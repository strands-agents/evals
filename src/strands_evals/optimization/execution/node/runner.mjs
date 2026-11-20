// The Node worker that runs a TypeScript (or JavaScript) agent file for strands-evals' optimizer.
//
// Usage: node runner.mjs <agent file>. One JSON object per line in on stdin, one per line out on stdout, matched by
// `id`; requests run concurrently. The file builds its agent with `createHarness` from `@strands-agents/harness`, as
// `/export` writes it. Every request runs the file again as a fresh module, so each trial gets a fresh agent and fresh
// objects; modules the file imports are loaded once and shared. Requests:
//
//   describe {attr, paths}           -> {index, values: [{value} | {missing}]}  the agent's call and its values
//   probe    {attr, index, values}   -> {ok} | {ok: false, status, error_type, error}  build it once, run nothing
//   run      {attr, index, values, input, trace_id, span_id, trace_attributes}
//                                    -> {status, output?, error_type?, error?, spans}
//   cancel   {target}                -> (no reply) abort that run
//
// `values` is a list of [path, value]: set on the `index`-th `createHarness` call of the file's run. Every call gets
// fresh `session` and `memory` folders (memory a copy of the configured one) and `printer: false` unless it gives
// one. A run's agent runs inside the given trace context, so its spans belong to the Python trial's trace; the
// spans of that trace are returned with the reply, crashes included. `status` is `completed`, `agent_failure`, or
// `infrastructure_invalid` (throttling, network, credentials, a model service's 5xx).

import { AsyncLocalStorage } from 'node:async_hooks'
import { cpSync, existsSync, mkdtempSync, rmSync } from 'node:fs'
import { createRequire, register } from 'node:module'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { pathToFileURL } from 'node:url'

const protocol = process.stdout.write.bind(process.stdout)
process.stdout.write = process.stderr.write.bind(process.stderr) // whatever the agent prints stays off the protocol
const send = (message) => void protocol(`${JSON.stringify(message)}\n`)

const [major, minor] = process.versions.node.split('.').map(Number)
const agentPath = resolve(process.argv[2] ?? '')
const agentURL = pathToFileURL(agentPath).href

const setup = (async () => {
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`Node.js ${process.versions.node} can't run TypeScript files directly; use Node.js 22.18 or later`)
  }
  // Resolved as the agent file resolves them, so the runner uses the project's own packages.
  const fromAgent = createRequire(agentPath)
  let harnessPath
  try {
    harnessPath = fromAgent.resolve('@strands-agents/harness')
  } catch {
    throw new Error(`${agentPath} can't import @strands-agents/harness; run \`npm install @strands-agents/harness\``)
  }
  const fromHarness = createRequire(harnessPath)
  const load = (name) => import(pathToFileURL(fromHarness.resolve(name)).href)
  const [otel, traceNode] = await Promise.all([load('@opentelemetry/api'), load('@opentelemetry/sdk-trace-node')])
  const collector = new Collector()
  const provider = new traceNode.NodeTracerProvider({ spanProcessors: [collector] })
  provider.register() // the global provider, with async context, so the agent's spans parent to the trial's span
  register(new URL('./hooks.mjs', import.meta.url), { data: { harnessURL: pathToFileURL(harnessPath).href } })
  return { otel, collector, harnessURL: pathToFileURL(harnessPath).href }
})()
setup.catch(() => undefined)

/** Keeps the finished spans of the traces a run is waiting on. */
class Collector {
  constructor() {
    this.traces = new Map()
  }
  watch(traceId) {
    if (!this.traces.has(traceId)) this.traces.set(traceId, [])
  }
  take(traceId) {
    const spans = this.traces.get(traceId) ?? []
    this.traces.delete(traceId)
    return withoutRepeatedModelCalls(spans).map(spanJson)
  }
  onStart() {}
  onEnd(span) {
    this.traces.get(span.spanContext().traceId)?.push(span)
  }
  forceFlush() {
    return Promise.resolve()
  }
  shutdown() {
    return Promise.resolve()
  }
}

// A model provider's SDK may trace its own call during the Strands `chat` span (e.g. the Anthropic SDK's
// `anthropic.messages.create`, parented elsewhere), also as `gen_ai.operation.name` `chat`: one model call read twice.
// Strands spans carry `gen_ai.event.start_time`; another `chat` span whose time overlaps a Strands one's, give or
// take `SLACK` (the two may be timed on clocks a millisecond or so apart), is that call again, and is left out.
const SLACK = 100_000_000n // 100 ms, in nanoseconds
const isChat = (span) => span.attributes['gen_ai.operation.name'] === 'chat'
const isStrands = (span) => 'gen_ai.event.start_time' in span.attributes
const toNanos = ([seconds, nanoseconds]) => BigInt(seconds) * 1000000000n + BigInt(nanoseconds)

function withoutRepeatedModelCalls(spans) {
  const calls = spans.filter((span) => isChat(span) && isStrands(span))
  const overlaps = (a, b) =>
    toNanos(a.startTime) <= toNanos(b.endTime) + SLACK && toNanos(a.endTime) + SLACK >= toNanos(b.startTime)
  return spans.filter((span) => !isChat(span) || isStrands(span) || !calls.some((call) => overlaps(span, call)))
}

const nanos = ([seconds, nanoseconds]) => (BigInt(seconds) * 1000000000n + BigInt(nanoseconds)).toString()

const spanJson = (span) => ({
  name: span.name,
  trace_id: span.spanContext().traceId,
  span_id: span.spanContext().spanId,
  parent_span_id: span.parentSpanContext?.spanId ?? span.parentSpanId ?? null,
  kind: span.kind,
  attributes: { ...span.attributes },
  events: span.events.map((event) => ({
    name: event.name,
    time_unix_nano: nanos(event.time),
    attributes: { ...(event.attributes ?? {}) },
  })),
  status: { code: span.status.code, message: span.status.message ?? null },
  start_time_unix_nano: nanos(span.startTime),
  end_time_unix_nano: nanos(span.endTime),
  scope: span.instrumentationScope?.name ?? span.instrumentationLibrary?.name ?? null,
})

// The `createHarness` calls of one run of the file, and what to give the agent's.
const capture = new AsyncLocalStorage()
const warned = new Set()

globalThis[Symbol.for('strands-evals.createHarness')] = async (real, options = {}) => {
  const run = capture.getStore()
  if (!run) return real.createHarness(options)
  const index = run.calls.length
  const call = { written: options }
  run.calls.push(call) // in call order, before any await
  let trial = options
  if (index === run.index) {
    for (const [path, value] of run.values) trial = setAt(trial, path, value)
    if (run.traceAttributes) trial = { ...trial, traceAttributes: { ...trial.traceAttributes, ...run.traceAttributes } }
  }
  trial = freshState(trial, run.folder(index))
  if (trial.printer === undefined) trial = { ...trial, printer: false }
  call.agent = await real.createHarness(trial)
  return call.agent
}

/** `document` with `value` at `path`, copying each object on the way instead of changing it. */
function setAt(document, path, value) {
  if (path.length === 0) return value
  const [head, ...rest] = path
  const copy = Array.isArray(document) ? [...document] : { ...(document ?? {}) }
  copy[head] = setAt(copy[head], rest, value)
  return copy
}

function freshState(options, folder) {
  let trial = options
  for (const key of ['session', 'memory']) {
    const value = trial[key] ?? true
    if (value === false) continue
    if (typeof value !== 'boolean' && (typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype)) {
      if (!warned.has(key)) {
        warned.add(key)
        process.stderr.write(`strands-evals: the agent's ${key} is an object, so every trial shares its state\n`)
      }
      continue
    }
    const config = typeof value === 'object' ? { ...value } : {}
    if (key === 'memory') {
      const source = resolve(config.dir ?? './.agent/memory')
      if (existsSync(source)) cpSync(source, join(folder, key), { recursive: true })
    }
    trial = { ...trial, [key]: { ...config, dir: join(folder, key) } }
  }
  return trial
}

let runs = 0

/** Runs the file once with the values on its `index`-th call; returns the agent and a cleanup. */
async function build(attr, index, values, traceAttributes) {
  const folders = []
  const run = {
    calls: [],
    index,
    values,
    traceAttributes,
    folder: () => {
      const folder = mkdtempSync(join(tmpdir(), 'strands-evals-trial-'))
      folders.push(folder)
      return folder
    },
  }
  const cleanup = async () => {
    for (const call of run.calls) if (call.agent) await release(call.agent)
    for (const folder of folders) rmSync(folder, { recursive: true, force: true })
  }
  try {
    const namespace = await capture.run(run, () => import(`${agentURL}?run=${++runs}`))
    const agent = namespace[attr]
    return { agent, run, namespace, cleanup }
  } catch (error) {
    await cleanup()
    throw error
  }
}

async function release(agent) {
  try {
    agent.cancel?.()
  } finally {
    await agent.shutdown?.()
  }
}

const jsonSafe = (value) =>
  value === null || ['string', 'number', 'boolean'].includes(typeof value) || Array.isArray(value)
    ? true
    : typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype

async function describe({ attr, paths }) {
  const built = await build(attr, -1, [], undefined)
  try {
    if (built.namespace[attr] === undefined) return { error: `${agentPath} exports no ${attr}` }
    const index = built.run.calls.findIndex((call) => call.agent === built.agent)
    if (index < 0) {
      return {
        error:
          `${attr} in ${agentPath} isn't an agent a createHarness call built; export the agent createHarness ` +
          'returns, as /export writes it',
      }
    }
    const written = built.run.calls[index].written
    const { DEFAULT_HARNESS_AGENT_CONFIG: defaults = {} } = await import((await setup).harnessURL)
    const values = paths.map((path) => {
      let value = path[0] in written ? written[path[0]] : defaults[path[0]]
      if (!(path[0] in written) && !(path[0] in defaults)) return { missing: `the createHarness call has no ${path[0]}` }
      for (const part of path.slice(1)) {
        if (value === null || typeof value !== 'object' || !(part in value)) {
          return { missing: 'the createHarness call has nothing at that path' }
        }
        value = value[part]
      }
      return jsonSafe(value) ? { value: value ?? null } : { missing: `its value is a ${value?.constructor?.name}` }
    })
    return { index, values }
  } finally {
    await built.cleanup()
  }
}

async function probe({ attr, index, values }) {
  let built
  try {
    built = await build(attr, index, values, undefined)
    if (built.run.calls[index]?.agent !== built.agent) throw new Error(`${attr} was built by another createHarness call`)
    return { ok: true }
  } catch (error) {
    return { ok: false, status: classify(error), ...describeError(error) }
  } finally {
    await built?.cleanup()
  }
}

const running = new Map()

async function runCase({ id, attr, index, values, input, trace_id, span_id, trace_attributes }) {
  const { otel, collector } = await setup
  const controller = new AbortController()
  running.set(id, controller)
  const parent = trace_id
    ? otel.trace.setSpanContext(otel.ROOT_CONTEXT, {
        traceId: trace_id,
        spanId: span_id,
        traceFlags: otel.TraceFlags.SAMPLED,
        isRemote: true,
      })
    : otel.ROOT_CONTEXT
  if (trace_id) collector.watch(trace_id)
  let built
  try {
    const result = await otel.context.with(parent, async () => {
      built = await build(attr, index, values, trace_attributes)
      if (built.run.calls[index]?.agent !== built.agent) throw new Error(`${attr} was built by another createHarness call`)
      return built.agent.invoke(typeof input === 'string' ? input : JSON.stringify(input), {
        cancelSignal: controller.signal,
      })
    })
    return { status: 'completed', output: String(result) }
  } catch (error) {
    return { status: classify(error), ...describeError(error) }
  } finally {
    running.delete(id)
    await built?.cleanup()
  }
}

// Mirrors the Python optimizer's `classify`: the model service, the network, or credentials failing is
// infrastructure (re-run, never scored); anything else is the agent's failure.
const INFRASTRUCTURE_NAMES = new Set([
  'ModelThrottledError',
  'ModelNetworkError',
  'ThrottlingException',
  'TooManyRequestsException',
  'ServiceUnavailableException',
  'InternalServerException',
  'ModelStreamErrorException',
  'ModelNotReadyException',
  'ExpiredTokenException',
  'ExpiredToken',
  'CredentialsProviderError',
  'TokenProviderError',
])
const INFRASTRUCTURE_CODES = new Set([
  'ECONNRESET',
  'ECONNREFUSED',
  'ETIMEDOUT',
  'ENOTFOUND',
  'EAI_AGAIN',
  'EPIPE',
  'EMFILE',
  'ENFILE',
])

function classify(error, depth = 0) {
  if (!error || typeof error !== 'object' || depth > 5) return 'agent_failure'
  const status = error.$metadata?.httpStatusCode
  if (
    INFRASTRUCTURE_NAMES.has(error.name) ||
    INFRASTRUCTURE_NAMES.has(error.constructor?.name) ||
    INFRASTRUCTURE_CODES.has(error.code) ||
    status === 429 ||
    (typeof status === 'number' && status >= 500)
  ) {
    return 'infrastructure_invalid'
  }
  for (const inner of [error.cause, ...(Array.isArray(error.errors) ? error.errors : [])]) {
    if (classify(inner, depth + 1) === 'infrastructure_invalid') return 'infrastructure_invalid'
  }
  return 'agent_failure'
}

const describeError = (error) => ({
  error_type: (error && (error.name || error.constructor?.name)) || 'Error',
  error: error instanceof Error ? error.message : String(error),
})

async function answer(line) {
  let request
  try {
    request = JSON.parse(line)
  } catch {
    process.stderr.write(`strands-evals runner ignored a line that isn't JSON: ${line.slice(0, 200)}\n`)
    return
  }
  try {
    if (request.type === 'cancel') {
      running.get(request.target)?.abort()
      return
    }
    await setup
    const handler = { describe, probe, run: runCase }[request.type]
    if (!handler) throw new Error(`unknown request type ${request.type}`)
    const reply = await handler(request)
    if (request.type === 'run' && request.trace_id) reply.spans = (await setup).collector.take(request.trace_id)
    send({ ...reply, id: request.id })
  } catch (error) {
    send({ id: request.id, setup_error: describeError(error).error })
  }
}

const pending = new Set()
createInterface({ input: process.stdin })
  .on('line', (line) => {
    const answering = answer(line)
    pending.add(answering)
    void answering.finally(() => pending.delete(answering))
  })
  .on('close', () => void Promise.allSettled([...pending]).then(() => process.exit(0))) // finish what's in flight
