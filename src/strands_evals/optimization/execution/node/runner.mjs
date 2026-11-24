// The Node worker that runs a TypeScript (or JavaScript) agent file for strands-evals' optimizer.
//
// Usage: node runner.mjs <agent file>. One JSON object per line in on stdin, one per line out on stdout, matched by
// `id`; requests run concurrently. The file builds its agent with `createHarness` from `@strands-agents/harness`, as
// `/export` writes it. Every request but `turn` and `release` runs the file again as a fresh module, so each trial gets
// a fresh agent and fresh objects (a session's turns reuse the agent its `start` built); modules the file imports are
// loaded once and shared. Requests:
//
//   describe {attr, paths}           -> {index, values: [{value} | {missing}]}  the agent's call and its values
//   probe    {attr, index, values}   -> {ok} | {ok: false, status, error_type, error}  build it once, run nothing
//   run      {attr, index, values, input, trace_id, span_id, trace_attributes}
//                                    -> {status, output?, result?, error_type?, error?, spans}
//   start    {attr, index, values, trace_id, span_id, trace_attributes}
//                                    -> {ok, session, spans} | {ok: false, status, error_type, error, spans}
//                                       hold an agent open
//   turn     {session, input, trace_id, span_id}
//                                    -> {status, output?, result?, error_type?, error?, spans}  ask it again; the
//                                       turns of one session run one at a time, a cancelled one finishing first
//   release  {session, wait_ms?}     -> {ok}  let the agent go and remove its folders, once a turn still running
//                                       ends (waited for `wait_ms`, 30 s by default, at most)
//   cancel   {target}                -> (no reply) abort that run or turn; release the session a start opened
//
// `values` is a list of [path, value]: set on the `index`-th `createHarness` call of the file's run. Every call gets
// fresh `session` and `memory` folders (memory a copy of the configured one) and `printer: false` unless it gives
// one. A run's agent runs inside the given trace context, so its spans belong to the Python trial's trace; the
// spans of that trace are returned with the reply, crashes included. `status` is `completed`, `agent_failure`, or
// `infrastructure_invalid` (throttling, network, credentials, a model service's 5xx).
//
// `run` builds a fresh agent, asks it once, and lets it go. A conversation is `start`, then `turn` for each message
// (the same agent, keeping what was said), then `release`; the one who started it releases it. `output` is the
// answer as text; `result` is its JSON-safe heart (the stop reason, the final message, any structured output).

import { AsyncLocalStorage } from 'node:async_hooks'
import { randomUUID } from 'node:crypto'
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
  // Strands records each message a model call sends as one event of its span, so OpenTelemetry's default of 128
  // events would drop the start of a long conversation: no limit unless OTEL_SPAN_EVENT_COUNT_LIMIT sets one (empty
  // for none, as in Python). Each span's dropped count goes back with it, so Python warns when that limit drops some.
  const limit = process.env.OTEL_SPAN_EVENT_COUNT_LIMIT?.trim()
  const eventCountLimit = !limit ? Infinity : /^\d+$/.test(limit) ? Number(limit) : undefined
  const provider = new traceNode.NodeTracerProvider({
    spanProcessors: [collector],
    ...(eventCountLimit === undefined ? {} : { spanLimits: { eventCountLimit } }), // else the SDK reads the value
  })
  provider.register() // the global provider, with async context, so the agent's spans parent to the trial's span
  register(new URL('./hooks.mjs', import.meta.url), { data: { harnessURL: pathToFileURL(harnessPath).href } })
  return { otel, collector, harnessURL: pathToFileURL(harnessPath).href }
})()
setup.catch(() => undefined)

/**
 * Keeps the finished spans of the traces requests are waiting on. Requests of one trial share its trace (two sessions
 * at once, or a session and a run), so each `watch` is matched by one `take`: a take returns the spans finished since
 * the last one, and the trace is let go only when the last request watching it is done.
 */
class Collector {
  constructor() {
    this.traces = new Map()
  }
  watch(traceId) {
    const trace = this.traces.get(traceId)
    if (trace) trace.watching += 1
    else this.traces.set(traceId, { watching: 1, spans: [] })
  }
  /**
   * Ends one watch and returns the spans finished since the last take. A request the caller cancelled passes
   * `leaveToOthers`: while another request of the trace still runs, the spans stay for that one's reply, which the
   * caller reads for sure; only the last takes them, onto a late reply the caller reads if the trial is still open.
   */
  take(traceId, { leaveToOthers = false } = {}) {
    const trace = this.traces.get(traceId)
    if (!trace) return []
    trace.watching -= 1
    if (leaveToOthers && trace.watching > 0) return []
    const spans = trace.spans
    trace.spans = []
    if (trace.watching === 0) this.traces.delete(traceId)
    return withoutRepeatedModelCalls(spans).map(spanJson)
  }
  onStart() {}
  onEnd(span) {
    this.traces.get(span.spanContext().traceId)?.spans.push(span)
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
  dropped_events: span.droppedEventsCount ?? 0,
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
  // Never throws: a trial whose agent already answered must not fail because letting it go did.
  const cleanup = async () => {
    for (const call of run.calls) {
      if (!call.agent) continue
      try {
        await release(call.agent)
      } catch (error) {
        process.stderr.write(`strands-evals: shutting an agent down failed: ${describeError(error).error}\n`)
      }
    }
    for (const folder of folders) {
      try {
        rmSync(folder, { recursive: true, force: true })
      } catch (error) {
        process.stderr.write(`strands-evals: removing ${folder} failed: ${describeError(error).error}\n`)
      }
    }
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
    return { index, values, written: Object.keys(written) }
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

/** The trace context a request's agent runs inside: the Python trial's span, or nothing. */
const parentContext = (otel, trace_id, span_id) =>
  trace_id
    ? otel.trace.setSpanContext(otel.ROOT_CONTEXT, {
        traceId: trace_id,
        spanId: span_id,
        traceFlags: otel.TraceFlags.SAMPLED,
        isRemote: true,
      })
    : otel.ROOT_CONTEXT

/** The JSON-safe heart of an invoke's AgentResult, or null when none of it survives JSON. */
function resultJson(result) {
  if (!result || typeof result !== 'object') return null
  try {
    return JSON.parse(
      JSON.stringify({
        stop_reason: result.stopReason ?? null,
        last_message: result.lastMessage ?? null,
        structured_output: result.structuredOutput ?? null,
      })
    )
  } catch {
    return null
  }
}

async function runCase({ attr, index, values, input, trace_id, span_id, trace_attributes }, signal) {
  const { otel } = await setup
  let built
  try {
    const result = await otel.context.with(parentContext(otel, trace_id, span_id), async () => {
      built = await build(attr, index, values, trace_attributes)
      if (built.run.calls[index]?.agent !== built.agent) throw new Error(`${attr} was built by another createHarness call`)
      return built.agent.invoke(typeof input === 'string' ? input : JSON.stringify(input), { cancelSignal: signal })
    })
    return { status: 'completed', output: String(result), result: resultJson(result) }
  } catch (error) {
    return { status: classify(error), ...describeError(error) }
  } finally {
    await built?.cleanup()
  }
}

// Conversations held open by `start`: the same agent answers every `turn` of its session until `release`. Ids are
// random, so an id from before a worker restart never names a session the new worker opened for another trial.
const sessions = new Map()
// The session each `start` request opened, by request id, until it is released: a `cancel` aimed at a start that
// already replied releases it, since the caller cancelled before it could learn the id and will never release it.
const startedBy = new Map()

async function startSession({ id, attr, index, values, trace_id, span_id, trace_attributes }, signal) {
  const { otel } = await setup
  let built
  try {
    built = await otel.context.with(parentContext(otel, trace_id, span_id), () =>
      build(attr, index, values, trace_attributes)
    )
    if (built.run.calls[index]?.agent !== built.agent) throw new Error(`${attr} was built by another createHarness call`)
  } catch (error) {
    await built?.cleanup()
    return { ok: false, status: classify(error), ...describeError(error) }
  }
  if (signal.aborted) {
    // Cancelled while building: nobody will ever hear of this agent, so let it go now.
    await built.cleanup()
    return { ok: false, status: 'agent_failure', error_type: 'Cancelled', error: 'the start was cancelled' }
  }
  const session = randomUUID()
  sessions.set(session, { built, startId: id, idle: Promise.resolve() })
  startedBy.set(id, session)
  return { ok: true, session }
}

async function turn(request, signal) {
  const { session, input, trace_id, span_id } = request
  const { otel } = await setup
  const open = sessions.get(session)
  if (!open) {
    // Released, never started, or lost to a worker restart: the Python side reads this as infrastructure.
    return {
      status: 'infrastructure_invalid',
      error_type: 'NodeSessionGone',
      error: `no open session ${session}: it was released, or the worker restarted since it started`,
    }
  }
  // One invocation at a time: a turn waits for the one before it, including one the caller cancelled, which keeps
  // running until the agent notices the cancel (an agent asked twice at once raises ConcurrentInvocationError). The
  // session is idle again once the turn's reply is sent (`answer` calls `request.replied`), so a `release` waiting
  // for it replies after it, and the turn's spans reach the caller first.
  const before = open.idle
  open.idle = new Promise((resolve) => (request.replied = resolve))
  try {
    await before
    if (signal.aborted) return { status: 'agent_failure', error_type: 'Cancelled', error: 'the turn was cancelled' }
    if (!sessions.has(session)) {
      return { status: 'infrastructure_invalid', error_type: 'NodeSessionGone', error: `session ${session} was released` }
    }
    const result = await otel.context.with(parentContext(otel, trace_id, span_id), () =>
      open.built.agent.invoke(typeof input === 'string' ? input : JSON.stringify(input), { cancelSignal: signal })
    )
    return { status: 'completed', output: String(result), result: resultJson(result) }
  } catch (error) {
    return { status: classify(error), ...describeError(error) }
  }
}

// How long `release` waits for a turn still running (one the caller cancelled that hasn't stopped yet).
const RELEASE_WAIT_MS = 30_000

async function releaseSession({ session, wait_ms = RELEASE_WAIT_MS }) {
  const open = sessions.get(session)
  sessions.delete(session)
  if (!open) return { ok: true }
  startedBy.delete(open.startId)
  // A cancelled turn still winding down would write its session snapshot back after the folders go, so wait for it.
  // One that ignores the cancel (a stuck tool) is waited for `wait_ms` at most: then the agent is let go anyway, and
  // its folders are removed again once the turn ends.
  let timer
  const idle = await Promise.race([
    open.idle.then(() => true),
    new Promise((resolve) => (timer = setTimeout(() => resolve(false), wait_ms))),
  ])
  clearTimeout(timer)
  if (!idle) {
    process.stderr.write(`strands-evals: session ${session} still has a turn running after ${wait_ms} ms; released\n`)
    void open.idle.then(() => open.built.cleanup())
  }
  await open.built.cleanup()
  return { ok: true }
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
  if (request.type === 'cancel') {
    running.get(request.target)?.abort()
    const session = startedBy.get(request.target)
    if (session !== undefined) await releaseSession({ session })
    return
  }
  // Registered before anything is awaited, so a cancel that follows the request always finds it.
  const controller = new AbortController()
  running.set(request.id, controller)
  let watching
  try {
    const { collector } = await setup
    if (['run', 'start', 'turn'].includes(request.type) && request.trace_id) {
      collector.watch(request.trace_id)
      watching = request.trace_id
    }
    const handler = { describe, probe, run: runCase, start: startSession, turn, release: releaseSession }[request.type]
    if (!handler) throw new Error(`unknown request type ${request.type}`)
    const reply = await handler(request, controller.signal)
    if (watching) {
      reply.spans = collector.take(watching, { leaveToOthers: controller.signal.aborted })
      watching = undefined
    }
    send({ ...reply, id: request.id })
  } catch (error) {
    send({ id: request.id, setup_error: describeError(error).error })
  } finally {
    running.delete(request.id)
    if (watching) (await setup).collector.take(watching, { leaveToOthers: true }) // a failed request carries no spans
    request.replied?.()
  }
}

const pending = new Set()
createInterface({ input: process.stdin })
  .on('line', (line) => {
    const answering = answer(line)
    pending.add(answering)
    void answering.finally(() => pending.delete(answering))
  })
  .on('close', () =>
    void Promise.allSettled([...pending]).then(async () => {
      // Finish what's in flight, then let go of any conversation nobody released, folders included.
      await Promise.allSettled([...sessions.values()].map(({ built }) => built.cleanup()))
      process.exit(0)
    })
  )
