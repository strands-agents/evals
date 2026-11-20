// Set up an optimization of a strands-workspace agent with the installed Strands CLI's own code.
// node prepare.mjs inspect [WORKSPACE]   -> JSON: the CLI, the agent, the experiments, and the knobs
// node prepare.mjs setup [WORKSPACE] --experiment NAME [--tune KEY]... [--model ID]... [--baseline-only]
//   [--final-pick N:Q]                   -> JSON: the run folder, the tuned variables, and the run/resume commands
import { existsSync, readdirSync, readFileSync, realpathSync } from 'node:fs'
import { mkdir, writeFile } from 'node:fs/promises'
import { delimiter, dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

const BRIDGE = ['evals-agent/entry-files.js', 'temporary-evals-fork/typescript-runner-main.js']
const AGENTS = { 'agent.py': ['python', 'create_harness'], 'agent.ts': ['typescript', 'createHarness'] }

const reply = process.stdout.write.bind(process.stdout)
process.stdout.write = process.stderr.write.bind(process.stderr)
const fail = (message) => {
  process.stderr.write(`prepare: error: ${message}\n`)
  process.exit(2)
}

const OPTIONS = {
  experiment: { type: 'string' },
  tune: { type: 'string', multiple: true, default: [] },
  model: { type: 'string', multiple: true, default: [] },
  'baseline-only': { type: 'boolean', default: false },
  'final-pick': { type: 'string' },
}
const { values: args, positionals } = (() => {
  try {
    return parseArgs({ allowPositionals: true, options: OPTIONS })
  } catch (error) {
    fail(error.message)
  }
})()
const [action, workspacePath = 'strands-workspace'] = positionals
if (action !== 'inspect' && action !== 'setup') fail('give inspect or setup')

const findBridge = () => {
  const tried = []
  for (const name of ['strands-dev', 'strands']) {
    const found = (process.env.PATH ?? '').split(delimiter).map((folder) => join(folder, name)).find(existsSync)
    if (!found) continue
    const cli = dirname(dirname(realpathSync(found)))
    const bridge = join(cli, 'dist/src/dev')
    if (BRIDGE.every((file) => existsSync(join(bridge, file)))) return bridge
    tried.push(`${name} at ${cli} has no evals bridge`)
  }
  fail(
    `no installed Strands CLI includes the evals bridge (${tried.join('; ') || 'no strands-dev or strands on PATH'}). ` +
      'Install a Strands CLI build that ships dist/src/dev/evals-agent and the TypeScript runner, then try again.'
  )
}
const bridge = findBridge()
const load = (path) => import(pathToFileURL(join(bridge, path)).href)
const shell = (word) => `'${String(word).replaceAll("'", "'\\''")}'`

const findAgent = async (workspace, directory) => {
  const { readDevWorkspace } = await load('workspace.js')
  if (existsSync(join(directory, '.config'))) return { ...(await readDevWorkspace(directory)), directory }
  const found = Object.keys(AGENTS).filter((file) => existsSync(join(workspace, file)))
  if (found.length !== 1) fail(`${workspace} needs exactly one of agent.py and agent.ts, found ${found.length}`)
  const [language, call] = AGENTS[found[0]]
  const entrypoint = join(workspace, found[0])
  if (!readFileSync(entrypoint, 'utf8').includes(call)) fail(`${entrypoint} does not call ${call}(...)`)
  return { directory, root: workspace, entrypoint, config: { language } }
}

const listExperiments = (directory) => {
  const folder = join(directory, 'experiments')
  const names = existsSync(folder) ? readdirSync(folder).filter((name) => !name.startsWith('.')) : []
  return names.flatMap((name) => {
    const file = join(folder, name, `${name}.json`)
    if (!existsSync(file)) return []
    try {
      const { cases = [], evaluators = [] } = JSON.parse(readFileSync(file, 'utf8'))
      return [{ name, file, cases: cases.length, evaluators: evaluators.map((e) => e.evaluator_type) }]
    } catch (error) {
      return [{ name, file, problem: `not JSON (${error.message})` }]
    }
  })
}

const shown = ({ key, label, kind, detail, unavailable, choices, current, entries, enabled }) => ({
  key,
  label,
  kind,
  detail,
  ...(unavailable ? { unavailable } : {}),
  ...(choices ? { choices } : {}),
  ...(current === undefined ? {} : { current: typeof current === 'string' ? current.slice(0, 80) : current }),
  ...(entries ? { entries: entries.map((entry) => entry.label) } : {}),
  recommended: enabled,
})

const setup = async (agent, knobs, experiments) => {
  const chosen = experiments.find((experiment) => experiment.name === args.experiment)
  if (!chosen) fail(`no experiment named ${args.experiment}; the workspace has ${experiments.map((e) => e.name).join(', ') || 'none'}`)
  const tune = args.tune.length > 0 ? args.tune : ['instructions']
  const tunable = knobs.filter((knob) => !knob.unavailable).map((knob) => knob.key)
  const unknown = tune.filter((key) => !tunable.includes(key))
  if (unknown.length > 0) fail(`cannot tune ${unknown.join(', ')}; the tunable knobs are ${tunable.join(', ')}`)
  const { temporarySpaceVariables } = await load('temporary-evals-fork/knobs.js')
  const enabled = knobs.map((knob) => ({ ...knob, enabled: tune.includes(knob.key) }))
  const variables = temporarySpaceVariables(enabled, args.model)
  if (variables.length === 0) fail('the chosen knobs give nothing to tune; a model needs another model id to try')
  const [finalists, lowerBound] = (args['final-pick'] ?? '').split(':').map(Number)
  const layout = await load('temporary-evals-fork/layout.js')
  const { temporaryRunnableExperiment } = await load('temporary-evals-fork/experiment-files.js')
  const { prepareTemporaryOptimizerEnvironment } = await load('temporary-evals-fork/environment.js')
  const { temporaryTypescriptRunnerOf, temporaryEvalsEnvironment } = await load('temporary-evals-fork/evals-process.js')
  const { writeEvalsOptimizeEntry } = await load('evals-agent/entry-files.js')
  const language = agent.config.language
  const python = await prepareTemporaryOptimizerEnvironment(
    { workspaceDirectory: agent.directory, projectRoot: agent.root, language },
    { onProgress: (line) => process.stderr.write(`${line}\n`) }
  )
  await layout.ensureTemporaryExperimentsFolder(agent.directory)
  const home = layout.temporaryExperimentHome(agent.directory, chosen.file)
  const files = layout.temporaryOptimizationFiles(home, layout.temporaryRunStamp(new Date()))
  if (existsSync(files.folder)) fail(`${files.folder} already exists; wait a second, then try again`)
  await mkdir(dirname(files.space), { recursive: true })
  await writeFile(files.space, `${JSON.stringify({ variables }, null, 2)}\n`)
  const experiment = resolve(await temporaryRunnableExperiment(chosen.file, files.copy))
  const flags = await writeEvalsOptimizeEntry(files.task, {
    agentFile: agent.entrypoint,
    experimentFile: experiment,
    customEvaluators: [],
    ...temporaryTypescriptRunnerOf({ config: { language } }),
    spaceFile: files.space,
    settings: ['instructions'],
    baselineOnly: args['baseline-only'],
    ...(finalists ? { finalPick: { finalists, lowerBound } } : {}),
  })
  // The environment the CLI starts the optimizer with: its bin first on PATH, and AWS_DEFAULT_REGION from AWS_REGION.
  const region = process.env.AWS_REGION
  const env = temporaryEvalsEnvironment(python, region ? { AWS_REGION: region } : {})
  const start = [
    `cd ${shell(agent.root)} &&`,
    `PATH=${shell(env.PATH)}:"$PATH"`,
    ...(env.AWS_DEFAULT_REGION ? [`AWS_DEFAULT_REGION=${shell(env.AWS_DEFAULT_REGION)}`] : []),
    `${shell(python)} -m strands_evals.cli optimize`,
  ].join(' ')
  const run = [experiment, ...flags, '--base-dir', files.record].map(shell).join(' ')
  return { folder: files.folder, variables, command: `${start} run ${run} --rich`, resume: `${start} resume` }
}

try {
  const workspace = resolve(workspacePath)
  const directory = join(workspace, 'strands-dev')
  if (!existsSync(directory) && !Object.keys(AGENTS).some((file) => existsSync(join(workspace, file))))
    fail(`${workspace} is not a strands-workspace`)
  if (action === 'setup' && !args.experiment) fail('setup needs --experiment NAME, a name from inspect')
  const agent = await findAgent(workspace, directory)
  const { readTemporaryKnobs } = await load('temporary-evals-fork/knobs.js')
  const knobs = await readTemporaryKnobs(agent)
  const experiments = listExperiments(directory)
  const output =
    action === 'inspect'
      ? {
          cli: dirname(dirname(dirname(bridge))),
          agent: { file: agent.entrypoint, root: agent.root, language: agent.config.language },
          experiments,
          knobs: knobs.map(shown),
        }
      : await setup(agent, knobs, experiments)
  reply(`${JSON.stringify(output, null, 2)}\n`)
} catch (error) {
  fail(error instanceof Error ? error.message : String(error))
}
