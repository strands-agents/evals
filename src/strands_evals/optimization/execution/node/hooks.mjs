// Module hooks for runner.mjs: the user's own modules (not node_modules) import `@strands-agents/harness` through a
// shim whose `createHarness` the runner can watch and give a trial's values; everything else it exports is the real
// module's. Registered by runner.mjs with `module.register`; `data.harnessURL` is the real module.

const SHIM = 'strands-evals:harness-shim'
let harnessURL

export async function initialize(data) {
  harnessURL = data.harnessURL
}

export async function resolve(specifier, context, next) {
  const parent = context.parentURL ?? ''
  if (specifier === '@strands-agents/harness' && parent.startsWith('file:') && !parent.includes('/node_modules/')) {
    return { url: SHIM, shortCircuit: true }
  }
  return next(specifier, context)
}

export async function load(url, context, next) {
  if (url !== SHIM) return next(url, context)
  const real = JSON.stringify(harnessURL)
  const source = [
    `export * from ${real}`,
    `import * as real from ${real}`,
    `export const createHarness = (options) => globalThis[Symbol.for('strands-evals.createHarness')](real, options)`,
  ].join('\n')
  return { format: 'module', source, shortCircuit: true }
}
