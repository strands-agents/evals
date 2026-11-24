import { createHarness } from '@strands-agents/harness'
import { Deterministic } from './world.ts'

export const agent = await createHarness({
  model: new Deterministic(),
  instructions: 'You answer questions. Give the exact answer.',
  effort: 'auto',
  builtinTools: [],
  builtinPlugins: [],
  skills: false,
})
