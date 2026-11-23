import { createHarness } from '@strands-agents/harness'
import { TwoTurnDeterministic } from './world.ts'

export const agent = await createHarness({
  model: new TwoTurnDeterministic(),
  instructions: 'You answer questions.',
  effort: 'auto',
  builtinTools: [],
  builtinPlugins: [],
  skills: false,
})
