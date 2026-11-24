/* Runs every suite, one process each, so one crash cannot hide the others. */
import { spawnSync } from 'node:child_process';

// The optimizer-run fixture is generated, not committed: write it before any suite reads it.
const gen = spawnSync(process.env.PYTHON || 'python3',
  [new URL('./fixtures/make_optimizer_run.py', import.meta.url).pathname],
  { stdio: ['ignore', 'ignore', 'inherit'] });
if (gen.status !== 0) {
  console.error('could not write fixtures/optimizer-run, which the optimization, server and markdown suites read.'
    + '\nIt needs a Python 3; override the interpreter with PYTHON=/path/to/python npm test.');
  process.exit(1);
}

const suites = [
  'sample.mjs',
  'extras.mjs',
  'optimization.mjs',
  'server.mjs',
  'guide.mjs',
  'charts.mjs',
  'live.mjs',
  'fidelity.mjs',
  'lenses.mjs',
  'sessions.mjs',
  'markdown.mjs',
  'editor.mjs',
  'empty.mjs',
];
let failed = 0;

for (const suite of suites) {
  const result = spawnSync(process.execPath, [new URL(`./${suite}`, import.meta.url).pathname], {
    stdio: 'inherit',
  });
  if (result.status !== 0) failed++;
}

if (failed) {
  console.log(`\n${failed} of ${suites.length} suites failed`);
  process.exit(1);
}
console.log(`\nall ${suites.length} suites passed`);
