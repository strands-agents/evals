/* Runs every suite in one process-per-suite, so one crash cannot hide the others. */
import { spawnSync } from 'node:child_process';

/* The optimizer-run fixture is 84 files, 80 of them recorded conversations named by content hash.
   Committed, it buried every review of this directory in generated JSON — so it is written here
   instead, from the generator beside it, which produces it byte for byte. */
const gen = spawnSync(process.env.PYTHON || 'python3',
  [new URL('./fixtures/make_optimizer_run.py', import.meta.url).pathname],
  { stdio: ['ignore', 'ignore', 'inherit'] });
if (gen.status !== 0) {
  console.error('could not write fixtures/optimizer-run — every optimization assertion reads it.'
    + '\nIt needs a Python 3; override the interpreter with PYTHON=/path/to/python npm test.');
  process.exit(1);
}

const suites = ['standalone.mjs', 'fileorigin.mjs', 'extras.mjs', 'optimization.mjs', 'server.mjs'];
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
