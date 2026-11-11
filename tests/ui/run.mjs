/* Runs every suite in one process-per-suite, so one crash cannot hide the others. */
import { spawnSync } from 'node:child_process';

const suites = ['standalone.mjs', 'fileorigin.mjs', 'extras.mjs', 'server.mjs'];
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
