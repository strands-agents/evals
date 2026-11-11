/* Server mode, driven against the REAL Python server: the page's requests go out over HTTP
 * to `strands-evals ui`, so the contract is tested rather than a mock of it. */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { page, runner, tick, waitFor } from './harness.mjs';

const PY = process.env.PYTHON || 'python3';
const TOKEN = 'test-token-abc';
const SERVE = new URL('./serve.py', import.meta.url).pathname;
const { ok, done } = runner('server');

// ── a tree with one paired run, one bare experiment, and two files that cannot be used ──
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ui-server-'));
const experiment = {
  cases: [
    { name: 'alpha', session_id: 'sid-alpha', input: 'ask alpha', expected_output: 'A',
      expected_interactions: [{ turn: 1 }] },
    { name: 'beta', session_id: 'sid-beta', input: 'ask beta' },
  ],
  evaluators: [{ evaluator_type: 'OutputEvaluator' }, { evaluator_type: 'RubricEvaluator', name: 'rubric', threshold: 0.8 }],
};
const report = {
  overall_score: 0.5,
  cases: [
    { name: 'alpha', session_id: 'sid-alpha', evaluator: 'OutputEvaluator', input: 'ask alpha',
      expected_output: 'A', actual_output: 'A' },
    { name: 'beta', session_id: 'sid-beta', evaluator: 'OutputEvaluator', input: 'ask beta',
      actual_trajectory: { session_id: 'sid-beta', traces: [{ spans: [{ name: 'agent' }] }] } },
  ],
  scores: [1.0, 0.0],
  test_passes: [true, false],
  detailed_results: [
    [{ score: 1.0, test_pass: true, label: 'match', reason: 'same' }],
    [{ score: 0.0, test_pass: false, label: 'no_match', reason: 'wrong' }],
  ],
};
fs.writeFileSync(path.join(root, 'experiment.json'), JSON.stringify(experiment, null, 2));
fs.writeFileSync(path.join(root, 'report.json'), JSON.stringify(report, null, 2));
fs.mkdirSync(path.join(root, 'nested'));
fs.writeFileSync(path.join(root, 'nested/experiment.json'), JSON.stringify({ cases: [], evaluators: [] }));
fs.writeFileSync(path.join(root, 'notes.json'), '{"just": "notes"}');
fs.writeFileSync(path.join(root, 'half-written.json'), '{"cases": [');

// ── boot it on a port the OS says is free ──
const port = await new Promise(resolve => {
  const probe = net.createServer();
  probe.listen(0, '127.0.0.1', () => {
    const { port } = probe.address();
    probe.close(() => resolve(port));
  });
});
const proc = spawn(PY, [SERVE, root, String(port), TOKEN], { stdio: ['ignore', 'pipe', 'inherit'] });
await new Promise((resolve, reject) => {
  proc.stdout.on('data', chunk => String(chunk).includes('ready') && resolve());
  proc.on('exit', code => reject(new Error('server exited ' + code)));
  setTimeout(() => reject(new Error('server did not start')), 10000);
});
const base = `http://127.0.0.1:${port}`;

// jsdom has no fetch; forward the page's relative requests to the real server.
const forward = async (url, opts = {}) => fetch(base + url, opts);

try {
  const p = await page({ fetchImpl: forward, url: `${base}/?t=${TOKEN}` });
  const { $, $$, text, click, type, window, errs } = p;
  await waitFor(() => $$('#runs-rows tr.run').length > 0, 'the run list');

  // ── mode and provenance ──
  ok(/served · saves to disk/.test(text('#mode-tag')), 'mode tag says it saves to disk', text('#mode-tag'));
  ok(text('#root-path') === fs.realpathSync(root), 'shows the served directory', text('#root-path'));
  ok($('[data-act="rescan"]').hidden === false, 'Rescan offered when served');
  ok($('[data-act="pick"]').hidden === true, 'folder picker hidden when served');
  ok(text('.tab.on') === 'Runs', 'lands on the run list', text('.tab.on'));

  // ── discovery came from the server ──
  const rows = $$('#runs-rows tr.run');
  ok(rows.length === 2, 'two runs discovered', rows.length);
  const paired = rows.find(r => /experiment.json/.test(r.textContent) && !/nested/.test(r.textContent));
  ok(/pairing verified/.test(paired.textContent), 'session_ids verify the pair', paired.textContent);
  ok(/1 failing/.test(paired.textContent), 'failing count from the server', paired.textContent);

  // ── files it could not use are named, with the reason ──
  const skips = text('#skips');
  ok(/2 JSON files skipped/.test(skips), 'skipped files counted', skips);
  ok(/notes.json/.test(skips) && /not an experiment or a report/.test(skips), 'names the unrelated file', skips);
  ok(/half-written.json/.test(skips) && /invalid JSON/.test(skips), 'names the broken file with a reason', skips);

  // ── selecting a run fetches both documents ──
  await click(paired);
  await waitFor(() => $$('#case-list li[data-i]').length > 0, 'the loaded cases');
  ok($$('#case-list li[data-i]').length === 2, 'cases loaded over HTTP', $$('#case-list li[data-i]').length);
  ok(/Edits save back to experiment.json/.test(text('#exp-sub')), 'says where a save goes', text('#exp-sub'));
  ok(window.location.hash.includes('run=experiment.json'), 'URL carries the selection', window.location.hash);

  // ── the server checked each evaluator against the real registry and constructor ──
  const chips = $$('#evs .evchip');
  ok(chips[0].className.includes('error'), 'unconstructable evaluator flagged on its chip', chips[0].className);
  ok(/will not load/.test(chips[0].textContent), 'chip says it will not load', chips[0].textContent);
  ok(chips[1].className.includes('warning'), 'unregistered evaluator marked, not failed', chips[1].className);
  ok(/custom/.test(chips[1].textContent), 'chip calls it custom', chips[1].textContent);
  const notes = text('#evnotes');
  ok(/1 evaluator problem/.test(notes), 'errors counted below the strip', notes);
  ok(/OutputEvaluator/.test(notes) && /needs a 'rubric' setting/.test(notes), 'names the evaluator and the fix', notes);
  ok(!/RubricEvaluator/.test(notes), 'a custom evaluator is not called a problem', notes);

  // ── a Session-shaped actual_trajectory must not read as absent ──
  await click('.tab[data-view="report"]');
  const betaRow = $$('#rep-rows tr.case').find(r => /beta/.test(r.textContent));
  await click(betaRow);
  const detail = $$('#rep-rows tr.detail').map(r => r.textContent.replace(/\s+/g, ' ')).join(' ');
  ok(/Actual trajectory/.test(detail), 'trajectory row shown', detail.slice(0, 200));
  ok(/recorded session, not a step list/.test(detail), 'a Session is labelled, not called none', detail.slice(0, 400));
  ok(!/Actual trajectory none captured/.test(detail), 'never claims a present trajectory is absent');
  ok(!/object Object/.test(detail), 'no [object Object] in the report detail');

  // ── save writes the real file ──
  await click('.tab[data-view="experiment"]');
  await click($$('#case-list li[data-i]')[0]);
  await type('#form [data-k="expected_output"]', 'an edited expectation');
  ok(/1 unsaved change/.test(text('#dirty')), 'edit marks dirty', text('#dirty'));
  await click('#save');
  await waitFor(() => text('#saved'), 'the save result');
  ok(/saved experiment.json/.test(text('#saved')), 'save reports success', text('#saved'));
  ok(text('#dirty') === 'No unsaved changes', 'clean after save', text('#dirty'));
  const onDisk = JSON.parse(fs.readFileSync(path.join(root, 'experiment.json'), 'utf8'));
  ok(onDisk.cases[0].expected_output === 'an edited expectation', 'the file on disk changed',
    JSON.stringify(onDisk.cases[0]));
  ok(onDisk.cases[0].session_id === 'sid-alpha', 'session_id preserved', onDisk.cases[0].session_id);
  ok(JSON.stringify(onDisk.cases[0].expected_interactions) === '[{"turn":1}]',
    'a field the form does not expose survived', JSON.stringify(onDisk.cases[0].expected_interactions));
  ok(JSON.stringify(onDisk.evaluators) === JSON.stringify(experiment.evaluators),
    'evaluators preserved byte-for-byte', JSON.stringify(onDisk.evaluators));

  // ── a write that would clobber someone else's edit is refused ──
  await type('#form [data-k="expected_output"]', 'my edit');
  const stolen = JSON.parse(fs.readFileSync(path.join(root, 'experiment.json'), 'utf8'));
  stolen.cases[1].input = 'edited by someone else';
  await new Promise(r => setTimeout(r, 1100));          // mtime granularity
  fs.writeFileSync(path.join(root, 'experiment.json'), JSON.stringify(stolen, null, 2));
  await click('#save');
  await waitFor(() => /changed on disk|saved/.test(text('#saved')), 'the stale save result');
  ok(/changed on disk/.test(text('#saved')), 'stale save refused', text('#saved'));
  ok($('#saved .reload'), 'offers to reload from disk');
  ok(JSON.parse(fs.readFileSync(path.join(root, 'experiment.json'), 'utf8')).cases[1].input
    === 'edited by someone else', 'the other edit survived');

  // ── reloading recovers, and the other edit is visible ──
  await click('#saved .reload');
  await waitFor(() => text('#dirty') === 'No unsaved changes', 'the reload');
  ok(text('#dirty') === 'No unsaved changes', 'reload drops the stale edit', text('#dirty'));
  await click($$('#case-list li[data-i]')[1]);
  ok($('#form [data-k="input"]').value === 'edited by someone else', 'shows what is now on disk',
    $('#form [data-k="input"]').value);

  // ── evaluator config edits round-trip through a save ──
  await click($$('#evs .evchip')[1]);
  await type('#evform [data-evk="threshold"]', '0.42');
  await click('#save');
  await waitFor(() => /saved/.test(text('#saved')), 'the evaluator save');
  ok(/saved/.test(text('#saved')), 'evaluator edit saved', text('#saved'));
  const after = JSON.parse(fs.readFileSync(path.join(root, 'experiment.json'), 'utf8'));
  ok(after.evaluators[1].threshold === 0.42, 'threshold written', JSON.stringify(after.evaluators[1]));
  ok(after.evaluators[1].evaluator_type === 'RubricEvaluator', 'evaluator_type untouched',
    after.evaluators[1].evaluator_type);

  // ── the server is the one that refuses a bad document ──
  const direct = await fetch(`${base}/api/experiment?path=experiment.json`, {
    method: 'PUT',
    headers: { 'X-Evals-Token': TOKEN, 'Content-Type': 'application/json' },
    body: JSON.stringify({ document: { cases: [{ name: 'no-input' }], evaluators: [] } }),
  });
  ok(direct.status === 422, 'server rejects an invalid case', direct.status);

  const noToken = await fetch(`${base}/api/runs`);
  ok(noToken.status === 401, 'api needs the token', noToken.status);

  done(errs);
} finally {
  proc.kill();
}
