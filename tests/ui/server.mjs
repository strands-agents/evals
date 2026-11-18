/* Served mode against the real Python server: the page's requests go out over HTTP to
   `strands-evals ui`, so the contract is tested rather than a mock of it. */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { page, runner, waitFor } from './harness.mjs';

const PY = process.env.PYTHON || 'python3';
const TOKEN = 'test-token-abc';
const SERVE = new URL('./serve.py', import.meta.url).pathname;
const { ok, done } = runner('server');

// ── a tree with one paired run, one bare experiment, two unusable files and an optimizer run ──
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ui-server-'));
const onDisk = name => JSON.parse(fs.readFileSync(path.join(root, name), 'utf8'));
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
fs.cpSync(new URL('./fixtures/optimizer-run/', import.meta.url).pathname, path.join(root, 'tuning'),
  { recursive: true });

// ── the real server, on a port the OS says is free ──
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
// jsdom has no fetch: forward the page's relative requests to the real server.
const requested = [];
const forward = async (url, opts = {}) => { requested.push(String(url)); return fetch(base + url, opts); };
// Boot ends by rendering the list and choosing a tab, so a row captured before it finishes is stale.
const served = async (route = '/') => {
  const p = await page({ fetchImpl: forward, url: `${base}${route}?t=${TOKEN}` });
  await waitFor(() => p.$('.app').dataset.ready, 'boot to finish');
  return p;
};

try {
  const p = await served();
  const { $, $$, text, click, type, window, errs } = p;

  // ── mode, and the discovery the server did ──
  ok(/served · saves to disk/.test(text('#mode-tag')) && text('.tab.on') === 'Runs',
    'boots on the run list, in a mode that saves to disk', text('#mode-tag'));
  ok(text('#root-path') === fs.realpathSync(root), 'shows the served directory', text('#root-path'));
  ok(!$('[data-act="rescan"]').hidden && $('[data-act="pick"]').hidden, 'Rescan is offered in place of the folder picker');

  const rows = $$('#runs-rows tr.run');
  const paired = rows.find(r => /experiment.json/.test(r.textContent) && !/nested/.test(r.textContent));
  ok(rows.length === 2 && /pairing verified/.test(text(paired)) && /1 failing/.test(text(paired)),
    'two runs discovered, the pair verified by session_id and its failing row counted', text('#runs-rows'));

  const skipped = Object.fromEntries($$('#skips li').map(li => [li.querySelector('code').textContent, text(li)]));
  ok(Object.keys(skipped).sort().join() === 'half-written.json,notes.json'
    && /invalid JSON/.test(skipped['half-written.json']) && /not an experiment or a report/.test(skipped['notes.json']),
    'only the two files it could not use are skipped, each with its reason, and none of the optimizer run’s own',
    text('#skips'));

  // ── the optimizer run, summarised by the server before the page has read its log ──
  await click('#sec-opt');
  const optRow = $('#opt-runs-rows tr[data-opt-run]');
  ok(/support-tuning-20261110-091500-7af3e1/.test(text(optRow)) && /c10/.test(text(optRow.children[2]))
    && /10 candidates/.test(text(optRow.children[3])) && /6 iterations/.test(text(optRow.children[3])),
    'the run is listed with its champion and the size of the search', text(optRow));
  const result = text(optRow.children[1]);
  ok(/\+0\.143/.test(result) && /held out/.test(result) && /clear of zero/.test(result),
    'and the result the server read off the held-out report', result);

  // ── its log, fetched over the API ──
  await click(optRow);
  await waitFor(() => $$('#opt-series-chart circle').length > 0, 'the objective chart');
  ok(/support-tuning-20261110-091500-7af3e1/.test(text('#opt-sub')) && text('#path') === 'tuning/events.jsonl',
    'opening it names the run and the log it read', `${text('#opt-sub')} | ${text('#path')}`);
  const best = $$('#opt-stats .stat').find(el => /^Best on training/.test(text(el.querySelector('.k'))));
  ok(best && text(best.querySelector('.v')) === '0.67', 'the best feasible objective is derived from the fetched log',
    best && text(best));
  ok($$('#opt-series-chart circle').length === 8 && $$('#opt-usage-chart rect.col').length === 6,
    'as are the objective and cost charts',
    `${$$('#opt-series-chart circle').length} marks, ${$$('#opt-usage-chart rect.col').length} columns`);
  // The candidate table is built from the `trial` events, so it shows the whole log arrived.
  const candidate = id => $$('#opt-cands tr.case').find(r => r.textContent.includes(id));
  await click(candidate('c10'));
  ok($$('#opt-cands tr.case').length === 11 && $$('#opt-cands tr.detail table.trials tr').length === 15,
    'and a row per candidate, the champion listing every trial recorded for it',
    `${$$('#opt-cands tr.case').length} candidates, ${$$('#opt-cands tr.detail table.trials tr').length} trial rows`);

  // ── a trial's transcript: a second fetch, for a file named by a hash in the log ──
  const trialRow = name => $$('#opt-cands tr.trial').find(r => r.children[0].textContent.includes(name));
  await click(trialRow('train_case_01'));
  await waitFor(() => !!$('#opt-cands tr.tdetail .trial-panel .msg'), 'the fetched transcript');
  ok(text('#opt-cands tr.tdetail .tr-count') === '3 spans of a recorded session'
    && $$('#opt-cands tr.tdetail .call').some(c => /lookup_order/.test(c.textContent)),
    'a transcript is fetched from the run directory and read as a session, tool call included',
    text('#opt-cands tr.tdetail .tr-count'));
  await click(trialRow('train_case_01'));

  await click(candidate('c6'));
  await click(trialRow('val_case_02'));
  await waitFor(() => /not in the run directory/.test(text('#opt-cands tr.tdetail') || ''),
    'the answer for a transcript the server does not have');
  ok(/transcripts\/4ed70e0cfecb415b\.json, which is not in the run directory/.test(text('#opt-cands tr.tdetail')),
    'one the server answers 404 for is reported as a file the run is missing, by name',
    text('#opt-cands tr.tdetail').slice(0, 200));

  // ── the effects model: a third fetch, for a file beside the log ──
  await waitFor(() => $$('#opt-eff-chart .eff-row').length > 0, 'the fetched effects model');
  const estimates = $$('#opt-eff-chart .eff-row text.tick').map(t => t.textContent).join();
  ok(!$('#opt-effects').hidden && estimates === '+0.12 · 0.96,+0.03 · 0.73,-0.01 · 0.41'
    && /fitted to 48 observations/.test(text('#opt-eff-cap')),
    'the effects drawer shows the estimates out of the fetched file, and what they were fitted to',
    `${estimates} | ${text('#opt-eff-cap')}`);
  await click('.tab[data-view="runs"]');

  // ── selecting a run fetches both documents ──
  await click(paired);
  await waitFor(() => $$('#case-list li[data-i]').length > 0, 'the loaded cases');
  ok($$('#case-list li[data-i]').length === 2 && /Edits save back to experiment.json/.test(text('#exp-sub')),
    'the cases load over HTTP, and the page says where a save goes', text('#exp-sub'));
  ok(window.location.pathname === '/runs/experiment.json/experiment/cases/alpha', 'the address is a path naming the run, the view and the case',
    window.location.pathname);

  // ── the server checked each evaluator against the real registry and constructor ──
  const chips = $$('#evs .evchip');
  ok(chips[0].classList.contains('error') && /will not load/.test(text(chips[0])),
    'an evaluator that cannot be constructed is flagged on its chip', text(chips[0]));
  ok(chips[1].classList.contains('warning') && /custom/.test(text(chips[1])),
    'an unregistered one is marked custom, not failed', text(chips[1]));
  const notes = text('#evnotes');
  ok(/1 evaluator problem/.test(notes) && /OutputEvaluator/.test(notes) && /needs a 'rubric' setting/.test(notes)
    && !/RubricEvaluator/.test(notes), 'and only the error is listed below, with its evaluator and the fix', notes);

  // ── a Session-shaped actual_trajectory must not read as absent ──
  await click('.tab[data-view="report"]');
  await click($$('#rep-rows tr.case').find(r => /beta/.test(r.textContent)));
  ok(/Actual trajectory ?1 span of a recorded session/.test(text('#rep-rows tr.detail')),
    'a Session trajectory in a report is drawn as a trace', text('#rep-rows tr.detail').slice(0, 300));

  // ── Save writes the real file ──
  await click('.tab[data-view="experiment"]');
  await click($$('#case-list li[data-i]')[0]);
  await type('#form [data-k="expected_output"]', 'an edited expectation');
  await click('#save');
  await waitFor(() => text('#saved'), 'the save result');
  ok(/saved experiment.json/.test(text('#saved')) && text('#dirty') === 'No unsaved changes',
    'a save is reported, leaving nothing unsaved', `${text('#saved')} | ${text('#dirty')}`);
  const edited = JSON.parse(JSON.stringify(experiment));
  edited.cases[0].expected_output = 'an edited expectation';
  ok(JSON.stringify(onDisk('experiment.json')) === JSON.stringify(edited),
    'the file on disk holds the edit and is otherwise as it was, fields the form does not show included',
    JSON.stringify(onDisk('experiment.json')));

  // ── a write that would clobber someone else's edit is refused ──
  await type('#form [data-k="expected_output"]', 'my edit');
  const theirs = onDisk('experiment.json');
  theirs.cases[1].input = 'edited by someone else';
  await new Promise(r => setTimeout(r, 1100));          // mtime granularity
  fs.writeFileSync(path.join(root, 'experiment.json'), JSON.stringify(theirs, null, 2));
  await click('#save');
  await waitFor(() => /changed on disk|saved/.test(text('#saved')), 'the stale save result');
  ok(/changed on disk/.test(text('#saved')) && text('#saved .reload') === 'Reload from disk',
    'a save over a file changed since it was loaded is refused, with an offer to reload', text('#saved'));
  ok(JSON.stringify(onDisk('experiment.json')) === JSON.stringify(theirs), 'leaving the other edit on disk');

  await click('#saved .reload');
  await waitFor(() => text('#dirty') === 'No unsaved changes', 'the reload');
  await click($$('#case-list li[data-i]')[1]);
  ok($('#form [data-k="input"]').value === 'edited by someone else', 'and the reload shows it',
    $('#form [data-k="input"]').value);

  // ── an evaluator setting round-trips through a save ──
  await click($$('#evs .evchip')[1]);
  await type('#evform [data-evk="threshold"]', '0.42');
  await click('#save');
  await waitFor(() => /saved/.test(text('#saved')), 'the evaluator save');
  const rubric = JSON.stringify(onDisk('experiment.json').evaluators[1]);
  ok(rubric === JSON.stringify({ ...experiment.evaluators[1], threshold: 0.42 }),
    'the edited threshold is written, the rest of the evaluator untouched', rubric);

  // ── what the server itself refuses ──
  const invalid = await fetch(`${base}/api/experiment?path=experiment.json`, {
    method: 'PUT',
    headers: { 'X-Evals-Token': TOKEN, 'Content-Type': 'application/json' },
    body: JSON.stringify({ document: { cases: [{ name: 'no-input' }], evaluators: [] } }),
  });
  ok(invalid.status === 422, 'the server rejects a case with no input', invalid.status);
  const noToken = await fetch(`${base}/api/runs`);
  ok(noToken.status === 401, 'and a request without the token', noToken.status);

  // ── the link names the open optimizer run, and a link naming one opens it ──
  const older = path.join(root, 'older-tuning');
  fs.mkdirSync(older);
  fs.writeFileSync(path.join(older, 'run.json'), JSON.stringify({ ...onDisk('tuning/run.json'), run_id: 'older-run' }));
  fs.copyFileSync(path.join(root, 'tuning/events.jsonl'), path.join(older, 'events.jsonl'));
  const anHourAgo = new Date(Date.now() - 3600e3);
  for (const name of ['run.json', 'events.jsonl']) fs.utimesSync(path.join(older, name), anHourAgo, anHourAgo);

  const picking = await served('/optimization');
  ok(/support-tuning/.test(picking.text('#opt-sub')), 'a link naming no optimizer run opens the newest',
    picking.text('#opt-sub'));
  await picking.click(picking.$$('#opt-runs-rows tr[data-opt-run]').find(r => /older-run/.test(r.textContent)));
  await waitFor(() => picking.$('#v-opt-overview').classList.contains('on'), 'the picked run to open');
  ok(/older-run/.test(picking.text('#opt-sub')) && picking.window.location.pathname === '/optimization/older-tuning/overview',
    'picking another opens it and names it in the address', picking.window.location.pathname);
  picking.window.history.back();
  await waitFor(() => picking.$('#v-opt-runs').classList.contains('on'), 'Back to return to the list');
  ok(picking.window.location.pathname === '/optimization', 'and Back returns to the list of runs', picking.window.location.pathname);
  picking.window.history.forward();
  await waitFor(() => picking.$('#v-opt-overview').classList.contains('on'), 'Forward to reopen the run');

  const linked = await served('/optimization/older-tuning/overview');
  ok(/older-run/.test(linked.text('#opt-sub')), 'and a link naming that run opens it, not the newest',
    linked.text('#opt-sub'));
  const effects = requested.filter(url => url.includes('effects.json')).map(url => decodeURIComponent(url.split('path=')[1]));
  ok(effects.includes('tuning/effects.json') && !effects.includes('older-tuning/effects.json'),
    'the effects model is read where the run has one, and not asked for where it has none', [...new Set(effects)].join());
  await linked.click('.tab[data-view="opt-runs"]');
  await linked.click(linked.$$('#opt-runs-rows tr[data-opt-run]').find(r => /support-tuning/.test(r.textContent)));
  await waitFor(() => /support-tuning/.test(linked.text('#opt-sub')), 'the other run to open');
  linked.window.history.go(-2);
  await waitFor(() => /older-run/.test(linked.text('#opt-sub')), 'Back to reopen the run the address names');
  ok(linked.window.location.pathname === '/optimization/older-tuning/overview', 'Back to a run\'s address reopens that run',
    linked.window.location.pathname);

  // ── the folder changes under an open page: the next poll shows it, and what was open stays open ──
  const asked = [];
  const live = await page({
    visible: true, url: `${base}/optimization/tuning/candidates?t=${TOKEN}`,
    fetchImpl: (url, opts) => { asked.push(String(url).split('?')[0]); return forward(url, opts); },
  });
  await waitFor(() => live.$('.app').dataset.ready, 'boot to finish');
  await live.click(live.$$('#opt-cands tr.case').find(r => r.textContent.includes('c10')));
  await live.click(live.$$('#opt-cands tr.trial').find(r => r.children[0].textContent.includes('train_case_01')));
  await waitFor(() => !!live.$('#opt-cands tr.tdetail .msg'), 'the transcript');
  const nested = () => live.text(live.$$('#runs-rows tr.run').find(r => /nested/.test(r.textContent)));
  const seen = () => [live.$$('#opt-cands tr.trial').length, !!live.$('#opt-cands tr.tdetail .msg'), /not run yet/.test(nested())];
  asked.length = 0;
  await live.window.pollNow();
  ok(asked.join() === '/api/changes' && String(seen()) === '15,true,true',
    'a poll over an unchanged folder asks the server one thing and changes nothing', `${asked.join()} | ${seen()}`);

  const logged = fs.readFileSync(path.join(root, 'tuning/events.jsonl'), 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
  const late = logged.find(e => e.event === 'trial' && e.record.candidate_id === 'c10');
  late.record.case_name = 'late_case';
  fs.appendFileSync(path.join(root, 'tuning/events.jsonl'), JSON.stringify(late) + '\n');
  fs.writeFileSync(path.join(root, 'nested/report.json'), JSON.stringify(report));
  await live.window.pollNow();
  ok(String(seen()) === '16,true,false' && /2 rows/.test(nested()),
    'a trial appended to the log and a report written beside an experiment both show on the next poll, '
    + 'with the open transcript still open', `${seen()} | ${nested()}`);

  done(errs.concat(picking.errs, linked.errs, live.errs));
} finally {
  proc.kill();
  fs.rmSync(root, { recursive: true, force: true });
}
