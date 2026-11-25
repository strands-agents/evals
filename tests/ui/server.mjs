/* One folder holding everything, against the real Python server: the page's requests go out over HTTP to
   `strands-evals ui`, so the contract is tested rather than a mock of it. */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ask, page, realServer, runner, waitFor } from './harness.mjs';

const TOKEN = 'test-token-abc';
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

const { base } = await realServer(root, TOKEN);
// jsdom has no fetch: forward the page's relative requests to the real server.
const requested = [];
const forward = async (url, opts = {}) => { requested.push(String(url)); return ask(base, url, opts); };
// Boot ends by rendering the list and choosing a tab, so a row captured before it finishes is stale.
const served = async (route = '/') => {
  const p = await page({ fetchImpl: forward, url: `${base}${route}?t=${TOKEN}` });
  await waitFor(() => p.$('.app').dataset.ready, 'boot to finish');
  return p;
};

try {
  const p = await served();
  const { $, $$, text, click, type, window } = p;

  // ── the discovery the server did ──
  ok($('#read-only').hidden && text('.tab.on') === 'Reports', 'boots on Reports, with nothing said of read-only', text('.tab.on'));
  ok(text('#path') === fs.realpathSync(root), 'the bar names the served directory', text('#path'));
  ok(!$('[data-act="rescan"]').hidden, 'Rescan is offered');

  const found = $$('#reports-rows tr.row');
  ok(found.length === 1 && $$('#experiments-rows tr.row').length === 2
    && /experiment\.json pairing verified/.test(text(found[0])) && /1 failing/.test(text(found[0])),
  'one report and two experiments discovered, the report paired by session_id and its failing row counted', text('#reports-rows'));

  const skipped = Object.fromEntries($$('#skips li').map(li => [li.querySelector('code').textContent, text(li)]));
  ok(Object.keys(skipped).sort().join() === 'half-written.json,notes.json'
    && /invalid JSON/.test(skipped['half-written.json']) && /not an experiment, a report, a session or a diagnosis/.test(skipped['notes.json']),
    'only the two files it could not use are skipped, each with its reason, and none of the optimizer run’s own',
    text('#skips'));

  // ── the optimizer run, summarised by the server before the page has read its log ──
  await click('.tab[data-view="opt-runs"]');
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
  ok(/support-tuning-20261110-091500-7af3e1/.test(text('#crumbs')) && text('#path') === fs.realpathSync(root),
    'opening it names the run, under a bar that still names the folder', `${text('#crumbs')} | ${text('#path')}`);
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
  await click('.tab[data-view="experiments"]');

  // ── opening an experiment fetches its document ──
  await click('#experiments-rows tr.row[data-experiment="experiment.json"]');
  await waitFor(() => $$('#case-list li[data-i]').length > 0, 'the loaded cases');
  ok($$('#case-list li[data-i]').length === 2 && /Edits save back to experiment.json/.test(text('#exp-sub')),
    'the cases load over HTTP, and the page says where a save goes', text('#exp-sub'));
  ok(window.location.pathname === '/experiments/experiment.json/cases/alpha', 'the address is a path naming the experiment and the case',
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
  await click('#facts a[data-report]');
  await waitFor(() => $$('#rep-rows tr.case').length > 0, 'the loaded report');
  ok(window.location.pathname === '/reports/report.json' && text('#crumbs') === 'Reports/report.json',
    'its report, linked from the header, is fetched and has a path of its own', window.location.pathname);
  await click($$('#rep-rows tr.case').find(r => /beta/.test(r.textContent)));
  ok(/Actual trajectory ?1 span of a recorded session/.test(text('#rep-rows tr.detail')),
    'a Session trajectory in a report is drawn as a trace', text('#rep-rows tr.detail').slice(0, 300));

  // ── Save writes the real file ──
  await click('#facts a[data-experiment]');
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
  const noToken = await fetch(`${base}/api/scan`);
  ok(noToken.status === 401, 'and a request without the token', noToken.status);

  // ── a link naming a report, or a case of an experiment, opens it ──
  const direct = await served('/reports/report.json');
  ok(direct.$('#v-report').classList.contains('on') && direct.text('#crumbs') === 'Reports/report.json' && direct.text('#r-score') === '0.50',
    'a path naming a report opens that report', direct.text('#crumbs'));
  ok(!direct.$('#experiments-rows tr.row.on'), 'and reads no experiment with it');
  const onCase = await served('/experiments/experiment.json/cases/beta');
  ok(onCase.$('#v-experiment').classList.contains('on') && onCase.text('#case-list li.on .nm') === 'beta',
    'a path naming a case of an experiment opens it on that case', onCase.text('#case-list li.on'));

  // ── the link names the open optimizer run, and a link naming one opens it ──
  const older = path.join(root, 'older-tuning');
  fs.mkdirSync(older);
  fs.writeFileSync(path.join(older, 'run.json'), JSON.stringify({ ...onDisk('tuning/run.json'), run_id: 'older-run' }));
  fs.copyFileSync(path.join(root, 'tuning/events.jsonl'), path.join(older, 'events.jsonl'));
  const anHourAgo = new Date(Date.now() - 3600e3);
  for (const name of ['run.json', 'events.jsonl']) fs.utimesSync(path.join(older, name), anHourAgo, anHourAgo);

  const picking = await served('/optimization');
  ok(/support-tuning/.test(picking.text('#opt-runs-rows tr')) && !picking.$('#opt-runs-rows tr.on'),
    'a link naming no optimizer run lists them, newest first, and marks none: the reader has opened none',
    picking.text('#opt-runs-rows tr'));
  await picking.click(picking.$$('#opt-runs-rows tr[data-opt-run]').find(r => /older-run/.test(r.textContent)));
  await waitFor(() => picking.$('#v-opt-overview').classList.contains('on'), 'the picked run to open');
  ok(/older-run/.test(picking.text('#crumbs')) && picking.window.location.pathname === '/optimization/older-tuning/result',
    'picking another opens it and names it in the address', picking.window.location.pathname);
  await picking.travel(-1);
  await waitFor(() => picking.$('#v-opt-runs').classList.contains('on'), 'Back to return to the list');
  ok(picking.window.location.pathname === '/optimization', 'and Back returns to the list of runs', picking.window.location.pathname);
  await picking.travel(1);
  await waitFor(() => picking.$('#v-opt-overview').classList.contains('on'), 'Forward to reopen the run');

  const linked = await served('/optimization/older-tuning/result');
  ok(/older-run/.test(linked.text('#crumbs')), 'and a link naming that run opens it, not the newest',
    linked.text('#crumbs'));
  const effects = requested.filter(url => url.includes('effects.json')).map(url => decodeURIComponent(url.split('path=')[1]));
  ok(effects.includes('tuning/effects.json') && !effects.includes('older-tuning/effects.json'),
    'the effects model is read where the run has one, and not asked for where it has none', [...new Set(effects)].join());
  await linked.click('.tab[data-view="opt-runs"]');
  await linked.click(linked.$$('#opt-runs-rows tr[data-opt-run]').find(r => /support-tuning/.test(r.textContent)));
  await waitFor(() => /support-tuning/.test(linked.text('#crumbs')), 'the other run to open');
  await linked.travel(-2);
  await waitFor(() => /older-run/.test(linked.text('#crumbs')), 'Back to reopen the run the address names');
  ok(linked.window.location.pathname === '/optimization/older-tuning/result', 'Back to a run\'s address reopens that run',
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
  const nested = () => live.text('#experiments-rows tr.row[data-experiment="nested/experiment.json"]');
  const seen = () => [live.$$('#opt-cands tr.trial').length, !!live.$('#opt-cands tr.tdetail .msg'), /not run yet/.test(nested())];
  asked.length = 0;
  await live.poll();
  ok(asked.join() === '/api/changes' && String(seen()) === '15,true,true',
    'a poll over an unchanged folder asks the server one thing and changes nothing', `${asked.join()} | ${seen()}`);

  const logged = fs.readFileSync(path.join(root, 'tuning/events.jsonl'), 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
  const late = logged.find(e => e.event === 'trial' && e.record.candidate_id === 'c10');
  late.record.case_name = 'late_case';
  fs.appendFileSync(path.join(root, 'tuning/events.jsonl'), JSON.stringify(late) + '\n');
  fs.writeFileSync(path.join(root, 'nested/report.json'), JSON.stringify(report));
  await live.poll();
  const written = live.text('#reports-rows tr.row[data-report="nested/report.json"]') || '';
  ok(String(seen()) === '16,true,false' && /report\.json 0\.50/.test(nested()) && /2 rows/.test(written) && /nested\/experiment\.json/.test(written),
    'a trial appended to the log and a report written beside an experiment both show on the next poll, '
    + 'with the open transcript still open', `${seen()} | ${nested()} | ${written}`);

  done();
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
