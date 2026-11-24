/* Keeping up with the folder: the page asks whether the tree changed and re-reads what is open,
   without replacing work in progress. Against a fake server whose tree the test changes. */
import { fakeServer, page, runner, tick } from './harness.mjs';

const { ok, done } = runner('live');

const report = (...passes) => ({
  overall_score: passes.filter(Boolean).length / passes.length,
  cases: passes.map((_, i) => ({ name: 'ab'[i], evaluator: 'X', input: 'q' })),
  scores: passes.map(Number), test_passes: passes,
});
const experiment = second => ({
  cases: [{ name: 'a', session_id: 's-a', input: 'first question' }, { name: 'b', session_id: 's-b', input: second }],
  evaluators: [],
});
const listed = (path, modified, more) =>
  ({ path, name: path, dir: '.', size: 1, modified, cases: 2, evaluators: [], rows: 2, overall_score: 0.5, ...more });

/* The tree the fake server reads: a test edits it, then moves `mark` as a real change would. */
const tree = {
  mark: 'm1',
  experiments: [listed('e.json', 5, { reports: ['r.json'] })],
  reports: [listed('r.json', 5, { experiment: 'e.json', link: null }), listed('other.json', 5, { experiment: null, link: null })],
  sessions: [{ path: 's.json', name: 's.json', dir: '.', size: 1, modified: 5, session_id: 's', traces: 1, spans: 1, seconds: 1, diagnoses: [] }],
  files: {
    's.json': { modified: 5, document: { session_id: 's', traces: [{ trace_id: 't', session_id: 's', spans: [{
      span_info: { span_id: 'a', session_id: 's', parent_span_id: null, start_time: '2026-01-05T09:00:00.000000Z', end_time: '2026-01-05T09:00:01.000000Z' },
      span_type: 'invoke_agent', user_prompt: 'hello', agent_response: 'hi', available_tools: [],
    }] }] } },
    'e.json': { modified: 5, document: experiment('second question') },
    'r.json': { modified: 5, document: report(true, false) },
    'other.json': { modified: 5, document: report(true, true) },
  },
};
const write = (path, document) => {
  const modified = tree.files[path].modified + 1;
  tree.files[path] = { modified, document };
  [...tree.experiments, ...tree.reports].find(m => m.path === path).modified = modified;
  tree.mark += '+';
};
const server = fakeServer({
  mark: () => tree.mark,
  scan: () => [{ root: '/served', depth: 3, read_only: false, fingerprint: tree.mark, experiments: tree.experiments, reports: tree.reports,
    sessions: tree.sessions, skipped: [], optimizer_runs: [] }],
  file: path => ({ path, ...tree.files[path] }),
  /* Only `e.json` is ever saved: refused, as the server refuses, when it changed since the editor read it. */
  put: body => {
    if (body.modified !== tree.files['e.json'].modified) return [{ error: 'e.json changed on disk since it was loaded' }, 409];
    write('e.json', body.document);
    return [{ path: 'e.json', modified: tree.files['e.json'].modified, cases: body.document.cases.length }];
  },
});
const asked = () => server.sent.splice(0).map(r => r.url.split('?')[0]).join(' ');
/* Hold the answer for a file until the test lets it go, as a slow read would. */
const slow = new Map();
const hold = path => {
  let release;
  slow.set(path, new Promise(resolve => { release = resolve; }));
  return async () => { slow.delete(path); release(); await tick(); };
};
const fetchImpl = async (url, opts) => {
  await slow.get(new URL(String(url), 'http://x').searchParams.get('path'));
  return server(url, opts);
};

const p = await page({ fetchImpl, visible: true });
await p.openExperiment();
const rows = () => p.$$('#experiments-rows tr.row');

// ── nothing changed: one cheap question, and nothing re-read ──
asked();
await p.poll();
ok(asked() === '/api/changes', 'a poll over an unchanged folder asks only whether it changed');

// ── an experiment appears ──
tree.experiments.push(listed('new/e.json', 5, { dir: 'new', reports: [] }));
tree.mark += '+';
rows()[0].focus();
await p.poll();
ok(rows().length === 2 && rows()[0].classList.contains('on'),
  'a new experiment shows up in the list with nothing clicked, and the open one stays marked', p.text('#experiments-rows'));
ok(p.d.activeElement === rows()[0], 'and the keyboard is still on the row it was on', p.d.activeElement.tagName);
ok(asked() === '/api/changes /api/scan', 'the open files are not fetched again when they did not change');

// ── the open report is re-run: it is re-read, and the reader keeps their place ──
await p.openReport();
await p.click('#rep-seg button[data-f="fail"]');
p.$('#rep-rows tr.case').focus();
await p.click('#rep-rows tr.case');
ok(p.d.activeElement === p.$('#rep-rows tr.case[aria-expanded="true"]'), 'opening a row leaves the keyboard on it', p.d.activeElement.tagName);
write('r.json', report(false, false));
asked();
await p.poll();
ok(asked() === '/api/changes /api/scan /api/file', 'only the report is fetched again, not the experiment open beside it');
ok(p.text('#r-score') === '0.00' && p.$$('#rep-rows tr.case').length === 2, 'a report written again is shown',
  `${p.text('#r-score')} · ${p.$$('#rep-rows tr.case').length} rows`);
ok(p.$('#rep-seg button.on').dataset.f === 'fail' && p.$('#rep-rows tr.case[aria-expanded="true"]'),
  'with the filter and the open row left as they were');
ok(p.d.activeElement === p.$('#rep-rows tr.case[aria-expanded="true"]'), 'and the keyboard still on that row', p.d.activeElement.tagName);

// ── the open experiment changes on disk: a clean editor follows it, keeping the case being read ──
await p.openExperiment();
await p.click(p.$$('#case-list li[data-i]')[1]);
write('e.json', experiment('rewritten elsewhere'));
await p.poll();
const input = () => p.$('#form [data-k="input"]').value;
ok(input() === 'rewritten elsewhere', 'an untouched editor reloads, still on the case it was showing', input());

// ── but never over unsaved edits ──
await p.type('#form [data-k="input"]', 'my edit');
write('e.json', experiment('rewritten again'));
await p.poll();
ok(input() === 'my edit' && /changed on disk/.test(p.text('#saved')),
  'unsaved edits are kept, and the page says the file moved under them', `${input()} | ${p.text('#saved')}`);
await p.click('#saved .reload');
await p.click(p.$$('#case-list li[data-i]')[1]);
ok(input() === 'rewritten again', 'the offer beside it reloads from disk', input());

// ── nor over an edit made while the changed file was being read ──
{
  write('e.json', experiment('rewritten under a slow read'));
  const release = hold('e.json');
  const polled = p.window.pollNow();
  await tick();
  await p.type('#form [data-k="input"]', 'typed meanwhile');
  await release();
  await polled;
  ok(input() === 'typed meanwhile' && /changed on disk/.test(p.text('#saved')),
    'an edit made during the re-read is kept, and the page says the file moved under it', `${input()} | ${p.text('#saved')}`);
  await p.click('#saved .reload');
  await p.click(p.$$('#case-list li[data-i]')[1]);
}

// ── nor over text that is typed but not applied yet ──
{
  const half = '[{"node_name": "search"';
  await p.type('#form [data-k="expected_interactions"]', half);
  write('e.json', experiment('rewritten under half-written JSON'));
  await p.poll();
  ok(p.$('#form [data-k="expected_interactions"]').value === half && /changed on disk/.test(p.text('#saved')),
    'JSON that does not parse yet is not an applied edit, and is kept all the same', `${p.$('#form [data-k="expected_interactions"]').value} | ${p.text('#saved')}`);
  await p.click('#saved .reload');
  await p.click(p.$$('#case-list li[data-i]')[1]);
}

// ── nor over the name of a tool that is being typed ──
{
  await p.click('#form [data-add]');
  const box = () => p.$('#form .chipin');
  box().value = 'sear';
  box().dispatchEvent(new p.window.Event('input', { bubbles: true }));
  write('e.json', experiment('rewritten under a tool name'));
  await p.poll();
  ok(box() && box().value === 'sear' && /changed on disk/.test(p.text('#saved')),
    'a tool name not yet added is kept too', `${box() && box().value} | ${p.text('#saved')}`);
  await p.click('#saved .reload');
  await p.click(p.$$('#case-list li[data-i]')[1]);
}

// ── the case being read is renamed on disk ──
{
  const renamed = experiment('renamed on disk');
  renamed.cases[1].name = 'z';
  write('e.json', renamed);
  await p.poll();
  ok(p.window.location.pathname === '/experiments/e.json/cases/a' && p.text('#form-title') === 'a',
    'when the case being read is gone from the file, the address moves to the case shown in its place', p.window.location.pathname);
  write('e.json', experiment('named as before'));
  await p.poll();
  await p.click(p.$$('#case-list li[data-i]')[1]);
}

// ── Rescan re-reads what is open as well as the lists ──
write('e.json', experiment('rewritten a third time'));
await p.click('[data-act="rescan"]');
ok(input() === 'rewritten a third time', 'Rescan shows the open experiment as it is on disk now, not only the lists', input());

// ── a reload the reader walked away from does not move what Save is checked against ──
{
  await p.type('#form [data-k="input"]', 'mine');
  write('e.json', experiment('theirs'));
  await p.click('#save');
  ok(/changed on disk/.test(p.text('#saved')), 'a save over a file that changed on disk is refused', p.text('#saved'));
  const release = hold('e.json');
  await p.click('#saved .reload');
  await p.click('.tab[data-view="reports"]');
  await release();
  await p.openExperiment();
  await p.click('#save');
  ok(input() === 'mine' && tree.files['e.json'].document.cases[1].input === 'theirs' && /changed on disk/.test(p.text('#saved')),
    'the edits are still there, and saving them is still refused', `${input()} | ${tree.files['e.json'].document.cases[1].input} | ${p.text('#saved')}`);
  await p.click('#saved .reload');
  await p.click(p.$$('#case-list li[data-i]')[1]);
}

// ── a file that arrives after the reader went elsewhere does not take the page ──
{
  let release = hold('r.json');
  await p.openReport('r.json');
  await p.openReport('other.json');
  await release();
  ok(p.text('#crumbs') === 'Reports/other.json' && p.text('#r-score') === '1.00',
    'a report that loads late does not replace the one opened after it', `${p.text('#crumbs')} ${p.text('#r-score')}`);

  await p.openReport('r.json');
  write('r.json', report(true, true));
  release = hold('r.json');
  const polled = p.window.pollNow();
  await tick();
  await p.openReport('other.json');
  await release();
  await polled;
  ok(p.text('#crumbs') === 'Reports/other.json', 'nor does a re-read of the report that was open before it', p.text('#crumbs'));

  release = hold('s.json');
  await p.openSession();
  await p.openReport('other.json');
  await release();
  ok(p.text('#crumbs') === 'Reports/other.json', 'nor does a session that loads late', p.text('#crumbs'));
}

// ── a step through history that a click overtakes ──
{
  const at = () => `${p.view()} ${p.window.location.pathname}`;
  await p.openReport('r.json');
  await p.openReport('other.json');
  let release = hold('r.json');
  await p.travel(-1);
  await p.click('.tab[data-view="experiments"]');
  await release();
  ok(at() === 'v-experiments /experiments', 'Back to a report that is slow to load, then a click elsewhere: the page stays where the click went', at());
  await p.travel(-1);
  ok(at() === 'v-report /reports/r.json' && p.text('#crumbs') === 'Reports/r.json', 'and the click was a step of its own, which Back undoes', at());

  release = hold('other.json');
  p.window.visit('/reports/other.json');
  await tick();
  await p.click('.tab[data-view="experiments"]');
  await release();
  ok(at() === 'v-experiments /experiments', 'the same when the slow report was asked for by a link in the page', at());

  await p.click('.tab[data-view="reports"]');
  const first = hold('r.json'), second = hold('other.json');
  await p.openReport('r.json');
  await p.openReport('other.json');
  await first();
  const early = at();
  await second();
  ok(early === 'v-reports /reports' && at() === 'v-report /reports/other.json' && p.text('#crumbs') === 'Reports/other.json',
    'of two reports asked for in a row the second is the one opened, whichever answers first', `${early} | ${at()}`);
}

// ── an optimizer run whose log arrives after the reader went elsewhere ──
{
  const run = dir => ({ id: dir, dir, run_id: dir, name: dir, events_path: `${dir}/events.jsonl`, config: {},
    candidates: 0, iterations: 0, modified: 1, has_effects: false });
  const runs = fakeServer({ scan: { optimizer_runs: [run('first'), run('second')] }, file: () => ({ events: [] }) });
  const q = await page({ url: 'http://127.0.0.1:9999/optimization', fetchImpl: async (url, opts) => {
    await slow.get(new URL(String(url), 'http://x').searchParams.get('path'));
    return runs(url, opts);
  } });
  const release = hold('second/events.jsonl');
  await q.click(q.$$('#opt-runs-rows tr[data-opt-run]')[1]);
  await q.click('.tab[data-view="reports"]');
  await release();
  ok(`${q.view()} ${q.window.location.pathname}` === 'v-reports /reports',
    'a run that loads late does not take the page back from where the reader went', `${q.view()} ${q.window.location.pathname}`);
}

// ── a tab nobody is looking at stays quiet, and catches up when it is looked at ──
Object.defineProperty(p.d, 'hidden', { value: true, configurable: true });
tree.experiments.pop();
tree.mark += '+';
asked();
await p.poll();
ok(asked() === '', 'a hidden tab does not poll');
Object.defineProperty(p.d, 'hidden', { value: false, configurable: true });
p.d.dispatchEvent(new p.window.Event('visibilitychange'));
await tick(4);
ok(rows().length === 1, 'and catches up as soon as it is shown again', rows().length);

done();
