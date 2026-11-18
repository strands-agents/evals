/* Keeping up with the folder: a served page asks whether the tree changed and re-reads what is open,
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
const listed = (path, modified) => ({ path, name: path, size: 1, modified, cases: 2, evaluators: [], rows: 2, overall_score: 0.5 });

/* The tree the fake server reads: a test edits it, then moves `mark` as a real change would. */
const tree = {
  mark: 'm1',
  runs: [{ id: 'e.json', dir: '.', link: null, experiment: listed('e.json', 5), report: listed('r.json', 5) }],
  files: { 'e.json': { modified: 5, document: experiment('second question') }, 'r.json': { modified: 5, document: report(true, false) } },
};
const write = (path, document) => {
  const modified = tree.files[path].modified + 1;
  tree.files[path] = { modified, document };
  tree.runs[0][path === 'e.json' ? 'experiment' : 'report'].modified = modified;
  tree.mark += '+';
};
const server = fakeServer({
  mark: () => tree.mark,
  runs: () => [{ root: '/served', depth: 3, read_only: false, fingerprint: tree.mark, runs: tree.runs, skipped: [], optimizer_runs: [] }],
  file: path => ({ path, ...tree.files[path] }),
});
const asked = () => server.sent.splice(0).map(r => r.url.split('?')[0]).join(' ');

const p = await page({ fetchImpl: server, visible: true });
await p.click('#runs-rows tr.run');
const poll = async () => { await p.window.pollNow(); await tick(); };

// ── nothing changed: one cheap question, and nothing re-read ──
asked();
await poll();
ok(asked() === '/api/changes', 'a poll over an unchanged folder asks only whether it changed');

// ── a run appears ──
tree.runs.push({ id: 'new/e.json', dir: 'new', link: null, report: null, experiment: listed('new/e.json', 5) });
tree.mark += '+';
await poll();
ok(p.$$('#runs-rows tr.run').length === 2 && p.$$('#runs-rows tr.run')[0].classList.contains('on'),
  'a new run shows up in the list with nothing clicked, and the open run stays selected', p.text('#runs-rows'));
ok(asked() === '/api/changes /api/runs', 'the open files are not fetched again when they did not change');

// ── the open report is re-run: it is re-read, and the reader keeps their place ──
await p.click('.tab[data-view="report"]');
await p.click('#rep-seg button[data-f="fail"]');
await p.click('#rep-rows tr.case');
write('r.json', report(false, false));
await poll();
ok(p.text('#r-score') === '0.00' && p.$$('#rep-rows tr.case').length === 2, 'a report written again is shown',
  `${p.text('#r-score')} · ${p.$$('#rep-rows tr.case').length} rows`);
ok(p.$('#rep-seg button.on').dataset.f === 'fail' && p.$('#rep-rows tr.case[aria-expanded="true"]'),
  'with the filter and the open row left as they were');

// ── the open experiment changes on disk: a clean editor follows it, keeping the case being read ──
await p.click('.tab[data-view="experiment"]');
await p.click(p.$$('#case-list li[data-i]')[1]);
write('e.json', experiment('rewritten elsewhere'));
await poll();
const input = () => p.$('#form [data-k="input"]').value;
ok(input() === 'rewritten elsewhere', 'an untouched editor reloads, still on the case it was showing', input());

// ── but never over unsaved edits ──
await p.type('#form [data-k="input"]', 'my edit');
write('e.json', experiment('rewritten again'));
await poll();
ok(input() === 'my edit' && /changed on disk/.test(p.text('#saved')),
  'unsaved edits are kept, and the page says the file moved under them', `${input()} | ${p.text('#saved')}`);
await p.click('#saved .reload');
await p.click(p.$$('#case-list li[data-i]')[1]);
ok(input() === 'rewritten again', 'the offer beside it reloads from disk', input());

// ── a tab nobody is looking at stays quiet, and catches up when it is looked at ──
Object.defineProperty(p.d, 'hidden', { value: true, configurable: true });
tree.runs.pop();
tree.mark += '+';
asked();
await poll();
ok(asked() === '', 'a hidden tab does not poll');
Object.defineProperty(p.d, 'hidden', { value: false, configurable: true });
p.d.dispatchEvent(new p.window.Event('visibilitychange'));
await tick(4);
ok(p.$$('#runs-rows tr.run').length === 1, 'and catches up as soon as it is shown again', p.$$('#runs-rows tr.run').length);

done(p.errs);
