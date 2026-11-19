/* Reports side by side: picking them in the list and from a report's header, the comparison page
   (summary, evaluator means, the verdict grid, oldest against newest), and its address. */
import { fakeServer, page, runner, served } from './harness.mjs';

const { ok, same, done } = runner('compare');

/* Rows as `[case, evaluator, score, verdict]`, the verdict one of pass, fail or na. `started` is when the run began. */
const report = (overall, rows, started) => ({
  overall_score: overall,
  cases: rows.map(([name, evaluator]) => ({ name, evaluator, input: 'q' })),
  scores: rows.map(r => r[2]),
  test_passes: rows.map(r => r[3] !== 'fail'),
  detailed_results: rows.map(r => [{ score: r[2], test_pass: r[3] !== 'fail', label: r[3] === 'na' ? 'not_applicable' : 'judged' }]),
  ...(started && { run_info: { start_time: started } }),
});
const experiment = { cases: ['a', 'b', 'c'].map(name => ({ name, session_id: `s-${name}`, input: 'q' })), evaluators: [] };

/* Three runs of the same cases. Their paths sort a, b, c; they ran c, a, b. */
const OLD = report(0.58, [['a', 'X', 1, 'pass'], ['a', 'Y', 0.9, 'pass'], ['b', 'X', 0, 'fail'], ['c', 'X', 0.3, 'fail'], ['d', 'X', 0.7, 'pass']],
  '2026-03-01T09:00:00Z');
const MID = report(0.45, [['a', 'X', 1, 'pass'], ['a', 'Y', 0.5, 'fail'], ['b', 'X', 0, 'fail'], ['c', 'X', 0.3, 'fail']],
  '2026-03-02T09:00:00Z');
const NEW = report(0.5, [['a', 'X', 1, 'pass'], ['a', 'Y', 0.2, 'fail'], ['b', 'X', 0, 'fail'], ['b', 'Y', 1, 'na'], ['c', 'X', 0.8, 'pass']],
  '2026-03-03T09:00:00Z');
const THREE = {
  'a/experiment.json': experiment, 'a/report.json': MID,
  'b/experiment.json': experiment, 'b/report.json': NEW,
  'c/experiment.json': experiment, 'c/report.json': OLD,
};

const box = (p, path) => p.$(`#reports-rows input[data-pick="${path}"]`);
const cells = (p, row) => p.$$('#cmp-summary tr').map(tr => p.text(tr.children[row]));
const compared = p => p.$$('#facts .chip').map(c => p.text(c).replace(/×$/, '').trim());
const address = (...paths) => '/reports/compare' + (paths.length ? '?' : '') + paths.map(path => 'r=' + encodeURIComponent(path)).join('&');
const here = p => p.window.location.pathname + p.window.location.search;

const p = await served(THREE);

// ── the list: tick reports, then compare them ──
ok(p.$$('#reports-rows input[type="checkbox"]').length === 3 && p.$('#cmp-go').hidden
  && !p.$('#cmp-hint').hidden && p.text('#cmp-hint') === 'Tick two or more reports to compare them.',
'every report in the list has a box, and the list says what ticking them is for', p.text('#cmp-hint'));
await p.click(box(p, 'c/report.json'));
ok(p.view() === 'v-reports' && box(p, 'c/report.json').checked && p.$('#cmp-go').hidden,
  'ticking a box does not open its report, and one report is not a comparison', p.view());
await p.click(box(p, 'a/report.json'));
ok(!p.$('#cmp-go').hidden && p.text('#cmp-go') === 'Compare (2)' && p.$('#cmp-hint').hidden,
  'two ticked reports offer Compare (2) in its place', p.text('#cmp-go'));
await p.click(box(p, 'b/report.json'));
await p.click(box(p, 'b/report.json'));
ok(p.text('#cmp-go') === 'Compare (2)' && !box(p, 'b/report.json').checked, 'unticking takes a report back out', p.text('#cmp-go'));
await p.click(box(p, 'b/report.json'));
ok(p.text('#cmp-go') === 'Compare (3)', 'and a third joins them', p.text('#cmp-go'));
ok(box(p, 'a/report.json').tabIndex === 0 && box(p, 'a/report.json').getAttribute('aria-label') === 'Compare a/report.json'
  && p.$('#cmp-go').tagName === 'BUTTON' && p.$('#cmp-go').tabIndex === 0,
'the boxes and the button are reached with Tab, and a box says which report it picks');
await p.key('Enter', box(p, 'a/report.json'));
ok(p.view() === 'v-reports', 'Enter on a box does not open the report its row names', p.view());
const before = p.window.history.length;
await p.click('#cmp-go');

// ── three reports, oldest first by when each run started ──
ok(p.view() === 'v-compare' && p.text('#crumbs') === 'Reports/Compare' && p.$('.tab.on').dataset.view === 'reports',
  'Compare opens the comparison, under Reports', `${p.view()} ${p.text('#crumbs')}`);
same(compared(p), ['#1 c/report.json', '#2 a/report.json', '#3 b/report.json'],
  'the header names the reports compared, in the order they ran, not the order of their paths');
ok(here(p) === address('c/report.json', 'a/report.json', 'b/report.json') && p.window.history.length === before + 1,
  'the address names them, and opening the comparison is a step in history', here(p));
same(cells(p, 2), ['0.58', '0.45', '0.50'], 'the summary gives each report\'s overall score');
same(cells(p, 3), ['—', '▼ -0.13', '▲ +0.05'], 'and its change from the report before it');
same([cells(p, 4), cells(p, 5), cells(p, 6)], [['5', '4', '5'], ['3', '1', '2'], ['2', '3', '2']],
  'with its rows, how many pass and how many fail, a row that was not applicable counted as neither');
ok(!/by file time/.test(p.text('#cmp-summary')), 'a report that records its run is dated by it');
same(p.$$('#cmp-evals tr').map(tr => [...tr.children].map(c => p.text(c))), [
  ['Evaluator', '#1', '#2', '#3'],
  ['X', '0.50', '0.43 ▼ -0.07', '0.60 ▲ +0.17'],
  ['Y', '0.90', '0.50 ▼ -0.40', '0.20 ▼ -0.30'],
], 'each evaluator\'s mean is given per report over the rows it applied to, with its change');

// ── the grid: a line per case and evaluator, a column per report ──
same(p.tips('#cmp-grid rect.cell'), [
  'a · Y · #1 c/report.json · score 0.90 · pass', 'a · Y · #2 a/report.json · score 0.50 · fail', 'a · Y · #3 b/report.json · score 0.20 · fail',
  'c · X · #1 c/report.json · score 0.30 · fail', 'c · X · #2 a/report.json · score 0.30 · fail', 'c · X · #3 b/report.json · score 0.80 · pass',
  'd · X · #1 c/report.json · score 0.70 · pass',
  'b · Y · #3 b/report.json · not applicable',
  'a · X · #1 c/report.json · score 1.00 · pass', 'a · X · #2 a/report.json · score 1.00 · pass', 'a · X · #3 b/report.json · score 1.00 · pass',
  'b · X · #1 c/report.json · score 0.00 · fail', 'b · X · #2 a/report.json · score 0.00 · fail', 'b · X · #3 b/report.json · score 0.00 · fail',
], 'the rows that flipped or are missing somewhere come first, and a report that does not hold a row leaves a gap there');
same(p.$$('#cmp-grid rect.cell').map(c => c.classList[1]),
  ['pass', 'fail', 'fail', 'fail', 'fail', 'pass', 'pass', 'na', 'pass', 'pass', 'pass', 'fail', 'fail', 'fail'],
  'each cell is coloured by its verdict, so a gap is never read as a failure');
ok(p.text('#cmp-grid-cap') === '4 of 6 rows changed verdict or are missing from a report, and come first. '
  + 'A gap means that report does not hold the row.', 'the caption counts them', p.text('#cmp-grid-cap'));

// ── the oldest against the newest ──
ok(p.text('#cmp-ends-head') === 'From #1 to #3'
  && p.text('#cmp-ends-cap') === 'c/report.json against b/report.json. 2 rows changed score or verdict.',
'the last chart sets the first report against the last', `${p.text('#cmp-ends-head')} | ${p.text('#cmp-ends-cap')}`);
ok(p.text('#cmp-ends-legend') === '1 now pass1 now fail2 kept their verdict1 only in #31 only in #1',
  'counting what now passes, now fails, kept its verdict, and what only one of them holds', p.text('#cmp-ends-legend'));
same(p.tips('#cmp-ends g.bell'), ['a · Y · 0.90 to 0.20 · now fails', 'c · X · 0.30 to 0.80 · now passes'],
  'the rows that moved are drawn, the largest move first');
same(p.$$('#cmp-ends g.bell').map(g => g.classList[1]), ['down', 'up'], 'marked worse or better');

// ── a mark opens that row in its report ──
await p.click(p.$$('#cmp-ends g.bell')[1]);
ok(p.view() === 'v-report' && p.text('#crumbs') === 'Reports/b/report.json'
  && p.$('#rep-rows tr.case[data-i="4"]').getAttribute('aria-expanded') === 'true',
'a row that moved opens in the newest report, on that row', `${p.view()} ${p.text('#crumbs')}`);
ok(here(p) === '/reports/b/report.json/rows/c/X', 'which the address names', here(p));
await p.travel(-1);
ok(p.view() === 'v-compare' && compared(p).length === 3, 'Back from the report returns to the comparison', p.view());
await p.key('Enter', p.$('#cmp-grid rect.cell'));
ok(p.view() === 'v-report' && p.text('#crumbs') === 'Reports/c/report.json'
  && p.$('#rep-rows tr.case[data-i="1"]').getAttribute('aria-expanded') === 'true',
'a cell of the grid opens its row in the report of its column, from the keyboard too', `${p.view()} ${p.text('#crumbs')}`);
await p.travel(-1);

// ── removing a report ──
same(p.$$('#facts [data-cmp-remove]').map(b => b.getAttribute('aria-label')), [
  'Remove c/report.json from the comparison', 'Remove a/report.json from the comparison', 'Remove b/report.json from the comparison',
], 'each compared report can be removed, by a button that says so');
await p.click('#facts [data-cmp-remove="a/report.json"]');
same(compared(p), ['#1 c/report.json', '#2 b/report.json'], 'removing one leaves the others, renumbered');
ok(!p.$('#facts [data-cmp-remove]'), 'two reports is the fewest a comparison holds, so neither can be removed');
same([cells(p, 2), cells(p, 3)], [['0.58', '0.50'], ['—', '▼ -0.08']], 'the change is now from the one before it that is left');
ok(p.text('#cmp-ends-head') === 'From #1 to #2' && p.$$('#cmp-grid rect.cell').length === 10,
  'and every part of the page follows', `${p.text('#cmp-ends-head')} ${p.$$('#cmp-grid rect.cell').length}`);
ok(here(p) === address('c/report.json', 'b/report.json'), 'as does the address', here(p));

// ── Back, and the address on its own ──
await p.travel(-1);
ok(p.view() === 'v-compare' && compared(p).length === 3, 'Back restores the comparison of three', compared(p).join());
await p.travel(-1);
ok(p.view() === 'v-reports' && here(p) === '/reports' && box(p, 'c/report.json').checked,
  'Back again returns to the list, the boxes as they were left', `${p.view()} ${here(p)}`);
await p.visit(address('b/report.json', 'a/report.json'));
same([p.view(), compared(p)], ['v-compare', ['#1 a/report.json', '#2 b/report.json']],
  'an address naming two reports opens their comparison, oldest first whatever order it names them in');
ok(here(p) === address('a/report.json', 'b/report.json'), 'and is corrected to that order', here(p));
for (const [named, what, says] of [
  [['a/report.json'], 'one report', 'only a/report.json'], [['a/report.json', 'a/report.json'], 'the same report twice', 'only a/report.json'],
  [[], 'none', 'none'],
]) {
  await p.visit(address(...named));
  ok(p.view() === 'v-missing' && here(p) === address(...named) && p.text('#missing').includes(`this address names ${says}.`),
    `an address naming ${what} says a comparison needs two, and stays as typed`, `${p.view()} ${here(p)} ${p.text('#missing')}`);
}

// ── from a report's header ──
await p.openReport('a /');
same(p.$$('#cmp-with button').map(b => b.dataset.value), ['', 'b/report.json', 'c/report.json'],
  'a report\'s header offers the other reports to compare it with');
ok(p.text('#cmp-with summary') === 'Compare with…' && p.$('#cmp-with').closest('#actions') != null, 'under Compare with, among its actions',
  p.text('#cmp-with summary'));
await p.click('#cmp-with button[data-value="b/report.json"]');
same([p.view(), compared(p)], ['v-compare', ['#1 a/report.json', '#2 b/report.json']], 'picking one opens the comparison of the two');
same(p.$$('#cmp-with button').map(b => b.dataset.value), ['', 'c/report.json'], 'where the same control offers the reports not yet in it');
ok(p.text('#cmp-with summary') === 'Add a report…', 'as Add a report', p.text('#cmp-with summary'));
await p.click('#cmp-with button[data-value="c/report.json"]');
same(compared(p), ['#1 c/report.json', '#2 a/report.json', '#3 b/report.json'], 'adding one places it by when it ran');
ok(p.$('#cmp-with').hidden, 'and with every report compared there is nothing left to add');

// ── two reports, one made in code with no experiment file ──
{
  const q = await served({
    'suite/experiment.json': experiment, 'suite/report.json': NEW,
    'nightly/results.json': report(0.75, [['a', 'X', 1, 'pass'], ['a', 'Y', 0.5, 'pass'], ['zz', 'X', 0.5, 'fail']], '2026-02-01T00:00:00Z'),
  });
  ok(/run from code/.test(q.text(box(q, 'nightly/results.json').closest('tr'))), 'a report with no experiment file is listed as run from code');
  await q.click(box(q, 'nightly/results.json'));
  await q.click(box(q, 'suite/report.json'));
  await q.click('#cmp-go');
  same(compared(q), ['#1 nightly/results.json', '#2 suite/report.json'], 'and compares like any other');
  ok(q.text('#cmp-ends-legend') === '0 now pass1 now fail1 kept their verdict3 only in #21 only in #1',
    'on the rows the two share', q.text('#cmp-ends-legend'));
  ok(!q.$('#facts [data-cmp-remove]') && q.text('#cmp-ends-head') === 'From #1 to #2', 'two reports are compared first against last');
  ok(q.text('#cmp-sub') === '2 reports, oldest first. Rows are matched by case and evaluator, so reports of different experiments '
    + 'compare only where they hold the same rows.', 'the page says how rows are matched', q.text('#cmp-sub'));
}

// ── a row that was not applicable in one report and judged in the other ──
{
  const q = await served({
    'old/report.json': report(0.5, [['declined', 'X', 1, 'na'], ['judged', 'X', 1, 'na'], ['lapsed', 'X', 1, 'pass'], ['idle', 'X', 1, 'na'], ['held', 'X', 1, 'pass']],
      '2026-03-01T09:00:00Z'),
    'new/report.json': report(0.5, [['declined', 'X', 0, 'fail'], ['judged', 'X', 1, 'pass'], ['lapsed', 'X', 1, 'na'], ['idle', 'X', 1, 'na'], ['held', 'X', 1, 'pass']],
      '2026-03-02T09:00:00Z'),
  });
  await q.click(box(q, 'old/report.json'));
  await q.click(box(q, 'new/report.json'));
  await q.click('#cmp-go');
  ok(q.text('#cmp-ends-legend') === '1 now pass1 now fail2 kept their verdict1 now not applicable',
    'a row judged for the first time counts as passing or failing now, and one no longer judged is counted apart', q.text('#cmp-ends-legend'));
  ok(q.text('#cmp-grid-cap').startsWith('3 of 5 rows changed verdict'), 'which is how many the grid above it says changed', q.text('#cmp-grid-cap'));
  same(q.tips('#cmp-ends g.bell'), ['judged · X · — to 1.00 · now passes', 'declined · X · — to 0.00 · now fails', 'lapsed · X · 1.00 to — · now not applicable'],
    'and each mark says which it was');
  const counts = q.window.compareMarkdown().split('\n').filter(line => /^- (Now|Kept)/.test(line));
  same(counts, ['- Now pass: 1', '- Now fail: 1', '- Kept their verdict: 2', '- Now not applicable: 1'],
    'the copied Markdown counts the same four ways');
}

// ── two rows of one report that share a case name and an evaluator ──
{
  const q = await served({
    'old/report.json': report(0.5, [['twin', 'X', 1, 'pass'], ['twin', 'X', 0, 'fail']], '2026-03-01T09:00:00Z'),
    'new/report.json': report(1, [['twin', 'X', 1, 'pass'], ['twin', 'X', 1, 'pass']], '2026-03-02T09:00:00Z'),
  });
  await q.click(box(q, 'old/report.json'));
  await q.click(box(q, 'new/report.json'));
  await q.click('#cmp-go');
  ok(q.text('#cmp-ends-legend') === '1 now pass0 now fail1 kept their verdict',
    'they are matched in the order the reports hold them, each against its own', q.text('#cmp-ends-legend'));
  ok(q.text('#cmp-grid-cap').startsWith('1 of 2 rows changed verdict') && q.$$('#cmp-grid .cell').length === 4,
    'and the grid draws both', `${q.text('#cmp-grid-cap')} ${q.$$('#cmp-grid .cell').length}`);
}

// ── reports that share no rows ──
{
  const q = await served({
    'one/report.json': report(1, [['a', 'X', 1, 'pass'], ['b', 'X', 1, 'pass']], '2026-01-01T00:00:00Z'),
    'two/report.json': report(0, [['m', 'Z', 0, 'fail'], ['n', 'Z', 0, 'fail']], '2026-01-02T00:00:00Z'),
  });
  await q.openReport('one /');
  await q.click('#cmp-with button[data-value="two/report.json"]');
  ok(q.view() === 'v-compare' && q.text('#cmp-ends-cap') === 'one/report.json against two/report.json. The two share no case and evaluator.',
    'reports with no row in common still open, and the page says they share nothing', q.text('#cmp-ends-cap'));
  ok(q.text('#cmp-ends-legend') === '2 only in #22 only in #1' && !q.$('#cmp-ends svg'),
    'nothing is counted as passing or failing now, only what each one holds', q.text('#cmp-ends-legend'));
  same(q.$$('#cmp-grid rect.cell').map(c => c.classList[1]), ['pass', 'pass', 'fail', 'fail'],
    'and the grid has a gap for every row the other report lacks');
  same(q.$$('#cmp-evals tbody tr').map(tr => [...tr.children].map(c => q.text(c))), [['X', '1.00', '—'], ['Z', '—', '0.00']],
    'an evaluator one report never ran has no mean there');
}

// ── one report: nothing to compare ──
{
  const q = await served({ 'only/experiment.json': experiment, 'only/report.json': NEW });
  await q.click(box(q, 'only/report.json'));
  await q.openReport();
  ok(q.$('#cmp-go').hidden && q.$('#cmp-hint').hidden && q.$('#cmp-with').hidden,
    'a folder with one report offers no comparison, in the list or on the report');
}

// ── served: the address is a path with the reports in its query ──
{
  const listed = (path, modified, overall) =>
    ({ path, name: path, dir: '.', size: 1, modified, rows: 2, failing: 0, overall_score: overall, experiment: null, link: null });
  const two = (first, second) => report(0.5, [['a', 'X', first, 'pass'], ['b', 'X', second, 'pass']]);
  const tree = {
    mark: 'm1',
    reports: [listed('late.json', 30, 0.9), listed('early.json', 10, 0.4), listed('mid.json', 20, 0.6)],
    files: { 'late.json': two(1, 0.8), 'early.json': two(0.5, 0.3), 'mid.json': two(0.6, 0.6) },
  };
  const modified = path => tree.reports.find(m => m.path === path).modified;
  const server = () => fakeServer({
    mark: () => tree.mark,
    scan: () => [{ root: '/served', depth: 3, read_only: false, fingerprint: tree.mark, experiments: [], reports: tree.reports,
      skipped: [], optimizer_runs: [] }],
    file: path => ({ path, modified: modified(path), document: tree.files[path] }),
  });

  const q = await page({ fetchImpl: server(), visible: true });
  await q.click(box(q, 'late.json'));
  await q.click(box(q, 'early.json'));
  await q.click('#cmp-go');
  same(compared(q), ['#1 early.json', '#2 late.json'], 'reports that do not record their run are ordered by their files\' times');
  ok(q.$$('#cmp-summary tr').every(tr => / · by file time$/.test(q.text(tr.children[1]))), 'and the summary says that is what dates them',
    q.text('#cmp-summary tr'));
  ok(here(q) === address('early.json', 'late.json') + '&t=tok', 'served, the comparison is a path naming its reports, beside whatever else the address held',
    here(q));

  // A re-run of a compared report, and one that disappears.
  tree.files['late.json'] = two(0.1, 0.8);
  tree.reports[0].modified = 31;
  tree.mark = 'm2';
  await q.poll();
  same(q.tips('#cmp-ends g.bell'), ['b · X · 0.30 to 0.80', 'a · X · 0.50 to 0.10'], 'a compared report that is run again is re-read where it stands');
  await q.click('.tab[data-view="reports"]');
  ok(here(q) === '/reports?t=tok', 'leaving the comparison takes its reports out of the address', here(q));
  await q.travel(-1);
  ok(q.view() === 'v-compare' && here(q) === address('early.json', 'late.json') + '&t=tok', 'and Back returns to it', `${q.view()} ${here(q)}`);
  tree.reports = tree.reports.filter(m => m.path !== 'early.json');
  tree.mark = 'm3';
  await q.poll();
  ok(q.view() === 'v-reports' && here(q) === '/reports?t=tok', 'a comparison left with one report when the other is deleted gives way to the list',
    `${q.view()} ${here(q)}`);

  // Opened from the address alone.
  tree.reports = [listed('late.json', 31, 0.9), listed('early.json', 10, 0.4), listed('mid.json', 20, 0.6)];
  const link = address('late.json', 'mid.json', 'early.json');
  const r = await page({ fetchImpl: server(), url: `http://127.0.0.1:9999${link}`, token: 'tok' });
  same([r.view(), compared(r)], ['v-compare', ['#1 early.json', '#2 mid.json', '#3 late.json']],
    'a link to a comparison opens it with nothing clicked');
  ok(here(r) === address('early.json', 'mid.json', 'late.json'), 'and the address is put in the order shown', here(r));
  same(cells(r, 3), ['—', '▲ +0.20', '▲ +0.30'], 'with each report set against the one before it');
  await r.click('#facts [data-cmp-remove="mid.json"]');
  ok(here(r) === address('early.json', 'late.json'), 'removing a report rewrites the address', here(r));
  await r.travel(-1);
  same(compared(r), ['#1 early.json', '#2 mid.json', '#3 late.json'], 'and Back puts it back');

  // A mark whose report can no longer be read.
  const unread = new Set();
  const answers = server();
  const s = await page({ fetchImpl: (url, opts) => (unread.has(new URL(String(url), 'http://x').searchParams.get('path'))
    ? { ok: false, status: 404, statusText: 'Not Found', json: async () => ({ error: 'no such file' }) } : answers(url, opts)) });
  await s.openReport('mid.json');
  await s.click('#cmp-with button[data-value="late.json"]');
  unread.add('late.json');
  await s.click('#cmp-ends g.bell');
  ok(s.view() === 'v-missing' && s.text('#missing') === 'This one would not surfacelate.json could not be read: no such fileAll reports',
    'a mark whose report cannot be read says so on a page of its own', s.text('#missing'));
  await s.travel(-1);
  await s.travel(-1);
  ok(s.view() === 'v-report' && s.text('#crumbs') === 'Reports/mid.json' && !s.$('#rep-rows tr.case[aria-expanded="true"]'),
    'and opens no row in the report that was open before', `${s.view()} ${s.text('#crumbs')}`);

  // A compared report that changes and cannot be read again.
  unread.clear();
  const t = await page({ fetchImpl: s.window.fetch, url: `http://127.0.0.1:9999${address('mid.json', 'late.json')}`, token: 'tok', visible: true });
  unread.add('late.json');
  tree.reports[0].modified += 1;
  tree.mark = 'm4';
  await t.poll();
  ok(t.view() === 'v-compare' && here(t) === address('mid.json', 'late.json') && compared(t).join() === '#1 mid.json,#2 late.json',
    'a compared report that cannot be re-read leaves the comparison as it was read', `${t.view()} ${here(t)}`);
  await t.click('.tab[data-view="reports"]');
  ok(/could not re-read: no such file/.test(t.text('#v-reports')), 'and the list says the folder could not be re-read', t.text('#scan-err'));

  // An address that names too few reports to compare.
  const one = await page({ fetchImpl: server(), url: 'http://127.0.0.1:9999/reports/compare?r=mid.json', token: 'tok' });
  same([one.view(), here(one), one.text('#missing')],
    ['v-missing', '/reports/compare?r=mid.json', 'It takes two lily pads to compareA comparison sets two or more reports side by side, '
      + 'and this address names only mid.json.Open the reportAll reports'],
    'a comparison of one report says what a comparison needs, at the address as typed');
  await one.click('#missing .ways a');
  ok(one.view() === 'v-report' && one.text('#crumbs') === 'Reports/mid.json', 'and offers the report it names', one.view());
}

done();
