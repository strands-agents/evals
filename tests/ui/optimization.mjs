/* The Optimization view, driven from the committed fixture run.
 *
 * The numbers on this tab are *derived* — best-so-far, the budget curve, the status breakdown —
 * so the way they can be wrong is to be plausible. The assertions below are written against the
 * fixture's plan rather than against whatever the page currently draws: the best feasible
 * objective is c10's 0.671 and never c8's higher-but-infeasible 0.690, a candidate that was never
 * scored is absent from the chart rather than sitting at zero, and the budget line is cumulative.
 */
import fs from 'node:fs';
import { page, runner } from './harness.mjs';

const { ok, done } = runner('optimization');

const DIR = new URL('./fixtures/optimizer-run/', import.meta.url);
const EVENTS = fs.readFileSync(new URL('events.jsonl', DIR), 'utf8');
const CONFIG = fs.readFileSync(new URL('run.json', DIR), 'utf8');

const skips = p => p.$$('#skips li').map(li => li.textContent.replace(/\s+/g, ' ').trim());

const p = await page();                       // no fetch -> standalone
const { $, $$, text, click, openFolder } = p;

// ── the tab only exists when there is a run behind it ──
ok($('#tab-opt').hidden === true, 'Optimization tab hidden with no optimizer run');

await openFolder({ 'demo/run.json': CONFIG, 'demo/events.jsonl': EVENTS });
ok($('#tab-opt').hidden === false, 'tab appears once a run directory is read');
ok($('#opt-run').hidden === true, 'the run picker stays hidden for a single run');
ok(skips(p).length === 0, 'neither file of a recognised run is reported as unusable', skips(p).join(' | '));

await click('#tab-opt');
ok($('#v-optimization').classList.contains('on'), 'the tab opens the view');
ok(text('#path') === 'demo/events.jsonl', 'the path bar names the log', text('#path'));
ok(/support-tuning-20261110-091500-7af3e1/.test(text('#opt-sub')), 'names the run', text('#opt-sub'));
ok(/11 proposals over 6 iterations/.test(text('#opt-sub')), 'counts proposals and iterations', text('#opt-sub'));
ok(/Stopped: iterations/.test(text('#opt-sub')), 'reports the stop reason', text('#opt-sub'));
ok(/WeightedSum/.test(text('#opt-objective')), 'shows the objective', text('#opt-objective'));

// ── the hero number: best FEASIBLE, which is not the best scored ──
ok(text('#opt-best') === '0.67', 'best feasible objective is c10, not infeasible c8', text('#opt-best'));
ok(/c10/.test(text('#opt-best-cap')), 'and names the candidate it came from', text('#opt-best-cap'));

const stats = $$('#opt-stats .stat').map(s =>
  s.querySelector('.k').textContent + '=' + s.querySelector('.v').textContent);
ok(stats.includes('Champion=c10'), 'champion read from the end-phase report', stats.join(' '));
ok(stats.includes('Rollouts=64 / 240'), 'rollouts are the iterations\' own tally against the budget', stats.join(' '));
ok(stats.includes('Scored=8'), 'eight candidates were scored', stats.join(' '));
ok(stats.includes('Not scored=3'), 'and three never were', stats.join(' '));

// ── the objective chart ──
const pts = $$('#opt-obj-chart circle');
ok(pts.length === 8, 'one mark per SCORED proposal, not per proposal', pts.length);
const titles = pts.map(c => c.querySelector('title').textContent);
ok(!titles.some(t => /^c4 |^c7 |^c9 /.test(t)), 'unscored, invalid and duplicate candidates are not plotted', titles.join(' | '));
ok(titles.some(t => t.startsWith('c8 · infeasible · 0.69')), 'the infeasible candidate IS plotted', titles.join(' | '));

const cls = name => {
  const i = titles.findIndex(t => t.startsWith(name + ' '));
  return i < 0 ? 'absent' : pts[i].getAttribute('class');
};
ok(/infeasible/.test(cls('c8')), 'c8 is marked infeasible', cls('c8'));
ok(/champion/.test(cls('c10')), 'the champion is marked as such rather than merely feasible', cls('c10'));
ok(/feasible/.test(cls('c6')) && !/infeasible/.test(cls('c6')), 'an ordinary feasible candidate', cls('c6'));

/* The best-so-far line. In SVG a larger value is a smaller y, so improvement is a decreasing y
   and a chart that counted c8 would dip below c10's mark somewhere. */
const path = $('#opt-obj-chart path.best');
ok(!!path, 'the best-so-far line is drawn');
const points = (path.getAttribute('d') || '').split(/[ML]/).filter(Boolean)
  .map(seg => seg.split(',').map(Number));
const ys = points.map(([, y]) => y);
ok(ys.length > 1, 'the line has more than one point', ys.length);
ok(ys.every((y, i) => i === 0 || y <= ys[i - 1] + 1e-9), 'best-so-far never gets worse', ys.join(' '));
/* Every segment is horizontal or vertical. A diagonal between two candidates would draw a best
   that rose gradually between them, which is a measurement nobody took. */
const diagonal = points.findIndex(([x, y], i) =>
  i > 0 && Math.abs(x - points[i - 1][0]) > 1e-9 && Math.abs(y - points[i - 1][1]) > 1e-9);
ok(diagonal < 0, 'the line steps rather than sloping', diagonal < 0 ? '' : `segment ${diagonal}`);

const cy = name => {
  const i = titles.findIndex(t => t.startsWith(name + ' '));
  return i < 0 ? NaN : Number(pts[i].getAttribute('cy'));
};
const highest = Math.min(...ys);
ok(Math.abs(highest - cy('c10')) < 1e-6, 'the line tops out at c10, the best feasible candidate',
  `line ${highest} vs c10 ${cy('c10')}`);
ok(highest > cy('c8') + 1e-6, 'and stays strictly below c8, which broke a constraint',
  `line ${highest} vs c8 ${cy('c8')}`);

/* c0 is the first proposal and is feasible, so the line starts at its mark; a rule that seeded
   the running best with 0 instead of null would start the line at the bottom of the axis. */
ok(Math.abs(ys[0] - cy('c0')) < 1e-6, 'the line starts at the first feasible candidate',
  `line ${ys[0]} vs c0 ${cy('c0')}`);

// ── the budget chart ──
const budget = $$('#opt-budget-chart circle').map(c => c.querySelector('title').textContent);
ok(budget.length === 6, 'one mark per finished iteration', budget.length);
ok(budget[0] === 'iteration 0: +8, 8 total', 'first iteration', budget[0]);
ok(budget[5] === 'iteration 5: +8, 64 total', 'last iteration is cumulative, not per-iteration', budget[5]);
const cumulative = budget.map(t => Number(t.match(/(\d+) total/)[1]));
ok(cumulative.every((v, i) => i === 0 || v >= cumulative[i - 1]), 'cumulative never decreases', cumulative.join(' '));
ok(!!$('#opt-budget-chart line.cap-line'), 'the budget cap is drawn');
ok(/240-rollout budget/.test(text('#opt-budget-cap')), 'and named in the caption', text('#opt-budget-cap'));

// ── the status breakdown ──
const bars = $$('#opt-status-chart .legend span').map(s => s.textContent.replace(/\s+/g, ' ').trim());
ok(bars.some(b => b === 'completed 80'), '80 trials completed', bars.join(' | '));
ok(bars.some(b => b === 'agent failure 1'), 'the agent failure is counted', bars.join(' | '));
ok(bars.some(b => b === 'evaluator invalid 1'), 'and the evaluator failure too', bars.join(' | '));
ok(bars.some(b => /train 64/.test(b) && /validation 12/.test(b) && /test 6/.test(b)),
  'splits broken out', bars.join(' | '));
ok(/82 trials/.test(text('#opt-sub')), 'the subtitle counts every trial, not just the training ones', text('#opt-sub'));

// ── the proposals table ──
const rows = $$('#opt-rows tr').map(r => [...r.children].map(c => c.textContent.replace(/\s+/g, ' ').trim()));
ok(rows.length === 11, 'every proposal gets a row, scored or not', rows.length);
const byName = name => rows.find(r => r[0].startsWith(name));
ok(byName('c4')[2] === 'not scored', 'an unscored candidate says so rather than showing 0.00', byName('c4')[2]);
ok(byName('c8')[2] === '0.69' && byName('c8')[3] === 'infeasible', 'c8 shows its score and its outcome', byName('c8').join(' / '));
ok(/champion/.test(byName('c10')[0]), 'the champion is tagged in the table', byName('c10')[0]);
const invalid = rows.find(r => /no candidate/.test(r[0]));
ok(!!invalid && invalid[3] === 'invalid', 'the invalid patch is shown as having no candidate', (invalid || []).join(' / '));
ok(/not in the registry/.test(invalid[5]), 'with the reason it was rejected', invalid[5]);

ok(text('#opt-note') === '', 'no warnings for a clean log', text('#opt-note'));

// ── a log that is still being written, and one with a broken line ──
const p2 = await page();
const lines = EVENTS.split('\n').filter(Boolean);
await p2.openFolder({
  'demo/run.json': CONFIG,
  'demo/events.jsonl': lines.slice(0, 40).join('\n') + '\n' + lines[40].slice(0, 30),
});
await p2.click('#tab-opt');
ok(p2.text('#opt-note') === '', 'a half-written last line is not reported as damage', p2.text('#opt-note'));
ok(/Still running/.test(p2.text('#opt-sub')), 'a log with no stop event is still running', p2.text('#opt-sub'));
ok(p2.$$('#opt-obj-chart circle').length > 0, 'and it still draws what it has');

const p3 = await page();
await p3.openFolder({
  'demo/run.json': CONFIG,
  'demo/events.jsonl': lines.slice(0, 5).join('\n') + '\n{"event": "tri\n' + lines.slice(5).join('\n') + '\n',
});
await p3.click('#tab-opt');
ok(/line 6/.test(p3.text('#opt-note')), 'a broken line in the middle is reported with its number', p3.text('#opt-note'));
ok(p3.$('#opt-note').className.includes('warn'), 'and flagged');

// ── a run directory missing half of itself is not a run ──
const p4 = await page();
await p4.openFolder({ 'demo/run.json': CONFIG });
ok(p4.$('#tab-opt').hidden === true, 'a run.json with no event log is not an optimizer run');
ok(skips(p4).some(s => /demo\/run\.json/.test(s)),
  'and it is reported as unusable rather than vanishing', skips(p4).join(' | '));

const p5 = await page();
await p5.openFolder({ 'demo/events.jsonl': EVENTS });
ok(p5.$('#tab-opt').hidden === true, 'an event log with no config is not an optimizer run either');
ok(skips(p5).some(s => /no run\.json beside it/.test(s)),
  'and says what it was missing', skips(p5).join(' | '));

/* ── a run that starts with nothing to measure ──
   The fixture's first candidate is feasible, so it cannot tell a line that starts at the first
   *scored* candidate from one that starts at the left edge. This log opens with an unscored
   proposal, where the two differ. */
const jsonl = events => events.map(e => JSON.stringify(e)).join('\n') + '\n';
const iterationEnd = (iteration, proposals, rollouts) => ({
  event: 'iteration_end', iteration,
  record: { iteration, proposals, rollouts_spent: rollouts },
});

const pLate = await page();
await pLate.openFolder({
  'late/run.json': { run_id: 'late-start', objective: 'Maximize(H)' },
  'late/events.jsonl': jsonl([
    { event: 'config', iteration: 0 },
    iterationEnd(0, [{ candidate_id: 'c0', number: 0, outcome: 'unscored', objective: null }], 0),
    iterationEnd(1, [{ candidate_id: 'c1', number: 1, outcome: 'feasible', objective: 0.4 }], 4),
    iterationEnd(2, [{ candidate_id: 'c2', number: 2, outcome: 'feasible', objective: 0.6 }], 4),
  ]),
});
await pLate.click('#tab-opt');
const latePts = pLate.$$('#opt-obj-chart circle');
ok(latePts.length === 2, 'the unscored opening proposal is not plotted', latePts.length);
const lateLine = pLate.$('#opt-obj-chart path.best');
ok(!!lateLine, 'a line is still drawn');
const lateXs = (lateLine.getAttribute('d') || '').split(/[ML]/).filter(Boolean)
  .map(seg => Number(seg.split(',')[0]));
ok(Math.abs(Math.min(...lateXs) - Number(latePts[0].getAttribute('cx'))) < 1e-6,
  'the line starts at the first scored candidate, not at the left edge',
  `line ${Math.min(...lateXs)} vs first mark ${latePts[0].getAttribute('cx')}`);
ok(pLate.text('#opt-best') === '0.60', 'best is the better of the two measured', pLate.text('#opt-best'));

// ── a run with nothing measured at all, and a Pareto run ──
const pEmpty = await page();
await pEmpty.openFolder({
  'empty/run.json': { run_id: 'just-started', objective: 'Maximize(H)', is_pareto: true },
  'empty/events.jsonl': jsonl([{ event: 'config', iteration: 0 }]),
});
await pEmpty.click('#tab-opt');
ok(pEmpty.$('#tab-opt').hidden === false, 'a run with only a config event is still a run');
ok(pEmpty.text('#opt-best') === '—', 'no best objective is a dash, not 0.00', pEmpty.text('#opt-best'));
ok(/No candidate has been scored/.test(pEmpty.text('#opt-obj-chart')), 'says why the chart is empty',
  pEmpty.text('#opt-obj-chart'));
ok(/No iteration has finished/.test(pEmpty.text('#opt-budget-chart')), 'and the budget chart too',
  pEmpty.text('#opt-budget-chart'));
ok(/No proposals recorded/.test(pEmpty.text('#opt-rows')), 'and the table', pEmpty.text('#opt-rows'));
ok(/Pareto/.test(pEmpty.text('#opt-note')), 'a Pareto run is flagged, because one best understates it',
  pEmpty.text('#opt-note'));

// ── two runs in one tree ──
const p6 = await page();
await p6.openFolder({
  'a/run.json': CONFIG,
  'a/events.jsonl': EVENTS,
  'b/run.json': JSON.stringify({ ...JSON.parse(CONFIG), run_id: 'second-run', name: 'other' }),
  'b/events.jsonl': lines.slice(0, 20).join('\n') + '\n',
});
ok(p6.$('#opt-run').hidden === false, 'the picker appears for more than one run');
ok(p6.$$('#opt-run option').length === 2, 'both runs offered', p6.$$('#opt-run option').length);
await p6.click('#tab-opt');
const first = p6.text('#opt-sub');
const picker = p6.$('#opt-run');
picker.value = '1';
picker.dispatchEvent(new p6.window.Event('change', { bubbles: true }));
await new Promise(r => setTimeout(r, 20));
ok(p6.text('#opt-sub') !== first, 'picking the other run changes the view', p6.text('#opt-sub'));
ok(/second-run/.test(p6.text('#opt-sub')), 'and names it', p6.text('#opt-sub'));

done([p, p2, p3, p4, p5, p6, pLate, pEmpty].flatMap(each => each.errs));
