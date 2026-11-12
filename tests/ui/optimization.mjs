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
ok($('#opt-run-label').hidden === true, 'and so does its label, which would otherwise label nothing');
ok(skips(p).length === 0, 'neither file of a recognised run is reported as unusable', skips(p).join(' | '));

await click('#tab-opt');
ok($('#v-optimization').classList.contains('on'), 'the tab opens the view');
ok(text('#path') === 'demo/events.jsonl', 'the path bar names the log', text('#path'));
ok(/support-tuning-20261110-091500-7af3e1/.test(text('#opt-sub')), 'names the run', text('#opt-sub'));
ok(/11 proposals over 6 iterations/.test(text('#opt-sub')), 'counts proposals and iterations', text('#opt-sub'));
ok(/Stopped: iterations/.test(text('#opt-sub')), 'reports the stop reason', text('#opt-sub'));
ok(/WeightedSum/.test(text('#opt-objective')), 'shows the objective', text('#opt-objective'));

/* ── what the optimizer did, before anything it produced ──
   Every number on this tab is unreadable without the method: the search only ever sees train,
   the pick happens once on validation, and test is measured but never optimized against. The
   `Optimizer` docstring is explicit that the experiment "is never run itself", which is the one
   thing a reader coming from the Runs tab will assume the opposite of. */
const why = text('#opt-explain');
ok(/built 10 versions of your agent/.test(why),
  'the strip counts the configurations it built, not the 11 proposals it drew', why);
ok(/varying system_prompt, temperature, tool_descriptions, max_tokens/.test(why),
  'and names what it actually varied, taken from the proposals rather than the configured blocks', why);
ok(/was never run itself/.test(why),
  'and says the experiment supplied the cases and was not itself run — the thing a reader '
  + 'arriving from the Runs tab assumes the opposite of', why);
ok(/split 8 train · 4 validation · 3 test/.test(why), 'with the split sizes', why);
ok(/searched on train/.test(why) && /picked the champion once on validation/.test(why)
  && /measured test only to report it/.test(why),
  'and what each split was for, which is the whole method', why);

/* ── the result, which is what the page opens on ──
   The question a tuning run answers is whether the champion beats the baseline on cases the
   search never saw. That lives in the `test` event, and every figure in this card has a wrong
   neighbour in the log it could have come from instead: the training best (0.671), the validation
   objective (0.658), or a mean over the cases the search chose on. */
const rtab = () => $$('#opt-result .rtab tr').map(r =>
  [...r.children].map(c => c.textContent.replace(/\s+/g, ' ').trim()));
ok(/^Held out/.test(text('#opt-result .rlabel')), 'the headline is the held-out comparison',
  text('#opt-result .rlabel'));
ok(/3 test cases kept out of the search/.test(text('#opt-result .rlabel')),
  'and says how many cases are behind it', text('#opt-result .rlabel'));
const helpRow = rtab().find(r => r[0] === 'Helpfulness');
ok(helpRow && helpRow.slice(1).join(' | ') === '0.521 | → | 0.664 | +0.143 | [0.061, 0.225] | 1.274×',
  'the champion against the baseline on the test split, with its interval and ratio',
  (helpRow || []).join(' | '));
const toks = rtab().find(r => r[0] === 'tokens');
ok(toks && toks.slice(1).join(' | ') === '3190 | → | 3340 | +150 |  | ',
  'a metric the run had no interval for shows the means and the delta and stops there',
  (toks || []).join(' | '));
ok(rtab().length === 2, 'a row per goal metric, in the order the run declared them', rtab().length);
/* `tokens` went UP by 150, which is worse — the objective minimises it. Direction lives in the
   objective expression, so a green +150 would be the page inventing one from the metric's name. */
ok($$('#opt-result .delta').every(td => /flat/.test(td.className)),
  'no per-metric delta is coloured, because only the objective has a known direction',
  $$('#opt-result .delta').map(td => td.className).join(' | '));
const verdict = text('#opt-result .verdict');
ok(/improved by 0\.143/.test(verdict), 'the verdict quotes the objective difference', verdict);
ok(/\[0\.061, 0\.225\] clears zero/.test(verdict), 'and reads the interval as the verdict it is', verdict);
ok(/never saw/.test(verdict), 'and says the cases were held out', verdict);
ok(!/flattering/.test(verdict), 'a held-out result is not hedged as a flattering one', verdict);

// ── the winner: the configuration the run exists to produce, not just the name of it ──
ok(text('#opt-winner .wid') === 'c10', 'the champion is named', text('#opt-winner .wid'));
ok(text('#opt-winner .lineage') === 'c10 ← c6 ← c3 ← c1 ← c0',
  'with the chain of parents it was derived through, back to the baseline',
  text('#opt-winner .lineage'));
const changedKeys = $$('#opt-winner .setting .skey').map(s => s.textContent);
ok(changedKeys.join() === 'system_prompt,temperature',
  'only the settings that actually differ from the baseline are shown', changedKeys.join());
ok(/2 other settings unchanged from c0/.test(text('#opt-winner .unchanged')),
  'the ones that did not move are counted rather than listed', text('#opt-winner .unchanged'));
/* A tuned system prompt is many lines. Rendered as `was → now` on one line it is unreadable, so
   a long or multi-line value gets a block diff instead. */
const cell = (key, sel) => {
  const found = $$('#opt-winner .setting').find(s => new RegExp(key).test(s.textContent));
  return (found && (sel ? found.querySelector(sel) : found)) || null;
};
const block = cell('system_prompt', 'pre');
ok(!!block, 'a multi-line value is diffed as a block, not inline');
const diff = (block || {}).textContent || '';
ok(/^- You are the support assistant/m.test(diff) && /^\+ You are the support assistant/m.test(diff),
  'with the baseline marked - and the champion +', diff.slice(0, 80));
ok(/NW-000000/.test((cell('system_prompt', 'pre .ins') || {}).textContent || '')
  && !/NW-000000/.test((cell('system_prompt', 'pre .del') || {}).textContent || ''),
  'and the real text on each side, not a summary of it');
ok(!cell('temperature', 'pre') && text('#opt-winner .setting .inline') === '0.7 → 0.2',
  'a short value stays inline', ((cell('temperature') || {}).textContent || '').replace(/\s+/g, ' ').trim());

/* Copy and download hand the configuration off. jsdom has no clipboard, so stub one in: the
   point of the assertion is WHAT gets handed over, which no amount of UI review can check. */
let copied = null;
Object.defineProperty(p.window.navigator, 'clipboard', {
  configurable: true, value: { writeText: async t => { copied = t; } },
});
await click('#opt-winner [data-act="copy-winner"]');
ok(/copied 4 settings/.test(text('#opt-copied')), 'copying says what it put on the clipboard',
  text('#opt-copied'));
ok(JSON.stringify(JSON.parse(copied)) === JSON.stringify({
  system_prompt: JSON.parse(copied).system_prompt, temperature: 0.2,
  tool_descriptions: { order_lookup: 'Look up an order.', refund_start: 'Start a refund.' },
  max_tokens: 1024,
}), 'the whole configuration is copied, inherited settings included, not just what changed', copied);
ok(/NW-000000/.test(JSON.parse(copied).system_prompt) && JSON.parse(copied).temperature === 0.2,
  'and it is the champion\'s values, not the baseline\'s', JSON.parse(copied).temperature);
await click('#opt-winner [data-act="save-winner"]');
ok(p.window.__downloads.includes('champion.json'), 'download offers it as a file',
  p.window.__downloads.join(' | '));
ok(/downloaded champion.json/.test(text('#opt-copied')), 'and says so', text('#opt-copied'));

/* With no clipboard at all the button must not look like it worked. */
delete p.window.navigator.clipboard;
await click('#opt-winner [data-act="copy-winner"]');
ok(/could not reach the clipboard/.test(text('#opt-copied')),
  'a copy that could not happen says so rather than claiming success', text('#opt-copied'));

// ── the hero number: best FEASIBLE, which is not the best scored ──
ok(text('#opt-best') === '0.67', 'best feasible objective is c10, not infeasible c8', text('#opt-best'));
ok(/c10/.test(text('#opt-best-cap')), 'and names the candidate it came from', text('#opt-best-cap'));

const stats = $$('#opt-stats .stat').map(s =>
  s.querySelector('.k').textContent + '=' + s.querySelector('.v').textContent);
ok(stats.includes('Champion=c10'), 'champion read from the end-phase report', stats.join(' '));
ok(stats.includes('Rollouts=64 / 240'), 'rollouts are the iterations\' own tally against the budget', stats.join(' '));
ok(stats.length === 3, 'three figures in the headline, not every number the run produced', stats.join(' '));

/* The rest is diagnostics, so it starts folded away — the page opens on the objective and one
   metric. What is inside each drawer is summarised on the fold, including the scored-against-
   unscored split that used to compete for space in the headline. */
ok(['#opt-search', '#opt-cost', '#opt-diag'].every(id => $(id).open === false),
  'the search and diagnostics drawers start closed');
ok(/11 proposals over 6 iterations/.test(text('#opt-search-sum')),
  'the search drawer says how much work is folded inside it', text('#opt-search-sum'));
ok(text('#opt-cost-sum') === '277k tokens', 'the cost drawer says what the run cost', text('#opt-cost-sum'));
ok(text('#opt-diag-sum') === '82 trials', 'the diagnostics drawer counts the trials', text('#opt-diag-sum'));

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

/* ── one metric chart with a picker, not a panel per metric ──
   A panel each put a wall of charts on screen for something read one at a time. The picker has
   to carry the same guarantees the panels did: every metric on offer, its own axis units, and a
   redraw that actually changes the chart. */
const metricMarks = () => $$('#opt-metric-chart circle');
const metricTitles = () => metricMarks().map(c => c.querySelector('title').textContent);
const metricTicks = () => $$('#opt-metric-chart text.tick')
  .map(t => Number(t.textContent)).filter(v => !Number.isNaN(v));
const axlab = () => $$('#opt-metric-chart .axlab').map(t => t.textContent).join();
const pick = async (sel, value) => {
  $(sel).value = value;
  $(sel).dispatchEvent(new p.window.Event('change', { bubbles: true }));
  await new Promise(r => setTimeout(r, 20));
};

ok($$('#opt-metric-pick option').map(o => o.value).join() === 'Helpfulness,tokens',
  'every measured metric is on offer, in the order run.json declares its goal metrics',
  $$('#opt-metric-pick option').map(o => o.value).join());
ok($('#opt-metric-pick').value === 'Helpfulness', 'the first goal metric is what you land on',
  $('#opt-metric-pick').value);

const help = metricTitles();
ok(help.length === 8, 'one mark per candidate the metric was measured on', help.length);
ok(!help.some(t => /^c4 |^c7 |^c9 /.test(t)),
  'a candidate with no measurement is absent, not drawn at zero', help.join(' | '));
/* `label.measured` is the mean the search itself acted on, so it wins over the mean of the
   candidate's trials — c0's trials average 0.5199 where its recorded mean is 0.512, and c1's
   0.5555 against 0.548, so the two round to different numbers on screen. */
ok(help[0] === 'c0 · Helpfulness 0.51', 'the recorded mean, not the average of the trials', help[0]);
ok(help[1] === 'c1 · Helpfulness 0.55', 'and again where the two round apart', help[1]);
ok(help[7] === 'c10 · Helpfulness 0.67', 'the champion is measured like any other', help[7]);
ok(axlab() === 'Helpfulness', 'the axis is labelled with what it shows', axlab());
ok(Math.max(...metricTicks()) < 1, 'and scaled to the metric alone, in score units',
  metricTicks().join(' '));

/* The joining line stops at a candidate that was never measured. Drawn straight through, it
   would read as a value passing through a candidate nothing was recorded for. */
const segs = $$('#opt-metric-chart path.metric-line')
  .map(pth => pth.getAttribute('d').split(/[ML]/).filter(Boolean).map(s => s.split(',').map(Number)));
ok(segs.length === 2, 'the line breaks rather than spanning the gaps', segs.length);
ok(segs.map(s => s.length).join() === '4,2',
  'c0-c3 and c5-c6 are joined; c8 and c10 stand alone past c7 and c9', segs.map(s => s.length).join());
const markXs = metricMarks().map(c => Number(c.getAttribute('cx')));
const slot = Math.min(...markXs.slice(1).map((x, i) => x - markXs[i]));
const reach = segs.flatMap(s => s.slice(1).map(([x], i) => x - s[i][0])).find(dx => dx > slot + 1e-6);
ok(reach === undefined, 'no segment reaches over a candidate it has no value for', reach);

// Switching metric redraws in that metric's own units, not on an axis shared with the score.
await pick('#opt-metric-pick', 'tokens');
const tokenMarks = metricTitles();
ok(tokenMarks[0] === 'c0 · tokens 3180', 'a metric that is a count keeps its decimals off', tokenMarks[0]);
ok(tokenMarks[6] === 'c8 · tokens 4210', 'including the candidate that blew the 4000-token constraint', tokenMarks[6]);
ok(/infeasible/.test(metricMarks()[6].getAttribute('class')), 'which is marked infeasible here too',
  metricMarks()[6].getAttribute('class'));
ok(axlab() === 'tokens', 'the axis label follows the picker', axlab());
ok(Math.min(...metricTicks()) > 1000, 'and the axis is in token units', metricTicks().join(' '));
ok(/mean recorded for tokens/.test(text('#opt-metric-cap')), 'as does the caption', text('#opt-metric-cap'));
await pick('#opt-metric-pick', 'Helpfulness');
ok(metricTitles()[0] === 'c0 · Helpfulness 0.51', 'and switching back restores the first',
  metricTitles()[0]);

/* ── what each iteration cost, in the folded-away cost drawer ──
   One chart and a unit picker: tokens, calls and seconds are the same shape in almost every run,
   so three panels said it three times. */
ok($$('#opt-usage-pick option').map(o => o.textContent).join() === 'Tokens,Model calls,Seconds',
  'every usage measure the trials recorded is on offer',
  $$('#opt-usage-pick option').map(o => o.textContent).join());
ok($('#opt-usage-pick').value === 'tokens', 'tokens is the unit you land on', $('#opt-usage-pick').value);

const cols = () => $$('#opt-usage-chart rect.col');
const colTitles = () => cols().map(r => r.querySelector('title').textContent);
const tokCols = colTitles();
ok(tokCols.length === 6, 'a column per finished iteration', tokCols.length);
// The drawer is shut, so this is also the check that a shut drawer is drawn rather than deferred.
ok($('#opt-cost').open === false, 'and it is drawn while the drawer is still closed');
ok(tokCols[0] === 'iteration 0: 25440 over 8 trials', 'the iteration total over its own trials', tokCols[0]);
ok(tokCols[1] === 'iteration 1: 50640 over 16 trials',
  'an iteration that evaluated two candidates cost about twice as much', tokCols[1]);
ok(tokCols[5] === 'iteration 5: 26800 over 8 trials',
  'and the last column is that iteration alone, not a running total', tokCols[5]);
const tokVals = tokCols.map(t => Number(t.match(/: (\d+) /)[1]));
ok(tokVals.some((v, i) => i > 0 && v < tokVals[i - 1]),
  'the series falls as well as rises, so it is per-iteration and not cumulative', tokVals.join(' '));

/* The 18 trials after the last `iteration_end` — finalists and the test split — belong to no
   iteration. Folded into the last column they would overstate it by more than double. */
ok(tokVals.reduce((a, b) => a + b, 0) === 217480, 'the columns are the search cost, 217480 tokens',
  tokVals.reduce((a, b) => a + b, 0));
ok(/excluding the 18 trials recorded after the last finished iteration/.test(text('#opt-usage-cap')),
  'the caption says what is left out', text('#opt-usage-cap'));
ok(stats.includes('Tokens=277k'), 'while the stat is the whole run, 217480 plus the 59190 spent after it',
  stats.join(' '));

// Height follows the number: a column is only as tall as its share of the costliest iteration.
const heights = cols().map(r => Number(r.getAttribute('height')));
ok(heights[3] === Math.max(...heights), 'the tallest column is the costliest iteration', heights.join(' '));
ok(Math.abs(heights[1] / heights[3] - tokVals[1] / tokVals[3]) < 1e-6, 'heights are proportional to the totals',
  `${heights[1]}/${heights[3]} vs ${tokVals[1]}/${tokVals[3]}`);

await pick('#opt-usage-pick', 'model_calls');
ok(colTitles()[0] === 'iteration 0: 24 over 8 trials', 'a count of calls is a whole number', colTitles()[0]);
await pick('#opt-usage-pick', 'seconds');
ok(colTitles()[0] === 'iteration 0: 33.60 over 8 trials', 'and seconds are not', colTitles()[0]);

/* ── the candidates, browsable the way the Report tab's cases are ──
   A candidate IS an evaluation run of one configuration — `RunDirectory.reports()` rebuilds its
   trials into `EvaluationReport`s "as `Experiment` would return them". So the row opens onto what
   it changed and how every case went, and the figures in it have to come from the trials rather
   than from the summary the search kept. */
const candRows = () => $$('#opt-cands tr.case')
  .map(r => [...r.children].map(c => c.textContent.replace(/\s+/g, ' ').trim()));
const ids = () => $$('#opt-cands tr.case').map(r => r.children[0].textContent.replace(/\s+/g, ' ').trim());
const rows = candRows();
ok(rows.length === 11, 'every proposal gets a row, scored or not', rows.length);
ok(text('#opt-cand-count') === '11 of 11 candidates, 3 never scored',
  'the count carries the scored-against-unscored split', text('#opt-cand-count'));
const byName = name => rows.find(r => r[0].startsWith(name));
ok(byName('c4')[2] === 'not scored', 'an unscored candidate says so rather than showing 0.00', byName('c4')[2]);
ok(!$$('#opt-cands tr.case').find(r => /^c4/.test(r.textContent)).querySelector('.bar2'),
  'and draws no bar, which at zero width would read as a measurement that came out worst');
ok(byName('c8')[2] === '0.69' && byName('c8')[3] === 'infeasible',
  'c8 shows its score and its outcome', byName('c8').join(' / '));
ok(/champion/.test(byName('c10')[0]), 'the champion is tagged in the table', byName('c10')[0]);
ok(/baseline/.test(byName('c0')[0]), 'and the baseline, which is where every diff is measured from',
  byName('c0')[0]);
/* `changed_keys` is the diff against the PARENT, and c5 changed only its tool descriptions while
   differing from the baseline in three settings. The column has to be the parent diff it is. */
ok(byName('c5')[4] === 'tool_descriptions',
  'the column is what the proposal changed about its parent', byName('c5')[4]);
const invalid = rows.find(r => /no candidate/.test(r[0]));
ok(!!invalid && invalid[3] === 'invalid', 'the invalid patch is shown as having no candidate',
  (invalid || []).join(' / '));

// ── opening a candidate: its configuration against the baseline, and every case it ran ──
const openCand = async id => {
  await click($$('#opt-cands tr.case').find(r => r.children[0].textContent.trim().startsWith(id)));
};
const dl = () => {
  const row = $('#opt-cands tr.detail');
  return !row ? {} : Object.fromEntries([...row.querySelectorAll('dt')].map(dt =>
    [dt.textContent, dt.nextElementSibling.textContent.replace(/\s+/g, ' ').trim()]));
};
await openCand('c10');
ok(!!$('#opt-cands tr.detail'), 'a candidate row opens');
ok(/Combine c6's up-front id request/.test(dl()['Why it was tried'] || ''),
  'the rationale the proposer gave is what it leads with', dl()['Why it was tried']);
ok(dl()['Derived from'] === 'c6', 'with the parent it was derived from', dl()['Derived from']);
/* The diff is against the BASELINE, not the parent — how far it has moved from what you already
   had is what you decide with, and a parent is itself several steps from that. c10's parent c6
   has the same temperature as c3, so a parent diff would show temperature as unchanged. */
ok('Versus c0' in dl(), 'and the configuration diffed against the baseline, labelled as such',
  Object.keys(dl()).join(' | '));
const keys = () => $$('#opt-cands tr.detail .setting .skey').map(s => s.textContent);
ok(keys().join() === 'system_prompt,temperature', 'only the settings that actually differ',
  keys().join());
ok(/0\.7 → 0\.2/.test(dl()['Versus c0']), 'a short value inline, baseline first',
  dl()['Versus c0'].slice(0, 120));
const pre = $('#opt-cands tr.detail .setting pre');
ok(!!pre && /^- You are the support assistant/m.test(pre.textContent)
  && /NW-000000/.test(pre.querySelector('.ins').textContent),
  'a multi-line prompt as a block diff with the real text on each side',
  (pre || {}).textContent ? pre.textContent.slice(0, 60) : 'no block');
ok(dl().Measured === 'Helpfulness 0.67 · tokens 3350', 'what it measured, in the metric\'s own units',
  dl().Measured);
/* Trials grouped in the order the run uses the splits. Alphabetically it is "3 test · 8 train ·
   4 validation", which puts the held-out split first and reads as where the run started. */
ok(/^15 trials · 8 train · 4 validation · 3 test/.test(dl().Cases || ''),
  'its trials counted and grouped by split, in run order', dl().Cases);
const trialRows = () => $$('#opt-cands tr.detail table.trials tr')
  .map(r => [...r.children].map(c => c.textContent.replace(/\s+/g, ' ').trim()).join('/'));
ok(trialRows().length === 15, 'a line per trial, every split together', trialRows().length);
ok(trialRows()[0] === 'train_case_01/train/Helpfulness 0.49/pass',
  'naming the case, its split, what was judged and how it went', trialRows()[0]);

/* c0 is the baseline, so there is nothing to diff it against — "identical to c0" would be true
   and useless, and a key list would imply it changed something. */
await openCand('c10');
await openCand('c0');
ok(/this is the configuration the run started from/.test(dl().Configuration || ''),
  'the baseline says it is the baseline rather than being diffed against itself', dl().Configuration);
ok(!('Versus c0' in dl()), 'and carries no diff header');

/* An unscored candidate still has a configuration — it was built, the budget just could not
   afford to run it. Its diff is the evidence for that, and it has no trials to open. */
await openCand('c0');
await openCand('c4');
ok(/budget could not afford it/.test(dl()['Why it was tried'] || ''),
  'an unscored candidate says why it was never run', dl()['Why it was tried']);
ok(/tool_descriptions/.test(dl()['Versus c0'] || ''),
  'and still shows the configuration it would have run', (dl()['Versus c0'] || '').slice(0, 80));
ok(/no trial was recorded/.test(dl().Cases || ''),
  'with nothing to open, said rather than left as an empty table', dl().Cases);
ok(!$('#opt-cands tr.detail [data-open]'), 'and no button offering a report there is nothing behind');

/* The invalid patch never became a candidate, so `propose` never carried its values. Inventing a
   diff from `changed_keys` would be making one up. */
await openCand('c4');
await click($$('#opt-cands tr.case').find(r => /no candidate/.test(r.textContent)));
ok(/not in the registry/.test(dl()['Why it was tried'] || ''),
  'the invalid patch opens onto the reason it was rejected', dl()['Why it was tried']);
ok(/records no values for this candidate/.test(dl()['Versus c0'] || ''),
  'and says its configuration is not in the log rather than showing an empty diff',
  dl()['Versus c0']);
await click($$('#opt-cands tr.case').find(r => /no candidate/.test(r.textContent)));

// ── filters, over the thing they name ──
const seg = async f => { await click(`#opt-seg button[data-f="${f}"]`); };
await seg('feasible');
ok(ids().join(' ').replace(/#\d+|champion|baseline/g, '').replace(/\s+/g, ' ').trim()
  === 'c0 c1 c2 c3 c5 c6 c10',
  'Could ship is the feasible candidates — not c8, which scored best of all and broke a constraint',
  ids().join(' | '));
await seg('infeasible');
ok(ids().length === 1 && /c8/.test(ids()[0]), 'Broke a constraint is c8 alone', ids().join(' | '));
await seg('unscored');
ok(ids().length === 3, 'Never scored collects the unscored, the duplicate and the invalid patch',
  ids().join(' | '));
ok(ids().some(s => /c4/.test(s)) && ids().some(s => /c9/.test(s)) && ids().some(s => /no candidate/.test(s)),
  'all three of them, which differ only in why nothing was measured', ids().join(' | '));
ok(text('#opt-cand-count') === '3 of 11 candidates, 3 never scored',
  'the count follows the filter while the total does not', text('#opt-cand-count'));
await seg('all');
ok(ids().length === 11, 'and All puts them back', ids().length);

/* The search box reaches the configuration values, not just the names. A candidate is found by
   what it actually says — which is the only handle anyone has on a tuned prompt. */
const search = async value => {
  $('#opt-find').value = value;
  $('#opt-find').dispatchEvent(new p.window.Event('input', { bubbles: true }));
  await new Promise(r => setTimeout(r, 20));
};
await search('NW-000000');
ok(ids().length === 3 && ['c5', 'c8', 'c10'].every((c, i) => ids()[i].startsWith(c)),
  'searching the order-id format finds the three candidates whose values contain it',
  ids().join(' | '));
await search('clarifying');
ok(ids().length === 1 && ids()[0].startsWith('c2'),
  'and a phrase from one prompt finds only that candidate', ids().join(' | '));
await search('zzz');
ok(/No candidate matches/.test(text('#opt-cands')), 'a search with no hit says so', text('#opt-cands'));
await search('');

/* ── opening a candidate as a report ──
   The claim the whole table rests on: a candidate's trials are an evaluation run of it, so they
   can be read in the Report tab exactly as any run is. */
await openCand('c10');
await click('#opt-cands [data-open]');
ok(text('.tab.on') === 'Report', 'the jump lands on the Report tab', text('.tab.on'));
ok(text('#path') === 'support-tuning-20261110-091500-7af3e1 · c10',
  'and the path bar names the run and the candidate, not a file that does not exist', text('#path'));
ok(text('#rep-count') === '15 of 15 rows', 'every trial of that candidate is a row', text('#rep-count'));
ok(text('#r-score') === '0.67', 'the hero score is the mean over its trials', text('#r-score'));
const repRow = $$('#rep-rows tr.case').map(r => [...r.children].map(c => c.textContent.trim()));
ok(repRow[0].join(' | ') === 'train_case_01 | Helpfulness | 0.49 | pass',
  'the case, the evaluator that judged it, the score and the verdict', repRow[0].join(' | '));
await click($$('#rep-rows tr.case')[0]);
const repDetail = () => $('#rep-rows tr.detail').textContent.replace(/\s+/g, ' ');
ok(/<answer for train_case_01 under c10>/.test(repDetail()),
  'opening a case shows what that candidate actually answered', repDetail().slice(0, 160));
/* A trial record carries no case input. Printed through the report renderer unguarded that was
   the literal word "undefined" standing where the question should be. */
ok(/Inputnot recorded in this report/.test(repDetail()) && !/undefined/.test(repDetail()),
  'and an input the trial never recorded is named as absent, not printed as undefined',
  repDetail().slice(0, 120));
/* `transcript` is a `{hash, format}` pointer into `transcripts/`, not a step list. Handed to the
   trajectory renderer it came out labelled "recorded session", which is a filename called a
   conversation. */
ok(!/Actual trajectory/.test(repDetail()),
  'and the transcript reference is left out rather than dressed up as a trajectory',
  repDetail().slice(0, 300));

/* A candidate with a failed trial carries that failure into its report rather than dropping the
   row — a run that lost a case to a tool timeout scored lower for it, and the report must say so. */
await click('#tab-opt');
await openCand('c2');
await click('#opt-cands [data-open]');
ok(text('#rep-count') === '8 of 8 rows', 'its eight training trials', text('#rep-count'));
ok(/7 \/ 8/.test(text('#r-stats')), 'with the agent failure counted as a failing row', text('#r-stats'));
const failRow = $$('#rep-rows tr.case').find(r => /train_case_05/.test(r.textContent));
await click(failRow);
ok(/agent_failure: the refund tool timed out/.test(repDetail()),
  'and the error named, not just the status', repDetail().slice(0, 240));
await click('#tab-opt');

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
ok(/No candidate yet/.test(pEmpty.text('#opt-cands')), 'and the candidate table', pEmpty.text('#opt-cands'));
/* With nothing built, the sentence above would read "built 0 versions of your agent, varying
   your agent" — the method stated as a tally of nothing. */
ok(/No iteration has finished yet/.test(pEmpty.text('#opt-explain'))
  && !/built 0 versions/.test(pEmpty.text('#opt-explain')),
  'and the explainer describes the method without counting what it has not built',
  pEmpty.text('#opt-explain'));
ok(/searching for a configuration that scores better/.test(pEmpty.text('#opt-explain')),
  'still saying what the run is for', pEmpty.text('#opt-explain'));
ok(/Pareto/.test(pEmpty.text('#opt-note')), 'a Pareto run is flagged, because one best understates it',
  pEmpty.text('#opt-note'));
ok(pEmpty.$('#opt-metric-wrap').hidden === true,
  'a run that measured no metric hides the metric chart rather than framing an empty one');
ok(pEmpty.$('#opt-cost').hidden === true, 'and the cost drawer too, with nothing to put in it');

/* ── a candidate the optimizer never labelled ──
   Then the only measurement available is the mean over its own trials. The fixture labels every
   scored candidate, so it cannot tell that path from the one that reads `label.measured`. */
const trial = (candidate, score, usage) => ({
  event: 'trial', iteration: 0,
  record: {
    candidate_id: candidate, split: 'train', status: 'completed',
    evaluations: { Helpfulness: { score, reason: 'judged' } }, usage,
  },
});

const pFall = await page();
await pFall.openFolder({
  'fall/run.json': { run_id: 'no-labels', objective: 'Maximize(Helpfulness)' },
  'fall/events.jsonl': jsonl([
    { event: 'config', iteration: 0 },
    trial('c0', 0.2, { tokens: 100, model_calls: 1, seconds: 1.5 }),
    trial('c0', 0.4, { tokens: 300, model_calls: 3, seconds: 2.5 }),
    iterationEnd(0, [{ candidate_id: 'c0', number: 0, outcome: 'feasible', objective: 0.3 }], 2),
  ]),
});
await pFall.click('#tab-opt');
const fallMarks = pFall.$$('#opt-metric-chart circle').map(c => c.querySelector('title').textContent);
ok(fallMarks.length === 1, 'the unlabelled candidate is still measured', fallMarks.join(' | '));
ok(fallMarks[0] === 'c0 · Helpfulness 0.30', 'from the mean of its trials, not their sum or the last of them',
  fallMarks[0]);
ok(pFall.$$('#opt-metric-chart path.metric-line').length === 0,
  'a single point draws no line between itself and nothing');
ok(pFall.$('#opt-metric-pick').hidden === true, 'with one metric there is nothing to pick between');
const fallCols = pFall.$$('#opt-usage-chart rect.col').map(r => r.querySelector('title').textContent);
ok(fallCols[0] === 'iteration 0: 400 over 2 trials', 'usage is summed over the iteration, not averaged',
  fallCols[0]);
ok(!/excluding/.test(pFall.text('#opt-usage-cap')),
  'and with nothing recorded after the last iteration there is no caveat to make',
  pFall.text('#opt-usage-cap'));

/* ── the verdict is the interval, not the gap ──
   A card that only read `mean_diff` would print the same confident headline for a run that
   proved nothing and for one that went backwards. The three logs below differ ONLY in the
   interval on the objective, so nothing but reading it can tell them apart. */
const resultLog = (report, events = []) => jsonl([
  { event: 'config', iteration: 0 },
  { event: 'propose', iteration: 0, candidate_id: 'c0', number: 0, parent_id: 'c0',
    values: { prompt: 'be helpful', temperature: 0.7 } },
  { event: 'propose', iteration: 1, candidate_id: 'c1', number: 1, parent_id: 'c0',
    values: { prompt: 'be helpful and brief', temperature: 0.7 } },
  iterationEnd(0, [{ candidate_id: 'c0', number: 0, outcome: 'feasible', objective: 0.5 }], 2),
  iterationEnd(1, [{ candidate_id: 'c1', number: 1, outcome: 'feasible', objective: 0.6 }], 2),
  ...events,
  ...(report ? [{ event: 'test', iteration: 1, report }] : []),
]);
const testReport = (objective, means) => ({
  champion_id: 'c1', baseline_id: 'c0', objective,
  difference: { H: objective },
  means: means || { c1: { H: { mean: 0.6 } }, c0: { H: { mean: 0.5 } } },
});
const resultPage = async log => {
  const q = await page();
  await q.openFolder({ 'r/run.json': { run_id: 'verdicts', objective: 'Maximize(H)' }, 'r/events.jsonl': log });
  await q.click('#tab-opt');
  return q;
};
const said = q => q.text('#opt-result .verdict');

const pFlat = await resultPage(resultLog(testReport(
  { mean_diff: 0.1, ci_low: -0.04, ci_high: 0.24, n_cases: 6 })));
ok(/has not shown an improvement/.test(said(pFlat)),
  'an interval that includes zero is reported as no improvement, however large the gap', said(pFlat));
ok(/\[-0\.04, 0\.24\] includes zero/.test(said(pFlat)), 'and the interval is quoted', said(pFlat));
ok(!/clears zero/.test(said(pFlat)), 'and never called a clear win', said(pFlat));

const pWorse = await resultPage(resultLog(testReport(
  { mean_diff: -0.08, ci_low: -0.19, ci_high: -0.02, n_cases: 6 })));
ok(/0\.08 worse than the baseline/.test(said(pWorse)),
  'a negative interval is a regression, stated as a positive amount worse', said(pWorse));
ok(/would be a regression/.test(said(pWorse)), 'and named as one', said(pWorse));

/* An interval tight around zero must not be rounded into one that looks like it touches it.
   Two decimals would print [0.004, 0.21], which reads as straddling zero — the opposite verdict. */
const pFine = await resultPage(resultLog(testReport(
  { mean_diff: 0.1071, ci_low: 0.004, ci_high: 0.2103, n_cases: 6 })));
ok(/\[0\.004, 0\.2103\] clears zero/.test(said(pFine)),
  'a narrow interval keeps the digits the verdict turns on', said(pFine));

const pNoCI = await resultPage(resultLog(testReport({ mean_diff: 0.1, n_cases: 6 })));
ok(/moved by 0\.1 against the baseline/.test(said(pNoCI)) && /no interval/.test(said(pNoCI)),
  'a report with no interval says how certain cannot be read off it', said(pNoCI));

/* The validation split is what the search chose on, so it is the weaker evidence and is labelled
   as such rather than dressed up as a held-out result. */
const pVal = await resultPage(resultLog(null, [{
  event: 'picked', iteration: 1,
  report: { champion_id: 'c1', measured: { c1: { H: { mean: 0.6 } }, c0: { H: { mean: 0.5 } } },
    reason: 'c1 ranked first.' },
}]));
ok(/^Validation split/.test(pVal.text('#opt-result .rlabel')),
  'a run with no test split is not called held out', pVal.text('#opt-result .rlabel'));
ok(/flattering/.test(said(pVal)), 'and the gap is hedged as the selected-on gap it is', said(pVal));
ok(/no overall comparison/.test(said(pVal)),
  'a validation report carries no objective difference, and the card says so rather than '
  + 'leaving a half-finished sentence', said(pVal));
ok(pVal.$$('#opt-result .rtab tr').length === 1
  && [...pVal.$('#opt-result .rtab tr').children].map(c => c.textContent).join('|') === 'H|0.5|→|0.6|+0.1||',
  'it still shows the means it does have — `measured`, which is what a validation report calls them',
  pVal.text('#opt-result .rtab'));

const pNoBase = await resultPage(resultLog(null, [{
  event: 'picked', iteration: 1,
  report: { champion_id: 'c1', measured: { c1: { H: { mean: 0.6 } } } },
}]));
ok(/nothing here to compare them against/.test(said(pNoBase)),
  'a report with no baseline figures says so instead of leaving a column of dashes unexplained',
  said(pNoBase));
ok(!/flattering/.test(said(pNoBase)), 'and does not hedge a gap it never computed', said(pNoBase));

/* `propose.values` is the only copy of a candidate's configuration. A log without it cannot show
   one, and inventing a diff out of `changed_keys` would be making it up. */
const pBare = await resultPage(jsonl([
  { event: 'config', iteration: 0 },
  iterationEnd(0, [{ candidate_id: 'c0', number: 0, outcome: 'feasible', objective: 0.5 }], 2),
  { event: 'test', iteration: 0, report: testReport({ mean_diff: 0.1, ci_low: 0.02, ci_high: 0.2, n_cases: 4 }) },
]));
ok(pBare.text('#opt-winner .wid') === 'c1', 'the champion is still named', pBare.text('#opt-winner .wid'));
ok(/records no values/.test(pBare.text('#opt-winner')),
  'and the page says its configuration is not in the log rather than showing an empty diff',
  pBare.text('#opt-winner'));
ok(!pBare.$('#opt-winner .lineage'), 'with no propose event there is no lineage to draw');
ok(!pBare.$('#opt-winner [data-act="copy-winner"]'),
  'and no copy button, which would hand over nothing');

/* Mid-run, before any end phase: there is no result to show, and the training best is not one. */
ok(/No result yet/.test(p2.text('#opt-result')),
  'a run still searching says it has no result yet', p2.text('#opt-result'));
ok(/Best feasible objective so far/.test(p2.text('#opt-result'))
  && /on the training split/.test(p2.text('#opt-result')),
  'offering the training best, named as the training best', p2.text('#opt-result'));
ok(/No champion has been picked yet/.test(p2.text('#opt-winner')),
  'and no winner', p2.text('#opt-winner'));

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

done([p, p2, p3, p4, p5, p6, pLate, pEmpty, pFall, pFlat, pWorse, pFine, pNoCI, pVal, pNoBase, pBare]
  .flatMap(each => each.errs));
