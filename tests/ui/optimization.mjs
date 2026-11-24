/* The Optimization section: Runs, Setup, Overview, Progress, Candidates, Tunables, Trade-off.
   Driven from the generated fixture run, variants of it, and small logs written inline. */
import fs from 'node:fs';
import { SAMPLE, page, runner, served, tick } from './harness.mjs';

const { ok, done } = runner('optimization');

const DIR = new URL('./fixtures/optimizer-run/', import.meta.url);
if (!fs.existsSync(new URL('events.jsonl', DIR))) {
  console.error('fixtures/optimizer-run is missing: run `npm run fixture` (or `npm test`, which writes it first).');
  process.exit(1);
}
const read = name => fs.readFileSync(new URL(name, DIR), 'utf8');
const CONFIG = JSON.parse(read('run.json'));
const EVENTS = read('events.jsonl').split('\n').filter(Boolean).map(line => JSON.parse(line));
const EFFECTS = read('effects.json');
const TRANSCRIPTS = Object.fromEntries(fs.readdirSync(new URL('transcripts/', DIR))
  .filter(f => f.endsWith('.json')).map(f => [`demo/transcripts/${f}`, read(`transcripts/${f}`)]));
const RUN_ID = CONFIG.run_id;

/* ── assertions ── */
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const eq = (actual, expected, msg) => ok(same(actual, expected), msg, JSON.stringify(actual));
const has = (actual, pattern, msg) => ok(pattern.test(actual || ''), msg, actual);

/* ── building pages ── */
const jsonl = events => events.map(e => JSON.stringify(e)).join('\n') + '\n';
/* The fixture log with every event passed through `change`; returning nothing drops the event. */
const eventsWith = change => jsonl(EVENTS.map(e => change(structuredClone(e))).filter(Boolean));
const configWith = change => { const c = structuredClone(CONFIG); change(c); return c; };
const go = (q, tab) => q.click(`.tab[data-view="opt-${tab}"]`);

async function open(files, tab) {
  const q = await served(files);
  if (tab) await go(q, tab);
  return q;
}
/* The fixture run under `demo/`, with its config or log replaced and other files beside it. */
const openFixture = ({ config = CONFIG, events = jsonl(EVENTS), files = {}, tab } = {}) =>
  open({ 'demo/run.json': config, 'demo/events.jsonl': events, ...files }, tab);
/* A run written inline, under `r/`. */
const openLog = (config, events, files = {}) =>
  open({ 'r/run.json': { run_id: 'inline', ...config }, 'r/events.jsonl': jsonl(events), ...files }, 'overview');

const prop = (id, number, objective, outcome = objective == null ? 'unscored' : 'feasible') =>
  ({ candidate_id: id, number, outcome, objective });
const iterationEnd = (iteration, proposals, rollouts = 2) =>
  ({ event: 'iteration_end', iteration, record: { iteration, proposals, rollouts_spent: rollouts } });
const START = { event: 'config', iteration: 0 };

/* ── reading pages ── */
/* Text nodes joined by spaces: the page sets adjacent spans and <br>s with no whitespace between. */
const flat = el => {
  if (!el) return '';
  const walker = el.ownerDocument.createTreeWalker(el, 4 /* NodeFilter.SHOW_TEXT */);
  const out = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) out.push(n.textContent);
  return out.join(' ').replace(/\s+/g, ' ').trim();
};
const num = (el, attr) => Number(el.getAttribute(attr));
const rows = (q, sel) => q.$$(sel).map(tr => [...tr.children].map(flat));
const lines = (q, sel) => rows(q, sel).map(r => r.join(' | '));
const kv = (q, sel) => Object.fromEntries(q.$$(`${sel} > dt`).map(dt => [flat(dt), flat(dt.nextElementSibling)]));
const pairs = (q, sel) => Object.entries(kv(q, sel)).map(([k, v]) => `${k}: ${v}`);
const texts = (q, sel) => q.$$(sel).map(el => el.textContent);
const titles = (q, sel) => texts(q, `${sel} title`);
const pick = async (q, sel, value) => {
  q.$(sel).value = value;
  q.$(sel).dispatchEvent(new q.window.Event('change', { bubbles: true }));
  await tick();
};
const marks = (q, chart) => q.$$(`${chart} circle.pt`).map(c => {
  const title = c.querySelector('title').textContent;
  return {
    id: title.split(' · ')[0], title, cand: c.dataset.cand, cls: c.getAttribute('class').replace('pt ', ''),
    cx: num(c, 'cx'), cy: num(c, 'cy'),
  };
});
const byId = list => Object.fromEntries(list.map(m => [m.id, m]));
const pathPoints = el => el.getAttribute('d').split(/[ML]/).filter(Boolean)
  .map(seg => seg.trim().split(',').map(Number));
const skips = q => q.$$('#skips li').map(flat);
const stats = q => q.$$('#opt-stats .stat')
  .map(s => ['k', 'v', 'n'].map(c => (s.querySelector(`.${c}`) || {}).textContent).join(' | '));
const said = q => q.text('#opt-result .verdict');
/* Each bar of the result picture as `label / tone / side of the no-change line`. */
const strip = q => q.$$('#opt-result .rstrip .rs-row').map(g => {
  const zero = num(q.$('#opt-result .rs-zero'), 'x1'), bar = g.querySelector('line');
  const side = num(bar, 'x1') > zero ? 'right of' : num(bar, 'x2') < zero ? 'left of' : 'across';
  return `${flat(g.querySelector('.rs-lab'))} / ${g.getAttribute('class').replace('rs-row ', '')} / ${side} zero`;
});
const stripNote = q => texts(q, '#opt-result .rs-note').join(' | ');
/* The result cell of the first row in the run list, with the tone of its figure. */
const runResult = q => {
  const td = q.$('#opt-runs-rows tr[data-opt-run]').children[1];
  const tone = (td.querySelector('.delta') || { className: 'no figure' }).className.replace('delta ', '');
  return `${flat(td)} (${tone})`;
};
/* The trade-off marks that share a centre with a front ring. */
const ringed = q => {
  const rings = q.$$('#opt-to-chart circle.front-ring').map(c => `${num(c, 'cx')},${num(c, 'cy')}`);
  return marks(q, '#opt-to-chart').filter(m => rings.includes(`${m.cx},${m.cy}`)).map(m => m.id).sort().join();
};
const shown = q => q.$$('#opt-cands tr.case').map(tr => tr.dataset.k).join();
const candRow = (q, id) => q.$(`#opt-cands tr.case[data-k="${id}"]`);
/* Open one candidate and close every other, so `detail` and `panel` read the only open one. */
const openCand = async (q, id) => {
  const others = q.$$('#opt-cands tr.case[aria-expanded="true"]').map(tr => tr.dataset.k).filter(k => k !== id);
  for (const k of others) await q.click(candRow(q, k));
  if (candRow(q, id).getAttribute('aria-expanded') !== 'true') await q.click(candRow(q, id));
};
const openTrial = async (q, id, name) => {
  await openCand(q, id);
  await q.click(q.$$('#opt-cands tr.trial').find(tr => tr.children[0].textContent.includes(name)));
};
const detail = q => kv(q, '#opt-cands tr.detail > td > dl');
const panel = q => kv(q, '#opt-cands .trial-panel > dl');
const speakers = q => q.$$('#opt-cands .trial-panel .msg').map(m => flat(m.querySelector('.who'))).join(' | ');
const blockDiff = root => {
  const pre = root.querySelector('.setting pre');
  const lines = kind => (pre ? [...pre.querySelectorAll(`.${kind}`)] : []).map(el => el.textContent).join('\n');
  /* Line by line: what the baseline already said is kept, and only the new lines are marked added. */
  return /^ {2}You are the support assistant/.test(lines('same')) && /^\+ .*NW-000000/m.test(lines('ins'))
    && !/NW-000000/.test(lines('same') + lines('del'));
};

/* ══════════ the fixture run ══════════ */

// ── section visibility ──
{
  const none = await served(SAMPLE);
  ok(none.$('#sgroup-opt').hidden && none.$$('.tab[data-sec="optimization"]').every(t => t.hidden),
    'with no optimizer run there is no Optimization section, no Runs view and no tabs of a run');
  ok(none.$('#runs-sub').hidden && !none.$('#runs-sub .runrow'),
    'and nothing is listed under the Runs row it does not have');
}

// ── the runs listed under the Runs row ──
{
  const run = id => ({ run_id: id, objective: 'Quality' });
  const log = id => jsonl([START, { event: 'propose', iteration: 1, candidate_id: `${id}-c1`,
    parent_id: null, values: { temperature: 0.2 }, changed_keys: ['temperature'] }]);
  const q = await served({
    'a.report.json': { overall_score: 0.5, scores: [0.5], test_passes: [true],
      cases: [{ name: 'a', evaluator: 'E', input: 'q' }] },
    'evals/runs/nightly/run.json': run('tune-1'), 'evals/runs/nightly/events.jsonl': log('a'),
    'later/nightly/run.json': run('tune-2'), 'later/nightly/events.jsonl': log('b'),
    'alone/run.json': run('tune-3'), 'alone/events.jsonl': log('c'),
  });
  const sidebar = () => q.$$('#runs-sub .runrow').map(el => el.textContent);
  const marked = () => q.$$('#runs-sub .runrow.on').map(el => el.textContent);
  const row = name => q.$$('#runs-sub .runrow').find(el => el.textContent === name);

  /* The list starts folded away: the Runs row unfolds it, and the caret beside the row folds it back. */
  const caret = q.$('.fold[data-fold="runs-sub"]');
  ok(!caret.hidden && q.$('#runs-sub').classList.contains('folded') && caret.getAttribute('aria-expanded') === 'false',
    'the runs start folded under their row, with the caret that offers to unfold them');
  await q.click('.tab[data-view="opt-runs"]');
  ok(!q.$('#runs-sub').classList.contains('folded') && caret.getAttribute('aria-expanded') === 'true',
    'opening the Runs view unfolds them, saying so');
  await q.click('.tab[data-view="opt-runs"]');
  ok(q.$('#runs-sub').classList.contains('folded') && q.view() === 'v-opt-runs',
    'clicking the Runs row again, already on its view, folds them back and stays there');
  await q.click('.tab[data-view="opt-runs"]');
  ok(!q.$('#runs-sub').classList.contains('folded'), 'and a third click unfolds them again');

  eq(sidebar(), q.$$('#opt-runs-rows tr td:first-child .path').map(el => el.textContent),
    'every run is listed under the Runs row, in the order the Runs list has them');
  ok(sidebar().includes('evals/runs/nightly') && sidebar().includes('later/nightly'),
    'two runs whose directories share a name are named by the whole path, not the name twice',
    sidebar().join());
  eq(row('alone').getAttribute('title'), null,
    'and no row carries a tooltip, which would cover the rows beneath it');
  eq(marked(), [],
    'the Runs list selects a run before anyone asks for one, so the list marks none as open');

  /* The point of listing them: opening a run should not take a trip through the list first. */
  await q.click('.tab[data-view="reports"]');
  await q.click(row('later/nightly'));
  eq([q.view(), marked(), q.text('#opt-runs-rows tr.on td:first-child .path')],
    ['v-opt-overview', ['later/nightly'], 'later/nightly'],
    'clicking one opens that run on its overview from anywhere, and the list follows to it');
  await q.click('.tab[data-view="opt-runs"]');
  eq([q.view(), marked()], ['v-opt-runs', []],
    'the Runs row itself still opens the list of every run, which has no one run open');
  ok(!q.$('#sections .tcount'), 'no count on the row: they are all listed right under it');
  await q.click(caret);
  eq([q.$('#runs-sub').classList.contains('folded'), caret.getAttribute('aria-expanded'), q.view()],
    [true, 'false', 'v-opt-runs'],
    'the same caret folds the runs away again, and the open view stays where it was');

  /* However a run is opened, the list unfolds: the mark is never hidden in a folded one. */
  await q.click('#opt-runs-rows tr[data-opt-run]');
  ok(q.view() === 'v-opt-overview' && !q.$('#runs-sub').classList.contains('folded') && marked().length === 1,
    'opening a run unfolds the folded list, so the marked run is in sight', `${q.view()} ${marked()}`);
}
const p = await served({
  'demo/run.json': CONFIG, 'demo/events.jsonl': jsonl(EVENTS), 'demo/effects.json': EFFECTS, ...TRANSCRIPTS,
});
const { $, $$, text, click } = p;
eq($$('#sections .tab:not([hidden])').map(t => t.dataset.view).join(), 'experiments,reports,opt-runs',
  'a run directory gains the sidebar a Runs view, after the views of the folder itself');
eq([p.text('#sgroup-opt .sec'), $('#sgroup-opt .tab').textContent],
  ['Optimization', 'Runs'], 'under an Optimization heading, which names the section and opens nothing');
eq(skips(p), [], 'the transcripts and effects.json of a run are not reported as unusable files');

// ── Runs ──
await click('.tab[data-view="opt-runs"]');
eq($$('.tab.on').map(t => t.dataset.view).join(), 'opt-runs', 'which opens the list of runs');
ok($('#pagehead').hidden, 'which is a list, so no page header names anything');
eq(rows(p, '#opt-runs-rows tr[data-opt-run]').map(r => r.slice(0, 5)),
  [[`${RUN_ID} / support-tuning demo`, '+0.143 held out · clear of zero', 'c10', '10 candidates 6 iterations',
    'iterations']],
  'one row: the run, its result, champion, search size and stop reason');
const delta = $('#opt-runs-rows .delta');
ok(delta.className === 'delta up' && /\[0\.061, 0\.225\]/.test(delta.title),
  'the result is coloured by its interval, which is on hover', delta.outerHTML);

await click('#opt-runs-rows tr[data-opt-run]');
ok($('#v-opt-overview').classList.contains('on') && text('#path') === 'demo/events.jsonl',
  'a run row opens Overview on its event log', text('#path'));
eq([text('#crumbs'), text('#facts'), $$('#opt-tabs .tab:not([hidden])').map(t => t.dataset.view).join(), text('#opt-tabs .tab.on')],
  [`Runs/${RUN_ID}`, 'stopped: iterations · champion c10', 'opt-setup,opt-overview,opt-progress,opt-candidates,opt-tunables,opt-trade',
    'Overview'],
  'the page header names the run and holds its six views as a tab strip, the open one marked');
eq([$$('#sections .tab.on').map(t => t.dataset.view).join(), $('.tab[aria-current="page"]').dataset.view], ['opt-runs', 'opt-overview'],
  'while the sidebar stays on Runs, the list the run was opened from');
eq(text('#opt-sub'), 'reached max_iterations=6',
  'the header names the run and its status once, and the subtitle adds only what the stop said for itself');
eq(text('#opt-method-sum'),
  'Tuned system_prompt, temperature, tool_descriptions, max_tokens · 8 train · 4 validation · 3 test',
  'the method line names the keys the proposals changed and the split sizes');

// ── Setup and Tunables ──
await go(p, 'setup');
ok($('#v-opt-setup').classList.contains('on') && text('#path') === 'demo/run.json', 'Setup reads run.json',
  text('#path'));
ok($('#opt-source').hidden, 'no experiment is named when the folder holds none');
has(text('#opt-explain'), /built 10 versions .* searched on train, picked the champion once on validation/,
  'the account counts the candidates built, not the 11 proposals, and says what each split was for');
eq(lines(p, '#opt-space-rows tr'), [
  'system_prompt | text ≤ 2000 chars | 127 characters | 10 of 10 · 7 changed | always',
  'temperature | float 0 – 1 | 0.7 | 10 of 10 · 2 changed | always',
  'max_tokens | int 256 – 2048 step 256 | 1024 | 10 of 10 · 1 changed | always',
  'tool_descriptions | text | 35 characters | 10 of 10 · 2 changed | always',
  'escalation.policy | choice · 2 options | ask | 0 of 10 | always',
  'escalation.auto.threshold | float 0.1 – 0.9 | 0.5 | 0 of 10 | escalation.policy = auto',
], 'the declared space: domain, default, use and condition per key, untouched keys included');
eq(text('#opt-space-note'), '', 'nothing is flagged for a log that stays inside its space');

// ── Overview: the result card ──
eq(flat($('#opt-result .rlabel')), 'Result · held out 3 test cases kept out of the search',
  'the card is headed by the split behind it');
eq(lines(p, '#opt-result .rtab tbody tr'),
  ['Helpfulness | 0.521 | → | 0.664 | +0.143 | [0.061, 0.225] | 1.274×', 'tokens | 3190 | → | 3340 | +150 |  | '],
  'a row per goal metric; one with no interval stops at the change');
ok($$('#opt-result .delta').every(td => td.classList.contains('flat')),
  'no per-metric change is coloured: only the objective states a direction');
eq(said(p), 'Objective +0.143, 95% interval [0.061, 0.225]: clear of zero on cases the search never saw.',
  'the verdict is the reading of the objective interval');
eq(strip(p), ['Helpfulness ↑ / good / right of zero'],
  'one bar: the goal with an interval, the objective not drawn twice');
eq($('#opt-result .rs-row title').textContent,
  'Helpfulness: +0.143 [0.061, 0.225] · clear of no change, the way the goal wants',
  'hovering the bar gives the figures and the reading');
ok(/held-out cases/.test(stripNote(p)) && /nothing to draw for tokens \(no interval reported\)/.test(stripNote(p))
  && !$('#opt-result .rstrip').classList.contains('weak'),
  'the picture is full strength, says held out, and names the goal it could not draw', stripNote(p));

// ── a candidate that cannot ship says why, in its row and above its cases ──
{
  const q = await openFixture({ tab: 'candidates' });
  const why = 'Not met: tokens <= 4000 (measured tokens 4210). A constraint is judged on the candidate as a '
    + 'whole, so every one of its cases can pass and it still cannot ship.';
  await openCand(q, 'c8');
  eq(detail(q)['Why infeasible'], why, 'an infeasible candidate names the constraint it broke and what it measured');
  await q.click('#opt-cands [data-open]');
  ok(q.text('#facts').includes(`infeasible ${why}`) && q.text('#r-stats').startsWith('8 / 8'),
    'and says so again above its cases, every one of which passes', `${q.text('#facts')} | ${q.text('#r-stats')}`);
  await q.click('#crumbs [data-go="opt-candidates"]');
  await openCand(q, 'c10');
  ok(!Object.keys(detail(q)).some(k => k.startsWith('Why c') || k === 'Why feasible'), 'a feasible one has nothing to explain',
    Object.keys(detail(q)).join());

  const unranked = await openFixture({ tab: 'candidates', events: eventsWith(e => {
    if (e.event === 'label' && e.candidate_id === 'c5') Object.assign(e.label, { outcome: 'ineligible', reason: 'train_case_03 has no score.' });
    for (const prop of (e.record && e.record.proposals) || []) if (prop.candidate_id === 'c5') prop.outcome = 'ineligible';
    return e;
  }) });
  await openCand(unranked, 'c5');
  eq(detail(unranked)['Why ineligible'], 'train_case_03 has no score. Some training case was not measured, so it was left out of the ranking.',
    'an ineligible one gives the reason the log recorded');
}

// ── Overview: case by case, the champion against the baseline ──
{
  const q = await openFixture({ tab: 'overview' });
  const marks = () => q.$$('#opt-cases-chart g.bell');
  const tip = name => marks().map(g => g.querySelector('title').textContent).find(t => t.startsWith(name + ' '));
  const validation = EVENTS.find(e => e.event === 'trial' && e.record.candidate_id === 'c10'
    && e.record.case_name === 'val_case_01').record.evaluations.Helpfulness.score;
  eq(q.$$('#opt-cases-pick button').map(o => o.textContent), ['Helpfulness', 'Tokens', 'Seconds'],
    'cases are compared on each evaluator score and on what a trial cost');
  eq([marks().length, tip('test_case_01'), tip('train_case_01'), tip('val_case_01')], [
    15,
    'test_case_01 · test · baseline 0.521 · champion 0.664',
    'train_case_01 · train · baseline 0.332 · champion 0.491',
    `val_case_01 · validation · baseline not run · champion ${validation}`,
  ], 'one row per case either ran, held-out cases first, each with what both measured');
  has(q.text('#opt-cases-cap'), /Of 11 cases both ran, 11 improved and 0 got worse\. A case only one of them ran has a single mark\./,
    'the caption counts which way the cases went');
  await q.click('#opt-cases-pick button[data-value="u:tokens"]');
  eq(texts(q, '#opt-cases-pick button.on'), ['Tokens'], 'a few choices sit side by side, the chosen one marked');
  ok(marks()[0].classList.contains('down') && /lower is better/.test(q.text('#opt-cases-cap'))
    && tip('test_case_01') === 'test_case_01 · test · baseline 3190 · champion 3340',
  'on tokens a rise is drawn as worse', `${marks()[0].getAttribute('class')} | ${tip('test_case_01')}`);

  const kept = await openFixture({ tab: 'overview', events: eventsWith(e => {
    if (e.report && e.report.champion_id) e.report.champion_id = 'c0';
    return e;
  }) });
  ok(kept.$('#opt-cases').hidden, 'a run that kept its baseline has nothing to compare case by case');

  // A candidate's cases, opened as a report, are filled in from the experiment the run was tuned against.
  const asked = Object.fromEntries(CONFIG.split.train.map(name => [name, `What about ${name}?`]));
  const given = { cases: [...CONFIG.split.train, ...CONFIG.split.validation, ...CONFIG.split.test]
    .map(name => ({ name, input: asked[name] || 'held out', expected_output: 'the right answer' })), evaluators: [] };
  const sourced = await openFixture({ tab: 'candidates', files: { 'cases/experiment.json': given } });
  await openCand(sourced, 'c10');
  await sourced.click('#opt-cands [data-open]');
  await tick(4);
  await sourced.click('#rep-rows tr.case');
  has(flat(sourced.$('#rep-rows tr.detail')), /Input What about train_case_01\? Expected output the right answer Actual output · diff/,
    'a candidate\'s report shows what each case asked and expected, read off the experiment the run was tuned against');

  // The champion, at the top of the view.
  const hero = (page, part) => texts(page, `#opt-hero ${part}`).join(' | ');
  eq([q.$('#opt-hero .champ').className, hero(q, '.champ-name'), hero(q, '.champ-from'), hero(q, '.champ-delta'), hero(q, '.champ-read')],
    ['champ won', 'c10', 'against c0, the baseline, on 3 held-out cases', '+0.143', 'clear of zero'],
    'the Overview opens on the champion: who won, against what, by how much, and how that reads');
  await pick(q, '#opt-cases-pick', 'm:Helpfulness');
  eq([q.$$('#opt-hero .win').map(w => w.classList[1]).join(), hero(q, '.win-nums'), hero(q, '.win-delta'), hero(q, '.win-cases')],
    ['up,down', '0.521 → 0.664 | 3190 → 3340', '▲ +0.143 | ▼ +150', 'better on 11 of 11 cases | better on 0 of 11 cases'],
    'with each goal beside the baseline: a gain the goal wanted is a win, more tokens is not');
  eq([kept.$('#opt-hero .champ').className, hero(kept, '.champ-name'), kept.$('#opt-hero .champ-gain')],
    ['champ held', 'c0', null], 'a run that kept its baseline says the baseline held, and claims no gain');
  const unshown = await openFixture({ tab: 'overview', events: eventsWith(e => {
    if (e.event === 'test') e.report.objective.ci_low = -0.02;
    return e;
  }) });
  eq([unshown.$('#opt-hero .champ-gain').className, hero(unshown, '.champ-read')], ['champ-gain flat', 'not shown'],
    'an interval that includes zero is not dressed as a win');
  const running = await openFixture({ tab: 'overview', events: eventsWith(e => (['picked', 'test', 'finalists'].includes(e.event) ? null : e)) });
  eq([running.$('#opt-hero .champ').className, hero(running, '.champ-name'), hero(running, '.champ-from'), stats(running)[1]],
    ['champ none', 'not picked yet', 'Leading on training: c10 at 0.67.', 'Best on training | 0.67 | c10 · highest here'],
    'and a run with no champion names who is leading, without calling it "not the pick"');

  const failed = await openFixture({ tab: 'overview', events: eventsWith(e => {
    if (e.event === 'trial' && e.record.candidate_id === 'c10' && e.record.case_name === 'train_case_01') e.record.status = 'agent_failure';
    return e;
  }) });
  eq(failed.$$('#opt-cases-chart g.bell title').map(el => el.textContent).find(tip => tip.startsWith('train_case_01 ')),
    'train_case_01 · train · baseline 0.332 · champion not run', 'a trial that failed is not charted as a score');
}

// ── Overview: the champion configuration ──
eq([text('#opt-winner .wid'), text('#opt-winner .lineage')], ['c10', 'c10 ← c6 ← c3 ← c1 ← c0'],
  'the champion with its lineage back to the baseline');
eq(texts(p, '#opt-winner .setting .skey'), ['system_prompt', 'temperature'],
  'only the settings that differ from the baseline');
has(text('#opt-winner .unchanged'), /^2 other settings unchanged from c0/, 'the rest are counted');
ok(blockDiff($('#opt-winner')) && text('#opt-winner .setting .inline') === '0.7 → 0.2',
  'a multi-line value is a block diff, a short one stays inline');

const champion = EVENTS.find(e => e.event === 'propose' && e.candidate_id === 'c10').values;
let copied = null;
Object.defineProperty(p.window.navigator, 'clipboard', {
  configurable: true, value: { writeText: async t => { copied = t; } },
});
await click('#opt-winner [data-act="copy-winner"]');
ok(text('#opt-copied') === 'copied 4 settings' && same(JSON.parse(copied), champion),
  'copy hands over the whole champion configuration, inherited settings included', copied);
await click('#opt-winner [data-act="save-winner"]');
ok(p.window.__downloads.at(-1) === 'champion.json' && same(await p.downloaded(), champion),
  'download offers the same document as champion.json', p.window.__downloads.join());
delete p.window.navigator.clipboard;
await click('#opt-winner [data-act="copy-winner"]');
has(text('#opt-copied'), /could not reach the clipboard/, 'a copy that could not happen says so');

// ── Progress: the figures and the objective chart ──
eq(stats(p), ['Rollouts | 64 / 240 | spent of the budget',
  'Best on training | 0.67 | the champion, on the cases it searched over'],
  'two figures under the champion; the training best is feasible c10, not infeasible c8');
eq([text('#opt-search-sum'), text('#opt-cost-sum')], ['11 proposals over 6 iterations', '277k tokens · 82 trials'],
  'the search is counted in proposals, the cost over every trial');

const objective = marks(p, '#opt-series-chart');
eq(objective.map(m => `${m.id} ${m.cls}`), ['c0 feasible', 'c1 feasible', 'c2 feasible', 'c3 feasible',
  'c5 feasible', 'c6 feasible', 'c8 infeasible', 'c10 champion'],
  'one mark per scored proposal: unscored c4, the invalid patch and duplicate c9 are not plotted');
eq(objective[6].title, 'c8 · infeasible · 0.69', 'a mark is titled with candidate, outcome and objective');
const best = pathPoints($('#opt-series-chart path.best'));
const top = Math.min(...best.map(([, y]) => y));
const cyOf = id => byId(objective)[id].cy;
ok(best.every(([x, y], i) => i === 0 || (y <= best[i - 1][1] && (x === best[i - 1][0] || y === best[i - 1][1]))),
  'best-so-far steps rather than slopes, and never falls', JSON.stringify(best));
ok(best[0][1] === cyOf('c0') && top === cyOf('c10') && top > cyOf('c8'),
  'it starts at c0, tops out at c10 and stays below infeasible c8', `${best[0][1]} ${top} c8 ${cyOf('c8')}`);
eq(texts(p, '#opt-series-chart text.catlab').join(), 'c0,c1,c2,c3,c4,c5,c6,—,c8,c9,c10',
  'every proposal is named under its slot, plotted or not');
eq([texts(p, '#opt-series-chart text.itlab').join(), $$('#opt-series-chart line.itsep').length],
  ['iteration,0,1,2,3,4,5', 5], 'the iterations are a second row, divided where the search turned over');

// ── Progress: budget and trial outcomes ──
eq(titles(p, '#opt-budget-chart circle'), ['iteration 0: +8, 8 total', 'iteration 1: +16, 24 total',
  'iteration 2: +8, 32 total', 'iteration 3: +16, 48 total', 'iteration 4: +8, 56 total', 'iteration 5: +8, 64 total'],
  'the budget curve is cumulative, one mark per finished iteration');
ok($('#opt-budget-chart line.cap-line') && /240-rollout budget/.test(text('#opt-budget-cap')),
  'the budget cap is drawn and named', text('#opt-budget-cap'));
eq($$('#opt-status-chart .legend span').map(flat),
  ['completed 80', 'agent failure 1', 'evaluator invalid 1', 'test 6 · train 64 · validation 12'],
  'trials by how they ended and by split');

// ── Progress: one chart, with a picker over the objective and each metric ──
const series = () => marks(p, '#opt-series-chart');
const axis = () => ({
  label: text('#opt-series-chart .axlab'), ticks: texts(p, '#opt-series-chart text.tick').map(Number),
});
eq([texts(p, '#opt-series-pick button').join(), $('#opt-series-pick').value, text('#opt-series-head')],
  ['Objective,Helpfulness,tokens', 'objective', 'Objective by candidate'],
  'the picker offers the objective, then each metric');
const objectiveLegend = $$('#opt-series-legend span').length;
await pick(p, '#opt-series-pick', 'm:Helpfulness');
ok(objectiveLegend > 0 && $$('#opt-series-legend span').length === 0
  && text('#opt-series-head') === 'Helpfulness by candidate',
  'a metric chart has its own heading and no best-so-far legend', text('#opt-series-head'));
eq(series().map(m => m.title), ['c0 · Helpfulness 0.51', 'c1 · Helpfulness 0.55', 'c2 · Helpfulness 0.50',
  'c3 · Helpfulness 0.61', 'c5 · Helpfulness 0.59', 'c6 · Helpfulness 0.64', 'c8 · Helpfulness 0.69',
  'c10 · Helpfulness 0.67'],
  'the recorded means (c0 0.51, where its trials average 0.52), and no mark for an unmeasured candidate');
ok(axis().label === 'Helpfulness, higher is better' && Math.max(...axis().ticks) < 1,
  'the axis carries the goal direction and is scaled to the metric', JSON.stringify(axis()));
eq($$('#opt-series-chart path.metric-line').map(el => pathPoints(el).length), [4, 2],
  'the joining line breaks at an unmeasured candidate rather than spanning it');
await pick(p, '#opt-series-pick', 'm:tokens');
eq([series()[0].title, series()[6].title, series()[6].cls], ['c0 · tokens 3180', 'c8 · tokens 4210', 'infeasible'],
  'a count keeps its decimals off, and c8 is infeasible here too');
ok(axis().label === 'tokens, lower is better' && Math.min(...axis().ticks) > 1000,
  'the axis follows the picker, direction and units', JSON.stringify(axis()));

// ── Progress: what each iteration cost ──
const cols = () => titles(p, '#opt-usage-chart rect.col');
eq([texts(p, '#opt-usage-pick button').join(), $('#opt-usage-pick').value], ['Tokens,Model calls,Seconds', 'tokens'],
  'every usage measure recorded is on offer, tokens first');
eq(cols(), ['iteration 0: 25440 over 8 trials', 'iteration 1: 50640 over 16 trials',
  'iteration 2: 26480 over 8 trials', 'iteration 3: 54440 over 16 trials', 'iteration 4: 33680 over 8 trials',
  'iteration 5: 26800 over 8 trials'],
  'a column per finished iteration, summed over its own trials and not cumulative');
has(text('#opt-usage-cap'), /excluding the 18 trials recorded after the last finished iteration/,
  'trials after the last iteration are left out of the columns, and the caption says so');
const heights = $$('#opt-usage-chart rect.col').map(r => num(r, 'height'));
ok(Math.abs(heights[1] / heights[3] - 50640 / 54440) < 1e-6, 'column heights are proportional to the totals',
  heights.join(' '));
await pick(p, '#opt-usage-pick', 'model_calls');
const calls = cols()[0];
await pick(p, '#opt-usage-pick', 'seconds');
eq([calls, cols()[0]], ['iteration 0: 24 over 8 trials', 'iteration 0: 33.60 over 8 trials'],
  'the picker switches unit: calls are whole numbers, seconds are not');

// ── Tunables: which setting moved the metric, from effects.json ──
const effects = () => $$('#opt-eff-chart .eff-row').map(g => ({
  row: [g.querySelector('.eff-level').textContent, g.getAttribute('class').replace('eff-row ', ''),
    g.querySelector('text.tick').textContent].join(' / '),
  x: num(g.querySelector('.eff-dot'), 'cx'),
  lo: num(g.querySelector('.eff-ci'), 'x1'),
  hi: num(g.querySelector('.eff-ci'), 'x2'),
}));
const effZero = () => num($('#opt-eff-chart .eff-zero'), 'x1');
ok(!$('#opt-effects').hidden && text('#opt-eff-sum') === '3 estimates',
  'a run with a fitted model has the effects drawer', text('#opt-eff-sum'));
eq(texts(p, '#opt-eff-goal button'), ['Helpfulness · higher is better', 'tokens · lower is better'],
  'each goal is offered with the direction it wants');
has(text('#opt-eff-cap'), /for Helpfulness, fitted to 48 observations/,
  'the caption says how much the estimate rests on');
eq($$('#opt-eff-chart .eff-key').map(flat),
  ['system_prompt vs default', 'temperature vs (0.45, 0.75]', 'max_tokens vs (256, 1024]'],
  'a group per key in declared order, each naming its reference level');
const helpful = effects();
eq(helpful.map(e => e.row), ['rewritten / settled / +0.12 · 0.96', '(0.15, 0.45] / unsettled / +0.03 · 0.73',
  '(1024, 2048] / unsettled / -0.01 · 0.41'],
  'signed change and P(better) per level; only an interval clear of zero is settled');
const grid = $$('#opt-eff-chart line.grid-l').map(l => num(l, 'x1'));
ok(effZero() > Math.min(...grid) && effZero() < Math.max(...grid) && helpful[0].lo > effZero()
  && helpful[0].lo < helpful[0].x && helpful[0].x < helpful[0].hi
  && helpful[2].lo < effZero() && helpful[2].hi > effZero(),
  'zero is in view: the settled bar is wholly to one side of it with its dot inside, the unsettled one spans it',
  JSON.stringify(helpful));
ok(text('#opt-eff-note')
  === 'Not modelled: tool_descriptions (two values, one candidate each), escalation.policy (one value).'
  && !$('#opt-eff-note').classList.contains('warn'), 'the keys the model left out are named with why',
  text('#opt-eff-note'));
await pick(p, '#opt-eff-goal', 'tokens');
ok(effects()[0].row === 'rewritten / settled / +180 · 0.01' && effects()[0].x > effZero(),
  'for a minimised goal P(better) is flipped, while the bar still points the way the number went', effects()[0].row);
ok(/The fit is flagged.*under-states the spread of token counts/.test(text('#opt-eff-note'))
  && $('#opt-eff-note').classList.contains('warn'), 'a fit that failed its own check says so, as a warning',
  text('#opt-eff-note'));

// ── Candidates: the table ──
const ALL = 'c0,c1,c2,c3,c4,c5,c6,invalid-7,c8,c9,c10';
eq(lines(p, '#opt-cands tr.case'), [
  'c0 #0 baseline | 0 | 0.51 | feasible | —',
  'c1 #1 | 1 | 0.55 | feasible | system_prompt',
  'c2 #2 | 1 | 0.50 | feasible | system_prompt',
  'c3 #3 | 2 | 0.61 | feasible | system_prompt, temperature',
  'c4 #4 | 2 | not scored | unscored | tool_descriptions',
  'c5 #5 | 3 | 0.59 | feasible | tool_descriptions',
  'c6 #6 | 3 | 0.64 | feasible | system_prompt',
  'no candidate | 4 | not scored | invalid | —',
  'c8 #7 | 4 | 0.69 | infeasible | system_prompt, max_tokens',
  'c9 #8 | 5 | not scored | duplicate | system_prompt',
  'c10 #9 champion | 5 | 0.67 | feasible | system_prompt, temperature',
], 'a row per proposal, scored or not: iteration, objective, outcome and what it changed about its parent');
ok(!candRow(p, 'c4').querySelector('.bar2') && !!candRow(p, 'c8').querySelector('.bar2'),
  'an unscored candidate draws no score bar');

// ── Candidates: one candidate, and its trials ──
await openCand(p, 'c10');
const c10 = detail(p);
eq(Object.keys(c10), ['Why it was tried', 'Versus c0', 'Cases', 'Per case'], 'a candidate row opens');
has(c10['Why it was tried'], /^Combine c6's up-front id request/, 'with its rationale');
const tiles = $$('#opt-cands tr.detail .cand-sum .stat');
eq(tiles.map(flat), ['0.67 Objective ▲ +0.16 against c0', '0.67 Helpfulness ▲ +0.16 against c0',
  '3350 tokens ▲ +170 against c0', 'c10 ← c6 ← c3 ← c1 ← c0 Derived from'],
'and its figures at a glance: the objective and each metric beside the baseline, and the line it descends from');
eq(tiles.map(el => el.className), ['stat up', 'stat up', 'stat down', 'stat line'],
  'a change the goal wanted is marked good, and more tokens is not');
ok(same(texts(p, '#opt-cands tr.detail .setting .skey'), ['system_prompt', 'temperature'])
  && blockDiff($('#opt-cands tr.detail')) && /0\.7 → 0\.2/.test(c10['Versus c0']),
  'diffed against the baseline, not the parent c6, which has the same temperature', c10['Versus c0'].slice(0, 120));
has(c10.Cases, /^15 trials · 8 train · 4 validation · 3 test/, 'its trials are counted by split, in run order');
const trials = rows(p, '#opt-cands tr.trial');
ok(trials.length === 15 && trials[0].join(' | ') === 'train_case_01 | train | Helpfulness 0.49 | fail',
  'a line per trial: case, split, score and verdict', trials[0].join(' | '));
eq([...new Set(trials.map(t => t.at(-1)))].sort(), ['fail', 'pass'],
  'a completed trial reads pass or fail as its evaluator recorded');

await openTrial(p, 'c10', 'train_case_01');
eq(pairs(p, '#opt-cands .trial-panel > dl').map(line => line.replace(/^Transcript: .*/, 'Transcript')),
  ['Ended as: completed', 'Scored: Helpfulness 0.49 fail The reply leaves the customer to ask again.', 'Cost: 3350 tokens · 3 model calls · 4.20s',
    'Output: Order NW-104233 was delivered on 2 October. It was left at the front desk.', 'Transcript'],
  'a trial opens: how it ended, each score with its verdict and reason, the cost, the output, the transcript');

// ── Candidates: transcripts, in both shapes ──
for (const [cand, name, count, who] of [
  ['c10', 'train_case_01', '3 spans of a recorded session',
    '#0 invoke agent | #1 inference | user | assistant | #2 tool'],
  ['c0', 'train_case_02', '4 messages',
    'user | assistant 2345 in · 1005 out · 1800 cache read tokens | user | assistant'],
]) {
  if (cand !== 'c10') await openTrial(p, cand, name);
  eq([text('#opt-cands .tr-count'), speakers(p)], [count, who],
    `a transcript of ${count}: each part by kind or role`);
  const blocks = $$('#opt-cands .trial-panel .call').map(flat);
  ok(blocks.some(c => /^tool call lookup_order .*NW-104233/.test(c))
    && blocks.includes('tool result delivered 2026-10-02') && !!$('#opt-cands details.raw'),
    'with the tool call named, its result unwrapped from the JSON envelope, and the file one disclosure away',
    blocks.join(' | '));
}
has(text('#opt-cands .say.think'), /No order id in the message/,
  'reasoning is shown, set apart from what the model said');
await openTrial(p, 'c0', 'train_case_02');
ok(!$('#opt-cands tr.tdetail'), 'a second click closes the trial');

// ── Candidates: trials with no transcript, a failure, an evaluator error ──
await openTrial(p, 'c6', 'val_case_02');
has(panel(p).Transcript, /the log points at demo\/transcripts\/4ed70e0cfecb415b\.json ?, which is not in the run directory/,
  'a pointer whose file the run directory does not hold names that file');
await openTrial(p, 'c2', 'train_case_05');
eq(['Ended as', 'Attempts', 'Output', 'Transcript'].map(k => panel(p)[k]),
  ['agent_failure ToolError the refund tool timed out', '2, earlier: agent_failure', 'none recorded',
    'this trial recorded no transcript'],
  'a failed trial: status and error, earlier attempts, no output, and no transcript pointer at all');
await openTrial(p, 'c5', 'train_case_03');
ok(panel(p)['Evaluator errors'] === 'Helpfulness raised the judge returned no score for this answer'
  && !('Scored' in panel(p)), 'an evaluator that raised is named, with no score standing in for it',
  JSON.stringify(panel(p)).slice(0, 200));

// ── Candidates: the baseline, an unscored candidate, the invalid patch ──
await openCand(p, 'c0');
ok(/the configuration the run started from/.test(detail(p).Configuration) && !('Versus c0' in detail(p)),
  'the baseline is not diffed against itself');
await openCand(p, 'c4');
ok(/budget could not afford it/.test(detail(p)['Why it was tried'])
  && /tool_descriptions/.test(detail(p)['Versus c0']),
  'an unscored candidate says why it never ran and still shows its configuration');
ok(/no trial was recorded/.test(detail(p).Cases) && !$('#opt-cands tr.detail [data-open]'),
  'with no trials and no report to open');
await openCand(p, 'invalid-7');
ok(/not in the registry/.test(detail(p)['Why it was tried'])
  && /records no values for this candidate/.test(detail(p)['Versus c0']),
  'the invalid patch shows why it was rejected, and no diff invented from its changed keys');
await openCand(p, 'c10');

// ── Candidates: filters and search ──
for (const [filter, ids] of [
  ['feasible', 'c0,c1,c2,c3,c5,c6,c10'], ['infeasible', 'c8'], ['unscored', 'c4,invalid-7,c9'], ['all', ALL],
]) {
  await click(`#opt-seg button[data-f="${filter}"]`);
  eq([shown(p), text('#opt-cand-count')], [ids, `${ids.split(',').length} of 11 candidates, 3 never scored`],
    `filter "${filter}", with the count following it`);
}
for (const [query, ids] of [['NW-000000', 'c5,c8,c10'], ['clarifying', 'c2'], ['zzz', '']]) {
  await p.type('#opt-find', query);
  eq(shown(p), ids, `search "${query}" reaches the configuration values`);
}
has(text('#opt-cands'), /No candidate matches/, 'a search with no hit says so');
await p.type('#opt-find', '');

// ── Candidates: open as report ──
const repDetail = () => flat($('#rep-rows tr.detail'));
await click('#opt-cands [data-open]');
eq([$('.view.on').id, text('#path'), text('#rep-count'), text('#r-score')],
  ['v-report', `${RUN_ID} · c10`, '15 of 15 rows', '0.67'],
  'a candidate opens in the report page over its trials, named by run and candidate');
eq(lines(p, '#rep-rows tr.case')[0], 'train_case_01 | Helpfulness | 0.49 | fail',
  'a row per trial: case, evaluator, score, verdict');
await click('#rep-rows tr.case');
ok(/Order NW-104233 was delivered on 2 October\. It was left at the front desk\./.test(repDetail()) && /Input not recorded in this report/.test(repDetail())
  && !/undefined|Actual trajectory/.test(repDetail()),
  'a case shows the answer, names the input as not recorded, and does not pass the transcript off as a trajectory',
  repDetail().slice(0, 300));
await go(p, 'candidates');
await openCand(p, 'c2');
await click('#opt-cands [data-open]');
ok(text('#rep-count') === '8 of 8 rows' && /4 \/ 8/.test(text('#r-stats')),
  'failed trials stay in the report as failing rows', text('#r-stats'));
await click($$('#rep-rows tr.case').find(r => /train_case_05/.test(r.textContent)));
has(repDetail(), /agent_failure: the refund tool timed out/, 'with the error named, not just the status');
eq(text('#opt-note'), '', 'a clean log raises no warning');

/* ══════════ variants of the fixture run ══════════ */

// ── which experiment the split's cases came from ──
{
  const all = ['train', 'validation', 'test'].flatMap(part => CONFIG.split[part]);
  const named = names => ({ cases: names.map(n => ({ name: n, input: 'q' })), evaluators: [] });
  const bySession = { cases: all.map(n => ({ session_id: n, name: `case ${n}` })), evaluators: [] };
  const whole = 'Its 15 cases are cases of cases/experiment.json.';
  for (const [what, files, expected] of [
    ['every case, by name', { 'cases/experiment.json': named(all) }, whole],
    ['every case, by session id', { 'cases/experiment.json': bySession }, whole],
    ['part of the split', { 'cases/experiment.json': named(all.slice(0, 9)) },
      '9 of its 15 cases are cases of cases/experiment.json.'],
    ['two experiments, the fuller match wins',
      { 'a-few/experiment.json': named(all.slice(0, 3)), 'most/experiment.json': named(all.slice(0, 12)) },
      '12 of its 15 cases are cases of most/experiment.json.'],
    ['two experiments that fit equally, neither guessed',
      { 'a-copy/experiment.json': named(all), 'b-copy/experiment.json': named(all) },
      'Not matched to one experiment: 2 experiments hold 15 of its 15 cases and are equally near: '
        + 'a-copy/experiment.json, b-copy/experiment.json.'],
    ['no case in common', { 'other/experiment.json': named(['something_else']) }, ''],
  ]) {
    const q = await openFixture({ files, tab: 'setup' });
    eq(q.$('#opt-source').hidden ? '' : q.text('#opt-source'), expected, `experiment match: ${what}`);
  }

  // The optimizer writes fractions at `split.train` and the names it drew under `split.cases`.
  const drawn = configWith(c => {
    c.split = { train: 0.4, validation: 0.3, test: 0.3, seed: 0, cases: CONFIG.split };
  });
  const q = await openFixture({ config: drawn, files: { 'cases/experiment.json': named(all) }, tab: 'setup' });
  eq([q.text('#opt-method-sum').replace(/^.*? · /, ''), kv(q, '#opt-setup-kv').Cases, q.text('#opt-source')],
    ['8 train · 4 validation · 3 test', '8 train · 4 validation · 3 test split with seed 0', whole],
    'a split recorded as fractions, with the names under split.cases, is counted and matched by those names');
  await q.click('#opt-source a[data-experiment]');
  ok(q.$('#v-experiment').classList.contains('on') && q.text('#path') === 'cases/experiment.json'
    && q.$$('#case-list li[data-i]').length === 15 && !!q.$('#experiments-rows tr.row.on'),
    'the named experiment is a link to its cases, with its row marked in Experiments', q.text('#path'));
}

// ── a candidate's cases take the report page, under the run they came from ──
{
  const cases = [{ name: 'a', session_id: 's-a', input: 'q', expected_output: 'yes' },
    { name: 'b', session_id: 's-b', input: 'q2', expected_output: 'no' }];
  const experiment = { 'evals/experiment.json': { cases, evaluators: ['OutputEvaluator'] } };
  const report = {
    'evals/report.json': {
      overall_score: 1,
      cases: cases.map(c => ({ ...c, evaluator: 'OutputEvaluator', actual_output: c.expected_output })),
      scores: [1, 1], test_passes: [true, true],
      detailed_results: cases.map(() => [{ score: 1, test_pass: true, label: 'match' }]),
    },
  };
  const candidate = async q => {
    await go(q, 'candidates');
    await openCand(q, 'c10');
    await q.click('#opt-cands [data-open]');
  };

  const q = await openFixture({ files: { ...experiment, ...report } });
  await q.openReport();
  eq([q.text('#rep-count'), q.text('#crumbs')], ['2 of 2 rows', 'Reports/evals/report.json'],
    'a report from the folder is open under Reports');
  await candidate(q);
  eq([q.text('#rep-count'), q.text('#facts')], ['15 of 15 rows', 'The cases this candidate ran, as a report.'],
    'a candidate on the page says what its rows are, with nothing to add for one that can ship');
  eq([q.text('#crumbs'), q.$$('#crumbs a').map(a => a.dataset.go).join(), q.text('.tab.on'), q.$('#opt-tabs').hidden],
    [`Runs/${RUN_ID}/Candidates/c10`, 'opt-runs,opt-overview,opt-candidates', 'Runs', true],
    'the header names it Runs / run / Candidates / id, each part a way back, and the sidebar is on Runs');
  eq([texts(q, '#rep-cost-pick button'), q.$$('#rep-cost g.rbar').length, q.$$('#rep-heat rect.cell').length],
    [['Seconds', 'Tokens'], 15, 15],
    'and charted like any report, its cost per case read off what each trial recorded');
  ok(!q.$('#reports-rows tr.row.on'), 'no report in the Reports list is marked open while a candidate has the page');
  await q.click(q.$$('#crumbs a')[2]);
  ok(q.$('#v-opt-candidates').classList.contains('on') && !q.$('#opt-tabs').hidden, 'the Candidates crumb goes back to the run’s candidates',
    q.$('.view.on').id);
  await q.openReport();
  eq([q.text('#rep-count'), q.text('#crumbs'), q.text('#path'), q.$('#reports-rows tr.row').classList.contains('on')],
    ['2 of 2 rows', 'Reports/evals/report.json', 'evals/report.json', true],
    'opening the folder’s report again shows it, with its path, its row marked and the candidate gone');

  const q2 = await openFixture({ files: experiment });
  await candidate(q2);

  // The address tells the two apart, so Back and Forward land on the report each step was showing.
  const h = await openFixture({ files: { ...experiment, ...report } });
  await h.openReport();
  await candidate(h);
  const showing = () => `${h.text('#rep-count')} ${h.text('#crumbs').startsWith('Runs') ? 'candidate' : 'own'}`;
  const travel = async delta => { await h.travel(delta); return `${h.$('.view.on').id} ${showing()}`; };
  eq(h.window.location.pathname, '/optimization/demo/candidates/c10/report', 'a candidate on the report page is named in the address');
  eq([await travel(-2), await travel(2)], ['v-report 2 of 2 rows own', 'v-report 15 of 15 rows candidate'],
    'Back to the folder\'s report shows that report, and Forward brings the candidate\'s cases back');
  const reopened = await openFixture({ files: { ...experiment, ...report } });
  await reopened.visit(h.window.location.pathname);
  eq([`${reopened.$('.view.on').id} ${reopened.text('#rep-count')}`, reopened.text('#crumbs')],
    ['v-report 15 of 15 rows', `Runs/${RUN_ID}/Candidates/c10`], 'and the address alone reopens them');
  await q2.click('#crumbs [data-go="opt-candidates"]');
  ok(q2.$('#v-opt-candidates').classList.contains('on') && q2.text('#rep-count') === '',
    'the Candidates crumb goes there, leaving no report loaded', q2.text('#rep-count'));
  await q2.click('.tab[data-view="reports"]');
  ok(q2.$('#v-reports').classList.contains('on') && q2.$('#pagehead').hidden, 'so Reports is the list again, with no header', q2.$('.view.on').id);
}

// ── partial and malformed logs ──
{
  const cut = jsonl(EVENTS.slice(0, 40)) + JSON.stringify(EVENTS[40]).slice(0, 30);
  const running = await openFixture({ events: cut, tab: 'overview' });
  ok(running.text('#opt-note') === '' && running.text('#facts') === 'still running'
    && marks(running, '#opt-series-chart').length > 0,
    'a half-written last line is not damage: the run reads as still running and draws what it has',
    running.text('#opt-note'));
  ok(/No result yet/.test(running.text('#opt-result'))
    && /Best feasible objective so far: 0\.\d+ \(c\d\), on the training split/.test(running.text('#opt-result'))
    && /No champion has been picked yet/.test(running.text('#opt-winner')),
    'before the end phase there is no result and no champion, only the training best named as such',
    running.text('#opt-result'));

  const torn = `${jsonl(EVENTS.slice(0, 5))}{"event": "tri\n${jsonl(EVENTS.slice(5))}`;
  const broken = await openFixture({ events: torn, tab: 'overview' });
  ok(/line 6/.test(broken.text('#opt-note')) && broken.$('#opt-note').classList.contains('warn'),
    'a broken line in the middle is reported by number, as a warning', broken.text('#opt-note'));

  const configOnly = await open({ 'demo/run.json': CONFIG });
  ok(configOnly.$('.tab[data-view="opt-runs"]').hidden && skips(configOnly).some(s => /demo\/run\.json/.test(s)),
    'a run.json with no event log is not a run, and is reported as unusable', skips(configOnly).join(' | '));
  const logOnly = await open({ 'demo/events.jsonl': jsonl(EVENTS) });
  ok(logOnly.$('.tab[data-view="opt-runs"]').hidden && skips(logOnly).some(s => /no run\.json beside it/.test(s)),
    'an event log with no run.json is not one either, and says what it lacks', skips(logOnly).join(' | '));
}

// ── two runs in one tree ──
{
  const q = await open({
    'a/run.json': CONFIG, 'a/events.jsonl': jsonl(EVENTS),
    'b/run.json': { ...CONFIG, run_id: 'second-run', name: 'other' }, 'b/events.jsonl': jsonl(EVENTS.slice(0, 20)),
  }, 'overview');
  const on = () => q.$$('#opt-runs-rows tr[data-opt-run]').map(tr => tr.classList.contains('on'));
  ok(on().length === 2 && q.text('#opt-runs-count') === '2 runs', 'both runs are listed and counted',
    q.text('#opt-runs-count'));
  ok(q.text('#crumbs') === `Runs/${RUN_ID}` && same(on(), [true, false]), 'the first is open, and its row is marked',
    q.text('#crumbs'));
  await go(q, 'runs');
  await q.click(q.$$('#opt-runs-rows tr[data-opt-run]')[1]);
  ok(q.text('#crumbs') === 'Runs/second-run' && q.$('#v-opt-overview').classList.contains('on')
    && same(on(), [false, true]), 'picking the other run opens its Overview and moves the mark', q.text('#crumbs'));
  eq(q.window.location.pathname, '/optimization/b/overview', 'and the address names that run by its directory');
  await go(q, 'candidates');
  await openCand(q, 'c0');
  await go(q, 'runs');
  await q.click(q.$$('#opt-runs-rows tr[data-opt-run]')[0]);
  await go(q, 'candidates');
  ok(candRow(q, 'c0') && !q.$('#opt-cands tr.detail'), 'a candidate opened in one run is not open in the next, though both have a c0');
}

// ── a trial's verdict comes from its evaluators, not from having finished ──
{
  const unjudged = eventsWith(e => {
    for (const v of Object.values((e.record || {}).evaluations || {})) delete v.test_pass;
    return e;
  });
  const q = await openFixture({ events: unjudged, tab: 'candidates' });
  await openCand(q, 'c10');
  eq([...new Set(rows(q, '#opt-cands tr.trial').map(t => t.at(-1)))], ['completed'],
    'a trial with no verdict recorded claims neither pass nor fail');
  await openCand(q, 'c2');
  await q.click('#opt-cands [data-open]');
  eq(lines(q, '#rep-rows tr.case')[4], 'train_case_05 | Helpfulness | 0.37 | fail',
    'and in the report a trial whose agent failed is a failing row, whatever it scored');
}

// ── nothing a run record says is taken as markup ──
{
  const hostile = '<img src=x onerror="window.__xss=1">';
  const config = configWith(c => {
    c.budget = { max_rollouts: hostile, max_iterations: hostile };
    c.final_pick = { ...c.final_pick, n_finalists: hostile };
  });
  const events = eventsWith(e => {
    if (e.event === 'test') e.report.objective = { ...e.report.objective, n_cases: hostile, ci_low: 0.1, ci_high: 0.1 };
    if (e.event === 'trial' && e.record.case_name === 'train_case_01') e.record.status = 'view on app';
    return e;
  });
  const q = await openFixture({ config, events });
  for (const tab of ['setup', 'overview', 'progress', 'candidates']) await go(q, tab);
  ok(q.$$('main img').length === 0 && q.window.__xss === undefined,
    'a count or budget in the record never becomes an element', q.$$('main img').length);
  has(q.text('#opt-setup-kv'), /<img/, 'it is shown as the text it is');
  ok(q.$$('#opt-status-chart .view, #opt-status-chart .app').length === 0
    && q.$$('#opt-status-chart .other').length > 0,
    'an unknown trial status is drawn as "other", not used as a class name');
}

// ── chart marks: hover, legend focus, and a click through to the candidate ──
{
  const q = await openFixture({ tab: 'progress' });
  const over = async el => {
    el.dispatchEvent(new q.window.MouseEvent('mouseover', { bubbles: true, clientX: 40, clientY: 40 }));
    await tick();
  };
  const tip = q.$('.tip');
  const mark = () => q.$('#opt-series-chart circle.pt[data-cand="c8"]');
  const hiddenBefore = tip.hidden;
  await over(mark());
  ok(hiddenBefore && !tip.hidden, 'the tooltip appears only when a mark is hovered');
  eq([...tip.children].map(el => el.textContent), ['c8', 'infeasible', '0.69', 'Click to open in Candidates'],
    'headed by the candidate, with what the mark stood for and where a click goes');
  ok(mark().classList.contains('hot') && q.$('#opt-series-chart svg').classList.contains('hovering'),
    'the hovered mark is picked out');
  ok(mark().querySelector('title') === null && mark().getAttribute('aria-label').startsWith('c8'),
    'the native tooltip gives way without losing its label', mark().getAttribute('aria-label'));
  await over(q.$('#opt-series-head'));
  ok(tip.hidden && !mark().classList.contains('hot'), 'leaving the mark puts everything back');

  await over(q.$$('#opt-series-legend span').find(s => s.querySelector('i.infeasible')));
  const focused = q.$('#opt-series-chart').closest('.chart').dataset.focus;
  await over(q.$('#opt-series-head'));
  ok(focused === 'infeasible' && q.$('.chart[data-focus]') === null,
    'hovering a legend entry focuses its marks until the pointer leaves', focused);

  await q.type('#opt-find', 'no such candidate');
  await q.click(mark());
  ok(q.$('#v-opt-candidates').classList.contains('on') && !!candRow(q, 'c8').querySelector('.caret.open'),
    'clicking a mark opens that candidate on the Candidates tab, whatever filter was on');
}

// ── Setup, in both spellings of run.json ──
const WRITTEN = configWith(c => {
  for (const key of ['is_pareto', 'goal_metrics', 'blocks', 'runs', 'run_id']) delete c[key];
  delete c.split.seed;
  Object.assign(c, {
    objective: 'Pareto(Maximize(Helpfulness), Minimize(tokens))',
    metrics: {
      Helpfulness: { kind: 'evaluator', roles: ['objective'] },
      tokens: { kind: 'usage', roles: ['objective', 'constraint'] },
      Latency: { kind: 'usage', roles: ['constraint'] },
    },
    search: { blocks: [{ name: 'text', keys: ['system_prompt'] }, { name: 'numbers', keys: ['max_tokens'] }] },
    runs_per_candidate: 2,
    seed: 7,
    selection: { name: 'top_ranked', config: {} },
    score_model: { name: 'BayesBenchModel', config: {} },
    final_pick: { n_finalists: 3, rule: { name: 'Mean', config: {} } },
  });
});
{
  const constraints = 'Constraints: tokens <= 4000 met by 0.9 of the cases to count';
  const budget = 'Budget: 240 rollouts · 6 iterations';
  const tail = ['A failed agent scores: worst', 'Started in: /home/liz/support-agent'];
  for (const [spelling, config, brief, machinery] of [
    ['as events.py documents it', CONFIG,
      ['Objective: WeightedSum(Maximize(Helpfulness), Minimize(tokens, scale=1000))', constraints,
        'Cases: 8 train · 4 validation · 3 test split with seed 20261110', budget],
      ['Final pick: 3 finalists · top_ranked', 'Score model: bayesbench', ...tail, `Run id: ${RUN_ID}`]],
    ['as the optimizer writes it', WRITTEN,
      ['Objective: Pareto(Maximize(Helpfulness), Minimize(tokens))',
        'Shape: a Pareto front, so there is no single best value', constraints,
        'Cases: 8 train · 4 validation · 3 test split with seed 7', 'Repeats: each case run 2 times', budget],
      ['Final pick: 3 finalists · Mean', 'Parents chosen by: top_ranked', 'Score model: BayesBenchModel', ...tail,
        'Run id: demo']],
  ]) {
    const q = await openFixture({ config, tab: 'setup' });
    eq(pairs(q, '#opt-setup-kv'), brief,
      `Setup, ${spelling}: the brief, with no row restating the objective's goals or the blocks`);
    eq([...pairs(q, '#opt-mech-kv'), q.text('#opt-mech-sum')], [...machinery, `${machinery.length} settings`],
      `Setup, ${spelling}: the machinery, counted on its fold`);
    if (config !== WRITTEN) continue;
    eq(q.$$('#opt-space-rows tr').map(tr => (tr.querySelector('.tag') || {}).textContent || '-').join(),
      'text,-,numbers,-,-,-',
      'each key of the space is tagged with the block that declares it, and an unclaimed key with none');
  }

  const unseeded = { ...WRITTEN, seed: undefined, objective: 'Maximize(Helpfulness - 0.01 * tokens)' };
  const expression = await openFixture({ config: unseeded, tab: 'setup' });
  eq(['Goal metrics', 'Cases'].map(k => kv(expression, '#opt-setup-kv')[k]),
    ['Helpfulness, tokens', '8 train · 4 validation · 3 test'],
    'an objective over an expression names no goal, so the objective-role metrics are listed; no seed, none shown');
}

// ── Tunables: a space that does not cover the log, and a record with none ──
{
  const narrowed = configWith(c => {
    c.space = [null, ...c.space.filter(r => !['temperature', 'max_tokens'].includes(r.key))];
  });
  const partial = await openFixture({ config: narrowed, tab: 'tunables' });
  ok(partial.$$('#opt-space-rows tr').length === 4 && partial.$('#opt-space-note').classList.contains('warn')
    && partial.text('#opt-space-note')
      === 'The log also changed temperature, max_tokens, which this space does not declare.',
    'keys the log changed without the space declaring them are named, as a warning, and a null row is skipped', partial.text('#opt-space-note'));

  const spaceless = await openFixture({ config: configWith(c => { delete c.space; }), tab: 'setup' });
  const brief = kv(spaceless, '#opt-setup-kv');
  ok(spaceless.$('#opt-space-wrap').hidden && /system_prompt/.test(brief['Actually varied'])
    && brief['Tunable blocks'] === 'instructions, tools',
    'with no declared space there is no table, and Setup lists the blocks and the keys the log changed',
    JSON.stringify(brief));
}

/* ══════════ runs written inline ══════════ */

// ── Progress: where best-so-far starts, and whose the training best is ──
{
  const late = await openLog({ objective: 'Maximize(H)' }, [START, iterationEnd(0, [prop('c0', 0, null)], 0),
    iterationEnd(1, [prop('c1', 1, 0.4)], 4), iterationEnd(2, [prop('c2', 2, 0.6)], 4)]);
  const plotted = marks(late, '#opt-series-chart');
  const lineStart = Math.min(...pathPoints(late.$('#opt-series-chart path.best')).map(([x]) => x));
  ok(plotted.length === 2 && lineStart === plotted[0].cx,
    'an unscored opening proposal is not plotted, and the line starts at the first scored candidate',
    plotted.map(m => m.id).join());
  has(stats(late)[1], /^Best on training \| 0\.60 \|/, 'the best is the better of the two measured');

  const split = await openLog({ objective: 'Maximize(H)' }, [START,
    iterationEnd(0, [prop('c0', 0, 0.4)], 4), iterationEnd(1, [prop('c1', 1, 0.9)], 4),
    { event: 'picked', iteration: 2,
      report: { champion_id: 'c0', measured: { c0: { H: { mean: 0.6 } }, c1: { H: { mean: 0.3 } } } } }]);
  eq(stats(split), ['Rollouts | 8 | spent · no budget declared',
    'Best on training | 0.90 | c1 · highest here, not the pick'],
    'when validation picks another candidate, the training best names its own and says it was not the pick');
}

// ── a run with nothing measured yet, on a Pareto objective ──
{
  const q = await openLog({ objective: 'Maximize(H)', is_pareto: true }, [START]);
  ok(!q.$('.tab[data-view="opt-runs"]').hidden && stats(q)[1].startsWith('Best on training | — |'),
    'a run with only a config event is a run, with a dash for its best', stats(q)[1]);
  ok(/No candidate has been scored/.test(q.text('#opt-series-chart'))
    && /No iteration has finished/.test(q.text('#opt-budget-chart')) && /No candidate yet/.test(q.text('#opt-cands')),
    'each empty chart and the candidate table say why they are empty');
  ok(/No iteration has finished yet/.test(q.text('#opt-explain')) && !/built 0 versions/.test(q.text('#opt-explain')),
    'the account describes the method without counting what it has not built', q.text('#opt-explain'));
  has(q.text('#opt-note'), /Pareto/, 'a Pareto run is flagged');
  ok(q.$('#opt-series-pick').hidden && q.$('#opt-usage-wrap').hidden && !q.$('#opt-cost').hidden,
    'with no metric and no usage there is no series picker and no spend chart, but the cost drawer stays');
}

// ── a candidate the optimizer never labelled is measured from its trials ──
{
  const trial = (name, score, usage) => ({
    event: 'trial', iteration: 0,
    record: { candidate_id: 'c0', case_name: name, run_index: 0, split: 'train', status: 'completed',
      evaluations: { Helpfulness: { score, reason: 'judged' } }, usage },
  });
  const q = await openLog({ objective: 'Maximize(Helpfulness)' }, [START,
    trial('a', 0.2, { tokens: 100, model_calls: 1, seconds: 1.5 }),
    trial('b', 0.4, { tokens: 300, model_calls: 3, seconds: 2.5 }),
    iterationEnd(0, [prop('c0', 0, 0.3)], 2)]);
  eq(texts(q, '#opt-series-pick button').join(), 'Objective,Helpfulness', 'the picker offers only what was measured');
  await pick(q, '#opt-series-pick', 'm:Helpfulness');
  ok(same(marks(q, '#opt-series-chart').map(m => m.title), ['c0 · Helpfulness 0.30'])
    && q.$$('#opt-series-chart path.metric-line').length === 0,
    'the metric is the mean of its trials, and a single point draws no line',
    titles(q, '#opt-series-chart circle').join());
  ok(same(titles(q, '#opt-usage-chart rect.col'), ['iteration 0: 400 over 2 trials'])
    && !/excluding/.test(q.text('#opt-usage-cap')),
    'usage is summed over the iteration, with no caveat when nothing follows it', q.text('#opt-usage-cap'));
}

// ── a trial recorded again replaces the earlier record ──
{
  const trial = (status, score, more) => ({
    event: 'trial', iteration: 0,
    record: { candidate_id: 'c0', case_name: 'a', run_index: 0, split: 'train', status,
      evaluations: score == null ? {} : { H: { score } }, usage: { tokens: 100 }, ...more },
  });
  const q = await openLog({ objective: 'Maximize(H)' }, [START, trial('infrastructure_invalid', null),
    trial('completed', 0.8, { attempts: 2, previous_attempts: [{ status: 'infrastructure_invalid' }] }),
    iterationEnd(0, [prop('c0', 0, 0.8)], 1)]);
  eq([q.text('#opt-cost-sum'), q.$$('#opt-status-chart .legend span').map(flat).join(' | ')],
    ['100 tokens · 1 trial', 'completed 1 | train 1'], 'it is counted once, as it ended, and so is what it spent');
  await go(q, 'candidates');
  await openCand(q, 'c0');
  eq(lines(q, '#opt-cands tr.trial'), ['a | train | H 0.80 | completed'], 'and listed once under its candidate');
}

// ── proposals that built nothing new: a tie, a duplicate, two invalid patches ──
{
  const none = { candidate_id: null, number: null, outcome: 'invalid', objective: null };
  const q = await openLog({ objective: 'Maximize(H)' }, [START, iterationEnd(0, [prop('c0', 0, 0.4)]),
    iterationEnd(1, [prop('c1', 1, 0.6), prop('c2', 2, 0.6), prop('c1', 1, 0.6, 'duplicate'), none, none])]);
  eq([stats(q)[1], texts(q, '#opt-series-chart text.vlab').join()],
    ['Best on training | 0.60 | c1 · highest here', 'c1 · 0.60'],
    'of two candidates level at the top, the figure and the flag on the chart name the same one: the first to get there');
  await go(q, 'setup');
  has(q.text('#opt-explain'), /built 3 versions/, 'a proposal that repeats a candidate is not another version built');
  await go(q, 'candidates');
  await q.click(q.$$('#opt-cands tr.case').at(-1));
  eq(q.$$('#opt-cands tr.case').map(tr => tr.getAttribute('aria-expanded')).slice(-2), ['false', 'true'],
    'and of two invalid patches in one iteration, opening one leaves the other closed');
}

// ── the candidate axis: ids too long for a slot, and more labels than fit ──
{
  const digests = ['a8997951dbcb', 'da69b054a6d3', '2002243454a9', 'b85bcdfd5997', 'abbf9531c55d', '610a97189bad',
    'dd141fd3c81c', '85e2c6312229'];
  const ids = await openLog({ objective: 'Maximize(H)' },
    [START, ...digests.map((id, i) => iterationEnd(i, [prop(id, i + 1, 0.3 + i * 0.05)]))]);
  eq([texts(ids, '#opt-series-chart text.catlab').join(), texts(ids, '#opt-series-chart text.vlab').join()],
    ['#1,#2,#3,#4,#5,#6,#7,#8', '#8 · 0.65'],
    'a digest id gives way to the proposal number, on the axis and on the flag over the best mark');

  const many = await openLog({ objective: 'Maximize(H)' },
    [START, ...Array.from({ length: 24 }, (_, i) => iterationEnd(i, [prop(`c${i}`, i, 0.5)]))]);
  eq(texts(many, '#opt-series-chart text.catlab').join(), 'c0,c2,c4,c6,c8,c10,c12,c14,c16,c18,c20,c22',
    'labels are thinned by their width so none overlap');
}

// ── Overview: the verdict is the interval, in the card, the picture and the run list ──
const proposed = (id, number, prompt) => ({
  event: 'propose', iteration: number, candidate_id: id, number, parent_id: 'c0', values: { prompt, temperature: 0.7 },
});
const resultLog = (report, extra = []) => [START,
  proposed('c0', 0, 'be helpful'), proposed('c1', 1, 'be helpful and brief'),
  iterationEnd(0, [prop('c0', 0, 0.5)]), iterationEnd(1, [prop('c1', 1, 0.6)]),
  ...extra, ...(report ? [{ event: 'test', iteration: 1, report }] : [])];
const interval = (mean_diff, ci_low, ci_high) => ({ mean_diff, ci_low, ci_high, n_cases: 6 });
const MEANS = { c1: { H: { mean: 0.6 } }, c0: { H: { mean: 0.5 } } };
const testReport = (objective, more = {}) =>
  ({ champion_id: 'c1', baseline_id: 'c0', objective, difference: { H: objective }, means: MEANS, ...more });
const picked = report => ({ event: 'picked', iteration: 1, report: { champion_id: 'c1', ...report } });
const openResult = (report, { objective = 'Maximize(H)', extra, ...config } = {}) =>
  openLog({ objective, ...config }, resultLog(report, extra));

for (const [name, objective, verdict, bar, listed, read] of [
  ['straddles zero', interval(0.1, -0.04, 0.24),
    /\+0\.1, 95% interval \[-0\.04, 0\.24\]: includes zero, so this run has not shown an improvement/,
    'H ↑ / flat / across zero', '+0.1 held out · includes zero (flat)', 'not shown'],
  ['is below zero', interval(-0.08, -0.19, -0.02),
    /Objective -0\.08, .* below zero, so shipping the champion would be a regression/,
    'H ↑ / bad / left of zero', '-0.08 held out · below zero (down)', 'a regression'],
  ['is narrowly clear', interval(0.1071, 0.004, 0.2103), /\[0\.004, 0\.2103\]: clear of zero/,
    'H ↑ / good / right of zero', '+0.1071 held out · clear of zero (up)', 'clear of zero'],
  ['is absent', { mean_diff: 0.1, n_cases: 6 }, /\+0\.1 against the baseline, with no interval in the report/,
    null, '+0.1 held out · no interval (flat)', 'no interval'],
  ['has no width', interval(0.7, 0.7, 0.7),
    /\+0\.7 against the baseline, the same on every one of 6 test cases, so its 95% interval has no width/,
    null, '+0.7 held out · no width (flat)', 'no width'],
]) {
  const q = await openResult(testReport(objective));
  has(said(q), verdict, `an interval that ${name}: the verdict`);
  eq(q.$('#opt-result .rstrip') ? strip(q) : null, bar && [bar], `an interval that ${name}: the picture`);
  eq(runResult(q), listed, `an interval that ${name}: the run list`);
  /* An interval of no width is not an absent one: the card reads it as the run list and the verdict do. */
  eq(texts(q, '#opt-hero .champ-read').join(), read, `an interval that ${name}: the card`);
}

// ── Overview: a run with only a validation report ──
{
  const validated = await openResult(null, { extra: [picked({ measured: MEANS, reason: 'c1 ranked first.' })] });
  has(validated.text('#opt-result .rlabel'), /^Result · validation split/,
    'a run with no test split is not called held out');
  ok(/no overall comparison/.test(said(validated)) && /flattering/.test(said(validated)),
    'it has no objective difference to report, and hedges the gap it chose on', said(validated));
  eq(lines(validated, '#opt-result .rtab tbody tr'), ['H | 0.5 | → | 0.6 | +0.1 |  | '],
    'it still shows the means, which a validation report calls `measured`');
  ok(!validated.$('#opt-result .rstrip') && runResult(validated) === 'not measured (no figure)',
    'with no interval there is no picture, and the run list says not measured rather than 0', runResult(validated));

  const alone = await openResult(null, { extra: [picked({ measured: { c1: MEANS.c1 } })] });
  ok(/nothing here to compare them against/.test(said(alone)) && !/flattering/.test(said(alone)),
    'a report with no baseline figures says so, and hedges no gap', said(alone));

  const faint = picked({ measured: MEANS, difference: { H: interval(0.1, 0.02, 0.18) } });
  const weak = await openResult(null, { extra: [faint] });
  ok(weak.$('#opt-result .rstrip').classList.contains('weak') && !/held-out/.test(stripNote(weak)),
    'a validation comparison is drawn faint and does not claim its cases were held out', stripNote(weak));

  const compared = await openLog({ objective: 'Maximize(H)' },
    [picked({ baseline_id: 'c0', objective: { mean_diff: 0.2, ci_low: 0.05, ci_high: 0.35, n_cases: 4 } })]);
  eq(runResult(compared), '+0.2 validation · clear of zero (up)',
    'an objective difference on the validation split is listed as validation, not held out');
  const searching = await openLog({ objective: 'Maximize(H)' }, [START, { event: 'trial', iteration: 0 }]);
  eq(runResult(searching), 'not measured (no figure)', 'a run still searching has no result rather than a zero one');
}

// ── Overview: metrics the run was not aiming at, and the row format ──
{
  const usage = { tokens: { mean: 420 }, seconds: { mean: 0.002 }, model_calls: { mean: 1 } };
  const means = { c1: { ...MEANS.c1, ...usage }, c0: { ...MEANS.c0, ...usage, tokens: { mean: 470 } } };
  const q = await openResult(testReport(interval(0.1, 0.02, 0.18), { means }), { constraints: ['tokens <= 500'] });
  const leading = () => q.$$('#opt-result .rtab tbody:not([hidden]) tr').map(r => r.children[0].textContent).join();
  eq([leading(), q.text('#opt-result .rmore')], ['H,tokens', '+ 2 metrics the run was not aiming at'],
    'the table leads with the goal and the constrained metric, the others counted on a fold');
  await q.click('#opt-result .rmore');
  ok(leading() === 'H,tokens,seconds,model_calls' && /^−/.test(q.text('#opt-result .rmore')),
    'which opens onto them', leading());

  const unnamed = await openResult(testReport(interval(0.1, 0.02, 0.18)), { objective: 'Maximize(score)' });
  ok(!unnamed.$('#opt-result .rmore') && unnamed.$$('#opt-result .rtab tbody tr').length === 1,
    'a report whose metrics the config names none of is shown whole, with no fold');

  const uneven = { c1: { H: { mean: 1 } }, c0: { H: { mean: 0.1667 } } };
  const digits = await openResult(testReport(interval(0.8333, 0.4975, 1.1692), { means: uneven }));
  eq(lines(digits, '#opt-result .rtab tbody tr'), ['H | 0.1667 | → | 1.0000 | +0.8333 | [0.4975, 1.1692] | '],
    'every figure in a row carries the decimals the widest of them needs');
}

// ── Overview: the picture takes each direction from the objective expression ──
{
  const compare = (objective, difference, means) =>
    ({ champion_id: 'c1', baseline_id: 'c0', objective, difference, means });

  const min = await openResult(compare(interval(0.11, 0.03, 0.19), { tokens: interval(180, 60, 300) },
    { c1: { tokens: { mean: 3340 } }, c0: { tokens: { mean: 3160 } } }),
  { objective: 'Minimize(tokens) with Maximize(H)' });
  eq(strip(min), ['tokens ↓ / bad / right of zero', 'Objective ↑ / good rs-obj / right of zero'],
    'a rise in a minimised metric is bad, and the objective gets its own row when its interval is not a metric\'s');

  const plain = await openResult(compare(interval(0.1, 0.02, 0.18), { latency: interval(0.4, 0.1, 0.7) },
    { c1: { latency: { mean: 2.4 } }, c0: { latency: { mean: 2.0 } } }));
  eq(strip(plain)[0], 'latency / plain / right of zero',
    'a metric the objective states no direction for is neither good nor bad, and has no arrow');

  const oneWide = await openResult(compare(interval(0.3, 0.1, 0.5),
    { H: interval(0.3, 0.1, 0.5), tokens: interval(-331, -331, -331) },
    { c1: { H: { mean: 0.6 }, tokens: { mean: 369 } }, c0: { H: { mean: 0.3 }, tokens: { mean: 700 } } }),
  { objective: 'Maximize(H) with Minimize(tokens)' });
  ok(same(strip(oneWide), ['H ↑ / good / right of zero'])
    && /nothing to draw for tokens \(the same on every case\)/.test(stripNote(oneWide)),
    'a goal whose interval has no width gets no bar, and is named with that reason', stripNote(oneWide));
}

// ── Overview: a champion whose configuration the log does not carry ──
{
  const q = await openLog({ objective: 'Maximize(H)' }, [START, iterationEnd(0, [prop('c0', 0, 0.5)]),
    { event: 'test', iteration: 0, report: testReport(interval(0.1, 0.02, 0.2)) }]);
  ok(q.text('#opt-winner .wid') === 'c1' && /records no values/.test(q.text('#opt-winner')),
    'the champion is named, and its configuration said to be missing', q.text('#opt-winner'));
  ok(!q.$('#opt-winner .lineage') && !q.$('#opt-winner [data-act="copy-winner"]'),
    'with no lineage and nothing to copy');
}

// ── Candidates: transcript files that hold no conversation ──
{
  const docs = { spanless: { traces: [{ spans: [] }] }, listless: [] };
  const trial = name => ({
    event: 'trial', iteration: 0,
    record: { candidate_id: 'c0', case_name: name, split: 'train', status: 'completed', output: '', transcript: name },
  });
  const q = await openLog({ objective: 'Maximize(H)' },
    [START, ...Object.keys(docs).map(trial), iterationEnd(0, [prop('c0', 0, 0.5)])],
    Object.fromEntries(Object.entries(docs).map(([name, doc]) => [`r/transcripts/${name}.json`, doc])));
  await go(q, 'candidates');
  for (const [name, pattern] of [
    ['spanless', /recorded with no spans in it/], ['listless', /transcript is an empty list/],
  ]) {
    await openTrial(q, 'c0', name);
    has(panel(q).Transcript, pattern, `a ${name} transcript file says what it is instead of rendering nothing`);
  }
  eq(panel(q).Output, 'empty', 'and an empty output is told apart from none recorded');
}

// ── Trade-off ──
const labelled = (id, measured) => ({
  event: 'label', iteration: 1, candidate_id: id,
  label: { measured: Object.fromEntries(Object.entries(measured).map(([m, v]) => [m, { mean: v, n: 3 }])) },
});
const offered = (id, number, more) =>
  ({ event: 'propose', iteration: 1, candidate_id: id, number, parent_id: 'c0', ...more });
const BASELINE = { event: 'propose', iteration: 0, candidate_id: 'c0', number: 0, origin: 'baseline', parent_id: null };
/* The baseline c0 is proposed and labelled but is in no `iteration_end`, as real runs record it. */
const goalLog = (...ending) => [START,
  { ...BASELINE, values: { prompt: 'answer' } },
  ...[1, 2, 3, 4].map(n => offered(`c${n}`, n, { values: { prompt: `answer ${n}` } })),
  labelled('c0', { Quality: 0.5, tokens: 200 }),
  labelled('c1', { Quality: 0.9, tokens: 400 }),      // the best Quality, and the dearest
  labelled('c2', { Quality: 0.6, tokens: 100 }),      // the cheapest
  labelled('c3', { Quality: 0.4, tokens: 300 }),      // worse than c2 on both
  labelled('c4', { Quality: 0.7 }),                   // tokens never measured
  iterationEnd(1, [1, 2, 3, 4].map(n => prop(`c${n}`, n, 0.5)), 8),
  ...ending];
const FRONT = {
  event: 'picked', iteration: 2,
  report: { champion_id: 'c1', front: ['c1', 'c2'], measured: { c1: { Quality: { mean: 0.9 } } } },
};
const TWO_GOALS = 'Pareto(Maximize(Quality), Minimize(tokens))';
{
  const one = await openLog({ objective: 'Maximize(Quality)' }, goalLog(FRONT));
  ok(one.$('#opt-tradeoff').hidden && one.$('.tab[data-view="opt-trade"]').hidden
    && /^One goal, so nothing to trade offThis run optimized one goal, Quality, so it has a best value/.test(one.text('#opt-to-none')),
    'a run with one goal has no trade-off chart and no tab for it, and the view says why', one.text('#opt-to-none'));

  const two = await openLog({ objective: TWO_GOALS }, goalLog(FRONT));
  const at = byId(marks(two, '#opt-to-chart'));
  ok(!two.$('#opt-tradeoff').hidden && !two.$('.tab[data-view="opt-trade"]').hidden && two.$('#opt-to-none').hidden
    && two.$('#opt-to-x').hidden && two.$('#opt-to-y').hidden,
    'two goals: the chart and its tab are there, with no axis pickers');
  eq([two.text('#opt-to-sum'), two.text('#opt-to-head')], ['Quality ↑ against tokens ↓', 'tokens against Quality'],
    'named by its goals and their directions, the first goal across and the second up');
  eq([Object.keys(at).sort().join(), two.text('#opt-to-stats')],
    ['c0,c1,c2,c3', '4 candidates plotted. 1 other was not measured on both goals: c4.'],
    'every candidate measured on both goals is plotted, the baseline included, and the one that was not is named');
  ok(at.c2.cy < at.c1.cy && at.c1.cx > at.c2.cx && at.c3.cx < at.c2.cx && at.c3.cy > at.c2.cy,
    'both axes are turned so better is up and right: fewer tokens is higher, and c3, worse on both, is below c2',
    JSON.stringify(at));
  has(two.text('#opt-to-chart'), /fewer tokens is better, upward along this axis/,
    'the minimised axis says which way better is');
  eq([ringed(two), at.c0.cls, at.c1.cls], ['c1,c2', 'baseline', 'champion'],
    'the recorded front is ringed, and the baseline and champion are marked');
  const stair = two.$('#opt-to-chart path.front-line').getAttribute('d');
  ok(stair.includes(`H${at.c1.cx}`) && stair.includes(`H${at.c2.cx}`),
    'the front line has a step at each candidate on the front', stair);
  eq(['c0', 'c1', 'c2', 'c3'].map(id => at[id].cand || 'no link'), ['no link', 'c1', 'c2', 'c3'],
    'a mark links to its candidate only when the Candidates table has a row for it, which the baseline here has not');
  has(two.text('#opt-note'), /front is on the Trade-off tab/, 'the Pareto note points at this tab');
  await go(two, 'candidates');
  await openCand(two, 'c1');
  eq(flat(two.$('#opt-cands .cand-sum .stat.line')), 'c1 ← c0 Derived from',
    'a candidate descends from the baseline though no iteration lists the baseline as a proposal');
  ok(two.$('#opt-effects').hidden, 'a run with no effects.json has no effects drawer');

  // Where the front comes from: the end report, else the champion, else the last iteration's snapshot.
  has(two.text('#opt-to-cap'), /the 2 candidates on the front in its end-phase report/, 'front from the end report');
  for (const [source, ending, ids, caption] of [
    ['the champion', [{ ...FRONT, report: { ...FRONT.report, front: [] } }], 'c1', /the champion it picked/],
    ['the last snapshot', [{ ...iterationEnd(2, [], 0), best_after: ['c2'] }], 'c2',
      /the 1 candidate the last finished iteration had best so far/],
  ]) {
    const q = await openLog({ objective: TWO_GOALS }, goalLog(...ending));
    ok(ringed(q) === ids && caption.test(q.text('#opt-to-cap')), `front from ${source}`,
      `${ringed(q)}: ${q.text('#opt-to-cap')}`);
  }

  // Three goals: the recorded front spans more dimensions than the chart, so the pair's front is computed.
  const three = await openLog({ objective: 'Pareto(Maximize(Quality), Minimize(tokens), Minimize(seconds))' },
    [START, BASELINE, offered('c1', 1), offered('c2', 2),
      labelled('c0', { Quality: 0.5, tokens: 200, seconds: 3 }),
      labelled('c1', { Quality: 0.9, tokens: 400, seconds: 1 }),
      labelled('c2', { Quality: 0.6, tokens: 100, seconds: 9 }),
      iterationEnd(1, [prop('c1', 1, 0.9), prop('c2', 2, 0.6)], 4)]);
  ok(three.text('#opt-to-sum') === '3 goals' && !three.$('#opt-to-x').hidden
    && texts(three, '#opt-to-y button').join() === 'tokens ↓,seconds ↓',
    'three goals are counted, with a pair to pick and the goal on one axis off the other\'s list',
    texts(three, '#opt-to-y button').join());
  ok(ringed(three) === 'c1,c2' && /worked out here.*front over all 3/.test(three.text('#opt-to-cap')),
    'the front of the pair is computed with each goal\'s direction, and said to be',
    `${ringed(three)}: ${three.text('#opt-to-cap')}`);
  await pick(three, '#opt-to-y', 'seconds');
  eq([three.text('#opt-to-head'), ringed(three)], ['seconds against Quality', 'c1'],
    'picking another goal redraws the pair and its front');
}

// ── Tunables: effects files with nothing, or one goal, to show ──
{
  const quality = { goals: [{ metric: 'Quality', better: 'higher' }], metrics: { Quality: { observations: 12 } } };
  const unfitted = { ...quality, effects: [], factors: [], reason: 'too few candidates to separate the keys' };
  const empty = await openLog({ objective: 'Maximize(Quality)' }, goalLog(FRONT), { 'r/effects.json': unfitted });
  ok(empty.$('#opt-effects').hidden, 'a model that came back with no estimates has no drawer either');

  const fitted = {
    ...quality,
    factors: [{ key: 'prompt', kind: 'text', levels: ['default', 'rewritten'], reference: 'default' }],
    effects: [{ metric: 'Quality', key: 'prompt', level: 'rewritten', reference: 'default', mean: 0.2,
      ci90: [0.1, 0.3], p_positive: 0.98 }],
  };
  const aimless = await openLog({ objective: 'Maximize(Quality)' }, goalLog(FRONT), { 'r/effects.json': { ...fitted, goals: [] } });
  ok(aimless.$('#opt-effects').hidden && aimless.$$('#opt-cands tr.case').length === 4,
    'estimates with no goal to read them for have no drawer, and the rest of the run still draws', aimless.errs.join());
  const oneGoal = await openLog({ objective: 'Maximize(Quality)' }, goalLog(FRONT), { 'r/effects.json': fitted });
  ok(!oneGoal.$('#opt-effects').hidden && oneGoal.$('#opt-eff-goal').hidden
    && texts(oneGoal, '#opt-eff-chart .eff-row text.tick').join() === '+0.2 · 0.98',
    'one goal still gets the plot, with no goal picker', texts(oneGoal, '#opt-eff-chart .eff-row text.tick').join());
}

// ── the run's files are read on request ──
{
  const trial = EVENTS.find(e => e.event === 'trial' && e.record.candidate_id === 'c10' && e.record.case_name === 'train_case_01');
  const transcript = `demo/transcripts/${trial.record.transcript.hash}.json`;
  const experiment = { path: 'e.json', name: 'e.json', dir: '.', size: 1, modified: 5, cases: 1, evaluators: [], reports: [] };
  const run = { id: 'demo', dir: 'demo', run_id: RUN_ID, config: CONFIG, events_path: 'demo/events.jsonl', modified: 1,
    tuned_against: { path: 'e.json', status: 'verified', split_cases: 15, matched: 15 } };
  const folder = { root: '/served', depth: 3, read_only: false };
  /* What is on disk: the experiment's stamp, and whether the transcript has been written yet. */
  const disk = { stamp: 5, transcript: false };
  let saved = null;
  const answer = (body, status = 200) => ({ ok: status < 400, status, statusText: '', json: async () => structuredClone(body) });
  const q = await page({ fetchImpl: async (url, opts = {}) => {
    const [route, query] = String(url).split('?');
    const asked = new URLSearchParams(query).get('path');
    if (route === '/api/config') return answer(folder);
    if (route === '/api/scan') return answer({ ...folder, fingerprint: 'f', experiments: [experiment], reports: [], skipped: [], optimizer_runs: [run] });
    if (route === '/api/events') return answer({ events: EVENTS });
    if (opts.method === 'PUT') { saved = JSON.parse(opts.body); return answer({ path: asked, modified: 6, cases: 1 }); }
    if (asked === 'e.json') return answer({ modified: disk.stamp, document: { cases: [{ name: 'train_case_01', input: 'q' }], evaluators: [] } });
    return asked === transcript && disk.transcript
      ? answer({ document: JSON.parse(TRANSCRIPTS[transcript]) }) : answer({ error: 'no such file' }, 404);
  } });
  await go(q, 'candidates');
  await openTrial(q, 'c10', 'train_case_01');
  has(panel(q).Transcript, /which is not in the run directory/, 'a transcript the run directory lacks is named as missing');
  disk.transcript = true;
  await q.click('#v-opt-runs [data-act="rescan"]');
  await tick(4);
  eq(q.text('#opt-cands .tr-count'), '3 spans of a recorded session', 'and is read once it is there, when the folder is read again with its trial still open');

  await q.openExperiment();
  await q.type('#form [data-k="input"]', 'edited');
  disk.stamp = 9;
  await go(q, 'candidates');
  await q.click('#opt-cands [data-open]');
  await tick(4);
  await q.openExperiment();
  await q.click('#save');
  await tick(4);
  eq(saved && saved.modified, 5,
    'reading the tuned-against experiment for a candidate\'s cases leaves the editor\'s save guard on the copy it loaded');
}

done();
