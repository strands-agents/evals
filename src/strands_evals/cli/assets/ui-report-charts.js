'use strict';

/* ── the report, drawn ──
   Built from ROWS and BY_CASE each time a report loads. A card with nothing to say hides itself. */

/* `{ path, rows }` of the report this one is compared against, with `error` when it could not be read. */
let repOther = null;
let repCost = null;                       // the cost measure charted, kept across re-renders

const evaluatorName = r => r.evaluator == null ? '—' : String(r.evaluator);
/* A mark that opens its row in the table. Only a small chart puts every mark in the tab order. */
const rowLink = (r, focusable) => ` data-row="${r.i}"${focusable ? ' tabindex="0"' : ''}`;

function verdictGrid() {
  const evaluators = [...new Set(ROWS.map(evaluatorName))];
  const focusable = evaluators.length * BY_CASE.length <= 80;
  return heatmap(evaluators, BY_CASE.map(g => g.name), (e, c) => {
    const r = BY_CASE[c].rows.find(row => evaluatorName(row) === evaluators[e]);
    return r && {
      cls: r.na ? 'na' : r.pass ? 'pass' : 'fail',
      tip: `${r.name} · ${evaluators[e]} · `
        + (r.na ? 'not applicable' : `score ${fmt(r.score)} · ${r.pass ? 'pass' : 'fail'}`),
      attrs: rowLink(r, focusable),
    };
  });
}

function scoreSpread() {
  const by = new Map();
  for (const r of ROWS) {
    if (r.na || typeof r.score !== 'number') continue;
    by.set(evaluatorName(r), (by.get(evaluatorName(r)) || []).concat(r.score));
  }
  return by.size ? histRows([...by].map(([label, values]) => ({ label, values })), { axis: 'score', width: rowWidth(true) }) : '';
}

/* ── what each case took ── */

const COSTS = [['seconds', 'Seconds'], ['tokens', 'Tokens']];

/* An optimizer trial records its usage; a case run by `strands-evals run` has only its recorded session. */
const caseCost = c => c.usage && typeof c.usage === 'object'
  ? { seconds: c.usage.seconds, tokens: c.usage.tokens, modelCalls: c.usage.model_calls }
  : traceStats(c.actual_trajectory);

function renderCaseCost() {
  const costs = BY_CASE.map(g => ({ g, cost: caseCost(g.case) || {} }));
  const measures = COSTS.filter(([key]) => costs.some(x => x.cost[key] > 0));
  $('rep-cost-card').hidden = !measures.length;
  if (!measures.length) return;
  repCost = fillPick('rep-cost-pick', measures.map(([value, label]) => ({ value, label })), repCost);
  $('rep-cost-pick').hidden = measures.length < 2;
  const rows = costs.filter(x => x.cost[repCost] > 0).sort((a, b) => b.cost[repCost] - a.cost[repCost]);
  $('rep-cost-cap').textContent = 'What each case took, largest first'
    + (rows.length > CHART_ROWS ? `. The ${CHART_ROWS} largest of ${rows.length}.` : '.');
  $('rep-cost').innerHTML = barRows(rows.slice(0, CHART_ROWS).map(({ g, cost }) => ({
    label: g.name,
    value: cost[repCost],
    text: repCost === 'seconds' ? `${nice(cost.seconds)}s` : compact(cost.tokens),
    tip: `${g.name} · ${costText(cost)}`,
    attrs: rowLink(g.rows[0], true),
  })), { axis: repCost });
}

/* ── this report against another ── */

/** How the rows two reports share moved from `before` to `after`, matched by case and evaluator. */
function compareRows(before, after) {
  const key = r => `${r.name}\u0000${evaluatorName(r)}`;
  const passed = r => r.na ? null : !!r.pass;
  const was = new Map(before.map(r => [key(r), r]));
  const out = { fixed: 0, broke: 0, same: 0, added: 0, gone: 0, moved: [] };
  const seen = new Set();
  for (const r of after) {
    const old = was.get(key(r));
    seen.add(key(r));
    if (!old) { out.added += 1; continue; }
    const flip = passed(old) === false && passed(r) === true ? 'fixed'
      : passed(old) === true && passed(r) === false ? 'broke' : null;
    out[flip || 'same'] += 1;
    const from = old.na ? null : old.score, to = r.na ? null : r.score;
    if ((flip || from !== to) && (from != null || to != null)) out.moved.push({ row: r, from, to, flip });
  }
  out.gone = before.filter(r => !seen.has(key(r))).length;
  return out;
}

function renderReportCompare() {
  const others = RUNS.map(r => r.report).filter(m => m && (!repMeta || m.path !== repMeta.path));
  const set = (cap, chart, legend) => {
    $('rep-cmp-cap').textContent = cap;
    $('rep-cmp').innerHTML = chart || '';
    $('rep-cmp-legend').innerHTML = legend || '';
  };
  $('rep-cmp-pick').hidden = !others.length;
  $('rep-cmp-pick').innerHTML = '<option value="">Pick a report…</option>' + others.map(m =>
    `<option value="${esc(m.path)}"${repOther && repOther.path === m.path ? ' selected' : ''}>${esc(m.path)}</option>`).join('');
  if (!others.length) {
    return set('No other report in this folder. Run the experiment again with a different -o name and '
      + 'the two can be compared here.');
  }
  if (!repOther) return set('Pick an earlier report of the same cases to see which rows changed since.');
  if (repOther.error) return set(`Could not read ${repOther.path}: ${repOther.error}`);

  const c = compareRows(repOther.rows, ROWS);
  const moved = c.moved.sort((a, b) => !!b.flip - !!a.flip
    || Math.abs((b.to || 0) - (b.from || 0)) - Math.abs((a.to || 0) - (a.from || 0)));
  const several = new Set(ROWS.map(evaluatorName)).size > 1;
  const legend = [
    `<span><i class="up"></i>${c.fixed} now pass</span>`,
    `<span><i class="down"></i>${c.broke} now fail</span>`,
    `<span><i class="same"></i>${c.same} kept their verdict</span>`,
    c.added ? `<span>${c.added} only in this report</span>` : '',
    c.gone ? `<span>${c.gone} only in the other</span>` : '',
  ].join('');
  const shared = c.fixed + c.broke + c.same;
  set(`From ${repOther.path} to this report. ` + (!shared ? 'The two share no case and evaluator.'
    : !moved.length ? `All ${plural(shared, 'row')} they share scored the same.`
      : `${plural(moved.length, 'row')} changed score or verdict`
        + (moved.length > CHART_ROWS ? `; the ${CHART_ROWS} that moved most are drawn.` : '.')),
  moved.length ? dumbbells(moved.slice(0, CHART_ROWS).map(m => ({
    label: m.row.name,
    sub: several ? evaluatorName(m.row) : '',
    from: m.from, to: m.to,
    tip: `${m.row.name} · ${evaluatorName(m.row)} · ${fmt(m.from)} to ${fmt(m.to)}`
      + (m.flip ? ` · now ${m.flip === 'fixed' ? 'passes' : 'fails'}` : ''),
    attrs: rowLink(m.row, true),
  })), { axis: 'score', width: rowWidth(true) }) : '', shared ? legend : '');
}

$('rep-cmp-pick').onchange = async e => {
  const path = e.target.value, asked = repMeta;
  const meta = RUNS.map(r => r.report).find(m => m && m.path === path);
  let other = null;
  if (meta) {
    try { other = { path, rows: reportRows(await document_for(meta)) }; }
    catch (err) { other = { path, error: err.message }; }
  }
  if (repMeta !== asked) return;            // another report was opened while this one was read
  repOther = other;
  renderReportCompare();
};
$('rep-cost-pick').onchange = e => { repCost = e.target.value; renderCaseCost(); };

function renderReportCharts() {
  $('rep-charts').hidden = !ROWS.length;
  if (!ROWS.length) return;
  $('rep-heat').innerHTML = verdictGrid();
  const spread = scoreSpread();
  $('rep-spread-card').hidden = !spread;
  $('rep-spread').innerHTML = spread;
  renderCaseCost();
  renderReportCompare();
}

$('rep-charts').addEventListener('click', e => {
  const mark = e.target.closest('[data-row]');
  if (mark) openReportRow(+mark.dataset.row);
});
