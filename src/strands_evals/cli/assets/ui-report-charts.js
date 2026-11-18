'use strict';

/* ── the report, drawn ──
   Built from ROWS and BY_CASE each time a report loads. A card with nothing to say hides itself. */

let repCost = null;                       // the cost measure charted, kept across re-renders

function verdictGrid() {
  const evaluators = [...new Set(ROWS.map(evaluatorName))];
  const tabbable = evaluators.length * BY_CASE.length <= TAB_MARKS;
  return heatmap(evaluators, BY_CASE.map(g => g.name), (e, c) => {
    const r = BY_CASE[c].rows.find(row => evaluatorName(row) === evaluators[e]);
    return r && { cls: verdictOf(r), tip: `${rowTitle(r)} · ${verdictText(r)}`, attrs: rowLink(r, tabbable) };
  });
}

function scoreSpread() {
  const by = scoresByEvaluator(ROWS);
  return by.size ? histRows([...by].map(([label, values]) => ({ label, values })), { axis: 'score' }) : '';
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
    text: repCost === 'seconds' ? secondsText(cost.seconds) : compact(cost.tokens),
    tip: `${g.name} · ${costText(cost)}`,
    attrs: rowLink(g.rows[0], true),
  })), { axis: repCost });
}

$('rep-cost-pick').onchange = e => { repCost = e.target.value; renderCaseCost(); };

function renderReportCharts() {
  $('rep-charts').hidden = !ROWS.length;
  if (!ROWS.length) return;
  $('rep-heat').innerHTML = verdictGrid();
  const spread = scoreSpread();
  $('rep-spread-card').hidden = !spread;
  $('rep-spread').innerHTML = spread;
  renderCaseCost();
}
