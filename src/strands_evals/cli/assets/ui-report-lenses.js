'use strict';

/* ── two lenses on a report ──
   Its rows grouped by something every case carries, and a row's diagnosis read as one. */

/* What rows are grouped by: `m:<metadata key>`, `n:condition`, or '' for none. Then the one group shown, or null. */
let repGroup = '', repGroupOnly = null;

const NOT_SET = '(not set)';
/* A chaos experiment names each case `<case>|<condition>` and records the condition nowhere else. */
const conditionOf = name => name.lastIndexOf('|') < 0 ? null : name.slice(name.lastIndexOf('|') + 1);
/* A list of plain values reads as its items; anything deeper is grouped by its JSON. */
const unset = v => v == null || v === '';
const groupText = v => unset(v) ? NOT_SET
  : Array.isArray(v) && v.length && v.every(x => x == null || typeof x !== 'object') ? v.join(', ')
    : typeof v === 'object' ? JSON.stringify(v) : String(v);
const metadataOf = c => c.metadata && typeof c.metadata === 'object' && !Array.isArray(c.metadata) ? c.metadata : {};

/** The group a case falls in under the chosen grouping. */
const groupOf = g => groupText(repGroup === 'n:condition' ? conditionOf(g.name) : metadataOf(g.case)[repGroup.slice(2)]);
const groupLabel = () => repGroup === 'n:condition' ? 'condition' : repGroup.slice(2);

/** Every metadata key some case has a value for, and the condition where case names hold one. */
function groupChoices() {
  const keys = new Set(BY_CASE.flatMap(g => Object.entries(metadataOf(g.case)).filter(([, v]) => !unset(v)).map(([key]) => key)));
  const choices = [{ value: '', label: 'None' }].concat([...keys].sort().map(key => ({ value: 'm:' + key, label: key })));
  if (BY_CASE.some(g => conditionOf(g.name) != null)) choices.push({ value: 'n:condition', label: 'condition, from the name' });
  return choices;
}

/** Per group, in the order its first case appears: what it holds and how it scored. */
function groupSummary() {
  const by = new Map();
  for (const g of BY_CASE) {
    const value = groupOf(g);
    const s = by.get(value) || { value, cases: 0, rows: 0, passing: 0, failing: 0, scores: [] };
    by.set(value, s);
    s.cases += 1;
    s.rows += g.rows.length;
    for (const r of g.rows.filter(row => !row.na)) {
      s[r.pass ? 'passing' : 'failing'] += 1;
      if (typeof r.score === 'number') s.scores.push(r.score);
    }
  }
  return [...by.values()];
}

/** Cases put next to the others of their group, the groups in the order the summary lists them. */
function inGroupOrder(list) {
  if (!repGroup) return list;
  const order = new Map(groupSummary().map((s, i) => [s.value, i]));
  return list.slice().sort((a, b) => order.get(groupOf(a)) - order.get(groupOf(b)));
}

/** The row that names a group in the table, before the first case of each. */
function groupHead(list, at) {
  if (!repGroup || (at > 0 && groupOf(list[at - 1]) === groupOf(list[at]))) return '';
  return `<tr class="grouphead"><td colspan="4"><span class="path">${esc(groupLabel())}</span> ${esc(groupOf(list[at]))}</td></tr>`;
}

function renderLenses() {
  const diagnosed = ROWS.some(r => r.diagnosis);
  app.querySelector('#rep-seg [data-f="diag"]').hidden = !diagnosed;
  /* A report written again with no diagnosis in it no longer offers the filter that was on. */
  if (!diagnosed && repFilter === 'diag') setFilter('all');

  const choices = REP ? groupChoices() : [];
  $('rep-lens').hidden = choices.length < 2;
  if (choices.length < 2) { repGroup = ''; repGroupOnly = null; return; }
  repGroup = fillPick('rep-group-pick', choices, repGroup);
  const groups = repGroup ? groupSummary() : [];
  if (!groups.some(s => s.value === repGroupOnly)) repGroupOnly = null;
  $('rep-group-cap').textContent = !repGroup ? 'Group the rows by something the cases carry, to see how one slice of them did.'
    : repGroup === 'n:condition'
      ? 'Read from the part of each case name after "|", which is how a chaos experiment names its conditions. '
        + 'Select a group to see only its rows.'
      : `By the value of ${groupLabel()} in each case's metadata. Select a group to see only its rows.`;
  $('rep-groups').innerHTML = !repGroup ? '' : '<div class="scroll"><table class="groups"><thead><tr>'
    + `<th>${esc(groupLabel())}</th><th>Cases</th><th>Rows</th><th>Passing</th><th>Failing</th><th>Mean score</th>`
    + '</tr></thead><tbody>' + groups.map(s => {
    const judged = s.passing + s.failing;
    const on = s.value === repGroupOnly;
    return `<tr class="group${on ? ' on' : ''}" data-group="${esc(s.value)}" tabindex="0" aria-pressed="${on}">`
      + `<td>${esc(s.value)}</td><td>${s.cases}</td><td>${s.rows}</td>`
      + `<td>${judged ? `${s.passing} of ${judged}` : '<span class="path">—</span>'}</td>`
      + `<td>${s.failing ? `<span class="vb fail"><i></i>${s.failing}</span>` : '<span class="path">0</span>'}</td>`
      + `<td>${s.scores.length ? scoreBar(s.scores.reduce((a, b) => a + b, 0) / s.scores.length) : '<span class="path">—</span>'}</td></tr>`;
  }).join('') + '</tbody></table></div>';
}

$('rep-group-pick').onchange = e => {
  repGroup = e.target.value; repGroupOnly = null; repLimit = 300;
  renderLenses(); renderReport();
};
$('rep-groups').onclick = e => {
  const row = e.target.closest('tr.group');
  if (!row) return;
  repGroupOnly = repGroupOnly === row.dataset.group ? null : row.dataset.group;
  repLimit = 300;
  renderLenses(); renderReport();
};

/* A case whose rows carry a diagnosis is marked in the table; `diagnosisView` draws it in the opened row. */
const diagnosedTag = g => g.rows.some(r => r.diagnosis)
  ? ' <span class="tag" data-tip="A diagnosis of what went wrong is recorded for this case. Open a row to read it.">diagnosed</span>' : '';
