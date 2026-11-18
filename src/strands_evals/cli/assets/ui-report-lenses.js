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

/* ── a diagnosis: what failed, why, where, and what to do ── */

const plainWords = s => String(s).replace(/[_-]+/g, ' ').toLowerCase();
/* The detector's taxonomy writes a category as `<parent>-category-<kind>`. */
const categoryText = s => plainWords(String(s).replace('-category-', ': '));
const arrayOf = v => Array.isArray(v) ? v : [];
const diagnosedTag = g => g.rows.some(r => r.diagnosis)
  ? ' <span class="tag" data-tip="A diagnosis of what went wrong is recorded for this case. Open a row to read it.">diagnosed</span>' : '';

/* A span the diagnosis names. When the row's recorded session is on the page it points at the card that draws it. */
function spanRef(id, spans) {
  const at = spans ? spans.findIndex(s => s && s.span_info && s.span_info.span_id === id) : -1;
  if (at < 0) return `<span class="mono">${esc(id == null ? 'an unnamed span' : id)}</span>`;
  const span = spans[at];
  const what = span.span_type === 'execute_tool' ? `tool call ${(span.tool_call || {}).name || ''}`.trim()
    : { inference: 'model call', invoke_agent: 'agent' }[span.span_type] || 'span';
  return `<button class="spanref" type="button" data-span-at="${at}" data-tip="Show this span in the trace of this row.">`
    + `#${at} ${esc(what)}</button>`;
}

const asWritten = v => `<pre class="val">${esc(typeof v === 'string' ? v : JSON.stringify(v, null, 2))}</pre>`;

function failureItem(f, spans) {
  if (!f || typeof f !== 'object') return `<li>${asWritten(f)}</li>`;
  const evidence = arrayOf(f.evidence), confidence = arrayOf(f.confidence);
  return `<li><div class="diag-at">at ${spanRef(f.span_id, spans)}</div>` + arrayOf(f.category).map((category, i) =>
    `<div class="diag-cat"><span class="tag warn">${esc(categoryText(category))}</span>`
    + (typeof confidence[i] === 'number'
      ? `<span class="path" data-tip="How sure the detector was of this failure.">${Math.round(confidence[i] * 100)}% sure</span>` : '')
    + (evidence[i] ? `<p>${esc(evidence[i])}</p>` : '') + '</div>').join('') + '</li>';
}

function causeItem(c, spans) {
  if (!c || typeof c !== 'object') return `<li>${asWritten(c)}</li>`;
  const facts = [
    arrayOf(c.propagation_impact).length ? `Effect: ${arrayOf(c.propagation_impact).map(plainWords).join(', ')}` : '',
    c.failure_detection_timing ? `Noticed: ${plainWords(c.failure_detection_timing)}` : '',
    c.completion_status ? `Outcome: ${plainWords(c.completion_status)}` : '',
  ].filter(Boolean).join(' · ');
  return '<li><div class="diag-at">'
    + (c.causality ? `<span class="tag">${esc(plainWords(c.causality))}</span> ` : '')
    + `began at ${spanRef(c.location, spans)}`
    + (c.failure_span_id !== c.location ? `, seen at ${spanRef(c.failure_span_id, spans)}` : '') + '</div>'
    + (c.root_cause_explanation ? `<p>${esc(c.root_cause_explanation)}</p>` : '')
    + (facts ? `<p class="path">${esc(facts)}</p>` : '') + '</li>';
}

/** A row's diagnosis and recommendation as markup, or '' when it has neither. */
function diagnosisView(r, c) {
  const d = r.diagnosis;
  if (d == null && !r.recommendation) return '';
  const raw = d == null ? '' : '<details class="raw"><summary>the diagnosis as written</summary>'
    + `<pre class="out">${esc(JSON.stringify(d, null, 1))}</pre></details>`;
  const shaped = d && typeof d === 'object' && (Array.isArray(d.failures) || Array.isArray(d.root_causes));
  if (d != null && !shaped) {
    return '<div class="diag"><h4>Diagnosis</h4><p class="empty">This is not in the shape a diagnosis is written in, '
      + `so it is shown as it was written.</p>${asWritten(d)}`
      + (r.recommendation ? `<h5>What to do</h5><p>${esc(r.recommendation)}</p>` : '') + '</div>';
  }
  const spans = sessionSpans(c.actual_trajectory);
  const failures = d ? arrayOf(d.failures) : [], causes = d ? arrayOf(d.root_causes) : [];
  /* The report's own recommendation is these fixes joined, so it is shown only where there are none. */
  const fixes = new Map();
  for (const cause of causes) {
    const fix = cause && typeof cause.fix_recommendation === 'string' ? cause.fix_recommendation.trim() : '';
    if (fix && !fixes.has(fix)) fixes.set(fix, cause.fix_type);
  }
  const todo = fixes.size
    ? `<ul>${[...fixes].map(([fix, type]) =>
      `<li>${type ? `<span class="tag ok">${esc(plainWords(type))}</span> ` : ''}${esc(fix)}</li>`).join('')}</ul>`
    : r.recommendation ? `<p>${esc(r.recommendation)}</p>` : '';
  return '<div class="diag"><h4>Diagnosis</h4>'
    + (d ? `<p class="path">${plural(failures.length, 'failure')} · ${plural(causes.length, 'root cause')}`
      + `${d.session_id ? ` · session ${esc(d.session_id)}` : ''}</p>` : '')
    + (failures.length ? `<h5>What failed</h5><ul>${failures.map(f => failureItem(f, spans)).join('')}</ul>` : '')
    + (causes.length ? `<h5>Why</h5><ul>${causes.map(cause => causeItem(cause, spans)).join('')}</ul>` : '')
    + (todo ? `<h5>What to do</h5>${todo}` : '')
    + raw + '</div>';
}

/* A span named by a diagnosis is found by the number its card carries, however the trace lays its cards out. */
$('rep-rows').addEventListener('click', e => {
  const ref = e.target.closest('.spanref');
  if (!ref) return;
  const detail = ref.closest('tr.detail');
  const label = [...detail.querySelectorAll('.trace .msg > .who > span')].find(s => s.textContent === `#${ref.dataset.spanAt}`);
  if (!label) return;
  detail.querySelectorAll('.msg.lit').forEach(card => card.classList.remove('lit'));
  const card = label.closest('.msg');
  card.classList.add('lit');
  if (card.scrollIntoView) card.scrollIntoView({ block: 'center' });
});
