'use strict';

/* ── a diagnosis: what failed, why, where, and what to do ──
   Everything that draws one goes through `diagnosisView`: an opened report row and a session's page. */

const plainWords = s => String(s).replace(/[_-]+/g, ' ').toLowerCase();
/* The detector's taxonomy writes a category as `<parent>-category-<kind>`. */
const categoryText = s => plainWords(String(s).replace('-category-', ': '));
/* A span the diagnosis names. When its session is drawn on the page it points at the card that draws it. */
function spanRef(id, spans) {
  const at = spans ? spans.findIndex(s => s && s.span_info && s.span_info.span_id === id) : -1;
  if (at < 0) return `<span class="mono">${esc(id == null ? 'an unnamed span' : id)}</span>`;
  const span = spans[at];
  const what = span.span_type === 'execute_tool' ? `tool call ${(span.tool_call || {}).name || ''}`.trim()
    : { inference: 'model call', invoke_agent: 'agent' }[span.span_type] || 'span';
  return `<button class="spanref" type="button" data-span-at="${at}" data-tip="Show this span in the trace beside this diagnosis.">`
    + `#${at} ${esc(what)}</button>`;
}

const asWritten = v => `<pre class="val">${esc(typeof v === 'string' ? v : JSON.stringify(v, null, 2))}</pre>`;

function failureItem(f, spans) {
  if (!f || typeof f !== 'object') return `<li>${asWritten(f)}</li>`;
  const evidence = arrayOf(f.evidence), confidence = arrayOf(f.confidence);
  return `<li><div class="diag-at">at ${spanRef(f.span_id, spans)}</div>` + arrayOf(f.category).map((category, i) =>
    `<div class="diag-cat"><span class="tag bad">${esc(categoryText(category))}</span>`
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

/** A diagnosis as markup: what failed, why, and what to do. `d` is a `DiagnosisResult` or either half
    of one. `spans` are those of the session drawn beside it, so a span it names can point at its card.
    `recommendation` is a report row's own. '' when there is neither a diagnosis nor a recommendation. */
function diagnosisView(d, { spans = null, recommendation = '' } = {}) {
  if (d == null && !recommendation) return '';
  const shaped = d && typeof d === 'object' && (Array.isArray(d.failures) || Array.isArray(d.root_causes));
  if (d != null && !shaped) {
    return '<div class="diag"><h4>Diagnosis</h4><p class="empty">This is not in the shape a diagnosis is written in, '
      + `so it is shown as it was written.</p>${asWritten(d)}`
      + (recommendation ? `<h5>What to do</h5><p>${esc(recommendation)}</p>` : '') + '</div>';
  }
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
    : recommendation ? `<p>${esc(recommendation)}</p>` : '';
  return '<div class="diag"><h4>Diagnosis</h4>'
    + (d ? `<p class="path">${plural(failures.length, 'failure')} · ${plural(causes.length, 'root cause')}`
      + `${d.session_id ? ` · session ${esc(d.session_id)}` : ''}</p>` : '')
    + (d && !failures.length && !causes.length ? '<p class="empty">No failures were found.</p>' : '')
    + (failures.length ? `<h5>What failed</h5><ul>${failures.map(f => failureItem(f, spans)).join('')}</ul>` : '')
    + (causes.length ? `<h5>Why</h5><ul>${causes.map(cause => causeItem(cause, spans)).join('')}</ul>` : '')
    + (todo ? `<h5>What to do</h5>${todo}` : '')
    + (d ? rawView(d, 'the diagnosis as written') : '') + '</div>';
}

/* A span named by a diagnosis is found by the number its card carries, however the trace lays its cards out. */
app.addEventListener('click', e => {
  const ref = e.target.closest('.spanref');
  if (!ref) return;
  const detail = ref.closest('tr.detail, .view');
  const label = [...detail.querySelectorAll('.trace .msg > .who > span')].find(s => s.textContent === `#${ref.dataset.spanAt}`);
  if (!label) return;
  detail.querySelectorAll('.msg.lit').forEach(card => card.classList.remove('lit'));
  const card = label.closest('.msg');
  card.classList.add('lit');
  if (card.scrollIntoView) card.scrollIntoView({ block: 'center' });
});
