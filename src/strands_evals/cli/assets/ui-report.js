'use strict';

let REP = null, repMeta = null;
let ROWS = [], BY_CASE = [];
const REP_PAGE = 300;                     // the table draws this many cases, then as many again when asked
let repFilter = 'all', repQuery = '', openRows = new Set(), repLimit = REP_PAGE;
/* The candidate whose cases the report page shows in place of a report file, or null. */
let repCand = null;

function setFilter(f) {
  repFilter = f;
  app.querySelectorAll('#rep-seg button').forEach(b => b.classList.toggle('on', b.dataset.f === f));
}

/* Nothing left that keeps a row out of the table: no filter, no search, no one group shown. */
function showEveryRow() {
  setFilter('all');
  repQuery = ''; $('rep-find').value = '';
  repGroupOnly = null;
}

/* `keep` re-reads the same report: the filter, the search, the grouping and the open rows stay as they are. */
function loadReport(meta, doc, keep) {
  repMeta = meta; REP = doc;
  if (!keep) {
    openRows = new Set(); repLimit = REP_PAGE; repGroup = '';
    showEveryRow();
  }
  buildRows(); renderHero(); renderReportCharts(); renderLenses(); renderReport();
}

function clearReport() {
  REP = null; repMeta = null; repCand = null; ROWS = []; BY_CASE = [];
  renderReportCharts(); renderLenses();
  const score = $('r-score');
  score.textContent = '—'; score.classList.add('none');
  $('r-stats').innerHTML = '';
  $('r-passbar').hidden = true;
  $('rep-count').textContent = '';
  $('rep-more').hidden = true;
  $('rep-rows').innerHTML = '';
}

/* ══════════ a report's rows, and what every view of them reads them with ══════════ */

/** One row per case and evaluator of a report document. A case with no name is numbered by its place
    among the rows of its evaluator, so it has one name under each. */
function reportRows(doc) {
  const place = new Map();
  return (doc.cases || []).map((c, i) => {
    /* evaluate() returns a list of EvaluationOutput: a rubric judge has one per criterion. */
    const outputs = arrayOf((doc.detailed_results || [])[i]);
    const at = place.get(c.evaluator) || 0;
    place.set(c.evaluator, at + 1);
    return {
      i, case: c, evaluator: c.evaluator,
      name: c.name != null ? String(c.name) : `#${at}`,
      score: (doc.scores || [])[i],
      pass: (doc.test_passes || [])[i],
      reason: (doc.reasons || [])[i] || '',
      outputs,
      /* Mirrors EvaluationReport.is_applicable: not applicable only when every output declined
         to judge and passed. not_applicable with test_pass=false is a failure to judge. */
      na: outputs.length > 0 && outputs.every(o => o && o.label === 'not_applicable' && o.test_pass),
      diagnosis: (doc.diagnoses || [])[i],
      recommendation: (doc.recommendations || [])[i],
    };
  });
}

const evaluatorName = r => r.evaluator == null ? '—' : String(r.evaluator);
/** A row as a chart mark names it. */
const rowTitle = r => `${r.name} · ${evaluatorName(r)}`;
/** `pass`, `fail`, or `na` where the evaluator declined to judge. */
const verdictOf = r => (r.na ? 'na' : r.pass ? 'pass' : 'fail');
const verdictText = r => (r.na ? 'not applicable' : `score ${fmt(r.score)} · ${r.pass ? 'pass' : 'fail'}`);
/** How many of `rows` came to each verdict: `{ pass, fail, na }`. */
function verdictCounts(rows) {
  const n = { pass: 0, fail: 0, na: 0 };
  for (const r of rows) n[verdictOf(r)] += 1;
  return n;
}
/* A mean is taken over the rows an evaluator judged and gave a number. */
const scored = r => !r.na && typeof r.score === 'number';
/** Each evaluator's scores, by its name. An evaluator that scored no row is left out. */
function scoresByEvaluator(rows) {
  const by = new Map();
  for (const r of rows.filter(scored)) by.set(evaluatorName(r), (by.get(evaluatorName(r)) || []).concat(r.score));
  return by;
}

const TAB_MARKS = 80;                     // a chart of more marks than this keeps them out of the tab order
/** What makes a chart mark open a row of the table. */
const rowLink = (r, tabbable) => ` data-row="${r.i}"${tabbable ? ' tabindex="0"' : ''}`;

/* ══════════ the page ══════════ */

function buildRows() {
  ROWS = reportRows(REP);
  const byName = new Map();
  for (const r of ROWS) {
    if (!byName.has(r.name)) byName.set(r.name, { name: r.name, case: r.case, rows: [] });
    byName.get(r.name).rows.push(r);
  }
  BY_CASE = [...byName.values()];
}

function renderHero() {
  const score = $('r-score');
  score.textContent = fmt(REP.overall_score); score.classList.remove('none');
  const n = verdictCounts(ROWS), applicable = n.pass + n.fail;
  const bar = $('r-passbar');
  bar.hidden = !applicable;
  bar.title = `${n.pass} of ${plural(applicable, 'applicable row')} passing`;
  bar.firstElementChild.style.width = `${applicable ? (n.pass / applicable) * 100 : 0}%`;
  $('r-stats').innerHTML = [
    [`${n.pass} / ${applicable}`, 'Rows passing', 'data-tip="A row is one case scored by one evaluator. It passes when that evaluator says so."'],
    [BY_CASE.length, 'Cases', 'data-term="case"'],
    [new Set(ROWS.map(evaluatorName)).size, 'Evaluators', 'data-term="evaluator"'],
    [n.na, 'Not applicable', 'data-term="not_applicable"'],
  ].map(([v, k, tip]) => `<div class="stat" ${tip}><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');
}

const words = s => String(s == null ? '' : s).split(/\s+/);
const diff = (expected, actual) => diffSeq(words(expected), words(actual)).map(([kind, word]) =>
  kind === 'same' ? esc(word)
    : kind === 'del' ? `<span class="del"><s>${esc(word)}</s></span>` : `<span class="ins">${esc(word)}</span>`).join(' ');

/* A report trajectory is `list[Any] | Session`, and a Session serializes to an object. */
function trajectory(v, reference) {
  if (v == null) return '<span class="empty">none captured</span>';
  if (!Array.isArray(v)) {
    return sessionSpans(v) ? `<div class="trace">${traceView(v, 'the session as recorded')}</div>`
      : '<span class="path">recorded session, not a step list</span><br>' + json(v);
  }
  if (!v.length) return '<span class="empty">empty list</span>';
  return '<span class="mono">' + v.map(t => {
    const text = esc(stepText(t));
    const novel = Array.isArray(reference) && !reference.some(x => sameStep(x, t));
    return novel ? `<span class="ins">${text}</span>` : text;
  }).join(' → ') + '</span>';
}

/* Keyed on what the case declares, never the evaluator's class name: the report does not
   record which field an evaluator read. */
const ASPECTS = [
  {
    has: c => c.expected_output != null || c.actual_output != null,
    rows: c => {
      const actual = c.actual_output == null ? '—' : showValue(c.actual_output);
      if (c.expected_output == null) return [['Actual output', actual]];
      /* Words can be compared only where both sides are text. */
      const worded = typeof c.expected_output === 'string' && (c.actual_output == null || typeof c.actual_output === 'string');
      return [['Expected output', showValue(c.expected_output)], worded
        ? ['Actual output · diff', diff(c.expected_output, c.actual_output == null ? '' : c.actual_output)]
        : ['Actual output', actual]];
    },
  },
  {
    has: c => c.expected_trajectory != null || c.actual_trajectory != null,
    rows: c => [['Expected trajectory', trajectory(c.expected_trajectory)],
                ['Actual trajectory', trajectory(c.actual_trajectory, c.expected_trajectory)]],
  },
  {
    has: c => c.expected_interactions != null || c.actual_interactions != null,
    rows: c => [['Expected interactions', pre(c.expected_interactions)],
                ['Actual interactions', pre(c.actual_interactions)]],
  },
  {
    has: c => c.expected_environment_state != null || c.actual_environment_state != null,
    rows: c => [['Expected env state', pre(c.expected_environment_state)],
                ['Actual env state', pre(c.actual_environment_state)]],
  },
];

const passTag = pass => `<span class="vb ${pass ? 'pass' : 'fail'}"><i></i>${pass ? 'pass' : 'fail'}</span>`;
const verdictTag = r => (r.na
  ? '<span class="vb na" data-term="not_applicable"><i></i>not applicable</span>' : passTag(r.pass));

function matches(g) {
  if (repFilter === 'fail' && !g.rows.some(r => verdictOf(r) === 'fail')) return false;
  if (repFilter === 'na' && !g.rows.some(r => r.na)) return false;
  if (repFilter === 'diag' && !g.rows.some(r => r.diagnosis)) return false;
  if (repGroupOnly != null && groupOf(g) !== repGroupOnly) return false;
  if (!repQuery) return true;
  const hay = [g.name, g.case.input, g.case.expected_output, g.case.expected_assertion, g.case.actual_output].map(valueText)
    .concat(g.rows.map(r => [r.reason, r.evaluator]
      .concat(r.outputs.map(o => `${(o && o.label) || ''} ${(o && o.reason) || ''}`)).join(' ')))
    .join(' ').toLowerCase();
  return hay.includes(repQuery);
}

/* The table's line for one row. The case is named on the first of its rows. */
function rowLine(g, r, first) {
  const on = openRows.has(r.i);
  return `<tr class="case" data-i="${r.i}" tabindex="0" aria-expanded="${on}">
        <td>${first ? `<span class="caret${on ? ' open' : ''}"></span>${esc(g.name)}${diagnosedTag(g)}` : ''}</td>
        <td class="ev">${esc(evaluatorName(r))}</td>
        <td>${r.na ? '<span class="path">—</span>' : scoreBar(r.score)}</td>
        <td>${verdictTag(r)}</td></tr>`;
}

/* An opened row, in two columns: what the case asked and got back, then what this evaluator made of it. */
function rowDetail(c, r) {
  /* Optimizer trial records carry no input. */
  const asked = [['Input', c.input == null
    ? '<span class="empty">not recorded in this report</span>' : showValue(c.input)]];
  if (c.expected_assertion) asked.push(['Assertion', esc(c.expected_assertion)]);
  ASPECTS.filter(a => a.has(c)).forEach(a => asked.push(...a.rows(c)));

  const several = r.outputs.length > 1;
  const judged = [];
  if (!r.outputs.length) {
    judged.push(['Verdict', '<span class="empty">evaluator returned no outputs, which '
      + '_default_aggregator scores as test_pass=false</span>']);
  }
  r.outputs.forEach((o, n) => {
    const tag = several ? ` ${n + 1}/${r.outputs.length}` : '';
    /* One output is the row's own score and verdict, which head the card already. */
    if (several) {
      judged.push([`Verdict${tag}`, `${o && o.label ? `<span class="mono">${esc(o.label)}</span> · ` : ''}`
        + `score ${fmt(o && o.score)} · ${passTag(o && o.test_pass)}`]);
    }
    if (o && o.reason) judged.push([`Reason${tag}`, esc(o.reason)]);
    if (o && o.metadata && Object.keys(o.metadata).length) judged.push([`Metadata${tag}`, pre(o.metadata)]);
  });
  if (several) judged.push(['Aggregated', `score ${fmt(r.score)} · ${esc(r.reason)}`]);

  const only = !r.na && r.outputs.length === 1 && r.outputs[0] ? r.outputs[0].label : null;
  const blocks = rows => rows.map(([k, v]) => `<dt>${k}</dt>`
    + `<dd${/^(Expected|Actual|Metadata)/.test(k) ? ' class="m"' : ''}>${v}</dd>`).join('');
  return '<tr class="detail"><td colspan="4"><div class="rowd">'
    + `<dl class="blocks">${blocks(asked)}</dl>`
    + `<dl class="blocks judge ${verdictOf(r)}">`
    + `<dt class="judge-by">${esc(r.evaluator == null ? 'Evaluator' : r.evaluator)}</dt>`
    + `<dd class="judge-score">${r.na ? '' : scoreBar(r.score)}${verdictTag(r)}`
    + `${only && only !== r.evaluator ? `<span class="mono path">${esc(only)}</span>` : ''}</dd>`
    + `${blocks(judged)}</dl>`
    + diagnosisView(r.diagnosis, { spans: sessionSpans(c.actual_trajectory), recommendation: r.recommendation })
    + `</div>${rowActions(r)}</td></tr>`;
}

function renderReport() {
  if (!REP) return;
  const groups = inGroupOrder(BY_CASE.filter(matches));
  const visible = groups.slice(0, repLimit);
  const hidden = groups.length - visible.length;

  $('rep-count').textContent = `${groups.reduce((n, g) => n + g.rows.length, 0)} of ${plural(ROWS.length, 'row')}`;
  $('rep-more').hidden = !hidden;
  $('rep-more').textContent = hidden ? `Show ${plural(hidden, 'more case')}` : '';

  redraw($('rep-rows'), visible.map((g, at) => groupHead(visible, at) + g.rows.map((r, k) =>
    rowLine(g, r, k === 0) + (openRows.has(r.i) ? rowDetail(g.case, r) : '')).join('')).join('')
    /* Nothing matching is always the filter, the search or one group: offer the way back out of all three. */
    || (ROWS.length ? '<tr><td colspan="4" class="empty">No rows match'
      + ' <button class="btn" data-act="rep-all">Show every row</button></td></tr>'
      : `<tr><td colspan="4">${emptyState('Not a ripple in this report',
        'The file holds no rows: the run that wrote it had no case to evaluate.')}</td></tr>`));
}

$('rep-find').oninput = e => { repQuery = e.target.value.trim().toLowerCase(); repLimit = REP_PAGE; renderReport(); };
$('rep-seg').onclick = e => {
  const b = e.target.closest('button[data-f]');
  if (!b) return;
  setFilter(b.dataset.f); repLimit = REP_PAGE;
  renderReport();
};
$('rep-more').onclick = () => { repLimit += REP_PAGE; renderReport(); };

/** Open one row of the table and bring it into view, clearing whatever would hide it. The address then names it. */
function openReportRow(i) {
  showEveryRow();
  renderLenses();
  const at = inGroupOrder(BY_CASE).findIndex(g => g.rows.some(r => r.i === i));
  repLimit = Math.max(repLimit, at + 1);
  /* The address names the row added last, so one that is open already is added again. */
  openRows.delete(i);
  openRows.add(i);
  renderReport();
  writeAddress(currentView(), true);
  const row = $('rep-rows').querySelector(`tr.case[data-i="${i}"]`);
  if (row && row.scrollIntoView) row.scrollIntoView({ block: 'center' });
}

app.addEventListener('click', e => {
  const mark = e.target.closest('[data-row]');
  if (!mark) return;
  if (currentView() === 'report') openReportRow(+mark.dataset.row);
});

$('rep-rows').onclick = e => {
  if (e.target.closest('[data-act="rep-all"]')) {
    showEveryRow();
    renderLenses();
    renderReport();
    return;
  }
  const tr = e.target.closest('tr.case');
  if (!tr) return;
  const i = +tr.dataset.i;
  openRows.has(i) ? openRows.delete(i) : openRows.add(i);
  renderReport();
  writeAddress('report', true);
};
