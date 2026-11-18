'use strict';

let REP = null, repMeta = null;
let ROWS = [], BY_CASE = [];
let repFilter = 'all', repQuery = '', openRows = new Set(), repLimit = 300;
/* `{ meta, doc, cand }` while Optimization borrows the report panel for candidate `cand`, with
   `meta` null when nothing was displaced; null when not borrowed. */
let repDisplaced = null;

/* ── the report panel, on loan ── */
function borrowReport(what) {
  $('rep-borrowed').hidden = false;
  $('rep-borrowed').innerHTML = `<span>${what}</span>`
    + '<button class="btn" data-act="unborrow">'
    + (repDisplaced && repDisplaced.meta ? 'Back to this run&rsquo;s report' : 'Back to Candidates')
    + '</button>';
}

function returnReport() {
  const had = repDisplaced;
  repDisplaced = null;
  $('rep-borrowed').hidden = true;
  $('rep-borrowed').innerHTML = '';
  if (!had || !had.meta) { clearReport(); return show('opt-candidates'); }
  loadReport(had.meta, had.doc);
  show('report');
}

app.querySelector('#rep-borrowed').onclick = e => {
  if (e.target.closest('[data-act="unborrow"]')) returnReport();
};

/* `keep` re-reads the same report: the filter, the search and the open rows stay as they are. */
function loadReport(meta, doc, keep) {
  repMeta = meta; REP = doc;
  if (!keep) {
    openRows = new Set(); repFilter = 'all'; repQuery = ''; repLimit = 300;
    $('rep-find').value = '';
    app.querySelectorAll('#rep-seg button').forEach(b => b.classList.toggle('on', b.dataset.f === 'all'));
    repOther = null;
  }
  buildRows(); renderHero(); renderReportCharts(); renderReport();
}

function clearReport() {
  REP = null; repMeta = null; ROWS = []; BY_CASE = []; repOther = null;
  renderReportCharts();
  const score = $('r-score');
  score.textContent = '—'; score.classList.add('none');
  $('r-stats').innerHTML = '';
  $('r-passbar').hidden = true;
  $('rep-count').textContent = '';
  $('rep-more').hidden = true;
  const exp = (expMeta && expMeta.name) || 'experiment.json';
  $('rep-rows').innerHTML = blank('No report in this run',
    `This experiment has not been run yet, or its report is somewhere else. To run it: `
    + `strands-evals run ${esc(exp)} --agent module:attr -o ${esc(stem(exp) ? stem(exp) + '.report.json' : 'report.json')}, `
    + `then Rescan. ${docLink('cli/run/#experiment-file-run', 'empty-report', 'How to run an experiment')}`);
}

/* ══════════ report ══════════ */

/** One row per case and evaluator of a report document. */
function reportRows(doc) {
  const rows = (doc.cases || []).map((c, i) => ({
    i, case: c, evaluator: c.evaluator,
    name: c.name != null ? String(c.name) : `#${i}`,
    score: (doc.scores || [])[i],
    pass: (doc.test_passes || [])[i],
    reason: (doc.reasons || [])[i] || '',
    // evaluate() returns a list of EvaluationOutput: a rubric judge has one per criterion.
    outputs: (doc.detailed_results || [])[i] || [],
    diagnosis: (doc.diagnoses || [])[i],
    recommendation: (doc.recommendations || [])[i],
  }));

  // Mirrors EvaluationReport.is_applicable: not applicable only when every output declined
  // to judge and passed. not_applicable with test_pass=false is a failure to judge.
  rows.forEach(r => {
    r.na = r.outputs.length > 0
      && r.outputs.every(o => o && o.label === 'not_applicable' && o.test_pass);
  });
  return rows;
}

function buildRows() {
  ROWS = reportRows(REP);
  BY_CASE = [];
  const index = new Map();
  ROWS.forEach(r => {
    const key = r.name;
    let g = index.get(key);
    if (!g) { g = { name: key, case: r.case, rows: [] }; index.set(key, g); BY_CASE.push(g); }
    g.rows.push(r);
  });
}

function renderHero() {
  const score = $('r-score');
  score.textContent = fmt(REP.overall_score); score.classList.remove('none');
  const applicable = ROWS.filter(r => !r.na);
  const passing = applicable.filter(r => r.pass).length;
  const bar = $('r-passbar');
  bar.hidden = !applicable.length;
  bar.title = `${passing} of ${plural(applicable.length, 'applicable row')} passing`;
  bar.firstElementChild.style.width = `${applicable.length ? (passing / applicable.length) * 100 : 0}%`;
  $('r-stats').innerHTML = [
    [`${passing} / ${applicable.length}`, 'Rows passing', 'data-tip="A row is one case scored by one evaluator. It passes when that evaluator says so."'],
    [BY_CASE.length, 'Cases', 'data-term="case"'],
    [new Set(ROWS.map(r => r.evaluator)).size, 'Evaluators', 'data-term="evaluator"'],
    [ROWS.filter(r => r.na).length, 'Not applicable', 'data-term="not_applicable"'],
  ].map(([v, k, tip]) => `<div class="stat" ${tip}><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');
}

const words = s => String(s == null ? '' : s).split(/\s+/);
function diff(expected, actual) {
  const a = words(expected), b = words(actual), out = [];
  const lcs = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  for (let x = a.length - 1; x >= 0; x--)
    for (let y = b.length - 1; y >= 0; y--)
      lcs[x][y] = a[x] === b[y] ? lcs[x + 1][y + 1] + 1 : Math.max(lcs[x + 1][y], lcs[x][y + 1]);
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { out.push(esc(a[i])); i++; j++; }
    else if (lcs[i + 1][j] >= lcs[i][j + 1]) out.push(`<span class="del"><s>${esc(a[i++])}</s></span>`);
    else out.push(`<span class="ins">${esc(b[j++])}</span>`);
  }
  while (i < a.length) out.push(`<span class="del"><s>${esc(a[i++])}</s></span>`);
  while (j < b.length) out.push(`<span class="ins">${esc(b[j++])}</span>`);
  return out.join(' ');
}

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
    const novel = reference && Array.isArray(reference) && !reference.some(x => sameStep(x, t));
    return novel ? `<span class="ins">${text}</span>` : text;
  }).join(' → ') + '</span>';
}

/* Keyed on what the case declares, never the evaluator's class name: the report does not
   record which field an evaluator read. */
const ASPECTS = [
  {
    has: c => c.expected_output != null || c.actual_output != null,
    rows: c => c.expected_output == null
      ? [['Actual output', esc(c.actual_output == null ? '—' : c.actual_output)]]
      : [['Expected output', esc(c.expected_output)],
         ['Actual output · diff', diff(c.expected_output, c.actual_output == null ? '' : c.actual_output)]],
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

const verdictTag = r => r.na
  ? '<span class="vb na" data-term="not_applicable"><i></i>not applicable</span>'
  : (r.pass ? '<span class="vb pass"><i></i>pass</span>' : '<span class="vb fail"><i></i>fail</span>');

function matches(g) {
  if (repFilter === 'fail' && g.rows.every(r => r.pass || r.na)) return false;
  if (repFilter === 'na' && !g.rows.some(r => r.na)) return false;
  if (!repQuery) return true;
  const hay = [g.name, g.case.input, g.case.expected_output, g.case.expected_assertion, g.case.actual_output]
    .concat(g.rows.map(r => [r.reason, r.evaluator]
      .concat(r.outputs.map(o => `${(o && o.label) || ''} ${(o && o.reason) || ''}`)).join(' ')))
    .join(' ').toLowerCase();
  return hay.includes(repQuery);
}

function renderReport() {
  if (!REP) return;
  const groups = BY_CASE.filter(matches);
  const total = groups.reduce((n, g) => n + g.rows.length, 0);
  const visible = groups.slice(0, repLimit);
  const hidden = groups.length - visible.length;

  $('rep-count').textContent = `${total} of ${plural(ROWS.length, 'row')}`;
  $('rep-more').hidden = hidden <= 0;
  $('rep-more').textContent = hidden > 0 ? `Show ${hidden} more case${hidden === 1 ? '' : 's'}` : '';

  $('rep-rows').innerHTML = visible.map(g => g.rows.map((r, k) => {
    const on = openRows.has(r.i);
    const head = `<tr class="case" data-i="${r.i}" tabindex="0" aria-expanded="${on}">
        <td>${k === 0 ? `<span class="caret${on ? ' open' : ''}"></span>${esc(g.name)}` : ''}</td>
        <td class="ev">${esc(r.evaluator == null ? '—' : r.evaluator)}</td>
        <td>${r.na ? '<span class="path">—</span>'
        : scoreBar(r.score)}</td>
        <td>${verdictTag(r)}</td></tr>`;
    if (!on) return head;

    const c = g.case;
    /* Optimizer trial records carry no input. */
    const kv = [['Input', c.input == null
      ? '<span class="empty">not recorded in this report</span>' : esc(c.input)]];
    if (c.expected_assertion) kv.push(['Assertion', esc(c.expected_assertion)]);
    ASPECTS.filter(a => a.has(c)).forEach(a => a.rows(c).forEach(row => kv.push(row)));
    if (!r.outputs.length) {
      kv.push(['Verdict', '<span class="empty">evaluator returned no outputs — '
        + '_default_aggregator scores that test_pass=false</span>']);
    }
    r.outputs.forEach((o, n) => {
      const tag = r.outputs.length > 1 ? ` ${n + 1}/${r.outputs.length}` : '';
      const label = o && o.label ? `<span class="mono">${esc(o.label)}</span> · ` : '';
      kv.push([`Verdict${tag}`, `${label}score ${fmt(o && o.score)} · `
        + (o && o.test_pass ? '<span class="vb pass"><i></i>pass</span>'
                            : '<span class="vb fail"><i></i>fail</span>')]);
      if (o && o.reason) kv.push([`Reason${tag}`, esc(o.reason)]);
      if (o && o.metadata && Object.keys(o.metadata).length) kv.push([`Metadata${tag}`, pre(o.metadata)]);
    });
    if (r.outputs.length > 1) kv.push(['Aggregated', `score ${fmt(r.score)} · ${esc(r.reason)}`]);
    if (r.diagnosis) kv.push(['Diagnosis', pre(r.diagnosis)]);
    if (r.recommendation) kv.push(['Recommendation', esc(r.recommendation)]);

    return head + '<tr class="detail"><td colspan="4"><dl class="kv">'
      + kv.map(([k2, v]) => `<dt>${k2}</dt>`
        + `<dd${/^(Expected|Actual|Metadata|Diagnosis)/.test(k2) ? ' class="m"' : ''}>${v}</dd>`).join('')
      + '</dl></td></tr>';
  }).join('')).join('')
    || '<tr><td colspan="4" class="empty">No rows match</td></tr>';
}

$('rep-find').oninput = e => { repQuery = e.target.value.trim().toLowerCase(); repLimit = 300; renderReport(); };
$('rep-seg').onclick = e => {
  const b = e.target.closest('button[data-f]');
  if (!b) return;
  repFilter = b.dataset.f; repLimit = 300;
  app.querySelectorAll('#rep-seg button').forEach(x => x.classList.toggle('on', x === b));
  renderReport();
};
$('rep-more').onclick = () => { repLimit += 300; renderReport(); };
/** Open one row of the table and bring it into view, clearing a filter that would hide it. */
function openReportRow(i) {
  repFilter = 'all'; repQuery = '';
  $('rep-find').value = '';
  app.querySelectorAll('#rep-seg button').forEach(b => b.classList.toggle('on', b.dataset.f === 'all'));
  const at = BY_CASE.findIndex(g => g.rows.some(r => r.i === i));
  repLimit = Math.max(repLimit, at + 1);
  openRows.add(i);
  renderReport();
  const row = $('rep-rows').querySelector(`tr.case[data-i="${i}"]`);
  if (row && row.scrollIntoView) row.scrollIntoView({ block: 'center' });
}

$('rep-rows').onclick = e => {
  const tr = e.target.closest('tr.case');
  if (!tr) return;
  const i = +tr.dataset.i;
  openRows.has(i) ? openRows.delete(i) : openRows.add(i);
  renderReport();
};
