'use strict';

/* ── Markdown, and the copy actions that hand it over ──
   A report, an experiment and a session are written by the server (`_ui_markdown.py`), which holds
   their files. What is written here is what the server has no document for: a comparison, an
   optimizer run's result and a candidate. */

const mdPresent = v => v != null && v !== '';
const mdLine = v => valueText(v).replace(/[ \t\r\n]+/g, ' ').replace(/^ | $/g, '');
const mdScore = v => (typeof v === 'number' && isFinite(v) ? v.toFixed(2) : 'none');
const mdCell = text => text.replace(/\\/g, '\\\\').replace(/\|/g, '\\|');
const mdDocument = parts => parts.join('\n\n') + '\n';

/* The fence is longer than any run of backticks in the text, so the text cannot close it. */
function mdFenced(text, lang) {
  const longest = (text.match(/`+/g) || []).reduce((n, run) => Math.max(n, run.length), 0);
  const ticks = '`'.repeat(Math.max(3, longest + 1));
  return `${ticks}${lang}\n${text}\n${ticks}`;
}
const mdFence = v => mdFenced(valueText(v), typeof v === 'string' ? 'text' : 'json');
const mdBlock = (label, v) => (mdPresent(v) ? [`${label}:`, mdFence(v)] : []);

/* ── what the server writes ── */

async function markdownFrom(path, opts) {
  const res = await fetch(path, opts);
  if (res.ok) return res.text();
  /* The server says why in `error`. An answer that is not its own has only a status. */
  const said = await res.json().catch(() => null);
  throw new Error((said && said.error) || `${res.status} ${res.statusText}`.trim());
}
/** The Markdown of the file at a page address. */
const fileMarkdown = route => markdownFrom(route + '.md', { headers: { 'X-Evals-Token': TOKEN } });
/** The Markdown of a document only the page holds, or of one part of a document: `/api/markdown` says what `body` is. */
const writtenMarkdown = body => markdownFrom('/api/markdown', {
  method: 'POST', headers: { 'X-Evals-Token': TOKEN, 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

/* ── where a file is, and how to run one case of it ── */

/* Absolute, so the path works from anywhere. */
const filePath = rel => (rel === '.' ? ROOT : `${ROOT.replace(/\/$/, '')}/${rel}`);
const shellQuote = s => (/^[\w./:=@%+-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\\''")}'`);
/* `--case` takes a name, or a 0-based index where no case has that name. `MODULE:ATTR` stands for the
   agent under test, which no file here records. */
const runCommand = (experiment, which) =>
  `strands-evals run ${shellQuote(filePath(experiment))} --case ${shellQuote(String(which))} --agent MODULE:ATTR`;

/* The whole experiment. The report is named so the scan pairs it back: the experiment's name, less `.json`
   and a trailing `experiment`, plus `.report.json`. Both paths are the file's own, so the command runs from
   anywhere and writes beside the experiment. */
function experimentRunCommand(experiment) {
  const base = experiment.replace(/^.*\//, '');
  const stem = base.replace(/\.json$/i, '').replace(/(^|[._-])(experiment|exp)$/i, '');
  const out = experiment.slice(0, experiment.length - base.length) + (stem ? `${stem}.report.json` : 'report.json');
  return `strands-evals run ${shellQuote(filePath(experiment))} --agent MODULE:ATTR -o ${shellQuote(filePath(out))}`;
}

const openExperimentOf = () => (repCand || !repMeta ? null : listed(REPORTS, repMeta).experiment);
/* How the open report and its experiment are named to a reader: a candidate's report is not a file. */
const reportNames = () => {
  const experiment = openExperimentOf();
  return { report: repCand ? repMeta.path : filePath(repMeta.path), experiment: experiment ? filePath(experiment) : null };
};
/* The open report with `rows` in full and every other row as its name and evaluator alone, marked passing.
   The rows kept are then named, numbered and counted as in the whole report, and the rest of it is not sent. */
function reportWith(rows) {
  const kept = new Set(rows.map(r => r.i));
  const of = (list, other = null) => REP.cases.map((_, i) => (kept.has(i) ? (list || [])[i] ?? null : other));
  return {
    cases: REP.cases.map((c, i) => (kept.has(i) ? c : { name: c.name, evaluator: c.evaluator })),
    scores: of(REP.scores), test_passes: of(REP.test_passes, true), reasons: of(REP.reasons),
    detailed_results: of(REP.detailed_results), diagnoses: of(REP.diagnoses), recommendations: of(REP.recommendations),
  };
}
/* The command that runs a failing case again: the first that has a name to run it by. */
function failingCommand() {
  const experiment = openExperimentOf();
  const named = ROWS.find(r => verdictOf(r) === 'fail' && mdPresent(r.case.name));
  return experiment && named ? runCommand(experiment, named.case.name) : null;
}

/* ── an optimizer run ── */

const mdFigure = v => (typeof v === 'number' && isFinite(v) ? fine(v) : 'none');
const mdChange = (before, after) => (typeof before === 'number' && typeof after === 'number' ? signed(after - before) : '');

/* Each changed setting as a diff of its lines, as `settingDiff` draws it. */
function mdSettings(settings) {
  const mark = { same: '  ', del: '- ', ins: '+ ' };
  return settings.filter(s => s.changed).flatMap(s => [`### ${s.key}`,
    mdFenced(diffSeq(s.before === undefined ? [] : valueText(s.before).split('\n'), valueText(s.after).split('\n'))
      .map(([kind, line]) => mark[kind] + line).join('\n'), 'diff')]);
}

const mdRunFacts = run => [`- Run: ${run.run_id}`, `- Directory: \`${filePath(run.dir || '.')}\``];

/** A candidate: what it measured beside the baseline, and what it changed from it. */
function candidateMarkdown(p, d, run) {
  const chain = d.lineageOf(p.candidate_id);
  const base = p.isBaseline ? null : d.proposals.find(x => x.candidate_id === d.baselineId)
    || d.points.find(x => x.candidate_id === d.baselineId);
  const facts = mdRunFacts(run).concat([`- Iteration: ${p.iteration}`]);
  if (p.outcome) facts.push(`- Outcome: ${p.outcome}`);
  if (p.isChampion || p.isBaseline) facts.push(`- Standing: ${p.isChampion ? 'champion' : 'baseline'}`);
  if (chain.length > 1) facts.push(`- Derived from: ${chain.join(' ← ')}`);
  const figure = (name, value, was) => `| ${mdCell(name)} | ${base ? `${mdFigure(was)} | ` : ''}${mdFigure(value)} |`
    + (base ? ` ${mdChange(was, value)} |` : '');
  const figures = [
    base ? `| Metric | ${mdCell(d.baselineId)} | ${mdCell(p.candidate_id)} | Change |` : `| Metric | ${mdCell(p.candidate_id)} |`,
    base ? '| --- | --- | --- | --- |' : '| --- | --- |',
    figure('Objective', p.objective, base && base.objective),
    ...Object.keys(p.measured || {}).map(m => figure(m, p.measured[m], base && (base.measured || {})[m])),
  ];
  const changed = p.isBaseline || !p.settings ? [] : mdSettings(p.settings);
  return mdDocument([`# Candidate ${p.candidate_id}`, facts.join('\n'),
    ...mdBlock('Why it was tried', p.rationale), ...mdBlock(`Why ${p.outcome}`, outcomeWhy(p)),
    base ? `## Figures against ${d.baselineId}` : '## Figures', figures.join('\n'),
    ...(p.isBaseline ? ['## Configuration', 'This is the configuration the run started from.', ...mdBlock('Values', p.settings && Object.fromEntries(p.settings.map(s => [s.key, s.after])))]
      : [`## Changes against ${d.baselineId || 'the baseline'}`,
        ...(changed.length ? changed : [p.settings ? 'None: it is the same in every setting the log records.'
          : 'The log records no values for this candidate.'])]),
  ]);
}

/** The result of a run: the champion, what it gained on the baseline, and its configuration. */
function resultMarkdown(d, run) {
  const r = d.result, w = d.winner;
  const status = run.state === 'finished' && d.stop ? `finished: ${stopPhrase(d.stop)}` : runState(run);
  const facts = mdRunFacts(run).concat([`- Status: ${status}`,
    `- Champion: ${d.champion || 'not picked yet'}`]);
  if (w && w.lineage.length > 1) facts.push(`- Derived from: ${w.lineage.join(' ← ')}`);
  if (r && r.baselineId) {
    facts.push(`- Compared with: ${r.baselineId}, the baseline, on `
      + (r.heldOut ? plural(r.cases || 0, 'held-out case') : 'the validation split the search chose on'));
  }
  const o = r && r.objective;
  if (o && typeof o.mean_diff === 'number') {
    facts.push(`- Objective change: ${signed(o.mean_diff)}`
      + (typeof o.ci_low === 'number' && typeof o.ci_high === 'number' ? `, 95% interval [${fine(o.ci_low)}, ${fine(o.ci_high)}]` : ''));
  }
  const parts = [`# Optimizer run ${run.run_id}`, facts.join('\n')];
  if (r && r.metrics.length) {
    parts.push('## Metrics', ['| Metric | Baseline | Champion | Change |', '| --- | --- | --- | --- |',
      ...r.metrics.map(m => `| ${mdCell(m.metric)} | ${mdFigure(m.before)} | ${mdFigure(m.after)} | ${mdChange(m.before, m.after)} |`),
    ].join('\n'));
  }
  if (w && w.settings) {
    const changed = mdSettings(w.settings);
    parts.push(`## Changes against ${w.baselineId || 'the baseline'}`,
      ...(changed.length ? changed : ['None: the run ended with the configuration it started from.']),
      '## Configuration', mdFence(w.values));
  }
  return mdDocument(parts);
}

/* ── the buttons ── */

const mdButton = (kind, label, arg, tip) =>
  `<button class="btn" data-md="${kind}"${arg == null ? '' : ` data-arg="${esc(arg)}"`}${tip ? ` data-tip="${esc(tip)}"` : ''}>${label}</button>`;

/* The actions in the page header, for what the view has open. */
function pageActions(view) {
  const run = OPT_RUNS[optSel];
  if (view === 'report' && REP) {
    return mdButton('report', 'Copy as Markdown')
      + (ROWS.some(r => verdictOf(r) === 'fail') ? mdButton('failing', 'Copy failing rows as a prompt') : '')
      + (repCand ? '' : mdButton('path', 'Copy path', repMeta.path));
  }
  if (view === 'experiment' && EXP) {
    /* An experiment with no report offers the command that would make one. */
    return (reportsOf(listed(EXPERIMENTS, expMeta)).length ? '' : mdButton('run-command', 'Copy run command', null,
      'The command that runs this experiment and writes its report beside it. Fill in --agent with your agent\'s module:attr.'))
      + mdButton('experiment', 'Copy as Markdown') + mdButton('path', 'Copy path', expMeta.path);
  }
  if (view === 'session' && SES) return mdButton('session', 'Copy as Markdown') + mdButton('path', 'Copy path', sesMeta.path);
  if (SECTION_OF[view] === 'optimization' && view !== 'opt-runs' && run) {
    return (OPT ? mdButton('result', 'Copy result as Markdown') : '') + mdButton('path', 'Copy path', run.dir || '.');
  }
  return '';
}

/* Beside an opened report row. The command needs the experiment the report was made from. */
function rowActions(r) {
  const experiment = openExperimentOf();
  return '<div class="mdacts">' + mdButton('row', 'Copy as Markdown', r.i)
    + (repCand ? '' : mdButton('row-link', 'Copy link', r.i))
    + (experiment && mdPresent(r.case.name) ? mdButton('row-command', 'Copy command', r.i) : '') + '</div>';
}

const candidateActions = p => (p.candidate_id ? '<div class="mdacts">'
  + mdButton('candidate', 'Copy as Markdown', p.candidate_id) + mdButton('candidate-link', 'Copy link', p.candidate_id)
  + '</div>' : '');

/* The traces drawn so far, so a button can name the one it copies by its place here.
   The open session has none: its page header copies it whole. */
const TRACE_DOCS = [];
function traceCopy(doc) {
  if (doc === SES) return '';
  let n = TRACE_DOCS.indexOf(doc);
  if (n < 0) n = TRACE_DOCS.push(doc) - 1;
  return `<button class="btn tr-copy" data-md="trace" data-arg="${n}">Copy as Markdown</button>`;
}

const candidateOf = id => OPT && OPT.proposals.find(p => p.candidate_id === id);

/* What each button copies, as text or a promise of it, or nothing when what it names is no longer open.
   A file is read at its address. What is not one (a candidate's report, edits not yet saved) and a part
   of one (a row, a case, a trace) is sent to be written. */
const MD_COPY = {
  report: () => REP && (repCand ? writtenMarkdown({ kind: 'report', path: repMeta.path, document: REP })
    : fileMarkdown(`/reports${idPath(repMeta.path)}`)),
  failing: () => REP && writtenMarkdown({ kind: 'failing', document: reportWith(ROWS.filter(r => verdictOf(r) === 'fail')),
    ...reportNames(), command: failingCommand() }),
  path: rel => filePath(rel),
  row: i => ROWS[i] && writtenMarkdown({ kind: 'row', row: ROWS[i].i, document: reportWith([ROWS[i]]), ...reportNames() }),
  'row-link': i => ROWS[i] && location.origin + `/reports${idPath(repMeta.path)}${rowPath(ROWS[i])}`,
  'row-command': i => ROWS[i] && runCommand(openExperimentOf(), ROWS[i].case.name),
  session: () => SES && fileMarkdown(`/sessions${idPath(sesMeta.path)}`),
  experiment: () => EXP && (countDirty()
    ? writtenMarkdown({ kind: 'experiment', path: expMeta.path, document: experimentDocument(), reports: listed(EXPERIMENTS, expMeta).reports || [] })
    : fileMarkdown(`/experiments${idPath(expMeta.path)}`)),
  case: () => cases[sel] && writtenMarkdown({ kind: 'case', document: cases[sel], index: sel, experiment: filePath(expMeta.path) }),
  'case-command': () => cases[sel] && runCommand(expMeta.path, mdPresent(cases[sel].name) ? cases[sel].name : sel),
  'run-command': () => expMeta && experimentRunCommand(expMeta.path),
  candidate: id => candidateOf(id) && candidateMarkdown(candidateOf(id), OPT, OPT_RUNS[optSel]),
  'candidate-link': id => location.origin + `/optimization${idPath(OPT_RUNS[optSel].id)}/candidates/${encodeURIComponent(id)}`,
  result: () => OPT && resultMarkdown(OPT, OPT_RUNS[optSel]),
  trace: n => TRACE_DOCS[n] && writtenMarkdown({ kind: 'trace', document: TRACE_DOCS[n], cost: costText(traceStats(TRACE_DOCS[n])) }),
};

/* Counts the copies begun: text that arrives after a later copy was asked for is not the one wanted. */
let copying = 0;

/* The button says what happened, and why when the text could not be written, then goes back to what it does. */
app.addEventListener('click', async e => {
  const b = e.target.closest('[data-md]');
  if (!b) return;
  const text = MD_COPY[b.dataset.md](b.dataset.arg);
  if (!text) return;
  const mine = ++copying;
  let why = '';
  const wanted = Promise.resolve(text).then(now => (mine === copying ? now : Promise.reject(new Error('overtaken'))),
    err => { why = err.message; throw err; });
  if (!b.dataset.label) b.dataset.label = b.textContent;
  const copied = await copyText(wanted);
  if (mine !== copying) return;
  b.textContent = copied ? 'Copied' : why ? `Could not copy: ${why}` : 'Could not copy';
  b.classList.toggle('refused', !!why);
  setTimeout(() => { b.textContent = b.dataset.label; b.classList.remove('refused'); }, why ? 6000 : 1600);
});
