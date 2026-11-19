'use strict';

/* ── Markdown, and the copy actions that hand it over ──
   What is open, written to paste into a conversation with a coding agent. The server writes a
   report, an experiment and a session the same way (`_ui_markdown.py`), for a reader with no
   browser, and `tests/ui/markdown.mjs` holds the two to the same text. */

const mdPresent = v => v != null && v !== '';
const mdLine = v => valueText(v).replace(/[ \t\r\n]+/g, ' ').replace(/^ | $/g, '');
const mdScore = v => (typeof v === 'number' && isFinite(v) ? v.toFixed(2) : 'none');
const mdCell = text => text.replace(/\\/g, '\\\\').replace(/\|/g, '\\|');
const mdName = (v, i) => (mdPresent(v) ? mdLine(v) : `#${i}`);
const mdDocument = parts => parts.join('\n\n') + '\n';

/* The fence is longer than any run of backticks in the text, so the text cannot close it. */
function mdFenced(text, lang) {
  const longest = (text.match(/`+/g) || []).reduce((n, run) => Math.max(n, run.length), 0);
  const ticks = '`'.repeat(Math.max(3, longest + 1));
  return `${ticks}${lang}\n${text}\n${ticks}`;
}
const mdFence = v => mdFenced(valueText(v), typeof v === 'string' ? 'text' : 'json');
const mdBlock = (label, v) => (mdPresent(v) ? [`${label}:`, mdFence(v)] : []);

function mdTrajectory(label, v) {
  const spans = sessionSpans(v);
  return spans ? [`${label}: a recorded session of ${plural(spans.length, 'span')}, kept in the report file.`]
    : mdBlock(label, v);
}

const MD_ASPECTS = [['output', 'output'], ['trajectory', 'trajectory'], ['interactions', 'interactions'],
  ['environment_state', 'environment state']];

/* What a case asked for and, in a report, what came back. */
function mdCaseBlocks(c, sides) {
  const parts = [...mdBlock('Input', c.input), ...mdBlock('Assertion', c.expected_assertion)];
  for (const [field, words] of MD_ASPECTS) {
    for (const side of sides) {
      const label = `${side[0].toUpperCase()}${side.slice(1)} ${words}`;
      parts.push(...(field === 'trajectory' ? mdTrajectory : mdBlock)(label, c[`${side}_${field}`]));
    }
  }
  return parts;
}

/* ── a report ── */

const mdVerdict = r => ({ pass: 'passed', fail: 'failed', na: 'not applicable' }[verdictOf(r)]);
const mdEvaluator = r => (mdPresent(r.evaluator) ? mdLine(r.evaluator) : '(unnamed)');
/* Each row's case name. A case with no name is numbered by its place among the rows of its evaluator,
   so it has one name under each. */
function mdCaseNames(rows) {
  const place = new Map();
  return new Map(rows.map(r => {
    const at = place.get(mdEvaluator(r)) || 0;
    place.set(mdEvaluator(r), at + 1);
    return [r, mdName(r.case.name, at)];
  }));
}

/* `names` is `mdCaseNames` of the rows `r` is one of. `more` is facts about where the row is from,
   for a row copied on its own. */
function mdRowParts(r, names, heading, more = []) {
  const outputs = r.outputs.map(o => (o && typeof o === 'object' && !Array.isArray(o) ? o : {}));
  const only = outputs.length === 1 ? outputs[0] : {};
  const several = outputs.length > 1 ? outputs : [];
  const facts = [`- Score: ${mdScore(r.score)}`, `- Verdict: ${mdVerdict(r)}`];
  if (typeof only.label === 'string' && only.label) facts.push(`- Label: ${mdLine(only.label)}`);
  several.forEach((o, n) => facts.push(`- Output ${n + 1} of ${outputs.length}: score ${mdScore(o.score)}, `
    + (o.test_pass ? 'passed' : 'failed') + (typeof o.label === 'string' && o.label ? `, ${mdLine(o.label)}` : '')));
  return [`${heading} ${names.get(r)} · ${mdEvaluator(r)}`, facts.concat(more).join('\n'),
    ...mdBlock('Reason', mdPresent(r.reason) ? r.reason : only.reason),
    ...several.flatMap((o, n) => mdBlock(`Reason, output ${n + 1} of ${outputs.length}`, o.reason)),
    ...mdBlock('Diagnosis', r.diagnosis), ...mdBlock('Recommendation', r.recommendation),
    ...mdCaseBlocks(r.case, ['expected', 'actual'])];
}

/* The experiment a report was paired with and how well it fits, or why it has none, as the scan found it. */
const mdMadeFrom = (experiment, link) => (experiment ? experiment + (link ? ` (${link.status}: ${link.detail})` : '')
  : link ? `not paired: ${link.detail}` : 'none found in this folder; a report made in code has none');

/** A report: its figures, each failing row in full, then every row in one table. `experiment` and
    `link` are the scan's pairing of it. */
function reportMarkdown(path, doc, experiment, link) {
  const rows = reportRows(doc), n = verdictCounts(rows), names = mdCaseNames(rows);
  const facts = [
    `- Overall score: ${mdScore(doc.overall_score)}`,
    `- Rows: ${rows.length} (${n.pass} passed, ${n.fail} failed, ${n.na} not applicable)`,
    `- Cases: ${new Set(names.values()).size}`,
    `- Evaluators: ${[...new Set(rows.map(mdEvaluator))].join(', ') || 'none'}`,
    `- Experiment: ${mdMadeFrom(experiment, link)}`,
  ];
  const failing = rows.filter(r => verdictOf(r) === 'fail').flatMap(r => mdRowParts(r, names, '###'));
  const table = ['| Case | Evaluator | Score | Verdict |', '| --- | --- | --- | --- |',
    ...rows.map(r => `| ${mdCell(names.get(r))} | ${mdCell(mdEvaluator(r))} | ${mdScore(r.score)} | ${mdVerdict(r)} |`)];
  return mdDocument([`# Report: ${path}`, facts.join('\n'), '## Failing rows',
    ...(failing.length ? failing : ['None.']), '## All rows', table.join('\n')]);
}

/* ── an experiment ── */

function mdEvaluatorParts(e, i) {
  const { evaluator_type: kind, name, ...settings } = e;
  const title = mdName(mdPresent(name) ? name : kind, i)
    + (mdPresent(name) && mdPresent(kind) ? ` (${mdLine(kind)})` : '');
  return [`### ${title}`, ...(Object.keys(settings).length ? mdBlock('Settings', settings) : [])];
}

const mdCaseParts = (c, i, heading) =>
  [`${heading} ${mdName(c.name, i)}`, ...mdCaseBlocks(c, ['expected']), ...mdBlock('Metadata', c.metadata)];

/** An experiment: its evaluators with their settings, then every case. */
function experimentMarkdown(path, doc, reports) {
  const given = arrayOf(doc.cases).map(c => (isObject(c) ? c : {}));
  const evaluators = arrayOf(doc.evaluators).filter(isObject);
  const facts = [
    `- Cases: ${given.length}`,
    `- Evaluators: ${evaluators.length}`,
    `- Reports: ${[...reports].sort().join(', ') || 'none found in this folder'}`,
  ];
  const judged = evaluators.flatMap(mdEvaluatorParts);
  const entries = given.flatMap((c, i) => mdCaseParts(c, i, '###'));
  return mdDocument([`# Experiment: ${path}`, facts.join('\n'), '## Evaluators',
    ...(judged.length ? judged : ['None.']), '## Cases', ...(entries.length ? entries : ['None.'])]);
}

/* ── where a file is, and how to run one case of it ── */

/* Absolute where the page knows the folder it is served from, so the path works from anywhere. */
const filePath = rel => (MODE !== 'server' || !ROOT ? rel : rel === '.' ? ROOT : `${ROOT.replace(/\/$/, '')}/${rel}`);
const shellQuote = s => (/^[\w./:=@%+-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\\''")}'`);
/* `--case` takes a name, or a 0-based index where no case has that name. */
const runCommand = (experiment, which) =>
  `strands-evals run ${shellQuote(filePath(experiment))} --case ${shellQuote(String(which))} --agent module:attr`;

const addressOf = route => (MODE === 'server' ? location.origin + route : `${location.href.split('#')[0]}#${route}`);

const openExperimentOf = () => (repCand || !repMeta ? null : listed(REPORTS, repMeta).experiment);
const reportPath = () => (repCand ? repMeta.path : filePath(repMeta.path));

function rowMarkdown(r) {
  const experiment = openExperimentOf();
  return mdDocument(mdRowParts(r, mdCaseNames(ROWS), '#', [
    `- Report: \`${reportPath()}\``,
    ...(experiment ? [`- Experiment: \`${filePath(experiment)}\``] : []),
  ]));
}

/** Every failing row of the open report, under one line saying what to do with them. */
function failingPrompt() {
  const rows = ROWS.filter(r => verdictOf(r) === 'fail'), names = mdCaseNames(ROWS);
  const experiment = openExperimentOf();
  const named = rows.find(r => mdPresent(r.case.name));
  const facts = experiment ? [`- Experiment: \`${filePath(experiment)}\``] : [];
  if (experiment && named) facts.push(`- To run one case again: \`${runCommand(experiment, named.case.name)}\``);
  return mdDocument([
    `${rows.length === 1 ? 'This row fails' : `These ${rows.length} rows fail`} in the evaluation report `
      + `\`${reportPath()}\`. Find the cause of each failure and fix it.`,
    ...(facts.length ? [facts.join('\n')] : []),
    ...rows.flatMap(r => mdRowParts(r, names, '##')),
  ]);
}

function caseMarkdown(c, i) {
  const parts = mdCaseParts(c, i, '#');
  return mdDocument([parts[0], `- Experiment: \`${filePath(expMeta.path)}\``, ...parts.slice(1)]);
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
  const facts = mdRunFacts(run).concat([`- Status: ${d.stop ? `stopped: ${d.stop.reason}` : 'still running'}`,
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

/* ── a trace ── */

/* What a session names is written outside a fence, where a line break could start a heading. */
const mdFlat = v => String(v).replace(/[ \t\r\n]+/g, ' ').replace(/^ | $/g, '');
const mdCall = (name, input) =>
  [`Tool call \`${name ? mdFlat(name) : '?'}\`:`, mdFenced(JSON.stringify(input === undefined ? null : input, null, 2), 'json')];
const mdResult = (text, status) => [`Tool result${status ? ` (${mdFlat(status)})` : ''}:`, mdFenced(text, 'text')];

/* The same two block vocabularies `contentBlock` reads, in full rather than clipped. */
function mdContent(b) {
  if (b == null || typeof b !== 'object') return [mdFenced(b == null ? '' : String(b), 'text')];
  if (typeof b.text === 'string') return [mdFence(b.text)];
  if (b.toolUse) return mdCall(b.toolUse.name, b.toolUse.input);
  if (b.content_type === 'tool_use') return mdCall(b.name, b.arguments);
  if (b.toolResult) return mdResult(...strandsResult(b.toolResult));
  if (b.content_type === 'tool_result') return mdResult(...recordedResult(b));
  if (b.reasoningContent) return ['Reasoning:', mdFenced(((b.reasoningContent.reasoningText || {}).text) || '', 'text')];
  return [mdFence(b)];
}

function mdMessage(m, heading) {
  const usage = usageLine(((m && m.metadata) || {}).usage);
  return [`${heading} ${m && m.role ? mdFlat(m.role) : 'unknown'}`, ...(usage ? [usage] : []), ...messageBlocks(m).flatMap(mdContent)];
}

/* The heading carries the span's `span_id`, which is what a diagnosis names it by. */
function mdSpan(s, i, h = '##') {
  const kind = (s && s.span_type) || 'span';
  const id = isObject(s) && isObject(s.span_info) ? s.span_info.span_id : null;
  const head = name => `${h} Span ${i}: ${name}${mdPresent(id) ? ` (span_id: ${mdLine(id)})` : ''}`;
  if (kind === 'inference' && Array.isArray(s.messages)) return [head('inference'), ...s.messages.flatMap(m => mdMessage(m, `${h}#`))];
  if (kind === 'execute_tool') {
    const call = s.tool_call || {};
    return [head('tool'), ...mdCall(call.name, call.arguments), ...mdResult(...recordedResult(s.tool_result || {}))];
  }
  if (kind === 'invoke_agent') {
    return [head('invoke agent'), ...mdBlock('System prompt', s.system_prompt), ...mdBlock('Prompt', s.user_prompt),
      ...mdBlock('Response', s.agent_response)];
  }
  return [head(mdFlat(kind)), mdFence(s)];
}

/** A recorded session or a list of messages as a conversation, tool calls and results included. */
function traceMarkdown(doc) {
  const cost = costText(traceStats(doc));
  const spans = sessionSpans(doc);
  if (spans) {
    return mdDocument([`# Trace: ${plural(spans.length, 'span')} of a recorded session`, ...(cost ? [cost] : []),
      ...spans.flatMap((s, i) => mdSpan(s, i))]);
  }
  if (Array.isArray(doc)) {
    return mdDocument([`# Trace: ${plural(doc.length, 'message')}`, ...(cost ? [cost] : []),
      ...doc.flatMap((m, i) => mdMessage(m, `## Message ${i + 1}:`))]);
  }
  return mdDocument(['# Trace', mdFence(doc)]);
}

/* ── a session, and a diagnosis whose session is not here ── */

/** A session: what it holds, each diagnosis made of it, then every trace span by span. A span keeps
    the number its card has on the page. `found` is the diagnoses, as `{ meta, doc }`. */
function sessionMarkdown(path, doc, seconds, found) {
  const traces = sessionTraces(doc);
  const facts = [`- Session: ${mdLine(doc.session_id)}`, `- Traces: ${traces.length}`,
    `- Spans: ${sessionSpans(doc).length}`];
  if (seconds != null) facts.push(`- Time: ${secondsText(seconds)}`);
  facts.push(`- Diagnoses: ${found.map(f => f.meta.path).join(', ') || 'none found'}`);
  let first = 0;
  return mdDocument([`# Session: ${path}`, facts.join('\n'),
    ...found.flatMap(f => [`## Diagnosis: ${f.meta.path}`, mdFence(f.doc)]),
    ...traces.flatMap((tr, at) => {
      const spans = tr.spans.flatMap((s, i) => mdSpan(s, first + i, '###'));
      first += tr.spans.length;
      return [`## Trace ${at + 1} of ${traces.length}${mdPresent(tr.id) ? `: ${mdLine(tr.id)}` : ''}`,
        ...(spans.length ? spans : ['No spans.'])];
    })]);
}

function diagnosisMarkdown(path, doc) {
  return mdDocument([`# Diagnosis: ${path}`, [
    `- Session: ${mdPresent(doc.session_id) ? `${mdLine(doc.session_id)}, which is not in the folder` : 'not named'}`,
    `- Failures: ${arrayOf(doc.failures).length}`, `- Root causes: ${arrayOf(doc.root_causes).length}`,
  ].join('\n'), mdFence(doc)]);
}

/* ── a comparison ── */

/** The open comparison: each report, each evaluator's mean in each, then what changed from the oldest to the newest. */
function compareMarkdown() {
  const no = CMP.map((_, i) => cmpNo(i)), last = CMP.length - 1;
  const table = (columns, rows) => [`| ${columns.join(' | ')} |`, `|${' --- |'.repeat(columns.length)}`, ...rows].join('\n');
  const reports = table(['#', 'Report', 'When', 'Overall score', 'Rows', 'Passing', 'Failing'], CMP.map((c, i) => {
    const n = verdictCounts(c.rows);
    return `| ${no[i]} | ${mdCell(c.meta.path)} | ${new Date(c.at).toISOString()}${c.dated ? '' : ', by file time'} | `
      + `${mdScore(c.meta.overall_score)} | ${c.rows.length} | ${n.pass} | ${n.fail} |`;
  }));
  const { names, means } = evaluatorMeans();
  const evaluators = table(['Evaluator', ...no], names.map(name =>
    `| ${mdCell(mdLine(name))} | ${means.map(m => (m.has(name) ? mdScore(m.get(name)) : 'no row')).join(' | ')} |`));
  const c = compareEndRows();
  /* A row that was not applicable at one end has no verdict to have kept. */
  const verdict = m => (m.flip ? `now ${m.flip === 'fixed' ? 'passes' : 'fails'}`
    : m.from == null || m.to == null ? mdVerdict(m.row) : `still ${m.row.pass ? 'passes' : 'fails'}`);
  const moved = table(['Case', 'Evaluator', no[0], no[last], 'Verdict'], c.moved.map(m =>
    `| ${mdCell(mdLine(m.row.name))} | ${mdCell(mdLine(evaluatorName(m.row)))} | ${mdScore(m.from)} | ${mdScore(m.to)} | ${verdict(m)} |`));
  return mdDocument([`# Comparison of ${plural(CMP.length, 'report')}`,
    'Oldest first. Rows are matched by case and evaluator, so reports of different experiments compare only where they hold the same rows.',
    reports, '## Evaluator means', names.length ? evaluators : 'No row has a score.',
    `## From ${no[0]} to ${no[last]}`,
    [`- Now pass: ${c.fixed}`, `- Now fail: ${c.broke}`, `- Kept their verdict: ${c.same}`,
      `- Only in ${no[last]}: ${c.added}`, `- Only in ${no[0]}: ${c.gone}`].join('\n'),
    c.moved.length ? moved : c.fixed + c.broke + c.same ? 'Every row they share scored the same.' : 'The two share no case and evaluator.']);
}

/* ── the buttons ── */

const mdButton = (kind, label, arg) =>
  `<button class="btn" data-md="${kind}"${arg == null ? '' : ` data-arg="${esc(arg)}"`}>${label}</button>`;

/* The actions in the page header, for what the view has open. */
function pageActions(view) {
  const run = OPT_RUNS[optSel];
  if (view === 'report' && REP) {
    return mdButton('report', 'Copy as Markdown')
      + (ROWS.some(r => verdictOf(r) === 'fail') ? mdButton('failing', 'Copy failing rows as a prompt') : '')
      + (repCand ? '' : mdButton('path', 'Copy path', repMeta.path));
  }
  if (view === 'compare' && CMP.length) return mdButton('compare', 'Copy as Markdown');
  if (view === 'experiment' && EXP) return mdButton('experiment', 'Copy as Markdown') + mdButton('path', 'Copy path', expMeta.path);
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

/* What each button copies, or nothing when what it names is no longer open. */
const MD_COPY = {
  report: () => REP && reportMarkdown(repMeta.path, REP, openExperimentOf(), repCand ? null : listed(REPORTS, repMeta).link),
  failing: () => REP && failingPrompt(),
  path: rel => filePath(rel),
  row: i => ROWS[i] && rowMarkdown(ROWS[i]),
  'row-link': i => ROWS[i] && addressOf(`/reports${idPath(repMeta.path)}${rowPath(ROWS[i])}`),
  'row-command': i => ROWS[i] && runCommand(openExperimentOf(), ROWS[i].case.name),
  compare: () => CMP.length && compareMarkdown(),
  session: () => SES && (isSession(sesMeta) ? sessionMarkdown(sesMeta.path, SES, listed(SESSIONS, sesMeta).seconds, sesFound)
    : diagnosisMarkdown(sesMeta.path, SES)),
  experiment: () => EXP && experimentMarkdown(expMeta.path, experimentDocument(), listed(EXPERIMENTS, expMeta).reports || []),
  case: () => cases[sel] && caseMarkdown(cases[sel], sel),
  'case-command': () => cases[sel] && runCommand(expMeta.path, mdPresent(cases[sel].name) ? cases[sel].name : sel),
  candidate: id => candidateOf(id) && candidateMarkdown(candidateOf(id), OPT, OPT_RUNS[optSel]),
  'candidate-link': id => addressOf(`/optimization${idPath(OPT_RUNS[optSel].id)}/candidates/${encodeURIComponent(id)}`),
  result: () => OPT && resultMarkdown(OPT, OPT_RUNS[optSel]),
  trace: n => TRACE_DOCS[n] && traceMarkdown(TRACE_DOCS[n]),
};

/* The button says what happened, then goes back to what it does. */
app.addEventListener('click', async e => {
  const b = e.target.closest('[data-md]');
  if (!b) return;
  const text = MD_COPY[b.dataset.md](b.dataset.arg);
  if (!text) return;
  if (!b.dataset.label) b.dataset.label = b.textContent;
  b.textContent = await copyText(text) ? 'Copied' : 'Could not copy';
  setTimeout(() => { b.textContent = b.dataset.label; }, 1600);
});
