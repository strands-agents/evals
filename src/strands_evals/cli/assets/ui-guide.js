'use strict';

/* Guidance and outbound links. What the page tells a user about a view, a field, a setting or a
   term is written here once, beside the documentation page that says more. */

const DOCS = 'https://strandsagents.com/docs/user-guide/evals-sdk/';
/* Campaign tags in the form the Strands site's analytics reads. `utm_content` names the placement. */
const CAMPAIGN = 'utm_source=strands-evals-ui&utm_medium=product&utm_campaign=evals-ui';

/** A documentation URL. `path` may end in `#anchor`; `placement` says which link on the page it is. */
function docUrl(path, placement) {
  const [page, anchor] = path.split('#');
  return `${DOCS}${page}?${CAMPAIGN}&utm_content=${encodeURIComponent(placement)}${anchor ? '#' + anchor : ''}`;
}
const docLink = (path, placement, label) =>
  `<a class="doc" href="${esc(docUrl(path, placement))}" target="_blank" rel="noopener">${esc(label)}</a>`;

/* ══════════ what things mean ══════════ */

const TERMS = {
  experiment: ['Experiment', 'What you run: a list of cases and the evaluators applied to each one. Running it produces a report.', 'how-evaluation-works/#what-an-evaluation-is-made-of'],
  case: ['Case', 'One test: the input sent to your agent, plus optional expectations that evaluators compare against.', 'how-evaluation-works/#what-an-evaluation-is-made-of'],
  evaluator: ['Evaluator', 'Scores one case and returns a score, a pass or fail, and its reasoning. Most use a model as a judge; some are plain code.', 'evaluators/#choose-an-evaluator'],
  report: ['Report', 'The result of running an experiment: a score, a verdict and a reason for every case and evaluator.', 'cli/report/'],
  score: ['Score', 'The number an evaluator gives a case, usually from 0 to 1. The overall score is the mean over the rows that applied.', 'how-evaluation-works/#the-levels-an-evaluator-scores-at'],
  verdict: ['Verdict', 'Pass or fail as the evaluator itself decided. This page applies no threshold of its own.', 'evaluators/#choose-an-evaluator'],
  not_applicable: ['Not applicable', 'The evaluator had nothing to judge on this case, for example no expected output. These rows are left out of the overall score.', 'evaluators/#choose-an-evaluator'],
  trajectory: ['Trajectory', 'What the agent did along the way: a list such as the tools it called, or a Session holding the full trace.', 'how-evaluation-works/#what-an-evaluation-is-made-of'],
  pairing: ['Pairing', 'A report is matched to an experiment beside it with the same name, or else to the nearest one holding all its cases. Verified means every case in the report is in that experiment.', 'how-to/serialization/'],
  overall: ['Overall score', 'The mean score over every row that applied. Rows an evaluator marked not applicable are left out.', 'how-evaluation-works/#the-levels-an-evaluator-scores-at'],
  draft: ['Draft', 'A case whose metadata says status: draft. Review drafts steps through them so each can be accepted or dropped.', 'how-to/experiment_management/#using-metadata-for-organization'],
  session: ['Session', 'Everything recorded for one conversation with an agent: one or more traces, saved as a file.', 'how-evaluation-works/#what-an-evaluation-is-made-of'],
  trace: ['Trace', 'One run of the agent inside a session, made of spans: each agent invocation, model call and tool call.', 'how-evaluation-works/#what-an-evaluation-is-made-of'],
  diagnosis: ['Diagnosis', 'What the detectors found in a session: the spans where something failed, and the root cause behind each.', 'detectors/'],
  iteration: ['Iteration', 'One round of the search: candidates are proposed, run on the training cases, and ranked.', ''],
  constraint: ['Constraint', 'A limit a candidate has to stay within, such as a token budget. One that does not is infeasible.', ''],
  budget: ['Budget', 'The most rollouts or iterations the run was allowed to spend before it had to stop.', ''],
  rollout: ['Rollout', 'One run of the agent on one case. Rollouts are what the budget counts.', ''],
  candidate: ['Candidate', 'One configuration the optimizer built and ran your cases against.', ''],
  baseline: ['Baseline', 'The configuration the run started from. Every result is read against it.', ''],
  champion: ['Champion', 'The configuration the run picked. It is the baseline when nothing feasible beat it.', ''],
  feasible: ['Feasible', 'Measured on every training case and meeting every constraint.', ''],
  infeasible: ['Infeasible', 'A constraint was not met. It can still be a parent of later candidates.', ''],
  ineligible: ['Ineligible', 'Some training case was not measured, so it could not be ranked.', ''],
  invalid: ['Invalid', 'The proposed change failed validation, so no candidate was built.', ''],
  unscored: ['Unscored', 'The budget could not cover it.', ''],
  duplicate: ['Duplicate', "Its values equal another candidate's, so it took that candidate's objective.", ''],
  split: ['Train, validation, test', 'The search scores candidates on train, picks the champion once on validation, and measures test only to report the result.', ''],
  objective: ['Objective', 'The expression the run maximises or minimises. Goals named in it decide which direction is better.', ''],
  interval: ['95% interval', 'The range the true change is likely to lie in. If it includes zero, the run has not shown a change.', ''],
};

/* Case fields: what each is for, what it accepts, and where the docs cover it. */
const FIELD_GUIDE = {
  name: ['A label for the case. Results are listed under it.', 'Text. Keep it unique: two cases with one name cannot be told apart in a report.', 'how-to/experiment_management/#naming-conventions'],
  input: ['What your task receives: the prompt or request sent to the agent.', 'Required. Text, or whatever your task accepts.', 'how-evaluation-works/#what-an-evaluation-is-made-of'],
  expected_output: ['The answer you expect. Evaluators that compare outputs read it.', 'Text. Leave it empty and an output evaluator reports not applicable.', 'evaluators/output_evaluator/'],
  expected_assertion: ['A plain-language statement of what success looks like, for a judge to check.', 'One or more sentences. Used by assertion-based evaluators such as GoalSuccessRateEvaluator.', 'evaluators/goal_success_rate_evaluator/'],
  expected_trajectory: ['The steps you expect the agent to take, such as the tools it calls, in order.', 'A list of tool names, or recorded calls. Leave it empty and a trajectory evaluator reports not applicable.', 'evaluators/trajectory_evaluator/'],
  expected_interactions: ['The expected exchange between agents or components, for multi-agent systems.', 'A JSON list with one entry per interaction.', 'evaluators/interactions_evaluator/'],
  expected_environment_state: ['The state you expect the environment to be in afterwards.', 'A JSON list of named states, checked by StateEquals.', 'evaluators/deterministic_evaluators/#environment-state-evaluators'],
  metadata: ['Free-form keys that travel with the case, for filtering and grouping.', 'Any keys. A value that reads as JSON is stored as that type; quote it to keep it text. status: draft marks a case for the review pass.', 'how-to/experiment_management/#using-metadata-for-organization'],
};

/* Settings that several evaluators share. A setting not listed here is explained on the evaluator's own page. */
const SETTING_GUIDE = {
  name: 'The label this evaluator carries in the report. Needed when the same evaluator appears twice.',
  rubric: 'The criteria the judge scores against, in plain language.',
  model: 'The model that judges. Empty means the default Bedrock model.',
  system_prompt: "Instructions for the judge. Empty uses the evaluator's own prompt.",
  include_inputs: "Whether the judge sees the case's input as well as the output.",
  version: "Which version of the evaluator's built-in prompt to use.",
  value: 'The value the output is checked against.',
  case_sensitive: 'Whether letter case has to match.',
  tool_name: 'The tool that must appear in the trajectory.',
  skill_name: 'The skill that must appear in the trajectory.',
  pass_threshold: 'The score a case must reach to pass.',
};

/* Built-in evaluators whose documentation page is not their own name in snake case. */
const EVALUATOR_PAGES = {
  ToolParameterAccuracyEvaluator: 'tool_parameter_evaluator',
  ToolSelectionAccuracyEvaluator: 'tool_selection_evaluator',
  StructuredOutputSimilarity: 'structured_output_evaluator',
  Equals: 'deterministic_evaluators', Contains: 'deterministic_evaluators', StartsWith: 'deterministic_evaluators',
  StateEquals: 'deterministic_evaluators', ToolCalled: 'deterministic_evaluators', SkillInvoked: 'deterministic_evaluators',
  TypedRubricEvaluator: '', Evaluator: 'custom_evaluator',
};
/* Summary and settings of every built-in evaluator, as the server read them off the library. */
let EVALUATOR_INFO = {};

/** The documentation page for an evaluator type; a type the library does not ship gets the custom-evaluator page. */
function evaluatorDoc(type) {
  const builtin = type in EVALUATOR_INFO || type in EVALUATOR_PAGES || /^[A-Z]\w*Evaluator$/.test(type);
  if (!builtin) return ['evaluators/custom_evaluator/', 'evaluator-custom'];
  const page = type in EVALUATOR_PAGES ? EVALUATOR_PAGES[type]
    : type.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase();
  return [`evaluators/${page ? page + '/' : ''}`, `evaluator-${type}`];
}

/* ══════════ how to use each view ══════════ */

const VIEW_GUIDE = {
  missing: {
    steps: [
      'The address names something this folder does not hold, or that could not be read. The page says what it looked for.',
      'Use a link on the page or the sidebar to go to a list of what is here.',
      'The page reads the folder again every few seconds and opens what the address names once it is there.',
    ],
    terms: ['report', 'experiment'],
  },
  reports: {
    steps: [
      'Each row is a <b>report</b>: the result of one run. It names the <b>experiment</b> file it was made from, or says it was run from code.',
      'Select a row to open the report. Select the experiment beside it to open its cases.',
      'No report yet? Run an experiment, then Rescan: <code>strands-evals run FILE --agent module:attr -o NAME.report.json</code>',
    ],
    terms: ['report', 'experiment', 'pairing'],
    links: [['quickstart/', 'Quickstart'], ['cli/run/#experiment-file-run', 'Running an experiment file'], ['cli/report/', 'The report command']],
  },
  experiments: {
    steps: [
      'Each row is an <b>experiment</b> file: the cases and evaluators a run uses, and the reports made from it.',
      'Select a row to open its cases. Select a report beside it to open that report.',
      'An experiment built in code and never saved has no file here. Its reports are still under Reports.',
    ],
    terms: ['experiment', 'case', 'evaluator'],
    links: [['how-to/serialization/', 'Experiment files'], ['cli/generate/', 'Generating a starter experiment'], ['quickstart/', 'Quickstart']],
  },
  experiment: {
    steps: [
      'An experiment is a list of <b>cases</b> and the <b>evaluators</b> that score them. Evaluators are the chips above the cases.',
      'Select a case to edit it. The ? beside each field says what it is for and what it accepts.',
      'Select an evaluator to see what it checks and change its settings.',
      'Save writes the file back. Cases marked draft can be accepted or dropped in one review pass.',
      '<b>Copy as Markdown</b> at the top copies the experiment as text for a coding agent. The selected case has its own copy, and the command that runs it alone.',
      'An experiment with no report offers <b>Copy run command</b> at the top: the command that runs it whole, once <code>--agent</code> names your agent.',
    ],
    terms: ['case', 'evaluator', 'trajectory'],
    links: [['how-evaluation-works/#what-an-evaluation-is-made-of', 'What an evaluation is made of'], ['evaluators/#choose-an-evaluator', 'Choosing an evaluator'], ['how-to/serialization/', 'Experiment files']],
  },
  report: {
    steps: [
      'One row per case and evaluator. The score is the evaluator\'s number; the verdict is its own pass or fail.',
      'Open a row to see the input, the expected and actual output, and the reason the evaluator gave.',
      'Filter to Failing to see what to fix first. Rows marked not applicable are left out of the overall score.',
      'The charts show where the failures sit, how scores are spread, and what each case took. Select a mark to open its row.',
      'Group by lets you split the rows by anything the cases carry, such as a tag, a red-team category or a chaos condition. A row with a diagnosis says what failed, why, and what to change.',
      'To hand it to a coding agent, <b>Copy as Markdown</b> at the top copies the whole report and <b>Copy failing rows as a prompt</b> only what fails. An open row has its own copy actions and a link.',
    ],
    terms: ['score', 'verdict', 'not_applicable'],
    links: [['how-evaluation-works/#the-levels-an-evaluator-scores-at', 'How evaluators score'], ['detectors/', 'Diagnosing failures'], ['cli/report/', 'The report command']],
  },
  sessions: {
    steps: [
      'Each row is a <b>session</b>: the recorded traces of one conversation, saved with <code>strands-evals fetch ... -o FILE</code> or from code.',
      'Select a row to read it, one trace after another.',
      'A <b>diagnosis</b> shows on the page of the session it was made from. One whose session is not in this folder is listed here on its own.',
    ],
    terms: ['session', 'trace', 'diagnosis'],
    links: [['cli/', 'CLI reference'], ['detectors/', 'Diagnosing failures']],
  },
  session: {
    steps: [
      'Read it top to bottom. Each <b>trace</b> has its own heading and what it cost; the time between traces is not counted.',
      'A trace with several agent invocations is split into turns: the prompt and the reply first, then the model and tool calls behind them.',
      'No diagnosis yet? Make one, then Rescan: <code>strands-evals diagnose FILE -o NAME.diagnosis.json</code>',
      '<b>Copy as Markdown</b> at the top copies the session and its diagnoses as text for a coding agent.',
    ],
    terms: ['session', 'trace', 'diagnosis'],
    links: [['detectors/', 'Diagnosing failures'], ['cli/', 'CLI reference']],
  },
  'opt-runs': {
    steps: [
      'Each row is one optimizer run in this folder: what it achieved, the configuration it picked, and why it stopped.',
      'Select a run to open its result. Its other views are the tabs at the top of its page.',
    ],
    terms: ['champion', 'rollout'],
  },
  'opt-setup': {
    steps: [
      'What the run was told to do, as its run.json records it: the objective, the constraints, the cases and the budget.',
      'Nothing here can be edited. It is the record of a finished run.',
    ],
    terms: ['objective', 'split', 'rollout'],
  },
  'opt-overview': {
    steps: [
      'The first card is the result: the champion against the baseline, on cases kept out of the search.',
      'Read the interval, not the gap. An interval that includes zero means the run has not shown a change.',
      'The second card is the configuration to take away. Copy it or download it.',
      'Case by case shows which cases the change came from, for each score and for tokens and time.',
      '<b>Copy result as Markdown</b> at the top copies the result as text for a coding agent.',
    ],
    terms: ['champion', 'baseline', 'interval', 'split'],
  },
  'opt-progress': {
    steps: [
      'One mark per candidate, in the order the search tried them. The line is the best feasible objective so far.',
      'Hover a mark for its figures. Select it to open that candidate.',
    ],
    terms: ['candidate', 'feasible', 'infeasible', 'ineligible', 'rollout'],
  },
  'opt-candidates': {
    steps: [
      'Every configuration the run tried. Open one to see what it changed from the baseline and how each case went.',
      'Open a case to read the answer the agent gave and the conversation behind it.',
      'An open candidate has <b>Copy as Markdown</b> and a link of its own.',
    ],
    terms: ['candidate', 'feasible', 'infeasible', 'ineligible', 'invalid', 'unscored', 'duplicate'],
  },
  'opt-tunables': {
    steps: [
      'The settings the run was allowed to change, what each could be, and how many candidates changed it.',
      'When the run has an effects model beside it, the chart estimates which setting moved the metric.',
    ],
    terms: ['candidate', 'baseline', 'interval'],
  },
  'opt-trade': {
    steps: [
      'One goal against another. Both axes are turned so that better is up and to the right.',
      'A mark above and right of another beats it on both goals.',
    ],
    terms: ['objective', 'champion', 'baseline'],
  },
};

const GUIDE_KEY = 'strands-evals-guide';

/* Closed until it is asked for, and after that as it was left. */
const guideWanted = () => document.cookie.split('; ').includes(`${GUIDE_KEY}=on`);

/** Fill the guide panel for a view: what to do there, the terms it uses, and where the docs say more. */
function renderGuide(view) {
  const guide = VIEW_GUIDE[view];
  const terms = guide.terms.map(key => `<dt>${esc(TERMS[key][0])}</dt><dd>${esc(TERMS[key][1])}</dd>`).join('');
  const links = (guide.links || [['', 'Evals documentation']]).map(([path, label]) =>
    docLink(path, `guide-${view}`, label)).join('');
  $('guide-body').innerHTML = `<h2>${esc(VIEW_NAME[view])}</h2>`
    + `<ol>${guide.steps.map(step => `<li>${step}</li>`).join('')}</ol>`
    + `<dl>${terms}</dl><p class="guide-links">${links}</p>`;
}

function setGuide(on) {
  app.dataset.guide = on ? 'on' : 'off';
  $('guide').hidden = !on;
  app.querySelector('[data-guide-toggle]').setAttribute('aria-expanded', String(on));
}

/* ══════════ help on demand ══════════ */

/** A ? button that opens the explanation of one field, setting, evaluator or term. */
const helpButton = (kind, key, label) =>
  `<button class="help" type="button" data-help="${kind}:${esc(key)}" aria-label="About ${esc(label)}">?</button>`;

/** `{ title, text, values, doc: [path, placement] }` for a help key, or null when there is nothing to say. */
function helpFor(kind, key) {
  if (kind === 'term' && TERMS[key]) {
    const [title, text, path] = TERMS[key];
    return { title, text, doc: [path, `term-${key}`] };
  }
  if (kind === 'field' && FIELD_GUIDE[key]) {
    const [text, values, path] = FIELD_GUIDE[key];
    return { title: key, text, values, doc: [path, `field-${key}`] };
  }
  if (kind === 'evaluator') {
    const info = EVALUATOR_INFO[key];
    const settings = info && info.settings.length
      ? 'Settings: ' + info.settings.map(s => s.name + (s.required ? ' (required)' : '')).join(', ') + '.' : '';
    return {
      title: key,
      text: info ? info.summary : 'Not one of the built-in evaluators. To run this experiment it has to be passed in as a custom evaluator.',
      values: settings,
      doc: evaluatorDoc(key),
    };
  }
  if (kind === 'setting') {
    const [type, name] = key.split(/\.(.*)/s);
    const known = ((EVALUATOR_INFO[type] || {}).settings || []).find(s => s.name === name);
    const state = known ? (known.required ? 'Required.' : `Default: ${known.default}.`) : '';
    if (!SETTING_GUIDE[name] && !state) return null;
    return { title: name, text: SETTING_GUIDE[name] || "Explained on this evaluator's documentation page.", values: state, doc: evaluatorDoc(type) };
  }
  return null;
}

const helpPop = document.createElement('div');
helpPop.className = 'help-pop';
helpPop.setAttribute('role', 'note');
helpPop.hidden = true;
let helpFrom = null;

function closeHelp() {
  if (!helpFrom) return;
  helpFrom.setAttribute('aria-expanded', 'false');
  helpPop.hidden = true;
  helpFrom = null;
}

function openHelp(button) {
  const [kind, key] = button.dataset.help.split(/:(.*)/s);
  const help = helpFor(kind, key);
  if (!help) return;
  helpPop.innerHTML = `<b>${esc(help.title)}</b><p>${esc(help.text)}</p>`
    + (help.values ? `<p class="values">${esc(help.values)}</p>` : '')
    + docLink(help.doc[0], help.doc[1], 'Read the docs');
  /* Placed after the button in the document so Tab reaches its link next; fixed so it adds no layout. */
  button.after(helpPop);
  const box = button.getBoundingClientRect();
  helpPop.style.left = `${Math.max(8, Math.min(box.left, innerWidth - 348))}px`;
  helpPop.style.top = `${box.bottom + 6}px`;
  helpPop.hidden = false;
  button.setAttribute('aria-expanded', 'true');
  helpFrom = button;
}

app.addEventListener('click', e => {
  const button = e.target.closest('[data-help]');
  if (button) {
    const again = button === helpFrom;
    closeHelp();
    if (!again) openHelp(button);
  } else if (!e.target.closest('.help-pop')) {
    closeHelp();
  }
  if (e.target.closest('[data-guide-toggle], [data-guide-close]')) {
    const on = app.dataset.guide !== 'on';
    setGuide(on);
    remember(GUIDE_KEY, on ? 'on' : 'off');
    /* The content column just changed width, and charts are drawn to the width they have. */
    if (REP) renderReportCharts();
    if (OPT) renderOptimization();
  }
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeHelp(); });

/* Links written in the markup name their page in `data-doc`; the address is filled in here so the
   campaign tags are spelled in one place. */
app.querySelectorAll('a[data-doc]').forEach(a => {
  a.href = docUrl(a.dataset.doc, a.dataset.placement);
  a.target = '_blank';
  a.rel = 'noopener';
});
setGuide(guideWanted());
