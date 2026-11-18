/* What the page hands a coding agent: Markdown for what is open, the path and the command that go
   with it, and an address for one report row or one candidate. The server writes a report, an
   experiment and a session as Markdown too, so the two are run over the same files here and compared. */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { page, runner, waitFor } from './harness.mjs';

const { ok, done } = runner('markdown');
const PY = process.env.PYTHON || 'python3';
const TOKEN = 'markdown-token';
const is = (actual, expected, msg) => ok(actual === expected, msg, `\n--- got\n${actual}\n--- wanted\n${expected}`);

const SESSION = {
  session_id: 's-3',
  traces: [{ spans: [
    { span_type: 'invoke_agent', system_prompt: 'You are terse.', user_prompt: 'Where is order 7?', agent_response: 'In transit.' },
    { span_type: 'inference', messages: [{ role: 'assistant', content: [
      { text: 'I will look it up.' }, { content_type: 'tool_use', name: 'lookup', arguments: { order: 7 } }] }] },
    { span_type: 'execute_tool', tool_call: { name: 'lookup', arguments: { order: 7 } }, tool_result: { content: '{"status": "in transit"}' } },
  ] }],
};
const MESSAGES = [
  { role: 'user', content: [{ text: 'Refund order 7.' }] },
  { role: 'assistant', content: [{ toolUse: { name: 'refund', input: { order: 7 } } }], metadata: { usage: { inputTokens: 12, outputTokens: 3 } } },
  { role: 'user', content: [{ toolResult: { status: 'error', content: [{ text: 'no such order' }] } }] },
  { role: 'assistant', content: 'I could not find order 7.' },
];
const EXPERIMENT = {
  cases: [
    { name: 'refund|policy', session_id: 's-1', input: 'Can I get a refund?', expected_output: 'Yes, within 30 days.', metadata: { tier: 'gold' } },
    { session_id: 's-2', input: { q: [1, 2] }, expected_assertion: 'It looks the order up.', expected_trajectory: ['lookup'] },
  ],
  evaluators: [{ evaluator_type: 'Equals' }, { evaluator_type: 'OutputEvaluator', name: 'tone', rubric: 'Be polite.\nBe specific.', threshold: 0.8 }],
};
/* Rows chosen for what is awkward to write: a pipe and a newline in the names, backticks in a value,
   two outputs on one row, a row with no name, one with no evaluator, one never judged. */
const REPORT = {
  overall_score: 0.375,
  cases: [
    { name: 'refund|policy', session_id: 's-1', evaluator: 'Equals', input: 'Can I get a refund?',
      expected_output: 'Yes, within 30 days.', actual_output: 'No.\n```\nsee policy\n```' },
    { name: 'refund|policy', session_id: 's-1', evaluator: 'tone\nof voice', input: 'Can I get a refund?',
      expected_output: 'Yes, within 30 days.', actual_output: 'No.\n```\nsee policy\n```' },
    { session_id: 's-3', evaluator: 'Equals', input: 'Where is order 7?', expected_trajectory: ['lookup'], actual_trajectory: SESSION },
    { name: 'declined', session_id: 's-4', input: 'hello' },
    { name: 'unjudged', session_id: 's-5', evaluator: 'Equals', input: '__NUMBERS__', actual_output: MESSAGES, expected_interactions: [{ turn: 1 }] },
  ],
  scores: [0, 0.125, 1, 1],
  /* The declined row is marked failed here: declining wins, so it must not be listed as a failure. */
  test_passes: [false, false, true, false],
  reasons: ['expected "Yes"', 'one of two failed'],
  detailed_results: [
    [{ score: 0, test_pass: false, label: 'no_match', reason: 'expected "Yes"' }],
    [{ score: 0.135, test_pass: true, label: 'polite', reason: 'polite enough' }, { score: 1.005, test_pass: false, reason: 'not specific' }],
    [{ score: 1, test_pass: true, label: 'match' }],
    [{ score: 1, test_pass: true, label: 'not_applicable' }],
  ],
  diagnoses: [null, { category: 'prompt', detail: 'no policy text' }],
  recommendations: [null, 'Add the refund policy to the prompt.'],
};
/* A session chosen the same way: a trace with no spans, one with no id, tool output in each shape the
   page unwraps, token counts in each size it shortens, a span of no known kind and one that is not a span. */
const stamp = seconds => ({ start_time: '2026-11-03T09:00:00.000000Z', end_time: `2026-11-03T09:00:0${seconds}0000Z` });
const RECORDED = {
  session_id: 's-9',
  traces: [
    { trace_id: 't-1', spans: [
      { span_type: 'invoke_agent', span_info: { span_id: 'agent', ...stamp('4.25') }, user_prompt: 'Refund order 7.', agent_response: 'Done.' },
      { span_type: 'inference', span_info: { span_id: 'model', ...stamp('1.00') }, messages: [
        { role: 'assistant', content: [{ content_type: 'tool_use', name: 'refund', arguments: { order: 7 } }],
          metadata: { usage: { inputTokens: 12345, outputTokens: 30, cacheReadInputTokens: 2500000 } } },
        { role: 'tool', content: [{ content_type: 'tool_result', content: '{"output": "refunded"}' }, { content_type: 'tool_result', error: 'timed out' }] },
        { role: 'assistant', content: 'All done.' },
        'not a message',
      ] },
      { span_type: 'execute_tool', span_info: { span_id: 'tool', ...stamp('2.00') }, tool_call: { name: 'refund', arguments: { order: 7 } },
        tool_result: { content: ' [1, {"a": 2.0}] ' } },
      { span_type: 'execute_tool', tool_call: {}, tool_result: { content: '{"a": NaN}', error: true } },
    ] },
    { trace_id: 't-2', spans: [] },
    { spans: [{ span_type: 'custom', note: 'kept as written' }, null] },
  ],
};
const DIAGNOSIS = {
  session_id: 's-9',
  failures: [{ span_id: 'tool', category: ['wrong_tool'], confidence: [0.9], evidence: ['It refunded twice.'] }],
  root_causes: [{ failure_span_id: 'tool', location: 'model', root_cause_explanation: 'No limit is described.', fix_recommendation: 'Describe one.' }],
};
const LOST = { session_id: 's-gone', failures: [{ span_id: 'x', category: ['loop'] }], root_causes: [] };
/* Written as text: keys out of JavaScript's order and numbers JSON.stringify would have rewritten. */
const NUMBERS = '{"b": 1.0, "10": 2, "2": [1e21, 1e-7, 0.30000000000000004, 12345678901234567890, true, null, "é \\u0001 \\" \\\\"], "empty": {}, "none": []}';

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ui-markdown-'));
const write = (rel, text) => {
  fs.mkdirSync(path.dirname(path.join(root, rel)), { recursive: true });
  fs.writeFileSync(path.join(root, rel), text);
};
write('support/support.experiment.json', JSON.stringify(EXPERIMENT));
write('support/support.report.json', JSON.stringify(REPORT).replace('"__NUMBERS__"', NUMBERS));
write('code/nightly.report.json', JSON.stringify({ overall_score: 1, cases: [{ name: 'a', evaluator: 'Equals' }, { name: 'b|c', evaluator: 'Equals' }],
  scores: [1, 1], test_passes: [true, true] }));
write('code/weekly.report.json', JSON.stringify({ overall_score: 0.25, run: { started_at: '2026-11-02T10:00:00Z' },
  cases: [{ name: 'a', evaluator: 'Equals' }, { name: 'b|c', evaluator: 'Equals' }], scores: [0, 0.5], test_passes: [false, true] }));
/* The older report says nothing of when it ran, so it is placed by its file's time. */
fs.utimesSync(path.join(root, 'code/nightly.report.json'), Date.UTC(2026, 10, 1) / 1000, Date.UTC(2026, 10, 1) / 1000);
write('empty/experiment.json', JSON.stringify({ cases: [], evaluators: [] }));
write('sessions/refund.session.json', JSON.stringify(RECORDED));
write('sessions/refund.diagnosis.json', JSON.stringify(DIAGNOSIS));
write('sessions/lost.diagnosis.json', JSON.stringify(LOST));
fs.cpSync(new URL('./fixtures/optimizer-run/', import.meta.url).pathname, path.join(root, 'tuning'), { recursive: true });
const ROOT = fs.realpathSync(root);

const port = await new Promise(resolve => {
  const probe = net.createServer();
  probe.listen(0, '127.0.0.1', () => { const { port } = probe.address(); probe.close(() => resolve(port)); });
});
const proc = spawn(PY, [new URL('./serve.py', import.meta.url).pathname, root, String(port), TOKEN], { stdio: ['ignore', 'pipe', 'inherit'] });
await new Promise((resolve, reject) => {
  proc.stdout.on('data', chunk => String(chunk).includes('ready') && resolve());
  proc.on('exit', code => reject(new Error('server exited ' + code)));
});
const base = `http://127.0.0.1:${port}`;
const fromServer = async address => (await fetch(base + address, { headers: { 'X-Evals-Token': TOKEN } })).text();

/* A page whose clipboard is the test's: `copy` clicks a button and returns what it put there. */
async function open(options) {
  const p = await page(options);
  Object.defineProperty(p.window.navigator, 'clipboard', { configurable: true, value: { writeText: async text => { p.copied = text; } } });
  p.copy = async target => { p.copied = null; await p.click(target); return p.copied; };
  return p;
}
async function served(route = '/') {
  const p = await open({ fetchImpl: (url, opts = {}) => fetch(base + url, opts), url: `${base}${route}?t=${TOKEN}` });
  await waitFor(() => p.$('.app').dataset.ready, 'boot to finish');
  return p;
}
const row = (p, i) => p.$(`#rep-rows tr.case[data-i="${i}"]`);
const beside = (p, i) => `#rep-rows tr.case[data-i="${i}"] + tr.detail`;
const labels = (p, within) => p.$$(`${within} [data-md]`).map(b => b.textContent).join(' | ');

const REPORT_FILE = `${ROOT}/support/support.report.json`;
const EXPERIMENT_FILE = `${ROOT}/support/support.experiment.json`;
const ROW_1 = `# refund|policy · tone of voice

- Score: 0.13
- Verdict: failed
- Output 1 of 2: score 0.14, passed, polite
- Output 2 of 2: score 1.00, failed
- Report: \`${REPORT_FILE}\`
- Experiment: \`${EXPERIMENT_FILE}\`

Reason:

\`\`\`text
one of two failed
\`\`\`

Reason, output 1 of 2:

\`\`\`text
polite enough
\`\`\`

Reason, output 2 of 2:

\`\`\`text
not specific
\`\`\`

Diagnosis:

\`\`\`json
{
  "category": "prompt",
  "detail": "no policy text"
}
\`\`\`

Recommendation:

\`\`\`text
Add the refund policy to the prompt.
\`\`\`

Input:

\`\`\`text
Can I get a refund?
\`\`\`

Expected output:

\`\`\`text
Yes, within 30 days.
\`\`\`

Actual output:

\`\`\`\`text
No.
\`\`\`
see policy
\`\`\`
\`\`\`\`
`;
const SESSION_TRACE = `# Trace: 3 spans of a recorded session

1 model call · 1 tool call

## Span 0: invoke agent

System prompt:

\`\`\`text
You are terse.
\`\`\`

Prompt:

\`\`\`text
Where is order 7?
\`\`\`

Response:

\`\`\`text
In transit.
\`\`\`

## Span 1: inference

### assistant

\`\`\`text
I will look it up.
\`\`\`

Tool call \`lookup\`:

\`\`\`json
{
  "order": 7
}
\`\`\`

## Span 2: tool

Tool call \`lookup\`:

\`\`\`json
{
  "order": 7
}
\`\`\`

Tool result:

\`\`\`text
{
 "status": "in transit"
}
\`\`\`
`;
const SESSION_FILE = `# Session: sessions/refund.session.json

- Session: s-9
- Traces: 3
- Spans: 6
- Time: 4.25s
- Diagnoses: sessions/refund.diagnosis.json

## Diagnosis: sessions/refund.diagnosis.json

\`\`\`json
${JSON.stringify(DIAGNOSIS, null, 2)}
\`\`\`

## Trace 1 of 3: t-1

### Span 0: invoke agent

Prompt:

\`\`\`text
Refund order 7.
\`\`\`

Response:

\`\`\`text
Done.
\`\`\`

### Span 1: inference

#### assistant

12k in · 30 out · 2.50M cache read tokens

Tool call \`refund\`:

\`\`\`json
{
  "order": 7
}
\`\`\`

#### tool

Tool result:

\`\`\`text
refunded
\`\`\`

Tool result (error):

\`\`\`text
timed out
\`\`\`

#### assistant

\`\`\`text
All done.
\`\`\`

#### unknown

\`\`\`text

\`\`\`

### Span 2: tool

Tool call \`refund\`:

\`\`\`json
{
  "order": 7
}
\`\`\`

Tool result:

\`\`\`text
[
 1,
 {
  "a": 2
 }
]
\`\`\`

### Span 3: tool

Tool call \`?\`:

\`\`\`json
null
\`\`\`

Tool result (error):

\`\`\`text
{"a": NaN}
\`\`\`

## Trace 2 of 3: t-2

No spans.

## Trace 3 of 3

### Span 4: custom

\`\`\`json
{
  "span_type": "custom",
  "note": "kept as written"
}
\`\`\`

### Span 5: span

\`\`\`json

\`\`\`
`;
const MESSAGE_TRACE = `# Trace: 4 messages

2 assistant messages · 1 tool call · 15 tokens

## Message 1: user

\`\`\`text
Refund order 7.
\`\`\`

## Message 2: assistant

12 in · 3 out tokens

Tool call \`refund\`:

\`\`\`json
{
  "order": 7
}
\`\`\`

## Message 3: user

Tool result (error):

\`\`\`text
no such order
\`\`\`

## Message 4: assistant

\`\`\`text
I could not find order 7.
\`\`\`
`;

try {
  // ── a report: the header's actions, and the server's Markdown for the same file ──
  const p = await served();
  await p.openReport('support');
  await waitFor(() => p.$$('#rep-rows tr.case').length === 5, 'the report');
  is(labels(p, '#actions'), 'Copy as Markdown | Copy failing rows as a prompt | Copy path', 'an open report offers its Markdown, its failing rows and its path');

  const copied = await p.copy('#actions [data-md="report"]');
  is(p.text('#actions [data-md="report"]'), 'Copied', 'the button says it copied');
  is(copied, await fromServer('/reports/support/support.report.json.md'), 'the page and the server write the same Markdown for a report');
  is(copied.split('\n## ')[0], `# Report: support/support.report.json

- Overall score: 0.38
- Rows: 5 (1 passed, 3 failed, 1 not applicable)
- Cases: 4
- Evaluators: Equals, tone of voice, (unnamed)
- Experiment: support/support.experiment.json
`, 'it opens with the figures and the experiment the report was made from');
  ok(copied.includes(`### refund|policy · Equals

- Score: 0.00
- Verdict: failed
- Label: no_match

Reason:

\`\`\`text
expected "Yes"
\`\`\`

Input:

\`\`\`text
Can I get a refund?
\`\`\`

Expected output:

\`\`\`text
Yes, within 30 days.
\`\`\`

Actual output:

\`\`\`\`text
No.
\`\`\`
see policy
\`\`\`
\`\`\`\`

### refund|policy · tone of voice`), 'a failing row carries its reason, input, expected and actual output, in a fence its value cannot close', copied.slice(0, 900));
  ok(copied.includes(`### unjudged · Equals

- Score: none
- Verdict: failed

Input:

\`\`\`json
{
  "2": [
    1e+21,
    1e-7,
    0.30000000000000004,
    12345678901234567000,
    true,
    null,
    "é \\u0001 \\" \\\\"
  ],
  "10": 2,
  "b": 1,
  "empty": {},
  "none": []
}
\`\`\`

Actual output:

\`\`\`json
${JSON.stringify(MESSAGES, null, 2)}
\`\`\`

Expected interactions:

\`\`\`json
[
  {
    "turn": 1
  }
]
\`\`\`

## All rows

| Case | Evaluator | Score | Verdict |
| --- | --- | --- | --- |
| refund\\|policy | Equals | 0.00 | failed |
| refund\\|policy | tone of voice | 0.13 | failed |
| #2 | Equals | 1.00 | passed |
| declined | (unnamed) | 1.00 | not applicable |
| unjudged | Equals | none | failed |
`), 'structured values print as the page prints them, and every row is in one table at the end, a pipe in a name escaped and a row with no name numbered', copied.slice(-1600));
  ok(!copied.includes('### #2') && !copied.includes('### declined'), 'a passing row and one that was not applicable are not among the failing ones');
  await waitFor(() => p.text('#actions [data-md="report"]') === 'Copy as Markdown', 'the button to read as before');

  const prompt = await p.copy('#actions [data-md="failing"]');
  is(prompt.split('\n\n## ')[0], `These 3 rows fail in the evaluation report \`${REPORT_FILE}\`. Find the cause of each failure and fix it.

- Experiment: \`${EXPERIMENT_FILE}\`
- To run one case again: \`strands-evals run ${EXPERIMENT_FILE} --case 'refund|policy' --agent module:attr\``,
  'the prompt opens with one instruction, the files by their full paths and the command for one case');
  is(prompt.split('\n').filter(line => line.startsWith('## ')).join(), '## refund|policy · Equals,## refund|policy · tone of voice,## unjudged · Equals',
    'and carries each failing row, and no other');
  is(await p.copy('#actions [data-md="path"]'), REPORT_FILE, 'the path copied is the absolute one, which works from any directory');

  // ── one row: its Markdown, its command, and an address of its own ──
  await p.click(row(p, 1));
  is(p.window.location.pathname, '/reports/support/support.report.json/rows/refund%7Cpolicy/tone%0Aof%20voice',
    'opening a row puts its case and evaluator in the address');
  is(labels(p, beside(p, 1)), 'Copy as Markdown | Copy link | Copy command', 'the open row offers its Markdown, a link and the command to run it again');
  is(await p.copy(`${beside(p, 1)} [data-md="row"]`), ROW_1, 'a row copies with where it is from, each output of its evaluator and what was diagnosed');
  is(await p.copy(`${beside(p, 1)} [data-md="row-link"]`), `${base}/reports/support/support.report.json/rows/refund%7Cpolicy/tone%0Aof%20voice`, 'its link is that address');
  is(await p.copy(`${beside(p, 1)} [data-md="row-command"]`), `strands-evals run ${EXPERIMENT_FILE} --case 'refund|policy' --agent module:attr`,
    'its command runs that one case of the experiment, with the name quoted for a shell');

  await p.click(row(p, 2));
  is(p.window.location.pathname, '/reports/support/support.report.json/rows/%232/Equals', 'a row with no name is addressed by its number');
  ok(!p.$(`${beside(p, 2)} [data-md="row-command"]`), 'and has no command, since there is no name to run it by');
  is(await p.copy(`${beside(p, 2)} [data-md="trace"]`), SESSION_TRACE, 'a recorded session copies span by span');
  ok((await p.copy(`${beside(p, 2)} .mdacts [data-md="row"]`)).includes('Actual trajectory: a recorded session of 3 spans, kept in the report file.'),
    'the row itself says a session was recorded instead of printing it');
  await p.click(row(p, 4));
  is(await p.copy(`${beside(p, 4)} [data-md="trace"]`), MESSAGE_TRACE, 'a conversation copies message by message, with each tool call and its result');
  await p.click(row(p, 4));
  await p.click(row(p, 2));
  is(p.window.location.pathname, '/reports/support/support.report.json/rows/refund%7Cpolicy/tone%0Aof%20voice', 'closing it leaves the address on the row still open');
  await p.click(row(p, 1));
  is(p.window.location.pathname, '/reports/support/support.report.json', 'and with no row open the address is the report’s');

  // ── following a row's address ──
  const linked = await served('/reports/support/support.report.json/rows/refund%7Cpolicy/tone%0Aof%20voice');
  is(linked.$$('#rep-rows tr.case[aria-expanded="true"]').map(tr => tr.dataset.i).join(), '1', 'a row’s address opens that row and not the other row of its case');
  is(linked.window.location.pathname, '/reports/support/support.report.json/rows/refund%7Cpolicy/tone%0Aof%20voice', 'and the address stays on it');
  const shown = [];
  linked.window.Element.prototype.scrollIntoView = function () { shown.push(this.dataset.i); };
  linked.window.history.pushState(null, '', '/reports/support/support.report.json/rows/declined');
  linked.window.dispatchEvent(new linked.window.PopStateEvent('popstate'));
  await waitFor(() => row(linked, 3).getAttribute('aria-expanded') === 'true', 'the row the address names');
  is(shown.join(), '3', 'a row with no evaluator is addressed by its case alone, and is brought into view');

  // ── a row's address while a lens groups the table and shows one group of it ──
  await linked.click(linked.$$('#rep-group-pick button').find(b => b.textContent === 'condition, from the name'));
  await linked.click(linked.$$('#rep-groups tr.group').find(tr => linked.text(tr.cells[0]) === 'policy'));
  ok(row(linked, 1) && !row(linked, 4), 'with one group shown, the rows of the others are out of the table');
  linked.window.history.pushState(null, '', '/reports/support/support.report.json/rows/unjudged/Equals');
  linked.window.dispatchEvent(new linked.window.PopStateEvent('popstate'));
  await waitFor(() => row(linked, 4) && row(linked, 4).getAttribute('aria-expanded') === 'true', 'the row outside the group shown');
  ok(shown.pop() === '4' && linked.$$('#rep-rows tr.grouphead').length === 2 && !linked.$('#rep-groups tr.group.on'),
    'the address of a row in another group opens it and brings it into view: the table stays grouped, with every group back in it');
  await linked.click(row(linked, 0));
  is(linked.window.location.pathname, '/reports/support/support.report.json/rows/refund%7Cpolicy/Equals', 'and a row opened in a grouped table is still the one the address names');

  // ── a report run from code has no experiment, so no command ──
  await p.click('.tab[data-view="reports"]');
  await p.openReport('nightly');
  await waitFor(() => p.$$('#rep-rows tr.case').length === 2, 'the second report');
  is(labels(p, '#actions'), 'Copy as Markdown | Copy path', 'a report with nothing failing offers no prompt');
  ok((await p.copy('#actions [data-md="report"]')).includes('- Experiment: none found, so this report was run from code\n\n## Failing rows\n\nNone.\n'),
    'its Markdown says it was run from code and that nothing fails', p.copied);
  await p.click(row(p, 0));
  is(labels(p, beside(p, 0)), 'Copy as Markdown | Copy link', 'and its row has no command to offer');

  // ── an experiment and one case of it ──
  await p.click('.tab[data-view="experiments"]');
  await p.openExperiment('support');
  await waitFor(() => p.$$('#case-list li[data-i]').length === 2, 'the cases');
  is(labels(p, '#actions'), 'Copy as Markdown | Copy path', 'an open experiment offers its Markdown and its path');
  const experiment = await p.copy('#actions [data-md="experiment"]');
  is(experiment, await fromServer('/experiments/support/support.experiment.json.md'), 'the page and the server write the same Markdown for an experiment');
  is(experiment, `# Experiment: support/support.experiment.json

- Cases: 2
- Evaluators: 2
- Reports: support/support.report.json

## Evaluators

### Equals

### tone (OutputEvaluator)

Settings:

\`\`\`json
{
  "rubric": "Be polite.\\nBe specific.",
  "threshold": 0.8
}
\`\`\`

## Cases

### refund|policy

Input:

\`\`\`text
Can I get a refund?
\`\`\`

Expected output:

\`\`\`text
Yes, within 30 days.
\`\`\`

Metadata:

\`\`\`json
{
  "tier": "gold"
}
\`\`\`

### #1

Input:

\`\`\`json
{
  "q": [
    1,
    2
  ]
}
\`\`\`

Assertion:

\`\`\`text
It looks the order up.
\`\`\`

Expected trajectory:

\`\`\`json
[
  "lookup"
]
\`\`\`
`, 'which lists each evaluator with its settings, then each case');
  is(await p.copy('#actions [data-md="path"]'), EXPERIMENT_FILE, 'its path is absolute too');

  is(labels(p, '#case-acts'), 'Copy as Markdown | Copy command', 'the selected case has its own actions');
  is(await p.copy('#case-acts [data-md="case"]'), `# refund|policy

- Experiment: \`${EXPERIMENT_FILE}\`

Input:

\`\`\`text
Can I get a refund?
\`\`\`

Expected output:

\`\`\`text
Yes, within 30 days.
\`\`\`

Metadata:

\`\`\`json
{
  "tier": "gold"
}
\`\`\`
`, 'a case copies with the experiment it belongs to');
  is(await p.copy('#case-acts [data-md="case-command"]'), `strands-evals run ${EXPERIMENT_FILE} --case 'refund|policy' --agent module:attr`, 'and with the command that runs it alone');
  await p.type('#form [data-k="input"]', 'Can I still get a refund?');
  ok((await p.copy('#case-acts [data-md="case"]')).includes('```text\nCan I still get a refund?\n```'), 'what is copied is the case as it reads on the page, unsaved edits included', p.copied);
  await p.click('#revert');
  await p.click('#case-list li[data-i="1"]');
  is(await p.copy('#case-acts [data-md="case-command"]'), `strands-evals run ${EXPERIMENT_FILE} --case 1 --agent module:attr`, 'a case with no name is run by its position');

  await p.click('.tab[data-view="experiments"]');
  await p.openExperiment('empty');
  await waitFor(() => /No cases/.test(p.text('#form')), 'the empty experiment');
  ok(p.$('#case-acts').hidden, 'with no case there is nothing to copy, and no action offering to');

  // ── a comparison ──
  await p.click('.tab[data-view="reports"]');
  await p.openReport('nightly');
  await p.click('#cmp-with button[data-value="code/weekly.report.json"]');
  await waitFor(() => p.$('#v-compare').classList.contains('on'), 'the comparison');
  is(labels(p, '#actions'), 'Copy as Markdown', 'a comparison offers its Markdown');
  ok(!p.$('#cmp-with').hidden && p.text('#cmp-with summary') === 'Add a report…', 'beside the control that adds a report to it', p.text('#cmp-with summary'));
  is(await p.copy('#actions [data-md="compare"]'), `# Comparison of 2 reports

Oldest first. Rows are matched by case and evaluator, so reports of different experiments compare only where they hold the same rows.

| # | Report | When | Overall score | Rows | Passing | Failing |
| --- | --- | --- | --- | --- | --- | --- |
| #1 | code/nightly.report.json | 2026-11-01T00:00:00.000Z, by file time | 1.00 | 2 | 2 | 0 |
| #2 | code/weekly.report.json | 2026-11-02T10:00:00.000Z | 0.25 | 2 | 1 | 1 |

## Evaluator means

| Evaluator | #1 | #2 |
| --- | --- | --- |
| Equals | 1.00 | 0.25 |

## From #1 to #2

- Now pass: 0
- Now fail: 1
- Kept their verdict: 1
- Only in #2: 0
- Only in #1: 0

| Case | Evaluator | #1 | #2 | Verdict |
| --- | --- | --- | --- | --- |
| a | Equals | 1.00 | 0.00 | now fails |
| b\\|c | Equals | 1.00 | 0.50 | still passes |
`, 'which holds each report, each evaluator’s mean in each, then what flipped and what moved between the oldest and the newest');
  await p.click('#cmp-with button[data-value="support/support.report.json"]');
  await waitFor(() => p.$$('#pagehead .chip').length === 3, 'the third report');
  const three = await p.copy('#actions [data-md="compare"]');
  ok(three.startsWith('# Comparison of 3 reports\n') && three.includes('\n| Evaluator | #1 | #2 | #3 |\n| --- | --- | --- | --- |\n')
    && /\n\| tone of voice \| no row \| no row \| 0\.13 \|\n/.test(three) && three.includes('\n## From #1 to #3\n')
    && three.endsWith('- Only in #3: 5\n- Only in #1: 2\n\nThe two share no case and evaluator.\n'),
  'a third report adds a column, an evaluator a report has no row from says so, and ends that share nothing say that', three);

  // ── a session: the header's actions, and the server's Markdown for the same file ──
  await p.click('.tab[data-view="sessions"]');
  await p.openSession('refund.session');
  await waitFor(() => p.$('#v-session').classList.contains('on'), 'the session');
  is(labels(p, '#actions'), 'Copy as Markdown | Copy path', 'an open session offers its Markdown and its path');
  const session = await p.copy('#actions [data-md="session"]');
  is(session, await fromServer('/sessions/sessions/refund.session.json.md'), 'the page and the server write the same Markdown for a session');
  is(session, SESSION_FILE, 'which holds what it recorded, the diagnosis made of it, then each trace span by span, numbered as on the page');
  is(await p.copy('#actions [data-md="path"]'), `${ROOT}/sessions/refund.session.json`, 'its path is the session file’s');
  is(JSON.stringify(JSON.parse(await fromServer('/sessions/sessions/refund.session.json.json'))), JSON.stringify(RECORDED), 'and as JSON the server answers with the file’s document');

  await p.click('.tab[data-view="sessions"]');
  await p.openSession('lost');
  await waitFor(() => p.text('#crumbs').includes('lost.diagnosis.json'), 'the diagnosis on its own');
  const lost = await p.copy('#actions [data-md="session"]');
  is(lost, await fromServer('/sessions/sessions/lost.diagnosis.json.md'), 'the page and the server write the same Markdown for a diagnosis whose session is not here');
  is(lost, `# Diagnosis: sessions/lost.diagnosis.json

- Session: s-gone, which is not in the folder
- Failures: 1
- Root causes: 0

\`\`\`json
${JSON.stringify(LOST, null, 2)}
\`\`\`
`, 'which says so, and gives the diagnosis as it was written');
  ok((await fromServer('/sessions/sessions/refund.diagnosis.json.md')).startsWith('# Diagnosis: sessions/refund.diagnosis.json\n\n- Session: s-9, in sessions/refund.session.json\n'),
    'asked for by its own file, a diagnosis names the session file it was made from');

  // ── an optimizer run: its result, and one candidate ──
  await p.click('#sec-opt');
  await p.click('#opt-runs-rows tr[data-opt-run]');
  await waitFor(() => !!p.$('#opt-winner [data-md="result"]'), 'the run');
  await p.click('.tab[data-view="opt-overview"]');
  is(labels(p, '#actions'), 'Copy result as Markdown | Copy path', 'an open run offers its result and its directory');
  is(await p.copy('#actions [data-md="path"]'), `${ROOT}/tuning`, 'the path is the run directory');
  const result = await p.copy('#actions [data-md="result"]');
  const config = fs.readFileSync(path.join(root, 'tuning/events.jsonl'), 'utf8').split('\n').filter(Boolean)
    .map(line => JSON.parse(line)).find(e => e.event === 'propose' && e.candidate_id === 'c10').values;
  ok(result.startsWith(`# Optimizer run support-tuning-20261110-091500-7af3e1

- Run: support-tuning-20261110-091500-7af3e1
- Directory: \`${ROOT}/tuning\`
- Status: stopped: `) && /\n- Champion: c10\n- Derived from: c10 ← [^\n]*c0\n- Compared with: c0, the baseline, on \d+ held-out cases\n- Objective change: \+0\.143, 95% interval \[/.test(result),
  'the result names the run, the champion, what it descends from and what it gained', result.slice(0, 500));
  ok(/\n## Metrics\n\n\| Metric \| Baseline \| Champion \| Change \|\n\| --- \| --- \| --- \| --- \|\n\| /.test(result)
    && /\n## Changes against c0\n\n### [^\n]+\n\n```diff\n/.test(result) && /\n### temperature\n\n```diff\n- 0\.7\n\+ 0\.2\n```\n/.test(result),
  'then each metric beside the baseline, and each changed setting as a diff', result);
  is(JSON.stringify(JSON.parse(result.split('## Configuration\n\n```json\n')[1].replace(/\n```\n$/, ''))), JSON.stringify(config),
    'and ends with the whole configuration of the champion');
  is(await p.copy('#opt-winner [data-md="result"]'), result, 'the same is offered beside the champion’s configuration');

  await p.click('.tab[data-view="opt-candidates"]');
  const candidate = id => p.$$('#opt-cands tr.case').find(tr => tr.dataset.k === id);
  await p.click(candidate('c10'));
  is(p.window.location.pathname, '/optimization/tuning/candidates/c10', 'opening a candidate puts it in the address');
  is(labels(p, '#opt-cands tr.detail .mdacts'), 'Copy as Markdown | Copy link', 'the open candidate offers its Markdown and a link');
  is(await p.copy('#opt-cands tr.detail [data-md="candidate-link"]'), `${base}/optimization/tuning/candidates/c10`, 'the link is that address');
  const card = await p.copy('#opt-cands tr.detail [data-md="candidate"]');
  ok(card.startsWith(`# Candidate c10

- Run: support-tuning-20261110-091500-7af3e1
- Directory: \`${ROOT}/tuning\`
- Iteration: `) && /\n- Standing: champion\n- Derived from: c10 ← /.test(card), 'a candidate says which run it is of and where it stands in it', card.slice(0, 400));
  ok(/\n## Figures against c0\n\n\| Metric \| c0 \| c10 \| Change \|\n\| --- \| --- \| --- \| --- \|\n\| Objective \| [\d.]+ \| [\d.]+ \| \+[\d.]+ \|\n/.test(card)
    && /\n## Changes against c0\n\n### /.test(card) && /\n### temperature\n\n```diff\n- 0\.7\n\+ 0\.2\n```\n/.test(card),
  'then its figures against the baseline and what it changed from it', card);
  await p.click(candidate('c10'));
  is(p.window.location.pathname, '/optimization/tuning/candidates', 'closing it takes it out of the address');
  await p.click(candidate('c0'));
  const first = await p.copy('#opt-cands tr.detail [data-md="candidate"]');
  ok(/\n- Standing: baseline\n/.test(first) && /\n## Figures\n\n\| Metric \| c0 \|\n\| --- \| --- \|\n/.test(first)
    && first.includes('## Configuration\n\nThis is the configuration the run started from.\n\nValues:\n\n```json\n{'),
  'the baseline has nothing to be compared with, so it gives its own figures and configuration', first);

  const pointed = await served('/optimization/tuning/candidates/c10');
  await waitFor(() => pointed.$$('#opt-cands tr.case[aria-expanded="true"]').length > 0, 'the candidate the address names');
  is(pointed.$$('#opt-cands tr.case[aria-expanded="true"]').map(tr => tr.dataset.k).join(), 'c10', 'a candidate’s address opens that candidate');
  ok(pointed.$('#v-opt-candidates').classList.contains('on') && pointed.window.location.pathname === '/optimization/tuning/candidates/c10',
    'on the Candidates page, and the address stays on it', pointed.window.location.pathname);

  // ── a candidate's cases on the report page are not a file ──
  await p.click('#opt-cands tr.detail [data-open]');
  is(labels(p, '#actions').replace(' | Copy failing rows as a prompt', ''), 'Copy as Markdown', 'a candidate’s cases have no path to copy');
  await p.click(row(p, 0));
  is(labels(p, `${beside(p, 0)} .mdacts`), 'Copy as Markdown', 'and a row of them has no link or command');

  // ── with no server: paths as the folder gives them, addresses after the # ──
  const alone = await open({ url: 'http://127.0.0.1:9999/index.html' });
  await alone.openReport();
  is(await alone.copy('#actions [data-md="path"]'), 'report.json', 'with no server the path is the one relative to the folder');
  await alone.click(row(alone, 0));
  const hash = alone.window.location.hash;
  ok(hash.startsWith('#/reports/report.json/rows/order_status_lookup/'), 'and a row’s address follows the #', hash);
  is(await alone.copy(`${beside(alone, 0)} [data-md="row-link"]`), `http://127.0.0.1:9999/index.html${hash}`, 'which is the link it copies');
  const back = await open({ url: `http://127.0.0.1:9999/index.html${hash}` });
  is(back.$$('#rep-rows tr.case[aria-expanded="true"]').map(tr => tr.dataset.i).join(), '0', 'and opening that link opens the row');

  // ── a clipboard that cannot be reached ──
  delete alone.window.navigator.clipboard;
  await alone.click('#actions [data-md="report"]');
  is(alone.text('#actions [data-md="report"]'), 'Could not copy', 'a copy that did not happen says so');

  done([...p.errs, ...linked.errs, ...pointed.errs, ...alone.errs, ...back.errs]);
} finally {
  proc.kill();
  fs.rmSync(root, { recursive: true, force: true });
}
