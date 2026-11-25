/* The Report's charts and the trace view inside a report row: the verdict grid, the score
   distribution, and what each case cost. */
import { runner, served } from './harness.mjs';

const { ok, same, done } = runner('charts');

/* Rows as `[case, evaluator, score, verdict]`, the verdict one of pass, fail or na. */
const report = (rows, trajectories = {}) => ({
  overall_score: 0.5,
  cases: rows.map(([name, evaluator]) => ({ name, evaluator, input: 'q', actual_trajectory: trajectories[name] || null })),
  scores: rows.map(r => r[2]),
  test_passes: rows.map(r => r[3] !== 'fail'),
  detailed_results: rows.map(r => [{ score: r[2], test_pass: r[3] !== 'fail', label: r[3] === 'na' ? 'not_applicable' : 'judged' }]),
});
const experiment = { cases: ['a', 'b', 'c'].map(name => ({ name, session_id: `s-${name}`, input: 'q' })), evaluators: [] };
/* A recorded session: one model call and one tool call, `seconds` long. */
const session = seconds => {
  const at = s => ({ start_time: '2026-01-01T00:00:00Z', end_time: `2026-01-01T00:00:0${s}Z` });
  return { session_id: 's', traces: [{ spans: [
    { span_type: 'invoke_agent', span_info: at(seconds), user_prompt: 'q', agent_response: 'r' },
    { span_type: 'inference', span_info: at(1), messages: [{ role: 'assistant', content: [{ content_type: 'text', text: 'looking' }] }] },
    { span_type: 'execute_tool', span_info: at(1), tool_call: { name: 'lookup', arguments: {} }, tool_result: { content: 'found' } },
  ] }] };
};

const NOW = [['a', 'X', 1, 'pass'], ['a', 'Y', 0.2, 'fail'], ['b', 'X', 0, 'fail'], ['b', 'Y', 1, 'na'], ['c', 'X', 0.8, 'pass']];
const THEN = [['a', 'X', 1, 'pass'], ['a', 'Y', 0.9, 'pass'], ['b', 'X', 0, 'fail'], ['c', 'X', 0.3, 'fail'], ['d', 'X', 1, 'pass']];

const p = await served({
  'now/experiment.json': experiment, 'now/report.json': report(NOW, { a: session(5), c: session(2) }),
  'then/experiment.json': experiment, 'then/report.json': report(THEN),
});
await p.openReport('now /');

// ── the verdict grid: one cell per row of the report, a gap where an evaluator did not run ──
same(p.tips('#rep-heat rect.cell'), [
  'a · X · score 1.00 · pass', 'b · X · score 0.00 · fail', 'c · X · score 0.80 · pass',
  'a · Y · score 0.20 · fail', 'b · Y · not applicable',
], 'the grid has a cell per case and evaluator that ran, saying what it scored');
same(p.$$('#rep-heat rect.cell').map(c => c.classList[1]), ['pass', 'fail', 'pass', 'fail', 'na'],
  'coloured by the verdict, not applicable apart from both');
same([p.$$('#rep-heat path.cell-mark').length, !!p.$('#v-report .legend .cellkey.fail'), !!p.$('#v-report .legend .cellkey.pass')],
  [2, true, true], 'a failing cell is crossed as well as red, so it is told from a passing one without its colour, and the key shows both');
ok(p.$$('#rep-heat rect.cell.fail').every(cell => cell.nextElementSibling.matches('path.cell-mark')),
  'and the cross is drawn on the cell that failed');

await p.type('#rep-find', 'matches nothing');
ok(!p.$('#rep-rows tr.case'), 'with the table filtered down to nothing');
await p.click(p.$$('#rep-heat rect.cell')[1]);
const opened = p.$('#rep-rows tr.case[aria-expanded="true"]');
ok(opened && opened.dataset.i === '2' && p.$('#rep-find').value === '',
  'selecting a cell opens its row in the table, clearing the filter that hid it', opened && opened.dataset.i);
ok(p.window.location.pathname === '/reports/now/report.json/rows/b/X', 'and the address names the row', p.window.location.pathname);

// ── the score distribution: per evaluator, over the rows it applied to ──
same(p.tips('#rep-spread line.hmean'), ['X · mean 0.6 over 3 rows', 'Y · mean 0.2 over 1 row'],
  'the distribution is per evaluator and leaves out the rows it did not apply to');
ok(p.$$('#rep-spread rect.hbin').length === 4, 'with a bar wherever scores fall', p.$$('#rep-spread rect.hbin').length);

// ── what each case cost, read off its recorded session ──
same(p.$$('#rep-cost text.vlab').map(t => t.textContent), ['5s', '2s'], 'cases with a recorded session are charted by how long they took, longest first');
same(p.tips('#rep-cost g.rbar'), ['a · 1 model call · 1 tool call · 5s', 'c · 1 model call · 1 tool call · 2s'],
  'with what the session counted');

// ── a recorded session in a report row is drawn as a trace ──
await p.click(p.$$('#rep-heat rect.cell')[2]);
const detail = p.$('#rep-rows tr.detail .trace');
ok(detail && p.text(detail.querySelector('.tr-count')) === '3 spans of a recorded session'
  && p.text(detail.querySelector('.tr-sum')) === '1 model call · 1 tool call · 2s'
  && [...detail.querySelectorAll('.call')].some(c => /lookup/.test(c.textContent)),
'the row shows its session span by span, with the tool call', detail && p.text(detail).slice(0, 200));

// ── an opened row: what the case asked and got back, beside what the evaluator made of it ──
{
  const card = p.$('#rep-rows tr.case[data-i="2"] + tr.detail .judge');
  ok(card.classList.contains('fail') && p.text(card.querySelector('.judge-by')) === 'X' && /^0\.00 ?fail ?judged$/.test(p.text(card.querySelector('.judge-score'))),
    'the judgement is a card of its own: the evaluator, its score, its verdict and the label it gave', p.text(card));
  ok(/^Input ?q/.test(p.text('#rep-rows tr.case[data-i="2"] + tr.detail .blocks:not(.judge)')), 'beside the case itself');
}

// ── a report with no session recorded for any case ──
ok(!p.$('#rep-cost-card').hidden, 'the cost chart is there while there is something to cost');
await p.openReport('then /');
ok(p.$('#rep-cost-card').hidden && !p.$('#rep-charts').hidden && p.$$('#rep-heat rect.cell').length === 5,
  'a report with nothing recorded to cost drops the cost chart and keeps the others', p.$$('#rep-heat rect.cell').length);

done();
