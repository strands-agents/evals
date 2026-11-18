/* The Report's charts and the trace view inside a report row: the verdict grid, the score
   distribution, this report against another, and what each case cost. */
import { page, runner, tick } from './harness.mjs';

const { ok, done } = runner('charts');
const pages = [];
const same = (actual, expected, msg) => ok(JSON.stringify(actual) === JSON.stringify(expected), msg, JSON.stringify(actual));

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

async function open(files) {
  const p = await page();
  pages.push(p);
  await p.openFolder(files);
  await p.click(p.$$('#runs-rows tr.run').find(r => /now/.test(r.textContent)));
  await p.click('.tab[data-view="report"]');
  return p;
}
const tips = (p, selector) => p.$$(selector).map(el => el.querySelector('title').textContent);
const pick = async (p, selector, value) => {
  p.$(selector).value = value;
  p.$(selector).dispatchEvent(new p.window.Event('change', { bubbles: true }));
  await tick();
};

const p = await open({
  'now/experiment.json': experiment, 'now/report.json': report(NOW, { a: session(5), c: session(2) }),
  'then/experiment.json': experiment, 'then/report.json': report(THEN),
});

// ── the verdict grid: one cell per row of the report, a gap where an evaluator did not run ──
same(tips(p, '#rep-heat rect.cell'), [
  'a · X · score 1.00 · pass', 'b · X · score 0.00 · fail', 'c · X · score 0.80 · pass',
  'a · Y · score 0.20 · fail', 'b · Y · not applicable',
], 'the grid has a cell per case and evaluator that ran, saying what it scored');
same(p.$$('#rep-heat rect.cell').map(c => c.classList[1]), ['pass', 'fail', 'pass', 'fail', 'na'],
  'coloured by the verdict, not applicable apart from both');

await p.type('#rep-find', 'matches nothing');
ok(!p.$('#rep-rows tr.case'), 'with the table filtered down to nothing');
await p.click(p.$$('#rep-heat rect.cell')[1]);
const opened = p.$('#rep-rows tr.case[aria-expanded="true"]');
ok(opened && opened.dataset.i === '2' && p.$('#rep-find').value === '',
  'selecting a cell opens its row in the table, clearing the filter that hid it', opened && opened.dataset.i);

// ── the score distribution: per evaluator, over the rows it applied to ──
same(tips(p, '#rep-spread line.hmean'), ['X · mean 0.6 over 3 rows', 'Y · mean 0.2 over 1 row'],
  'the distribution is per evaluator and leaves out the rows it did not apply to');
ok(p.$$('#rep-spread rect.hbin').length === 4, 'with a bar wherever scores fall', p.$$('#rep-spread rect.hbin').length);

// ── against another report ──
ok(!p.$('#rep-cmp svg') && /Pick an earlier report/.test(p.text('#rep-cmp-cap')), 'nothing is compared until a report is picked');
same(p.$$('#rep-cmp-pick option').map(o => o.value), ['', 'then/report.json'], 'the picker offers the other reports in the folder');
await pick(p, '#rep-cmp-pick', 'then/report.json');
ok(p.text('#rep-cmp-legend') === '1 now pass1 now fail2 kept their verdict1 only in this report1 only in the other',
  'the rows both reports hold are counted by what happened to their verdict', p.text('#rep-cmp-legend'));
same(tips(p, '#rep-cmp g.bell'), ['a · Y · 0.90 to 0.20 · now fails', 'c · X · 0.30 to 0.80 · now passes'],
  'and the ones that moved are drawn, the largest move first');
same(p.$$('#rep-cmp g.bell').map(g => g.classList[1]), ['down', 'up'], 'marked worse or better');
await p.click(p.$$('#rep-cmp g.bell')[1]);
ok(p.$('#rep-rows tr.case[data-i="4"]').getAttribute('aria-expanded') === 'true', 'a row there opens in the table too');

// ── what each case cost, read off its recorded session ──
same(p.$$('#rep-cost text.vlab').map(t => t.textContent), ['5s', '2s'], 'cases with a recorded session are charted by how long they took, longest first');
same(tips(p, '#rep-cost g.rbar'), ['a · 1 model call · 1 tool call · 5s', 'c · 1 model call · 1 tool call · 2s'],
  'with what the session counted');

// ── a recorded session in a report row is drawn as a trace ──
const detail = p.$('#rep-rows tr.detail .trace');
ok(detail && p.text(detail.querySelector('.tr-count')) === '3 spans of a recorded session'
  && p.text(detail.querySelector('.tr-sum')) === '1 model call · 1 tool call · 2s'
  && [...detail.querySelectorAll('.call')].some(c => /lookup/.test(c.textContent)),
'the row shows its session span by span, with the tool call', detail && p.text(detail).slice(0, 200));

// ── a report alone in its folder, with no sessions ──
{
  const q = await open({ 'now/experiment.json': experiment, 'now/report.json': report(NOW) });
  ok(q.$('#rep-cost-card').hidden, 'no cost chart without anything recorded to cost');
  ok(q.$('#rep-cmp-pick').hidden && /No other report in this folder/.test(q.text('#rep-cmp-cap')),
    'and the comparison says there is nothing to compare against', q.text('#rep-cmp-cap'));
}

// ── no report, no charts ──
{
  const q = await page();
  pages.push(q);
  await q.openFolder({ 'now/experiment.json': experiment });
  await q.click('#runs-rows tr.run');
  ok(q.$('#rep-charts').hidden, 'a run with no report shows no charts');
}

done(pages.flatMap(q => q.errs));
