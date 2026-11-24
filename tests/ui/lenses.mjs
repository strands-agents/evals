/* Two lenses on a report: its rows grouped by what the cases carry, and a row's diagnosis read as
   what failed, why, where and what to do. */
import { fakeServer, page, runner, served, tick } from './harness.mjs';

const { ok, same, done } = runner('lenses');

const NA = [{ score: 1, test_pass: true, label: 'not_applicable' }];
/* name, evaluator, score, pass, metadata, the evaluator's outputs */
const report = rows => ({
  overall_score: 0.5,
  cases: rows.map(([name, evaluator, , , metadata]) => ({ name, evaluator, input: 'q', ...(metadata ? { metadata } : {}) })),
  scores: rows.map(r => r[2]), test_passes: rows.map(r => r[3]),
  detailed_results: rows.map(r => r[5] || []),
});
const BILLING = { team: 'billing', tags: ['vip', 'eu'], context: '', success_criteria: null };
const TAGGED = report([
  ['order', 'X', 1, true, { ...BILLING, limits: { turns: 3 } }],
  ['order', 'Y', 0.2, false, { ...BILLING, limits: { turns: 3 } }],
  ['login', 'X', 0, false, { team: 'accounts', tags: ['eu'] }],
  ['refund', 'X', 1, true, BILLING],
  ['plain', 'X', 0.5, true],
  ['plain', 'Y', 1, true, null, NA],
]);
const CHAOS = report([
  ['order|baseline', 'X', 1, true], ['order|tool_timeout', 'X', 0, false],
  ['refund|baseline', 'X', 1, true], ['refund|tool_timeout', 'X', 0.5, true],
]);
const BARE = report([['a', 'X', 1, true], ['b', 'X', 0, false]]);

const lens = p => {
  const choices = () => p.$$('#rep-group-pick button').map(b => p.text(b));
  const choose = label => p.click(p.$$('#rep-group-pick button').find(b => p.text(b) === label));
  const summary = () => p.$$('#rep-groups tr.group').map(tr => [...tr.cells].map(td => p.text(td)));
  const group = value => p.$$('#rep-groups tr.group').find(tr => p.text(tr.cells[0]) === value);
  /* The table as read down the page: a group's heading, then `case/evaluator` for each of its rows. */
  const table = () => {
    let name = '';
    return p.$$('#rep-rows tr.grouphead, #rep-rows tr.case').map(tr => {
      if (tr.classList.contains('grouphead')) return `[${p.text(tr)}]`;
      name = p.text(tr.cells[0]).replace(/ diagnosed$/, '') || name;
      return `${name}/${p.text(tr.cells[1])}`;
    });
  };
  return { choices, choose, summary, group, table };
};

// ── grouping by what the cases carry ──
{
  const p = await served({ 'tagged/report.json': TAGGED, 'chaos/report.json': CHAOS, 'bare/report.json': BARE });
  await p.openReport('tagged');
  const { choices, choose, summary, group, table } = lens(p);

  same(choices(), ['None', 'limits', 'tags', 'team'],
    'the picker offers none and every metadata key some case has a value for, leaving out a key that is blank in all of them');
  ok(!p.$('#rep-lens').hidden && !p.$('#rep-groups tr.group') && !p.$('#rep-rows tr.grouphead')
    && p.text('#rep-group-cap') === 'Group the rows by something the cases carry, to see how one slice of them did.',
  'with none chosen there is no summary and the table is as it was', p.text('#rep-group-cap'));
  same(table(), ['order/X', 'order/Y', 'login/X', 'refund/X', 'plain/X', 'plain/Y'], 'ungrouped, the rows keep the order of the report');

  await choose('team');
  same(p.$$('#rep-groups th').map(th => p.text(th)), ['team', 'Cases', 'Rows', 'Passing', 'Failing', 'Mean score'],
    'the summary is headed by the key it groups by');
  same(summary(), [
    ['billing', '2', '3', '2 of 3', '1', '0.73'],
    ['accounts', '1', '1', '0 of 1', '1', '0.00'],
    ['(not set)', '1', '2', '1 of 1', '0', '0.50'],
  ], 'each value has its cases, rows, passing, failing and mean score, a not-applicable row counted as a row and not judged');
  ok(group('billing').querySelector('.bar2 .fill').style.width === '73%', 'the mean score is drawn as a bar',
    group('billing').querySelector('.bar2 .fill').style.width);
  same(table(), ['[team billing]', 'order/X', 'order/Y', 'refund/X', '[team accounts]', 'login/X', '[team (not set)]', 'plain/X', 'plain/Y'],
    'the table puts the cases of a value together under its name');
  ok(p.text('#rep-group-cap') === "By the value of team in each case's metadata. Select a group to see only its rows.",
    'the caption says where the value was read from', p.text('#rep-group-cap'));

  await p.click(group('billing'));
  same(table(), ['[team billing]', 'order/X', 'order/Y', 'refund/X'], 'selecting a group shows only its rows');
  ok(p.text('#rep-count') === '3 of 6 rows' && group('billing').getAttribute('aria-pressed') === 'true'
    && group('accounts').getAttribute('aria-pressed') === 'false',
  'the count follows, and the group says it is the one selected', p.text('#rep-count'));
  await p.click('#rep-seg button[data-f="fail"]');
  same(table(), ['[team billing]', 'order/X', 'order/Y'], 'the Failing filter narrows within the selected group');
  await p.click('#rep-seg button[data-f="all"]');
  await p.key('Enter', group('billing'));
  ok(p.text('#rep-count') === '6 of 6 rows' && !p.$('#rep-groups tr.group.on'), 'selecting it again, by keyboard too, shows every group',
    p.text('#rep-count'));

  await choose('tags');
  same(summary().map(s => s.slice(0, 2)), [['vip, eu', '2'], ['eu', '1'], ['(not set)', '1']],
    'a list value groups by its items read as text');
  await choose('limits');
  same(summary().map(s => s.slice(0, 2)), [['{"turns":3}', '1'], ['(not set)', '3']], 'a dict value groups by its JSON');
  await p.click(group('{"turns":3}'));
  same(table(), ['[limits {"turns":3}]', 'order/X', 'order/Y'], 'and a structured value can be selected like any other');
  ok(!p.text('#rep-lens').includes('[object Object]') && !p.text('#rep-rows').includes('[object Object]'),
    'no structured value is printed as [object Object]');

  await p.openReport('chaos');
  same(choices(), ['None', 'condition, from the name'], 'case names that end in |condition offer the condition as a group');
  ok(p.$('#rep-group-pick button.on').textContent === 'None' && !p.$('#rep-groups tr.group'),
    'another report opens ungrouped');
  await choose('condition, from the name');
  same(summary(), [['baseline', '2', '2', '2 of 2', '0', '1.00'], ['tool_timeout', '2', '2', '1 of 2', '1', '0.25']],
    'the rows of a chaos report are summarised per condition');
  ok(p.text('#rep-group-cap') === 'Read from the part of each case name after "|", which is how a chaos experiment names its '
    + 'conditions. Select a group to see only its rows.', 'and the caption says the condition was read from the name',
  p.text('#rep-group-cap'));
  same(table(), ['[condition baseline]', 'order|baseline/X', 'refund|baseline/X', '[condition tool_timeout]', 'order|tool_timeout/X',
    'refund|tool_timeout/X'], 'the table groups by it too');

  await p.openReport('bare');
  ok(p.$('#rep-lens').hidden && p.$('#rep-seg [data-f="diag"]').hidden && !p.$('#rep-rows tr.grouphead')
    && p.$$('#rep-rows tr.case').length === 2,
  'a report whose cases carry no metadata has no Group by control and no Diagnosed filter, and its rows are untouched');
}

// ── the grouping is kept when the report is written again ──
{
  const listed = path => ({ path, name: path, dir: '.', size: 1, modified: 5, rows: 6, overall_score: 0.5, evaluators: [],
    experiment: null, link: null });
  const tree = { mark: 'm1', reports: [listed('r.json'), listed('other.json')],
    files: { 'r.json': { modified: 5, document: TAGGED }, 'other.json': { modified: 5, document: TAGGED } } };
  const server = fakeServer({
    mark: () => tree.mark,
    scan: () => [{ root: '/served', depth: 3, read_only: false, fingerprint: tree.mark, experiments: [], reports: tree.reports,
      skipped: [], optimizer_runs: [] }],
    file: path => ({ path, ...tree.files[path] }),
  });
  const p = await page({ fetchImpl: server, visible: true });
  await p.openReport('r.json');
  const { choose, summary, group, table } = lens(p);
  await choose('team');
  await p.click(group('accounts'));

  const rerun = JSON.parse(JSON.stringify(TAGGED));
  rerun.scores[2] = 1; rerun.test_passes[2] = true;
  tree.files['r.json'] = { modified: 6, document: rerun };
  tree.reports[0].modified = 6;
  tree.mark += '+';
  await p.window.pollNow(); await tick();

  ok(p.text('#rep-group-pick button.on') === 'team' && group('accounts').classList.contains('on'),
    'a live refresh keeps the grouping and the selected group', p.text('#rep-group-pick'));
  same([summary()[1], table()], [['accounts', '1', '1', '1 of 1', '0', '1.00'], ['[team accounts]', 'login/X']],
    'and the figures are those of the report as it is now');

  await p.openReport('other.json');
  ok(p.text('#rep-group-pick button.on') === 'None' && p.text('#rep-count') === '6 of 6 rows',
    'opening a different report starts ungrouped', p.text('#rep-count'));

  const rewrite = async (document, modified) => {
    tree.files['other.json'] = { modified, document };
    tree.reports[1].modified = modified;
    tree.mark += '+';
    await p.window.pollNow(); await tick();
  };
  await rewrite({ ...TAGGED, diagnoses: [{ failures: [], root_causes: [] }] }, 6);
  await p.click('#rep-seg [data-f="diag"]');
  same(table(), ['order/X', 'order/Y'], 'a diagnosis written while the report is open can be filtered by');
  await rewrite(TAGGED, 7);
  ok(p.$('#rep-seg button.on').dataset.f === 'all' && p.text('#rep-count') === '6 of 6 rows',
    'and when the report is written again with none, the filter it no longer offers is dropped', p.text('#rep-count'));
}

// ── a diagnosis read as one ──
{
  const span = (n, more) => ({ span_info: { span_id: `span-${n}`, trace_id: 't', session_id: 'sess-1' }, ...more });
  const SESSION = { session_id: 'sess-1', traces: [{ trace_id: 't', session_id: 'sess-1', spans: [
    span(0, { span_type: 'invoke_agent', user_prompt: 'Where is my order?', agent_response: 'It shipped.' }),
    span(1, { span_type: 'inference', messages: [{ role: 'assistant', content: [{ text: 'I will look it up.' }] }] }),
    span(2, { span_type: 'execute_tool', tool_call: { name: 'lookup_order', arguments: { customer: 'c-1' } },
      tool_result: { content: 'no such order', error: 'no such order' } }),
  ] }] };
  const FIX = 'Say in the tool description that it takes an order id.';
  const cause = more => ({
    failure_span_id: 'span-2', location: 'span-1', causality: 'PRIMARY_FAILURE', propagation_impact: ['TASK_TERMINATION', 'INCORRECT_PATH'],
    failure_detection_timing: 'ONLY_AT_TASK_END', completion_status: 'COMPLETE_FAILURE',
    root_cause_explanation: 'The model passed a customer id where an order id was needed.',
    fix_type: 'TOOL_DESCRIPTION_FIX', fix_recommendation: FIX, ...more,
  });
  const DIAGNOSIS = {
    session_id: 'sess-1',
    failures: [{ span_id: 'span-2', category: ['execution-error-category-timeout', 'HALLUCINATION'], confidence: [0.9, 0.55],
      evidence: ['The tool returned an error.', 'The reply cites an order <b>that</b> does not exist.'] }],
    root_causes: [
      cause(),
      cause({ failure_span_id: 'span-9', location: 'span-9', causality: 'SECONDARY_FAILURE', propagation_impact: [],
        root_cause_explanation: 'The reply did not say the lookup failed.' }),
    ],
  };
  const HALF = { failures: [null, { span_id: 'gone', category: 'not a list' }], root_causes: 'nope' };
  const rows = [
    ['lookup', 'X', 0, false], ['lookup', 'Y', 1, true], ['fine', 'X', 1, true],
    ['odd', 'X', 0, false], ['half', 'X', 0, false], ['advice', 'X', 0, false],
  ];
  const DIAGNOSED = {
    ...report(rows),
    diagnoses: [DIAGNOSIS, DIAGNOSIS, null, 'the detector timed out', HALF, null],
    recommendations: [FIX, FIX, null, 'Run it again.', null, 'Add a retry.'],
  };
  DIAGNOSED.cases[0].actual_trajectory = SESSION;
  DIAGNOSED.cases[1].actual_trajectory = SESSION;

  const p = await served({ 'run/report.json': DIAGNOSED });
  await p.openReport('run');
  const { table } = lens(p);
  const marked = () => p.$$('#rep-rows tr.case').filter(tr => tr.querySelector('.tag')).map(tr => p.text(tr.cells[0]));
  same(marked(), ['lookup diagnosed', 'odd diagnosed', 'half diagnosed'], 'a case with a diagnosis is marked on its first row');

  ok(!p.$('#rep-seg [data-f="diag"]').hidden && p.text('#rep-seg [data-f="diag"]') === 'Diagnosed',
    'a report with diagnoses offers a Diagnosed filter');
  await p.click('#rep-seg [data-f="diag"]');
  same(table(), ['lookup/X', 'lookup/Y', 'odd/X', 'half/X'], 'which shows only the cases that have one');
  await p.click('#rep-seg [data-f="all"]');

  for (const row of p.$$('#rep-rows tr.case')) await p.click(`#rep-rows tr.case[data-i="${row.dataset.i}"]`);
  const detail = i => p.$(`#rep-rows tr.case[data-i="${i}"] + tr.detail`);
  const diag = i => detail(i).querySelector('.diag');
  const parts = el => Object.fromEntries([...el.querySelectorAll('h5')].map(h => [h.textContent, h.nextElementSibling]));
  ok(p.$$('#rep-rows tr.detail').length === 6, 'every row opens, whatever its diagnosis holds', p.$$('#rep-rows tr.detail').length);

  const full = parts(diag(0));
  same(Object.keys(full), ['What failed', 'Why', 'What to do'], 'a diagnosis reads as what failed, why, and what to do');
  ok(p.text(diag(0).querySelector('h4 + p')) === '1 failure · 2 root causes · session sess-1',
    'headed by how much it found and in which session', p.text(diag(0).querySelector('h4 + p')));
  same(p.$$('.diag-cat').slice(0, 2).map(c => p.text(c)), [
    'execution error: timeout90% sureThe tool returned an error.',
    'hallucination55% sureThe reply cites an order <b>that</b> does not exist.',
  ], 'each failure category reads as words, with how sure the detector was and its evidence as text');
  ok(p.text(full['What failed'].querySelector('.diag-at')) === 'at #2 tool call lookup_order',
    'a failure says where it happened, by the span of the trace beside it', p.text(full['What failed'].querySelector('.diag-at')));

  const why = [...full.Why.children];
  ok(p.text(why[0].querySelector('.diag-at')) === 'primary failure began at #1 model call, seen at #2 tool call lookup_order'
    && p.text(why[0].querySelector('p')) === 'The model passed a customer id where an order id was needed.'
    && p.text(why[0].querySelector('p.path')) === 'Effect: task termination, incorrect path · Noticed: only at task end · Outcome: complete failure',
  'a root cause says where it began and was seen, why, and what it did to the run', p.text(why[0]));
  ok(p.text(why[1].querySelector('.diag-at')) === 'secondary failure began at span-9' && !why[1].querySelector('.spanref'),
    'a span that is not in the recorded session is named, not linked', p.text(why[1].querySelector('.diag-at')));
  same([...full['What to do'].children].map(li => p.text(li)), [`tool description fix ${FIX}`],
    'what to do lists each fix once, with the kind of fix it is');
  same(JSON.parse(diag(0).querySelector('details.raw pre').textContent), DIAGNOSIS, 'the diagnosis as written is one disclosure away');
  ok(!diag(0).querySelector('details.raw').open && p.text(diag(0).querySelector('details.raw summary')) === 'the diagnosis as written',
    'and stays closed until asked for');

  const lit = () => [...detail(0).querySelectorAll('.trace .msg.lit')].map(c => p.text(c.querySelector('.who')));
  await p.click(full['What failed'].querySelector('.spanref'));
  same(lit(), ['#2tool'], 'a span named by the diagnosis marks its card in the trace of the row');
  await p.click(why[0].querySelector('.spanref'));
  same(lit(), ['#1inference'], 'and choosing another moves the mark');

  ok(!diag(2) && detail(2).querySelector('.blocks'), 'a row with no diagnosis shows none, and the rest of the row as before');

  ok(p.text(diag(3).querySelector('p.empty')) === 'This is not in the shape a diagnosis is written in, so it is shown as it was written.'
    && diag(3).querySelector('pre.val').textContent === 'the detector timed out'
    && p.text(parts(diag(3))['What to do']) === 'Run it again.',
  'a diagnosis that is not in the usual shape is shown as written, with its recommendation', p.text(diag(3)));

  const half = parts(diag(4));
  ok(Object.keys(half).join() === 'What failed' && half['What failed'].children.length === 2
    && p.text(half['What failed'].children[0]) === 'null' && p.text(half['What failed'].children[1]) === 'at gone'
    && !diag(4).querySelector('.spanref')
    && p.text(diag(4).querySelector('h4 + p')) === '2 failures · 0 root causes',
  'a diagnosis with parts missing or mistyped shows what it can read', p.text(diag(4)));
  same(JSON.parse(diag(4).querySelector('details.raw pre').textContent), HALF, 'and still offers itself as written');

  same([Object.keys(parts(diag(5))), p.text(parts(diag(5))['What to do'])], [['What to do'], 'Add a retry.'],
    'a recommendation recorded with no diagnosis is shown as what to do');
  ok(!/\[object Object\]|undefined/.test(p.text('#rep-rows')),
    'nothing in a diagnosis is printed as [object Object] or undefined');
}

done();
