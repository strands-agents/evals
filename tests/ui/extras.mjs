/* Edge cases: big reports, deep links, partial documents, runs missing a half. */
import { page, runner } from './harness.mjs';

const { ok, done } = runner('extras');
const errs = [];

// ── a report big enough to need the cap ──
{
  const n = 420;
  const cases = Array.from({ length: n }, (_, i) => ({
    name: `case_${i}`, session_id: `sid-${i}`, evaluator: 'OutputEvaluator',
    input: `q${i}`, expected_output: 'yes', actual_output: i % 7 ? 'yes' : 'no',
  }));
  const p = await page();
  await p.openFolder({
    'big/experiment.json': { cases: cases.map(({ name, session_id, input }) => ({ name, session_id, input })), evaluators: [] },
    'big/report.json': {
      overall_score: 0.857, cases,
      scores: cases.map((_, i) => (i % 7 ? 1 : 0)),
      test_passes: cases.map((_, i) => !!(i % 7)),
      detailed_results: cases.map((_, i) => [{ score: i % 7 ? 1 : 0, test_pass: !!(i % 7), label: i % 7 ? 'match' : 'no_match' }]),
    },
  });
  await p.click('#runs-rows tr.run');
  await p.click('.tab[data-view="report"]');
  ok(p.$$('#rep-rows tr.case').length === 300, 'first page is capped at 300 rows', p.$$('#rep-rows tr.case').length);
  ok(p.$('#rep-more').hidden === false, 'and says there are more');
  ok(/Show 120 more cases/.test(p.text('#rep-more')), 'naming how many', p.text('#rep-more'));
  ok(p.text('#rep-count') === `${n} of ${n} rows`, 'the count is the real total, not the shown one', p.text('#rep-count'));
  await p.click('#rep-more');
  ok(p.$$('#rep-rows tr.case').length === n, 'show-more reveals the rest', p.$$('#rep-rows tr.case').length);
  ok(p.$('#rep-more').hidden === true, 'and the button goes away');
  errs.push(...p.errs);
}

// ── a deep link opens the case it names ──
{
  const p = await page({ url: 'http://127.0.0.1/?x=1#tab=report&run=experiment.json&case=refund_edge_case' });
  ok(p.text('.tab.on') === 'Report', 'the tab from the hash wins', p.text('.tab.on'));
  ok(p.$$('#case-list li.on')[0]?.textContent.includes('refund_edge_case'),
    'the case from the hash is selected', p.$$('#case-list li.on')[0]?.textContent);
  await p.click('.tab[data-view="experiment"]');
  ok(/case=refund_edge_case/.test(p.window.location.hash), 'the hash keeps naming it', p.window.location.hash);
  errs.push(...p.errs);
}

// ── a report that omits every optional array still renders ──
{
  const p = await page();
  await p.openFolder({
    'bare/experiment.json': { cases: [{ name: 'a', session_id: 's-a', input: 'q' }], evaluators: [] },
    'bare/report.json': {
      overall_score: 1.0,
      cases: [{ name: 'a', session_id: 's-a', evaluator: 'OutputEvaluator', input: 'q', actual_output: 'yes' }],
      scores: [1.0], test_passes: [true],
    },
  });
  await p.click('#runs-rows tr.run');
  await p.click('.tab[data-view="report"]');
  ok(p.$$('#rep-rows tr.case').length === 1, 'the row renders', p.$$('#rep-rows tr.case').length);
  ok(p.text('#r-score') === '1.00', 'score renders', p.text('#r-score'));
  await p.click('#rep-rows tr.case');
  const detail = p.text('#rep-rows tr.detail');
  ok(/evaluator returned no outputs/.test(detail), 'says the row carries no outputs rather than inventing a verdict', detail);
  ok(p.errs.length === 0, 'no errors on a report missing its optional arrays', p.errs.join('; '));
  errs.push(...p.errs);
}

// ── a report that does not belong to the experiment beside it says so ──
{
  const p = await page();
  await p.openFolder({
    'wrong/experiment.json': { cases: [{ name: 'a', session_id: 's-a', input: 'q' }], evaluators: [] },
    'wrong/report.json': {
      overall_score: 0.0,
      cases: [{ name: 'somebody-elses-case', session_id: 's-z', evaluator: 'X' }],
      scores: [0.0], test_passes: [false],
    },
  });
  const row = p.$('#runs-rows tr.run');
  ok(/does not match/.test(row.textContent), 'mismatched pairing is flagged', row.textContent.replace(/\s+/g, ' '));
  errs.push(...p.errs);
}

// ── the default filenames pair even when another experiment is in the folder ──
{
  const p = await page();
  await p.openFolder({
    'd/experiment.json': { cases: [{ name: 'a', session_id: 's-a', input: 'q' }], evaluators: [] },
    'd/report.json': {
      overall_score: 1, cases: [{ name: 'a', session_id: 's-a', evaluator: 'X' }],
      scores: [1], test_passes: [true],
    },
    'd/experiment.saved.json': { cases: [{ name: 'a', session_id: 's-a', input: 'q' }], evaluators: [] },
  });
  const rows = p.$$('#runs-rows tr.run');
  ok(rows.length === 2, 'two experiments, one report', rows.length);
  const paired = rows.find(r => />\s*experiment.json|experiment.json<|experiment.json/.test(r.children[1].textContent)
    && !/saved/.test(r.children[1].textContent));
  ok(/pairing verified/.test(paired.textContent), 'experiment.json paired with report.json',
    paired.textContent.replace(/\s+/g, ' '));
  const other = rows.find(r => /saved/.test(r.children[1].textContent));
  ok(/not run yet/.test(other.children[2].textContent), 'the other experiment is left unpaired',
    other.children[2].textContent);
  errs.push(...p.errs);
}

// ── a run with only a report, and one with only an experiment ──
{
  const p = await page();
  await p.openFolder({
    'only-report/report.json': {
      overall_score: 0.4,
      cases: [{ name: 'a', evaluator: 'X', input: 'q', actual_output: 'no' }],
      scores: [0.4], test_passes: [false],
    },
    'only-exp/experiment.json': { cases: [{ name: 'b', session_id: 's-b', input: 'q' }], evaluators: [] },
  });
  const rows = p.$$('#runs-rows tr.run');
  ok(rows.length === 2, 'both halves listed as runs', rows.length);
  const reportOnly = rows.find(r => /only-report/.test(r.textContent));
  ok(/none/.test(reportOnly.children[1].textContent), 'report-only run says it has no experiment',
    reportOnly.children[1].textContent);
  const expOnly = rows.find(r => /only-exp/.test(r.textContent));
  ok(/not run yet/.test(expOnly.children[2].textContent), 'experiment-only run says it has not run',
    expOnly.children[2].textContent);

  await p.click(reportOnly);
  ok(p.text('.tab.on') === 'Report', 'picking a report-only run opens the report', p.text('.tab.on'));
  await p.click('.tab[data-view="experiment"]');
  ok(/No experiment in this run/.test(p.text('#form')), 'the editor says why it is empty', p.text('#form'));
  ok(p.$('#save').disabled && p.$('#add-case').disabled, 'and offers no controls that cannot work');
  await p.click('.tab[data-view="report"]');
  ok(p.text('#r-score') === '0.40', 'the report still loads', p.text('#r-score'));

  await p.click(p.$$('#runs-rows tr.run').find(r => /only-exp/.test(r.textContent)));
  await p.click('.tab[data-view="report"]');
  ok(/No report in this run/.test(p.text('#rep-rows')), 'and the report side says it has not run',
    p.text('#rep-rows'));
  errs.push(...p.errs);
}

done(errs);
