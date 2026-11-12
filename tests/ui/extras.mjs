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

/* ── the experiment and report cells open their own tab ──
 * The row already selects the run; which of its two halves you wanted is in the cell you
 * clicked, and a run list is read by scanning those columns. */
{
  const p = await page();
  await p.openFolder({
    'alpha/experiment.json': {
      cases: [{ name: 'a', session_id: 's-a', input: 'q', expected_output: 'yes' }],
      evaluators: [{ evaluator_type: 'OutputEvaluator', threshold: 0.8 }],
    },
    'alpha/report.json': {
      overall_score: 0.25,
      cases: [{ name: 'a', session_id: 's-a', evaluator: 'X', input: 'q', actual_output: 'no' }],
      scores: [0.25], test_passes: [false],
    },
    'beta/experiment.json': { cases: [{ name: 'b', session_id: 's-b', input: 'q' }], evaluators: [] },
  });
  const rowOf = name => p.$$('#runs-rows tr.run').find(r => new RegExp(name).test(r.textContent));

  const expCell = rowOf('alpha').children[1];
  ok(expCell.dataset.goto === 'experiment' && expCell.classList.contains('goto'),
    'the experiment cell is marked as a jump', expCell.className + ' ' + expCell.dataset.goto);
  await p.click(expCell);
  ok(p.text('.tab.on') === 'Experiment', 'clicking it lands on the Experiment tab', p.text('.tab.on'));
  ok(p.$$('#case-list li[data-i]').length === 1, 'with that run loaded, not just the tab switched',
    p.$$('#case-list li[data-i]').length);

  /* From another run, so the jump goes through selection rather than the already-here
     shortcut — and the default for a run that has both halves is the Experiment tab, which
     is what the cell has to override. */
  await p.click('.tab[data-view="runs"]');
  await p.click(rowOf('beta'));
  await p.click('.tab[data-view="runs"]');
  await p.click(rowOf('alpha').children[2]);
  ok(p.text('.tab.on') === 'Report', 'the report cell lands on the Report tab', p.text('.tab.on'));
  ok(p.text('#r-score') === '0.25', 'showing that run’s report', p.text('#r-score'));

  // And the same cell on the run already open is the shortcut, which must land identically.
  await p.click('.tab[data-view="runs"]');
  await p.click(rowOf('alpha').children[2]);
  ok(p.text('.tab.on') === 'Report', 'and so does it on the run already open', p.text('.tab.on'));

  /* A run with no report must not have its empty cell jump to an empty Report tab — the
     cell is only a jump when there is something behind it. */
  await p.click('.tab[data-view="runs"]');
  const noReport = rowOf('beta').children[2];
  ok(!noReport.dataset.goto && !noReport.classList.contains('goto'),
    'the "not run yet" cell is not a jump', noReport.className);
  await p.click(noReport);
  ok(p.text('.tab.on') === 'Experiment', 'clicking it does what the row does', p.text('.tab.on'));

  /* Jumping between the tabs of the run you are already on must not re-read it from disk:
     the unsaved-changes prompt only fires when the run changes, so a silent reload here
     would drop the edit without ever asking. */
  await p.type('#form [data-k="expected_output"]', 'edited, not saved');
  await p.click('.tab[data-view="runs"]');
  await p.click(rowOf('beta').children[1]);
  ok(p.text('.tab.on') === 'Experiment', 'the jump still happens', p.text('.tab.on'));
  ok(p.$('#form [data-k="expected_output"]').value === 'edited, not saved',
    'and the unsaved edit survives it', p.$('#form [data-k="expected_output"]').value);
  ok(/1 unsaved change/.test(p.text('#dirty')), 'still reported as unsaved', p.text('#dirty'));
  errs.push(...p.errs);
}

/* ── the properties are set in ink you can read ──
 * Field labels, column headers and the detail list's keys name what you are looking at, so
 * none of them may be set in the page's faintest ink. */
{
  const p = await page();
  await p.openFolder({
    'c/experiment.json': {
      cases: [{ name: 'a', session_id: 's-a', input: 'q', expected_output: 'yes' }],
      // The rubric carries markup, because picking the key out of `name=value` means the
      // value is now interpolated into HTML and has to arrive escaped.
      evaluators: [{ evaluator_type: 'OutputEvaluator', threshold: 0.8, rubric: 'say <b>yes</b>' }],
    },
    'c/report.json': {
      overall_score: 1, cases: [{ name: 'a', session_id: 's-a', evaluator: 'X', input: 'q', actual_output: 'yes' }],
      scores: [1], test_passes: [true],
    },
  });
  await p.click('#runs-rows tr.run');
  /* jsdom does not substitute custom properties, so a computed `color` comes back as the
     literal `var(--inkN)` it was declared with. That is exactly what is being asserted: which
     ink the stylesheet reaches for, not what a particular theme resolves it to. */
  const ink = sel => p.window.getComputedStyle(p.$(sel)).color.trim();
  const FAINT = 'var(--ink3)';
  ok(ink('.field .hint') === FAINT, 'a hint is still the faintest ink, as the reference',
    ink('.field .hint'));
  ok(ink('.field > label') !== FAINT, 'a case field label is not', ink('.field > label'));

  /* And the faintest ink itself has to clear WCAG AA on the card it sits on, or promoting
     the labels would just have moved the problem to the hints. */
  const lum = hex => {
    const ch = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map(v => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2];
  };
  const faintHex = p.window.getComputedStyle(p.$('.app')).getPropertyValue('--ink3').trim();
  // `--card` is declared as `var(--card, #ffffff)` so a host theme can override it; untouched,
  // as it is here, the ground is white.
  const ratio = (lum('#ffffff') + 0.05) / (lum(faintHex) + 0.05);
  ok(ratio >= 4.5, 'the muted ink clears 4.5:1 on an untinted card', `${faintHex} is ${ratio.toFixed(2)}:1`);

  // An evaluator's settings read as `name=value`, with the name picked out of it.
  const chipKey = p.$('#evs .evchip .ck');
  ok(chipKey && chipKey.textContent === 'threshold', 'the evaluator setting names its key',
    chipKey && chipKey.textContent);
  ok(/threshold=0.8/.test(p.text('#evs .evchip .c')), 'and still shows the value beside it',
    p.text('#evs .evchip .c'));
  ok(/say <b>yes<\/b>/.test(p.text('#evs .evchip .c')), 'a value containing markup is shown as text',
    p.text('#evs .evchip .c'));
  ok(!p.$('#evs .evchip .c b'), 'not parsed as markup',
    p.$('#evs .evchip .c').innerHTML);

  await p.click('.tab[data-view="report"]');
  ok(ink('table.rep th') !== FAINT, 'a report column header is not the faintest ink', ink('table.rep th'));
  await p.click('#rep-rows tr.case');
  ok(ink('#rep-rows tr.detail .kv dt') !== FAINT, 'nor is a detail property name',
    ink('#rep-rows tr.detail .kv dt'));
  errs.push(...p.errs);
}

done(errs);
