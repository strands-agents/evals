/* Standalone mode: the page opened with no server behind it. */
import { page, runner } from './harness.mjs';

const { ok, done } = runner('standalone');
const { $, $$, text, click, type, saved, openFolder, openExperiment, openReport, errs } = await page();   // no fetch, so /api/config fails
const caseCount = () => $$('#case-list li[data-i]').length;
const on = () => $('.view.on').id;

// ── boot: the bundled sample ──
ok(text('.tab.on') === 'Reports' && $$('.view.on').length === 1, 'lands on Reports, the one view showing', text('.tab.on'));
ok(/standalone.*download/.test(text('#mode-tag')), 'the mode tag says standalone, and that saves download', text('#mode-tag'));
ok($('[data-act="rescan"]').hidden && !$('[data-act="pick"]').hidden, 'Open folder is offered in place of Rescan');
ok(/sample/.test(text('#reports-sub')), 'and the page says it is showing its sample', text('#reports-sub'));

const report = text('#reports-rows'), experiment = text('#experiments-rows');
ok($$('#reports-rows tr.row').length === 1 && /0\.92/.test(report) && /5 failing/.test(report)
  && /experiment\.json ?pairing verified/.test(report),
'the sample is one report, summarised, naming the experiment it was made from with the pairing checked', report);
ok($$('#experiments-rows tr.row').length === 1 && /13 cases · 3 evaluators/.test(experiment) && /report\.json 0\.92/.test(experiment),
  'and one experiment, its cases and evaluators counted and its report beside it', experiment);
ok($('#pagehead').hidden && caseCount() === 0 && text('#r-score') === '—', 'nothing is open until it is asked for');

// ── a report opens on its own page, and names the experiment it was made from ──
await openReport();
ok(on() === 'v-report' && text('#crumbs') === 'Reports/report.json' && text('#r-score') === '0.92' && caseCount() === 0,
  'opening a report shows it under a header that says what it is, and loads nothing else', text('#crumbs'));
ok(text('.tab.on') === 'Reports' && /^0\.92 overall · 39 rows · 5 failing · of experiment\.json ?pairing verified$/.test(text('#facts')),
  'the sidebar still marks the list it came from, and the header says how it went', text('#facts'));
const stats = $$('#r-stats .stat').map(s => `${s.querySelector('.v').textContent} ${s.querySelector('.k').textContent}`);
ok(stats[0] === '31 / 36 Rows passing' && stats[3] === '3 Not applicable',
  'rows an evaluator declined are counted apart from the rows passing', stats.join(', '));
await click('#facts a[data-experiment]');
ok(on() === 'v-experiment' && text('#crumbs') === 'Experiments/experiment.json' && caseCount() === 13
  && /^13 cases · 3 evaluators · Report: report\.json 0\.92$/.test(text('#facts')),
'the experiment named there opens its cases, under a header that links back to its report', text('#facts'));
await click('#crumbs a');
ok(on() === 'v-experiments', 'and the first part of the header leads back to the list', on());
await openExperiment();

// ── evaluators and their settings ──
const chips = () => $$('#evs .evchip').map(c => [...c.children].map(el => el.textContent).join(' '));
ok(chips().join(' | ') === 'OutputEvaluator | ToolTrajectoryEvaluator threshold=0.9 | support_rubric RubricEvaluator',
  'a chip shows the name or else the class, the class beside a name, and any settings', chips().join(' | '));
// Only the server can check an evaluator against the registry.
ok(text('#evnotes') === '', 'with no verdict on any of them without a server', text('#evnotes'));

await click('#evs .evchip');
ok($('#evform').hidden === false, 'clicking a chip opens its settings');
await click('[data-evclose]');
ok($('#evform').hidden === true, 'and Close shuts them');

await click($$('#evs .evchip')[1]);
await type('#evform [data-evk="threshold"]', '0.75');
ok(text('#dirty') === '1 unsaved change', 'editing a setting is an unsaved change', text('#dirty'));
await click('#revert');
ok(text('#dirty') === 'No unsaved changes' && /threshold=0.9/.test(chips()[1]), 'which Revert undoes', chips()[1]);

// ── adding, saving and deleting a case ──
const before = caseCount();
await click('#add-case');
ok(caseCount() === before + 1 && !$('#save').disabled, 'a new case joins the list, unsaved', caseCount());
await click('#save');
ok(/has no input/.test(text('#saved')) && await saved() === null, 'Save refuses a case with no input', text('#saved'));
await type('#form [data-k="input"]', 'a brand new question');
await click('#save');
ok((await saved()).cases[before].input === 'a brand new question' && text('#dirty') === 'No unsaved changes',
  'and downloads the file once it has one', text('#saved'));
await click('#del-case');
ok(caseCount() === before, 'Delete removes the case', caseCount());

// The report could not tell two cases of one name apart.
await click($$('#case-list li[data-i]')[0]);
await type('#form [data-k="name"]', $$('#case-list li[data-i] .nm')[1].textContent);
await click('#save');
ok(/two cases are named/.test(text('#saved')), 'Save refuses two cases of one name', text('#saved'));
await click('#revert');

// ── an opened folder, and trajectories that are not lists of strings ──
await openFolder({
  'tree/experiment.json': {
    cases: [
      { name: 'dicts', session_id: 'sid-1', input: 'hi',
        expected_trajectory: [{ name: 'search', arguments: { q: 'rome' } }, { tool: 'reply' }] },
      { name: 'session', session_id: 'sid-2', input: 'hi',
        expected_trajectory: { session_id: 's1', spans: [{ name: 'agent' }] } },
      { name: 'strings', session_id: 'sid-3', input: 'hi', expected_trajectory: ['search', 'reply'] },
    ],
    evaluators: [{ evaluator_type: 'ToolTrajectoryEvaluator' }],
  },
});
ok($$('#experiments-rows tr.row').length === 1 && !$('#reports-rows tr.row') && /folder you opened/.test(text('#experiments-sub'))
  && on() === 'v-experiments' && caseCount() === 0,
'an opened folder replaces the sample, and with no reports in it the page opens on its experiments', text('#experiments-sub'));
await openExperiment();
const trajectory = () => text('[aria-labelledby="label-expected_trajectory"]');
ok(caseCount() === 3 && /search\(q="rome"\)/.test(trajectory()) && /reply\(\)/.test(trajectory()),
  'a recorded call reads as its name and arguments, not [object Object]', trajectory());
await click($$('#case-list li[data-i]')[1]);
ok(/session_id/.test(trajectory()), 'a Session trajectory is shown as JSON, not reported as absent', trajectory());
await click($$('#case-list li[data-i]')[2]);
ok($$('#form .chips .chip').length === 2 && !!$('#form .chips .addchip'), 'a list of tool names is chips that can be added to',
  $$('#form .chips .chip').length);

done(errs);
