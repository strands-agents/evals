/* The page from end to end over one experiment and its report: the lists, a report, the cases editor and Save. */
import { SAMPLE, page, runner, served } from './harness.mjs';

const { ok, done } = runner('sample');
const sample = await served(SAMPLE);
const { $, $$, text, click, type, saved, openExperiment, openReport, view: on } = sample;
const caseCount = () => $$('#case-list li[data-i]').length;

// ── boot ──
ok(text('.tab.on') === 'Reports' && $$('.view.on').length === 1, 'lands on Reports, the one view showing', text('.tab.on'));
ok(text('#path') === sample.root && $('#read-only').hidden && !$('[data-act="rescan"]').hidden,
  'the bar names the folder being served, which can be read again, with nothing said of read-only', text('#path'));

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
ok(chips().join(' | ') === 'OutputEvaluator will not load | ToolTrajectoryEvaluator threshold=0.9 custom | support_rubric RubricEvaluator custom',
  'a chip shows the name or else the class, the class beside a name, any settings, and what the server made of it', chips().join(' | '));
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
ok(/has no input/.test(text('#saved')) && saved() === null, 'Save refuses a case with no input', text('#saved'));
await type('#form [data-k="input"]', 'a brand new question');
await click('#save');
ok(saved().cases[before].input === 'a brand new question' && text('#dirty') === 'No unsaved changes',
  'and writes the file once it has one', text('#saved'));
await click('#del-case');
ok(caseCount() === before, 'Delete removes the case', caseCount());

// The report could not tell two cases of one name apart.
await click($$('#case-list li[data-i]')[0]);
await type('#form [data-k="name"]', $$('#case-list li[data-i] .nm')[1].textContent);
await click('#save');
ok(/two cases are named/.test(text('#saved')), 'Save refuses two cases of one name', text('#saved'));
await click('#revert');

// ── a folder with no report, and trajectories that are not lists of strings ──
{
  const { $, $$, text, click, openExperiment, view: on } = await served({
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
  const caseCount = () => $$('#case-list li[data-i]').length;
  ok($$('#experiments-rows tr.row').length === 1 && !$('#reports-rows tr.row') && on() === 'v-experiments' && caseCount() === 0,
    'with no reports in the folder the page opens on its experiments', on());
  await openExperiment();
  const trajectory = () => text('[aria-labelledby="label-expected_trajectory"]');
  ok(caseCount() === 3 && /search\(q="rome"\)/.test(trajectory()) && /reply\(\)/.test(trajectory()),
    'a recorded call reads as its name and arguments, not [object Object]', trajectory());
  await click($$('#case-list li[data-i]')[1]);
  ok(/session_id/.test(trajectory()), 'a Session trajectory is shown as JSON, not reported as absent', trajectory());
  await click($$('#case-list li[data-i]')[2]);
  ok($$('#form .chips .chip').length === 2 && !!$('#form .chips .addchip'), 'a list of tool names is chips that can be added to',
    $$('#form .chips .chip').length);

}

// ── served with --read-only ──
{
  const shut = await served(SAMPLE, { readOnly: true });
  ok(!shut.$('#read-only').hidden && shut.text('#read-only') === 'read-only', 'a server started with --read-only says so in the bar',
    shut.text('#read-only'));
  await shut.openExperiment();
  await shut.type('#form [data-k="input"]', 'an edit with nowhere to go');
  ok(shut.text('#exp-sub') === 'Read-only: the server was started with --read-only.' && shut.text('#dirty') === '1 unsaved change'
    && shut.$('#save').disabled && shut.$('#save').title === 'the server was started with --read-only' && shut.$('#review-case').disabled,
  'and the editor says why an edit cannot be saved', `${shut.text('#exp-sub')} ${shut.text('#dirty')}`);
}

// ── opened without its server ──
{
  const lone = await page();
  ok(lone.view() === 'v-missing' && lone.text('#missing') === 'No pond to look inThis page is served by strands-evals ui, and nothing is '
    + 'answering it. Run strands-evals ui FOLDER and open the link it prints.' && !!lone.$('#missing svg.frog'),
  'a page with no server behind it says how to start one', lone.text('#missing'));
  ok(lone.$('.app').classList.contains('unserved') && !lone.$('#reports-rows tr.row'), 'and shows nothing else');
  /* What a browser does with a request from a file: refused before anything is sent. */
  const disk = await page({ url: 'file:///home/someone/ui.html', fetchImpl: async () => { throw new TypeError('Failed to fetch'); } });
  ok(disk.view() === 'v-missing' && /Run strands-evals ui FOLDER/.test(disk.text('#missing')) && disk.$('.app').dataset.ready === '1',
    'and so does one opened from disk', disk.text('#missing'));
}

done();
