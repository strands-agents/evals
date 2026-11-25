/* What the page says when there is nothing to show: an address that names something that is not
   there, a list or a view with nothing in it, a file that will not load. Each says why, under the frog. */
import { SAMPLE, fakeServer, page, runner, served } from './harness.mjs';

const { ok, done } = runner('empty');

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const eq = (actual, expected, msg) => ok(same(actual, expected), msg, JSON.stringify(actual));

const GONE = 'It may have been deleted, renamed or moved, or the link may be from another folder.';
/* The frog sits first in the block, hidden from a screen reader, above the headline. */
const frog = (p, within) => !!p.$(`${within} .nores > svg.frog[aria-hidden="true"]:first-child + b`);
const marked = p => p.$$('.sections .tab.on').map(t => t.textContent).join();
const ways = p => p.$$('#missing .ways a').map(a => `${a.textContent} ${a.getAttribute('href')}`);
const here = p => p.window.location.pathname + p.window.location.search;
const way = (p, label) => p.$$('.ways a').find(a => a.textContent === label);

// ── an address that names something that is not there, in a folder of one experiment and its report ──
for (const [route, tab, text, links, what] of [
  ['/nope/at/all', '',
    'Page not foundNo page lives at /nope/at/all. The address may be mistyped, or from an older version of this page.',
    ['All reports /reports', 'All experiments /experiments'], 'an address the page does not have'],
  ['/reports/gone.json', 'Reports', `Report not foundNo report named gone.json is in this folder. ${GONE}`,
    ['All reports /reports'], 'a report that is not in the folder'],
  ['/experiments/deep/gone.json', 'Experiments', `Experiment not foundNo experiment named deep/gone.json is in this folder. ${GONE}`,
    ['All experiments /experiments'], 'an experiment that is not'],
  ['/sessions/gone.json', 'Sessions', `Session not foundNo session named gone.json is in this folder. ${GONE}`,
    ['All sessions /sessions'], 'a session that is not'],
  ['/experiments/report.json', 'Experiments', 'Wrong kind of filereport.json is a report, not an experiment.',
    ['Open it under Reports /reports/report.json', 'All experiments /experiments'], 'a file of another kind than the address says'],
  ['/experiments/experiment.json/cases/no%20such%20case', 'Experiments',
    'Case not foundExperiment experiment.json has no case named no such case. It may have been renamed or deleted.',
    ['Open the experiment /experiments/experiment.json', 'All experiments /experiments'], 'a case the experiment does not hold'],
  ['/experiments/experiment.json/cases/%E0%A4%A', 'Experiments',
    'Case not foundExperiment experiment.json has no case named %E0%A4%A. It may have been renamed or deleted.',
    ['Open the experiment /experiments/experiment.json', 'All experiments /experiments'], 'a case named by a malformed escape'],
  ['/reports/report.json/rows/nobody/Judge', 'Reports',
    'Row not foundReport report.json has no row for case nobody and evaluator Judge. The report may have been written again since.',
    ['Open the report /reports/report.json', 'All reports /reports'], 'a row the report does not hold'],
  ['/optimization/tuning/result', 'Runs', `Run not foundNo optimizer run is at tuning in this folder. ${GONE}`,
    ['All reports /reports'], 'an optimizer run in a folder that has none'],
]) {
  const p = await served(SAMPLE, { route });
  eq([p.view(), here(p), p.text('#missing .nores'), ways(p), marked(p), frog(p, '#missing')],
    ['v-missing', route, text + links.map(l => l.split(' /')[0]).join(''), links, tab, true],
    `${route}: ${what} says so under the frog, keeps its address and marks the list it would be in`);
  ok(p.$('#pagehead').hidden, `${route}: with no page header`, p.$('#pagehead').hidden);
}

// ── what was asked for comes from the address, so it is shown as text ──
{
  const p = await served(SAMPLE, { route: '/reports/%3Cimg%20src%3Dx%20onerror%3Dalert(1)%3E.json' });
  ok(p.text('#missing code') === '<img src=x onerror=alert(1)>.json' && !p.$('#missing img'),
    'a name in the address is shown as it was written, not as markup', p.$('#missing').innerHTML.slice(0, 200));
}

// ── the ways out are steps Back can undo ──
{
  const p = await served(SAMPLE, { route: '/experiments/report.json' });
  await p.click(way(p, 'Open it under Reports'));
  eq([p.view(), here(p), p.text('#crumbs')], ['v-report', '/reports/report.json', 'Reports/report.json'],
    'a way out of a not-found page opens what it names, at its address');
  await p.travel(-1);
  eq([p.view(), here(p), p.text('#missing b')], ['v-missing', '/experiments/report.json', 'Wrong kind of file'],
    'and Back returns to the page that said it was not found');
  await p.click('.tab[data-view="reports"]');
  eq([p.view(), here(p)], ['v-reports', '/reports'], 'the sidebar leaves it as it leaves any page');
  await p.visit('/experiments/experiment.json/cases/order_status_lookup');
  eq([p.view(), p.text('#case-list li.on .nm')], ['v-experiment', 'order_status_lookup'],
    'and an address that names something that is there opens it from a not-found page');
}

// ── inside an optimizer run ──
const START = { event: 'config', iteration: 0 };
const jsonl = events => events.map(e => JSON.stringify(e)).join('\n') + '\n';
const propose = (id, values) => ({ event: 'propose', iteration: 1, candidate_id: id, parent_id: null, values, changed_keys: Object.keys(values) });
const RUN = {
  'r/run.json': { run_id: 'tune-1', objective: 'Quality' },
  'r/events.jsonl': jsonl([START, propose('c1', { temperature: 0.2 }),
    { event: 'iteration_end', iteration: 1, record: { iteration: 1, proposals: [{ candidate_id: 'c1', number: 1, outcome: 'unscored', objective: null }], rollouts_spent: 0 } }]),
};
{
  const p = await served(RUN);
  for (const [route, text, links, what] of [
    ['/optimization/r/bogus', 'Run not foundNo optimizer run is at r/bogus in this folder. Did you mean run tune-1?',
      ['Open tune-1 /optimization/r/result', 'All optimizer runs /optimization'], 'a view the run does not have'],
    ['/optimization/elsewhere/result', `Run not foundNo optimizer run is at elsewhere in this folder. ${GONE}`,
      ['All optimizer runs /optimization'], 'a run that is not in the folder'],
    ['/optimization/r/candidates/c9', 'Candidate not foundRun tune-1 has no candidate named c9. The link may be from another run.',
      ['Candidates of this run /optimization/r/candidates', 'All optimizer runs /optimization'], 'a candidate the run does not hold'],
    ['/optimization/r/candidates/c1/report',
      'No cases to reportCandidate c1 of run tune-1 recorded no trial, so it has no cases to show as a report.',
      ['Open the candidate /optimization/r/candidates/c1', 'Candidates of this run /optimization/r/candidates'],
      'the report of a candidate that ran no case'],
  ]) {
    await p.visit(route);
    eq([p.view(), here(p), p.text('#missing .nores'), ways(p), marked(p)],
      ['v-missing', route, text + links.map(l => l.split(' /')[0]).join(''), links, 'Runs'],
      `${route}: ${what} says so, under Optimization`);
  }
  await p.click(way(p, 'Open the candidate'));
  eq([p.view(), here(p), p.$('#opt-cands tr.case').getAttribute('aria-expanded')],
    ['v-opt-candidates', '/optimization/r/candidates/c1', 'true'], 'and the way to the candidate opens its row');
  await p.visit('/optimization/r');
  eq([p.view(), here(p)], ['v-opt-overview', '/optimization/r/result'],
    'the address of a run with no view named opens its Result view');
}

// ── a run with nothing in it yet ──
{
  const p = await served({ 'r/run.json': { run_id: 'fresh', objective: 'Quality' }, 'r/events.jsonl': jsonl([START]) });
  await p.click('#opt-runs-rows tr');
  const blank = view => [p.$(`#v-${view}`).classList.contains('blank'), p.text(`#v-${view} .optnone`)];
  eq([p.text('#opt-result'), frog(p, '#opt-result')],
    ['No result yetNo candidate has been picked yet. The comparison appears once the run reaches its end phase.', true],
    'Result says there is no result yet, and why');
  eq(blank('opt-progress'), [true, 'The search has not startedNo candidate has been proposed and no iteration has '
    + 'finished, so there is nothing to chart. The run is not finished.'],
  'Iterations says the search has not started, in place of its empty charts');
  eq([p.text('#opt-cands'), frog(p, '#opt-cands')], ['No candidate yetCandidates appear here as each iteration finishes.', true],
    'Candidates says none has been built');
  eq(p.text('#opt-space-sub'), 'This run.json declares no search space: it was written before the '
    + 'field existed. The configuration above names the keys the log changed, which is as close as the record gets.',
  'the Tunables section of Setup says the run declares no search space');
  eq([p.$('#opt-to-none').hidden, p.text('#opt-to-none'), frog(p, '#opt-to-none')],
    [false, 'One goal, so nothing to trade offThis run optimized one goal, so it has a best value rather than a trade-off. '
      + 'Iterations charts it by candidate.', true], 'Trade-off says a run with one goal has none');
  eq(blank('opt-setup'), [false, ''], 'and Setup, which has a run.json to show, is drawn');

  for (const [file, held] of Object.entries(RUN)) p.write(file, held);
  await p.click('[data-act="rescan"]');
  eq([blank('opt-progress'), blank('opt-candidates')], [[false, ''], [false, '']],
    'a run that has proposed a candidate draws its views again');
}
{
  const p = await served({ 'r/run.json': {}, 'r/events.jsonl': jsonl([START]) });
  await p.click('#opt-runs-rows tr');
  eq([p.$('#v-opt-setup').classList.contains('blank'), p.text('#v-opt-setup .optnone'), frog(p, '#v-opt-setup .optnone')],
    [true, 'No setup recordedThis run’s run.json is empty, so there is no record of what the run was told to do.', true],
    'a run whose run.json is empty says there is no setup to show');
}

// ── a run whose log cannot be read ──
{
  const run = { id: 'tuning', dir: 'tuning', run_id: 'tune-2', name: 'tune-2', events_path: 'tuning/events.jsonl', config: {},
    candidates: 0, iterations: 0, modified: 1, has_effects: false };
  const server = fakeServer({ scan: { optimizer_runs: [run] } });
  const p = await page({ url: 'http://127.0.0.1:9999/optimization/tuning/iterations',
    fetchImpl: (url, opts) => (String(url).startsWith('/api/events')
      ? { ok: false, status: 500, statusText: '', json: async () => ({ error: 'the log is not UTF-8' }) } : server(url, opts)) });
  const said = 'This run could not be readCould not read tuning/events.jsonl: the log is not UTF-8All optimizer runs';
  eq([p.view(), p.window.location.pathname, p.$$('.view.blank').length, p.text('#v-opt-progress .optnone'), frog(p, '#v-opt-progress .optnone')],
    ['v-opt-progress', '/optimization/tuning/iterations', 5, said, true],
    'every view of a run that could not be read says so, with the error the server gave');
  await p.click('#v-opt-progress .ways a');
  eq([p.view(), p.window.location.pathname], ['v-opt-runs', '/optimization'], 'and leads back to the list of runs');
}

// ── lists with nothing in them ──
{
  const p = await page({ fetchImpl: fakeServer(), url: 'http://127.0.0.1:9999/' });
  eq([p.text('#reports-rows'), frog(p, '#reports-rows')],
    ['No reports here yetNo report JSON under /served. How to run an experiment', true],
    'a folder with no report says so, and how to make one');
  await p.click('.tab[data-view="experiments"]');
  eq([p.text('#experiments-rows'), frog(p, '#experiments-rows')],
    ['No experiments here yetNo experiment JSON under /served. Create your first experiment', true],
    'and the same for experiments');
  ok(p.$('#reports-tree').hidden && p.$('#experiments-tree').hidden
    && p.$$('.fold[data-fold]').every(f => f.hidden),
  'with nothing found, nothing is listed under the sidebar rows and no caret offers to fold it');
}
{
  const p = await page({ fetchImpl: fakeServer(), url: 'http://127.0.0.1:9999/optimization' });
  eq([p.view(), p.window.location.pathname, p.text('#opt-runs-rows'), frog(p, '#opt-runs-rows'), p.$('.tab[data-view="opt-runs"]').hidden],
    ['v-opt-runs', '/optimization', 'No optimizer runs here yetNo directory under /served holds a run.json with an '
      + 'events.jsonl beside it.', true, false],
    'the address of the runs list in a folder with no run shows it empty, says what a run is, and keeps its tab');
}

// ── a report with no rows, an experiment with no cases ──
{
  const p = await served({
    'run/experiment.json': { cases: [], evaluators: [] },
    'run/report.json': { overall_score: 0, scores: [], cases: [], test_passes: [] },
  });
  await p.openReport();
  eq([p.view(), p.text('#rep-rows'), frog(p, '#rep-rows')],
    ['v-report', 'No rows in this reportThe file holds no rows: the run that wrote it had no case to evaluate.', true],
    'a report with no rows says so');
  await p.click('.tab[data-view="experiments"]');
  await p.openExperiment();
  eq([p.text('#form'), frog(p, '#form')],
    ['No cases yetThis experiment’s case list is empty. Use + Case to add one.', true],
    'and an experiment with no cases says how to add one');
  eq([p.text('#case-list'), p.text('#case-count')], ['No cases yet', '0 of 0 cases'],
    'its case list says it is empty, not that a search found no match');
}
{
  const p = await served(SAMPLE, { route: '/reports/report.json' });
  await p.type('#rep-find', 'no row says this');
  ok(p.text('#rep-rows') === 'No rows match Show every row' && !p.$('#rep-rows .frog'),
    'a search that matches nothing is not an empty report', p.text('#rep-rows'));
  await p.click('#rep-rows [data-act="rep-all"]');
  ok(p.$('#rep-find').value === '' && p.$$('#rep-rows tr.case').length > 0,
    'and it offers the way back: the search is cleared and every row is there again', p.text('#rep-count'));
}

// ── a file that will not load, and one that turns up later ──
{
  const report = (path, modified) => ({ path, name: path, dir: '.', size: 1, modified, overall_score: 1, rows: 1, failing: 0, experiment: null, link: null });
  const DOC = { overall_score: 1, scores: [1], cases: [{ name: 'a', evaluator: 'E' }], test_passes: [true] };
  const tree = { mark: 'm1', reports: [report('bad.json', 1)], readable: new Set() };
  const server = fakeServer({ scan: () => [{ root: '/served', depth: 3, read_only: false, fingerprint: tree.mark,
    experiments: [], reports: tree.reports, skipped: [], optimizer_runs: [] }], mark: () => tree.mark });
  const fetchImpl = (url, opts) => {
    const path = new URL(String(url), 'http://x').searchParams.get('path');
    if (!String(url).startsWith('/api/file')) return server(url, opts);
    return tree.readable.has(path)
      ? { ok: true, status: 200, statusText: '', json: async () => ({ document: DOC, modified: 2 }) }
      : { ok: false, status: 404, statusText: '', json: async () => ({ error: `${path} is not a report` }) };
  };
  const p = await page({ fetchImpl, url: 'http://127.0.0.1:9999/reports', visible: true });
  await p.openReport('bad.json');
  eq([p.view(), p.window.location.pathname, p.text('#missing .nores'), frog(p, '#missing'), marked(p)],
    ['v-missing', '/reports/bad.json', 'This file could not be readbad.json could not be read: bad.json is not a reportAll reports', true, 'Reports'],
    'a report that will not load says why on a page of its own, at the report’s address');
  ok(way(p, 'All reports').getAttribute('href') === '/reports', 'its way out is a path', way(p, 'All reports').getAttribute('href'));
  await p.travel(-1);
  eq([p.view(), p.window.location.pathname], ['v-reports', '/reports'], 'and Back returns to the list');
  await p.travel(1);
  eq(p.view(), 'v-missing', 'Forward tries it again, and says so again');

  tree.readable.add('bad.json');
  tree.mark = 'm2';
  await p.poll();
  eq([p.view(), p.window.location.pathname, p.text('#crumbs')], ['v-report', '/reports/bad.json', 'Reports/bad.json'],
    'once the folder changes and the file can be read, the page opens it with nothing clicked');

  const q = await page({ fetchImpl, url: 'http://127.0.0.1:9999/reports/late.json', visible: true });
  eq([q.view(), q.text('#missing b')], ['v-missing', 'Report not found'], 'a report that is not there yet is not found');
  tree.reports = tree.reports.concat(report('late.json', 3));
  tree.readable.add('late.json');
  tree.mark = 'm3';
  await q.poll();
  eq([q.view(), q.window.location.pathname, q.text('#crumbs')], ['v-report', '/reports/late.json', 'Reports/late.json'],
    'and is opened at the same address when a later scan finds it');
}

// ── an address the page does not have ──
{
  const p = await page({ fetchImpl: fakeServer(), url: 'http://127.0.0.1:9999/nope' });
  eq([p.view(), p.window.location.pathname, p.text('#missing b'), ways(p)],
    ['v-missing', '/nope', 'Page not found', ['All reports /reports', 'All experiments /experiments']],
    'a path that is no page of the app is not found, and stays in the address bar');
  const home = await page({ fetchImpl: fakeServer(), url: 'http://127.0.0.1:9999/index.html' });
  eq(home.view(), 'v-reports', 'and /index.html is the page itself');
}

done();
