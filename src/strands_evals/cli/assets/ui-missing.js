'use strict';

/* ── not found ──
   An address that names something the folder does not hold is not corrected to a list: the page says
   what it looked for, and the address stays as typed. */

const named = s => `<code>${esc(s)}</code>`;
const GONE = 'It may have been deleted, renamed or moved, or the link may be from another folder.';

/* A link to another address of the page, followed in place. */
const wayTo = (route, label) =>
  `<a href="${esc(route)}" data-route="${esc(route)}">${esc(label)}</a>`;
const LIST_WAY = {
  reports: ['/reports', 'All reports'], experiments: ['/experiments', 'All experiments'],
  sessions: ['/sessions', 'All sessions'], 'opt-runs': ['/optimization', 'All optimizer runs'],
};
const listWay = list => wayTo(...LIST_WAY[list]);
/* The lists that have something in them, and Reports and Experiments whether they do or not. */
const homeWays = () => ['reports', 'experiments'].concat(sessionRows().length ? ['sessions'] : [],
  OPT_RUNS.length ? ['opt-runs'] : []).map(listWay);

/** Say that `route` names something that is not here. `list` is the list it would be in, if it has one. */
function showMissing(route, list, title, body, ways) {
  MISSING = { route, list };
  $('missing').innerHTML = emptyState(title, `${body}<span class="ways">${ways.join('')}</span>`);
  show('missing');
}

/* The kinds of file an address can name. `key` is its name in a parsed address, `list` the list and
   address its files are under. */
const FILE_KINDS = [
  { key: 'report', a: 'a report', files: () => REPORTS, list: 'reports' },
  { key: 'experiment', a: 'an experiment', files: () => EXPERIMENTS, list: 'experiments' },
  { key: 'session', a: 'a session or a diagnosis', files: () => SESSIONS.concat(DIAGNOSES), list: 'sessions' },
];
const holds = (kind, path) => kind.files().some(m => m.path === path);

/* The report row and the candidate an address names, in what is open. */
const rowNamed = want => ROWS.find(r => r.name === want.row
  && (want.evaluator == null || String(r.evaluator) === want.evaluator));
const candidateNamed = id => OPT.proposals.find(p => p.candidate_id === id);

/** What an address names that the folder does not hold, as `[list, title, body, ways]`, or null. */
function notThere(want) {
  if (want.tab === 'missing') {
    return [null, 'This lily pad is empty', `No page lives at ${named(currentRoute())}. `
      + 'The address may be mistyped, or from an older version of this page.', homeWays()];
  }
  for (const kind of FILE_KINDS) {
    const { key, a, list } = kind, path = want[key];
    if (!path || holds(kind, path)) continue;
    const other = FILE_KINDS.find(k => holds(k, path));
    return other
      ? [list, 'Wrong lily pad', `${named(path)} is ${other.a}, not ${a}.`,
        [wayTo(`/${other.list}${idPath(path)}`, `Open it under ${VIEW_NAME[other.list]}`), listWay(list)]]
      : [list, `That ${key} hopped away`, `No ${key} named ${named(path)} is in this folder. ${GONE}`, [listWay(list)]];
  }
  if (want.opt && !OPT_RUNS.some(r => r.id === want.opt)) {
    /* The last part may be a view the page does not have, of a run that is here. */
    const at = want.opt.lastIndexOf('/');
    const run = OPT_RUNS.find(r => r.id === (at < 0 ? '.' : want.opt.slice(0, at)));
    return ['opt-runs', 'That run hopped away', `No optimizer run is at ${named(want.opt)} in this folder. `
      + (run ? `Did you mean run <b>${esc(run.run_id)}</b>?` : GONE),
    (run ? [wayTo(`/optimization${idPath(run.id)}/overview`, `Open ${run.run_id}`)] : [])
      .concat(listWay(OPT_RUNS.length ? 'opt-runs' : 'reports'))];
  }
  return null;
}

/** What an address names inside something that did open and does not hold it, the same way. */
function notInside(want) {
  if (want.case != null && expMeta && expMeta.path === want.experiment && !cases.some(c => c.name === want.case)) {
    return ['experiments', 'No such case in this pond',
      `Experiment ${named(want.experiment)} has no case named ${named(want.case)}. It may have been renamed or deleted.`,
      [wayTo(`/experiments${idPath(want.experiment)}`, 'Open the experiment'), listWay('experiments')]];
  }
  if (want.row != null && !repCand && repMeta && repMeta.path === want.report && !rowNamed(want)) {
    return ['reports', 'No such row in this pond', `Report ${named(want.report)} has no row for case ${named(want.row)}`
      + `${want.evaluator == null ? '' : ` and evaluator ${named(want.evaluator)}`}. The report may have been written again since.`,
    [wayTo(`/reports${idPath(want.report)}`, 'Open the report'), listWay('reports')]];
  }
  const id = want.cand || want.candidate, run = OPT_RUNS[optSel];
  if (!id || !OPT || !run || run.id !== want.opt) return null;
  const p = candidateNamed(id), runPath = `/optimization${idPath(run.id)}`;
  if (!p) {
    return ['opt-runs', 'That candidate hopped away',
      `Run <b>${esc(run.run_id)}</b> has no candidate named ${named(id)}. The link may be from another run.`,
      [wayTo(`${runPath}/candidates`, 'Candidates of this run'), listWay('opt-runs')]];
  }
  if (want.cand && !p.trials.length) {
    const why = outcomeWhy(p);
    return ['opt-runs', 'Nothing to report from this lily pad',
      `Candidate ${named(id)} of run <b>${esc(run.run_id)}</b> recorded no trial, so it has no cases to show as a report.`
        + (why ? ` ${esc(why)}` : ''),
      [wayTo(`${runPath}/candidates/${encodeURIComponent(id)}`, 'Open the candidate'), wayTo(`${runPath}/candidates`, 'Candidates of this run')]];
  }
  return null;
}

/** After the folder is read again: what was not there may be now. */
async function retryMissing() {
  if (currentView() !== 'missing') return;
  following = true;
  try { await goTo(readAddress()); } finally { following = false; }
}

app.addEventListener('click', e => {
  const way = e.target.closest('[data-route]');
  /* A click that asks for a new tab or window is the browser's to follow. */
  if (!way || e.metaKey || e.ctrlKey || e.shiftKey || e.button) return;
  e.preventDefault();
  visit(way.dataset.route);
});
