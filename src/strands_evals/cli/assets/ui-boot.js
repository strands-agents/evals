'use strict';

/* ══════════ boot ══════════ */

function bundled() {
  const size = doc => JSON.stringify(doc, null, 2).length;
  return {
    experiments: [{
      path: 'experiment.json', name: 'experiment.json', dir: '.', obj: SAMPLE_EXPERIMENT, modified: 0,
      size: size(SAMPLE_EXPERIMENT),
      cases: SAMPLE_EXPERIMENT.cases.length, evaluators: SAMPLE_EXPERIMENT.evaluators, reports: ['report.json'],
    }],
    reports: [{
      path: 'report.json', name: 'report.json', dir: '.', obj: SAMPLE_REPORT, modified: 0,
      size: size(SAMPLE_REPORT), ...reportMeta(SAMPLE_REPORT),
      experiment: 'experiment.json', link: linkStatus(SAMPLE_EXPERIMENT, SAMPLE_REPORT),
    }],
  };
}

/* Open what an address names, then show its view. `force` opens without asking about unsaved edits.
   Resolves true when it stopped because the page moved on before a file it asked for arrived. */
async function goTo(want, force) {
  /* Opening things on the way can move the address, and a not-found page keeps the one asked for. */
  const asked = currentRoute(), was = MISSING;
  const lost = notThere(want);
  if (lost) return showMissing(asked, ...lost);
  /* A file that never arrived either would not load, which the page now says, or was overtaken. */
  const overtaken = () => MISSING === was;
  if (want.report && (repCand || !repMeta || repMeta.path !== want.report) && !await openReport(want.report)) return overtaken();
  if (want.experiment && (!expMeta || expMeta.path !== want.experiment)
    && !await openExperiment(want.experiment, { force })) return overtaken();
  if (want.compare && !await openCompare(want.compare, true)) return overtaken();
  if (want.session && (!sesMeta || sesMeta.path !== want.session) && !await openSession(want.session)) return overtaken();
  const opt = want.opt ? OPT_RUNS.findIndex(r => r.id === want.opt) : -1;
  if (opt >= 0 && opt !== optSel) await selectOptimizerRun(opt);
  const inside = notInside(want);
  if (inside) return showMissing(asked, ...inside);
  /* Unsaved edits may have kept another experiment open: the case is picked only in the one the address names. */
  const here = want.case && expMeta && expMeta.path === want.experiment;
  const at = here ? cases.findIndex(c => c.name === want.case) : -1;
  if (at >= 0 && at !== sel) { sel = at; renderForm(); }
  const candidate = want.cand && OPT ? candidateNamed(want.cand) : null;
  if (candidate && repCand !== candidate.candidate_id) openCandidateReport(candidate);
  /* A run that could not be read has no candidate to report on: its own views say why. */
  show(want.cand && !candidate ? 'opt-candidates' : want.tab || homeView(), { silent: true });
  const row = want.row == null ? null : rowNamed(want);
  if (row && !repCand) openReportRow(row.i);
  if (want.candidate) openCandidate(want.candidate);
}

async function boot() {
  let config = null;
  try { config = await api('/api/config'); } catch { /* no server behind the page */ }
  clearExperiment(); clearReport();
  if (config) {
    MODE = 'server'; ROOT = config.root; READONLY = config.read_only; DEPTH = config.depth;
    EVALUATOR_INFO = config.evaluators || {};
    await rescan();
    startPolling();
  } else {
    MODE = 'standalone'; READONLY = false;
    folderLabel = 'bundled sample';
    showingSample = true;
    ({ experiments: EXPERIMENTS, reports: REPORTS } = bundled());
    setSessions({});
    renderLists();
  }

  await goTo(readAddress(), true);
  refreshDirty();
  /* The first step says everything that is open, so Back to it can reopen the same things. */
  writeAddress(currentView(), true);
  /* Set last: boot is async and ends by choosing a view, so drive the page only after this. */
  app.dataset.ready = '1';
}

/* Back, Forward, or an address typed into the open page: go where it says without adding a step. */
async function followAddress() {
  if (currentRoute() === lastRoute) return;
  lastRoute = currentRoute();
  following = true;
  let overtaken;
  try {
    overtaken = await goTo(readAddress());
  } finally {
    following = false;
  }
  /* Where the address could not be followed (unsaved edits kept, a file that is gone), it is corrected.
     Where a click overtook it, the click is a step of its own. */
  writeAddress(currentView(), !overtaken);
}
/* One at a time: a step taken while the one before it is still loading waits its turn. */
let followed = Promise.resolve();
const followInTurn = () => { followed = followed.then(followAddress, followAddress); };
/** Go to an address of the page as if it had been typed: a step Back can undo. */
function visit(route) {
  try {
    history.pushState(null, '', MODE === 'server' ? route : `${location.pathname}${location.search}#${route}`);
  } catch {
    location.hash = route;                     // file://, where the change of hash is what gets followed
    return;
  }
  followInTurn();
}
window.addEventListener('popstate', followInTurn);
window.addEventListener('hashchange', followInTurn);

boot();
