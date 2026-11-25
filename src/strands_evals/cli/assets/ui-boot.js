'use strict';

/* ══════════ boot ══════════ */

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
  if (want.session && (!sesMeta || sesMeta.path !== want.session) && !await openSession(want.session)) return overtaken();
  const opt = want.opt ? OPT_RUNS.findIndex(r => r.id === want.opt) : -1;
  if (opt >= 0 && opt !== optSel && !await openOptimizerRun(opt)) return overtaken();
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

/* The page is drawn from what its server reads. Opened any other way, it says how to start one, and nothing else. */
function unserved() {
  app.classList.add('unserved');
  MISSING = { route: currentRoute(), list: 'reports' };
  $('missing').innerHTML = emptyState('No server is answering',
    'This page is served by <code>strands-evals ui</code>, and nothing is answering it. '
      + 'Run <code>strands-evals ui FOLDER</code> and open the link it prints.');
  show('missing', { silent: true });
  app.dataset.ready = '1';
}

async function boot() {
  clearExperiment(); clearReport();
  let config;
  try { config = await api('/api/config'); } catch { return unserved(); }
  ROOT = config.root; READONLY = config.read_only; DEPTH = config.depth;
  EVALUATOR_INFO = config.evaluators || {};
  await rescan();
  startPolling();

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
  history.pushState(null, '', route);
  followInTurn();
}
window.addEventListener('popstate', followInTurn);

boot();
