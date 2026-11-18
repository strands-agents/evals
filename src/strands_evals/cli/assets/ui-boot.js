'use strict';

/* ══════════ boot ══════════ */

function bundled() {
  const size = doc => JSON.stringify(doc, null, 2).length;
  return [{
    id: 'experiment.json', dir: 'bundled sample',
    experiment: {
      path: 'experiment.json', name: 'experiment.json', obj: SAMPLE_EXPERIMENT, modified: 0,
      size: size(SAMPLE_EXPERIMENT),
      cases: SAMPLE_EXPERIMENT.cases.length, evaluators: SAMPLE_EXPERIMENT.evaluators,
    },
    report: {
      path: 'report.json', name: 'report.json', obj: SAMPLE_REPORT, modified: 0,
      size: size(SAMPLE_REPORT), ...reportMeta(SAMPLE_REPORT),
    },
    link: linkStatus(SAMPLE_EXPERIMENT, SAMPLE_REPORT),
  }];
}

async function boot() {
  let config = null;
  try { config = await api('/api/config'); } catch { /* no server behind the page */ }
  if (config) {
    MODE = 'server'; ROOT = config.root; READONLY = config.read_only; DEPTH = config.depth;
    EVALUATOR_INFO = config.evaluators || {};
    await rescan();
    startPolling();
  } else {
    MODE = 'standalone'; READONLY = false;
    folderLabel = 'bundled sample';
    showingSample = true;
    RUNS = bundled();
    renderRuns();
  }

  clearExperiment(); clearReport();
  const want = readAddress();
  let index = want.run ? RUNS.findIndex(r => r.id === want.run) : -1;
  if (index < 0 && MODE !== 'server' && RUNS.length) index = 0;     // the sample is the point
  if (index >= 0) {
    await selectRun(index, { force: true, view: want.tab || 'runs' });
    if (want.case) {
      const at = cases.findIndex(c => c.name === want.case);
      if (at >= 0) { sel = at; renderForm(); }
    }
  } else {
    renderRuns();
    show(want.tab || 'runs', { silent: true });
  }
  followReport(want);
  refreshDirty();
  /* The first step says everything that is open, so Back to it can reopen the same things. */
  writeAddress(currentView(), true);
  /* Set last: boot is async and ends by choosing a tab, so drive the page only after this. */
  app.dataset.ready = '1';
}

/* A report address that names a candidate shows that candidate's cases; one that does not gives the
   panel back to the run's own report. */
function followReport(want) {
  if (want.tab !== 'report') return;
  const p = want.cand && OPT ? OPT.proposals.find(x => x.candidate_id === want.cand) : null;
  if (p && p.trials.length) {
    if (!repDisplaced || repDisplaced.cand !== p.candidate_id) openCandidateReport(p);
  } else if (repDisplaced) {
    returnReport();
  }
}

/* Back, Forward, or an address typed into the open page: go where it says without adding a step. */
async function followAddress() {
  if (currentRoute() === lastRoute) return;
  lastRoute = currentRoute();
  const want = readAddress();
  following = true;
  try {
    const run = want.run ? RUNS.findIndex(r => r.id === want.run) : -1;
    if (run >= 0 && run !== selRun) await selectRun(run, { view: want.tab || 'runs' });
    const opt = want.opt ? OPT_RUNS.findIndex(r => r.id === want.opt) : -1;
    if (opt >= 0 && opt !== optSel) await selectOptimizerRun(opt);
    const at = want.case ? cases.findIndex(c => c.name === want.case) : -1;
    if (at >= 0 && at !== sel) { sel = at; renderForm(); }
    followReport(want);
    show(want.tab || 'runs', { silent: true });
  } finally {
    following = false;
  }
  /* Where the address could not be followed (unsaved edits kept, a run that is gone), it is corrected. */
  writeAddress(currentView(), true);
}
window.addEventListener('popstate', followAddress);
window.addEventListener('hashchange', followAddress);

boot();
