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

/* Open what an address names, then show its view. `force` opens without asking about unsaved edits. */
async function goTo(want, force) {
  if (want.report && (repCand || !repMeta || repMeta.path !== want.report)) await openReport(want.report);
  if (want.experiment && (!expMeta || expMeta.path !== want.experiment)) await openExperiment(want.experiment, { force });
  if (want.session && (!sesMeta || sesMeta.path !== want.session)) await openSession(want.session);
  const opt = want.opt ? OPT_RUNS.findIndex(r => r.id === want.opt) : -1;
  if (opt >= 0 && opt !== optSel) await selectOptimizerRun(opt);
  const at = want.case ? cases.findIndex(c => c.name === want.case) : -1;
  if (at >= 0 && at !== sel) { sel = at; renderForm(); }
  const candidate = want.cand && OPT ? OPT.proposals.find(p => p.candidate_id === want.cand) : null;
  if (candidate && candidate.trials.length && repCand !== candidate.candidate_id) openCandidateReport(candidate);
  show(want.tab || homeView(), { silent: true });
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
  try {
    await goTo(readAddress());
  } finally {
    following = false;
  }
  /* Where the address could not be followed (unsaved edits kept, a file that is gone), it is corrected. */
  writeAddress(currentView(), true);
}
window.addEventListener('popstate', followAddress);
window.addEventListener('hashchange', followAddress);

boot();
