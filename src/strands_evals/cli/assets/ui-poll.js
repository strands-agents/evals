'use strict';

/* ── keeping up with the folder ──
   Served mode only: when the server's fingerprint of the tree moves, re-read the run list and what is open. */

const POLL_MS = 5000;
let polling = false;

/** Re-read the open experiment and report where their files changed. Unsaved edits are never replaced. */
async function refreshOpenRun() {
  const run = RUNS[selRun];
  if (!run) return;
  const stale = (listed, loaded) => listed && (!loaded || loaded.path !== listed.path || loaded.modified !== listed.modified);

  /* While Optimization has the report panel, the run's own report is the one waiting to come back. */
  const shown = repDisplaced ? repDisplaced.meta : repMeta;
  if (stale(run.report, shown)) {
    const doc = await document_for(run.report);
    if (repDisplaced) Object.assign(repDisplaced, { meta: run.report, doc });
    else loadReport(run.report, doc, shown && shown.path === run.report.path);
  }

  if (!stale(run.experiment, expMeta)) return;
  if (countDirty() || review || $('v-experiment').inert) {
    note('This file changed on disk while you were editing it.', 'stale');
    return;
  }
  const at = cases[sel] && cases[sel].name;
  loadExperiment(run.experiment, await document_for(run.experiment));
  const again = cases.findIndex(c => c.name === at);
  if (again > 0) { sel = again; renderForm(); }
}

async function pollNow() {
  if (MODE !== 'server' || document.hidden || polling) return;
  polling = true;
  try {
    let mark;
    try { mark = (await api('/api/changes')).fingerprint; } catch { return; }   // no answer: the next tick asks again
    if (!mark || mark === treeMark || !await rescan(true)) return;
    try { await refreshOpenRun(); } catch (err) { scanError = err.message; renderRuns(); }
  } finally {
    polling = false;
  }
}

function startPolling() {
  setInterval(pollNow, POLL_MS);
  document.addEventListener('visibilitychange', pollNow);
}
