'use strict';

/* ── keeping up with the folder ──
   Served mode only: when the server's fingerprint of the tree moves, re-read the lists and what is open. */

const POLL_MS = 5000;
let polling = false;

/** Re-read the open report and the open experiment where their files changed. Unsaved edits are never replaced. */
async function refreshOpen() {
  await refreshSession();
  const changed = (list, loaded) => {
    const now = loaded && list.find(m => m.path === loaded.path);
    return now && now.modified !== loaded.modified ? now : null;
  };
  const report = repCand ? null : changed(REPORTS, repMeta);
  if (report) loadReport(report, await document_for(report), true);

  const experiment = changed(EXPERIMENTS, expMeta);
  if (!experiment) return;
  if (countDirty() || review || $('v-experiment').inert) {
    note('This file changed on disk while you were editing it.', 'stale');
    return;
  }
  const at = cases[sel] && cases[sel].name;
  loadExperiment(experiment, await document_for(experiment));
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
    try { await refreshOpen(); } catch (err) { scanError = `could not re-read: ${err.message}`; }
    renderLists();
  } finally {
    polling = false;
  }
}

function startPolling() {
  setInterval(pollNow, POLL_MS);
  document.addEventListener('visibilitychange', pollNow);
}
