'use strict';

/* ── keeping up with the folder ──
   Served mode only: when the server's fingerprint of the tree moves, or Rescan is pressed, re-read the
   lists and what is open. */

const POLL_MS = 5000;
let polling = false;

/** Re-read the open report and the open experiment where their files changed. Unsaved edits are never replaced,
    and neither is anything opened while the file was being read. */
async function refreshOpen() {
  await refreshSession();
  const changed = (list, loaded) => {
    const now = loaded && list.find(m => m.path === loaded.path);
    return now && now.modified !== loaded.modified ? now : null;
  };
  const report = repCand ? null : changed(REPORTS, repMeta);
  if (report) {
    const was = repMeta, doc = await document_for(report);
    if (repMeta === was) loadReport(report, doc, true);
  }

  const experiment = changed(EXPERIMENTS, expMeta);
  if (!experiment) return;
  if (countDirty() || review || $('v-experiment').inert) {
    note('This file changed on disk while you were editing it.', 'stale');
    return;
  }
  const was = expMeta, at = cases[sel] && cases[sel].name;
  const doc = await document_for(experiment);
  if (expMeta !== was || countDirty()) return;
  loadExperiment(experiment, doc);
  const again = cases.findIndex(c => c.name === at);
  if (again > 0) { sel = again; renderForm(); }
}

/** Re-read the folder, then whatever is open from it. */
async function rereadFolder() {
  if (!await rescan(true)) return;
  try { await refreshOpen(); await refreshCompare(); } catch (err) { scanError = `could not re-read: ${err.message}`; }
  renderLists();
}

async function pollNow() {
  if (MODE !== 'server' || document.hidden || polling) return;
  polling = true;
  try {
    let mark;
    try { mark = (await api('/api/changes')).fingerprint; } catch { return; }   // no answer: the next tick asks again
    if (mark && mark !== treeMark) await rereadFolder();
  } finally {
    polling = false;
  }
}

function startPolling() {
  setInterval(pollNow, POLL_MS);
  document.addEventListener('visibilitychange', pollNow);
}

app.querySelectorAll('[data-act="rescan"]').forEach(b => { b.onclick = rereadFolder; });
