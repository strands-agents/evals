'use strict';

/* ── keeping up with the folder ──
   When the server's fingerprint of the tree moves, or Rescan is pressed, re-read the
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
  const was = expMeta, read = await readFile(experiment);
  if (expMeta !== was) return;
  /* Asked after the read: an edit made while it was on its way counts too. */
  if (countDirty() || unapplied() || review || $('v-experiment').inert) {
    note('This file changed on disk while you were editing it.', 'stale');
    return;
  }
  const at = cases[sel] && cases[sel].name;
  loadExperimentFile(experiment, read);
  const again = cases.findIndex(c => c.name === at);
  if (again > 0) { sel = again; renderForm(); }
  followCase();
}

/** Re-read the folder, then whatever is open from it. */
async function rereadFolder() {
  if (!await rescan(true)) return;
  try { await refreshOpen(); } catch (err) { scanError = `could not re-read: ${err.message}`; }
  renderLists();
  await retryMissing();
}

async function pollNow() {
  if (document.hidden || polling) return;
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
