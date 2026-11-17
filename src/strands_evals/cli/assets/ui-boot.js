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
  const want = readHash();
  let config = null;
  try { config = await api('/api/config'); } catch { /* no server behind the page */ }
  if (config) {
    MODE = 'server'; ROOT = config.root; READONLY = config.read_only; DEPTH = config.depth;
    await rescan();
  } else {
    MODE = 'standalone'; READONLY = false;
    folderLabel = 'bundled sample';
    showingSample = true;
    RUNS = bundled();
    renderRuns();
  }

  clearExperiment(); clearReport();
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
  refreshDirty();
  /* Set last: boot is async and ends by choosing a tab, so drive the page only after this. */
  app.dataset.ready = '1';
}

boot();
