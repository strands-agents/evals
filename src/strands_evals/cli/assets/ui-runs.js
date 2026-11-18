'use strict';

let RUNS = [], selRun = -1, skipped = [], truncated = false, folderLabel = '';
let showingSample = false;
let scanError = '';                          // why the last scan failed, if it did
let treeMark = null;                         // the server's fingerprint of the tree when it was last scanned

/* ══════════ runs view ══════════ */

const blank = (title, body) => `<tr><td colspan="4"><div class="nores"><b>${title}</b>${body}</div></td></tr>`;

const LINK_TAG = {
  verified: ['ok', 'pairing verified'],
  partial: ['warn', 'partial match'],
  mismatch: ['warn', 'does not match'],
  unverified: ['', 'unverified pairing'],
};

function renderRuns() {
  $('mode-tag').textContent = MODE === 'server'
    ? (READONLY ? 'served · read-only' : 'served · saves to disk')
    : 'standalone file · saves download';
  $('mode-tag').className = 'mode ' + (MODE === 'server' ? (READONLY ? 'ro' : 'live') : 'ro');
  $('mode-tag').dataset.tip = MODE !== 'server' ? 'Opened without a server, so nothing can be written back. Save downloads a copy.'
    : READONLY ? 'Served with --read-only. Cases can be read but not saved.'
      : 'Served by strands-evals ui. Save writes the experiment file in place.';
  $('root-path').textContent = ROOT || folderLabel || '';
  $('runs-sub').textContent = MODE === 'server'
    ? `Everything ${DEPTH != null ? DEPTH + ' levels ' : ''}under the directory the CLI was started in.`
    : showingSample
      ? 'No server behind this page, so it is showing the sample it carries. Open a folder to read your own.'
      : `Reading the folder you opened${folderLabel ? ` (${folderLabel})` : ''}. Edits save as a download, not back to these files.`;
  app.querySelectorAll('[data-act="rescan"]').forEach(b => { b.hidden = MODE !== 'server'; });
  app.querySelector('[data-act="pick"]').hidden = MODE === 'server';

  $('runs-count').innerHTML = plural(RUNS.length, 'run')
    + (truncated ? ' · <span class="tag warn">scan hit the file cap</span>' : '');

  const failed = scanError
    ? `<tr><td colspan="4"><span class="tag warn">could not scan: ${esc(scanError)}</span></td></tr>` : '';
  $('runs-rows').innerHTML = failed + (RUNS.length ? RUNS.map((r, i) => {
    const e = r.experiment, p = r.report;
    const mod = Math.max((e && e.modified) || 0, (p && p.modified) || 0);
    const label = r.dir && r.dir !== '.' ? r.dir : (e || p).name;
    const sub = r.dir && r.dir !== '.' ? ` <span class="dim">/ ${esc((e || p).name)}</span>` : '';
    const link = r.link && LINK_TAG[r.link.status]
      ? `<br><span class="tag ${LINK_TAG[r.link.status][0]}" title="${esc(r.link.detail)}">${LINK_TAG[r.link.status][1]}</span>`
      : '';
    return `<tr class="run${i === selRun ? ' on' : ''}" data-run="${i}" tabindex="0">
        <td>${esc(label)}${sub}</td>
        <td class="ev${e ? ' goto' : ''}"${e ? ' data-goto="experiment" title="open these cases in the Experiment tab"' : ''}>${e
        ? `${plural(e.cases, 'case')} · ${plural((e.evaluators || []).length, 'evaluator')}`
          + `<br><span class="path">${esc(e.name)} · ${KB(e.size)}</span>`
        : '<span class="empty">none</span>'}</td>
        <td class="ev${p ? ' goto' : ''}"${p ? ' data-goto="report" title="open these results in the Report tab"' : ''}>${p
        ? `${scoreBar(p.overall_score)} · ${plural(p.rows, 'row')}`
          + (p.failing ? ` · <span class="vb fail"><i></i>${p.failing} failing</span>` : '')
          + `<br><span class="path">${esc(p.name)} · ${KB(p.size)}</span>${link}`
        : '<span class="empty">not run yet</span>'}</td>
        <td class="ev">${when(mod)}</td></tr>`;
  }).join('') : failed ? '' : blank(
    'Nothing to show here',
    MODE === 'server'
      ? `No experiment or report JSON under ${esc(ROOT)}. `
        + docLink('quickstart/#run-your-first-evaluation', 'empty-runs', 'Create your first experiment')
      : 'Open a folder to look for experiments and reports.'));

  $('skips').innerHTML = skipped.length
    ? `<details class="skips"><summary>${plural(skipped.length, 'file')} skipped</summary><ul>`
      + skipped.map(s => `<li><code>${esc(s.path)}</code> — ${esc(s.why)}</li>`).join('')
      + '</ul></details>'
    : '';
}

app.querySelectorAll('[data-act="rescan"]').forEach(b => { b.onclick = () => rescan(true); });

/* ══════════ loading a run ══════════ */

async function document_for(meta) {
  if (!meta) return null;
  if (meta.obj) return meta.obj;                       // standalone: already in memory
  const res = await api('/api/file?path=' + encodeURIComponent(meta.path));
  meta.modified = res.modified;                        // refresh the save guard's baseline
  if (res.evaluator_notes) meta.evaluator_notes = res.evaluator_notes;
  return res.document;
}

async function rescan(keepSelection) {
  const previous = selRun >= 0 && RUNS[selRun] ? RUNS[selRun].id : null;
  let res;
  try {
    res = await api('/api/runs');
  } catch (err) {
    scanError = err.message;
    renderRuns();
    return null;
  }
  scanError = '';
  ROOT = res.root; DEPTH = res.depth; READONLY = res.read_only;
  RUNS = res.runs; skipped = res.skipped || []; truncated = !!res.truncated;
  treeMark = res.fingerprint;
  selRun = keepSelection && previous ? RUNS.findIndex(r => r.id === previous) : -1;
  renderRuns();

  await loadOptimizerRuns(res.optimizer_runs || [], keepSelection);
  return res;
}

async function selectRun(i, opts) {
  const run = RUNS[i];
  if (!run) return;
  const dirty = countDirty();
  if (!(opts && opts.force) && dirty && i !== selRun
      && !confirm(`${plural(dirty, 'unsaved change')} will be lost. Switch run?`)) return;
  selRun = i;
  repDisplaced = null;
  $('rep-borrowed').hidden = true;
  $('rep-borrowed').innerHTML = '';
  try {
    const [expDoc, repDoc] = await Promise.all([document_for(run.experiment), document_for(run.report)]);
    expDoc ? loadExperiment(run.experiment, expDoc) : clearExperiment();
    repDoc ? loadReport(run.report, repDoc) : clearReport();
  } catch (err) {
    clearExperiment(); clearReport();
    $('runs-rows').insertAdjacentHTML('afterbegin',
      `<tr><td colspan="4"><span class="tag warn">could not load: ${esc(err.message)}</span></td></tr>`);
    return;
  }
  renderRuns();
  const target = (opts && opts.view) || (run.experiment ? 'experiment' : 'report');
  show(target);
}

$('runs-rows').onclick = e => {
  const tr = e.target.closest('tr.run');
  if (!tr) return;
  const cell = e.target.closest('td[data-goto]');
  const i = +tr.dataset.run;
  /* Re-selecting the current run would re-read from disk and drop unsaved edits. */
  if (i === selRun) return show(cell ? cell.dataset.goto : (RUNS[i].experiment ? 'experiment' : 'report'));
  selectRun(i, cell ? { view: cell.dataset.goto } : undefined);
};
