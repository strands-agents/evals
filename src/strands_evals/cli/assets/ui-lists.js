'use strict';

/* What the folder holds: every report and every experiment file, each report knowing the
   experiment it was made from. A report made in code has none, and one experiment can have many. */
let REPORTS = [], EXPERIMENTS = [], skipped = [], truncated = false, folderLabel = '';
let showingSample = false;
let scanError = '';                          // why the last scan or open failed, if it did
let treeMark = null;                         // the server's fingerprint of the tree when it was last scanned

/* ══════════ the two lists ══════════ */

const blank = (title, body) => `<tr><td colspan="4"><div class="nores"><b>${title}</b>${body}</div></td></tr>`;

const LINK_TAG = {
  verified: ['ok', 'pairing verified'],
  partial: ['warn', 'partial match'],
  mismatch: ['warn', 'does not match'],
  unverified: ['', 'unverified pairing'],
};
const linkTag = link => (link && LINK_TAG[link.status]
  ? `<span class="tag ${LINK_TAG[link.status][0]}" title="${esc(link.detail)}">${LINK_TAG[link.status][1]}</span>` : '');

/** A link that opens a report or an experiment: `kind` is `report` or `experiment`. */
const openLink = (kind, path, label) =>
  `<a data-${kind}="${esc(path)}" tabindex="0" role="link">${esc(label || path)}</a>`;
const FROM_CODE = '<span class="empty" data-tip="No experiment file holds these cases, as when the report was '
  + 'made with run_evaluations() in code and saved with to_file().">run from code</span>';
/* The experiment a report was made from, as its list row and its page header both say it. */
const madeFrom = r => (r.experiment ? `${openLink('experiment', r.experiment)} ${linkTag(r.link)}` : FROM_CODE);

const filed = m => `${m.dir && m.dir !== '.' ? `<span class="dim">${esc(m.dir)} /</span> ` : ''}${esc(m.name)}`
  + `<br><span class="path">${KB(m.size)}</span>`;

function renderLists() {
  $('mode-tag').textContent = MODE === 'server'
    ? (READONLY ? 'served · read-only' : 'served · saves to disk')
    : 'standalone file · saves download';
  $('mode-tag').className = 'mode ' + (MODE === 'server' ? (READONLY ? 'ro' : 'live') : 'ro');
  $('mode-tag').dataset.tip = MODE !== 'server' ? 'Opened without a server, so nothing can be written back. Save downloads a copy.'
    : READONLY ? 'Served with --read-only. Cases can be read but not saved.'
      : 'Served by strands-evals ui. Save writes the experiment file in place.';
  $('root-path').textContent = ROOT || folderLabel || '';
  const from = MODE === 'server'
    ? `Everything ${DEPTH != null ? DEPTH + ' levels ' : ''}under the directory the CLI was started in.`
    : showingSample
      ? 'No server behind this page, so it is showing the sample it carries. Open a folder to read your own.'
      : `Reading the folder you opened${folderLabel ? ` (${folderLabel})` : ''}. Edits save as a download, not back to these files.`;
  $('reports-sub').textContent = `${from} A report is the result of one run, whether it was run from an experiment file or in code.`;
  $('experiments-sub').textContent = `${from} An experiment file holds the cases and evaluators a run uses.`;
  app.querySelectorAll('[data-act="rescan"]').forEach(b => { b.hidden = MODE !== 'server'; });
  app.querySelector('[data-act="pick"]').hidden = MODE === 'server';

  const capped = truncated ? ' · <span class="tag warn">scan hit the file cap</span>' : '';
  $('reports-count').innerHTML = plural(REPORTS.length, 'report') + capped;
  $('experiments-count').innerHTML = plural(EXPERIMENTS.length, 'experiment') + capped;
  const failed = scanError ? `<tr><td colspan="4"><span class="tag warn">${esc(scanError)}</span></td></tr>` : '';
  const nothing = (what, link) => (failed ? '' : blank(`No ${what} here`, MODE === 'server'
    ? `No ${what} JSON under ${esc(ROOT)}. ${link}` : 'Open a folder to look for experiments and reports.'));

  $('reports-rows').innerHTML = failed + (REPORTS.map(r => {
    const open = !repCand && repMeta && repMeta.path === r.path;
    return `<tr class="row${open ? ' on' : ''}" data-report="${esc(r.path)}" tabindex="0">
        <td>${filed(r)}</td>
        <td class="ev">${scoreBar(r.overall_score)} · ${plural(r.rows, 'row')}`
      + (r.failing ? ` · <span class="vb fail"><i></i>${r.failing} failing</span>` : '') + `</td>
        <td class="ev">${madeFrom(r)}</td>
        <td class="ev">${when(r.modified)}</td></tr>`;
  }).join('') || nothing('report', docLink('cli/run/#experiment-file-run', 'empty-reports', 'How to run an experiment')));

  $('experiments-rows').innerHTML = failed + (EXPERIMENTS.map(e => {
    const made = e.reports.map(path => REPORTS.find(r => r.path === path)).filter(Boolean);
    return `<tr class="row${expMeta && expMeta.path === e.path ? ' on' : ''}" data-experiment="${esc(e.path)}" tabindex="0">
        <td>${filed(e)}</td>
        <td class="ev">${plural(e.cases, 'case')} · ${plural((e.evaluators || []).length, 'evaluator')}</td>
        <td class="ev">${made.map(r => `${openLink('report', r.path, r.name)} ${fmt(r.overall_score)}`).join('<br>')
          || '<span class="empty">not run yet</span>'}</td>
        <td class="ev">${when(e.modified)}</td></tr>`;
  }).join('') || nothing('experiment', docLink('quickstart/#run-your-first-evaluation', 'empty-experiments', 'Create your first experiment')));

  $('skips').innerHTML = skipped.length
    ? `<details class="skips"><summary>${plural(skipped.length, 'file')} skipped</summary><ul>`
      + skipped.map(s => `<li><code>${esc(s.path)}</code> — ${esc(s.why)}</li>`).join('')
      + '</ul></details>'
    : '';
  renderPageHead(currentView());
}

app.querySelectorAll('[data-act="rescan"]').forEach(b => { b.onclick = () => rescan(true); });

/* ══════════ reading the folder, and opening what is in it ══════════ */

async function document_for(meta) {
  if (!meta) return null;
  if (meta.obj) return meta.obj;                       // standalone: already in memory
  const res = await api('/api/file?path=' + encodeURIComponent(meta.path));
  meta.modified = res.modified;                        // refresh the save guard's baseline
  if (res.evaluator_notes) meta.evaluator_notes = res.evaluator_notes;
  return res.document;
}

/* `keepOpen` re-reads the folder under an open page, so what is open in an optimizer run stays open. */
async function rescan(keepOpen) {
  let res;
  try {
    res = await api('/api/scan');
  } catch (err) {
    scanError = `could not scan: ${err.message}`;
    renderLists();
    return null;
  }
  scanError = '';
  ROOT = res.root; DEPTH = res.depth; READONLY = res.read_only;
  REPORTS = res.reports; EXPERIMENTS = res.experiments;
  skipped = res.skipped || []; truncated = !!res.truncated;
  treeMark = res.fingerprint;
  renderLists();

  await loadOptimizerRuns(res.optimizer_runs || [], keepOpen);
  return res;
}

/* A file that will not load says so on the lists, which is where the page goes back to. */
function openFailed(path, err) {
  scanError = `could not load ${path}: ${err.message}`;
  renderLists();
  show(homeView());
}

/** Open one report on its own page. The experiment editor, and any edits in it, are left alone. */
async function openReport(path) {
  const meta = REPORTS.find(r => r.path === path);
  if (!meta) return;
  let doc;
  try { doc = await document_for(meta); } catch (err) { return openFailed(path, err); }
  closeCandidate();
  loadReport(meta, doc);
  scanError = '';
  renderLists();
  show('report');
}

/** Open one experiment's cases. Asks first when that would drop unsaved edits to another one. */
async function openExperiment(path, opts) {
  const meta = EXPERIMENTS.find(e => e.path === path);
  if (!meta) return;
  /* Opening the one already open again would re-read it from disk and drop unsaved edits. */
  if (expMeta && expMeta.path === path && !(opts && opts.force)) return show('experiment');
  const dirty = countDirty();
  if (!(opts && opts.force) && dirty
      && !await ask(`${dirty} unsaved ${dirty === 1 ? 'change' : 'changes'} will be lost if you open another experiment.`,
        'Discard and open', 'Stay')) return;
  let doc;
  try { doc = await document_for(meta); } catch (err) { return openFailed(path, err); }
  loadExperiment(meta, doc);
  scanError = '';
  renderLists();
  show('experiment');
}

/* A row, or a link naming a report or an experiment anywhere on the page, opens it. */
app.addEventListener('click', e => {
  const named = e.target.closest('[data-report], [data-experiment]');
  if (!named) return;
  e.preventDefault();
  if (named.dataset.report != null) openReport(named.dataset.report);
  else openExperiment(named.dataset.experiment);
});
