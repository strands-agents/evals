'use strict';

/* What the folder holds: every report and every experiment file, each report knowing the
   experiment it was made from. A report made in code has none, and one experiment can have many. */
let REPORTS = [], EXPERIMENTS = [], skipped = [], truncated = false, folderLabel = '';
let showingSample = false;
let scanError = '';                          // why the last scan or open failed, if it did
let treeMark = null;                         // the server's fingerprint of the tree when it was last scanned

/* ══════════ the two lists ══════════ */

const blank = (title, body) => `<tr><td colspan="4">${emptyState(title, body)}</td></tr>`;

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

/* The reports made from an experiment that the folder still holds. */
const reportsOf = e => e.reports.map(path => REPORTS.find(r => r.path === path)).filter(Boolean);
const experimentSize = e => `${plural(e.cases, 'case')} · ${plural((e.evaluators || []).length, 'evaluator')}`;

const filed = m => `${m.dir && m.dir !== '.' ? `<span class="dim">${esc(m.dir)} /</span> ` : ''}${esc(m.name)}`
  + `<br><span class="path">${KB(m.size)}</span>`;

function renderLists() {
  $('mode-tag').textContent = MODE === 'server'
    ? (READONLY ? 'served · read-only' : 'served · saves to disk')
    : 'standalone file · saves download';
  $('mode-tag').className = 'mode ' + (MODE === 'server' && !READONLY ? 'live' : 'ro');
  $('mode-tag').dataset.tip = MODE !== 'server' ? 'Opened without a server, so nothing can be written back. Save downloads a copy.'
    : READONLY ? 'Served with --read-only. Cases can be read but not saved.'
      : 'Served by strands-evals ui. Save writes the experiment file in place.';
  $('root-path').textContent = ROOT || folderLabel;
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
  const nothing = (what, link) => (failed ? '' : blank(`No ${what}s in this pond yet`, MODE === 'server'
    ? `No ${what} JSON under ${esc(ROOT)}. ${link}` : 'Open a folder to look for experiments and reports.'));

  redraw($('reports-rows'), failed + (REPORTS.map(r => {
    const open = !repCand && repMeta && repMeta.path === r.path;
    return `<tr class="row${open ? ' on' : ''}" data-report="${esc(r.path)}" tabindex="0">
        <td><label class="pickbox"><input type="checkbox" data-pick="${esc(r.path)}"${cmpPick.has(r.path) ? ' checked' : ''}
          aria-label="Compare ${esc(r.path)}"></label><span>${filed(r)}</span></td>
        <td class="ev">${scoreBar(r.overall_score)} · ${plural(r.rows, 'row')}`
      + (r.failing ? ` · <span class="vb fail"><i></i>${r.failing} failing</span>` : '') + `</td>
        <td class="ev">${madeFrom(r)}</td>
        <td class="ev">${when(r.modified)}</td></tr>`;
  }).join('') || nothing('report', docLink('cli/run/#experiment-file-run', 'empty-reports', 'How to run an experiment'))));

  redraw($('experiments-rows'), failed + (EXPERIMENTS.map(e =>
    `<tr class="row${expMeta && expMeta.path === e.path ? ' on' : ''}" data-experiment="${esc(e.path)}" tabindex="0">
        <td>${filed(e)}</td>
        <td class="ev">${experimentSize(e)}</td>
        <td class="ev">${reportsOf(e).map(r => `${openLink('report', r.path, r.name)} ${fmt(r.overall_score)}`).join('<br>')
          || '<span class="empty">not run yet</span>'}</td>
        <td class="ev">${when(e.modified)}</td></tr>`).join('')
    || nothing('experiment', docLink('quickstart/#run-your-first-evaluation', 'empty-experiments', 'Create your first experiment'))));

  $('skips').innerHTML = skipped.length
    ? `<details class="skips"><summary>${plural(skipped.length, 'file')} skipped</summary><ul>`
      + skipped.map(s => `<li><code>${esc(s.path)}</code> — ${esc(s.why)}</li>`).join('')
      + '</ul></details>'
    : '';
  renderComparePick();
  renderPageHead(currentView());
}

/* ══════════ reading the folder, and opening what is in it ══════════ */

/* A file as the server holds it now: `{ document, modified, evaluator_notes }`. Standalone, the copy in memory. */
const readFile = async meta => (meta.obj ? { document: meta.obj }
  : api('/api/file?path=' + encodeURIComponent(meta.path)));
const document_for = async meta => (await readFile(meta)).document;

/* Put a file as read into the editor. Save is checked against the time it had when read, so that time
   moves only here: a read that is dropped must leave it alone. */
function loadExperimentFile(meta, read) {
  if (read.modified != null) meta.modified = read.modified;
  if (read.evaluator_notes) meta.evaluator_notes = read.evaluator_notes;
  loadExperiment(meta, read.document);
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
  setSessions(res);
  renderLists();

  await loadOptimizerRuns(res.optimizer_runs || [], keepOpen);
  return res;
}

/* A file that will not load says so on a page of its own, at the address that names the file. */
function openFailed(path, err) {
  const list = REPORTS.some(r => r.path === path) ? 'reports' : EXPERIMENTS.some(e => e.path === path) ? 'experiments' : 'sessions';
  showMissing(`/${list}${idPath(path)}`, list, 'This one would not surface',
    `${named(path)} could not be read: ${esc(err.message)}`, [listWay(list)]);
}

/* What `read` loads for the file at `path`, or null when it would not load or the page moved on while it did. */
async function readToOpen(path, read) {
  const from = moves;
  try {
    const got = await read();
    return moves === from ? got : null;
  } catch (err) {
    if (moves === from) openFailed(path, err);
    return null;
  }
}

/* Whether the open experiment's unsaved edits may be dropped: asked in the page when there are any. */
async function mayDiscard(doing) {
  const dirty = countDirty();
  return !dirty || ask(`${plural(dirty, 'unsaved change')} will be lost if you ${doing}.`, 'Discard and open', 'Stay');
}

/** Open one report on its own page. The experiment editor, and any edits in it, are left alone. */
async function openReport(path) {
  const meta = REPORTS.find(r => r.path === path);
  const doc = meta && await readToOpen(path, () => document_for(meta));
  if (!doc) return;
  repCand = null;
  loadReport(meta, doc);
  scanError = '';
  renderLists();
  show('report');
}

/** Open one experiment's cases. Asks first when that would drop unsaved edits to another one. */
async function openExperiment(path, opts) {
  const meta = EXPERIMENTS.find(e => e.path === path);
  if (!meta) return;
  const force = opts && opts.force;
  /* Opening the one already open again would re-read it from disk and drop unsaved edits. */
  if (expMeta && expMeta.path === path && !force) return show('experiment');
  if (!force && !await mayDiscard('open another experiment')) return;
  const read = await readToOpen(path, () => readFile(meta));
  if (!read) return;
  loadExperimentFile(meta, read);
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
