'use strict';

/* What the folder holds: every report and every experiment file, each report knowing the
   experiment it was made from. A report made in code has none, and one experiment can have many. */
let REPORTS = [], EXPERIMENTS = [], skipped = [], truncated = false;
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
const FROM_CODE = '<span class="empty" data-tip="No experiment file in this folder holds these cases, as when the '
  + 'report was made with run_evaluations() in code and saved with to_file().">run from code</span>';
/* The experiment a report was made from, as its list row and its page header both say it. With none,
   a link says why the scan could not decide; no link is none found. */
const madeFrom = r => (r.experiment ? `${openLink('experiment', r.experiment)} ${linkTag(r.link)}`
  : r.link ? `<span class="empty" data-tip="${esc(r.link.detail)}. A tie is not guessed.">not paired</span>` : FROM_CODE);

/* The reports made from an experiment that the folder still holds. */
const reportsOf = e => e.reports.map(path => REPORTS.find(r => r.path === path)).filter(Boolean);
const experimentSize = e => `${plural(e.cases, 'case')} · ${plural((e.evaluators || []).length, 'evaluator')}`;

const filed = m => `${m.dir && m.dir !== '.' ? `<span class="dim">${esc(m.dir)} /</span> ` : ''}${esc(m.name)}`
  + `<br><span class="path">${KB(m.size)}</span>`;

/* ── the files, under the views that name them ──

   Like the optimizer runs under Runs: every report and every experiment is listed again right under
   its view, so opening one is a click from any page rather than a trip through the list first. The
   lists stay there and the rows above them still open them, with the columns a sidebar row has no
   space for. Two files that share a name are told apart by the whole path. */

/* The file whose page is showing, if a page of one is: the list views show all of them and mark
   none, and a candidate read as a report is a page of its run, not of a report file. */
const openReportPath = () => (currentView() === 'report' && !repCand && repMeta ? repMeta.path : null);
const openExperimentPath = () => (currentView() === 'experiment' && expMeta ? expMeta.path : null);

/** Draw one tree. Nothing is drawn before a file is found, and the caret that folds it waits too.
    No tooltip on a row: anchored under the row, it would cover the rows beneath it. */
function renderFileTree(id, kind, metas, open) {
  const names = metas.map(m => m.name);
  const twice = new Set(names.filter((n, i) => names.indexOf(n) !== i));
  $(id).hidden = !metas.length;
  app.querySelector(`.fold[data-fold="${id}"]`).hidden = !metas.length;
  redraw($(id), metas.map(m => {
    const on = open === m.path;
    return `<a class="runrow${on ? ' on' : ''}" data-${kind}="${esc(m.path)}" tabindex="0" role="link"`
      + `${on ? ' aria-current="true"' : ''}>${esc(twice.has(m.name) ? m.path : m.name)}</a>`;
  }).join(''));
}

function renderFileTrees() {
  renderFileTree('reports-tree', 'report', REPORTS, openReportPath());
  renderFileTree('experiments-tree', 'experiment', EXPERIMENTS, openExperimentPath());
}

/** Move the marks to the files the page is showing. Called on every view change, which is cheaper
    than redrawing and is all that changes. A list holding the open file unfolds: however the file
    was opened, the mark is never left hidden in a folded list. */
function markFileTrees() {
  const mark = (id, key, path) => {
    if (path != null) setFolded(id, false);
    for (const el of $(id).querySelectorAll('a.runrow')) {
      const here = path != null && el.dataset[key] === path;
      el.classList.toggle('on', here);
      if (here) el.setAttribute('aria-current', 'true');
      else el.removeAttribute('aria-current');
    }
  };
  mark('reports-tree', 'report', openReportPath());
  mark('experiments-tree', 'experiment', openExperimentPath());
}

function renderLists() {
  $('read-only').hidden = !READONLY;
  const from = `Everything ${DEPTH != null ? DEPTH + ' levels ' : ''}under the directory the CLI was started in.`;
  $('reports-sub').textContent = `${from} A report is the result of one run, whether it was run from an experiment file or in code.`;
  $('experiments-sub').textContent = `${from} An experiment file holds the cases and evaluators a run uses.`;

  const capped = truncated ? ' · <span class="tag warn">scan hit the file cap</span>' : '';
  $('reports-count').innerHTML = plural(REPORTS.length, 'report') + capped;
  $('experiments-count').innerHTML = plural(EXPERIMENTS.length, 'experiment') + capped;
  const failed = scanError ? `<tr><td colspan="4"><span class="tag warn">${esc(scanError)}</span></td></tr>` : '';
  const nothing = (what, link) => (failed ? '' : blank(`No ${what}s in this pond yet`, `No ${what} JSON under ${esc(ROOT)}. ${link}`));

  redraw($('reports-rows'), failed + (REPORTS.map(r => {
    const open = !repCand && repMeta && repMeta.path === r.path;
    return `<tr class="row${open ? ' on' : ''}" data-report="${esc(r.path)}" tabindex="0">
        <td>${filed(r)}</td>
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
      + skipped.map(s => `<li><code>${esc(s.path)}</code>: ${esc(s.why)}</li>`).join('')
      + '</ul></details>'
    : '';
  renderFileTrees();
  renderPageHead(currentView());
}

/* ══════════ reading the folder, and opening what is in it ══════════ */

/* A file as the server holds it now: `{ document, modified, evaluator_notes }`. */
const readFile = meta => api('/api/file?path=' + encodeURIComponent(meta.path));
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

/* Counts the reads begun to open something: of two asked for in a row, the later is the one wanted. */
let opening = 0;

/* What `read` loads for the file at `path`, or null when it would not load or the page moved on while it did. */
async function readToOpen(path, read) {
  const from = moves, mine = ++opening;
  const wanted = () => moves === from && opening === mine;
  try {
    const got = await read();
    return wanted() ? got : null;
  } catch (err) {
    if (wanted()) openFailed(path, err);
    return null;
  }
}

/* Whether the open experiment's unsaved edits may be dropped: asked in the page when there are any. */
async function mayDiscard(doing) {
  const dirty = countDirty();
  return !dirty || ask(`${plural(dirty, 'unsaved change')} will be lost if you ${doing}.`, 'Discard and open', 'Stay');
}

/** Open one report on its own page. The experiment editor, and any edits in it, are left alone.
    Like the other opens, resolves false when the file never arrived: it would not load, or the page moved on first. */
async function openReport(path) {
  const meta = REPORTS.find(r => r.path === path);
  const doc = meta && await readToOpen(path, () => document_for(meta));
  if (!doc) return false;
  repCand = null;
  loadReport(meta, doc);
  scanError = '';
  renderLists();
  show('report');
  return true;
}

/** Open one experiment's cases. Asks first when that would drop unsaved edits to another one, and
    staying with those is not a file that never arrived. */
async function openExperiment(path, opts) {
  const meta = EXPERIMENTS.find(e => e.path === path);
  if (!meta) return false;
  const force = opts && opts.force;
  /* Opening the one already open again would re-read it from disk and drop unsaved edits. */
  if (expMeta && expMeta.path === path && !force) { show('experiment'); return true; }
  if (!force && !await mayDiscard('open another experiment')) return true;
  const read = await readToOpen(path, () => readFile(meta));
  if (!read) return false;
  loadExperimentFile(meta, read);
  scanError = '';
  renderLists();
  show('experiment');
  return true;
}

/* A row, or a link naming a report or an experiment anywhere on the page, opens it. */
app.addEventListener('click', e => {
  const named = e.target.closest('[data-report], [data-experiment]');
  if (!named) return;
  e.preventDefault();
  if (named.dataset.report != null) openReport(named.dataset.report);
  else openExperiment(named.dataset.experiment);
});
