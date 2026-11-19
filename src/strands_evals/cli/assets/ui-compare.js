'use strict';

/* ── reports side by side ──
   Any two or more reports, oldest first. Rows are matched by case and evaluator, so reports of
   different experiments compare only where they hold the same rows. */

/* The reports being compared, oldest first: `{ meta, rows, at, dated }` each. */
let CMP = [];
/* The reports ticked in the list, by path. */
const cmpPick = new Set();
const GRID_ROWS = 60;                     // the grid draws this many rows; its caption says how many it left out

/* A report's rows as `[key, row]`, the key its case and evaluator. Rows that share both are told apart
   by their order, so the second of them is matched with the second in another report. */
function cmpKeyed(rows) {
  const seen = new Map();
  return rows.map(r => {
    const key = `${r.name}\u0000${evaluatorName(r)}`, nth = seen.get(key) || 0;
    seen.set(key, nth + 1);
    return [nth ? `${key}\u0000${nth}` : key, r];
  });
}
const cmpNo = i => `#${i + 1}`;

/** How the rows two reports share moved from `before` to `after`, matched by case and evaluator. A row
    judged only in `after` counts by its verdict there; one judged only in `before` has `lapsed`. */
function compareRows(before, after) {
  const passed = r => r.na ? null : !!r.pass;
  const was = new Map(cmpKeyed(before));
  const out = { fixed: 0, broke: 0, same: 0, lapsed: 0, added: 0, gone: 0, moved: [] };
  for (const [key, r] of cmpKeyed(after)) {
    const old = was.get(key);
    if (!old) { out.added += 1; continue; }
    was.delete(key);
    const then = passed(old), now = passed(r);
    const flip = now === then || now === null ? null : now ? 'fixed' : 'broke';
    const lapsed = now === null && then !== null;
    out[flip || (lapsed ? 'lapsed' : 'same')] += 1;
    const from = old.na ? null : old.score, to = r.na ? null : r.score;
    if ((flip || from !== to) && (from != null || to != null)) out.moved.push({ row: r, from, to, flip, lapsed });
  }
  out.gone = was.size;
  return out;
}

/* A report made by an SDK that records the run says when it ran; an older one has only its file's time. */
async function comparable(meta) {
  const doc = await document_for(meta);
  const started = Date.parse(((doc && doc.run_info) || {}).start_time);
  const dated = Number.isFinite(started);
  return { meta, rows: reportRows(doc), at: dated ? started : (meta.modified || 0) * 1000, dated };
}

/** Compare these reports. Fewer than two that are in the folder is not a comparison. `quiet` loads
    without changing the view, for an address being followed. */
async function openCompare(paths, quiet) {
  const metas = [...new Set(paths)].map(path => REPORTS.find(r => r.path === path)).filter(Boolean);
  CMP = [];
  if (metas.length < 2) return false;
  const read = [];
  for (const meta of metas) {
    const one = await readToOpen(meta.path, () => comparable(meta));
    if (!one) return false;
    read.push(one);
  }
  setCompared(read);
  if (!quiet) show('compare');
  return true;
}

function setCompared(read) {
  CMP = read.sort((a, b) => a.at - b.at || a.meta.path.localeCompare(b.meta.path));
  renderCompare();
}

/** Re-read the comparison after the folder changed. A report that is gone leaves it, and one that
    cannot be read leaves the comparison as it was. */
async function refreshCompare() {
  if (!CMP.length) return;
  const was = CMP, now = was.map(c => REPORTS.find(m => m.path === c.meta.path)).filter(Boolean);
  if (now.length === was.length && now.every((m, i) => m.modified === was[i].meta.modified)) return;
  const read = now.length < 2 ? [] : await Promise.all(now.map(comparable));
  if (CMP !== was) return;                   // another comparison was opened while this one was read
  setCompared(read);
  if (!read.length && currentView() === 'compare') show('reports');
}

/* ── the page ── */

const cmpDelta = (to, from) => {
  if (to == null || from == null || to === from) return '';
  return `<span class="delta ${to > from ? 'up' : 'down'}">${to > from ? '▲ +' : '▼ '}${esc(fmt(to - from))}</span>`;
};

function compareSummary() {
  return CMP.map((c, i) => {
    const n = verdictCounts(c.rows);
    const before = i ? CMP[i - 1].meta.overall_score : null;
    return `<tr><td class="mono">${cmpNo(i)}</td><td>${openLink('report', c.meta.path)}`
      + `<br><span class="path">${when(c.at / 1000)}${c.dated ? '' : ' · by file time'}</span></td>`
      + `<td>${scoreBar(c.meta.overall_score)}</td><td>${cmpDelta(c.meta.overall_score, before) || '<span class="path">—</span>'}</td>`
      + `<td class="n">${c.rows.length}</td><td class="n">${n.pass}</td><td class="n">${n.fail}</td></tr>`;
  }).join('');
}

/** Each evaluator's mean over the rows it applied to, report by report: `{ names, means }`. */
function evaluatorMeans() {
  const means = CMP.map(c => new Map([...scoresByEvaluator(c.rows)].map(([name, scores]) => [name, mean(scores)])));
  return { names: [...new Set(means.flatMap(m => [...m.keys()]))], means };
}

function compareEvaluators() {
  const { names, means } = evaluatorMeans();
  const head = `<tr><th data-term="evaluator">Evaluator</th>`
    + `${CMP.map((c, i) => `<th data-tip="${esc(c.meta.path)}">${cmpNo(i)}</th>`).join('')}</tr>`;
  const body = names.map(name => `<tr><td>${esc(name)}</td>` + means.map((m, i) => {
    const earlier = means.slice(0, i).reverse().find(x => x.has(name));
    return `<td>${m.has(name) ? `${fmt(m.get(name))} ${cmpDelta(m.get(name), earlier ? earlier.get(name) : null)}`
      : '<span class="path" data-tip="This report has no row from this evaluator.">—</span>'}</td>`;
  }).join('') + '</tr>').join('');
  return `<thead>${head}</thead><tbody>${body}</tbody>`;
}

/* Every case-and-evaluator row against every report. The rows that changed come first. */
function compareGrid() {
  const held = CMP.map(c => new Map(cmpKeyed(c.rows)));
  const keys = [...new Set(held.flatMap(m => [...m.keys()]))];
  const rowAt = k => held.find(m => m.has(k)).get(k);
  const several = new Set(keys.map(k => evaluatorName(rowAt(k)))).size > 1;
  const label = k => (several ? rowTitle(rowAt(k)) : rowAt(k).name);
  const changed = k => new Set(held.map(m => (m.has(k) ? verdictOf(m.get(k)) : 'gap'))).size > 1;
  const moved = keys.filter(changed);
  const drawn = moved.concat(keys.filter(k => !changed(k))).slice(0, GRID_ROWS);
  const tabbable = drawn.length * CMP.length <= TAB_MARKS;
  $('cmp-grid-cap').textContent = `${moved.length} of ${plural(keys.length, 'row')} changed verdict or `
    + `${moved.length === 1 ? 'is' : 'are'} missing from a report, and come first. A gap means that report does not hold the row.`
    + (keys.length > drawn.length ? ` The first ${drawn.length} are drawn.` : '');
  return heatmap(drawn.map(label), CMP.map((_, i) => cmpNo(i)), (r, c) => {
    const row = held[c].get(drawn[r]);
    return row && {
      cls: verdictOf(row),
      tip: `${rowTitle(row)} · ${cmpNo(c)} ${CMP[c].meta.path} · ${verdictText(row)}`,
      attrs: rowLink(row, tabbable, CMP[c].meta.path),
    };
  });
}

/** The oldest against the newest, the rows that flipped first and then those whose score moved furthest. */
function compareEndRows() {
  const c = compareRows(CMP[0].rows, CMP[CMP.length - 1].rows);
  c.moved.sort((a, b) => !!b.flip - !!a.flip
    || Math.abs((b.to || 0) - (b.from || 0)) - Math.abs((a.to || 0) - (a.from || 0)));
  return c;
}

/* The oldest against the newest: what flipped, what moved, and what only one of them holds. */
function compareEnds() {
  const last = CMP.length - 1;
  const c = compareEndRows(), moved = c.moved;
  const several = new Set(CMP[last].rows.map(evaluatorName)).size > 1;
  const shared = c.fixed + c.broke + c.same + c.lapsed;
  $('cmp-ends-head').textContent = `From ${cmpNo(0)} to ${cmpNo(last)}`;
  $('cmp-ends-cap').textContent = `${CMP[0].meta.path} against ${CMP[last].meta.path}. `
    + (!shared ? 'The two share no case and evaluator.'
      : !moved.length ? `All ${plural(shared, 'row')} they share scored the same.`
        : `${plural(moved.length, 'row')} changed score or verdict`
          + (moved.length > CHART_ROWS ? `; the ${CHART_ROWS} that moved most are drawn.` : '.'));
  $('cmp-ends').innerHTML = moved.length ? dumbbells(moved.slice(0, CHART_ROWS).map(m => ({
    label: m.row.name,
    sub: several ? evaluatorName(m.row) : '',
    from: m.from, to: m.to,
    tip: `${rowTitle(m.row)} · ${fmt(m.from)} to ${fmt(m.to)}`
      + (m.flip ? ` · now ${m.flip === 'fixed' ? 'passes' : 'fails'}` : m.lapsed ? ' · now not applicable' : ''),
    attrs: rowLink(m.row, true, CMP[last].meta.path),
  })), { axis: 'score' }) : '';
  $('cmp-ends-legend').innerHTML = [
    shared ? `<span><i class="up"></i>${c.fixed} now pass</span>` : '',
    shared ? `<span><i class="down"></i>${c.broke} now fail</span>` : '',
    shared ? `<span><i class="same"></i>${c.same} kept their verdict</span>` : '',
    c.lapsed ? `<span>${c.lapsed} now not applicable</span>` : '',
    c.added ? `<span>${c.added} only in ${cmpNo(last)}</span>` : '',
    c.gone ? `<span>${c.gone} only in ${cmpNo(0)}</span>` : '',
  ].join('');
}

function renderCompare() {
  if (!CMP.length) return;
  $('cmp-sub').textContent = `${plural(CMP.length, 'report')}, oldest first. Rows are matched by case and evaluator, `
    + 'so reports of different experiments compare only where they hold the same rows.';
  $('cmp-summary').innerHTML = compareSummary();
  $('cmp-evals').innerHTML = compareEvaluators();
  $('cmp-grid').innerHTML = compareGrid();
  compareEnds();
}

/* ── the header: what is compared, and how to change it ── */

function compareHead() {
  const removable = CMP.length > 2;
  return [[['Reports', 'reports'], ['Compare']], CMP.map((c, i) =>
    `<span class="chip"><span class="ord">${cmpNo(i)}</span> ${esc(c.meta.path)}`
    + (removable ? `<button class="x" type="button" data-cmp-remove="${esc(c.meta.path)}" `
      + `aria-label="Remove ${esc(c.meta.path)} from the comparison">&times;</button>` : '') + '</span>').join(' ')];
}

/* One picker in the header: from a report it starts a comparison, in a comparison it adds a report. */
function renderCompareAction(view) {
  const on = view === 'compare' ? CMP.length > 0 : view === 'report' && !!repMeta && !repCand;
  const taken = view === 'compare' ? CMP.map(c => c.meta.path) : on ? [repMeta.path] : [];
  const others = REPORTS.filter(m => !taken.includes(m.path));
  $('cmp-with').hidden = !on || !others.length;
  /* The header is redrawn whenever the folder is re-read: a list someone has open is left alone. */
  const offer = [view, ...taken, '', ...others.map(m => m.path)].join('\n');
  if ($('cmp-with').hidden || $('cmp-with').dataset.offer === offer) return;
  fillPick('cmp-with', [{ value: '', label: view === 'compare' ? 'Add a report…' : 'Compare with…' },
    ...others.map(m => ({ value: m.path, label: m.path }))], '', true);
  $('cmp-with').dataset.offer = offer;
  $('cmp-with').dataset.from = taken.join('\n');
}

$('cmp-with').onchange = e => {
  if (e.target.value) openCompare(e.target.dataset.from.split('\n').concat(e.target.value));
};

$('pagehead').addEventListener('click', e => {
  const remove = e.target.closest('[data-cmp-remove]');
  if (remove) openCompare(CMP.map(c => c.meta.path).filter(path => path !== remove.dataset.cmpRemove));
});

/* ── picking reports in the list ── */

function renderComparePick() {
  for (const path of [...cmpPick]) if (!REPORTS.some(r => r.path === path)) cmpPick.delete(path);
  $('cmp-go').hidden = cmpPick.size < 2;
  $('cmp-hint').hidden = cmpPick.size >= 2 || REPORTS.length < 2;
  $('cmp-go').textContent = `Compare (${cmpPick.size})`;
}

/* The box sits in a row that opens its report when selected, so the tick stops there. */
$('reports-rows').addEventListener('click', e => {
  const box = e.target.closest('input[data-pick]');
  if (!e.target.closest('.pickbox')) return;
  e.stopPropagation();
  if (!box) return;
  if (box.checked) cmpPick.add(box.dataset.pick); else cmpPick.delete(box.dataset.pick);
  renderComparePick();
});
$('cmp-go').onclick = () => openCompare([...cmpPick]);
