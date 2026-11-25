'use strict';

/* ── what every chart is drawn with ──
   Hand-rolled SVG: the page must work with no network and no build step. */

/** The width to draw a chart at: the inside of its card, so its type comes out at the page's own size
    on a wide window and on a narrow one. Read off the content column, because the card itself may be
    hidden while it is drawn. `unlaid` is the width to use when the page is not laid out. */
function cardWidth(unlaid) {
  const main = app.querySelector('main'), style = getComputedStyle(main);
  const column = main.clientWidth - (parseFloat(style.paddingLeft) || 0) - (parseFloat(style.paddingRight) || 0);
  return column > 0 ? Math.max(280, Math.round(column - 54)) : unlaid;
}

/* `t` holds the axis title above the plot, `b` two rows under it: tick labels, then the axis name. */
const PAD = { t: 28, r: 16, b: 48, l: 50 };
const PLOT_W = 640;                       // when the page is not laid out
/* A chart with axes is `W` by `H`. `sized()` sets both for the card before one is drawn: as wide as
   the card, and tall enough to read without growing as fast as the card widens. */
let W = PLOT_W, H = 244, IN_W = W - PAD.l - PAD.r, IN_H = H - PAD.t - PAD.b, FLOOR = PAD.t + IN_H;
function sized() {
  W = cardWidth(PLOT_W);
  H = Math.round(Math.min(320, Math.max(228, W * 0.38)));
  IN_W = W - PAD.l - PAD.r;
  IN_H = H - PAD.t - PAD.b;
  FLOOR = PAD.t + IN_H;
}

/* Never stretched far past the size it was drawn for, however wide the card. */
const svg = inner => `<svg viewBox="0 0 ${W} ${H}" style="max-width:${Math.round(W * 1.15)}px" role="img">${inner}</svg>`;

/* An axis is named in a sentence's case, whatever case the name was written in. */
const axisName = s => String(s).charAt(0).toUpperCase() + String(s).slice(1);

/** Draw the charts again at the width the content column has now. A report's and a run's are drawn
    to the width they had when they were read, so anything that changes that width asks for this. */
function redrawCharts() {
  if (REP) renderReportCharts();
  if (OPT) renderOptimization();
}
/* A window being dragged to a new size settles before the charts follow it, and one that only
   changed height does not move them at all. */
let chartsAt = 0, resizing = 0;
window.addEventListener('resize', () => {
  clearTimeout(resizing);
  resizing = setTimeout(() => {
    const now = cardWidth(0);
    if (now === chartsAt) return;
    chartsAt = now;
    redrawCharts();
  }, 150);
});

/* Round steps (1, 2, 2.5 or 5 of a power of ten), with the domain widened out to them. */
function niceTicks(lo, hi, want) {
  if (!(isFinite(lo) && isFinite(hi))) return { lo: 0, hi: 1, step: 0.5, ticks: [0, 0.5, 1] };
  if (hi <= lo) { const pad = Math.abs(hi) * 0.1 || 1; lo = hi - pad; hi = hi + pad; }
  /* Divided by the tick count, not the gaps, so the smaller candidate step is chosen. */
  const n = Math.max(2, want || 5);
  const raw = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = ([1, 2, 2.5, 5, 10].find(m => m * mag >= raw * (1 - 1e-9)) || 10) * mag;
  const first = Math.floor(lo / step + 1e-9) * step;
  const last = Math.ceil(hi / step - 1e-9) * step;
  const ticks = [];
  for (let i = 0; first + i * step <= last + step * 1e-6; i++) {
    const v = first + i * step;
    // Avoids printing zero as "-0".
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return { lo: first, hi: last, step, ticks };
}

/** How many decimals a step needs, so 0.25 labels as "0.25" and 20000 does not label as "20000.00". */
const dpOf = step => {
  const s = step.toPrecision(12).replace(/0+$/, '');
  const dot = s.indexOf('.');
  return dot < 0 || /e/i.test(s) ? 0 : Math.min(6, s.length - dot - 1);
};
const tickText = (v, step) => Math.abs(v) >= 1e4 ? compact(v) : v.toFixed(dpOf(step));

/* `lo` greater than `hi` runs the axis backwards. */
function yAxis(lo, hi, label) {
  const t = niceTicks(Math.min(lo, hi), Math.max(lo, hi), 5);
  const [from, to] = lo > hi ? [t.hi, t.lo] : [t.lo, t.hi];
  const span = (to - from) || 1;
  const y = v => FLOOR - ((v - from) / span) * IN_H;
  let out = `<line class="ax" x1="${PAD.l}" y1="${PAD.t}" x2="${PAD.l}" y2="${FLOOR}"/>`;
  for (const v of t.ticks) {
    const yy = y(v);
    out += `<line class="grid-l" x1="${PAD.l}" y1="${yy}" x2="${PAD.l + IN_W}" y2="${yy}"/>`
      + `<text class="tick" x="${PAD.l - 7}" y="${yy + 3.5}" text-anchor="end">`
      + `${esc(tickText(v, t.step))}</text>`;
  }
  if (label) out += `<text class="axlab" x="2" y="${PAD.t - 12}">${esc(axisName(label))}</text>`;
  return { y, out, lo: from, hi: to };
}

/* `lo` greater than `hi` runs the axis backwards. */
function xAxis(lo, hi, label) {
  const back = lo > hi;
  const t = niceTicks(Math.min(lo, hi), Math.max(lo, hi), 5);
  const [from, to] = back ? [t.hi, t.lo] : [t.lo, t.hi];
  const span = (to - from) || 1;
  const x = v => PAD.l + ((v - from) / span) * IN_W;
  let out = `<line class="ax" x1="${PAD.l}" y1="${FLOOR}" x2="${PAD.l + IN_W}" y2="${FLOOR}"/>`;
  for (const v of t.ticks) {
    const xx = x(v);
    // Anchored by where it landed, not by its place in the list: the list may run right to left.
    const anchor = xx <= PAD.l + 1 ? 'start' : xx >= PAD.l + IN_W - 1 ? 'end' : 'middle';
    out += `<line class="grid-l" x1="${xx}" y1="${PAD.t}" x2="${xx}" y2="${FLOOR}"/>`
      + `<text class="tick" x="${xx}" y="${FLOOR + 15}" text-anchor="${anchor}">`
      + `${esc(tickText(v, t.step))}</text>`;
  }
  if (label) {
    out += `<text class="axlab" x="${PAD.l + IN_W / 2}" y="${H - 6}" text-anchor="middle">`
      + `${esc(axisName(label))}</text>`;
  }
  return { x, out, lo: from, hi: to };
}

const shortId = id => {
  const s = String(id == null ? '' : id);
  return s.length <= 8 ? s : s.slice(0, 6) + '…';
};

/* The SVG is built as a string, so text cannot be measured. Label widths are estimated:
   12px of the page's sans runs about 6.9px a glyph. */
const GLYPH = 6.9;
const labWidth = s => s.length * GLYPH + 8;

/** A figure written next to a mark, turned inward when the mark is near an edge. */
function flag(px, py, label) {
  const anchor = px > PAD.l + IN_W - 52 ? 'end' : px < PAD.l + 52 ? 'start' : 'middle';
  const dx = anchor === 'end' ? 4 : anchor === 'start' ? -4 : 0;
  const dy = py < PAD.t + 14 ? 15 : -9;      // under the mark when there is no room above it
  return `<text class="vlab" x="${px + dx}" y="${py + dy}" text-anchor="${anchor}">`
    + `${esc(label)}</text>`;
}

/* ── charts that are a list of labelled rows over one shared scale ── */
const ROW_W = 900;                        // when the page is not laid out
const ROWP = { row: 24, head: 26, lab: 200, right: 18 };
const CHART_ROWS = 30;                    // a row chart lists this many; its caption says how many it left out

/** The width to draw a row chart at: the inside of its card. */
const rowWidth = () => cardWidth(ROW_W);
/* The label column gives way on a narrow card. */
const labOf = width => Math.min(ROWP.lab, Math.round(width * 0.38));

const clipText = (s, room) => s.length > room ? s.slice(0, Math.max(1, room - 1)) + '…' : s;

/** A row's label, cut to the label column; the whole of it stays in the tooltip. `sub` is a dim suffix. */
function rowText(row, cy, lab) {
  const name = String(row.label), sub = row.sub == null ? '' : String(row.sub);
  const room = Math.floor((lab - 12) / GLYPH) - (sub ? sub.length + 1 : 0);
  return `<text class="rowlab" x="2" y="${cy}" dy="0.32em"><title>${esc(sub ? `${name} · ${sub}` : name)}</title>`
    + esc(clipText(name, room)) + (sub ? `<tspan class="rowsub"> ${esc(sub)}</tspan>` : '') + '</text>';
}

const rowSvg = (width, height, inner) =>
  `<svg viewBox="0 0 ${width} ${height}" style="max-width:${Math.round(width * 1.15)}px" role="img">${inner}</svg>`;

/* `rows` carry `label` and an optional `sub`. Returns the scale, each row's centre, and `frame`,
   which closes the SVG around the marks. */
function rowPlot(rows, lo, hi, { axis } = {}) {
  const width = rowWidth();
  const t = niceTicks(lo, hi, width < 600 ? 4 : 5);
  const left = labOf(width), right = width - ROWP.right;
  const span = (t.hi - t.lo) || 1;
  const x = v => left + ((v - t.lo) / span) * (right - left);
  const y = i => ROWP.head + i * ROWP.row + ROWP.row / 2;
  const floor = ROWP.head + rows.length * ROWP.row;
  let out = '';
  for (const v of t.ticks) {
    const xx = x(v);
    const anchor = xx <= left + 1 ? 'start' : xx >= right - 1 ? 'end' : 'middle';
    out += `<line class="grid-l" x1="${xx}" y1="${ROWP.head - 4}" x2="${xx}" y2="${floor}"/>`
      + `<text class="tick" x="${xx}" y="${ROWP.head - 8}" text-anchor="${anchor}">${esc(tickText(v, t.step))}</text>`;
  }
  rows.forEach((row, i) => { out += rowText(row, y(i), left); });
  if (axis) {
    out += `<text class="axlab" x="${(left + right) / 2}" y="${floor + 18}" text-anchor="middle">${esc(axisName(axis))}</text>`;
  }
  const frame = body => rowSvg(width, floor + (axis ? 26 : 6), out + body);
  return { x, y, left, right, lo: t.lo, hi: t.hi, frame };
}

/* The whole row answers the pointer, not only its marks. */
const rowHit = (p, i) =>
  `<rect class="hit" x="${p.left}" y="${p.y(i) - ROWP.row / 2}" width="${p.right - p.left}" height="${ROWP.row}"/>`;

/** Before and after on each row: `{ label, sub, from, to, tip, attrs }`. A row missing one end draws
    the end it has. `opts.better` is `lower` when a smaller value is the improvement. */
function dumbbells(rows, opts) {
  const values = rows.flatMap(r => [r.from, r.to]).filter(v => v != null);
  const p = rowPlot(rows, Math.min(...values), Math.max(...values), opts);
  return p.frame(rows.map((r, i) => {
    const cy = p.y(i);
    const moved = r.from != null && r.to != null && r.from !== r.to;
    const cls = !moved ? 'same' : (r.to > r.from) === (opts.better !== 'lower') ? 'up' : 'down';
    return `<g class="bell ${cls}"${r.attrs || ''}><title>${esc(r.tip)}</title>${rowHit(p, i)}`
      + (moved ? `<line class="bell-l" x1="${p.x(r.from)}" y1="${cy}" x2="${p.x(r.to)}" y2="${cy}"/>` : '')
      + (r.from != null ? `<circle class="bell-a" cx="${p.x(r.from)}" cy="${cy}" r="4"/>` : '')
      + (r.to != null ? `<circle class="bell-b" cx="${p.x(r.to)}" cy="${cy}" r="4.5"/>` : '') + '</g>';
  }).join(''));
}

/** One bar from zero on each row: `{ label, sub, value, text, tip, attrs }`; `text` is written at the bar's end. */
function barRows(rows, opts) {
  const p = rowPlot(rows, 0, Math.max(...rows.map(r => r.value)) || 1, opts);
  return p.frame(rows.map((r, i) => {
    const end = p.x(r.value);
    return `<g class="rbar"${r.attrs || ''}><title>${esc(r.tip)}</title>${rowHit(p, i)}`
      + `<rect class="col" x="${p.left}" y="${p.y(i) - 6}" width="${Math.max(1, end - p.left)}" height="12"/>`
      + `<text class="vlab" x="${end > p.right - 60 ? end - 6 : end + 6}" y="${p.y(i)}" dy="0.32em" `
      + `text-anchor="${end > p.right - 60 ? 'end' : 'start'}">${esc(r.text)}</text></g>`;
  }).join(''));
}

/** A histogram on each row: `{ label, sub, values }`. Every row counts on one scale, so heights compare
    between rows; the tick is the row's mean. */
function histRows(rows, opts) {
  const bins = 20;
  const all = rows.flatMap(r => r.values);
  const p = rowPlot(rows, Math.min(...all), Math.max(...all), opts);
  const span = (p.hi - p.lo) || 1, width = (p.right - p.left) / bins;
  const counted = rows.map(r => {
    const counts = new Array(bins).fill(0);
    for (const v of r.values) counts[Math.min(bins - 1, Math.floor(((v - p.lo) / span) * bins))] += 1;
    return counts;
  });
  const most = Math.max(1, ...counted.flat());
  return p.frame(rows.map((r, i) => {
    const base = p.y(i) + ROWP.row / 2 - 2;
    const bars = counted[i].map((n, b) => {
      if (!n) return '';
      const tall = Math.max(2, (n / most) * (ROWP.row - 5));
      const from = p.lo + (b / bins) * span, to = p.lo + ((b + 1) / bins) * span;
      return `<rect class="hbin" x="${p.left + b * width + 0.5}" y="${base - tall}" width="${width - 1}" height="${tall}">`
        + `<title>${esc(`${r.label} · ${fine(from)} to ${fine(to)} · ${plural(n, 'row')}`)}</title></rect>`;
    }).join('');
    if (!r.values.length) return bars;
    const average = mean(r.values), at = p.x(average);
    return bars + `<line class="hmean" x1="${at}" y1="${base - ROWP.row + 5}" x2="${at}" y2="${base + 2}">`
      + `<title>${esc(`${r.label} · mean ${fine(average)} over ${plural(r.values.length, 'row')}`)}</title></line>`;
  }).join(''));
}

/** A grid of cells. `rows` and `cols` are labels; `cell(r, c)` returns `{ cls, tip, attrs, mark }`, or
    null for a gap. A cell with `mark` carries a cross, so it is told from the rest by more than its
    colour. Column labels are written only while there is room for them: level where each fits under
    its own column, and turned where it does not. */
function heatmap(rows, cols, cell) {
  const width = rowWidth();
  const left = labOf(width), top = 4;
  const wide = (width - ROWP.right - left) / Math.max(1, cols.length);
  const gap = wide > 6 ? 2 : 0, labelled = wide >= 22;
  /* Level where every label fits under its own column, or the columns are wide enough that one cut
     to its column still says most of itself. Turned otherwise: a turned label has more room. */
  const room = Math.floor((wide - gap - 4) / GLYPH);
  const level = labelled && (wide >= 110 || cols.every(label => String(label).length <= room));
  const floor = top + rows.length * ROWP.row;
  const tall = ROWP.row - 2, arm = Math.min(4, (wide - gap) / 2 - 2, tall / 2 - 2);
  let out = '';
  rows.forEach((label, r) => {
    const y = top + r * ROWP.row;
    out += rowText({ label }, y + ROWP.row / 2, left);
    cols.forEach((_, c) => {
      const got = cell(r, c);
      if (!got) return;
      const x = left + c * wide, across = Math.max(1, wide - gap);
      out += `<rect class="cell ${got.cls}"${got.attrs || ''} x="${x}" y="${y + 1}" `
        + `width="${across}" height="${tall}"><title>${esc(got.tip)}</title></rect>`;
      if (got.mark && arm >= 2) {
        const cx = x + across / 2, cy = y + 1 + tall / 2;
        out += `<path class="cell-mark" d="M${cx - arm},${cy - arm}L${cx + arm},${cy + arm}`
          + `M${cx - arm},${cy + arm}L${cx + arm},${cy - arm}"/>`;
      }
    });
  });
  if (labelled) {
    cols.forEach((label, c) => {
      const x = left + (c + 0.5) * wide - gap / 2, y = floor + (level ? 16 : 12);
      out += level
        ? `<text class="catlab" x="${x}" y="${y}" text-anchor="middle">`
          + `<title>${esc(String(label))}</title>${esc(clipText(String(label), room))}</text>`
        : `<text class="catlab" transform="rotate(-35 ${x} ${y})" x="${x}" y="${y}" text-anchor="end">`
          + `<title>${esc(String(label))}</title>${esc(clipText(String(label), 18))}</text>`;
    });
  }
  return rowSvg(width, floor + (!labelled ? 6 : level ? 24 : 72), out);
}
