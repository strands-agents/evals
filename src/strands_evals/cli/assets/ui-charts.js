'use strict';

/* ── charts ──
   Hand-rolled SVG: the page must work with no server, no network and no build step. */

/* `t` holds the axis title above the plot, `b` two rows under it: tick labels, then the axis name. */
const W = 640, H = 244, PAD = { t: 26, r: 16, b: 46, l: 48 };
const IN_W = W - PAD.l - PAD.r, IN_H = H - PAD.t - PAD.b;
const FLOOR = PAD.t + IN_H;

const svg = inner => `<svg viewBox="0 0 ${W} ${H}" role="img">`
  + `<rect class="plot" x="${PAD.l}" y="${PAD.t}" width="${IN_W}" height="${IN_H}"/>${inner}</svg>`;

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
  if (label) out += `<text class="axlab" x="2" y="${PAD.t - 11}">${esc(label)}</text>`;
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
      + `<text class="tick" x="${xx}" y="${FLOOR + 14}" text-anchor="${anchor}">`
      + `${esc(tickText(v, t.step))}</text>`;
  }
  if (label) {
    out += `<text class="axlab" x="${PAD.l + IN_W / 2}" y="${H - 6}" text-anchor="middle">`
      + `${esc(label)}</text>`;
  }
  return { x, out, lo: from, hi: to };
}

/* ── the bottom of a chart whose x axis is a list, not a scale ── */
const shortId = id => {
  const s = String(id == null ? '' : id);
  return s.length <= 8 ? s : s.slice(0, 6) + '…';
};

/* The SVG is built as a string, so text cannot be measured. Label widths are estimated:
   10.5px of the page's sans runs about 6.1px a glyph. */
const GLYPH = 6.1;
const labWidth = s => s.length * GLYPH + 8;

function candLabel(p, slot) {
  const id = String(p.candidate_id == null ? '' : p.candidate_id);
  const num = p.number == null ? '' : `#${p.number}`;
  if (id && labWidth(id) <= slot) return id;
  if (num && num.length < id.length) return num;
  // An invalid proposal has neither an id nor a number.
  return (id ? shortId(id) : num) || '—';
}

function candAxis(items, x) {
  const slot = IN_W / Math.max(1, items.length);
  const labels = items.map(p => candLabel(p, slot));
  // Thinned by the widest label, so none overlap.
  const every = Math.max(1, Math.ceil(labels.reduce((m, s) => Math.max(m, labWidth(s)), 0) / slot));
  const groups = [];
  let out = '', lastAt = -every;
  items.forEach((p, i) => {
    const last = groups[groups.length - 1];
    if (last && last.iteration === p.iteration) last.to = i;
    else groups.push({ iteration: p.iteration, from: i, to: i });
    if (i - lastAt >= every) {
      lastAt = i;
      out += `<text class="catlab" x="${x(i)}" y="${FLOOR + 15}" text-anchor="middle">`
        + `${esc(labels[i])}</text>`;
    }
  });
  if (groups.length > 1) {
    out += `<text class="itlab" x="0" y="${FLOOR + 31}">iteration</text>`;
    groups.forEach((g, gi) => {
      if (gi) {
        const mid = (x(g.from) + x(groups[gi - 1].to)) / 2;
        out += `<line class="itsep" x1="${mid}" y1="${PAD.t}" x2="${mid}" y2="${FLOOR + 22}"/>`;
      }
      out += `<text class="itlab" x="${(x(g.from) + x(g.to)) / 2}" y="${FLOOR + 31}" `
        + `text-anchor="middle">${esc(g.iteration)}</text>`;
    });
  }
  return out;
}

/** A figure written next to a mark, turned inward when the mark is near an edge. */
function flag(px, py, label) {
  const anchor = px > PAD.l + IN_W - 52 ? 'end' : px < PAD.l + 52 ? 'start' : 'middle';
  const dx = anchor === 'end' ? 4 : anchor === 'start' ? -4 : 0;
  const dy = py < PAD.t + 14 ? 15 : -9;      // under the mark when there is no room above it
  return `<text class="vlab" x="${px + dx}" y="${py + dy}" text-anchor="${anchor}">`
    + `${esc(label)}</text>`;
}

function objectiveChart(d) {
  if (!d.scored.length) {
    return '<div class="empty">No candidate has been scored yet, so there is no objective to plot.</div>';
  }
  const values = d.scored.map(p => p.objective);
  let lo = Math.min(...values), hi = Math.max(...values);
  // Small: the axis is rounded out to whole steps below, which supplies the headroom.
  const pad = ((hi - lo) || Math.abs(hi) || 1) * 0.05;
  lo -= pad; hi += pad;

  const n = d.proposals.length;
  const x = i => PAD.l + (n === 1 ? IN_W / 2 : (i / (n - 1)) * IN_W);
  const { y, out: axis } = yAxis(lo, hi, 'objective — higher is better');

  let marks = '', best = null;
  d.proposals.forEach((p, i) => {
    if (p.objective == null) return;    // never plotted as zero: it was not measured
    const cls = p.candidate_id === d.champion ? 'champion'
      : p.outcome === 'feasible' || p.outcome === 'infeasible' ? p.outcome : 'ineligible';
    marks += `<circle class="pt ${cls}" data-cand="${esc(p.candidate_id)}" tabindex="0" cx="${x(i)}" cy="${y(p.objective)}" r="4">`
      + `<title>${esc(`${p.candidate_id} · ${p.outcome} · ${nice(p.objective)}`)}</title></circle>`;
    if (p.outcome === 'feasible' && (!best || p.objective > best.p.objective)) {
      best = { p, px: x(i), py: y(p.objective) };
    }
  });
  const xticks = candAxis(d.proposals, x)
    + (best ? flag(best.px, best.py,
      // Named the way the axis under it names candidates, so the flag and the tick agree.
      `${candLabel(best.p, IN_W / Math.max(1, d.proposals.length))} · ${nice(best.p.objective)}`) : '');

  /* A step, not a slope: nothing was measured between two candidates. */
  const steps = [];
  d.proposals.forEach((p, i) => {
    if (p.best == null) return;
    const px = x(i), py = y(p.best);
    if (steps.length) steps.push(`L${px},${steps[steps.length - 1].split(',')[1]}`);
    steps.push(`${steps.length ? 'L' : 'M'}${px},${py}`);
  });
  const line = steps.length > 1 ? `<path class="best" d="${steps.join(' ')}"/>` : '';

  return svg(axis + line + marks + xticks);
}

function budgetChart(d) {
  if (!d.iterations.length) return '<div class="empty">No iteration has finished yet.</div>';
  const cap = d.maxRollouts;
  const hi = Math.max(d.spent, cap || 0) || 1;
  const { y, out: axis } = yAxis(0, hi, 'rollouts run, cumulative');
  const pts = [{ iteration: '', cumulative: 0 }].concat(d.iterations);
  const n = pts.length;
  const x = i => PAD.l + (n === 1 ? 0 : (i / (n - 1)) * IN_W);

  const path = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i)},${y(p.cumulative)}`).join(' ');
  const area = `${path} L${x(n - 1)},${y(0)} L${x(0)},${y(0)} Z`;
  let marks = '';
  d.iterations.forEach((p, i) => {
    marks += `<circle class="pt data" cx="${x(i + 1)}" cy="${y(p.cumulative)}" r="3">`
      + `<title>${esc(`iteration ${p.iteration}: +${p.rollouts}, ${p.cumulative} total`)}</title></circle>`;
  });
  const last = d.iterations[d.iterations.length - 1];
  const spentFlag = last
    ? flag(x(d.iterations.length), y(last.cumulative),
      cap != null ? `${last.cumulative} of ${cap}` : `${last.cumulative}`)
    : '';
  const capLine = cap != null
    ? `<line class="cap-line" x1="${PAD.l}" y1="${y(cap)}" x2="${PAD.l + IN_W}" y2="${y(cap)}"/>`
      + `<text class="tick" x="${PAD.l + IN_W}" y="${y(cap) - 5}" text-anchor="end">`
      + `budget ${esc(cap)}</text>`
    : '';
  const xticks = d.iterations.map((p, i) =>
    `<text class="catlab" x="${x(i + 1)}" y="${FLOOR + 15}" text-anchor="middle">`
    + `${esc(p.iteration)}</text>`).join('')
    + `<text class="axlab" x="${PAD.l + IN_W / 2}" y="${H - 6}" text-anchor="middle">iteration</text>`;
  return svg(axis + `<path class="budget-fill" d="${area}"/><path class="budget-line" d="${path}"/>`
    + capLine + marks + xticks + spentFlag);
}

function metricChart(d, metric) {
  const values = d.proposals.map(p => p.measured[metric]).filter(v => v != null);
  if (!values.length) return '<div class="empty">Nothing measured this metric yet.</div>';
  let lo = Math.min(...values), hi = Math.max(...values);
  const pad = ((hi - lo) || Math.abs(hi) || 1) * 0.05;
  lo -= pad; hi += pad;

  const n = d.proposals.length;
  const x = i => PAD.l + (n === 1 ? IN_W / 2 : (i / (n - 1)) * IN_W);
  const goal = d.goals.find(g => g.metric === metric);
  const { y, out: axis } = yAxis(lo, hi,
    goal ? `${metric} — ${goal.dir === 'min' ? 'lower' : 'higher'} is better` : metric);

  /* Joined only between consecutive measured candidates. */
  let marks = '', segments = '', run = [], champ = null;
  const flush = () => {
    if (run.length > 1) segments += `<path class="metric-line" d="${run.join(' ')}"/>`;
    run = [];
  };
  d.proposals.forEach((p, i) => {
    const v = p.measured[metric];
    if (v == null) { flush(); return; }
    run.push(`${run.length ? 'L' : 'M'}${x(i)},${y(v)}`);
    const cls = p.candidate_id === d.champion ? 'champion'
      : p.outcome === 'infeasible' ? 'infeasible' : 'feasible';
    marks += `<circle class="pt ${esc(cls)}" data-cand="${esc(p.candidate_id)}" tabindex="0" cx="${x(i)}" cy="${y(v)}" r="3.5">`
      + `<title>${esc(`${p.candidate_id} · ${metric} ${nice(v)}`)}</title></circle>`;
    if (p.candidate_id === d.champion) champ = { px: x(i), py: y(v), v };
  });
  flush();
  const xticks = candAxis(d.proposals, x)
    + (champ ? flag(champ.px, champ.py, `champion · ${nice(champ.v)}`) : '');
  return svg(axis + segments + marks + xticks);
}

/* ── one goal against another ──
   Both axes are oriented so better is up and to the right, whichever way the goal wants. */

/** The measured value of a goal, signed so that bigger is always better. */
const oriented = (goal, p) => (goal.dir === 'min' ? -1 : 1) * p.measured[goal.metric];

/** The candidates no other candidate beats on both of these two goals. */
function front2(points, xg, yg) {
  return points.filter(a => !points.some(b => {
    const [bu, bv, au, av] = [oriented(xg, b), oriented(yg, b), oriented(xg, a), oriented(yg, a)];
    return bu >= au && bv >= av && (bu !== au || bv !== av);
  }));
}

/* The boundary of what a set of marks dominates, stepped because nothing was measured
   between two candidates. Only marks on the boundary get a step. */
function staircase(marks) {
  if (!marks.length) return null;
  const byX = marks.slice().sort((a, b) => b.px - a.px || a.py - b.py);
  const edge = [];
  let top = Infinity;
  for (const m of byX) if (m.py < top) { edge.push(m); top = m.py; }
  edge.reverse();
  let line = `M${PAD.l},${edge[0].py}`;
  edge.forEach((m, i) => {
    line += `H${m.px}V${i + 1 < edge.length ? edge[i + 1].py : PAD.t + IN_H}`;
  });
  return { line, region: `${line}H${PAD.l}Z`, on: edge.length };
}

function tradeoffChart(d, xg, yg, frontIds) {
  const plotted = d.points.filter(p =>
    p.measured[xg.metric] != null && p.measured[yg.metric] != null);
  if (plotted.length < 2) {
    return '<div class="empty">Fewer than two candidates have both goals measured, so there is '
      + 'nothing to trade off yet.</div>';
  }
  const span = goal => {
    const values = plotted.map(p => p.measured[goal.metric]);
    let lo = Math.min(...values), hi = Math.max(...values);
    const pad = ((hi - lo) || Math.abs(hi) || 1) * 0.04;
    lo -= pad; hi += pad;
    // Better to the right, and better up: a goal that wants less runs its axis backwards.
    return goal.dir === 'min' ? [hi, lo] : [lo, hi];
  };
  const label = (goal, towards) =>
    `${goal.dir === 'min' ? 'fewer' : 'higher'} ${goal.metric} is better — ${towards} this axis`;
  const { x, out: xax } = xAxis(...span(xg), label(xg, 'rightward along'));
  const { y, out: yax } = yAxis(...span(yg), label(yg, 'upward along'));

  const on = new Set(frontIds);
  const marks = plotted.map(p => ({
    p,
    px: x(p.measured[xg.metric]),
    py: y(p.measured[yg.metric]),
  }));
  const steps = staircase(marks.filter(m => on.has(m.p.candidate_id)));
  const region = steps ? `<path class="front-fill" d="${steps.region}"/>` : '';
  const line = steps ? `<path class="front-line" d="${steps.line}"/>` : '';

  const dots = marks.map(({ p, px, py }) => {
    const cls = p.isChampion ? 'champion' : p.isBaseline ? 'baseline'
      : p.outcome === 'infeasible' ? 'infeasible'
        : p.outcome === 'feasible' || p.outcome == null ? 'feasible' : 'ineligible';
    const ring = on.has(p.candidate_id)
      ? `<circle class="front-ring" cx="${px}" cy="${py}" r="7.5"/>` : '';
    const label = p.isBaseline ? ' · baseline' : p.isChampion ? ' · champion' : '';
    const link = d.proposals.some(x => x.candidate_id === p.candidate_id) ? ` data-cand="${esc(p.candidate_id)}" tabindex="0"` : '';
    return ring + `<circle class="pt ${cls}"${link} cx="${px}" cy="${py}" r="4">`
      + `<title>${esc(`${p.candidate_id}${label} · ${xg.metric} ${nice(p.measured[xg.metric])}`
        + ` · ${yg.metric} ${nice(p.measured[yg.metric])}`)}</title></circle>`;
  }).join('');

  return svg(yax + xax + region + line + dots);
}

/* ── which setting moved the metric ──
   Forest plot of `effects.json`: change from each key's reference level, with a 90% interval. */

const EFF = { row: 22, head: 20, lab: 152, right: 104 };

const signed = v => (v > 0 ? '+' : '') + fine(v);
/* `p_positive` is the probability the metric went up, so it is flipped for a goal that wants less. */
const pBetter = (e, better) =>
  better === 'lower' ? 1 - (e.p_positive || 0) : (e.p_positive || 0);

function effectsChart(ef, goal) {
  const rows = (ef.effects || []).filter(e => e && e.metric === goal.metric
    && Array.isArray(e.ci90) && e.ci90.length === 2 && typeof e.mean === 'number');
  if (!rows.length) {
    return '<div class="empty">The model estimated no effect for this metric.</div>';
  }
  const order = new Map((ef.factors || []).map((f, i) => [f.key, i]));
  const sorted = rows.slice().sort((a, b) =>
    (order.has(a.key) ? order.get(a.key) : 1e6) - (order.has(b.key) ? order.get(b.key) : 1e6));
  const groups = [];
  for (const e of sorted) {
    const last = groups[groups.length - 1];
    if (last && last.key === e.key) last.rows.push(e);
    else groups.push({ key: e.key, reference: e.reference, rows: [e] });
  }

  /* Zero is always in view: every bar is a change from the reference level. */
  const lo = Math.min(0, ...rows.map(e => e.ci90[0]));
  const hi = Math.max(0, ...rows.map(e => e.ci90[1]));
  const pad = ((hi - lo) || Math.abs(hi) || 1) * 0.08;
  const t = niceTicks(lo - pad, hi + pad, 5);
  const span = (t.hi - t.lo) || 1;
  const left = EFF.lab, right = W - EFF.right;
  const x = v => left + ((v - t.lo) / span) * (right - left);
  const floor = EFF.head + groups.reduce((n, g) => n + g.rows.length + 1, 0) * EFF.row + 4;
  const height = floor + 18;

  let ticks = '';
  for (const v of t.ticks) {
    const xx = x(v);
    const anchor = xx <= left + 1 ? 'start' : xx >= right - 1 ? 'end' : 'middle';
    ticks += `<line class="grid-l" x1="${xx}" y1="${EFF.head - 4}" x2="${xx}" y2="${floor}"/>`
      + `<text class="tick" x="${xx}" y="${EFF.head - 8}" text-anchor="${anchor}">`
      + `${esc(tickText(v, t.step))}</text>`;
  }
  ticks += `<line class="eff-zero" x1="${x(0)}" y1="${EFF.head - 4}" x2="${x(0)}" y2="${floor}"/>`
    + `<text class="eff-foot" x="${x(0)}" y="${floor + 13}" text-anchor="middle">no change</text>`
    + `<text class="eff-foot" x="${right + 6}" y="${EFF.head - 8}">change · P(better)</text>`;

  let body = '', y = EFF.head;
  for (const g of groups) {
    body += `<text class="eff-key" x="2" y="${y + EFF.row / 2}" dy="0.32em">${esc(g.key)}`
      + `<tspan class="eff-ref"> vs ${esc(g.reference)}</tspan></text>`;
    y += EFF.row;
    for (const e of g.rows) {
      const cy = y + EFF.row / 2;
      y += EFF.row;
      const settled = e.ci90[0] > 0 || e.ci90[1] < 0;
      const p = pBetter(e, goal.better);
      body += `<g class="eff-row ${settled ? 'settled' : 'unsettled'}">`
        + `<title>${esc(`${e.key} = ${e.level} vs ${g.reference}: ${signed(e.mean)}`
          + ` [${fine(e.ci90[0])} – ${fine(e.ci90[1])}], P(better) ${p.toFixed(2)}`)}</title>`
        + `<text class="eff-level" x="14" y="${cy}" dy="0.32em">${esc(e.level)}</text>`
        + `<line class="eff-ci" x1="${x(e.ci90[0])}" y1="${cy}" x2="${x(e.ci90[1])}" y2="${cy}"/>`
        + `<circle class="eff-dot" cx="${x(e.mean)}" cy="${cy}" r="4"/>`
        + `<text class="tick" x="${right + 6}" y="${cy}" dy="0.32em">`
        + `${esc(signed(e.mean))} · ${esc(p.toFixed(2))}</text></g>`;
    }
  }
  return `<svg viewBox="0 0 ${W} ${height}" role="img">${ticks}${body}</svg>`;
}

function usageChart(d, key, label) {
  if (!d.iterations.length) return '<div class="empty">No iteration has finished yet.</div>';
  const values = d.iterations.map(it => it.usage[key] || 0);
  const hi = Math.max(...values) || 1;
  const { y, out: axis } = yAxis(0, hi, `${label.toLowerCase()} spent in that iteration`);
  const n = d.iterations.length;
  const slot = IN_W / n;
  const width = Math.min(34, slot * 0.6);
  const labelled = slot > 36;

  let bars = '', xticks = '';
  d.iterations.forEach((it, i) => {
    const centre = PAD.l + slot * (i + 0.5);
    const value = it.usage[key] || 0;
    const top = y(value);
    bars += `<rect class="col" x="${centre - width / 2}" y="${top}" width="${width}" `
      + `height="${Math.max(0, FLOOR - top)}">`
      + `<title>${esc(`iteration ${it.iteration}: ${nice(value)} over ${plural(it.usage.trials, 'trial')}`)}`
      + `</title></rect>`;
    if (labelled) {
      bars += `<text class="vlab" x="${centre}" y="${top - 6}" text-anchor="middle">`
        + `${esc(compact(value))}</text>`;
    }
    xticks += `<text class="catlab" x="${centre}" y="${FLOOR + 15}" text-anchor="middle">`
      + `${esc(it.iteration)}</text>`;
  });
  xticks += `<text class="axlab" x="${PAD.l + IN_W / 2}" y="${H - 6}" text-anchor="middle">iteration</text>`;
  return svg(axis + bars + xticks);
}

/* A status the record schema does not name must not become a class name. */
const statusClass = s => STATUSES.includes(s) ? s : 'other';

function statusChart(d) {
  if (!d.trials) return '<div class="empty">No trial has been recorded yet.</div>';
  const present = STATUSES.filter(s => d.statuses[s]);
  const other = Object.keys(d.statuses).filter(s => !STATUSES.includes(s));
  const bars = present.concat(other).map(s => {
    const pctWidth = (d.statuses[s] / d.trials) * 100;
    return `<span class="${statusClass(s)}" style="width:${pctWidth.toFixed(2)}%" title="${esc(s)}: ${esc(d.statuses[s])}"></span>`;
  }).join('');
  const legend = present.concat(other).map(s =>
    `<span><i class="${statusClass(s)}"></i>`
    + `${esc(s.replace(/_/g, ' '))} ${esc(d.statuses[s])}</span>`).join('');
  const splits = Object.keys(d.bySplit).sort()
    .map(s => `${esc(s)} ${d.bySplit[s]}`).join(' · ');
  return `<div class="bars">${bars}</div><div class="legend">${legend}</div>`
    + (splits ? `<div class="legend"><span>${splits}</span></div>` : '');
}
