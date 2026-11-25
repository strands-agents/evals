'use strict';

/* ── the optimizer's charts ── */

/* ── the bottom of a chart whose x axis is a list, not a scale ── */
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
    /* Beside the axis, where the scale's own figures are, so it never runs into the first group's number. */
    out += `<text class="itlab" x="${PAD.l - 10}" y="${FLOOR + 31}" text-anchor="end">iteration</text>`;
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

function objectiveChart(d) {
  if (!d.scored.length) {
    return '<div class="empty">No candidate has been scored yet, so there is no objective to plot.</div>';
  }
  sized();
  const values = d.scored.map(p => p.objective);
  let lo = Math.min(...values), hi = Math.max(...values);
  // Small: the axis is rounded out to whole steps below, which supplies the headroom.
  const pad = ((hi - lo) || Math.abs(hi) || 1) * 0.05;
  lo -= pad; hi += pad;

  const n = d.proposals.length;
  const x = i => PAD.l + (n === 1 ? IN_W / 2 : (i / (n - 1)) * IN_W);
  const { y, out: axis } = yAxis(lo, hi, 'objective, higher is better');

  let marks = '';
  d.proposals.forEach((p, i) => {
    if (p.objective == null) return;    // never plotted as zero: it was not measured
    const cls = p.candidate_id === d.champion ? 'champion'
      : p.outcome === 'feasible' || p.outcome === 'infeasible' ? p.outcome : 'ineligible';
    marks += `<circle class="pt ${cls}" data-cand="${esc(p.candidate_id)}" tabindex="0" cx="${x(i)}" cy="${y(p.objective)}" r="4">`
      + `<title>${esc(`${p.candidate_id} · ${p.outcome} · ${nice(p.objective)}`)}</title></circle>`;
  });
  const lead = d.bestCandidate;
  const xticks = candAxis(d.proposals, x)
    + (lead ? flag(x(d.proposals.indexOf(lead)), y(lead.objective),
      /* Named the way the axis under it names candidates, so the flag and the tick agree. On a card
         too narrow for both, the figure alone: the name is on the axis under the mark. */
      W < 420 ? nice(lead.objective) : `${candLabel(lead, IN_W / Math.max(1, n))} · ${nice(lead.objective)}`) : '');

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
  sized();
  const cap = d.maxRollouts;
  const hi = Math.max(d.spent, cap || 0) || 1;
  const { y, out: axis } = yAxis(0, hi, 'rollouts run, cumulative');
  /* The curve starts from nothing spent, a point before the first iteration, and ends with the end phase. */
  const last = d.iterations[d.iterations.length - 1].cumulative;
  const steps = d.spent > last
    ? d.iterations.concat({ iteration: 'end', rollouts: d.spent - last, cumulative: d.spent })
    : d.iterations;
  const pts = [{ cumulative: 0 }].concat(steps);
  const n = pts.length;
  const x = i => PAD.l + (i / (n - 1)) * IN_W;

  const path = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i)},${y(p.cumulative)}`).join(' ');
  const area = `${path} L${x(n - 1)},${y(0)} L${x(0)},${y(0)} Z`;
  let marks = '';
  steps.forEach((p, i) => {
    const name = p.iteration === 'end' ? 'the end phase' : `iteration ${p.iteration}`;
    marks += `<circle class="pt data" cx="${x(i + 1)}" cy="${y(p.cumulative)}" r="3">`
      + `<title>${esc(`${name}: +${p.rollouts}, ${p.cumulative} total`)}</title></circle>`;
  });
  const spentFlag = flag(x(n - 1), y(d.spent), cap != null ? `${d.spent} of ${cap}` : `${d.spent}`);
  const capLine = cap != null
    ? `<line class="cap-line" x1="${PAD.l}" y1="${y(cap)}" x2="${PAD.l + IN_W}" y2="${y(cap)}"/>`
      + `<text class="tick" x="${PAD.l + IN_W}" y="${y(cap) - 5}" text-anchor="end">`
      + `budget ${esc(cap)}</text>`
    : '';
  const xticks = steps.map((p, i) =>
    `<text class="catlab" x="${x(i + 1)}" y="${FLOOR + 15}" text-anchor="middle">`
    + `${esc(p.iteration)}</text>`).join('')
    + `<text class="axlab" x="${PAD.l + IN_W / 2}" y="${H - 6}" text-anchor="middle">Iteration</text>`;
  return svg(axis + `<path class="budget-fill" d="${area}"/><path class="budget-line" d="${path}"/>`
    + capLine + marks + xticks + spentFlag);
}

function metricChart(d, metric) {
  const values = d.proposals.map(p => p.measured[metric]).filter(v => v != null);
  if (!values.length) return '<div class="empty">Nothing measured this metric yet.</div>';
  sized();
  let lo = Math.min(...values), hi = Math.max(...values);
  const pad = ((hi - lo) || Math.abs(hi) || 1) * 0.05;
  lo -= pad; hi += pad;

  const n = d.proposals.length;
  const x = i => PAD.l + (n === 1 ? IN_W / 2 : (i / (n - 1)) * IN_W);
  const dir = goalDir(d, metric);
  const { y, out: axis } = yAxis(lo, hi, dir ? `${metric}, ${dir === 'min' ? 'lower' : 'higher'} is better` : metric);

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
    marks += `<circle class="pt ${cls}" data-cand="${esc(p.candidate_id)}" tabindex="0" cx="${x(i)}" cy="${y(v)}" r="3.5">`
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

/* `plotted` is the candidates measured on both goals, `frontIds` the ones to ring and step through. */
function tradeoffChart(d, plotted, xg, yg, frontIds) {
  if (plotted.length < 2) {
    return '<div class="empty">Fewer than two candidates have both goals measured, so there is '
      + 'nothing to trade off yet.</div>';
  }
  sized();
  const span = goal => {
    const values = plotted.map(p => p.measured[goal.metric]);
    let lo = Math.min(...values), hi = Math.max(...values);
    const pad = ((hi - lo) || Math.abs(hi) || 1) * 0.04;
    lo -= pad; hi += pad;
    // Better to the right, and better up: a goal that wants less runs its axis backwards.
    return goal.dir === 'min' ? [hi, lo] : [lo, hi];
  };
  const label = (goal, towards) =>
    `${goal.dir === 'min' ? 'fewer' : 'higher'} ${goal.metric} is better, ${towards} this axis`;
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

const EFF = { row: 24, head: 22, lab: 230, right: 124 };
/* A key is written in the mono face: 12px of it runs 7.2px a glyph. */
const MONO_GLYPH = 7.2;

const signed = v => (v > 0 ? '+' : '') + fine(v);
/* `p_positive` is the probability the metric went up, so it is flipped for a goal that wants less. */
const pBetter = (e, better) =>
  better === 'lower' ? 1 - (e.p_positive || 0) : (e.p_positive || 0);

/** The estimates of `effects.json` for one metric that have a mean and an interval to draw. */
const effectRows = (ef, metric) => ef.effects.filter(e => e && e.metric === metric
  && Array.isArray(e.ci90) && e.ci90.length === 2 && typeof e.mean === 'number');

function effectsChart(ef, goal) {
  const rows = effectRows(ef, goal.metric);
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

  sized();
  /* Zero is always in view: every bar is a change from the reference level. */
  const lo = Math.min(0, ...rows.map(e => e.ci90[0]));
  const hi = Math.max(0, ...rows.map(e => e.ci90[1]));
  const pad = ((hi - lo) || Math.abs(hi) || 1) * 0.08;
  const t = niceTicks(lo - pad, hi + pad, 5);
  const span = (t.hi - t.lo) || 1;
  /* The label column gives way on a narrow card, and what does not fit it is cut: the whole is in the tooltip. */
  const left = Math.min(EFF.lab, Math.round(W * 0.36)), right = W - EFF.right;
  const x = v => left + ((v - t.lo) / span) * (right - left);
  const floor = EFF.head + groups.reduce((n, g) => n + g.rows.length + 1, 0) * EFF.row + 4;
  const height = floor + 20;
  const keyRoom = Math.floor((left - 6) / MONO_GLYPH), levelRoom = Math.floor((left - 20) / GLYPH);

  let ticks = '';
  for (const v of t.ticks) {
    const xx = x(v);
    const anchor = xx <= left + 1 ? 'start' : xx >= right - 1 ? 'end' : 'middle';
    ticks += `<line class="grid-l" x1="${xx}" y1="${EFF.head - 4}" x2="${xx}" y2="${floor}"/>`
      + `<text class="tick" x="${xx}" y="${EFF.head - 8}" text-anchor="${anchor}">`
      + `${esc(tickText(v, t.step))}</text>`;
  }
  ticks += `<line class="eff-zero" x1="${x(0)}" y1="${EFF.head - 4}" x2="${x(0)}" y2="${floor}"/>`
    + `<text class="eff-foot" x="${x(0)}" y="${floor + 14}" text-anchor="middle">no change</text>`
    + `<text class="eff-foot" x="${right + 8}" y="${EFF.head - 8}">change · P(better)</text>`;

  let body = '', y = EFF.head;
  for (const g of groups) {
    const key = clipText(String(g.key), keyRoom);
    const against = clipText(` vs ${g.reference}`, Math.max(0, keyRoom - key.length));
    body += `<text class="eff-key" x="2" y="${y + EFF.row / 2}" dy="0.32em">`
      + `<title>${esc(`${g.key} vs ${g.reference}`)}</title>${esc(key)}`
      + `<tspan class="eff-ref">${esc(against.length > 4 ? against : '')}</tspan></text>`;
    y += EFF.row;
    for (const e of g.rows) {
      const cy = y + EFF.row / 2;
      y += EFF.row;
      const settled = e.ci90[0] > 0 || e.ci90[1] < 0;
      const p = pBetter(e, goal.better);
      body += `<g class="eff-row ${settled ? 'settled' : 'unsettled'}">`
        + `<title>${esc(`${e.key} = ${e.level} vs ${g.reference}: ${signed(e.mean)}`
          + ` [${fine(e.ci90[0])} – ${fine(e.ci90[1])}], P(better) ${p.toFixed(2)}`)}</title>`
        + `<text class="eff-level" x="14" y="${cy}" dy="0.32em">${esc(clipText(String(e.level), levelRoom))}</text>`
        + `<line class="eff-ci" x1="${x(e.ci90[0])}" y1="${cy}" x2="${x(e.ci90[1])}" y2="${cy}"/>`
        + `<circle class="eff-dot" cx="${x(e.mean)}" cy="${cy}" r="4"/>`
        + `<text class="tick" x="${right + 8}" y="${cy}" dy="0.32em">`
        + `${esc(signed(e.mean))} · ${esc(p.toFixed(2))}</text></g>`;
    }
  }
  return `<svg viewBox="0 0 ${W} ${height}" style="max-width:${Math.round(W * 1.15)}px" role="img">${ticks}${body}</svg>`;
}

function usageChart(d, key, label) {
  if (!d.iterations.length) return '<div class="empty">No iteration has finished yet.</div>';
  sized();
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
  xticks += `<text class="axlab" x="${PAD.l + IN_W / 2}" y="${H - 6}" text-anchor="middle">Iteration</text>`;
  return svg(axis + bars + xticks);
}

/* A status the record schema does not name must not become a class name. */
const statusClass = s => STATUSES.includes(s) ? s : 'other';

function statusChart(d) {
  if (!d.trials) return '<div class="empty">No trial has been recorded yet.</div>';
  /* The schema's statuses in its order, then any other the log holds. */
  const found = STATUSES.filter(s => d.statuses[s]).concat(Object.keys(d.statuses).filter(s => !STATUSES.includes(s)));
  /* Each part grows by its share, with a gap between two and a width no part falls under: one trial
     in a thousand is still there to see, and no two parts touch. */
  const bars = found.map(s => `<span class="${statusClass(s)}" style="flex-grow:${esc(d.statuses[s])}"`
    + ` title="${esc(s)}: ${esc(d.statuses[s])}"></span>`).join('');
  const legend = found.map(s =>
    `<span><i class="${statusClass(s)}"></i>`
    + `${esc(s.replace(/_/g, ' '))} ${esc(d.statuses[s])}</span>`).join('');
  const splits = Object.keys(d.bySplit).sort()
    .map(s => `${esc(s)} ${d.bySplit[s]}`).join(' · ');
  return `<div class="bars">${bars}</div><div class="legend">${legend}</div>`
    + (splits ? `<div class="legend"><span>${splits}</span></div>` : '');
}

/* ── case by case: the champion against the baseline ── */

/** What both candidates' trials can be compared on: each evaluator's score, then what a trial cost. */
function caseMeasures(d) {
  const trials = [d.baselineId, d.champion].flatMap(id => d.trialsBy.get(id) || []);
  const number = v => (typeof v === 'number' ? v : null);
  const measures = [...new Set(trials.flatMap(t => Object.keys(t.evaluations || {})))].map(metric => ({
    value: 'm:' + metric, label: metric, axis: metric,
    better: goalDir(d, metric) === 'min' ? 'lower' : 'higher',
    of: t => number(scoreOf((t.evaluations || {})[metric])),
  }));
  for (const [key, label] of [['tokens', 'Tokens'], ['seconds', 'Seconds']]) {
    if (!trials.some(t => (t.usage || {})[key] > 0)) continue;
    measures.push({ value: 'u:' + key, label, axis: key, better: 'lower', of: t => number((t.usage || {})[key]) });
  }
  return measures;
}

/** `{ name, split, from, to }` per case: the mean of a measure under the baseline and under the champion,
    over the trials that completed. Held-out cases come first. */
function caseRows(d, measure) {
  const means = id => {
    const by = new Map();
    for (const t of d.trialsBy.get(id) || []) {
      const v = trialFailed(t) ? null : measure.of(t);
      if (v == null) continue;
      const key = `${t.split || ''}\u0000${t.case_name}`;
      by.set(key, (by.get(key) || []).concat(v));
    }
    return new Map([...by].map(([key, vs]) => [key, vs.reduce((a, b) => a + b, 0) / vs.length]));
  };
  const before = means(d.baselineId), after = means(d.champion);
  const rank = split => (['test', 'validation', 'train'].indexOf(split) + 1) || 4;
  return [...new Set([...before.keys(), ...after.keys()])].map(key => {
    const [split, name] = key.split('\u0000');
    return { name, split, from: before.has(key) ? before.get(key) : null, to: after.has(key) ? after.get(key) : null };
  }).sort((a, b) => rank(a.split) - rank(b.split) || a.name.localeCompare(b.name));
}

/** Of the cases both candidates ran: how many there are, and how many moved the way the measure calls better or worse. */
function caseMoves(rows, measure) {
  const moved = rows.filter(r => r.from != null && r.to != null && r.to !== r.from);
  const better = moved.filter(r => (r.to > r.from) === (measure.better !== 'lower')).length;
  return { both: rows.filter(r => r.from != null && r.to != null).length, better, worse: moved.length - better };
}
