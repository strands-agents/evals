'use strict';

/* ── the result ── */

/* Each row is scaled to its own interval, so widths do not compare across rows. Only
   intervals with width are drawn: `[149, 149]` has no spread to show. */
const RS = { row: 24, head: 22, lab: 112, right: 112, foot: 32 };

/** The goals the objective expression states plus the metrics the constraints name. */
function aimedMetrics(d) {
  const f = d.facts || {};
  const set = new Set([...(f.goalMetrics || []), ...(f.goals || []).map(g => g.metric)]);
  for (const con of ((d.config || {}).constraints || [])) {
    const named = String(con).match(/[A-Za-z_][\w.]*/);
    if (named) set.add(named[0]);
  }
  return set;
}

function resultStrip(d) {
  const r = d.result;
  if (!r) return '';
  /* `d.goals` holds only goals the trials measured. Directions here come from the record's goals. */
  const goalDir = m => ((((d.facts || {}).goals || []).find(g => g.metric === m) || {}).dir) || null;
  const rows = [];
  /* A `null` bound is no interval reported, and `ci_low === ci_high` has nothing to draw. */
  const interval = o => o && typeof o.ci_low === 'number' && typeof o.ci_high === 'number'
    && o.ci_high > o.ci_low;
  for (const m of r.metrics) {
    if (!interval(m.diff)) continue;      // only the goal metrics get a `difference` entry
    rows.push({
      label: m.metric,
      dir: goalDir(m.metric),
      lo: m.diff.ci_low,
      hi: m.diff.ci_high,
      mean: typeof m.diff.mean_diff === 'number' ? m.diff.mean_diff : (m.diff.ci_low + m.diff.ci_high) / 2,
    });
  }
  /* On a single-goal run the objective is that metric reported twice: drawn once. */
  const o = r.objective;
  if (interval(o) && typeof o.mean_diff === 'number'
    && !rows.some(x => x.lo === o.ci_low && x.hi === o.ci_high && x.mean === o.mean_diff)) {
    rows.push({ label: 'Objective', dir: 'max', lo: o.ci_low, hi: o.ci_high, mean: o.mean_diff, objective: true });
  }
  if (!rows.length) return '';

  const drawn = new Set(rows.map(x => x.label));
  const aimed = aimedMetrics(d);
  const missing = r.metrics.filter(x => aimed.has(x.metric) && !drawn.has(x.metric));
  const missWhy = m => `${m.metric} (${m.diff && typeof m.diff.ci_low === 'number'
    && typeof m.diff.ci_high === 'number' ? 'the same on every case' : 'no interval reported'})`;

  const left = RS.lab, right = W - RS.right;
  const zero = (left + right) / 2;
  const half = (right - left) / 2 - 10;
  const height = RS.head + rows.length * RS.row + RS.foot;

  let body = '';
  rows.forEach((row, i) => {
    const cy = RS.head + i * RS.row + RS.row / 2;
    const m = Math.max(Math.abs(row.lo), Math.abs(row.hi), Math.abs(row.mean)) || 1;
    const x = v => zero + (v / m) * half;
    const crosses = row.lo <= 0 && row.hi >= 0;
    const kind = crosses ? 'flat'
      : !row.dir ? 'plain'
        : (row.lo > 0) === (row.dir === 'max') ? 'good' : 'bad';
    const reading = crosses
      ? 'crosses no change, so this run has not shown it moved'
      : kind === 'plain'
        ? 'clear of no change, in a metric the objective states no direction for'
        : kind === 'good' ? 'clear of no change, the way the goal wants' : 'clear of no change, the wrong way';
    const verdict = crosses ? 'not shown'
      : kind === 'plain' ? 'changed' : kind === 'good' ? 'better' : 'worse';
    const arrow = row.dir ? `<tspan class="rs-dir"> ${row.dir === 'max' ? '↑' : '↓'}</tspan>` : '';
    body += `<g class="rs-row ${kind}${row.objective ? ' rs-obj' : ''}">`
      + `<title>${esc(`${row.label}: ${signed(row.mean)} [${fine(row.lo)}, ${fine(row.hi)}] — ${reading}`)}`
      + `</title>`
      + `<text class="rs-lab" x="0" y="${cy}" dy="0.32em">${esc(row.label)}${arrow}</text>`
      + `<line class="rs-ci" x1="${x(row.lo)}" y1="${cy}" x2="${x(row.hi)}" y2="${cy}"/>`
      + `<circle class="rs-dot" cx="${x(row.mean)}" cy="${cy}" r="4"/>`
      + `<text class="rs-read" x="${W - RS.right + 14}" y="${cy}" dy="0.32em">${esc(verdict)}</text>`
      + '</g>';
  });

  return `<div class="rstrip${r.heldOut ? '' : ' weak'}">`
    + `<svg viewBox="0 0 ${W} ${height}" role="img" aria-label="the change in each goal metric, `
    + 'against a line at no change">'
    + `<line class="rs-zero" x1="${zero}" y1="${RS.head - 5}" x2="${zero}" y2="${height - RS.foot + 3}"/>`
    + `<text class="rs-note" x="${zero}" y="${RS.head - 9}" text-anchor="middle">no change</text>`
    + `<text class="rs-note" x="0" y="${height - 17}">`
    + `${r.heldOut ? 'champion against baseline, on the held-out cases' : 'champion against baseline'}`
    + `${missing.length ? esc(` · nothing to draw for ${missing.map(missWhy).join(', ')}`) : ''}`
    + '</text>'
    + `<text class="rs-note" x="0" y="${height - 4}">each bar is the 95% interval of the change,`
    + ' scaled to its own row — so widths do not compare</text>'
    + body + '</svg></div>';
}

function resultCard(d) {
  if (!d.result) {
    const why = d.stop
      ? 'This run recorded no end-phase report, so there is no champion to compare.'
      : 'No candidate has been picked yet — the comparison appears once the run reaches its end phase.';
    const far = d.best == null ? '' : ` Best feasible objective so far: <b>${esc(fmt(d.best))}</b>`
      + `${d.bestCandidate ? ` (${esc(d.bestCandidate.candidate_id)})` : ''}, on the training split.`;
    return `<p class="rlabel">No result yet</p><p class="verdict">${why}${far}</p>`;
  }

  const r = d.result;
  const label = r.heldOut
    ? ['Result · held out', `${plural(r.cases || 0, 'test case')} kept out of the search`]
    : ['Result · validation split', 'the search chose on these cases, so this is weaker evidence'];
  /* One number format per row: the value needing the most decimals sets all of them. */
  const rowFmt = vals => {
    const nums = vals.filter(v => typeof v === 'number' && isFinite(v));
    if (!nums.length) return fine;
    if (nums.some(v => Math.abs(v) >= 10)) return nice;     // a token count is not a 4dp figure
    const dp = Math.max(...nums.map(v => {
      const dot = String(Number(v.toFixed(4))).indexOf('.');
      return dot < 0 ? 0 : String(Number(v.toFixed(4))).length - dot - 1;
    }));
    return v => v.toFixed(dp);
  };
  const row = m => {
    const delta = m.before != null && m.after != null ? m.after - m.before : null;
    const ciLow = m.diff && typeof m.diff.ci_low === 'number' ? m.diff.ci_low : null;
    const n = rowFmt([m.before, m.after, delta, ciLow, ciLow == null ? null : m.diff.ci_high]);
    const ci = ciLow == null ? '' : `[${n(ciLow)}, ${n(m.diff.ci_high)}]`;
    // The ratio keeps `fine`: it is on its own scale, not the metric's.
    const ratio = m.ratio && typeof m.ratio.mean_diff === 'number' ? `${fine(m.ratio.mean_diff)}×` : '';
    return `<tr><th>${esc(m.metric)}</th>`
      + `<td class="num">${m.before == null ? '—' : esc(n(m.before))}</td>`
      + `<td class="arrow">→</td><td class="num">${m.after == null ? '—' : esc(n(m.after))}</td>`
      + `<td class="num delta flat">${delta == null ? '' : (delta > 0 ? '+' : '') + esc(n(delta))}</td>`
      + `<td class="ci">${esc(ci)}</td><td class="num ci">${esc(ratio)}</td></tr>`;
  };
  const aimed = aimedMetrics(d);
  const lead = r.metrics.filter(m => aimed.has(m.metric));
  const rest = lead.length ? r.metrics.filter(m => !aimed.has(m.metric)) : [];
  const head = '<thead><tr><th>Metric</th><th>Baseline</th><th></th><th>Champion</th>'
    + '<th>Change</th><th title="of the change, resampled over the cases">95% interval</th>'
    + '<th title="champion ÷ baseline">× baseline</th></tr></thead>';
  const table = `<table class="rtab">${head}<tbody>`
    + `${(lead.length ? lead : r.metrics).map(row).join('')}</tbody>`
    + (rest.length ? `<tbody class="rest" id="opt-rest" hidden>${rest.map(row).join('')}</tbody>` : '')
    + '</table>'
    + (rest.length
      ? `<button class="rmore" data-act="opt-rest">+ ${plural(rest.length, 'metric')} `
        + 'the run was not aiming at</button>'
      : '');

  const o = r.objective;
  let verdict;
  if (!o || typeof o.mean_diff !== 'number') {
    verdict = 'This report records no overall comparison against the baseline — only the per-metric '
      + 'means above — so how much of the gap is real cannot be read off it.';
  } else if (typeof o.ci_low !== 'number' || typeof o.ci_high !== 'number') {
    verdict = `Objective <b>${esc(signed(o.mean_diff))}</b> against the baseline, with no interval `
      + 'in the report — so how certain that is cannot be read off it.';
  } else if (o.ci_high === o.ci_low) {
    /* Every case reported the same difference: that is no interval, not a narrow one. */
    verdict = `Objective <b>${esc(signed(o.mean_diff))}</b> against the baseline — the same on every `
      + `one of ${plural(r.cases || 0, 'test case')}, so its 95% interval has no width and says `
      + 'nothing about how far this would carry to other cases.';
  } else {
    const span = `95% interval [${esc(fine(o.ci_low))}, ${esc(fine(o.ci_high))}]`;
    const reading = o.ci_low > 0
      ? `clear of zero${r.heldOut ? ' on cases the search never saw' : ''}.`
      : o.ci_high < 0
        ? 'below zero, so shipping the champion would be a regression.'
        : 'includes zero, so <b>this run has not shown an improvement</b>. More cases or more '
          + 'runs per case would narrow it.';
    verdict = `Objective <b>${esc(signed(o.mean_diff))}</b>, ${span} — ${reading}`;
  }
  if (!r.heldOut && r.baselineId) {
    verdict += ' The gap is likely flattering.';
  }
  /* A `picked` report only carries figures for the finalists, and the baseline is rarely one. */
  if (!r.baselineId) {
    verdict += ' It has no figures for the baseline, so the champion\'s numbers stand alone —'
      + ' there is nothing here to compare them against.';
  }

  return `<p class="rlabel">${esc(label[0])}<span class="rq">${esc(label[1])}</span></p>`
    + `<p class="verdict lead">${verdict}</p>` + table + resultStrip(d);
}

/** `a → b` for short values, a two-line diff for anything long or multi-line. */
function settingDiff(s) {
  /* A sampled float arrives at full precision, e.g. 0.6932267914762588. */
  const asText = v => v === undefined ? '(unset)'
    : typeof v === 'string' ? v
      : typeof v === 'number' ? fine(v) : JSON.stringify(v);
  const before = asText(s.before), after = asText(s.after);
  const long = [before, after].some(v => v.length > 60 || v.includes('\n'));
  if (!s.changed) return `<div class="inline">${esc(after)}</div>`;
  if (!long) {
    return `<div class="inline">${s.before === undefined ? '' : `<span class="was">${esc(before)}</span> → `}`
      + `${esc(after)}</div>`;
  }
  return `<pre>${s.before === undefined ? '' : `<span class="del">- ${esc(before)}</span>\n`}`
    + `<span class="ins">+ ${esc(after)}</span></pre>`;
}

/* ── the winner ── */
function winnerCard(d) {
  if (!d.winner) {
    return '<p class="wlabel">Champion configuration</p>'
      + '<p class="verdict">No champion has been picked yet.</p>';
  }
  const w = d.winner;
  const chain = w.lineage.length > 1
    ? `<span class="lineage">${esc(w.lineage.join(' ← '))}</span>` : '';
  const head = '<p class="wlabel">Champion configuration</p>'
    + `<div><span class="wid">${esc(w.id)}</span>${chain}</div>`;

  if (!w.settings) {
    return head + '<p class="verdict">The log records no values for this candidate, so its '
      + 'configuration cannot be shown. Only a <code>propose</code> event carries them.</p>';
  }
  const changed = w.settings.filter(s => s.changed);
  const same = w.settings.length - changed.length;
  const body = changed.map(s =>
    `<div class="setting"><span class="skey">${esc(s.key)}</span>`
    + `${w.baselineId ? '' : ' <span class="tag">no baseline to compare</span>'}`
    + settingDiff(s) + '</div>').join('')
    || `<p class="unchanged">Nothing differs from ${esc(w.baselineId || 'the baseline')} — the run `
      + 'ended with the configuration it started from.</p>';
  const unchanged = same
    ? `<p class="unchanged">${plural(same, 'other setting')} unchanged from `
      + `${esc(w.baselineId || 'the baseline')}.</p>` : '';
  return head + body + unchanged
    + '<div class="wacts"><button class="btn" data-act="copy-winner">Copy configuration</button>'
    + '<button class="btn" data-act="save-winner">Download champion.json</button>'
    + '<span class="said" id="opt-copied"></span></div>';
}
