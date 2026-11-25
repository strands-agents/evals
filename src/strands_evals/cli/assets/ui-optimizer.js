'use strict';

let OPT_RUNS = [], optSel = -1, OPT = null;
/* The run the reader last had a page of open, by id. The page keeps one run selected from the start,
   so that every view has a run to draw: that is not the reader having opened it. */
let optSeen = null;
let optError = '';                            // why the selected run could not be read, if it could not
/* The run's `effects.json`, or null when it has none that can be read. */
let EFFECTS = null;
/* Picker selections, kept across re-renders. */
let optSeries = null, optUsage = null;
let optToX = null, optToY = null;
let optEffGoal = null;
let optCases = null;

/* ══════════ the runs in the sidebar ══════════

   Every run is listed under the Runs view that names them, so opening one is a click from any page
   rather than a trip through the list first. The list stays there and the Runs row still opens it,
   with the columns a sidebar row has no space for. */

const baseOf = path => path.slice(path.lastIndexOf('/') + 1);

/** Every run, in the order the scan found them, which is the order the Runs list starts in. A run is a directory, so it is
    named by that directory; since this is one flat list rather than a tree, two runs whose
    directories share a name are named by the whole path instead.

    A run is marked only while a page of that run is showing. The Runs list is not a page of one run:
    it shows all of them, and selects the first before anyone asks for one, so marking it there would
    say the reader opened a run they have not. A candidate read as a report is a page of its run,
    which is why the section is what the mark is read off, not the view. */
function runRows() {
  const named = OPT_RUNS.map(r => (r.dir && r.dir !== '.' ? baseOf(r.dir) : r.run_id));
  const twice = new Set(named.filter((name, i) => named.indexOf(name) !== i));
  const view = currentView();
  const here = SECTION_OF[listOf(view)] === 'optimization' && view !== 'opt-runs';
  return OPT_RUNS.map((r, i) => ({
    name: twice.has(named[i]) && r.dir && r.dir !== '.' ? r.dir : named[i],
    id: r.id,
    on: here && (OPT_RUNS[optSel] || {}).id === r.id,
  }));
}

/** Draw them under the Runs row. Nothing is drawn there before a run is found, and the row, the
    caret that folds the list, and the Optimization heading over it are hidden then too. No count on
    the row: they are all listed right under it. No tooltip on a row: anchored under the row, it
    would cover the rows beneath it; the Runs view has the full paths. */
function renderRunList() {
  const runs = runRows();
  $('runs-sub').hidden = !runs.length;
  app.querySelector('.fold[data-fold="runs-sub"]').hidden = !runs.length;
  redraw($('runs-sub'), runs.map(r =>
    `<a class="runrow${r.on ? ' on' : ''}" data-opt-run-id="${esc(r.id)}" tabindex="0" role="link"`
    + `${r.on ? ' aria-current="true"' : ''}>${esc(r.name)}</a>`).join(''));
}

/** Move the mark to the run the page is showing. Called on every view change, which is cheaper than
    redrawing and is all that changes. The list unfolds while a run is marked: however the run was
    opened, the mark is never left hidden in a folded list. */
function markRunList() {
  const rows = runRows();
  if (rows.some(r => r.on)) setFolded('runs-sub', false);
  const on = new Map(rows.map(r => [r.id, r.on]));
  for (const el of $('runs-sub').querySelectorAll('[data-opt-run-id]')) {
    const here = on.get(el.dataset.optRunId) === true;
    el.classList.toggle('on', here);
    if (here) el.setAttribute('aria-current', 'true');
    else el.removeAttribute('aria-current');
  }
}

/* Clicking one opens that run on its overview, wherever the reader was: that is the point of
   listing them. The run the page already has open needs no reading. */
app.addEventListener('click', e => {
  const named = e.target.closest('[data-opt-run-id]');
  if (!named) return;
  e.preventDefault();
  const at = OPT_RUNS.findIndex(r => r.id === named.dataset.optRunId);
  if (at < 0) return;
  if (at === optSel) { show('opt-overview'); return; }
  openOptimizerRun(at).then(arrived => arrived && show('opt-overview'));
});

/* ── the run list, and the run that is open ── */

/* A view of the open run with nothing to draw says why in place of its content. */
function optNothing(view, html) {
  $('v-' + view).classList.toggle('blank', !!html);
  $('v-' + view).querySelector('.optnone').innerHTML = html || '';
}

/* The objective's interval as the run list reads it: `[tone, words]` by where it sits. The tone
   follows the interval, not the delta: one that straddles zero has shown nothing. */
const RUN_READING = {
  none: ['flat', 'no interval'], point: ['flat', 'no width'],
  above: ['up', 'clear of zero'], below: ['down', 'below zero'], across: ['flat', 'includes zero'],
};

function runResultCell(r) {
  const res = r.result, o = res && res.objective, side = intervalSide(o);
  if (!side) return '<span class="empty">not measured</span>';
  const [tone, reading] = RUN_READING[side];
  const hover = `objective ${signed(o.mean_diff)}`
    + (side === 'none' ? ', no interval recorded' : `, 95% interval [${fine(o.ci_low)}, ${fine(o.ci_high)}]`)
    + ` on the ${res.held_out ? 'held-out test' : 'validation'} split`
    + (o.n_cases ? `, ${plural(o.n_cases, 'case')}` : '');
  return `<span class="delta ${tone}" title="${esc(hover)}">${esc(signed(o.mean_diff))}</span>`
    + `<br><span class="path">${res.held_out ? 'held out' : 'validation'} · ${reading}</span>`;
}

/** The runs list's status cell: why a finished run stopped, else where it is, with how to go on where it can. */
function runStateCell(r) {
  if (r.state === 'finished' && r.stop_reason) return esc(stopPhrase({ reason: r.stop_reason }).replace(/^it /, ''));
  const resumable = r.state === 'halted' || r.state === 'interrupted';
  return `<span class="empty">${esc(runState(r))}</span>`
    + (resumable ? '<br><span class="path">resume it to continue</span>' : '');
}

function renderOptimizerRuns() {
  const any = OPT_RUNS.length > 0;
  /* The Optimization section and its Runs list show in every folder, as Reports and Experiments do, so a
     folder with no run says so there. A run's own page tabs wait for a run to be found. */
  app.querySelectorAll('.tab[data-sec="optimization"]').forEach(t => {
    t.hidden = t.dataset.view !== 'opt-runs'
      && (!any || (t.dataset.multi !== undefined && !(OPT && OPT.goals.length >= 2)));
  });

  $('opt-runs-sub').textContent = any
    ? 'Each one is a directory holding a run.json and the events.jsonl the run appended to.'
    : '';
  const runs = arranged('opt-runs', OPT_RUNS);
  $('opt-runs-count').textContent = any ? listCount('opt-runs', runs.length, OPT_RUNS.length) : '';
  $('opt-runs-rows').innerHTML = any ? runs.map(r => {
    const alias = r.name && r.name !== r.run_id ? ` <span class="dim">/ ${esc(r.name)}</span>` : '';
    const at = r.dir && r.dir !== '.' ? r.dir : r.events_path;
    /* A row is named by its place among every run, whatever order or filter the list is showing. */
    return `<tr class="run${r.id === optSeen ? ' on' : ''}" data-opt-run="${OPT_RUNS.indexOf(r)}" tabindex="0" aria-describedby="row-hint">
        <td>${esc(r.run_id)}${alias}<br><span class="path">${esc(at)}</span></td>
        <td class="ev">${runResultCell(r)}</td>
        <td class="ev">${r.champion_id ? esc(r.champion_id) : '<span class="empty">none picked</span>'}</td>
        <td class="ev">${plural(r.candidates, 'candidate')}`
      + `<br><span class="path">${plural(r.iterations, 'iteration')}</span></td>
        <td class="ev">${runStateCell(r)}</td>
        <td class="ev">${when(r.modified)}</td></tr>`;
  }).join('') || noMatch('opt-runs', 6) : `<tr><td colspan="6">${emptyState('No optimizer runs here yet',
    `No directory under ${esc(ROOT)} holds a run.json with an events.jsonl beside it.`)}</td></tr>`;

  renderRunList();
  /* The list can be shown empty; a view of one run cannot outlive the run. */
  if (!any && SECTION_OF[currentView()] === 'optimization' && currentView() !== 'opt-runs') show(homeView());
}

/** Mark the run the reader last opened in the Runs list. Called on every view change, like the mark in the sidebar. */
function markRunRows() {
  for (const tr of $('opt-runs-rows').querySelectorAll('tr[data-opt-run]')) {
    tr.classList.toggle('on', (OPT_RUNS[Number(tr.dataset.optRun)] || {}).id === optSeen);
  }
}

$('opt-runs-rows').onclick = e => {
  const tr = e.target.closest('tr[data-opt-run]');
  if (!tr) return;
  const i = Number(tr.dataset.optRun);
  if (i === optSel) return show('opt-overview');
  openOptimizerRun(i).then(arrived => arrived && show('opt-overview'));
};

/** Select a run that was asked for. Like the other opens, resolves false when the page moved on while its log was read. */
const openOptimizerRun = async index =>
  !!await readToOpen(OPT_RUNS[index].events_path, async () => { await selectOptimizerRun(index); return true; });

/* `keep` re-reads the folder: when the open run is still there, what is open inside it stays open. */
async function loadOptimizerRuns(runs, keep) {
  const open = (OPT_RUNS[optSel] || {}).id;
  const previous = open || readAddress().opt;
  OPT_RUNS = runs;
  optSel = previous ? OPT_RUNS.findIndex(r => r.id === previous) : -1;
  const same = keep && open && optSel >= 0;
  if (optSel < 0 && OPT_RUNS.length) optSel = 0;
  renderOptimizerRuns();
  if (OPT_RUNS.length) await selectOptimizerRun(optSel, same);
  else { OPT = null; renderOptimization(); }
}

async function selectOptimizerRun(index, keep) {
  optSel = index;
  const run = OPT_RUNS[index];
  if (keep) {
    /* Transcripts are named by content, so one already read is still right; a miss may now be there. */
    for (const [hash, got] of TRANSCRIPTS) if (got.state !== 'ok') TRANSCRIPTS.delete(hash);
  } else {
    OPT = null;
    /* Two runs can hold a candidate of the same id (a resumed run does), and a transcript
       state of `missing` is an answer about one run directory: nothing opened carries over. */
    optOpen.clear();
    openTrial = null;
    TRANSCRIPTS.clear();
    EFFECTS = null;
  }
  optError = '';
  if (!run) { renderOptimization(); return; }
  try {
    const events = run.eventsCache || (await api('/api/events?path=' + encodeURIComponent(run.events_path))).events;
    run.eventsCache = events;
    /* Awaited so the run renders once, with the effects drawer already settled. */
    const effects = await readEffects(run);
    if (OPT_RUNS[optSel] !== run) return;      // a later selection superseded this one
    OPT = deriveOptimization(events, run.config);
    EFFECTS = effects;
  } catch (err) {
    if (OPT_RUNS[optSel] !== run) return;
    OPT = null;
    optError = `Could not read ${run.events_path}: ${err.message}`;
  }
  readOpenTranscript();
  renderOptimization();
}

/** A file of the run directory, as its path under the served folder. */
const runFile = (run, name) => `${run.dir && run.dir !== '.' ? run.dir + '/' : ''}${name}`;

/* `effects.json` is fitted after the run, so most runs have none: one that cannot be read is no different. */
async function readEffects(run) {
  if (run.has_effects === false) return null;
  try {
    return (await api('/api/file?path=' + encodeURIComponent(runFile(run, 'effects.json')))).document;
  } catch {
    return null;
  }
}

function renderOptimization() {
  const run = OPT_RUNS[optSel], d = OPT;
  renderOptimizerRuns();
  renderPageHead(currentView());
  if (!d) {
    $('opt-sub').textContent = optError;
    ['opt-hero', 'opt-series-chart', 'opt-series-legend', 'opt-budget-chart', 'opt-status-chart',
      'opt-usage-chart', 'opt-to-chart', 'opt-to-legend', 'opt-result', 'opt-winner', 'opt-cands',
      'opt-explain', 'opt-method-sum', 'opt-setup-kv', 'opt-mech-kv', 'opt-stats']
      .forEach(id => { $(id).innerHTML = ''; });
    ['opt-setup-sub', 'opt-cand-sub', 'opt-cand-count'].forEach(id => { $(id).textContent = ''; });
    ['opt-source', 'opt-mech', 'opt-tradeoff', 'opt-to-none', 'opt-effects', 'opt-space-wrap', 'opt-cost', 'opt-cases']
      .forEach(id => { $(id).hidden = true; });
    Object.keys(OPT_SEGMENT).forEach(view => optNothing(view, optError && emptyState('This run could not be read',
      `${esc(optError)}<span class="ways">${listWay('opt-runs')}</span>`)));
    return;
  }

  /* The page header names the run and its status: here, only what the stop said for itself. */
  $('opt-sub').textContent = d.stop && d.stop.message ? `Stop reason: ${d.stop.message}` : '';
  $('opt-method-sum').innerHTML = methodLine(d);
  $('opt-hero').innerHTML = championHero(d);
  $('opt-stats').innerHTML = [
    optStat(d.maxRollouts != null ? `${d.spent} / ${d.maxRollouts}` : String(d.spent), 'Rollouts',
      d.maxRollouts != null ? 'spent of the budget' : 'spent · no budget declared'),
    optStat(d.best == null ? '—' : fmt(d.best), 'Best on training',
      d.bestCandidate
        ? (d.champion && d.bestCandidate.candidate_id === d.champion
          ? 'the champion, on the cases it searched over'
          : `${shortId(d.bestCandidate.candidate_id)} · highest here${d.champion ? ', not the pick' : ''}`)
        : 'on the cases it searched over',
      d.bestCandidate
        ? `candidate ${d.bestCandidate.candidate_id} scored highest on the training split; the `
          + 'champion is chosen on the validation split, so the two can differ'
        : ''),
  ].join('');
  $('opt-result').innerHTML = resultCard(d);
  $('opt-winner').innerHTML = winnerCard(d);
  renderCaseChart(d);
  renderRunNotes(d, run);
  renderCost(d);

  $('opt-search-sum').textContent = `${plural(d.proposals.length, 'proposal')} over `
    + plural(d.iterations.length, 'iteration');
  renderSeries(d);
  $('opt-budget-cap').textContent = 'A rollout is one case run once. '
    + (d.maxRollouts != null
      ? `Cumulative, against the ${d.maxRollouts}-rollout budget.`
      : 'Cumulative. The run config names no budget.');
  $('opt-budget-chart').innerHTML = budgetChart(d);

  $('opt-tradeoff').hidden = d.goals.length < 2;
  $('opt-to-none').hidden = d.goals.length >= 2;
  $('opt-to-none').innerHTML = d.goals.length >= 2 ? '' : emptyState('One goal, so nothing to trade off',
    `This run optimized one goal${d.goals.length ? `, ${esc(d.goals[0].metric)}, ` : ', '}`
      + 'so it has a best value rather than a trade-off. Iterations charts it by candidate.');
  if (d.goals.length >= 2) renderTradeoff(d);

  renderOptimizerSetup(d, run);
  renderEffects();
  Object.keys(OPT_SEGMENT).forEach(view => optNothing(view, ''));
  if (!Object.keys(d.config).length) {
    optNothing('opt-setup', emptyState('No setup recorded',
      'This run’s run.json is empty, so there is no record of what the run was told to do.'));
  }
  if (!d.proposals.length && !d.iterations.length) {
    optNothing('opt-progress', emptyState('The search has not started',
      'No candidate has been proposed and no iteration has finished, so there is nothing to chart. '
        + (run.state === 'running' ? 'This fills in as the run writes to its log.'
          : d.stop ? `The run stopped: ${esc(stopPhrase(d.stop))}.` : `The run is ${esc(runState(run))}.`)));
  }
  $('opt-cand-sub').textContent = 'Each one is a configuration the optimizer built and ran your cases against.';
  renderCandidates();
}

/* ── Result ── */

function optStat(value, key, hint, title) {
  return `<div class="stat"${title ? ` title="${esc(title)}"` : ''}>`
    + `<span class="v">${esc(value)}</span><span class="k">${esc(key)}</span>`
    + (hint ? `<span class="n">${esc(hint)}</span>` : '') + '</div>';
}

function methodLine(d) {
  const e = d.explain;
  const what = e.tuned.length
    ? `Tuned ${e.tuned.map(k => `<code>${esc(k)}</code>`).join(', ')}`
    : (e.blocks.length ? `Tuned its ${e.blocks.map(esc).join(' and ')}` : 'Searching over your agent');
  if (!e.sizes.length) return what;
  const where = e.sizes.map(s => `${s.n} ${esc(s.name)}`).join(' · ');
  return `${what} <span class="path">· ${where}</span>`;
}

/* Where the champion's gain came from: each case under the baseline and under the champion. */
function renderCaseChart(d) {
  const measures = d.champion && d.baselineId && d.champion !== d.baselineId ? caseMeasures(d) : [];
  $('opt-cases').hidden = !measures.length;
  if (!measures.length) return;
  optCases = fillPick('opt-cases-pick', measures, optCases);
  $('opt-cases-pick').hidden = measures.length < 2;
  const m = measures.find(x => x.value === optCases);
  const rows = caseRows(d, m);
  const { both, better, worse } = caseMoves(rows, m);
  $('opt-cases-cap').textContent = `${m.label} on each case, from the baseline to the champion`
    + `${m.better === 'lower' ? ', where lower is better' : ''}. Of ${plural(both, 'case')} both ran, `
    + `${better} improved and ${worse} got worse.`
    + (both < rows.length ? ' A case only one of them ran has a single mark.' : '')
    + (rows.length > CHART_ROWS ? ` The first ${CHART_ROWS} of ${rows.length} are drawn.` : '');
  const show = v => v == null ? 'not run' : fine(v);
  $('opt-cases-chart').innerHTML = dumbbells(rows.slice(0, CHART_ROWS).map(r => ({
    label: r.name, sub: r.split, from: r.from, to: r.to,
    tip: `${r.name} · ${r.split || 'no split'} · baseline ${show(r.from)} · champion ${show(r.to)}`,
  })), { axis: m.axis, better: m.better });
}

/* Result: what the run spent, by iteration and by how each trial ended. */
function renderCost(d) {
  $('opt-cost').hidden = false;
  $('opt-usage-wrap').hidden = d.usageKeys.length === 0;
  if (d.usageKeys.length) {
    optUsage = fillPick('opt-usage-pick', d.usageKeys.map(u => ({ value: u.key, label: u.label })), optUsage);
    const unit = d.usageKeys.find(u => u.key === optUsage);
    $('opt-usage-pick').hidden = d.usageKeys.length < 2;
    /* Trials after the last `iteration_end` are the end phase or an iteration still in flight. */
    $('opt-usage-cap').textContent = `Summed over that iteration's own trials and proposer calls, every role and `
      + `every attempt (the agent, its judges, the rollouts' other agents, the proposers), not cumulative`
      + (d.endPhase.trials
        ? `, excluding the ${plural(d.endPhase.trials, 'trial')} recorded after the last finished iteration.`
        : '.');
    $('opt-usage-chart').innerHTML = usageChart(d, unit.key, unit.label);
  }
  $('opt-status-cap').textContent = 'Every recorded trial by how it ended. A trial is one case run once.';
  $('opt-status-chart').innerHTML = statusChart(d);
  const tokens = d.iterations.reduce((sum, it) => sum + it.usage.tokens, 0) + d.endPhase.tokens;
  $('opt-cost-sum').textContent = [tokens ? `${compact(tokens)} tokens` : '', plural(d.trials, 'trial')]
    .filter(Boolean).join(' · ');
}

/* Result: what a reader should know before trusting the figures above. */
function renderRunNotes(d, run) {
  const notes = [];
  if (run.truncated) {
    notes.push(`Only the first ${run.events} events were read, so these numbers cover part of the run.`);
  }
  if (run.bad_lines && run.bad_lines.length) {
    notes.push(`${plural(run.bad_lines.length, 'line')} in the log could not be parsed `
      + `(${run.bad_lines.map(b => 'line ' + b.line).join(', ')}).`);
  }
  if (d.isPareto) {
    notes.push(d.goals.length >= 2
      ? 'This run uses a Pareto objective, so a single best value understates it: its front is '
        + 'on the Trade-off tab.'
      : 'This run uses a Pareto objective, so a single best value understates it, and '
        + `${d.goals.length ? 'only one of its goals was' : 'none of its goals were'} measured, `
        + 'so there is no trade-off to plot.');
  }
  $('opt-note').textContent = notes.join(' ');
  $('opt-note').className = 'optnote' + (notes.length ? ' warn' : '');
}

$('opt-cases-pick').onchange = e => { optCases = e.target.value; renderCaseChart(OPT); };

$('opt-usage-pick').onchange = e => { optUsage = e.target.value; renderCost(OPT); };

/* ── Iterations ── */

/* Iterations: one chart over the objective or a metric, by candidate. */
function renderSeries(d) {
  /* Metric values are prefixed `m:` so a metric named "objective" cannot collide with the objective. */
  const series = [{ value: 'objective', label: 'Objective' },
    ...d.metrics.map(m => ({ value: 'm:' + m, label: m }))];
  optSeries = fillPick('opt-series-pick', series, optSeries);
  $('opt-series-pick').hidden = series.length < 2;
  const metric = optSeries.startsWith('m:') ? optSeries.slice(2) : null;
  $('opt-series-head').textContent = axisName(`${metric || 'Objective'} by candidate`);
  $('opt-series-cap').textContent = metric
    ? `The mean recorded for ${metric} on each candidate it was measured on: the metric itself, `
      + 'not its contribution to the objective.'
    : 'One mark per proposal, in the order the search drew them. The candidate it built is under '
      + 'its mark, by name or by its number in the Candidates table where the name will not '
      + 'fit, and the iterations are grouped under those. The line is the best feasible '
      + 'objective so far, so it steps up where a candidate beat everything before it.';
  $('opt-series-chart').innerHTML = metric ? metricChart(d, metric) : objectiveChart(d);
  $('opt-series-legend').innerHTML = metric ? '' : ['feasible', 'infeasible', 'ineligible'].map(o =>
    `<span data-term="${o}"><i class="${o}"></i>${o}</span>`
  ).join('') + (d.champion ? '<span data-term="champion"><i class="champion"></i>champion</span>' : '')
    + '<span><i class="ln"></i>best feasible so far</span>';
}

$('opt-series-pick').onchange = e => { optSeries = e.target.value; renderSeries(OPT); };

/* ── Trade-off ── */

/* Two goals: the run's recorded front is drawn. More: the chosen pair's front is computed here. */
function renderTradeoff(d) {
  const many = d.goals.length > 2;
  const option = g => ({ value: g.metric, label: `${g.metric} ${g.dir === 'min' ? '↓' : '↑'}` });
  optToX = fillPick('opt-to-x', d.goals.map(option), optToX);
  const xg = d.goals.find(g => g.metric === optToX);
  // Never the same goal on both axes.
  optToY = fillPick('opt-to-y', d.goals.filter(g => g.metric !== optToX).map(option), optToY);
  const yg = d.goals.find(g => g.metric === optToY);
  $('opt-to-x').hidden = !many;
  $('opt-to-y').hidden = !many;

  const plotted = d.points.filter(p =>
    p.measured[xg.metric] != null && p.measured[yg.metric] != null);
  const recorded = d.front.ids.filter(id => plotted.some(p => p.candidate_id === id));
  const stated = !many && recorded.length > 0;
  const ids = stated ? recorded : front2(plotted, xg, yg).map(p => p.candidate_id);

  const whose = {
    report: `the ${plural(ids.length, 'candidate')} on the front in its end-phase report`,
    champion: 'the champion it picked',
    snapshot: `the ${plural(ids.length, 'candidate')} the last finished iteration had best so far`,
  }[d.front.source];
  const short = { report: 'the recorded front', champion: 'the champion', snapshot: 'best so far' };

  $('opt-to-head').textContent = axisName(`${yg.metric} against ${xg.metric}`);
  $('opt-to-sum').textContent = many
    ? `${d.goals.length} goals`
    : d.goals.map(g => `${g.metric} ${g.dir === 'min' ? '↓' : '↑'}`).join(' against ');
  $('opt-to-cap').textContent = 'One mark per candidate, both axes turned so better is up and to '
    + 'the right: a mark above and right of another beats it on both goals. The step line is the '
    + 'edge of what ' + (stated
      ? `the run itself called best: ${whose}.`
      : `${plural(ids.length, 'candidate')} dominate on this pair of goals, worked out here: the `
        + 'set the run recorded is a front over all '
        + `${d.goals.length} of them, which this chart cannot show.`);
  $('opt-to-chart').innerHTML = tradeoffChart(d, plotted, xg, yg, ids);

  $('opt-to-legend').innerHTML = '<span><i class="ln front"></i>'
    + `${stated ? short[d.front.source] : 'front of this pair'}</span>`
    + '<span><i class="shade"></i>dominated</span>'
    + (plotted.some(p => p.isBaseline) ? '<span data-term="baseline"><i class="hollow"></i>baseline</span>' : '')
    + (plotted.some(p => p.isChampion) && !(stated && d.front.source === 'champion')
      ? '<span data-term="champion"><i class="champion"></i>champion</span>' : '');

  const missing = d.points.filter(p => !plotted.includes(p));
  $('opt-to-stats').textContent = `${plural(plotted.length, 'candidate')} plotted`
    + (missing.length
      ? `. ${plural(missing.length, 'other')} ${missing.length === 1 ? 'was' : 'were'} not `
        + `measured on both goals: ${missing.map(p => p.candidate_id).join(', ')}.`
      : '.');
}

$('opt-to-x').onchange = e => { optToX = e.target.value; renderTradeoff(OPT); };
$('opt-to-y').onchange = e => { optToY = e.target.value; renderTradeoff(OPT); };

/* ── Setup ── */

/** The experiment whose cases this run split up, if one in this folder has them. */
function sourceLine(run) {
  const t = run.tuned_against;
  if (!t) return '';
  if (t.status === 'tie') return `Not matched to one experiment: ${esc(t.detail)}.`;
  const link = openLink('experiment', t.path);
  return t.status === 'verified'
    ? `Its ${plural(t.split_cases, 'case')} are cases of ${link}.`
    : `${esc(t.matched)} of its ${plural(t.split_cases, 'case')} are cases of ${link}.`;
}

function explainStrip(d) {
  const e = d.explain;
  const runs = e.runs > 1 ? `, each case run ${plural(e.runs, 'time')}` : '';
  const built = e.candidates
    ? `It built <b>${esc(e.candidates)}</b> versions of your agent and ran your cases against `
      + `each one${runs}.`
    : 'It builds versions of your agent and runs your cases against each one, searching for a '
      + 'configuration that scores better. No iteration has finished yet, so there is nothing it '
      + 'built to show.';
  const split = e.sizes.length
    ? ' It searched on train, picked the champion once on validation, and measured test only to report it.'
    : '';
  return `${built} Your experiment supplied the cases and evaluators and was never run itself.${split}`;
}

function renderOptimizerSetup(d, run) {
  const c = d.config, f = d.facts;
  const rows = [], mech = [];
  const mono = v => `<span class="mono">${esc(v)}</span>`;

  $('opt-setup-sub').textContent = 'What this run was told to do, as its run.json records it. Read-only: '
    + 'nothing here writes that file.';
  $('opt-explain').innerHTML = explainStrip(d);
  const source = sourceLine(run);
  $('opt-source').innerHTML = source;
  $('opt-source').hidden = !source;
  $('opt-source').title = source
    ? 'Matched by case id. A run record does not name the experiment it was given.' : '';

  rows.push(['Objective', c.objective ? mono(c.objective) : '<span class="empty">none recorded</span>']);
  if (f.isPareto) rows.push(['Shape', 'a Pareto front, so there is no single best value']);
  const unnamed = f.goalMetrics.filter(m => !f.goals.some(g => g.metric === m));
  if (unnamed.length) rows.push(['Goal metrics', unnamed.map(esc).join(', ')]);
  const feasible = c.feasibility_threshold != null
    ? `<span class="path">met by ${esc(c.feasibility_threshold)} of the cases to count</span>` : '';
  rows.push(['Constraints', (c.constraints || []).length
    ? c.constraints.map(x => mono(x)).join('<br>') + (feasible ? `<br>${feasible}` : '')
    : '<span class="empty">none</span>']);

  if (d.explain.sizes.length) {
    rows.push(['Cases', d.explain.sizes.map(s => `${s.n} ${esc(s.name)}`).join(' · ')
      + (f.seed != null ? ` <span class="path">split with seed ${esc(f.seed)}</span>` : '')]);
  }
  if (d.explain.runs > 1) rows.push(['Repeats', `each case run ${plural(d.explain.runs, 'time')}`]);

  const budget = c.budget || {};
  const limits = [
    budget.max_rollouts != null ? `${esc(budget.max_rollouts)} rollouts` : null,
    budget.max_iterations != null ? `${esc(budget.max_iterations)} iterations` : null,
    budget.max_seconds != null ? `${esc(budget.max_seconds)} seconds` : null,
  ].filter(Boolean);
  rows.push(['Budget', limits.length ? limits.join(' · ') : '<span class="empty">none set</span>']);

  if (!d.space.length) {
    if (f.blocks.length) rows.push(['Tunable blocks', f.blocks.map(esc).join(', ')]);
    if (d.explain.tuned.length) rows.push(['Actually varied', d.explain.tuned.map(k => mono(k)).join(', ')]);
  }
  const pick = c.final_pick || {};
  if (pick.n_finalists != null || f.pickRule) {
    mech.push(['Final pick', [pick.n_finalists != null ? plural(pick.n_finalists, 'finalist') : null,
      f.pickRule ? esc(f.pickRule) : null].filter(Boolean).join(' · ')]);
  }
  if (f.selection) mech.push(['Parents chosen by', esc(f.selection)]);
  if (f.scoreModel) mech.push(['Score model', esc(f.scoreModel)]);
  if (c.agent_failure_score) mech.push(['A failed agent scores', esc(c.agent_failure_score)]);
  if (c.workdir) mech.push(['Started in', mono(c.workdir)]);
  mech.push(['Run id', mono(c.run_id || run.run_id)]);

  const list = pairs => pairs.map(([k, v]) =>
    `<dt>${esc(k)}</dt><dd${/^(Objective|Constraints|Started in|Run id)$/.test(k) ? ' class="m"' : ''}>${v}</dd>`
  ).join('');
  $('opt-setup-kv').innerHTML = list(rows);
  $('opt-mech-kv').innerHTML = list(mech);
  $('opt-mech').hidden = false;
  $('opt-mech-sum').textContent = plural(mech.length, 'setting');
  renderOptimizerSpace(d);
}

/* ── Tunables: what the search could change, and which setting moved the metric ── */

function domainText(r) {
  if (r.kind === 'int' || r.kind === 'float') {
    const step = r.step != null && !(r.kind === 'int' && r.step === 1) ? ` step ${fine(r.step)}` : '';
    return `${r.kind} ${r.min == null ? '?' : fine(r.min)} – ${r.max == null ? '?' : fine(r.max)}`
      + `${r.log ? ' log' : ''}${step}`;
  }
  if (r.kind === 'text') return r.sizeCap ? `text ≤ ${compact(r.sizeCap)} chars` : 'text';
  if (r.kind === 'skills') return 'skill library';
  return r.options ? `${r.kind} · ${plural(r.options.length, 'option')}` : r.kind || '—';
}

/* A text key's default is recorded as a digest and a length, never the prompt itself. */
function defaultText(r) {
  const v = r.default;
  if (v && typeof v === 'object' && !Array.isArray(v)) {
    return typeof v.length === 'number'
      ? `<span class="path">${plural(v.length, 'character')}</span>` : '<span class="path">recorded</span>';
  }
  if (v === undefined) return '<span class="empty">none</span>';
  return `<span class="mono">${esc(typeof v === 'number' ? fine(v) : String(v))}</span>`;
}

function renderOptimizerSpace(d) {
  const space = d.space;
  $('opt-space-wrap').hidden = !space.length;
  const total = d.points.length;
  $('opt-space-sub').textContent = space.length
    ? `${plural(space.length, 'key')} the optimizer could set, as run.json declares them. `
      + 'A key with no candidate behind it was never varied.'
    : 'This run.json declares no search space: it was written before the field existed. The '
      + 'configuration above names the keys the log changed, which is as close as the record gets.';
  if (!space.length) {
    $('opt-space-note').textContent = '';
    $('opt-space-note').className = 'optnote';
    return;
  }
  $('opt-space-rows').innerHTML = space.map(r => {
    const block = r.block ? ` <span class="tag">${esc(r.block)}</span>` : '';
    const used = `${r.active} of ${total}`
      + (r.changed ? ` <span class="path">· ${r.changed} changed</span>` : '');
    return `<tr><td><span class="mono">${esc(r.key)}</span>${block}</td>`
      + `<td>${esc(domainText(r))}</td><td>${defaultText(r)}</td>`
      + `<td>${used}</td>`
      + `<td>${r.when
        ? `<span class="mono">${esc(r.when[0])}</span> = <span class="mono">${esc(r.when[1])}</span>`
        : '<span class="path">always</span>'}</td></tr>`;
  }).join('');
  $('opt-space-note').textContent = d.undeclared.length
    ? `The log also changed ${d.undeclared.join(', ')}, which this space does not declare.` : '';
  $('opt-space-note').className = 'optnote' + (d.undeclared.length ? ' warn' : '');
}

/* With no goal there is nothing to pick an estimate for. */
const usableEffects = doc => !!doc && typeof doc === 'object' && Array.isArray(doc.effects) && !!doc.metrics
  && Array.isArray(doc.goals) && doc.goals.some(g => g && g.metric);

function renderEffects() {
  const ef = usableEffects(EFFECTS) && EFFECTS.effects.length ? EFFECTS : null;
  $('opt-effects').hidden = !ef;
  if (!ef) return;

  const goals = ef.goals.filter(g => g && g.metric);
  optEffGoal = fillPick('opt-eff-goal', goals.map(g => ({
    value: g.metric,
    label: `${g.metric} · ${g.better === 'lower' ? 'lower' : 'higher'} is better`,
  })), optEffGoal);
  $('opt-eff-goal').hidden = goals.length < 2;
  const goal = goals.find(g => g.metric === optEffGoal);

  const m = ef.metrics[goal.metric] || {};
  const diag = m.diagnostics || null;
  $('opt-eff-sum').textContent = plural(effectRows(ef, goal.metric).length, 'estimate');
  $('opt-eff-cap').textContent = `Each level against its key's reference level, for ${goal.metric}`
    + `${typeof m.observations === 'number' ? `, fitted to ${plural(m.observations, 'observation')}` : ''}`
    + '. The bar is the 90% interval; one that crosses zero is drawn faint, because the run\'s '
    + 'trials do not settle it.';
  $('opt-eff-chart').innerHTML = effectsChart(ef, goal);

  const notes = [];
  if ((ef.omitted || []).length) {
    notes.push(`Not modelled: ${ef.omitted.map(o => `${o.key} (${o.why})`).join(', ')}.`);
  }
  if (diag && diag.flagged) {
    notes.push('The fit is flagged, so read these estimates as rough'
      + `${(diag.messages || []).length ? `: ${diag.messages.join('; ')}` : '.'}`);
  }
  $('opt-eff-note').textContent = notes.join(' ');
  $('opt-eff-note').className = 'optnote' + (diag && diag.flagged ? ' warn' : '');
}

$('opt-eff-goal').onchange = e => { optEffGoal = e.target.value; renderEffects(); };
