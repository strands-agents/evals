'use strict';

/* ══════════ optimization ══════════ */

/* Derived from the append-only `events.jsonl` described in `optimization/record/events.py`. */

const STATUSES = ['completed', 'agent_failure', 'evaluator_invalid', 'infrastructure_invalid'];

/* Every named stop reason ends a run on purpose: the search finished, it did not fail.
   An unknown reason is shown as the log wrote it. */
const STOP_PHRASES = {
  budget: 'it used its rollout budget',
  iterations: 'it used its iteration cap',
  exhausted: 'the schedule had nothing left to try',
  setup: 'its baseline could not be measured',
};
const stopPhrase = stop => STOP_PHRASES[stop.reason] || stop.reason;

/* Where a run is, as the scan tells it from the run's lock and its log: `running` only while a process holds
   the lock, so a run that was killed, cancelled or halted by a proposer error stops reading as running. */
const RUN_STATES = {
  running: 'still running',
  finished: 'finished',
  halted: 'halted by a proposer error',
  interrupted: 'interrupted',
  unknown: 'not finished',
};
const runState = run => RUN_STATES[(run || {}).state] || RUN_STATES.unknown;

/* The case names of one split, in either shape a record writes them: `split.cases.<part>` or `split.<part>`. */
const splitCases = (split, part) => {
  const s = split || {};
  const nested = (s.cases || {})[part];
  if (Array.isArray(nested)) return nested;
  return Array.isArray(s[part]) ? s[part] : [];
};

/* An evaluation and a measured mean are each a bare number, or a record holding one. */
const scoreOf = v => (v && typeof v === 'object' ? v.score : v);
const meanOf = v => (v && typeof v === 'object' ? v.mean : v);
const trialFailed = t => !!t.status && t.status !== 'completed';
/* A later `trial` for the same candidate, run and case replaces the earlier one. */
const trialKey = rec => JSON.stringify([rec.candidate_id, rec.run_index, rec.case_name]);

/* `run.json` facts under every name the record has used: schema 1 writes `metrics` roles,
   `search.blocks`, `runs_per_candidate`, and marks a Pareto run only in its objective string. */
function configFacts(cfg) {
  cfg = cfg || {};
  const metrics = cfg.metrics && typeof cfg.metrics === 'object' && !Array.isArray(cfg.metrics)
    ? Object.keys(cfg.metrics).filter(key => {
      const roles = (cfg.metrics[key] || {}).roles;
      return !Array.isArray(roles) || roles.includes('objective');
    })
    : [];
  const grouped = ((cfg.search || {}).blocks || [])
    .map(b => b && (b.name || b.key)).filter(Boolean);
  // `kind` in the fixture, `name` in the records on disk; both are the class that was used.
  const named = v => (v && typeof v === 'object' && (v.kind || v.name)) || null;
  const num = (...values) => values.find(v => typeof v === 'number') ?? null;
  /* A goal's direction is stated only in the objective expression. A goal over an expression
     rather than a bare metric names no single metric and yields none. */
  const goals = [];
  for (const m of String(cfg.objective || '').matchAll(/(Maximize|Minimize)\(\s*([A-Za-z_][\w.]*)\s*[,)]/g)) {
    goals.push({ metric: m[2], dir: m[1] === 'Maximize' ? 'max' : 'min' });
  }
  return {
    goals,
    goalMetrics: (cfg.goal_metrics || []).length ? cfg.goal_metrics : metrics,
    blocks: (cfg.blocks || []).length ? cfg.blocks : grouped,
    runs: num(cfg.runs, cfg.runs_per_candidate),
    isPareto: !!cfg.is_pareto || /^Pareto\(/.test(cfg.objective || ''),
    // The seed belongs to the split it made; the examples also keep the run's own seed at the top.
    seed: (cfg.split || {}).seed != null ? cfg.split.seed : (cfg.seed != null ? cfg.seed : null),
    scoreModel: named(cfg.score_model),
    pickRule: named((cfg.final_pick || {}).rule),
    selection: named(cfg.selection),
  };
}

/* A goal's direction as the objective states it, whether or not any trial measured the goal. */
const goalDir = (d, metric) => (d.facts.goals.find(g => g.metric === metric) || {}).dir || null;

function deriveOptimization(events, config) {
  const cfg = config || {};
  const facts = configFacts(cfg);
  const proposals = [];      // one row per proposal, in the order the search made them
  const statuses = {};       // trial status -> count
  const bySplit = {};        // split -> trial count
  const iterations = [];     // {iteration, rollouts, cumulative}
  let champion = null, validation = null, test = null, stop = null;
  let bestAfter = null;

  /* `propose` carries the rationale and parent, `iteration_end` the outcome and objective:
     joined on candidate id. An invalid patch has no `propose`. */
  const proposed = new Map();
  for (const e of events) {
    if (e && e.event === 'propose' && e.candidate_id) {
      /* Only the first `propose` carries `values`. A later one is a rescore and must not overwrite. */
      if (!proposed.has(e.candidate_id)) proposed.set(e.candidate_id, e);
      else if (e.values && !proposed.get(e.candidate_id).values) proposed.set(e.candidate_id, e);
    }
  }

  /* `label.measured` is what the search acted on, so it wins over the trial average. */
  const measured = new Map();        // candidate -> {metric: mean}
  const fromTrials = new Map();      // candidate -> {metric: {sum, n}}
  const trialsBy = new Map();        // candidate -> [trial record], in log order
  const labelled = new Map();        // candidate -> why the ranking labelled it as it did
  const metricNames = [];
  const seeMetric = name => { if (!metricNames.includes(name)) metricNames.push(name); };

  /* What a run spent, as `OptimizationResult.cost` counts it: every role (the agent, its judges, the
     rollouts' other agents, the proposers) and every attempt, the ones run again included. */
  const blankUsage = () => ({ tokens: 0, model_calls: 0, seconds: 0, trials: 0 });
  const addSpend = (into, usage) => {
    for (const key of ['tokens', 'model_calls', 'seconds']) {
      if (usage && typeof usage[key] === 'number' && isFinite(usage[key])) into[key] += usage[key];
    }
  };
  const addUsage = (into, rec) => {
    into.trials += 1;
    /* The record holds one judge spend per evaluator; an earlier attempt holds their total. */
    Object.values(rec.judge_usage && typeof rec.judge_usage === 'object' ? rec.judge_usage : {})
      .forEach(u => addSpend(into, u));
    for (const spent of [rec, ...(Array.isArray(rec.previous_attempts) ? rec.previous_attempts : [])]) {
      if (!spent || typeof spent !== 'object') continue;
      addSpend(into, spent.usage);
      addSpend(into, spent.benchmark_usage);
      if (spent !== rec) addSpend(into, spent.judge_usage);
    }
  };
  let pending = blankUsage();

  const lastTrial = new Map();
  events.forEach((e, i) => { if (e && e.event === 'trial') lastTrial.set(trialKey(e.record || {}), i); });

  /* What a trial cost the budget, as the optimizer's `spent` counts it: its last attempt and every one
     it replaced. Records from before `rollouts` count one per attempt. */
  const rolloutsOf = rec => [rec, ...(Array.isArray(rec.previous_attempts) ? rec.previous_attempts : [])]
    .reduce((sum, a) => sum + (a && typeof a.rollouts === 'number' ? a.rollouts : 1), 0);

  // Every trial's rollouts: the baseline and the end phase have no `iteration_end` of their own.
  let spent = 0;
  for (const [i, e] of events.entries()) {
    if (!e || typeof e !== 'object') continue;
    switch (e.event) {
      case 'trial': {
        const rec = e.record || {};
        if (lastTrial.get(trialKey(rec)) !== i) break;
        const status = rec.status || 'unknown';
        statuses[status] = (statuses[status] || 0) + 1;
        const split = rec.split || 'unknown';
        bySplit[split] = (bySplit[split] || 0) + 1;
        addUsage(pending, rec);
        spent += rolloutsOf(rec);

        const cid = rec.candidate_id;
        if (cid) {
          if (!trialsBy.has(cid)) trialsBy.set(cid, []);
          trialsBy.get(cid).push(rec);
          if (!fromTrials.has(cid)) fromTrials.set(cid, {});
          const totals = fromTrials.get(cid);
          for (const [metric, value] of Object.entries(rec.evaluations || {})) {
            const score = scoreOf(value);
            if (typeof score !== 'number' || !isFinite(score)) continue;
            seeMetric(metric);
            if (!totals[metric]) totals[metric] = { sum: 0, n: 0 };
            totals[metric].sum += score;
            totals[metric].n += 1;
          }
        }
        break;
      }
      case 'label': {
        const label = e.label || {};
        if (e.candidate_id) {
          labelled.set(e.candidate_id, {
            reason: typeof label.reason === 'string' ? label.reason : '',
            unmet: Array.isArray(label.unmet_constraints) ? label.unmet_constraints.map(String) : [],
          });
        }
        if (!e.candidate_id || !label.measured || typeof label.measured !== 'object') break;
        const means = {};
        for (const [metric, value] of Object.entries(label.measured)) {
          const mean = meanOf(value);
          if (typeof mean !== 'number' || !isFinite(mean)) continue;
          seeMetric(metric);
          means[metric] = mean;
        }
        measured.set(e.candidate_id, Object.assign(measured.get(e.candidate_id) || {}, means));
        break;
      }
      case 'proposer_usage':
        addSpend(pending, e.usage);
        break;
      case 'iteration_end': {
        const rec = e.record || {};
        // Since the iteration before, so the first one also carries the baseline's.
        const rollouts = spent - (iterations.length ? iterations[iterations.length - 1].cumulative : 0);
        const iteration = rec.iteration != null ? rec.iteration : e.iteration;
        iterations.push({ iteration, rollouts, cumulative: spent, usage: pending });
        pending = blankUsage();
        if (e.best_after && typeof e.best_after === 'object') bestAfter = e.best_after;
        for (const p of (rec.proposals || [])) {
          const origin = proposed.get(p.candidate_id);
          proposals.push({
            candidate_id: p.candidate_id,
            number: p.number,
            iteration,
            parent_id: p.parent_id,
            origin: p.origin,
            outcome: p.outcome,
            objective: typeof p.objective === 'number' ? p.objective : null,
            changed_keys: p.changed_keys || [],
            rationale: p.rationale || (origin ? origin.rationale : '') || '',
          });
        }
        break;
      }
      case 'picked':
        validation = e.report || null;
        if (validation && validation.champion_id) champion = validation.champion_id;
        break;
      case 'test':
        test = e.report || null;
        if (test && test.champion_id) champion = test.champion_id;
        break;
      case 'stop':
        stop = { reason: e.reason, message: e.message };
        break;
    }
  }

  // Whatever never got flushed was spent after the last iteration ended: the end phase.
  const endPhase = pending;

  /* Best so far counts only `feasible` candidates: one that breaks a constraint cannot ship. */
  let running = null;
  for (const p of proposals) {
    if (p.outcome === 'feasible' && p.objective != null) {
      running = running == null ? p.objective : Math.max(running, p.objective);
    }
    p.best = running;          // null until the first feasible candidate is scored
  }

  const meansOf = id => {
    const own = measured.get(id) || {};
    const averaged = fromTrials.get(id) || {};
    const out = {};
    for (const metric of metricNames) {
      if (own[metric] != null) out[metric] = own[metric];
      else if (averaged[metric] && averaged[metric].n) out[metric] = averaged[metric].sum / averaged[metric].n;
    }
    return out;
  };
  for (const p of proposals) p.measured = meansOf(p.candidate_id);

  const declared = facts.goalMetrics.filter(m => metricNames.includes(m));
  const metrics = declared.concat(metricNames.filter(m => !declared.includes(m)))
    .filter(m => proposals.some(p => p.measured[m] != null));
  const goals = facts.goals.filter(g => metrics.includes(g.metric));

  const usageKeys = [
    { key: 'tokens', label: 'Tokens' },
    { key: 'model_calls', label: 'Model calls' },
    { key: 'seconds', label: 'Seconds' },
  ].filter(u => iterations.some(it => it.usage[u.key] > 0) || endPhase[u.key] > 0);

  /* Baseline: the one the test report names, else the opening candidate. Records on disk mark
     it `origin: "baseline"` with no parent, the fixture makes it its own parent, number 0. */
  const opened = [...proposed.values()].find(e => e.origin === 'baseline'
    || e.number === 0 || !e.parent_id || e.parent_id === e.candidate_id);
  const baselineId = (test && test.baseline_id)
    || (proposals.find(p => p.candidate_id && (p.parent_id === p.candidate_id || p.number === 0)) || {}).candidate_id
    || (opened || {}).candidate_id
    || (proposals[0] || {}).candidate_id
    || null;

  /* One point per candidate: a duplicate or a rescore is a second proposal of the same one.
     No `iteration_end` reports the baseline as a proposal, so it is added here. */
  const points = proposals.filter((p, i) =>
    p.candidate_id && proposals.findIndex(q => q.candidate_id === p.candidate_id) === i);
  const built = points.length;
  if (baselineId && !points.some(p => p.candidate_id === baselineId)) {
    const means = meansOf(baselineId);
    if (Object.keys(means).length) {
      points.push({
        candidate_id: baselineId, iteration: 0, outcome: null, objective: null,
        measured: means, isBaseline: true, isChampion: baselineId === champion,
      });
    }
  }

  /* A Pareto run's end report names its front. A scalarised run records an empty `front` and
     a champion. Before the end phase only the `iteration_end` snapshots exist. */
  const front = [
    { ids: (validation || {}).front, source: 'report' },
    { ids: (test || {}).front, source: 'report' },
    { ids: champion ? [champion] : null, source: 'champion' },
    { ids: bestAfter, source: 'snapshot' },
  ].find(s => Array.isArray(s.ids) && s.ids.length) || { ids: [], source: null };

  /* `test` is the held-out comparison, `picked` the validation split the search chose on. */
  const report = test || validation || null;
  // `TestReport.means` and `ValidationReport.measured` are the same shape under different names.
  const means = (report && (report.means || report.measured)) || {};
  const resultMetrics = [];
  if (champion && means[champion]) {
    const after = means[champion], before = means[baselineId] || {};
    const order = metrics.filter(m => after[m] != null)
      .concat(Object.keys(after).filter(m => !metrics.includes(m)));
    for (const metric of order) {
      resultMetrics.push({
        metric,
        before: typeof meanOf(before[metric]) === 'number' ? meanOf(before[metric]) : null,
        after: typeof meanOf(after[metric]) === 'number' ? meanOf(after[metric]) : null,
        // Only `difference` carries an interval, and only for the metrics the run had a goal for.
        diff: ((report.difference || {})[metric]) || null,
        ratio: ((report.ratio || {})[metric]) || null,
      });
    }
  }
  const result = resultMetrics.length ? {
    heldOut: !!test,
    baselineId: means[baselineId] ? baselineId : null,
    cases: (report.objective && report.objective.n_cases) || null,
    objective: report.objective || null,
    metrics: resultMetrics,
  } : null;

  /* `propose.values` is the only copy of a candidate's configuration. */
  const valuesOf = id => {
    const p = proposed.get(id);
    return p && p.values && typeof p.values === 'object' && !Array.isArray(p.values) ? p.values : null;
  };
  /* `propose` names a candidate's parent, and so does its row in `iteration_end`. */
  const parentOf = id => (proposed.get(id) || {}).parent_id
    || (proposals.find(p => p.candidate_id === id) || {}).parent_id;
  /** A candidate and its ancestors, newest first. Stops at one seen before: a baseline can be its own parent. */
  const lineageOf = id => {
    const chain = [];
    for (let at = id; at && !chain.includes(at); at = parentOf(at)) chain.push(at);
    return chain;
  };
  const baselineValues = baselineId ? valuesOf(baselineId) : null;
  /* Diffed against the baseline, not the parent. */
  const diffSettings = id => {
    const own = valuesOf(id);
    if (!own) return null;
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const keys = Object.keys(own).concat(Object.keys(baselineValues || {}).filter(k => !(k in own)));
    return keys.map(key => ({
      key,
      before: baselineValues ? baselineValues[key] : undefined,
      after: own[key],
      changed: !baselineValues || !same(baselineValues[key], own[key]),
    }));
  };
  const winner = champion ? {
    id: champion,
    lineage: lineageOf(champion),
    baselineId: baselineValues ? baselineId : null,
    values: valuesOf(champion),
    settings: diffSettings(champion),
  } : null;

  for (const p of proposals) {
    p.trials = trialsBy.get(p.candidate_id) || [];
    p.label = labelled.get(p.candidate_id) || { reason: '', unmet: [] };
    p.splits = {};
    for (const t of p.trials) p.splits[t.split || 'unknown'] = (p.splits[t.split || 'unknown'] || 0) + 1;
    p.settings = diffSettings(p.candidate_id);
    p.isChampion = !!champion && p.candidate_id === champion;
    p.isBaseline = !!baselineId && p.candidate_id === baselineId;
  }

  const sizes = ['train', 'validation', 'test']
    .map(name => ({ name, n: splitCases(cfg.split, name).length }))
    .filter(s => s.n);
  const tuned = [];
  for (const p of proposals) {
    for (const key of p.changed_keys) if (!tuned.includes(key)) tuned.push(key);
  }
  const explain = {
    sizes,
    tuned,
    blocks: facts.blocks,
    runs: facts.runs,
    candidates: built,
  };

  const blockOf = new Map();
  for (const b of ((cfg.search || {}).blocks || [])) {
    for (const key of ((b || {}).keys || [])) if (!blockOf.has(key)) blockOf.set(key, b.name || b.key);
  }
  const space = (Array.isArray(cfg.space) ? cfg.space : []).filter(row => row && typeof row === 'object').map(row => {
    const key = String(row.key == null ? '' : row.key);
    let active = 0, changed = 0;
    for (const p of points) {
      const values = valuesOf(p.candidate_id);
      if (values && key in values) active++;
    }
    for (const p of proposals) if (p.changed_keys.includes(key)) changed++;
    return {
      key,
      kind: row.kind || '',
      min: typeof row.min_value === 'number' ? row.min_value : null,
      max: typeof row.max_value === 'number' ? row.max_value : null,
      log: !!row.log,
      step: typeof row.step === 'number' ? row.step : null,
      options: Array.isArray(row.options) ? row.options : null,
      sizeCap: typeof row.size_cap === 'number' ? row.size_cap : null,
      default: row.default,
      /* `active_when` is `(key, value)`: the row exists only under that alternative. */
      when: Array.isArray(row.active_when) && row.active_when.length === 2 ? row.active_when : null,
      block: blockOf.get(key) || null,
      active, changed,
    };
  });
  const undeclared = space.length ? tuned.filter(k => !space.some(r => r.key === k)) : [];

  const budget = cfg.budget || {};
  const scored = proposals.filter(p => p.objective != null);
  return {
    result,
    winner,
    explain,
    space,
    undeclared,
    baselineId,
    lineageOf,
    trialsBy,
    config: cfg,
    proposals,
    points,
    front,
    scored,
    iterations,
    metrics,
    goals,
    usageKeys,
    endPhase,
    statuses,
    bySplit,
    trials: Object.values(statuses).reduce((a, b) => a + b, 0),
    best: running,
    /* On a tie, the first to get there: a later one did not beat it. */
    bestCandidate: running == null ? null : proposals.find(p => p.outcome === 'feasible' && p.objective === running),
    spent,
    maxRollouts: typeof budget.max_rollouts === 'number' ? budget.max_rollouts : null,
    champion, stop,
    isPareto: facts.isPareto,
    facts,
  };
}
