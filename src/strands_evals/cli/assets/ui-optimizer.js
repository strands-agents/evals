'use strict';

/* ── view ── */

let OPT_RUNS = [], optSel = -1, OPT = null;
/* Picker selections, kept across re-renders. */
let optSeries = null, optUsage = null;
let optToX = null, optToY = null;
let optEffGoal = null;
let optCases = null;
/* `{ state, doc }`: state is `ok`, `none` (no `effects.json` beside the run) or `error`. */
let EFFECTS = { state: 'none' };
let optQuery = '', optFilter = 'all';
let optError = '';                            // why the selected run could not be read, if it could not
const optOpen = new Set();
/* `<candidate_id>|<index into its trials>`, or null. One open at a time. */
let openTrial = null;
/* By hash: `{ state, ... }` where state is `loading`, `ok` (with `doc`), `missing` (file not
   there), `unread` (not in the opened folder) or `error`. Identical trials share one hash. */
const TRANSCRIPTS = new Map();

/** The experiment whose cases this run split up, if one in this folder has them. */
function sourceLine(run) {
  const t = run && run.tuned_against;
  if (!t) return '';
  const link = `<a data-exp="${esc(t.path)}" tabindex="0" role="link">${esc(t.path)}</a>`;
  return t.status === 'verified'
    ? `Its ${plural(t.split_cases, 'case')} are cases of ${link}.`
    : `${t.matched} of its ${plural(t.split_cases, 'case')} are cases of ${link}.`;
}

function optStat(value, key, hint, title) {
  return `<div class="stat"${title ? ` title="${esc(title)}"` : ''}>`
    + `<span class="v">${esc(value)}</span><span class="k">${esc(key)}</span>`
    + (hint ? `<span class="n">${esc(hint)}</span>` : '') + '</div>';
}

/* ── what the optimizer did ── */

function methodLine(d) {
  const e = d.explain;
  const what = e.tuned.length
    ? `Tuned ${e.tuned.map(k => `<code>${esc(k)}</code>`).join(', ')}`
    : (e.blocks.length ? `Tuned its ${e.blocks.map(b => esc(b)).join(' and ')}` : 'Searching over your agent');
  if (!e.sizes.length) return what;
  const where = e.sizes.map(s => `${s.n} ${esc(s.name)}`).join(' · ');
  return `${what} <span class="path">· ${where}</span>`;
}

function explainStrip(d) {
  const e = d.explain;
  const runs = e.runs && e.runs > 1 ? `, each case run ${plural(e.runs, 'time')}` : '';
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

/* ── one trial, and the conversation behind it ──
   The log carries a `{hash, format}` pointer to `transcripts/<hash>.json` in the run directory. */
const transcriptHash = t => {
  const v = t && t.transcript;
  if (typeof v === 'string') return v;                      // older records wrote the bare hash
  return v && typeof v === 'object' && typeof v.hash === 'string' ? v.hash : null;
};

/** Relative to the tree being read: the same path in both modes. */
const transcriptPath = (run, hash) =>
  `${run.dir && run.dir !== '.' ? run.dir + '/' : ''}transcripts/${hash}.json`;

async function fetchTranscript(run, hash) {
  const rel = transcriptPath(run, hash);
  TRANSCRIPTS.set(hash, { state: 'loading', rel });
  const done = v => {
    if (OPT_RUNS[optSel] !== run) return;      // the selection moved on while this was loading
    TRANSCRIPTS.set(hash, Object.assign({ rel }, v));
    renderCandidates();
  };
  try {
    /* A folder run holds its transcripts as Files: a hash with no File was not in the folder. */
    if (run.transcripts) {
      const file = run.transcripts.get(hash);
      done(file ? { state: 'ok', doc: JSON.parse(await file.text()) } : { state: 'unread' });
    } else if (MODE === 'server') {
      done({ state: 'ok', doc: (await api('/api/file?path=' + encodeURIComponent(rel))).document });
    } else {
      done({ state: 'unread' });
    }
  } catch (err) {
    done(err.status === 404 ? { state: 'missing' } : { state: 'error', why: err.message });
  }
}

/* `effects.json` is fitted after the run, so most runs have none and a 404 is ordinary. */
async function readEffects(run) {
  if (run.effectsDoc) return { state: 'ok', doc: run.effectsDoc };
  if (run.fromFolder || MODE !== 'server' || run.has_effects === false) return { state: 'none' };
  const rel = `${run.dir && run.dir !== '.' ? run.dir + '/' : ''}effects.json`;
  try {
    return { state: 'ok', doc: (await api('/api/file?path=' + encodeURIComponent(rel))).document };
  } catch (err) {
    return err.status === 404 ? { state: 'none' } : { state: 'error', why: err.message };
  }
}

const usableEffects = doc => !!doc && typeof doc === 'object'
  && Array.isArray(doc.effects) && Array.isArray(doc.goals) && !!doc.metrics;

function renderEffects() {
  const ef = EFFECTS.state === 'ok' && usableEffects(EFFECTS.doc) ? EFFECTS.doc : null;
  $('opt-effects').hidden = !ef || !ef.effects.length;
  if (!ef || !ef.effects.length) return;

  const goals = ef.goals.filter(g => g && g.metric);
  optEffGoal = fillPick('opt-eff-goal', goals.map(g => ({
    value: g.metric,
    label: `${g.metric} · ${g.better === 'lower' ? 'lower' : 'higher'} is better`,
  })), optEffGoal);
  $('opt-eff-goal').hidden = goals.length < 2;
  const goal = goals.find(g => g.metric === optEffGoal) || goals[0];

  const m = (ef.metrics || {})[goal.metric] || {};
  const diag = m.diagnostics || null;
  $('opt-eff-sum').textContent = plural(ef.effects.filter(e => e.metric === goal.metric).length, 'estimate');
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

function transcriptSlot(run, t) {
  const hash = transcriptHash(t);
  if (!hash) {
    return '<span class="empty">this trial recorded no transcript</span>';
  }
  const got = TRANSCRIPTS.get(hash);
  const where = `<code>${esc(transcriptPath(run, hash))}</code>`;
  if (!got || got.state === 'loading') return `<span class="empty">reading ${where}…</span>`;
  if (got.state === 'missing') {
    return `<span class="empty">the log points at ${where}, which is not in the run directory`
      + '</span>';
  }
  if (got.state === 'unread') {
    return `<span class="empty">${where} was not among the files read from the opened folder — `
      + `the picker scans ${SCAN_DEPTH} levels down, so a run nested deeper keeps its transcripts `
      + 'to itself. Serving the folder with <code>strands-evals ui</code> reads them on '
      + 'request.</span>';
  }
  if (got.state === 'error') return `<span class="empty">${where} could not be read: ${esc(got.why)}</span>`;
  return traceView(got.doc);
}

function trialPanel(t, run) {
  const kv = [];
  const bad = t.status && t.status !== 'completed';
  kv.push(['Ended as', (bad
    ? `<span class="tag warn">${esc(t.status)}</span>`
    : `<span class="tag ok">${esc(t.status || 'completed')}</span>`)
    + (t.error_type ? ` <span class="mono">${esc(t.error_type)}</span>` : '')
    + (t.error ? `<div class="why">${esc(t.error)}</div>` : '')]);

  if (typeof t.attempts === 'number' && t.attempts > 1) {
    const before = (t.previous_attempts || []).map(a =>
      esc((a && (a.status || a.error_type)) || 'unrecorded')).join(', ');
    kv.push(['Attempts', `${t.attempts}${before ? ` — earlier: ${before}` : ''}`]);
  }

  const scores = Object.entries(t.evaluations || {});
  if (scores.length) {
    kv.push(['Scored', scores.map(([m, v]) => {
      const score = v && typeof v === 'object' ? v.score : v;
      const why = v && typeof v === 'object' && v.reason ? v.reason : '';
      return `<div class="setting"><span class="skey">${esc(m)}</span> `
        + `<span class="mono">${typeof score === 'number' ? esc(fmt(score)) : '—'}</span>`
        + (v && typeof v.test_pass === 'boolean' ? ` ${verdictTag({ pass: v.test_pass })}` : '')
        + (why ? `<div class="why">${esc(why)}</div>` : '') + '</div>';
    }).join('')]);
  }

  /* A trial can be `completed` with an evaluator that raised: that is not a zero score. */
  const errors = Object.entries(t.evaluator_errors || {});
  if (errors.length) {
    kv.push(['Evaluator errors', errors.map(([m, why]) =>
      `<div class="setting"><span class="skey">${esc(m)}</span> `
      + `<span class="tag warn">raised</span><div class="why">${esc(typeof why === 'string'
        ? why : JSON.stringify(why))}</div></div>`).join('')]);
  }

  const u = t.usage || {};
  const cost = [
    typeof u.tokens === 'number' && u.tokens ? `${compact(u.tokens)} tokens` : '',
    typeof u.model_calls === 'number' && u.model_calls ? plural(u.model_calls, 'model call') : '',
    typeof u.seconds === 'number' && u.seconds ? `${nice(u.seconds)}s` : '',
  ].filter(Boolean).join(' · ');
  if (cost) kv.push(['Cost', `<span class="mono">${esc(cost)}</span>`]);

  kv.push(['Output', t.output == null
    ? '<span class="empty">none recorded</span>'
    : (String(t.output).trim() === ''
      ? '<span class="empty">empty</span>'
      : `<pre class="out">${clip(t.output, 4000)}</pre>`)]);

  kv.push(['Transcript', transcriptSlot(run, t)]);

  return `<div class="trial-panel"><dl class="kv">`
    + kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')
    + '</dl></div>';
}

/* ── candidates ── */
/** Why a candidate that ran cannot ship, in plain words, or '' for one that can. Its cases may all
    pass: the outcome is about the candidate as a whole, which a report of its cases does not show. */
function outcomeWhy(p) {
  const reason = p.label.reason.replace(/\.$/, '');
  if (p.outcome === 'infeasible') {
    const unmet = p.label.unmet.join('; ') || reason || 'a constraint';
    const measured = Object.keys(p.measured || {}).filter(m => unmet.includes(m))
      .map(m => `${m} ${nice(p.measured[m])}`).join(', ');
    return `Not met: ${unmet}${measured ? ` (measured ${measured})` : ''}. A constraint is judged on the `
      + 'candidate as a whole, so every one of its cases can pass and it still cannot ship.';
  }
  if (p.outcome === 'ineligible') {
    return `${reason ? reason + '. ' : ''}Some training case was not measured, so it was left out of the ranking.`;
  }
  return '';
}

/* Whether a trial's evaluators passed it, or null when none of them recorded a verdict. */
function trialVerdict(t) {
  const judged = Object.values(t.evaluations || {}).filter(v => v && typeof v.test_pass === 'boolean');
  return judged.length ? judged.every(v => v.test_pass) : null;
}

function candidateReport(p) {
  /* An `EvaluationReport`-shaped document for `loadReport`. One row per (case, metric):
     `evaluations` is metric -> {score, reason}. */
  const reportCases = [], scores = [], passes = [], reasons = [], detailed = [], measured = [];
  for (const t of p.trials) {
    const entries = Object.entries(t.evaluations || {});
    const rows = entries.length ? entries : [[null, null]];
    for (const [metric, value] of rows) {
      const score = value && typeof value === 'object' ? value.score : value;
      const reason = (value && typeof value === 'object' && value.reason) || '';
      const failed = t.status && t.status !== 'completed';
      const judged = value && typeof value.test_pass === 'boolean';
      const pass = !failed && (judged ? value.test_pass : typeof score === 'number');
      reportCases.push({
        name: t.case_name,
        evaluator: metric,
        split: t.split,
        actual_output: t.output == null ? null : t.output,
        /* `transcript` is a `{hash, format}` pointer, not a step list: not passed as a trajectory. */
        actual_trajectory: t.trajectory != null ? t.trajectory : null,
        actual_interactions: t.interactions,
        actual_environment_state: t.environment_state,
        usage: t.usage,
      });
      scores.push(typeof score === 'number' ? score : 0);
      passes.push(pass);
      measured.push(!failed && typeof score === 'number');
      reasons.push(failed ? `${t.status}${t.error ? `: ${t.error}` : ''}` : reason);
      detailed.push(metric == null ? [] : [{
        score: typeof score === 'number' ? score : 0,
        test_pass: pass,
        label: failed ? String(t.status) : (value && value.label) || metric,
        reason: failed ? `${t.status}${t.error ? `: ${t.error}` : ''}` : reason,
        not_applicable: false,
      }]);
    }
  }
  const real = scores.filter((_, i) => measured[i]);
  return {
    overall_score: real.length ? real.reduce((a, b) => a + b, 0) / real.length : 0,
    cases: reportCases, scores, test_passes: passes, reasons, detailed_results: detailed,
  };
}

/** Put one candidate's cases in the Report view, holding on to the report they displace. */
function openCandidateReport(p) {
  const run = OPT_RUNS[optSel] || {};
  /* Saved once: a second candidate must not overwrite the Evals report being held. */
  if (!repDisplaced) repDisplaced = { meta: repMeta, doc: REP };
  repDisplaced.cand = p.candidate_id;
  loadReport({ path: `${run.run_id || 'run'} · ${p.candidate_id}` }, candidateReport(p));
  const why = outcomeWhy(p);
  borrowReport(`Candidate <b>${esc(p.candidate_id)}</b> of ${esc(run.run_id || 'this run')}, `
    + 'from Optimization.' + (why ? ` <span class="tag warn">${esc(p.outcome)}</span> ${esc(why)}` : ''));
}

/** Open one candidate's row on the Candidates tab, clearing any filter that would hide it. */
function openCandidate(id) {
  if (!OPT || !OPT.proposals.some(p => p.candidate_id === id)) return;
  optQuery = ''; optFilter = 'all';
  $('opt-find').value = '';
  app.querySelectorAll('#opt-seg button').forEach(b => b.classList.toggle('on', b.dataset.f === 'all'));
  optOpen.add(id);
  renderCandidates();
  show('opt-candidates');
  const row = [...$('opt-cands').querySelectorAll('tr.case')].find(tr => tr.dataset.k === id);
  if (row && row.scrollIntoView) row.scrollIntoView({ block: 'center' });
}

const OUTCOME_TAG = { feasible: 'ok', infeasible: 'warn', invalid: 'warn' };

function candidateMatches(p) {
  if (optFilter === 'feasible' && p.outcome !== 'feasible') return false;
  if (optFilter === 'infeasible' && p.outcome !== 'infeasible') return false;
  if (optFilter === 'unscored' && p.objective != null) return false;
  if (!optQuery) return true;
  const hay = [p.candidate_id, p.outcome, p.rationale, p.origin, (p.changed_keys || []).join(' ')]
    .concat((p.settings || []).filter(s => s.changed)
      .map(s => `${s.key} ${typeof s.after === 'string' ? s.after : JSON.stringify(s.after)}`))
    .join(' ').toLowerCase();
  return hay.includes(optQuery);
}

function candidateDetail(p, d, run) {
  const kv = [];
  if (p.rationale) kv.push(['Why it was tried', esc(p.rationale)]);
  if (outcomeWhy(p)) kv.push([`Why ${p.outcome}`, esc(outcomeWhy(p))]);
  if (p.parent_id && p.parent_id !== p.candidate_id) kv.push(['Derived from', esc(p.parent_id)]);

  /* Diffed against the baseline. `changed_keys` in the row above is the diff against the parent. */
  const changed = (p.settings || []).filter(s => s.changed);
  const vs = `Versus ${d.baselineId || 'baseline'}`;
  if (p.isBaseline) {
    kv.push(['Configuration', `<span class="empty">this is the configuration the run started `
      + 'from, so there is nothing to diff it against</span>']);
  } else if (!p.settings) {
    kv.push([vs, '<span class="empty">the log records no values for this candidate — only a '
      + '<code>propose</code> event carries them</span>']);
  } else if (!changed.length) {
    kv.push([vs, `<span class="empty">identical to ${esc(d.baselineId || 'the baseline')} in every `
      + 'setting the log records</span>']);
  } else {
    kv.push([vs, changed.map(s => `<div class="setting"><span class="skey">${esc(s.key)}</span>`
      + settingDiff(s) + '</div>').join('')]);
  }

  const metrics = Object.keys(p.measured || {});
  if (metrics.length) {
    kv.push(['Measured', metrics.map(m =>
      `<span class="mono">${esc(m)} ${esc(nice(p.measured[m]))}</span>`).join(' · ')]);
  }

  if (!p.trials.length) {
    kv.push(['Cases', '<span class="empty">no trial was recorded for this candidate, so there is '
      + 'nothing to open</span>']);
  } else {
    const order = ['train', 'validation', 'test'];
    const splits = Object.keys(p.splits)
      .sort((a, b) => (order.indexOf(a) + 1 || 9) - (order.indexOf(b) + 1 || 9))
      .map(s => `${p.splits[s]} ${esc(s)}`).join(' · ');
    kv.push(['Cases', `${plural(p.trials.length, 'trial')} · ${splits}`
      + ` <button class="btn" data-open="${esc(p.candidate_id)}">Open as report</button>`]);
    const rows = p.trials.map((t, i) => {
      const scored = Object.entries(t.evaluations || {}).map(([m, v]) => {
        const score = v && typeof v === 'object' ? v.score : v;
        return `${esc(m)} ${typeof score === 'number' ? esc(fmt(score)) : '—'}`;
      }).join(' · ');
      const bad = t.status && t.status !== 'completed';
      const verdict = trialVerdict(t);
      /* Keyed by position: a trial recorded twice would collide under case and run index. */
      const key = `${p.candidate_id}|${i}`;
      const on = openTrial === key;
      const nth = typeof t.run_index === 'number' && p.trials.some(x => x.run_index > 0)
        ? ` <span class="path">run ${esc(t.run_index)}</span>` : '';
      const head = `<tr class="trial${on ? ' on' : ''}" data-trial="${esc(key)}" tabindex="0" aria-expanded="${on}">`
        + `<td><span class="caret${on ? ' open' : ''}"></span>`
        + `${esc(t.case_name == null ? '—' : t.case_name)}${nth}</td>`
        + `<td class="path">${esc(t.split || '')}</td>`
        + `<td>${scored || '<span class="path">not judged</span>'}</td>`
        + `<td>${bad ? `<span class="tag warn">${esc(t.status)}</span>`
          : verdict == null ? '<span class="vb na"><i></i>completed</span>' : verdictTag({ pass: verdict })}</td></tr>`;
      return on
        ? head + `<tr class="tdetail"><td colspan="4">${trialPanel(t, run || {})}</td></tr>`
        : head;
    }).join('');
    kv.push(['Per case', `<table class="trials"><tbody>${rows}</tbody></table>`]);
  }
  return '<tr class="detail"><td colspan="5"><dl class="kv">'
    + kv.map(([k, v]) => `<dt>${esc(k)}</dt>`
      + `<dd${/^(Derived|Measured|Per case)/.test(k) ? ' class="m"' : ''}>${v}</dd>`).join('')
    + '</dl></td></tr>';
}

function renderCandidates() {
  const d = OPT;
  if (!d) return;
  const shown = d.proposals.filter(candidateMatches);
  const unscored = d.proposals.length - d.scored.length;
  $('opt-cand-count').textContent = `${shown.length} of ${plural(d.proposals.length, 'candidate')}`
    + (unscored ? `, ${unscored} never scored` : '');

  const run = OPT_RUNS[optSel];

  $('opt-cands').innerHTML = shown.map(p => {
    const key = p.candidate_id || `invalid-${p.iteration}-${p.number}`;
    const on = optOpen.has(key);
    const name = p.candidate_id
      ? `<span class="caret${on ? ' open' : ''}"></span>${esc(p.candidate_id)}`
        + (p.number != null ? ` <span class="path">#${esc(p.number)}</span>` : '')
      : '<span class="path">no candidate</span>';
    const tags = (p.isChampion ? ' <span class="tag ok" data-term="champion">champion</span>' : '')
      + (p.isBaseline ? ' <span class="tag" data-term="baseline">baseline</span>' : '');
    const obj = p.objective == null
      ? '<span class="path">not scored</span>'
      : scoreBar(p.objective);
    const outcome = p.outcome
      ? `<span class="tag ${OUTCOME_TAG[p.outcome] || ''}"${TERMS[p.outcome] ? ` data-term="${esc(p.outcome)}"` : ''}>${esc(p.outcome)}</span>` : '';
    const head = `<tr class="case" data-k="${esc(key)}" tabindex="0" aria-expanded="${on}"><td>${name}${tags}</td>`
      + `<td>${esc(p.iteration)}</td><td>${obj}</td><td>${outcome}</td>`
      + `<td class="mono">${esc((p.changed_keys || []).join(', ') || '—')}</td></tr>`;
    return on ? head + candidateDetail(p, d, run) : head;
  }).join('')
    || `<tr><td colspan="5"><div class="nores">${d.proposals.length
      ? '<b>No candidate matches</b>Try a different filter, or clear the search.'
      : '<b>No candidate yet</b>Candidates appear here as each iteration finishes.'}`
      + '</div></td></tr>';
}

/* ── the trade-off panel ──
   Two goals: the run's recorded front is drawn. More: the chosen pair's front is computed here. */
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

  $('opt-to-head').textContent = `${yg.metric} against ${xg.metric}`;
  $('opt-to-sum').textContent = many
    ? `${d.goals.length} goals`
    : d.goals.map(g => `${g.metric} ${g.dir === 'min' ? '↓' : '↑'}`).join(' against ');
  $('opt-to-cap').textContent = 'One mark per candidate, both axes turned so better is up and to '
    + 'the right: a mark above and right of another beats it on both goals. The step line is the '
    + 'edge of what ' + (stated
      ? `the run itself called best — ${whose}.`
      : `${plural(ids.length, 'candidate')} dominate on this pair of goals, worked out here: the `
        + 'set the run recorded is a front over all '
        + `${d.goals.length} of them, which this chart cannot show.`);
  $('opt-to-chart').innerHTML = tradeoffChart(d, xg, yg, ids);

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

function renderOptimization() {
  const run = OPT_RUNS[optSel];
  renderOptimizerRuns();
  if (!run || !OPT) {
    $('opt-sub').textContent = optError || (OPT_RUNS.length
      ? 'Pick a run on the Runs tab.'
      : 'No optimizer run directory in this folder. One is a run.json with an events.jsonl beside it.');
    ['opt-hero', 'opt-series-chart', 'opt-series-legend', 'opt-budget-chart', 'opt-status-chart',
      'opt-usage-chart', 'opt-to-chart', 'opt-to-legend', 'opt-result', 'opt-winner', 'opt-cands',
      'opt-explain', 'opt-method-sum', 'opt-setup-kv', 'opt-mech-kv']
      .forEach(id => { $(id).innerHTML = ''; });
    $('opt-setup-sub').textContent = '';
    $('opt-source').hidden = true;
    $('opt-mech').hidden = true;
    $('opt-tradeoff').hidden = true;
    $('opt-to-none').hidden = true;
    $('opt-effects').hidden = true;
    $('opt-space-wrap').hidden = true;
    $('opt-cost').hidden = true;
    $('opt-cases').hidden = true;
    $('opt-cand-sub').textContent = '';
    $('opt-cand-count').textContent = '';
    $('opt-stats').innerHTML = '';
    return;
  }

  const d = OPT;
  $('opt-sub').textContent = run.run_id + (d.stop ? ` · stopped: ${d.stop.reason}` : ' · still running');
  $('opt-sub').title = (d.stop && d.stop.message) || '';

  const stats = [
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
  ];
  const tokens = d.iterations.reduce((sum, it) => sum + (it.usage.tokens || 0), 0) + (d.endPhase.tokens || 0);
  $('opt-stats').innerHTML = stats.join('');

  $('opt-hero').innerHTML = championHero(d);
  $('opt-result').innerHTML = resultCard(d);
  $('opt-winner').innerHTML = winnerCard(d);
  renderCaseChart(d);
  $('opt-search-sum').textContent = `${plural(d.proposals.length, 'proposal')} over `
    + plural(d.iterations.length, 'iteration');

  /* Metric values are prefixed `m:` so a metric named "objective" cannot collide with the objective. */
  const series = [{ value: 'objective', label: 'Objective' },
    ...d.metrics.map(m => ({ value: 'm:' + m, label: m }))];
  optSeries = fillPick('opt-series-pick', series, optSeries);
  $('opt-series-pick').hidden = series.length < 2;
  const metric = optSeries.startsWith('m:') ? optSeries.slice(2) : null;
  $('opt-series-head').textContent = `${metric || 'Objective'} by candidate`;
  $('opt-series-cap').textContent = metric
    ? `The mean recorded for ${metric} on each candidate it was measured on — the metric itself, `
      + 'not its contribution to the objective.'
    : 'One mark per proposal, in the order the search drew them. The candidate it built is under '
      + 'its mark — by name, or by its number in the Candidates table where the name will not '
      + 'fit — and the iterations are grouped under those. The line is the best feasible '
      + 'objective so far, so it steps up where a candidate beat everything before it.';
  $('opt-series-chart').innerHTML = metric ? metricChart(d, metric) : objectiveChart(d);
  $('opt-series-legend').innerHTML = metric ? '' : ['feasible', 'infeasible', 'ineligible'].map(o =>
    `<span data-term="${o}"><i class="${o}"></i>${o}</span>`
  ).join('') + (d.champion ? '<span data-term="champion"><i class="champion"></i>champion</span>' : '')
    + '<span><i class="ln"></i>best feasible so far</span>';

  $('opt-tradeoff').hidden = d.goals.length < 2;
  $('opt-to-none').hidden = d.goals.length >= 2;
  $('opt-to-none').textContent = d.goals.length >= 2 ? ''
    : `This run optimized one goal${d.goals.length ? ` — ${d.goals[0].metric} — ` : ', '}`
      + 'so it has a best value rather than a trade-off. Progress charts it by candidate.';
  if (d.goals.length >= 2) renderTradeoff(d);

  renderEffects();

  $('opt-cost').hidden = false;
  $('opt-usage-wrap').hidden = d.usageKeys.length === 0;
  if (d.usageKeys.length) {
    optUsage = fillPick('opt-usage-pick', d.usageKeys.map(u => ({ value: u.key, label: u.label })), optUsage);
    const unit = d.usageKeys.find(u => u.key === optUsage);
    $('opt-usage-pick').hidden = d.usageKeys.length < 2;
    /* Trials after the last `iteration_end` are the end phase or an iteration still in flight. */
    $('opt-usage-cap').textContent = `Summed over that iteration's own trials, not cumulative`
      + (d.endPhase.trials
        ? `, excluding the ${plural(d.endPhase.trials, 'trial')} recorded after the last finished iteration.`
        : '.');
    $('opt-usage-chart').innerHTML = usageChart(d, unit.key, unit.label);
  }

  $('opt-budget-cap').textContent = 'A rollout is one case run once. '
    + (d.maxRollouts != null
      ? `Cumulative, against the ${d.maxRollouts}-rollout budget.`
      : 'Cumulative. The run config names no budget.');
  $('opt-budget-chart').innerHTML = budgetChart(d);

  $('opt-status-cap').textContent = 'Every recorded trial by how it ended. A trial is one case run once.';
  $('opt-status-chart').innerHTML = statusChart(d);
  $('opt-cost-sum').textContent = [tokens ? `${compact(tokens)} tokens` : '', plural(d.trials, 'trial')]
    .filter(Boolean).join(' · ');

  $('opt-method-sum').innerHTML = methodLine(d);
  $('opt-setup-sub').textContent = 'What this run was told to do, as its run.json records it. Read-only: '
    + 'nothing here writes that file.';
  $('opt-explain').innerHTML = explainStrip(d);
  const source = sourceLine(run);
  $('opt-source').innerHTML = source;
  $('opt-source').hidden = !source;
  $('opt-source').title = source
    ? 'Matched by case id. A run record does not name the experiment it was given.' : '';
  renderOptimizerSetup(d, run);
  $('opt-cand-sub').textContent = 'Each one is a configuration the optimizer built and ran your cases against.';
  renderCandidates();

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
      : 'This run uses a Pareto objective, so a single best value understates it — and '
        + `${d.goals.length ? 'only one of its goals was' : 'none of its goals were'} measured, `
        + 'so there is no trade-off to plot.');
  }
  $('opt-note').textContent = notes.join(' ');
  $('opt-note').className = 'optnote' + (notes.length ? ' warn' : '');
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
  const both = rows.filter(r => r.from != null && r.to != null);
  const better = both.filter(r => r.to !== r.from && (r.to > r.from) === (m.better !== 'lower')).length;
  const worse = both.filter(r => r.to !== r.from).length - better;
  $('opt-cases-cap').textContent = `${m.label} on each case, from the baseline to the champion`
    + `${m.better === 'lower' ? ', where lower is better' : ''}. Of ${plural(both.length, 'case')} both ran, `
    + `${better} improved and ${worse} got worse.`
    + (both.length < rows.length ? ' A case only one of them ran has a single mark.' : '')
    + (rows.length > CHART_ROWS ? ` The first ${CHART_ROWS} of ${rows.length} are drawn.` : '');
  const show = v => v == null ? 'not run' : fine(v);
  $('opt-cases-chart').innerHTML = dumbbells(rows.slice(0, CHART_ROWS).map(r => ({
    label: r.name, sub: r.split, from: r.from, to: r.to,
    tip: `${r.name} · ${r.split || 'no split'} · baseline ${show(r.from)} · champion ${show(r.to)}`,
  })), { axis: m.axis, better: m.better });
}

/* `keep` re-reads the folder: when the open run is still there, what is open inside it stays open. */
async function loadOptimizerRuns(runs, keep) {
  const open = optSel >= 0 && OPT_RUNS[optSel] ? OPT_RUNS[optSel].id : null;
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
       state of `missing` is an answer about one run directory: reset both. */
    openTrial = null;
    TRANSCRIPTS.clear();
    EFFECTS = { state: 'none' };
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
  renderOptimization();
}

const optBlank = (title, body) =>
  `<tr><td colspan="6"><div class="nores"><b>${title}</b>${body}</div></td></tr>`;

/* The tone follows the interval, not the delta: one that straddles zero has shown nothing. */
function runResultCell(r) {
  const res = r.result, o = res && res.objective;
  if (!o || typeof o.mean_diff !== 'number') {
    return '<span class="empty">not measured</span>';
  }
  const interval = typeof o.ci_low === 'number' && typeof o.ci_high === 'number';
  let tone = 'flat', reading;
  if (!interval) reading = 'no interval';
  else if (o.ci_high === o.ci_low) reading = 'no width';
  else if (o.ci_low > 0) { tone = 'up'; reading = 'clear of zero'; }
  else if (o.ci_high < 0) { tone = 'down'; reading = 'below zero'; }
  else reading = 'includes zero';
  const where = res.held_out ? 'held out' : 'validation';
  const hover = `objective ${signed(o.mean_diff)}`
    + (interval ? `, 95% interval [${fine(o.ci_low)}, ${fine(o.ci_high)}]` : ', no interval recorded')
    + ` on the ${where === 'held out' ? 'held-out test' : 'validation'} split`
    + (o.n_cases ? `, ${plural(o.n_cases, 'case')}` : '');
  return `<span class="delta ${tone}" title="${esc(hover)}">${esc(signed(o.mean_diff))}</span>`
    + `<br><span class="path">${esc(where)} · ${esc(reading)}</span>`;
}

function renderOptimizerRuns() {
  const any = OPT_RUNS.length > 0;
  $('sec-opt').hidden = !any;
  $('sgroup-opt').hidden = !any;
  $('sections').classList.toggle('lonely', !any);
  app.querySelectorAll('.tab[data-sec="optimization"]').forEach(t => {
    t.hidden = !any || (t.dataset.multi !== undefined && !(OPT && OPT.goals.length >= 2));
  });

  $('opt-runs-sub').textContent = any
    ? 'Each one is a directory holding a run.json and the events.jsonl the run appended to.'
    : '';
  $('opt-runs-count').textContent = any ? plural(OPT_RUNS.length, 'run') : '';
  $('opt-runs-rows').innerHTML = any ? OPT_RUNS.map((r, i) => {
    const alias = r.name && r.name !== r.run_id ? ` <span class="dim">/ ${esc(r.name)}</span>` : '';
    const at = r.dir && r.dir !== '.' ? r.dir : r.events_path;
    return `<tr class="run${i === optSel ? ' on' : ''}" data-opt-run="${i}" tabindex="0">
        <td>${esc(r.run_id)}${alias}<br><span class="path">${esc(at)}</span></td>
        <td class="ev">${runResultCell(r)}</td>
        <td class="ev">${r.champion_id ? esc(r.champion_id) : '<span class="empty">none picked</span>'}</td>
        <td class="ev">${plural(r.candidates, 'candidate')}`
      + `<br><span class="path">${plural(r.iterations, 'iteration')}</span></td>
        <td class="ev">${r.stop_reason
        ? `<span class="tag">${esc(r.stop_reason)}</span>`
        : '<span class="empty">still running</span>'}</td>
        <td class="ev">${when(r.modified)}</td></tr>`;
  }).join('') : optBlank('No optimizer run here', MODE === 'server'
    ? `No directory under ${esc(ROOT)} holds a run.json with an events.jsonl beside it.`
    : 'Open a folder that holds an optimizer run directory.');

  if (!any && SECTION_OF[currentView()] === 'optimization') show('runs');
}

$('opt-runs-rows').onclick = e => {
  const tr = e.target.closest('tr[data-opt-run]');
  if (!tr) return;
  const i = Number(tr.dataset.optRun);
  if (i === optSel) return show('opt-overview');
  selectOptimizerRun(i).then(() => show('opt-overview'));
};

/* ── Setup ── */
function renderOptimizerSetup(d, run) {
  const c = d.config || {};
  const f = d.facts || configFacts(c);
  const rows = [], mech = [];
  const mono = v => `<span class="mono">${esc(v)}</span>`;

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
    mech.push(['Final pick', [pick.n_finalists != null ? `${plural(pick.n_finalists, 'finalist')}` : null,
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
  $('opt-mech').hidden = !mech.length;
  $('opt-mech-sum').textContent = plural(mech.length, 'setting');
  renderOptimizerSpace(d);
}

/* ── what the search was allowed to change ── */

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
      ? `<span class="path">${esc(plural(v.length, 'character'))}</span>` : '<span class="path">recorded</span>';
  }
  if (v === undefined) return '<span class="empty">none</span>';
  return `<span class="mono">${esc(typeof v === 'number' ? fine(v) : String(v))}</span>`;
}

function renderOptimizerSpace(d) {
  const space = d.space || [];
  $('opt-space-wrap').hidden = !space.length;
  const total = d.points.length;
  $('opt-space-sub').textContent = space.length
    ? `${plural(space.length, 'key')} the optimizer could set, as run.json declares them. `
      + 'A key with no candidate behind it was never varied.'
    : 'This run.json declares no search space — it was written before the field existed. Setup '
      + 'names the keys the log changed, which is as close as the record gets.';
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

/* `navigator.clipboard` needs a secure context, which `file://` is not, so there is a fallback. */
async function copyText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* fall through to the selection trick */ }
  try {
    const ta = Object.assign(document.createElement('textarea'), { value: text });
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const done = document.execCommand && document.execCommand('copy');
    ta.remove();
    return !!done;
  } catch { return false; }
}

/* Toggled in place: a re-render would close the rows the click just opened. */
$('opt-result').onclick = e => {
  if (!e.target.closest('[data-act="opt-rest"]')) return;
  const rest = $('opt-rest'), btn = $('opt-result').querySelector('.rmore');
  rest.hidden = !rest.hidden;
  const n = rest.querySelectorAll('tr').length;
  btn.textContent = rest.hidden
    ? `+ ${plural(n, 'metric')} the run was not aiming at`
    : `− ${plural(n, 'metric')} the run was not aiming at`;
};

$('opt-winner').onclick = async e => {
  const act = e.target.closest('[data-act]');
  if (!act || !OPT || !OPT.winner || !OPT.winner.values) return;
  const text = JSON.stringify(OPT.winner.values, null, 2) + '\n';
  const said = $('opt-copied');
  if (act.dataset.act === 'copy-winner') {
    said.textContent = await copyText(text)
      ? `copied ${plural(Object.keys(OPT.winner.values).length, 'setting')}`
      : 'could not reach the clipboard — the configuration is above to select by hand';
    return;
  }
  if (act.dataset.act === 'save-winner') {
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    const a = Object.assign(document.createElement('a'), { href: url, download: 'champion.json' });
    a.click();
    if (URL.revokeObjectURL) URL.revokeObjectURL(url);
    said.textContent = `downloaded champion.json · ${KB(text.length)}`;
  }
};
$('opt-find').oninput = e => { optQuery = e.target.value.trim().toLowerCase(); renderCandidates(); };
$('opt-seg').onclick = e => {
  const b = e.target.closest('button[data-f]');
  if (!b) return;
  optFilter = b.dataset.f;
  app.querySelectorAll('#opt-seg button').forEach(x => x.classList.toggle('on', x === b));
  renderCandidates();
};
$('opt-cands').onclick = e => {
  /* Handled before the row toggle: the button sits inside the row it would collapse. */
  const jump = e.target.closest('[data-open]');
  if (jump) {
    const p = OPT && OPT.proposals.find(x => x.candidate_id === jump.dataset.open);
    if (p && p.trials.length) {
      openCandidateReport(p);
      show('report');
    }
    return;
  }
  const trial = e.target.closest('tr[data-trial]');
  if (trial) {
    const key = trial.dataset.trial;
    openTrial = openTrial === key ? null : key;
    if (openTrial) {
      const at = key.lastIndexOf('|');
      const p = OPT.proposals.find(x => x.candidate_id === key.slice(0, at));
      const hash = p && transcriptHash(p.trials[Number(key.slice(at + 1))]);
      if (hash && !TRANSCRIPTS.has(hash)) fetchTranscript(OPT_RUNS[optSel] || {}, hash);
    }
    renderCandidates();
    return;
  }
  const tr = e.target.closest('tr.case');
  if (!tr) return;
  const key = tr.dataset.k;
  optOpen.has(key) ? optOpen.delete(key) : optOpen.add(key);
  renderCandidates();
};
$('opt-source').onclick = e => {
  const link = e.target.closest('a[data-exp]');
  if (!link) return;
  e.preventDefault();
  const i = RUNS.findIndex(r => r.experiment && r.experiment.path === link.dataset.exp);
  if (i < 0) return;
  if (i === selRun) return show('experiment');
  selectRun(i, { view: 'experiment' });
};
// Redrawing the whole view is cheap: the log is already parsed.
$('opt-series-pick').onchange = e => { optSeries = e.target.value; renderOptimization(); };
$('opt-usage-pick').onchange = e => { optUsage = e.target.value; renderOptimization(); };
$('opt-cases-pick').onchange = e => { optCases = e.target.value; renderCaseChart(OPT); };
$('opt-to-x').onchange = e => { optToX = e.target.value; renderOptimization(); };
$('opt-to-y').onchange = e => { optToY = e.target.value; renderOptimization(); };
$('opt-eff-goal').onchange = e => { optEffGoal = e.target.value; renderOptimization(); };
