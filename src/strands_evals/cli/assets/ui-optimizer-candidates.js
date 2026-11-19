'use strict';

/* ── Candidates ── */

let optQuery = '', optFilter = 'all';
const optOpen = new Set();
/* `<candidate_id>|<index into its trials>`, or null. One open at a time. */
let openTrial = null;
/* By hash: `{ state, ... }` where state is `loading`, `ok` (with `doc`), `missing` (file not
   there) or `error`. Identical trials share one hash. */
const TRANSCRIPTS = new Map();

const OUTCOME_TAG = { feasible: 'ok', infeasible: 'warn', invalid: 'warn' };

/** Why a candidate that ran cannot ship, in plain words, or '' for one that can. Its cases may all
    pass: the outcome is about the candidate as a whole, which a report of its cases does not show. */
function outcomeWhy(p) {
  const reason = p.label.reason.replace(/\.$/, '');
  if (p.outcome === 'infeasible') {
    const unmet = p.label.unmet.join('; ') || reason || 'a constraint';
    const measured = Object.keys(p.measured).filter(m => unmet.includes(m))
      .map(m => `${m} ${nice(p.measured[m])}`).join(', ');
    return `Not met: ${unmet}${measured ? ` (measured ${measured})` : ''}. A constraint is judged on the `
      + 'candidate as a whole, so every one of its cases can pass and it still cannot ship.';
  }
  if (p.outcome === 'ineligible') {
    return `${reason ? reason + '. ' : ''}Some training case was not measured, so it was left out of the ranking.`;
  }
  return '';
}

function candidateMatches(p) {
  if (optFilter === 'feasible' && p.outcome !== 'feasible') return false;
  if (optFilter === 'infeasible' && p.outcome !== 'infeasible') return false;
  if (optFilter === 'unscored' && p.objective != null) return false;
  if (!optQuery) return true;
  const hay = [p.candidate_id, p.outcome, p.rationale, p.origin, p.changed_keys.join(' ')]
    .concat((p.settings || []).filter(s => s.changed)
      .map(s => `${s.key} ${typeof s.after === 'string' ? s.after : JSON.stringify(s.after)}`))
    .join(' ').toLowerCase();
  return hay.includes(optQuery);
}

/* A candidate at a glance: where it came from, its objective and each metric it measured, each
   beside the baseline's so the change reads without opening anything. */
function candidateSummary(p, d) {
  /* An invalid patch built no candidate: what it descends from is its parent's line. */
  const chain = d.lineageOf(p.candidate_id || p.parent_id);
  const base = p.isBaseline ? null : d.points.find(x => x.candidate_id === d.baselineId);
  const show = v => Math.abs(v) >= 10 ? nice(v) : fmt(v);
  const tile = (name, value, was, dir) => {
    const delta = value != null && was != null ? value - was : null;
    const cls = !delta || !dir ? '' : (delta > 0) === (dir === 'max') ? ' up' : ' down';
    return `<div class="stat${cls}"><span class="v">${value == null ? '—' : esc(show(value))}</span>`
      + `<span class="k">${esc(name)}</span>`
      + (delta == null ? '' : `<span class="n">${delta > 0 ? '▲ +' : delta < 0 ? '▼ ' : ''}${esc(show(delta))} against ${esc(d.baselineId)}</span>`)
      + '</div>';
  };
  return '<div class="cand-sum">'
    + tile('Objective', p.objective, base && base.objective, 'max')
    + Object.keys(p.measured).map(m => tile(m, p.measured[m], base && base.measured[m], goalDir(d, m))).join('')
    + (chain.length > 1 ? `<div class="stat line"><span class="v">${esc(chain.join(' ← '))}</span>`
      + '<span class="k">Derived from</span></div>' : '')
    + '</div>';
}

/* Whether a trial's evaluators passed it, or null when none of them recorded a verdict. */
function trialVerdict(t) {
  const judged = Object.values(t.evaluations || {}).filter(v => v && typeof v.test_pass === 'boolean');
  return judged.length ? judged.every(v => v.test_pass) : null;
}

function candidateDetail(p, d, run) {
  const kv = [];
  if (p.rationale) kv.push(['Why it was tried', esc(p.rationale)]);
  const why = outcomeWhy(p);
  if (why) kv.push([`Why ${p.outcome}`, esc(why)]);

  /* Diffed against the baseline. `changed_keys` in the row above is the diff against the parent. */
  const changed = (p.settings || []).filter(s => s.changed);
  const vs = `Versus ${d.baselineId || 'baseline'}`;
  if (p.isBaseline) {
    kv.push(['Configuration', `<span class="empty">this is the configuration the run started `
      + 'from, so there is nothing to diff it against</span>']);
  } else if (!p.settings) {
    kv.push([vs, '<span class="empty">the log records no values for this candidate: only a '
      + '<code>propose</code> event carries them</span>']);
  } else if (!changed.length) {
    kv.push([vs, `<span class="empty">identical to ${esc(d.baselineId || 'the baseline')} in every `
      + 'setting the log records</span>']);
  } else {
    kv.push([vs, changed.map(s => `<div class="setting"><span class="skey">${esc(s.key)}</span>`
      + settingDiff(s) + '</div>').join('')]);
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
      const scored = Object.entries(t.evaluations || {})
        .map(([m, v]) => `${esc(m)} ${esc(fmt(scoreOf(v)))}`).join(' · ');
      const verdict = trialVerdict(t);
      const key = `${p.candidate_id}|${i}`;
      const on = openTrial === key;
      const nth = typeof t.run_index === 'number' && p.trials.some(x => x.run_index > 0)
        ? ` <span class="path">run ${esc(t.run_index)}</span>` : '';
      const head = `<tr class="trial${on ? ' on' : ''}" data-trial="${esc(key)}" tabindex="0" aria-expanded="${on}">`
        + `<td><span class="caret${on ? ' open' : ''}"></span>`
        + `${esc(t.case_name == null ? '—' : t.case_name)}${nth}</td>`
        + `<td class="path">${esc(t.split || '')}</td>`
        + `<td>${scored || '<span class="path">not judged</span>'}</td>`
        + `<td>${trialFailed(t) ? `<span class="tag warn">${esc(t.status)}</span>`
          : verdict == null ? '<span class="vb na"><i></i>completed</span>' : verdictTag({ pass: verdict })}</td></tr>`;
      return on
        ? head + `<tr class="tdetail"><td colspan="4">${trialPanel(t, run)}</td></tr>`
        : head;
    }).join('');
    kv.push(['Per case', `<table class="trials"><tbody>${rows}</tbody></table>`]);
  }
  return `<tr class="detail"><td colspan="5">${candidateSummary(p, d)}<dl class="kv">`
    + kv.map(([k, v]) => `<dt>${esc(k)}</dt>`
      + `<dd${k === 'Per case' ? ' class="m"' : ''}>${v}</dd>`).join('')
    + `</dl>${candidateActions(p)}</td></tr>`;
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
    /* An invalid patch has neither an id nor a number: its place in the log tells it from the next. */
    const key = p.candidate_id || `invalid-${d.proposals.indexOf(p)}`;
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
      + `<td class="mono">${esc(p.changed_keys.join(', ') || '—')}</td></tr>`;
    return on ? head + candidateDetail(p, d, run) : head;
  }).join('')
    || `<tr><td colspan="5">${d.proposals.length
      ? emptyState('No candidate matches', 'Try a different filter, or clear the search.')
      : emptyState('No candidate yet', 'Candidates appear here as each iteration finishes.')}</td></tr>`;
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
    openCandidateReport(OPT.proposals.find(p => p.candidate_id === jump.dataset.open));
    show('report');
    return;
  }
  const trial = e.target.closest('tr[data-trial]');
  if (trial) {
    const key = trial.dataset.trial;
    openTrial = openTrial === key ? null : key;
    readOpenTranscript();
    renderCandidates();
    return;
  }
  const tr = e.target.closest('tr.case');
  if (!tr) return;
  const key = tr.dataset.k;
  optOpen.has(key) ? optOpen.delete(key) : optOpen.add(key);
  renderCandidates();
  writeAddress('opt-candidates', true);
};

/* ── one trial, and the conversation behind it ── */

/* The log carries a `{hash, format}` pointer to `transcripts/<hash>.json` in the run directory. */
const transcriptHash = t => {
  const v = t && t.transcript;
  if (typeof v === 'string') return v;                      // older records wrote the bare hash
  return v && typeof v === 'object' && typeof v.hash === 'string' ? v.hash : null;
};

const transcriptPath = (run, hash) => runFile(run, `transcripts/${hash}.json`);

async function fetchTranscript(run, hash) {
  TRANSCRIPTS.set(hash, { state: 'loading' });
  const done = got => {
    if (OPT_RUNS[optSel] !== run) return;      // the selection moved on while this was loading
    TRANSCRIPTS.set(hash, got);
    renderCandidates();
  };
  try {
    done({ state: 'ok', doc: (await api('/api/file?path=' + encodeURIComponent(transcriptPath(run, hash)))).document });
  } catch (err) {
    done(err.status === 404 ? { state: 'missing' } : { state: 'error', why: err.message });
  }
}

/** Read the open trial's transcript, unless it has been read or is being read. */
function readOpenTranscript() {
  if (!openTrial || !OPT) return;
  const at = openTrial.lastIndexOf('|');
  const p = OPT.proposals.find(x => x.candidate_id === openTrial.slice(0, at));
  const hash = p && transcriptHash(p.trials[Number(openTrial.slice(at + 1))]);
  if (hash && !TRANSCRIPTS.has(hash)) fetchTranscript(OPT_RUNS[optSel], hash);
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
  if (got.state === 'error') return `<span class="empty">${where} could not be read: ${esc(got.why)}</span>`;
  return traceView(got.doc);
}

function trialPanel(t, run) {
  const kv = [];
  kv.push(['Ended as', (trialFailed(t)
    ? `<span class="tag warn">${esc(t.status)}</span>`
    : `<span class="tag ok">${esc(t.status || 'completed')}</span>`)
    + (t.error_type ? ` <span class="mono">${esc(t.error_type)}</span>` : '')
    + (t.error ? `<div class="why">${esc(t.error)}</div>` : '')]);

  if (typeof t.attempts === 'number' && t.attempts > 1) {
    const before = (t.previous_attempts || []).map(a =>
      esc((a && (a.status || a.error_type)) || 'unrecorded')).join(', ');
    kv.push(['Attempts', `${t.attempts}${before ? `, earlier: ${before}` : ''}`]);
  }

  const scores = Object.entries(t.evaluations || {});
  if (scores.length) {
    kv.push(['Scored', scores.map(([m, v]) => {
      const score = scoreOf(v);
      const why = (v && v.reason) || '';
      return `<div class="setting"><span class="skey">${esc(m)}</span> `
        + `<span class="mono">${esc(fmt(score))}</span>`
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
    : (valueText(t.output).trim() === ''
      ? '<span class="empty">empty</span>'
      : `<pre class="out">${clip(valueText(t.output), 4000)}</pre>`)]);

  kv.push(['Transcript', transcriptSlot(run, t)]);

  return `<div class="trial-panel"><dl class="kv">`
    + kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')
    + '</dl></div>';
}

/* ── a candidate's cases on the report page ── */

/* The cases an optimizer run was given, by name, when the experiment holding them is in this folder. */
async function sourceCases(run) {
  const named = run.tuned_against && EXPERIMENTS.find(e => e.path === run.tuned_against.path);
  try {
    const doc = named ? await document_for(named) : null;
    return doc ? new Map((doc.cases || []).map(c => [c.name, c])) : null;
  } catch {
    return null;
  }
}

/** A candidate's trials as an `EvaluationReport`-shaped document: one row per trial and evaluator, or one
    row for a trial no evaluator scored. `source` is the experiment's cases by name, when they are known. */
function candidateReport(p, source) {
  const reportCases = [], scores = [], passes = [], reasons = [], detailed = [], measured = [];
  for (const t of p.trials) {
    const entries = Object.entries(t.evaluations || {});
    const failed = trialFailed(t);
    /* A trial records what came back, not what was asked: that is read off the experiment's case. */
    const given = (source && source.get(t.case_name)) || {};
    for (const [metric, value] of entries.length ? entries : [[null, null]]) {
      const score = scoreOf(value);
      const judged = value && typeof value.test_pass === 'boolean';
      const pass = !failed && (judged ? value.test_pass : typeof score === 'number');
      const reason = failed ? `${t.status}${t.error ? `: ${t.error}` : ''}` : (value && value.reason) || '';
      reportCases.push({
        name: t.case_name,
        input: given.input,
        expected_output: given.expected_output,
        expected_assertion: given.expected_assertion,
        evaluator: metric,
        split: t.split,
        actual_output: t.output ?? null,
        /* `transcript` is a `{hash, format}` pointer, not a step list: not passed as a trajectory. */
        actual_trajectory: t.trajectory ?? null,
        actual_interactions: t.interactions,
        actual_environment_state: t.environment_state,
        usage: t.usage,
      });
      scores.push(typeof score === 'number' ? score : 0);
      passes.push(pass);
      measured.push(!failed && typeof score === 'number');
      reasons.push(reason);
      detailed.push(metric == null ? [] : [{
        score: typeof score === 'number' ? score : 0,
        test_pass: pass,
        label: failed ? String(t.status) : (value && value.label) || metric,
        reason,
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

/** Put one candidate's cases on the report page, in place of whatever report was there. */
function openCandidateReport(p) {
  const run = OPT_RUNS[optSel];
  repCand = p.candidate_id;
  const meta = { path: `${run.run_id} · ${p.candidate_id}` };
  loadReport(meta, candidateReport(p, run.sourceCases));
  if (run.sourceCases === undefined) {
    sourceCases(run).then(found => {
      run.sourceCases = found;
      /* Drawn again only if this candidate is still the one in the panel. */
      if (found && repMeta === meta) loadReport(meta, candidateReport(p, found), true);
    });
  }
  renderLists();
}
