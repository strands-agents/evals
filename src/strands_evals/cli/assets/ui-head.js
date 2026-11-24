'use strict';

/* ── the page header ──
   What is open is named here, not in the sidebar: where it came from, what it is, and its views. */

/* The last crumb is where the page is; the ones before it lead back. */
const crumbs = trail => trail.map(([label, view], i) => (i < trail.length - 1
  ? `<a data-go="${view}" tabindex="0" role="link">${esc(label)}</a>` : `<b>${esc(label)}</b>`))
  .join('<span class="sep">/</span>');

/* The listing knows what the folder holds now; the open copy is what was loaded. */
const listed = (list, meta) => list.find(m => m.path === meta.path) || meta;

function reportHead() {
  const r = listed(REPORTS, repMeta);
  return [[['Reports', 'reports'], [r.path]],
    `<b>${fmt(r.overall_score)}</b> overall · ${plural(r.rows, 'row')} · ${r.failing || 0} failing · `
      + (r.experiment ? `of ${madeFrom(r)}` : madeFrom(r))];
}

function experimentHead() {
  const e = listed(EXPERIMENTS, expMeta);
  const made = reportsOf(e);
  return [[['Experiments', 'experiments'], [e.path]],
    `${experimentSize(e)} · `
      + (made.length
        ? `${made.length === 1 ? 'Report' : 'Reports'}: ${made.map(r => `${openLink('report', r.path)} ${fmt(r.overall_score)}`).join(', ')}`
        : `<span class="empty">Not run yet.</span> Run it, then Rescan.`
          + ` ${docLink('cli/run/#experiment-file-run', 'unrun-experiment', 'How to run an experiment')}`)];
}

function runHead(view) {
  const run = OPT_RUNS[optSel];
  const trail = [['Runs', 'opt-runs'], [run.run_id, 'opt-overview']];
  if (view === 'report') {
    const p = OPT && OPT.proposals.find(c => c.candidate_id === repCand), why = p && outcomeWhy(p);
    return [trail.concat([['Candidates', 'opt-candidates'], [repCand]]), 'The cases this candidate ran, as a report.'
      + (why ? ` <span class="tag warn">${esc(p.outcome)}</span> ${esc(why)}` : '')];
  }
  const status = !OPT ? '' : OPT.stop ? `stopped: ${OPT.stop.reason}` : 'still running';
  return [trail.slice(0, 1).concat([[run.run_id]]),
    [status, OPT && OPT.champion ? `champion ${OPT.champion}` : ''].filter(Boolean).map(esc).join(' · ')];
}

function renderPageHead(view) {
  const run = OPT_RUNS[optSel] && (view === 'report' ? repCand : SECTION_OF[view] === 'optimization' && view !== 'opt-runs');
  const head = run ? runHead(view)
    : view === 'report' && repMeta ? reportHead()
    : view === 'experiment' && expMeta ? experimentHead()
    : view === 'session' && sesMeta ? sessionHead() : null;
  $('pagehead').hidden = !head;
  $('opt-tabs').hidden = !run || view === 'report';
  if (!head) return;
  $('crumbs').innerHTML = crumbs(head[0]);
  $('facts').innerHTML = head[1];
  $('copy-acts').innerHTML = pageActions(view);
}

$('crumbs').onclick = e => {
  const back = e.target.closest('[data-go]');
  if (!back) return;
  e.preventDefault();
  /* A candidate's cases are on the report page only while it is open: leaving by a crumb puts them away. */
  if (repCand) clearReport();
  show(back.dataset.go);
};
