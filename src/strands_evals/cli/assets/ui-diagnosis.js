'use strict';

/* ── what `strands-evals diagnose` found in a session ──
   Everything that draws a diagnosis goes through `diagnosisView`, so a new way of showing one lands here. */

/* The library's constants, as words: `SYSTEM_PROMPT_FIX` reads `system prompt fix`. */
const plainWord = v => String(v == null ? '' : v).replace(/_/g, ' ').toLowerCase();

/** A diagnosis as markup: each failure with the span it is on and the evidence for it, then each root
    cause with where it started and what would fix it. `d` is a `DiagnosisResult`, or either half of one. */
function diagnosisView(d) {
  const failures = Array.isArray(d && d.failures) ? d.failures : [];
  const causes = Array.isArray(d && d.root_causes) ? d.root_causes : [];
  if (!failures.length && !causes.length) return '<div class="empty">No failures were found in this session.</div>';

  const failure = f => {
    const kinds = (f.category || []).map((category, i) => {
      const sure = (f.confidence || [])[i], evidence = (f.evidence || [])[i];
      return `<div class="call"><span class="ckind"><b>${esc(category)}</b>`
        + `${typeof sure === 'number' ? ` · confidence ${fmt(sure)}` : ''}</span>`
        + `${evidence ? `<div class="say">${clip(evidence)}</div>` : ''}</div>`;
    }).join('');
    return `<div class="msg dg-failure"><div class="who"><b>failure</b><span>span <code>${esc(f.span_id)}</code></span></div>${kinds}</div>`;
  };
  const cause = c => {
    const facts = [
      c.location ? `started at span ${c.location}` : '', plainWord(c.causality),
      c.failure_detection_timing ? `noticed ${plainWord(c.failure_detection_timing)}` : '', plainWord(c.completion_status),
    ].filter(Boolean).join(' · ');
    return `<div class="msg dg-cause"><div class="who"><b>root cause</b><span>of the failure on span <code>${esc(c.failure_span_id)}</code></span></div>`
      + `<div class="say">${clip(c.root_cause_explanation)}</div>`
      + (c.fix_recommendation
        ? `<div class="call"><span class="ckind">fix${c.fix_type ? ` · <b>${esc(plainWord(c.fix_type))}</b>` : ''}</span>`
          + `<div class="say">${clip(c.fix_recommendation)}</div></div>` : '')
      + `<div class="dg-facts">${esc(facts)}</div></div>`;
  };
  return `<div class="tr-count">${plural(failures.length, 'failure')} · ${plural(causes.length, 'root cause')}</div>`
    + failures.map(failure).join('') + causes.map(cause).join('');
}
