## What

Judge evaluators silently mis-score correct agents when the trajectory they must
read overflows the judge model's context window. The oversized prompt is sent
anyway; the judge either truncates (scoring on a partial trace) or the request
fails and the run is recorded as a hard evaluator error — in both cases a correct
agent can be marked failing for a reason that has nothing to do with its behavior.

This PR adds **progressive trace disclosure**. Instead of inlining a whole
`Session` into the evaluation prompt, the evaluator can hand the judge a compact
per-span overview plus three read-only lookup tools (`list_spans`, `get_span`,
`search_spans`) and let it pull only the spans it needs to verify the rubric.

## The `disclosure` kwarg (new public API)

Every judge evaluator gains a `disclosure` constructor kwarg:

- `"auto"` (default) — preflight the rendered prompt; disclose **only** when it
  would overflow the judge's context window. Prompts that already fit are sent
  byte-for-byte unchanged, so existing behavior is preserved for the common case.
- `"always"` — disclose whenever a `Session` is available (useful for testing the
  disclosure path or standardizing judge inputs).
- `"never"` — always inline, restoring the prior behavior where a genuine overflow
  surfaces as a judge context-length error.

> Public API change → needs `needs-api-review`.

## How it works

- `_trace_index.py` (internal, underscore-prefixed): `TraceIndex` builds an
  in-memory index over the flattened, time-ordered spans. `for_judge()` returns
  `(overview_section, tools)`. Every tool return is capped at `max_read_chars` and
  paged, so no single call can overflow the judge in one shot.
- Base `Evaluator` gains the disclosure seam: `_render_with_disclosure` preflights
  the inline prompt (via `would_exceed_context`, tiktoken when available else a
  chars/4 estimate with a 0.65 safety margin), and swaps in the overview + tools
  when it would overflow. Judge context windows are looked up per model id
  (nova-micro 128K, nova-lite/pro 300K, claude 200K, else 200K default).
- All 18 judge evaluators are wired through the seam. Tool-level and skill
  evaluators resolve the index once per case rather than per tool/skill.

## Scope / non-goals

- This does **not** change how a real overflow is *reported* when disclosure is
  off or unavailable — under `"never"` (or if no `Session` is present) an overflow
  still surfaces as a judge context-length error. Mapping overflow to a
  could-not-evaluate status is a separate change (#399).
- `TraceIndex` is internal; callers do not construct it directly.

## Testing

- `tests/strands_evals/evaluators/test_evaluator_disclosure.py` — disclosure-mode
  validation, resolve-index matrix, inline byte-identity on the fits-path,
  prompt/tool swap on overflow, and e2e auto/never behavior.
- `tests/strands_evals/tools/` — `TraceIndex` overview paging, get/search
  windowing, regex vs literal search, whitespace normalization, empty-session and
  empty-pattern guards, and evaluator integration.
- `tests_integ/test_trace_index_judge_reliability.py` — judge reliability on
  oversized traces.
- Full suite: 2198 passed. mypy clean, ruff clean.
