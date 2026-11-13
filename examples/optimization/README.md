# Example optimization runs: what a run saves

`runs/` holds two finished optimizer runs, in the format the strands-evals optimizer writes
(`strands_evals.optimization`, run record `schema_version` 1). They exist so you can build against the format
before the optimizer itself is finished.

**The data is partly artificial.** The trials are real: a scripted agent and judge ran through the real trial
runner, so trajectories, usage, statuses, and evaluator outputs are what the code produces. The search
decisions around them are hand-written in the final format: which candidates were proposed, the fit, labels,
and the end phase. Their numbers are made up but consistent with the trials. Every fit estimate, `p_met`,
diagnostic, and `effects.json` value is invented. The field names and shapes are the real ones. A few
optimizer events may still gain or lose a field before the optimizer is merged.

## The two runs

The agent answers "What is the capital of X?". The baseline guesses from memory and often names the largest
city. Prompts that tell it to call a `lookup` tool get every answer right. Two evaluators score each answer:
`ExactAnswer` (deterministic, 0 or 1) and `Helpfulness` (an LLM judge, 0 to 1). Seven cases: 3 train,
2 validation, 2 test.

| Run | Objective | What it shows |
|---|---|---|
| `geo-<time>-<hex>/` | `WeightedSum(Maximize('ExactAnswer'), Maximize('Helpfulness'))`, limit `tokens <= 600` | One `Nested` round (an opener, then 4 inner steps); every outcome a proposal can have; every trial status; a refit that failed; a kill and resume; a re-scored candidate; a budget stop; then validation, the test report, effects, and the diff |
| `geo-pareto-<time>-<hex>/` | `Pareto(Maximize('ExactAnswer'), Minimize('tokens', scale=500))` | The proposers' per-objective values (`term_values`), the validated front (one member also changed `window_size`, so its `front/*.values.json` isn't empty), a test report per front member, `front/` files, a stop on a proposer error |

What happens in `geo-…`, by `iteration`:

| Iteration | Events to look at |
|---|---|
| 0 | the baseline: `propose` #0, its `trial`s (one repaired by a re-run pass, one re-judged), `score`, `fit` (not fitted: fewer than 2 candidates), `label` |
| 1 | round `[1]` opens: `select`, `propose` #1 with `role: "opener"` and a `patch` (its `observe` comes at the round's end), `evidence` for the other blocks, `iteration_end` |
| 2 | `proposer_retry`, `invalid` (+ its `observe` with `candidate_id: null`), `screened`, #2 (`feasible`), #3 (`infeasible`: over the token limit, one `agent_failure` trial) |
| 3 | #4 (`ineligible`: one `infrastructure_invalid` trial after 2 attempts), a `duplicate` of #2, a `fit` with `error` |
| 4 | `select` and `propose` #5 and one `trial`, then the process is killed. A `resume` event follows, then iteration 4 again: `select`, `propose` #5 again (same number), the remaining trials |
| 5 | #4 proposed again (`rescore: true`: only its failed trial runs again, now `feasible`), #6 `unscored` (budget), the opener's `observe` (round end, `source_id` = the round's best), `iteration_end`, `stop` (`budget`) |
| end | `score` on validation (one finalist has an `evaluator_invalid` trial), `finalists`, `picked`, `score` on test, `test` |

## Folder layout

```
<run_id>/                       [<name>-]<YYYYMMDD-HHMMSS UTC>-<6 hex>; the name is for people, never parsed
├── run.json                    the run's configuration (also the first line of events.jsonl)
├── events.jsonl                everything that happened, one JSON object per line, append-only
├── transcripts/<hash>.json     one trial's trajectory, stored once by content hash
├── effects.json                per-option effects from the score model, written at the end
├── best.diff                   the shipped candidate's text changes vs the baseline, for review
├── best.values.json            its changed numbers and choices: {key: {"old": v, "new": v}}
├── front/<number>-<id>.diff    (Pareto only) the same pair for each member of the validated front
├── front/<number>-<id>.values.json
└── candidates/<id>/optimization_files/   the candidate's copy of the user's files, with its edits
                                (only for values the agent reads from a file path)
```

`events.jsonl` is the only source of truth. Everything else can be derived from it, except the transcripts
(referenced from it) and the file copies.

## Reading `events.jsonl`

Every line has `event` (the type), `iteration` (0 = setup and the baseline), and `time` (ISO 8601, for display
only). Events of the end phase carry the last search iteration's number.

Rules a reader needs:

1. **Candidates are keyed by `candidate_id`** (12 hex: the hash of their values). `number` (0 = baseline,
   then 1, 2, …) is only for showing to people. Numbers can have gaps; #6 in `geo-…` was proposed but never
   scored.
2. **A candidate's values are in its first `propose` event.** A later `propose` with `rescore: true`
   re-scores the same candidate and has no `values`.
3. **A trial is keyed by `(candidate_id, run_index, case_name)`. The last `trial` event per key wins.** Earlier
   ones were replaced by a re-run or re-judge; the winner's `previous_attempts` describes them. Trial events
   of candidates scored at the same time interleave; every other event comes in a fixed order.
4. **Killed iterations.** Events of an iteration with no `iteration_end` (and no `stop`) before the next
   `resume` were cut off by a kill. That iteration runs again after the `resume`. Its trials are still valid
   and are reused as cache hits; its other events are superseded.
5. **Records nest whole:** `label`, `observation`, `evidence`, `record`, `report` hold the record objects
   below, as JSON.

### Event types

| Event | Fields |
|---|---|
| `config` | `config` = the contents of `run.json` |
| `propose` | `candidate_id`, `number`, `parent_id`, `origin` (block name, or `"baseline"`), `values`, `changed_keys`, `rationale` (the proposer's reason), `inherited` (key → the candidate it was copied from), `defaulted` (keys set to their default), `round` (round path, e.g. `[1]`, or null), `role` (`"opener"` / `"inner"` / null), `rescore`; `patch` on a round opener |
| `trial` | `record`: a trial record (below); its trajectory is replaced by `transcript: {hash, format}` |
| `score` | `candidate_id`, `split`, `case_names`, `runs`, `rollouts` (spent by this call), `objective_by_run` (case → one objective value per run; null = not measured), `statuses` (status → count), `unavailable` (case → run → metric → why a value was dropped) |
| `fit` | `candidates` (ids fitted), `is_fitted`, `error` (when the score model raised), `flagged`, `diagnostics` (per metric: held-out checks, prior scales, scale, noise, observation count) |
| `select` | `parent_id`, `strategy` |
| `label` | `candidate_id`, `label`: `outcome` (`feasible` / `infeasible` / `ineligible`), `reason`, `unmet_constraints`, `estimates` (fit estimate per goal metric), `p_met` (constraint → probability it is met), `is_from_fit`, `measured` (per metric `{mean, ci95, n}`) |
| `invalid` | `block`, `redraw`, `updates`, `reason`: a proposal rejected before scoring |
| `duplicate` | `block`, `redraw`, `candidate_id` (the existing candidate it repeats) |
| `unscored` | `candidate_id`, `pending`, `remaining`, `reserve`: the budget couldn't cover it |
| `proposer_retry` / `proposer_error` | `block`, `redraw`, `error_type`, `error`; an error ends the search |
| `screened` | `block`, `key`, `reason`: a proposal dropped for quoting the training data |
| `observe` | `block`, `candidate_id`, `observation`: what the proposer was told about its own proposal (`patch` incl. its own model `usage`, `objective`, `term_values`, `outcome`, `reason`, `violations`, `source_id`) |
| `evidence` | `block` (the proposer told), `evidence`: another block's candidate and its outcome |
| `iteration_end` | `record` (iteration, parent, round, role, blocks, per-proposal outcomes, rollouts spent), `exhausted`, `best_after` (the top candidates at that point) |
| `resume` | `spent`, `budget`, `code` (version, commit, dirty of the resuming code) |
| `stop` | `reason` (`budget`, `iterations`, `exhausted`, `proposer_error`, `setup`), `message` |
| `finalists` | `candidate_ids`, ranked: what goes to validation |
| `picked` | `report`: `champion_id`, `front` (Pareto), `objectives`, `estimates`, `measured`, `infeasible`, `problem`, `reason` |
| `test` | `report`: champion vs baseline on test (`objective`, `difference` and `ratio` per metric as `{mean_diff, ci_low, ci_high, n_cases}` at 90%, `means`, `missing`), or null with `reason`; `front`: one report per front member (Pareto) |

## Trial records

One per (candidate, case, run): `candidate_id`, `case_name`, `run_index`, `split`, `status`, `error_type`,
`error`, `output`, `interactions`, `environment_state`, and:

- `evaluations`: evaluator name → `{score, test_pass, reason, label, metadata}`. `details` holds each
  evaluator's per-row outputs.
- `evaluator_errors`: evaluator name → exception type. An evaluator is in `evaluations` or here, never both.
- `usage`: the agent's own spend: `tokens`, `input_tokens`, `output_tokens`, `cache_read_tokens`,
  `cache_write_tokens` (null = not reported, not 0), `model_calls`, `seconds`, `models`.
- `judge_usage`: evaluator name → the same shape: what each LLM judge spent. Kept apart from `usage`.
- `rollouts`: what this attempt cost the budget (1 when the agent ran, 0 for a re-judge on a saved output).
- `attempts` and `previous_attempts`: one `{status, error_type, error, usage, judge_usage, pass_index,
  rollouts}` per earlier attempt this record replaced.

| `status` | Meaning | In `geo-…` |
|---|---|---|
| `completed` | measured normally (an evaluator may still have failed: see `evaluator_errors`) | most trials |
| `agent_failure` | the agent raised; scored at each evaluator's worst end | #3, `capital_france` |
| `infrastructure_invalid` | the environment or model service failed; nothing measured; re-run later | #4, `capital_canada` (iteration 3) |
| `evaluator_invalid` | every evaluator failed; nothing measured | the last finalist on `capital_turkey` (validation) |

Transcripts (`transcripts/<hash>.json`) are a `Session` (`format: "session"`): `{session_id, traces: [{trace_id,
session_id, spans: [{span_type, span_info, messages, metadata}]}]}` with the agent's model calls, tool calls,
and messages. The judge's calls are never in it. `format: "list"` means the agent returned its own list.

## Costs

Nothing stores totals. Compute them from the events:

- **Rollouts spent:** over the last trial per key, `rollouts` plus each previous attempt's `rollouts`.
- **Agent cost:** the sum of `usage` over the last trial per key, per split.
- **Judge cost:** the sum of every `judge_usage` value over the last trial per key.
- **Retry cost:** `usage` and `judge_usage` of every `previous_attempts` entry.
- **Search cost:** the sum of `observation.patch.usage` over `observe` events (the proposers' own model spend).

## `run.json`

`cases`; `split` (settings, seed, the three case lists); `space` (every tunable parameter: `key`, `kind`,
`default`, bounds, `options`, `active_when`, file `source`; text defaults as `{sha256, length}`, never the
text); `objective` and `constraints` (as text); `metrics` (each goal metric's kind, roles, range, weight,
scale); `search` (blocks, proposers, schedule); `selection`; `score_model`; `final_pick`; `budget`;
`runs_per_candidate`; `agent_failure_score`; `feasibility_threshold`; `seed`; `proposals_per_iteration`;
`rerun_passes`; `max_workers`; `agent` (the optimized function, `module:qualname`); `name`; `workdir`
(diff paths are relative to it); `schema_version`; `code` (the optimizer version and git commit that wrote it).

Values are dotted keys. A key under a choice (`conversation_manager.summary.summary_ratio`) exists only while
that choice is selected (`active_when`).

## `effects.json`, diffs

- `effects.json`: `effects` (per metric and option: `key`, `level`, `reference`, `mean`, `ci90`, `p_positive`),
  `factors`, `omitted`, `goals`, `metrics` (the score model's scale, noise, diagnostics), `reason` (why
  `effects` is empty, when it is).
- `best.diff`: a unified diff of the shipped candidate's text changes. Hunks on files use the file's path;
  text that isn't in a file is headed `--- <key>`. For review only, not `git apply`. Empty when the baseline
  ships.
- `best.values.json`: the numbers and choices that changed. A key active on one side only has only that
  side's entry.

## Things to know

- `run.json` (`workdir`, `space[].source.path`) holds absolute paths from the machine that wrote the run.
- An evaluator's `label` is filled in `evaluations` but null in `details`. That's how the evaluation library
  reports it today, not an error in the record.

## Regenerating

From a checkout of the optimizer branch (`optimization-record`), with the test environment (it has scipy):

```bash
cd ../evals-candidate
hatch env run -e hatch-test.py3.14 -- python ../real_examples/24_example_run.py
```

This replaces `runs/` with new runs: new ids, timestamps, and folder names.
