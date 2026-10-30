"""The append-only `events.jsonl` event log."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

EventName = Literal[
    "config",
    "propose",
    "trial",
    "score",
    "fit",
    "select",
    "label",
    "invalid",
    "duplicate",
    "unscored",
    "proposer_retry",
    "proposer_error",
    "resume",
    "stop",
    "observe",
    "evidence",
    "screened",
    "iteration_end",
    "finalists",
    "selected",
    "test",
]


class EventLog:
    """Everything that happened, one JSON event per line; the first line is the run's config.

    Every event has `event`, `iteration` (0 for setup and the baseline), and `time` (ISO 8601, for reading
    only; resume never uses it). Records nest whole under the key shown, via `model_dump`.

    - `config`: the `run.json` contents, including `schema_version`.
    - `propose`: `candidate_id`, `number` (display only), `parent_id`, `origin`, `values` (the only copy of
      the candidate's typed values), `changed_keys`, `rationale`, `inherited`, `defaulted`, `round`, `role`;
      `rescore=true` for an ineligible candidate proposed again (then no `values`, the first number and
      parent kept); `patch` when its observe is deferred (a round's opener).
    - `trial`: `record` (`TrialRecord`, its trajectory replaced by `transcript`: `{hash, format}` or None),
      one per trial as soon as it finishes. A later one for the same (candidate, run, case) replaces it.
      Trials of concurrently scored candidates interleave; every other event is written in a fixed order.
    - `score`: `candidate_id`, `split`, `case_names`, `runs`, `rollouts`, `objective_by_run`, `statuses`
      (status -> trial count), `unavailable`.
    - `fit`: `candidates` (ids fitted), `fitted`, `error` (when the score model raised), `flagged`,
      `diagnostics`; one per step, `fitted=false` also when the step scored nothing new.
    - `select`: `parent_id`, `strategy`.
    - `label`: `candidate_id`, `label` (`CandidateLabel`).
    - `invalid`: `block`, `redraw`, `updates`, `reason` (no candidate).
    - `duplicate`: `block`, `redraw`, `candidate_id` (the existing one); `patch` and `round` for a round's
      opener.
    - `unscored`: `candidate_id`, `pending`, `remaining`, `reserve`.
    - `proposer_retry`, `proposer_error`: `block`, `redraw`, `error_type`, `error`.
    - `screened`: `block`, `key`, `reason`.
    - `resume`: `spent`, `budget`, `code` (the resuming strands-evals: version, git commit, dirty).
    - `stop`: `reason` (a `stop_reason`, or `"setup"` when the baseline isn't eligible), `message`.
    - `observe`: `block`, `candidate_id` (None for an invalid patch), `observation` (`Observation`).
    - `evidence`: `block` (the proposer told), `evidence` (`Evidence`).
    - `iteration_end`: `record` (`IterationRecord`), `exhausted`, `best_after` (a snapshot).
    - `finalists`: `candidate_ids` (ranked). `selected`: `report` (`ValidationReport`). `test`: `report`
      (`TestReport` or None), `reason` (also naming the front members it couldn't afford), `front` (under a
      `Pareto` objective, each tested front member's `TestReport`, the champion's included; `{}` otherwise).

    Resume reads: `spent` from the trial events (the last per key: its `rollouts` plus its earlier
    attempts'); candidates, numbers, parents, and
    rounds from `propose` and `iteration_end`; exhausted blocks and iterations from `iteration_end`; history
    from `select`; open rounds from openers with no observe yet; `stop_reason` from the last `stop`; proposer
    feedback and `search_cost` from `observe` / `evidence`; the fit sequence from `fit`; the end phase's
    progress from `finalists` / `selected` / `test`. Only search events up to the last `iteration_end` are
    replayed; trial events are always read (the cache).
    """

    def __init__(self, path: Path):
        raise NotImplementedError

    def append(self, event: EventName, **fields: Any) -> None:
        """Append one event as a JSON line, with one write and flush; the optimizer is the only writer.

        Args:
            event: The event name.
            **fields: JSON-serializable event fields.
        """
        raise NotImplementedError

    def read(self) -> list[dict[str, Any]]:
        """Every event, dropping a truncated last line."""
        raise NotImplementedError
