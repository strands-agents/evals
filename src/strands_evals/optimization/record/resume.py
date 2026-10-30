"""Rebuilding a run's state from its event log."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...types.optimization import (
    Evidence,
    IterationRecord,
    Observation,
    Patch,
    StopReason,
    TestReport,
    Usage,
    ValidationReport,
)
from .run_directory import RunDirectory


@dataclass(frozen=True)
class FitReplay:
    """One `fit` event, replayed in order on resume.

    Attributes:
        ids: The candidates fitted.
        fitted: False when the step scored nothing new or the model returned None or raised.
        error: `"<Type>: <message>"` when the model raised; then no fit is in use from there.
    """

    ids: list[str]
    fitted: bool
    error: str | None


@dataclass(frozen=True)
class OpenRound:
    """A `Nested` round started but not finished; `opener_id` None when the opener was unscored."""

    round: tuple[int, ...]
    opener_id: str | None
    block: str
    patch: Patch


@dataclass
class ResumeState:
    """State up to the last `iteration_end` event; a killed iteration is discarded and re-run.

    Attributes:
        parents: Candidate id -> parent id (None for the baseline).
        spent: Rebuilt from the saved trial records, not from `score` events: over the last record per
            key, `rollouts + sum(a.rollouts for a in earlier)` (re-runs included).
        iteration: The last iteration with an `iteration_end` event (0 = only setup and the baseline).
        candidates: Every saved candidate; their scores and the fit are rebuilt from their saved trials.
        exhausted: Blocks that produced zero candidates on their latest turn.
        feedback: Per block name, its `observe` and `evidence` events interleaved in event order; replayed
            through each block's proposer before the first new iteration.
        history: Parents chosen so far (`SelectionContext.history`).
        open_rounds: `Nested` rounds started but not finished, outermost first.
        fits: Each `fit` event, in order: fitted ones are refitted, unfitted ones without an error kept the
            previous fit, ones with an error mean no fit in use from there.
        iterations: Rebuilt from the events, so the result covers the whole run.
        search_cost: Summed from the `observe` events' `Patch.usage`.
        finalists: Already chosen by the end phase, reused rather than re-ranked.
        validation: The end phase already picked; only the test report is left.
        test_done: A `test` event exists, so the end phase is finished (a None `test` is a finished
            result too: skipped).
        test: The `test` event's report; None when skipped or not reached.
        test_reason: Why it was skipped (the `test` event's reason); None otherwise.
        front_test: The `test` event's front reports (`{}` unless the objective is a `Pareto`).
        stop_reason: The last `stop` event's reason.
    """

    parents: dict[str, str | None]
    spent: int
    iteration: int
    candidates: list[str]
    exhausted: set[str]
    feedback: dict[str, list[Observation | Evidence]]
    history: list[str]
    open_rounds: list[OpenRound]
    fits: list[FitReplay]
    iterations: list[IterationRecord]
    search_cost: Usage
    finalists: list[str] | None
    validation: ValidationReport | None
    test_done: bool
    test: TestReport | None
    test_reason: str | None
    front_test: dict[str, TestReport]
    stop_reason: StopReason | None


def load_state(run: RunDirectory, *, config: Mapping[str, Any]) -> ResumeState:
    """Compare `config` with `run.json` field by field, then rebuild the state from the event log.

    Compared (what `Optimizer.resume` is given again): cases, space (every stored `ParamInfo` field),
    objective and constraints, search (blocks, proposers and their `config()`, keys, schedule name and
    `config()`, redraws), selection (name and `config()`), score model (name and `config()`).
    Read (never passed again, rebuilt from `run.json`): split (settings, seed used, and case lists),
    `runs_per_candidate`, `crash_score`, `feasibility`, seed, `proposals_per_iteration`, `rerun_passes`,
    and the final pick (`FinalPick.from_config`). Budget, `max_workers`, and `on_iteration` may differ.

    Args:
        run: The reopened run directory.
        config: The resumed optimizer's config, built by the code that wrote `run.json`.

    Returns:
        The state up to the last `iteration_end` event.

    Raises:
        ValueError: A field that may not change differs, naming the field, e.g.
            "objective changed: 'Helpfulness' -> 'WeightedSum(...)'".
    """
    raise NotImplementedError
