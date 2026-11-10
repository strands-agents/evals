"""Rebuilding a run's state from its event log."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
from .events import _encode, closed_events
from .run_directory import RunDirectory


@dataclass(frozen=True)
class FitReplay:
    """One `fit` event, replayed in order on resume.

    Attributes:
        ids: The candidates fitted.
        is_fitted: False when the step scored nothing new or the model returned None or raised.
        error: `"<Type>: <message>"` when the model raised; then no fit is in use from there.
    """

    ids: list[str]
    is_fitted: bool
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
            key, `rollouts + sum(a.rollouts for a in previous_attempts)` (re-runs included).
        iteration: The last iteration with an `iteration_end` event (0 = only setup and the baseline).
        candidates: Every candidate proposed in a closed iteration, for duplicate checks (a killed
            iteration's are proposed again as new); their scores and the fit are rebuilt from their saved trials.
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
        is_test_done: A `test` event exists, so the end phase is finished (a None `test` is a finished
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
    is_test_done: bool
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
    `runs_per_candidate`, `agent_failure_score`, `feasibility_threshold`, seed, `proposals_per_iteration`,
    `rerun_passes`, and the final pick (`FinalPick.from_config`). Budget, `max_workers`, and `on_iteration` may differ.

    Args:
        run: The reopened run directory.
        config: The resumed optimizer's config, built by the code that wrote `run.json`.

    Returns:
        The state up to the last `iteration_end` event.

    Every search field comes only from closed iterations (`closed_events`), so a killed iteration is
    discarded and runs again; trials are always read, and `spent` counts every one.

    Raises:
        ValueError: A field that may not change differs, naming the field, e.g.
            "objective changed: 'Helpfulness' -> 'WeightedSum(...)'".
    """
    given = json.loads(json.dumps(dict(config), default=_encode, allow_nan=False))
    for field in _COMPARED:
        changed = _difference(run.config.get(field), given.get(field), field)
        if changed is not None:
            raise ValueError(f"{changed}; start a new run to change it")

    events = run.events.read()
    trials: dict[tuple[str, int, str], dict[str, Any]] = {}
    for event in events:
        if event["event"] == "trial":
            record = event["record"]
            trials[(record["candidate_id"], record["run_index"], record["case_name"])] = record
    spent = sum(r["rollouts"] + sum(a["rollouts"] for a in r["previous_attempts"]) for r in trials.values())

    state = ResumeState(
        parents={},
        spent=spent,
        iteration=0,
        candidates=[],
        exhausted=set(),
        feedback={},
        history=[],
        open_rounds=[],
        fits=[],
        iterations=[],
        search_cost=Usage.zero(),
        finalists=None,
        validation=None,
        is_test_done=False,
        test=None,
        test_reason=None,
        front_test={},
        stop_reason=None,
    )
    for event in closed_events(events):
        name = event["event"]
        if name == "propose":
            candidate_id = event["candidate_id"]
            if not event["rescore"] and candidate_id not in state.parents:
                state.parents[candidate_id] = event["parent_id"]
                state.candidates.append(candidate_id)
            if event.get("patch") is not None:
                state.open_rounds.append(
                    OpenRound(
                        tuple(event["round"]), candidate_id, event["origin"], Patch.model_validate(event["patch"])
                    )
                )
        elif name == "duplicate" and event.get("patch") is not None:
            state.open_rounds.append(
                OpenRound(
                    tuple(event["round"]), event["candidate_id"], event["block"], Patch.model_validate(event["patch"])
                )
            )
        elif name == "unscored":
            state.open_rounds = [
                replace(r, opener_id=None) if r.opener_id == event["candidate_id"] else r for r in state.open_rounds
            ]
        elif name == "observe":
            observation = Observation.model_validate(event["observation"])
            state.feedback.setdefault(event["block"], []).append(observation)
            if observation.patch.usage is not None:
                state.search_cost = state.search_cost + observation.patch.usage
            # An opener's deferred observe closes its round.
            for index, open_round in enumerate(state.open_rounds):
                if open_round.block == event["block"] and open_round.patch == observation.patch:
                    del state.open_rounds[index]
                    break
        elif name == "evidence":
            state.feedback.setdefault(event["block"], []).append(Evidence.model_validate(event["evidence"]))
        elif name == "select":
            state.history.append(event["parent_id"])
        elif name == "fit":
            state.fits.append(FitReplay(list(event["candidates"]), event["is_fitted"], event.get("error")))
        elif name == "iteration_end":
            record = IterationRecord.model_validate(event["record"])
            state.iterations.append(record)
            state.iteration = record.iteration
            for block in record.blocks:
                if block in event["exhausted"]:
                    state.exhausted.add(block)
                else:
                    state.exhausted.discard(block)
        elif name == "finalists":
            state.finalists = list(event["candidate_ids"])
        elif name == "picked":
            state.validation = ValidationReport.model_validate(event["report"])
        elif name == "test":
            state.is_test_done = True
            state.test = None if event["report"] is None else TestReport.model_validate(event["report"])
            state.test_reason = event.get("reason")
            state.front_test = {key: TestReport.model_validate(report) for key, report in event["front"].items()}
        elif name == "stop":
            state.stop_reason = event["reason"]
    return state


# What `Optimizer.resume` is given again; every other `run.json` field is read from it.
_COMPARED = ("cases", "space", "objective", "constraints", "search", "selection", "score_model")


def _difference(old: Any, new: Any, path: str) -> str | None:
    """Where `old` and `new` first differ, as `"<path> changed: <old> -> <new>"`; None when they are equal.

    Lists of `{"key": ...}` records (the space) are matched by key, so the message names the parameter.
    """
    if isinstance(old, list) and isinstance(new, list):
        if all(isinstance(item, dict) and "key" in item for item in [*old, *new]):
            old_by_key = {item["key"]: item for item in old}
            new_by_key = {item["key"]: item for item in new}
            pairs = [
                (f"{path}[{key!r}]", old_by_key.get(key), new_by_key.get(key)) for key in {**old_by_key, **new_by_key}
            ]
        elif len(old) == len(new):
            pairs = [(f"{path}[{index}]", a, b) for index, (a, b) in enumerate(zip(old, new, strict=True))]
        else:
            pairs = None
    elif isinstance(old, dict) and isinstance(new, dict):
        pairs = [(f"{path}.{key}", old.get(key), new.get(key)) for key in {**old, **new}]
    else:
        pairs = None
    if pairs is not None:
        return next((found for at, a, b in pairs if (found := _difference(a, b, at)) is not None), None)
    if old == new:
        return None
    shown = [text if len(text := repr(value)) <= 200 else text[:197] + "..." for value in (old, new)]
    return f"{path} changed: {shown[0]} -> {shown[1]}"
