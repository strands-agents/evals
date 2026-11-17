"""Rebuilding a run's state from its event log."""

from __future__ import annotations

import json
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
    ValidationReport,
)
from .events import _encode, closed_events
from .run_directory import RunDirectory

# What `Optimizer.resume` is given again; every other `run.json` field is read from it.
_COMPARED = ("cases", "space", "objective", "constraints", "search", "selection", "score_model")


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
    """A `Nested` round started but not finished."""

    round: tuple[int, ...]
    opener_id: str
    block: str
    patch: Patch


@dataclass(frozen=True)
class DrawReplay:
    """The killed iteration's `draw` event: what its proposers returned, replayed when it runs again.

    Attributes:
        iteration: The killed iteration.
        parent_id: The parent the patches apply to.
        blocks: The step's block names.
        n_candidates: How many candidates the step was asked for.
        draws: Per redraw (0 first), each proposal's patches (block name -> `Patch`) in the order returned.
    """

    iteration: int
    parent_id: str
    blocks: list[str]
    n_candidates: int
    draws: list[list[dict[str, Patch]]]


@dataclass
class ResumeState:
    """State up to the last closed iteration (one ending in `iteration_end`, or the partial iteration a `stop`
    ends); a killed iteration is discarded and re-run, on its recorded proposals when it has a `draw` event.

    Attributes:
        parents: Candidate id -> parent id (None for the baseline).
        spent: Rebuilt from the saved trial records, not from `score` events: over the last record per
            key, `rollouts + sum(a.rollouts for a in previous_attempts)` (re-runs included).
        iteration: The last iteration with an `iteration_end` event (0 = only setup and the baseline).
        candidates: Every candidate proposed in a closed iteration, for duplicate checks (a killed
            iteration's are saved again when it runs again); their scores and the fit are rebuilt from their
            saved trials.
        exhausted: Blocks that produced zero candidates on their latest turn.
        feedback: Per block name, its `observe` and `evidence` events interleaved in event order; replayed
            through each block's proposer before the first new iteration.
        history: Parents chosen so far (`SelectionContext.history`).
        open_rounds: `Nested` rounds started but not finished, outermost first.
        fits: Each `fit` event, in order: fitted ones are refitted, unfitted ones without an error kept the
            previous fit, ones with an error mean no fit in use from there.
        iterations: Rebuilt from the events, so the result covers the whole run.
        finalists: Already chosen by the end phase, reused rather than re-ranked.
        validation: The end phase already picked; only the test report is left.
        is_test_done: A `test` event exists, so the end phase is finished (a None `test` is a finished
            result too: skipped).
        test: The `test` event's report; None when skipped or not reached.
        test_reason: Why it was skipped (the `test` event's reason); None otherwise.
        front_test: The `test` event's front reports (`{}` unless the objective is a `Pareto`).
        test_failed_ids: The `test` event's tested candidates with no completed test trial.
        stop_reason: The last `stop` event's reason.
        stop_max_rollouts: The `max_rollouts` the search stopped under (the last `stop` event's); None before
            the search stopped, and for a `stop` event written before it recorded the budget.
        trials: (candidate id, run index, case name) -> the last `trial` event's saved record by the end of the
            closed iterations (every trial once the search stopped), so scores are rebuilt as the uninterrupted run
            had them (`RunDirectory.record_from` makes it a `TrialRecord`); a trial a killed iteration replaced is
            still a cache hit when that iteration runs again.
        draw: The killed iteration's last `draw` event (iteration `iteration + 1`, read across earlier resumes),
            so it runs again on the same proposals instead of asking its proposers again; None when the search
            stopped or the kill came before every proposal of that iteration was in.
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
    finalists: list[str] | None
    validation: ValidationReport | None
    is_test_done: bool
    test: TestReport | None
    test_reason: str | None
    front_test: dict[str, TestReport]
    test_failed_ids: list[str]
    stop_reason: StopReason | None
    stop_max_rollouts: int | None
    trials: dict[tuple[str, int, str], Mapping[str, Any]]
    draw: DrawReplay | None


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
        The state up to the last closed iteration.

    Every search field comes only from closed iterations (`closed_events`), so a killed iteration is
    discarded and runs again, on the proposals returned in `draw`; trials are always read, and `spent`
    counts every one.

    Raises:
        ValueError: A field that may not change differs, naming the field, e.g.
            "objective changed: 'Helpfulness' -> 'WeightedSum(...)'".
    """
    given = json.loads(json.dumps({field: config.get(field) for field in _COMPARED}, default=_encode, allow_nan=False))
    for field in _COMPARED:
        changed = _difference(run.config.get(field), given.get(field), field)
        if changed is not None:
            raise ValueError(f"{changed}; pass the run's {field} to resume it, or start a new run to change it")

    events = run.events.read()
    # The last trial per key, as RunDirectory.open indexed it.
    trials = run._trials.values()
    spent = sum(r["rollouts"] + sum(a["rollouts"] for a in r["previous_attempts"]) for r in trials)

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
        finalists=None,
        validation=None,
        is_test_done=False,
        test=None,
        test_reason=None,
        front_test={},
        test_failed_ids=[],
        stop_reason=None,
        stop_max_rollouts=None,
        trials={},
        draw=None,
    )
    closed = closed_events(events)
    kept = {id(event) for event in closed}
    if any(event["event"] == "stop" for event in closed):
        last = len(events) - 1  # the search is over: every trial counts
    else:
        last = max(
            (i for i, e in enumerate(events) if id(e) in kept and e["event"] not in ("config", "trial", "resume")),
            default=-1,
        )
    for event in events[: last + 1]:
        if event["event"] == "trial":
            record = event["record"]
            state.trials[(record["candidate_id"], record["run_index"], record["case_name"])] = record
    for event in closed:
        name = event["event"]
        if name == "propose":
            candidate_id = event["candidate_id"]
            if not event["rescore"] and candidate_id not in state.parents:
                state.parents[candidate_id] = event["parent_id"]
                state.candidates.append(candidate_id)
            for block, patch in (event.get("deferred_patches") or {}).items():  # one round per deferred block
                state.open_rounds.append(
                    OpenRound(tuple(event["round"]), candidate_id, block, Patch.model_validate(patch))
                )
        elif name == "duplicate" and event.get("patch") is not None:
            state.open_rounds.append(
                OpenRound(
                    tuple(event["round"]), event["candidate_id"], event["block"], Patch.model_validate(event["patch"])
                )
            )
        elif name == "observe":
            observation = Observation.model_validate(event["observation"])
            state.feedback.setdefault(event["block"], []).append(observation)
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
            state.test_failed_ids = list(event.get("failed", []))
        elif name == "stop":
            state.stop_reason = event["reason"]
            state.stop_max_rollouts = event.get("max_rollouts")
    if state.stop_reason is None:  # the killed iteration runs again; a later attempt's draw replaces an earlier one
        for event in events:
            if event["event"] == "draw" and event["iteration"] == state.iteration + 1:
                state.draw = DrawReplay(
                    event["iteration"],
                    event["parent_id"],
                    list(event["blocks"]),
                    event["n_candidates"],
                    [[{b: Patch.model_validate(p) for b, p in d.items()} for d in redraw] for redraw in event["draws"]],
                )
    return state


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
