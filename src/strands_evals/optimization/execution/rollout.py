"""The default rollout, the task wrappers `Experiment` drives, and the per-run trial log."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from ...case import Case
from ...types.optimization import TrialStatus
from .traces import TraceCapture

TaskFn = Callable[[Case], Any]
RolloutFn = Callable[[Callable[..., Any], Case], Any]
EnvironmentFn = Callable[
    [Case],
    Mapping[str, Any] | AbstractContextManager[Mapping[str, Any]] | AbstractAsyncContextManager[Mapping[str, Any]],
]


@dataclass
class TrialLogEntry:
    """What the task wrapper recorded for one trial.

    Attributes:
        status: The last attempt's status from `classify`, or `completed` when the rollout returned
            (`verifier_invalid` is set later, by `TrialRunner`).
        error_type: The last attempt's exception type name; None when it returned.
        error: The last attempt's exception message; None when it returned.
        attempts: Every attempt, the last one included.
        trace_id: The trial's trace (its `execute_case` span), which groups its agent and judge spans.
        attempt_span_id: The last attempt's span.
        earlier: `(span id, status, error_type, error)` for each attempt `Experiment` retried
            (throttling), oldest first.
    """

    status: TrialStatus
    error_type: str | None
    error: str | None
    attempts: int
    trace_id: int
    attempt_span_id: int
    earlier: list[tuple[int, TrialStatus, str | None, str | None]]


@dataclass
class TrialLog:
    """One run's trial status, written by `instrument`, keyed by case name.

    Lives here, not in `trials.py`, so the two modules don't import each other.

    Attributes:
        setup_error: A `_FactoryArgumentError` the bound factory raised, else None; `TrialRunner.run`
            raises it once the `Experiment` call returns.
    """

    setup_error: TypeError | None = None
    _entries: dict[str, TrialLogEntry] = field(default_factory=dict, repr=False)

    def record(self, case_name: str, entry: TrialLogEntry) -> None:
        """Store one trial's entry; called by `instrument`.

        Args:
            case_name: The case name.
            entry: What the wrapper recorded for that case's trial.
        """
        raise NotImplementedError

    def records(self) -> dict[str, TrialLogEntry]:
        """Case name -> the entry `instrument` recorded for its trial."""
        raise NotImplementedError


async def default_rollout(agent_factory: Callable[..., Any], case: Case) -> dict[str, Any]:
    """Build an agent and return `{"output": str(await agent.invoke_async(case.input))}`.

    Args:
        agent_factory: Builds a fresh agent; called with no arguments.
        case: The case to run.
    """
    raise NotImplementedError


def make_task(
    agent_factory: Callable[..., Any],
    rollout_fn: RolloutFn,
    environment: EnvironmentFn | None = None,
) -> TaskFn:
    """Wrap a rollout as a task: per-attempt environment, then `rollout_fn` inside a `rollout` span.

    `rollout_fn` receives `agent_factory._with_resources(resources)`: `default_rollout` calls it with no
    arguments; a custom rollout may pass the factory function's non-tunable kwargs. A resource or kwarg
    named like a declared marker, or a call kwarg the function doesn't accept, raises
    `_FactoryArgumentError`: a setup error, never a trial status. An error creating or entering the
    environment is raised as `InfrastructureError(...) from error`, so `classify` makes it
    `infrastructure_invalid` whatever its type, and the trial's `error_type` / `error` record the
    original `__cause__`. An error exiting it after the rollout returned only warns, and the trial
    keeps its result.

    Args:
        agent_factory: The candidate's bound factory.
        rollout_fn: Runs one case with a fresh agent.
        environment: Called once per attempt; returns resources or a context manager yielding them.

    Returns:
        `task(case)`, not yet instrumented.
    """
    raise NotImplementedError


def instrument(task: TaskFn, log: TrialLog, traces: TraceCapture) -> TaskFn:
    """Open one span per attempt, classify and re-raise errors, and attach the `Session` as trajectory.

    Called by `TrialRunner.run`, once per run with a fresh `TrialLog`. A `_FactoryArgumentError` is
    stored as `log.setup_error`, not classified. A rollout-returned `trajectory` that is not None wins;
    a returned dict without one gets a copy with the `Session` added, never the dict itself changed;
    a non-dict return becomes `{"output": value, "trajectory": session}`. No agent spans -> trajectory None.
    Attempts `Experiment` retried on throttling go into `TrialLogEntry.earlier`.

    Args:
        task: The task from `make_task` or `build_task`.
        log: A fresh log for this run.
        traces: The run's trace capture.

    Returns:
        The wrapped task that `Experiment` calls.
    """
    raise NotImplementedError
