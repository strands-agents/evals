"""The default rollout, the task wrappers `Experiment` drives, and the per-run trial log."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from ...case import Case
from ...telemetry import get_tracer
from ...types.optimization import TrialStatus
from ..declaration.tunable import _FactoryArgumentError
from .status import InfrastructureError, classify
from .traces import ATTEMPT_SPAN, ROLLOUT_SPAN, SPAN_KIND, TraceCapture

logger = logging.getLogger(__name__)

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
            (`evaluator_invalid` is set later, by `TrialRunner`).
        error_type: The last attempt's exception type name; None when it returned.
        error: The last attempt's exception message; None when it returned.
        attempts: Every attempt, the last one included.
        trace_id: The trial's trace (its `execute_case` span), which groups its agent and judge spans;
            hex, as in a `Session` (`format(trace_id, "032x")`).
        attempt_span_id: The last attempt's span, hex (`format(span_id, "016x")`).
        previous_attempts: `(span id, status, error_type, error)` for each attempt `Experiment` retried
            (throttling), oldest first.
    """

    status: TrialStatus
    error_type: str | None
    error: str | None
    attempts: int
    trace_id: str
    attempt_span_id: str
    previous_attempts: list[tuple[str, TrialStatus, str | None, str | None]]


@dataclass
class TrialLog:
    """One run's trial status, written by `instrument`, keyed by case name.

    Lives here, not in `trials.py`, so the two modules don't import each other.

    Attributes:
        setup_error: The first setup error, else None: a `_FactoryArgumentError` (from the bound factory,
            or an environment that gave no mapping) or a `RuntimeError` (trial spans not recorded).
            `TrialRunner.run` raises it once the `Experiment` call returns.
        trace_ids: Every trace `instrument` registered, with or without an entry (a setup error or a
            cancelled attempt records none), so `TrialRunner.run` can release them all in a `finally`.
    """

    setup_error: Exception | None = None
    trace_ids: set[str] = field(default_factory=set)
    _entries: dict[str, TrialLogEntry] = field(default_factory=dict, repr=False)

    def record(self, case_name: str, entry: TrialLogEntry) -> None:
        """Store one trial's entry; called by `instrument`.

        Args:
            case_name: The case name.
            entry: What the wrapper recorded for that case's trial.
        """
        self._entries[case_name] = entry

    def records(self) -> dict[str, TrialLogEntry]:
        """Case name -> the entry `instrument` recorded for its trial."""
        return dict(self._entries)


async def default_rollout(agent_factory: Callable[..., Any], case: Case) -> dict[str, Any]:
    """Build an agent and return `{"output": str(await agent.invoke_async(case.input))}`.

    Args:
        agent_factory: Builds a fresh agent; called with no arguments.
        case: The case to run.
    """
    agent = agent_factory()
    return {"output": str(await agent.invoke_async(case.input))}


def make_task(
    agent_factory: Callable[..., Any],
    rollout_fn: RolloutFn,
    environment: EnvironmentFn | None = None,
) -> TaskFn:
    """Wrap a rollout as a task: per-attempt environment, then `rollout_fn` inside a `rollout` span.

    `rollout_fn` receives `agent_factory._with_resources(resources)` (without `environment`,
    `agent_factory` itself): `default_rollout` calls it with no arguments; a custom rollout may pass the
    factory function's non-tunable kwargs. A setup error, never a trial status, raises
    `_FactoryArgumentError`: an environment that gives no mapping, a resource or kwarg named like a
    declared marker, a call kwarg the function doesn't accept, or a required parameter left unsupplied.
    An error creating or entering the environment is raised as `InfrastructureError(...) from error`, so
    `classify` makes it `infrastructure_invalid` whatever its type. An error exiting it after the rollout
    returned only warns, and the trial keeps its result. A sync `rollout_fn`, a sync `environment`, and a
    sync context manager's `__enter__` / `__exit__` run in a worker thread, as `Experiment` runs a sync
    task, so a slow setup doesn't stall concurrent trials; a context variable set there doesn't reach
    the rollout.

    Args:
        agent_factory: The candidate's bound factory.
        rollout_fn: Runs one case with a fresh agent.
        environment: Called once per attempt; returns resources or a context manager yielding them.

    Returns:
        `task(case)`, not yet instrumented.
    """

    async def task(case: Case) -> Any:
        if environment is None:
            return await _rollout(rollout_fn, agent_factory, case)
        try:
            provided = await asyncio.to_thread(environment, case)
            if isinstance(provided, AbstractAsyncContextManager):
                resources = await provided.__aenter__()
            elif isinstance(provided, AbstractContextManager):
                resources = await asyncio.to_thread(provided.__enter__)
            else:
                resources = provided
        except Exception as error:
            raise InfrastructureError(f"environment setup failed for case {case.name!r}: {error}") from error
        try:
            if not isinstance(resources, Mapping):
                if inspect.iscoroutine(resources):
                    resources.close()  # an `async def` environment, never awaited
                raise _FactoryArgumentError(
                    f"environment for case {case.name!r} gave {type(resources).__name__}, not a mapping of resources"
                )
            result = await _rollout(rollout_fn, agent_factory._with_resources(resources), case)
        except BaseException as error:
            await _exit_environment(provided, case, error)
            raise
        await _exit_environment(provided, case, None)
        return result

    return task


async def _rollout(rollout_fn: RolloutFn, agent_factory: Callable[..., Any], case: Case) -> Any:
    """Run `rollout_fn` inside the `rollout` span, whose duration is `Usage.seconds`."""
    with get_tracer().start_as_current_span(f"rollout {case.name}", attributes={SPAN_KIND: ROLLOUT_SPAN}):
        return await _call(rollout_fn, agent_factory, case)


async def _exit_environment(provided: Any, case: Case, error: BaseException | None) -> None:
    """Exit an environment context manager; an error exiting it only warns, and never hides `error`."""
    exc_info = (type(error), error, error.__traceback__) if error is not None else (None, None, None)
    try:
        if isinstance(provided, AbstractAsyncContextManager):
            await provided.__aexit__(*exc_info)
        elif isinstance(provided, AbstractContextManager):
            await asyncio.to_thread(provided.__exit__, *exc_info)
    except Exception as exit_error:
        logger.warning("case=<%s>, error=<%s> | environment failed to exit", case.name, exit_error)


async def _call(fn: Callable[..., Any], *args: Any) -> Any:
    """Await an async function; run a sync one in a worker thread (which keeps the current span)."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    result = await asyncio.to_thread(fn, *args)
    if inspect.isawaitable(result):
        result = await result
    return result


def instrument(task: TaskFn, log: TrialLog, traces: TraceCapture) -> TaskFn:
    """Open one span per attempt, classify and re-raise errors, and attach the `Session` as trajectory.

    Called by `TrialRunner.run`, once per run with a fresh `TrialLog`. A setup error is stored as
    `log.setup_error`, not classified: a `_FactoryArgumentError`, or a `RuntimeError` when the attempt
    span isn't recorded (an SDK disabled or sampling out), since every such trial would read as
    `completed` with no usage and no trajectory. A rollout-returned `trajectory` that is not None wins;
    a returned dict without one gets a copy with the `Session` added, never the dict itself changed;
    a non-dict return becomes `{"output": value, "trajectory": session}`. No agent spans -> trajectory None.
    Attempts `Experiment` retried on throttling go into `TrialLogEntry.previous_attempts`. An
    `InfrastructureError` raised from another error records that error's type and message.

    Args:
        task: The task from `make_task` or `build_task`.
        log: A fresh log for this run.
        traces: The run's trace capture.

    Returns:
        The wrapped task that `Experiment` calls.
    """

    async def instrumented(case: Case) -> Any:
        case_name = str(case.name)
        previous = log.records().get(case_name)
        earlier = []
        if previous is not None:
            earlier = [
                *previous.previous_attempts,
                (previous.attempt_span_id, previous.status, previous.error_type, previous.error),
            ]
        with get_tracer().start_as_current_span(
            f"attempt {case_name}", attributes={SPAN_KIND: ATTEMPT_SPAN}
        ) as attempt_span:
            if not attempt_span.is_recording():
                error = RuntimeError(
                    "trial spans aren't recorded (OTEL_SDK_DISABLED or a sampler such as OTEL_TRACES_SAMPLER="
                    "always_off / traceidratio), so the optimizer can't read usage or the trajectory; record every span"
                )
                log.setup_error = log.setup_error or error
                raise error
            context = attempt_span.get_span_context()
            trace_id = format(context.trace_id, "032x")
            attempt_span_id = format(context.span_id, "016x")
            traces.register(trace_id)
            log.trace_ids.add(trace_id)

            def record(status: TrialStatus, error: BaseException | None) -> None:
                if isinstance(error, InfrastructureError) and error.__cause__ is not None:
                    error = error.__cause__
                log.record(
                    case_name,
                    TrialLogEntry(
                        status=status,
                        error_type=type(error).__name__ if error is not None else None,
                        error=str(error) if error is not None else None,
                        attempts=len(earlier) + 1,
                        trace_id=trace_id,
                        attempt_span_id=attempt_span_id,
                        previous_attempts=earlier,
                    ),
                )

            try:
                output = await _call(task, case)
            except _FactoryArgumentError as error:
                log.setup_error = log.setup_error or error
                raise
            except Exception as error:
                record(classify(error), error)
                raise
            record(TrialStatus.COMPLETED, None)
        session = traces.session(trace_id, attempt_span_id)
        if isinstance(output, dict):
            if output.get("trajectory") is not None or session is None:
                return output
            return {**output, "trajectory": session}
        return {"output": output, "trajectory": session}

    return instrumented
