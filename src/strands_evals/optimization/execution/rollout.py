"""The default rollout, the task wrappers `Experiment` drives, and the per-run trial log."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import logging
from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from contextlib import AbstractAsyncContextManager, AbstractContextManager, suppress
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.trace import Span
from strands import Agent
from strands.multiagent.base import MultiAgentBase

from ...case import Case
from ...telemetry import get_tracer
from ...types.optimization import TrialStatus
from ..declaration.tunable import _FactoryArgumentError
from .node import NodeAgent
from .status import InfrastructureError, classify
from .traces import ATTEMPT_SPAN, FACTORY_AGENT, FACTORY_TAGGED, ROLLOUT_SPAN, SPAN_KIND, TraceCapture

logger = logging.getLogger(__name__)

TaskFn = Callable[[Case], Any]
RolloutFn = Callable[[Callable[..., Any], Case], Any]
EnvironmentFn = Callable[
    [Case],
    Mapping[str, Any] | AbstractContextManager[Mapping[str, Any]] | AbstractAsyncContextManager[Mapping[str, Any]],
]

# The pool sync functions run in, set by `TrialRunner.run` for its call and sized to its `max_workers`:
# asyncio's shared pool holds at most min(32, cpus + 4) threads, which would cap the sync trials running
# at once. None (outside a `TrialRunner` call) -> that shared pool.
_THREAD_POOL: contextvars.ContextVar[Executor | None] = contextvars.ContextVar("trial_thread_pool", default=None)

# Type names of objects a factory built that couldn't be tagged, each warned about once per process.
_UNTAGGED_WARNED: set[str] = set()


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
    """One trial's status (one case, one run), written by `instrument`, keyed by case name.

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

    The factory runs once per attempt, so models it builds should share one `boto3.Session` rather than
    each make their own (see `Optimizer`).

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

    `rollout_fn` receives a factory that calls `agent_factory._with_resources(resources)` (without
    `environment`, `agent_factory` itself): `default_rollout` calls it with no arguments; a custom rollout
    may pass the factory function's non-tunable kwargs. A setup error, never a trial status, raises
    `_FactoryArgumentError`: an environment that gives no mapping, a resource or kwarg named like a
    declared marker, a call kwarg the function doesn't accept, or a required parameter left unsupplied.
    An error creating or entering the environment is raised as `InfrastructureError(...) from error`, so
    `classify` makes it `infrastructure_invalid` whatever its type. An error exiting it after the rollout
    returned only warns, and the trial keeps its result. A sync `rollout_fn`, a sync `environment`, and a
    sync context manager's `__enter__` / `__exit__` run in a worker thread (in a `TrialRunner` call, its
    pool of `max_workers` threads), so a slow setup doesn't stall concurrent trials. Setup and teardown
    may run on different threads, so a thread-bound resource (e.g. a sqlite connection) must allow that,
    or come from an async context manager, which runs on the event loop. A trial cancelled while a sync
    `__enter__` runs waits for it to finish, exits the environment, then re-raises the cancellation (a
    second cancellation during that wait leaves the environment entered).

    The factory `rollout_fn` receives also tags each strands `Agent`, multi-agent system, or `NodeAgent` it returns,
    through its `trace_attributes`, to tell the candidate's model calls (its agents and everything under
    them: sub-agents, a `ToolSimulator` behind one of its tools) from every other agent's in the attempt,
    such as a simulated user or an agent the environment built. Those are `TrialRecord.benchmark_usage`,
    never the candidate's usage, and Reflection doesn't read them. So an agent the rollout builds itself,
    even from the parts of the one the factory returned, is not the candidate's, and an attempt whose
    factory agent never runs records no candidate usage. Anything else the factory returns (a config dict,
    a wrapper object) can't be tagged: a warning says so once per type, and every model call in the attempt
    counts as the candidate's, as on the `build_task` path.

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
        env_context = contextvars.copy_context()
        try:
            provided = await _in_thread(env_context.run, environment, case)
            if isinstance(provided, AbstractAsyncContextManager):
                resources = await provided.__aenter__()
            elif isinstance(provided, AbstractContextManager):
                entering = asyncio.ensure_future(_in_thread(env_context.run, provided.__enter__))
                try:
                    resources = await asyncio.shield(entering)
                except asyncio.CancelledError as cancelled:
                    # The worker thread can't be stopped: let it finish entering, exit, then cancel.
                    with suppress(Exception):
                        await asyncio.shield(entering)
                        await _exit_environment(provided, env_context, case, cancelled)
                    raise
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
            await _exit_environment(provided, env_context, case, error)
            raise
        await _exit_environment(provided, env_context, case, None)
        return result

    return task


async def _rollout(rollout_fn: RolloutFn, agent_factory: Callable[..., Any], case: Case) -> Any:
    """Run `rollout_fn` inside the `rollout` span, whose duration is `Usage.seconds`, with a tagging factory."""
    attributes: dict[str, str | bool] = {SPAN_KIND: ROLLOUT_SPAN, FACTORY_TAGGED: True}
    with get_tracer().start_as_current_span(f"rollout {case.name}", attributes=attributes) as span:
        return await _call(rollout_fn, _tagging(agent_factory, span), case)


def _tagging(agent_factory: Callable[..., Any], rollout_span: Span) -> Callable[..., Any]:
    """`agent_factory`, setting `FACTORY_AGENT` in the `trace_attributes` of each strands `Agent`, multi-agent
    system, or `NodeAgent` (set on its spans in Node) it builds (an async factory's once awaited).

    Strands reads `trace_attributes` at each invocation, so the agent's spans carry the tag. Arguments are
    passed through unchanged. Anything else built is returned as is, sets `FACTORY_TAGGED` False on the rollout
    span, and is warned about once per type.
    """

    def tag(built: Any) -> Any:
        attributes = getattr(built, "trace_attributes", None)
        if isinstance(built, (Agent, MultiAgentBase, NodeAgent)) and isinstance(attributes, dict):
            attributes[FACTORY_AGENT] = True
            return built
        rollout_span.set_attribute(FACTORY_TAGGED, False)
        if type(built).__qualname__ not in _UNTAGGED_WARNED:
            _UNTAGGED_WARNED.add(type(built).__qualname__)
            logger.warning(
                "built=<%s> | the factory returned no strands agent or multi-agent system, so other agents in the "
                "rollout can't be told apart | every model call counts as the candidate's and reflection reads every "
                "agent | return the agent itself to tell them apart",
                type(built).__qualname__,
            )
        return built

    async def tag_when_built(built: Any) -> Any:
        return tag(await built)

    def build(*args: Any, **kwargs: Any) -> Any:
        built = agent_factory(*args, **kwargs)
        return tag_when_built(built) if inspect.isawaitable(built) else tag(built)

    return build


async def _exit_environment(
    provided: Any, env_context: contextvars.Context, case: Case, error: BaseException | None
) -> None:
    """Exit an environment context manager in the context it was entered in; an error exiting it only
    warns, and never hides `error`."""
    exc_info = (type(error), error, error.__traceback__) if error is not None else (None, None, None)
    try:
        if isinstance(provided, AbstractAsyncContextManager):
            await provided.__aexit__(*exc_info)
        elif isinstance(provided, AbstractContextManager):
            await _in_thread(env_context.run, provided.__exit__, *exc_info)
    except Exception as exit_error:
        logger.warning("case=<%s>, error=<%s> | environment failed to exit", case.name, exit_error)


async def _call(fn: Callable[..., Any], *args: Any) -> Any:
    """Await an async function; run a sync one in a worker thread (which keeps the current span)."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    result = await _in_thread(fn, *args)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _in_thread(fn: Callable[..., Any], *args: Any) -> Any:
    """Run a sync function in `_THREAD_POOL` (else asyncio's shared pool), in a copy of the current
    context, as `asyncio.to_thread` does, so it keeps the current span."""
    pool = _THREAD_POOL.get()
    if pool is None:
        return await asyncio.to_thread(fn, *args)
    call = functools.partial(contextvars.copy_context().run, fn, *args)
    return await asyncio.get_running_loop().run_in_executor(pool, call)


def instrument(task: TaskFn, log: TrialLog, traces: TraceCapture) -> TaskFn:
    """Open one span per attempt, classify and re-raise errors, and attach the `Session` as trajectory.

    Called by `TrialRunner.run`, once per trial (one case, one run) with a fresh `TrialLog`. A setup error is stored as
    `log.setup_error`, not classified: a `_FactoryArgumentError`, or a `RuntimeError` when the attempt
    span isn't recorded (an SDK disabled or sampling out), since every such trial would read as
    `completed` with no usage and no trajectory. A rollout-returned `trajectory` that is not None wins;
    a returned dict without one gets a copy with the `Session` added, never the dict itself changed;
    a non-dict return becomes `{"output": value, "trajectory": session}`. No agent spans -> trajectory None.
    The `Session` keeps every agent in the attempt; when the factory's agents are told apart from the
    others (see `make_task`), each span's metadata says which (`CANDIDATE_SPAN_KEY`).
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
