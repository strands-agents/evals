"""The default rollout, the task wrappers `Experiment` drives, and the per-run trial log."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import logging
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import Executor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import AbstractAsyncContextManager, AbstractContextManager, suppress
from dataclasses import dataclass, field
from typing import Any

from strands import Agent
from strands.hooks import AfterModelCallEvent, AfterToolCallEvent, BeforeModelCallEvent, HookProvider, HookRegistry

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

# The pool sync functions run in, set by `TrialRunner.run` for its call and sized to its `max_workers`:
# asyncio's shared pool holds at most min(32, cpus + 4) threads, which would cap the sync trials running
# at once. None (outside a `TrialRunner` call) -> that shared pool.
_THREAD_POOL: contextvars.ContextVar[Executor | None] = contextvars.ContextVar("trial_thread_pool", default=None)

# What a rollout's own deadline raises: `asyncio.wait_for`, `Future.result(timeout=)`; one class from Python 3.11 on.
_DEADLINE_ERRORS = (TimeoutError, asyncio.TimeoutError, FuturesTimeoutError)


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

    `rollout_fn` receives a factory that builds with `agent_factory._with_resources(resources)` (without
    `environment`, with `agent_factory`) and adds the attempt's hooks to every `strands.Agent` it returns
    (see `_rollout`): `default_rollout` calls it with no arguments; a custom rollout may pass the
    factory function's non-tunable kwargs. A setup error, never a trial status, raises
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
    """Run `rollout_fn` inside the `rollout` span, whose duration is `Usage.seconds`.

    `rollout_fn` gets `agent_factory` wrapped so that every `Agent` it builds gets this attempt's
    `_AttemptHooks`. Two failures inside the agent then end the attempt as infrastructure:

    - A tool raised `InfrastructureError`, which Strands would hand to the model as an error result: every
      agent of the attempt stops before its next model call, and the attempt raises that error, whatever
      the rollout returned or raised after it (a setup error, `_FactoryArgumentError`, still wins). A later
      result of the same tool call that isn't an `InfrastructureError` (a hook retried it) clears it.
    - The rollout's own deadline (`TimeoutError`) passed after an agent's last model call failed on
      infrastructure (`classify`) and before it started another, e.g. while `ModelRetryStrategy` slept
      after throttling: raised as `InfrastructureError` from that model call's error, so the trial records
      the throttling. Any other deadline, e.g. one that passes during a model call, stays the rollout's
      `TimeoutError`.
    """
    hooks = _AttemptHooks()
    with get_tracer().start_as_current_span(f"rollout {case.name}", attributes={SPAN_KIND: ROLLOUT_SPAN}):
        failure: Exception | None = None
        try:
            result = await _call(rollout_fn, _adding_hooks(agent_factory, hooks), case)
        except _FactoryArgumentError:
            raise
        except Exception as error:
            failure = error
        finally:
            hooks.close()
        # Raised outside the `except`, so the tool's error doesn't gain the rollout's error as its context.
        tool_error = hooks.tool_error()
        if tool_error is not None:
            raise tool_error
        if failure is None:
            return result
        failed_call = hooks.failed_call()
        if isinstance(failure, _DEADLINE_ERRORS) and failed_call is not None:
            raise InfrastructureError(
                f"the rollout's deadline passed after a model call failed on infrastructure: {failed_call}"
            ) from failed_call
        raise failure


class _AttemptHooks(HookProvider):
    """The hooks `_rollout` adds to every `Agent` the factory builds during one attempt.

    Every callback is sync and only records or raises. They do nothing once the attempt is over
    (`close`), since a hook can't be removed from an agent the factory might return again; such an agent
    keeps one set of idle callbacks per earlier attempt.
    """

    def __init__(self) -> None:
        # (id(agent), toolUseId) -> the error; a later result for the same call (a hook retried it) clears it
        self._tool_errors: dict[tuple[int, str], InfrastructureError] = {}
        # id(agent) -> its last model call's infrastructure error, until the agent's next model call starts
        self._failed_calls: dict[int, Exception] = {}
        self._closed = False

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register the three callbacks.

        After* callbacks of one order run in reverse registration order, so `_after_model_call` runs
        before the agent's retry strategy (registered when the agent was built) sleeps.
        """
        registry.add_callback(AfterToolCallEvent, self._after_tool_call)
        registry.add_callback(BeforeModelCallEvent, self._before_model_call)
        registry.add_callback(AfterModelCallEvent, self._after_model_call)

    def close(self) -> None:
        """End the attempt: every callback does nothing from now on."""
        self._closed = True

    def tool_error(self) -> InfrastructureError | None:
        """The first `InfrastructureError` a tool raised that no retry cleared, else None."""
        return next(iter(self._tool_errors.values()), None)

    def failed_call(self) -> Exception | None:
        """The infrastructure error of an agent's last model call when it has started none since, else None."""
        return next(iter(self._failed_calls.values()), None)

    def _after_tool_call(self, event: AfterToolCallEvent) -> None:
        if self._closed:
            return
        key = (id(event.agent), str(event.tool_use.get("toolUseId")))
        if isinstance(event.exception, InfrastructureError):
            self._tool_errors[key] = event.exception
        else:
            self._tool_errors.pop(key, None)

    def _before_model_call(self, event: BeforeModelCallEvent) -> None:
        if self._closed:
            return
        tool_error = self.tool_error()
        if tool_error is not None:
            raise tool_error
        self._failed_calls.pop(id(event.agent), None)

    def _after_model_call(self, event: AfterModelCallEvent) -> None:
        if self._closed:
            return
        if event.exception is not None and classify(event.exception) is TrialStatus.INFRASTRUCTURE_INVALID:
            self._failed_calls[id(event.agent)] = event.exception
        else:
            self._failed_calls.pop(id(event.agent), None)


def _adding_hooks(agent_factory: Callable[..., Any], hooks: _AttemptHooks) -> Callable[..., Any]:
    """`agent_factory`, but every `Agent` it returns (or its coroutine returns) gets `hooks`; anything else
    it builds is returned as is."""

    def build(*args: Any, **kwargs: Any) -> Any:
        built = agent_factory(*args, **kwargs)
        if inspect.iscoroutine(built):
            return _add_hooks_when_built(built, hooks)
        _add_hooks(built, hooks)
        return built

    return build


async def _add_hooks_when_built(built: Coroutine[Any, Any, Any], hooks: _AttemptHooks) -> Any:
    agent = await built
    _add_hooks(agent, hooks)
    return agent


def _add_hooks(built: Any, hooks: _AttemptHooks) -> None:
    if isinstance(built, Agent):
        built.hooks.add_hook(hooks)


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
