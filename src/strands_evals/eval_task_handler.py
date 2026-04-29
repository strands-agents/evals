"""Decorator and handlers for wrapping task functions with evaluation behavior."""

import functools
import inspect
from collections.abc import Callable
from typing import Any

from strands import Agent

from .case import Case
from .mappers.strands_in_memory_session_mapper import StrandsInMemorySessionMapper
from .telemetry import StrandsEvalsTelemetry


class EvalTaskHandler:
    """Base handler that normalizes task function return values.

    Subclass to add behavior before/after task execution (e.g., telemetry collection).
    """

    def before(self, case: Case) -> None:
        """Called before the task function runs. Override to add setup logic."""
        pass

    def after(self, case: Case, result: Any) -> dict[str, Any]:
        """Called after the task function runs. Normalizes the result to a dict.

        Args:
            case: The test case that was executed.
            result: The raw return value from the task function.

        Returns:
            A dict compatible with Experiment (must have at least "output" key).
        """
        if isinstance(result, dict):
            return result
        return {"output": str(result)}


class TracedHandler(EvalTaskHandler):
    """Handler that collects OpenTelemetry spans and maps them to a Session.

    Use with @eval_task when your evaluators need trajectory data.

    This handler shares a single span exporter across calls. Use only with
    sequential execution (run_evaluations) or run_evaluations_async with
    max_workers=1. For concurrent execution, each worker needs its own
    TracedHandler instance.

    Args:
        mapper: Session mapper to use. Defaults to StrandsInMemorySessionMapper.

    Example:
        @eval_task(TracedHandler())
        def my_task():
            return Agent(model="...", tools=[calculator])
    """

    def __init__(self, mapper=None):
        self._telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
        self._mapper = mapper or StrandsInMemorySessionMapper()

    def before(self, case: Case) -> None:
        self._telemetry.in_memory_exporter.clear()

    def after(self, case: Case, result: Any) -> dict[str, Any]:
        processed = super().after(case, result)

        spans = list(self._telemetry.in_memory_exporter.get_finished_spans())
        session = self._mapper.map_to_session(spans, case.session_id)
        processed.setdefault("trajectory", session)

        return processed


def _accepts_case(fn: Callable) -> bool:
    """Check if a function accepts a positional argument."""
    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(params) >= 1


def eval_task(handler: EvalTaskHandler | None = None) -> Callable:
    """Decorator that wraps a task function with evaluation behavior.

    The decorated function can:
    - Take no arguments (simple) or a Case argument (for per-case customization)
    - Return an Agent (auto-invoked with case.input), a string, or a dict

    Args:
        handler: Handler that runs before/after the task function.
            Defaults to EvalTaskHandler (normalizes return values only).

    Example:
        @eval_task()
        def my_task():
            return Agent(model="...", tools=[calculator])

        @eval_task(TracedHandler())
        def my_task(case):
            tools = [calculator] if case.metadata.get("use_calc") else []
            return Agent(model="...", tools=tools)
    """
    if handler is None:
        handler = EvalTaskHandler()

    def decorator(fn: Callable) -> Callable[[Case], dict[str, Any]]:
        takes_case = _accepts_case(fn)

        @functools.wraps(fn)
        def wrapper(case: Case) -> dict[str, Any]:
            handler.before(case)

            result = fn(case) if takes_case else fn()

            if isinstance(result, Agent):
                result = str(result(case.input))

            return handler.after(case, result)

        return wrapper

    return decorator
