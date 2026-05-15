"""Chaos Experiment.

Composes the base Experiment to run ChaosCase objects through evaluators,
providing deterministic evaluation of agent resilience under tool failures.
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

from ..evaluators.evaluator import Evaluator
from ..experiment import Experiment
from ..types.evaluation_report import EvaluationReport
from ._context import _current_chaos_case
from .case import ChaosCase

logger = logging.getLogger(__name__)


class ChaosExperiment:
    """Runs ChaosCase objects through evaluators with chaos-aware dispatch.

    Sets the active ChaosCase via ContextVar before each task invocation so
    the ChaosPlugin can read the case's effects at hook time. The user's task
    body contains zero chaos concepts — the plugin reads the active case from
    the ContextVar.

    Use ``ChaosCase.expand()`` to generate the Cartesian product of base cases
    × effect sets before passing them to this experiment.

    Example::

        from strands_evals import Case
        from strands_evals.chaos import (
            ChaosCase,
            ChaosExperiment,
            ChaosPlugin,
        )
        from strands_evals.chaos.effects import ToolCallFailure

        chaos = ChaosPlugin()

        cases = [Case(input="Find flights to Tokyo", name="flight_search")]
        effect_sets = [
            {"search_tool": [ToolCallFailure(error_type="timeout")]},
            {"database_tool": [ToolCallFailure(error_type="network_error")]},
        ]
        chaos_cases = ChaosCase.expand(cases, effect_sets)

        def my_task(case):
            agent = Agent(tools=[search_tool, database_tool], plugins=[chaos])
            return {"output": str(agent(case.input))}

        experiment = ChaosExperiment(
            cases=chaos_cases,
            evaluators=[my_evaluator],
        )

        reports = experiment.run_evaluations(task=my_task)
    """

    def __init__(
        self,
        cases: list[ChaosCase],
        evaluators: Optional[list[Evaluator]] = None,
    ):
        """Initialize a ChaosExperiment.

        Args:
            cases: ChaosCase objects to evaluate. Use ``ChaosCase.expand()``
                to generate these from base cases and effect sets.
            evaluators: Evaluators to assess results.
        """
        self._cases = cases
        self._evaluators = evaluators

        # Internal Experiment with the chaos cases
        self._experiment = Experiment(
            cases=list(cases),
            evaluators=evaluators,
        )

    @property
    def cases(self) -> list[ChaosCase]:
        """The ChaosCase objects configured for this experiment."""
        return self._cases

    def _wrap_task(self, task: Callable[[ChaosCase], Any]) -> Callable[[ChaosCase], Any]:
        """Wrap a task function to activate the correct ChaosCase via ContextVar.

        Handles both sync and async tasks — returns a sync wrapper for sync tasks
        and an async wrapper for async tasks, so the base Experiment dispatches
        correctly.

        Args:
            task: The original task function (sync or async).

        Returns:
            A wrapped callable that sets/resets the ContextVar around each invocation.
        """
        import asyncio

        if asyncio.iscoroutinefunction(task):

            async def chaos_aware_task_async(case: ChaosCase) -> Any:
                token = _current_chaos_case.set(case)
                try:
                    return await task(case)
                finally:
                    _current_chaos_case.reset(token)

            return chaos_aware_task_async
        else:

            def chaos_aware_task(case: ChaosCase) -> Any:
                token = _current_chaos_case.set(case)
                try:
                    return task(case)
                finally:
                    _current_chaos_case.reset(token)

            return chaos_aware_task

    def run_evaluations(
        self,
        task: Callable[[ChaosCase], Any],
        **kwargs,
    ) -> list[EvaluationReport]:
        """Run evaluations across all ChaosCase objects.

        Delegates to run_evaluations_async with max_workers=1, mirroring the
        base Experiment pattern.

        Args:
            task: The task function to evaluate. Takes a ChaosCase and returns output.
                The task body should contain zero chaos concepts — just construct
                the agent with plugins=[chaos] and call it.
            **kwargs: Additional kwargs passed to the base Experiment.run_evaluations_async.

        Returns:
            List of EvaluationReport objects.

        Raises:
            ValueError: If an async task is passed (use run_evaluations_async instead).
        """
        import asyncio

        if asyncio.iscoroutinefunction(task):
            raise ValueError(
                "Async task is not supported in run_evaluations. Please use run_evaluations_async instead."
            )

        return asyncio.run(self.run_evaluations_async(task, max_workers=1, **kwargs))

    async def run_evaluations_async(
        self,
        task: Callable[[ChaosCase], Any],
        max_workers: int = 10,
        **kwargs,
    ) -> list[EvaluationReport]:
        """Run evaluations asynchronously across all ChaosCase objects.

        Wraps the user's task to set the ContextVar before each case execution.
        The base Experiment handles sync-to-async dispatch internally.

        Args:
            task: The task function (sync or async).
            max_workers: Maximum number of parallel workers.
            **kwargs: Additional kwargs passed to the base Experiment.run_evaluations_async.

        Returns:
            List of EvaluationReport objects.
        """
        wrapped = self._wrap_task(task)
        reports = await self._experiment.run_evaluations_async(wrapped, max_workers=max_workers, **kwargs)

        logger.info(
            "cases=<%d>, reports=<%d> | chaos experiment complete",
            len(self._cases),
            len(reports),
        )

        return reports
