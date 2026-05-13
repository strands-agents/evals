"""Chaos Experiment.

Composes the base Experiment to run test cases across multiple chaos scenarios,
providing deterministic evaluation of agent resilience under tool failures.
"""

import logging
import uuid
from collections.abc import Callable
from typing import Any, Optional

from ..case import Case
from ..evaluators.evaluator import Evaluator
from ..experiment import Experiment
from ..types.evaluation_report import EvaluationReport
from ._context import _current_scenario
from .scenario import ChaosScenario

logger = logging.getLogger(__name__)


class ChaosExperiment:
    """Runs cases × scenarios by composing the base Experiment.

    For each scenario, activates it via ContextVar, runs all cases through
    the evaluators, then resets. The user's task body contains zero chaos
    concepts — the plugin reads the active scenario from the ContextVar.

    Optionally includes a baseline run (no chaos) for comparison.

    Note: The experiment runs ``len(cases) × (len(scenarios) + 1)`` evaluations
    when ``include_baseline=True``, or ``len(cases) × len(scenarios)`` otherwise.
    Plan scenario counts accordingly — each combination triggers a full agent
    invocation.

    Example::

        from strands_evals.chaos import (
            ChaosExperiment,
            ChaosPlugin,
            ChaosScenario,
        )
        from strands_evals.chaos.effects import ToolCallFailure

        chaos = ChaosPlugin()

        scenarios = [
            ChaosScenario(
                name="search_timeout",
                effects={"search_tool": [ToolCallFailure(error_type="timeout")]},
            ),
            ChaosScenario(
                name="db_down",
                effects={"database_tool": [ToolCallFailure(error_type="network_error")]},
            ),
        ]

        def my_task(case):
            agent = Agent(tools=[search_tool, database_tool], plugins=[chaos])
            return {"output": str(agent(case.input))}

        experiment = ChaosExperiment(
            cases=[Case(input="Find flights to Tokyo", name="flight_search")],
            scenarios=scenarios,
            evaluators=[my_evaluator],
            include_baseline=True,
        )

        reports = experiment.run_evaluations(task=my_task)
    """

    def __init__(
        self,
        cases: list[Case],
        scenarios: list[ChaosScenario],
        evaluators: Optional[list[Evaluator]] = None,
        include_baseline: bool = True,
    ):
        """Initialize a ChaosExperiment.

        Args:
            cases: Test cases to evaluate.
            scenarios: List of chaos scenarios. Each scenario runs all cases.
                All effects in a scenario fire simultaneously in a single run.
            evaluators: Evaluators to assess results.
            include_baseline: If True, runs all cases with no chaos first for comparison.
        """
        self._original_cases = cases
        self._scenarios = scenarios
        self._evaluators = evaluators
        self._include_baseline = include_baseline

        # Build the expanded case list and internal maps
        self._expanded_cases: list[Case] = []
        self._scenario_by_session: dict[str, ChaosScenario] = {}
        self._original_case_name_by_session: dict[str, Optional[str]] = {}

        all_scenarios = [ChaosScenario(name="baseline")] if include_baseline else []
        all_scenarios.extend(scenarios)

        for case in cases:
            for scenario in all_scenarios:
                # Create expanded case with fresh session_id
                session_id = str(uuid.uuid4())
                expanded_case = case.model_copy(
                    update={
                        "name": f"{case.name}|{scenario.name}" if case.name else scenario.name,
                        "session_id": session_id,
                    }
                )
                self._expanded_cases.append(expanded_case)
                self._scenario_by_session[session_id] = scenario
                self._original_case_name_by_session[session_id] = case.name

        # Internal Experiment with expanded cases
        self._experiment = Experiment(
            cases=self._expanded_cases,
            evaluators=evaluators,
        )

    @property
    def scenarios(self) -> list[ChaosScenario]:
        """The chaos scenarios configured for this experiment."""
        return self._scenarios

    @property
    def cases(self) -> list[Case]:
        """The original (unexpanded) test cases."""
        return self._original_cases

    def get_scenario_for_session(self, session_id: str) -> Optional[ChaosScenario]:
        """Look up the scenario assigned to a given session_id.

        Useful for downstream aggregation and reporting.

        Args:
            session_id: The session_id of an expanded case.

        Returns:
            The ChaosScenario for that session, or None if not found.
        """
        return self._scenario_by_session.get(session_id)

    def get_original_case_name(self, session_id: str) -> Optional[str]:
        """Look up the original case name for a given session_id.

        Args:
            session_id: The session_id of an expanded case.

        Returns:
            The original case name, or None if not found.
        """
        return self._original_case_name_by_session.get(session_id)

    def _wrap_task(self, task: Callable[[Case], Any]) -> Callable[[Case], Any]:
        """Wrap a task function to activate the correct scenario via ContextVar.

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

            async def chaos_aware_task_async(case: Case) -> Any:
                scenario = self._scenario_by_session.get(case.session_id)
                token = _current_scenario.set(scenario)
                try:
                    return await task(case)
                finally:
                    _current_scenario.reset(token)

            return chaos_aware_task_async
        else:

            def chaos_aware_task(case: Case) -> Any:
                scenario = self._scenario_by_session.get(case.session_id)
                token = _current_scenario.set(scenario)
                try:
                    return task(case)
                finally:
                    _current_scenario.reset(token)

            return chaos_aware_task

    def run_evaluations(
        self,
        task: Callable[[Case], Any],
        **kwargs,
    ) -> list[EvaluationReport]:
        """Run evaluations across all (case × scenario) combinations.

        Delegates to run_evaluations_async with max_workers=1, mirroring the
        base Experiment pattern.

        Args:
            task: The task function to evaluate. Takes a Case and returns output.
                The task body should contain zero chaos concepts — just construct
                the agent with plugins=[chaos] and call it.
            **kwargs: Additional kwargs passed to the base Experiment.run_evaluations_async.

        Returns:
            List of EvaluationReport objects covering all scenarios.

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
        task: Callable[[Case], Any],
        max_workers: int = 10,
        **kwargs,
    ) -> list[EvaluationReport]:
        """Run evaluations asynchronously across all (case × scenario) combinations.

        Wraps the user's task to set the ContextVar before each case execution.
        The base Experiment handles sync-to-async dispatch internally.

        Args:
            task: The task function (sync or async).
            max_workers: Maximum number of parallel workers.
            **kwargs: Additional kwargs passed to the base Experiment.run_evaluations_async.

        Returns:
            List of EvaluationReport objects covering all scenarios.
        """
        wrapped = self._wrap_task(task)
        reports = await self._experiment.run_evaluations_async(wrapped, max_workers=max_workers, **kwargs)

        num_scenarios = len(self._scenarios) + (1 if self._include_baseline else 0)
        logger.info(
            "cases=<%d>, scenarios=<%d>, reports=<%d> | chaos experiment complete",
            len(self._original_cases),
            num_scenarios,
            len(reports),
        )

        return reports
