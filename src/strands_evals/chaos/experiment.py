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
from .aggregator import ChaosScenarioAggregator
from .aggregator_types import ChaosAggregationReport
from .scenario import ChaosScenario

logger = logging.getLogger(__name__)

# The baseline scenario — no chaos effects
_BASELINE_SCENARIO = ChaosScenario(name="baseline")


class ChaosExperiment:
    """Runs cases × scenarios by composing the base Experiment.

    For each scenario, activates it via ContextVar, runs all cases through
    the evaluators, then resets. The user's task body contains zero chaos
    concepts — the plugin reads the active scenario from the ContextVar.

    Example::

        from strands_evals.chaos import (
            ChaosExperiment,
            ChaosScenario,
            ChaosScenarioAggregator,
            ToolCallFailure,
            TruncateFields,
        )

        scenarios = [
            ChaosScenario(name="search_timeout", effects={"search_tool": [ToolCallFailure(error_type="timeout")]}),
            ChaosScenario(name="search_corrupt", effects={"search_tool": [TruncateFields(max_length=5)]}),
        ]

        experiment = ChaosExperiment(
            cases=test_cases,
            scenarios=scenarios,
            evaluators=[my_evaluator],
            aggregator=ChaosScenarioAggregator(),
        )

        reports = experiment.run_evaluations(task=my_task)
        aggregation_report = experiment.aggregate_evaluations()
        aggregation_report.run_display()
    """

    def __init__(
        self,
        cases: list[Case],
        scenarios: list[ChaosScenario],
        evaluators: Optional[list[Evaluator]] = None,
        include_baseline: bool = True,
        aggregator: Optional[ChaosScenarioAggregator] = None,
    ):
        """Initialize a ChaosExperiment.

        Args:
            cases: Test cases to evaluate.
            scenarios: List of chaos scenarios. Each scenario runs all cases.
                All effects in a scenario fire simultaneously in a single run.
            evaluators: Evaluators to assess results.
            include_baseline: If True, runs all cases with no chaos first for comparison.
            aggregator: Optional ChaosScenarioAggregator for cross-scenario analysis.
                If provided, aggregate_evaluations() can be called after run_evaluations().
        """
        self._original_cases = cases
        self._scenarios = scenarios
        self._evaluators = evaluators
        self._include_baseline = include_baseline
        self._aggregator = aggregator
        self._last_reports: list[EvaluationReport] = []

        # Build the expanded case list and internal maps
        self._expanded_cases: list[Case] = []
        self._scenario_by_session: dict[str, ChaosScenario] = {}
        self._original_case_name_by_session: dict[str, Optional[str]] = {}

        all_scenarios = []
        if include_baseline:
            all_scenarios.append(_BASELINE_SCENARIO)
        all_scenarios.extend(scenarios)

        for case in cases:
            for scenario in all_scenarios:
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

        # Auto-populate aggregator's known_tools from scenarios
        if self._aggregator is not None and not self._aggregator.known_tools:
            tools: set[str] = set()
            for scenario in scenarios:
                tools.update(scenario.effects.keys())
            self._aggregator.known_tools = sorted(tools)

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
        """Look up the scenario assigned to a given session_id."""
        return self._scenario_by_session.get(session_id)

    def get_original_case_name(self, session_id: str) -> Optional[str]:
        """Look up the original case name for a given session_id."""
        return self._original_case_name_by_session.get(session_id)

    def run_evaluations(
        self,
        task: Callable[[Case], Any],
        **kwargs,
    ) -> list[EvaluationReport]:
        """Run evaluations across all (case × scenario) combinations.

        Args:
            task: The task function to evaluate. Takes a Case and returns output.
            **kwargs: Additional kwargs passed to the base Experiment.run_evaluations.

        Returns:
            List of EvaluationReport objects covering all scenarios.
        """

        def chaos_aware_task(case: Case) -> Any:
            scenario = self._scenario_by_session.get(case.session_id)
            token = _current_scenario.set(scenario)
            try:
                return task(case)
            finally:
                _current_scenario.reset(token)

        reports = self._experiment.run_evaluations(chaos_aware_task, **kwargs)
        self._last_reports = reports

        num_scenarios = len(self._scenarios) + (1 if self._include_baseline else 0)
        logger.info(
            f"Chaos experiment complete: {len(reports)} reports "
            f"({len(self._original_cases)} cases × {num_scenarios} scenarios)"
        )

        return reports

    async def run_evaluations_async(
        self,
        task: Callable[[Case], Any],
        max_workers: int = 10,
        **kwargs,
    ) -> list[EvaluationReport]:
        """Run evaluations asynchronously across all (case × scenario) combinations."""
        import asyncio

        def chaos_aware_task(case: Case) -> Any:
            scenario = self._scenario_by_session.get(case.session_id)
            token = _current_scenario.set(scenario)
            try:
                return task(case)
            finally:
                _current_scenario.reset(token)

        async def chaos_aware_task_async(case: Case) -> Any:
            scenario = self._scenario_by_session.get(case.session_id)
            token = _current_scenario.set(scenario)
            try:
                if asyncio.iscoroutinefunction(task):
                    return await task(case)
                else:
                    return task(case)
            finally:
                _current_scenario.reset(token)

        if asyncio.iscoroutinefunction(task):
            reports = await self._experiment.run_evaluations_async(
                chaos_aware_task_async, max_workers=max_workers, **kwargs
            )
        else:
            reports = await self._experiment.run_evaluations_async(
                chaos_aware_task, max_workers=max_workers, **kwargs
            )

        self._last_reports = reports
        return reports

    def aggregate_evaluations(self) -> ChaosAggregationReport:
        """Aggregate the last run's evaluation reports into a ChaosAggregationReport.

        Must be called after run_evaluations(). Uses the aggregator passed to __init__.

        Returns:
            ChaosAggregationReport with .run_display() and .to_file() methods.

        Raises:
            RuntimeError: If no aggregator was configured or run_evaluations() hasn't been called.
        """
        if self._aggregator is None:
            raise RuntimeError(
                "No aggregator configured. Pass aggregator=ChaosScenarioAggregator() "
                "to ChaosExperiment.__init__()."
            )
        if not self._last_reports:
            raise RuntimeError(
                "No evaluation reports available. Call run_evaluations() first."
            )

        report = self._aggregator.aggregate(self._last_reports)
        report._reports = self._last_reports
        return report
