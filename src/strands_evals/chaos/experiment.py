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
from .plugin import ChaosPlugin
from .scenario import ChaosScenario

logger = logging.getLogger(__name__)

# The baseline scenario — no chaos effects
_BASELINE_SCENARIO = ChaosScenario(name="baseline")


class ChaosExperiment:
    """Runs cases × scenarios by composing the base Experiment.

    For each scenario, activates it via ContextVar, runs all cases through
    the evaluators, then resets. The user's task body contains zero chaos
    concepts — the plugin reads the active scenario from the ContextVar.

    Optionally includes a baseline run (no chaos) for comparison.

    Example::

        from strands_evals.chaos import (
            ChaosExperiment,
            ChaosPlugin,
            ChaosScenario,
            ChaosScenarioAggregator,
            ToolChaosEffect,
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

        # Aggregate and display separately
        aggregator = ChaosScenarioAggregator(known_tools=["search_tool", "database_tool"])
        aggregations = aggregator.aggregate(reports)
        display_chaos_aggregation(aggregations, reports=reports)
        experiment = ChaosExperiment(
            chaos_plugin=chaos,
            chaos_scenarios=scenarios,
            cases=cases,
            evaluators=[GoalSuccessRateEvaluator(), RecoveryStrategyEvaluator()],
            aggregator=ChaosScenarioAggregator(),
        )

        reports = experiment.run_evaluations(task=my_task)
        aggregation_report = experiment.aggregate_evaluations()
        aggregation_report.run_display()
        aggregation_report.to_file("chaos_report.json")
    """

    def __init__(
        self,
        cases: list[Case],
        scenarios: list[ChaosScenario],
        evaluators: Optional[list[Evaluator]] = None,
        include_baseline: bool = True,
        baseline_assertion: Optional[str] = None,
        aggregator: Optional[ChaosScenarioAggregator] = None,
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

        all_scenarios = []
        if include_baseline:
            all_scenarios.append(_BASELINE_SCENARIO)
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
            baseline_assertion: Optional assertion string for baseline evaluation.
            aggregator: Optional ChaosScenarioAggregator for cross-scenario analysis.
                If provided, aggregate_evaluations() can be called after run_evaluations().
                The aggregator auto-derives known_tools from chaos_scenarios.
        """
        super().__init__(cases=cases, evaluators=evaluators)
        self.chaos_plugin = chaos_plugin
        self.chaos_scenarios = chaos_scenarios
        self.include_baseline = include_baseline
        self.baseline_assertion = baseline_assertion
        self._aggregator = aggregator
        self._last_reports: list[EvaluationReport] = []

        # Auto-populate aggregator's known_tools from scenarios
        if self._aggregator is not None and not self._aggregator.known_tools:
            tools: set[str] = set()
            for scenario in chaos_scenarios:
                tools.update(scenario.tool_effects.keys())
            self._aggregator.known_tools = sorted(tools)

    def run_evaluations(
        self,
        task: Callable[[Case], Any],
        **kwargs,
    ) -> list[EvaluationReport]:
        """Run evaluations across all (case × scenario) combinations.

        Wraps the user's task function to set the ContextVar before each
        case execution, so the ChaosPlugin sees the correct scenario.
        Executes the task for each (scenario, case) pair:
        1. If include_baseline=True, runs all cases with no chaos active.
        2. For each scenario, activates it on the plugin, runs all cases,
           then clears.

        Results are stored internally for use by aggregate_evaluations().

        Args:
            task: The task function to evaluate. Takes a Case and returns output.
                The task body should contain zero chaos concepts — just construct
                the agent with plugins=[chaos] and call it.
            **kwargs: Additional kwargs passed to the base Experiment.run_evaluations.

        Returns:
            List of EvaluationReport objects covering all scenarios.
        """

        def chaos_aware_task(case: Case) -> Any:
            """Wrapper that activates the correct scenario via ContextVar."""
            scenario = self._scenario_by_session.get(case.session_id)
            token = _current_scenario.set(scenario)
            try:
                return task(case)
            finally:
                _current_scenario.reset(token)

        reports = self._experiment.run_evaluations(chaos_aware_task, **kwargs)

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
        """Run evaluations asynchronously across all (case × scenario) combinations.

        Same as run_evaluations but uses the async worker pool for parallelism.
        ContextVar ensures each case sees its own scenario even under concurrency.
        # Store for aggregate_evaluations()
        self._last_reports = all_reports
        return all_reports

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
        # Store the raw reports on the aggregation report for run_display()
        report._reports = self._last_reports
        return report

    def _tag_cases_with_scenario(
        self, scenario_name: str, scenario: Optional[ChaosScenario] = None
    ) -> list[Case]:
        """Create copies of cases with scenario name injected into metadata.

        Args:
            task: The task function (sync or async).
            max_workers: Maximum number of parallel workers.
            **kwargs: Additional kwargs passed to the base Experiment.run_evaluations_async.

        Returns:
            List of EvaluationReport objects covering all scenarios.
        """
        import asyncio

        def chaos_aware_task(case: Case) -> Any:
            """Wrapper that activates the correct scenario via ContextVar."""
            scenario = self._scenario_by_session.get(case.session_id)
            token = _current_scenario.set(scenario)
            try:
                return task(case)
            finally:
                _current_scenario.reset(token)
            # Store structured tool_effects for aggregator consumption
            if scenario is not None and scenario.tool_effects:
                tool_effects_serialized = {}
                for tool_name, effect_spec in scenario.tool_effects.items():
                    if isinstance(effect_spec, str):
                        tool_effects_serialized[tool_name] = effect_spec
                    elif hasattr(effect_spec, "value"):
                        tool_effects_serialized[tool_name] = effect_spec.value
                    elif hasattr(effect_spec, "effect"):
                        tool_effects_serialized[tool_name] = effect_spec.effect.value
                    else:
                        tool_effects_serialized[tool_name] = str(effect_spec)
                tagged.metadata["chaos_tool_effects"] = tool_effects_serialized

        async def chaos_aware_task_async(case: Case) -> Any:
            """Async wrapper that activates the correct scenario via ContextVar."""
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

        return reports
