"""Chaos Experiment.

Extends the base Experiment to run test cases across multiple chaos scenarios,
providing deterministic evaluation of agent resilience under tool failures.
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

from ..case import Case
from ..evaluators.evaluator import Evaluator
from ..experiment import Experiment
from ..types.evaluation_report import EvaluationReport
from .plugin import ChaosPlugin
from .scenario import ChaosScenario

logger = logging.getLogger(__name__)


class ChaosExperiment(Experiment):
    """Extends Experiment to run cases × chaos scenarios.

    For each scenario, sets it as active on the ChaosPlugin, runs all cases
    through the evaluators, then clears. Reports are tagged by scenario name.

    Optionally includes a baseline run (no chaos) for comparison.

    Example::

        from strands_evals.chaos import (
            ToolChaosEffect,
            ChaosExperiment,
            ChaosPlugin,
            ChaosScenario,
        )

        chaos = ChaosPlugin()
        agent = Agent(model=my_model, tools=[...], plugins=[chaos])

        scenarios = [
            ChaosScenario(name="search_timeout", tool_effects={"search_tool": ToolChaosEffect.TIMEOUT}),
            ChaosScenario(name="db_down", tool_effects={"database_tool": ToolChaosEffect.NETWORK_ERROR}),
        ]

        experiment = ChaosExperiment(
            chaos_plugin=chaos,
            chaos_scenarios=scenarios,
            cases=[Case(input="Find flights to Tokyo", name="flight_search")],
            evaluators=[my_evaluator],
            include_baseline=True,
        )

        reports = experiment.run_evaluations(task=lambda case: agent(case.input))
    """

    def __init__(
        self,
        chaos_plugin: ChaosPlugin,
        chaos_scenarios: list[ChaosScenario],
        cases: Optional[list[Case]] = None,
        evaluators: Optional[list[Evaluator]] = None,
        include_baseline: bool = True,
        baseline_assertion: Optional[str] = None,
    ):
        """Initialize a ChaosExperiment.

        Args:
            chaos_plugin: The ChaosPlugin instance attached to the agent.
            chaos_scenarios: List of scenarios to evaluate. Each scenario runs
                all cases independently.
            cases: Test cases to evaluate (same as base Experiment).
            evaluators: Evaluators to assess results (same as base Experiment).
            include_baseline: If True, runs all cases with no chaos first for comparison.
            baseline_assertion: Optional assertion string for baseline evaluation
                (e.g., "agent should respond correctly without any failures").
        """
        super().__init__(cases=cases, evaluators=evaluators)
        self.chaos_plugin = chaos_plugin
        self.chaos_scenarios = chaos_scenarios
        self.include_baseline = include_baseline
        self.baseline_assertion = baseline_assertion

    def run_evaluations(
        self,
        task: Callable[[Case], Any],
        **kwargs,
    ) -> list[EvaluationReport]:
        """Run evaluations across all scenarios (and optionally baseline).

        Executes the task for each (scenario, case) pair:
        1. If include_baseline=True, runs all cases with no chaos active.
        2. For each scenario, activates it on the plugin, runs all cases,
           then clears.

        Results from all runs are collected into a flat list of EvaluationReports,
        with each case tagged with its scenario name in metadata.

        Args:
            task: The task function to evaluate. Takes a Case and returns output.
            **kwargs: Additional kwargs passed to the base run_evaluations.

        Returns:
            List of EvaluationReport objects covering all scenarios.
        """
        all_reports: list[EvaluationReport] = []

        # Baseline run (no chaos)
        if self.include_baseline:
            logger.info("Running baseline evaluation (no chaos)")
            self.chaos_plugin.set_active_scenario(None)
            baseline_cases = self._tag_cases_with_scenario("baseline")
            original_cases = self._cases
            self._cases = baseline_cases
            try:
                reports = super().run_evaluations(task, **kwargs)
                all_reports.extend(reports)
            finally:
                self._cases = original_cases

        # Chaos scenario runs
        for scenario in self.chaos_scenarios:
            logger.info(f"Running chaos scenario: {scenario.name}")
            self.chaos_plugin.set_active_scenario(scenario)
            scenario_cases = self._tag_cases_with_scenario(scenario.name)
            original_cases = self._cases
            self._cases = scenario_cases
            try:
                reports = super().run_evaluations(task, **kwargs)
                all_reports.extend(reports)
            finally:
                self._cases = original_cases

        # Clear active scenario after all runs
        self.chaos_plugin.set_active_scenario(None)
        logger.info(
            f"Chaos experiment complete: {len(all_reports)} reports "
            f"({1 if self.include_baseline else 0} baseline + "
            f"{len(self.chaos_scenarios)} scenarios)"
        )

        return all_reports

    def _tag_cases_with_scenario(self, scenario_name: str) -> list[Case]:
        """Create copies of cases with scenario name injected into metadata.

        Args:
            scenario_name: The scenario name to tag.

        Returns:
            Deep copies of all cases with metadata["chaos_scenario"] set.
        """
        tagged_cases = []
        for case in self._cases:
            tagged = case.model_copy(deep=True)
            if tagged.metadata is None:
                tagged.metadata = {}
            tagged.metadata["chaos_scenario"] = scenario_name
            # Update case name to include scenario for report clarity
            if tagged.name:
                tagged.name = f"{tagged.name} [{scenario_name}]"
            else:
                tagged.name = f"[{scenario_name}]"
            tagged_cases.append(tagged)
        return tagged_cases
