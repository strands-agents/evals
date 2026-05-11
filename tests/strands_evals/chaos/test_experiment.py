"""Unit tests for ChaosExperiment."""

import pytest

from strands_evals import Case
from strands_evals.chaos import ChaosExperiment, ChaosScenario
from strands_evals.chaos._context import _current_scenario
from strands_evals.chaos.effects import CorruptValues, ToolCallFailure
from strands_evals.evaluators.evaluator import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput


class MockChaosEvaluator(Evaluator):
    """Simple evaluator that always passes."""

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        return [EvaluationOutput(score=1.0, test_pass=True, reason="Mock pass")]


@pytest.fixture
def cases():
    return [
        Case(name="case_a", input="hello"),
        Case(name="case_b", input="world"),
    ]


@pytest.fixture
def scenarios():
    return [
        ChaosScenario(
            name="search_timeout",
            effects={"search_tool": [ToolCallFailure(error_type="timeout")]},
        ),
        ChaosScenario(
            name="db_corrupt",
            effects={"db_tool": [CorruptValues(corrupt_ratio=0.8)]},
        ),
    ]


@pytest.fixture
def evaluator():
    return MockChaosEvaluator()


class TestChaosExperiment:
    """Tests for ChaosExperiment initialization and execution."""

    def test_expanded_cases_count_with_baseline(self, cases, scenarios, evaluator):
        experiment = ChaosExperiment(cases=cases, scenarios=scenarios, evaluators=[evaluator], include_baseline=True)
        # 2 cases × (2 scenarios + 1 baseline) = 6
        assert len(experiment._expanded_cases) == 6

    def test_expanded_cases_count_without_baseline(self, cases, scenarios, evaluator):
        experiment = ChaosExperiment(cases=cases, scenarios=scenarios, evaluators=[evaluator], include_baseline=False)
        # 2 cases × 2 scenarios = 4
        assert len(experiment._expanded_cases) == 4

    def test_expanded_case_names_include_scenario(self, cases, scenarios, evaluator):
        experiment = ChaosExperiment(cases=cases, scenarios=scenarios, evaluators=[evaluator], include_baseline=True)
        names = [c.name for c in experiment._expanded_cases]
        assert "case_a|baseline" in names
        assert "case_a|search_timeout" in names
        assert "case_a|db_corrupt" in names
        assert "case_b|baseline" in names
        assert "case_b|search_timeout" in names
        assert "case_b|db_corrupt" in names

    def test_each_expanded_case_has_unique_session_id(self, cases, scenarios, evaluator):
        experiment = ChaosExperiment(cases=cases, scenarios=scenarios, evaluators=[evaluator])
        session_ids = [c.session_id for c in experiment._expanded_cases]
        assert len(session_ids) == len(set(session_ids))

    def test_get_scenario_for_session(self, cases, scenarios, evaluator):
        experiment = ChaosExperiment(cases=cases, scenarios=scenarios, evaluators=[evaluator], include_baseline=True)
        # Pick an expanded case and verify its scenario maps correctly
        for expanded_case in experiment._expanded_cases:
            scenario = experiment.get_scenario_for_session(expanded_case.session_id)
            assert scenario is not None
            assert scenario.name in expanded_case.name

    def test_get_scenario_for_unknown_session(self, cases, scenarios, evaluator):
        experiment = ChaosExperiment(cases=cases, scenarios=scenarios, evaluators=[evaluator])
        assert experiment.get_scenario_for_session("nonexistent-id") is None

    def test_get_original_case_name(self, cases, scenarios, evaluator):
        experiment = ChaosExperiment(cases=cases, scenarios=scenarios, evaluators=[evaluator], include_baseline=True)
        for expanded_case in experiment._expanded_cases:
            original_name = experiment.get_original_case_name(expanded_case.session_id)
            assert original_name in ("case_a", "case_b")

    def test_get_original_case_name_unknown_session(self, cases, scenarios, evaluator):
        experiment = ChaosExperiment(cases=cases, scenarios=scenarios, evaluators=[evaluator])
        assert experiment.get_original_case_name("nonexistent-id") is None

    def test_context_var_set_and_reset(self, cases, scenarios, evaluator):
        """Verify the ContextVar is set to the correct scenario during task execution and reset after."""
        observed_scenarios = []

        def capturing_task(case: Case):
            scenario = _current_scenario.get()
            observed_scenarios.append((case.name, scenario.name if scenario else None))
            return "output"

        experiment = ChaosExperiment(cases=cases, scenarios=scenarios, evaluators=[evaluator], include_baseline=True)
        experiment.run_evaluations(task=capturing_task)

        # Should have 6 observations (2 cases × 3 scenarios)
        assert len(observed_scenarios) == 6

        # Verify baseline scenarios observed
        baseline_obs = [(name, sn) for name, sn in observed_scenarios if sn == "baseline"]
        assert len(baseline_obs) == 2

        # Verify chaos scenarios observed
        timeout_obs = [(name, sn) for name, sn in observed_scenarios if sn == "search_timeout"]
        assert len(timeout_obs) == 2

        # After all runs, the ContextVar should be back to None
        assert _current_scenario.get() is None

    def test_context_var_reset_on_task_exception(self, evaluator):
        """Verify the ContextVar is reset even if the task raises."""
        cases = [Case(name="failing", input="x")]
        scenarios_list = [ChaosScenario(name="chaos", effects={"t": [ToolCallFailure()]})]

        call_count = [0]

        def failing_task(case: Case):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Task failed")
            return "output"

        experiment = ChaosExperiment(
            cases=cases, scenarios=scenarios_list, evaluators=[evaluator], include_baseline=True
        )

        # The base Experiment should handle the exception internally
        # ContextVar should still be reset
        try:
            experiment.run_evaluations(task=failing_task)
        except Exception:
            pass

        assert _current_scenario.get() is None

    def test_returns_evaluation_reports(self, cases, scenarios, evaluator):
        """Verify run_evaluations returns reports."""

        def task(case: Case):
            return "output"

        experiment = ChaosExperiment(cases=cases, scenarios=scenarios, evaluators=[evaluator], include_baseline=True)
        reports = experiment.run_evaluations(task=task)

        assert len(reports) >= 1
        report = reports[0]
        # 2 cases × 3 scenarios = 6 scores
        assert len(report.scores) == 6
