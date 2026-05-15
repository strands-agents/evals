"""Unit tests for ChaosExperiment."""

import pytest

from strands_evals import Case
from strands_evals.chaos import ChaosCase, ChaosExperiment
from strands_evals.chaos._context import _current_chaos_case
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
def effect_maps():
    return {
        "search_timeout": {"search_tool": [ToolCallFailure(error_type="timeout")]},
        "db_corrupt": {"db_tool": [CorruptValues(corrupt_ratio=0.8)]},
    }


@pytest.fixture
def evaluator():
    return MockChaosEvaluator()


class TestChaosExperiment:
    """Tests for ChaosExperiment initialization and execution."""

    def test_cases_count_with_baseline(self, cases, effect_maps, evaluator):
        chaos_cases = ChaosCase.expand(cases, effect_maps, include_no_effect_baseline=True)
        experiment = ChaosExperiment(cases=chaos_cases, evaluators=[evaluator])
        # 2 cases × (2 effect maps + 1 baseline) = 6
        assert len(experiment.cases) == 6

    def test_cases_count_without_baseline(self, cases, effect_maps, evaluator):
        chaos_cases = ChaosCase.expand(cases, effect_maps)
        experiment = ChaosExperiment(cases=chaos_cases, evaluators=[evaluator])
        # 2 cases × 2 effect maps = 4
        assert len(experiment.cases) == 4

    def test_case_names_include_effect_map_key(self, cases, effect_maps, evaluator):
        chaos_cases = ChaosCase.expand(cases, effect_maps, include_no_effect_baseline=True)
        experiment = ChaosExperiment(cases=chaos_cases, evaluators=[evaluator])
        names = [c.name for c in experiment.cases]
        assert "case_a|baseline" in names
        assert "case_b|baseline" in names
        assert "case_a|search_timeout" in names
        assert "case_b|db_corrupt" in names

    def test_each_case_has_unique_session_id(self, cases, effect_maps, evaluator):
        chaos_cases = ChaosCase.expand(cases, effect_maps)
        experiment = ChaosExperiment(cases=chaos_cases, evaluators=[evaluator])
        session_ids = [c.session_id for c in experiment.cases]
        assert len(session_ids) == len(set(session_ids))

    def test_context_var_set_and_reset(self, cases, effect_maps, evaluator):
        """Verify the ContextVar is set to the correct ChaosCase during task execution and reset after."""
        observed_cases = []

        def capturing_task(case: ChaosCase):
            active_case = _current_chaos_case.get()
            observed_cases.append((case.name, active_case.name if active_case else None))
            return "output"

        chaos_cases = ChaosCase.expand(cases, effect_maps, include_no_effect_baseline=True)
        experiment = ChaosExperiment(cases=chaos_cases, evaluators=[evaluator])
        experiment.run_evaluations(task=capturing_task)

        # Should have 6 observations (2 cases × 3 conditions)
        assert len(observed_cases) == 6

        # Verify the ContextVar matched the case being executed
        for case_name, active_name in observed_cases:
            assert case_name == active_name

        # After all runs, the ContextVar should be back to None
        assert _current_chaos_case.get() is None

    def test_context_var_reset_on_task_exception(self, evaluator):
        """Verify the ContextVar is reset even if the task raises."""
        cases = [Case(name="failing", input="x")]
        effect_maps = {"chaos": {"t": [ToolCallFailure()]}}
        chaos_cases = ChaosCase.expand(cases, effect_maps, include_no_effect_baseline=True)

        call_count = [0]

        def failing_task(case: ChaosCase):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Task failed")
            return "output"

        experiment = ChaosExperiment(cases=chaos_cases, evaluators=[evaluator])

        # The base Experiment should handle the exception internally
        # ContextVar should still be reset
        try:
            experiment.run_evaluations(task=failing_task)
        except Exception:
            pass

        assert _current_chaos_case.get() is None

    def test_returns_evaluation_reports(self, cases, effect_maps, evaluator):
        """Verify run_evaluations returns reports."""

        def task(case: ChaosCase):
            return "output"

        chaos_cases = ChaosCase.expand(cases, effect_maps, include_no_effect_baseline=True)
        experiment = ChaosExperiment(cases=chaos_cases, evaluators=[evaluator])
        reports = experiment.run_evaluations(task=task)

        assert len(reports) >= 1
        report = reports[0]
        # 2 cases × 3 conditions = 6 scores
        assert len(report.scores) == 6
