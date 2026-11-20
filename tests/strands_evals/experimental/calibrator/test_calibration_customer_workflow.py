import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.experimental import Calibrator
from strands_evals.types import EvaluationData, EvaluationOutput


class _StableQualityEvaluator(Evaluator[str, str]):
    """Consistently separate the good and bad customer cases."""

    def __init__(self) -> None:
        super().__init__(name="stable-quality")
        self.calls = 0

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        self.calls += 1
        score = 1.0 if evaluation_case.input.startswith("good") else 0.0
        return [EvaluationOutput(score=score, test_pass=score == 1.0)]


class _ReplayNoisyEvaluator(Evaluator[str, str]):
    """Change across judge replays without distinguishing the cases."""

    def __init__(self) -> None:
        super().__init__(name="replay-noisy")
        self.calls = 0
        self._calls_by_input: dict[str, int] = {}

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        self.calls += 1
        call = self._calls_by_input.get(evaluation_case.input, 0)
        self._calls_by_input[evaluation_case.input] = call + 1
        score = float(call % 2)
        return [EvaluationOutput(score=score, test_pass=score == 1.0)]


def test_customer_can_collect_and_inspect_measurement_diagnostics() -> None:
    """Exercise calibration through the same public API a customer would use."""
    stable_evaluator = _StableQualityEvaluator()
    noisy_evaluator = _ReplayNoisyEvaluator()
    cases = [
        *[Case(name=f"good-case-{index}", input=f"good-{index}") for index in range(5)],
        *[Case(name=f"bad-case-{index}", input=f"bad-{index}") for index in range(5)],
    ]
    experiment = Experiment(
        cases=cases,
        evaluators=[
            stable_evaluator,
            noisy_evaluator,
        ],
    )
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    calibrator = Calibrator(
        step_sizes={
            "stable-quality": 1.0,
            "replay-noisy": 1.0,
        }
    )

    assert calibrator.weights == {
        "stable-quality": 0.5,
        "replay-noisy": 0.5,
    }
    assert calibrator.is_fitted is False
    assert calibrator.status == "uncalibrated"

    result = calibrator.run_evaluations(
        experiment,
        task,
        agent_runs=2,
        judge_replays=3,
    )

    expected_task_calls = len(cases) * 2
    expected_evaluator_calls = len(cases) * 2 * 3
    assert task_calls == expected_task_calls
    assert stable_evaluator.calls == expected_evaluator_calls
    assert noisy_evaluator.calls == expected_evaluator_calls
    assert result.score == pytest.approx(0.5)
    assert result.status == "uncalibrated"
    assert result.algorithm_id == "equal_weighting"
    assert calibrator.calibration_design == {
        "agent_runs": 2,
        "judge_replays": 3,
    }
    assert calibrator.repeatabilities == pytest.approx(
        {
            "stable-quality": 1.0,
            "replay-noisy": 0.0,
        }
    )
    assert calibrator.variance_components["stable-quality"]["judge"] == pytest.approx(0.0)
    assert calibrator.variance_components["replay-noisy"]["judge"] > 0.0
