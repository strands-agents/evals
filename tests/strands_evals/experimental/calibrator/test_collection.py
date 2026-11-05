import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.experimental import CalibrationResult, Calibrator
from strands_evals.types import EvaluationData, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport


class _CountingEvaluator(Evaluator[str, str]):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.calls = 0

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        self.calls += 1
        score = 1.0 if evaluation_case.input == "good" else 0.0
        return [EvaluationOutput(score=score, test_pass=score == 1.0)]


class _FailingEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        raise RuntimeError("judge crashed")


def _experiment(*evaluators: Evaluator[str, str]) -> Experiment[str, str]:
    return Experiment(
        cases=[
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=list(evaluators),
    )


def _report() -> EvaluationReport:
    return EvaluationReport(
        overall_score=0.5,
        scores=[1.0, 0.0],
        cases=[
            {"name": "good-case", "evaluator": "quality"},
            {"name": "bad-case", "evaluator": "quality"},
        ],
        test_passes=[True, False],
    )


def _repeated_report() -> list[list[EvaluationReport]]:
    report = _report()
    return [[report.model_copy(deep=True) for _ in range(2)] for _ in range(2)]


def test_run_evaluations_reuses_each_agent_output_for_judge_replays() -> None:
    first = _CountingEvaluator(name="first")
    second = _CountingEvaluator(name="second")
    experiment = _experiment(first, second)
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    calibrator = Calibrator(step_sizes={"first": 0.5, "second": 0.5})
    result = calibrator.run_evaluations(
        experiment,
        task,
        agent_runs=2,
        judge_replays=3,
    )

    assert task_calls == 4
    assert first.calls == 12
    assert second.calls == 12
    assert isinstance(result, CalibrationResult)
    assert result.score == pytest.approx(0.5)
    assert len(result.reports_by_run) == 2
    assert all(len(run_reports) == 3 for run_reports in result.reports_by_run)
    assert result.weights == {"first": 0.5, "second": 0.5}
    assert result.algorithm_id == "equal_weighting"
    assert result.algorithm_version == 1
    assert result.model_parameters == {
        "weights": {"first": 0.5, "second": 0.5},
    }
    assert result.status == "uncalibrated"


def test_calibration_result_is_a_point_in_time_copy() -> None:
    evaluator = _CountingEvaluator(name="quality")
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    result = calibrator.run_evaluations(
        _experiment(evaluator),
        lambda case: case.input,
        agent_runs=2,
        judge_replays=2,
    )
    original_score = result.score

    result.reports_by_run[0][0].scores[0] = 0.25
    assert result.weights is not None
    result.weights["quality"] = 0.25

    assert result.score == original_score
    assert calibrator.weights == {"quality": 1.0}


def test_run_evaluations_rejects_one_case_before_task_execution() -> None:
    evaluator = _CountingEvaluator(name="quality")
    experiment = Experiment(
        cases=[Case(name="only-case", input="good")],
        evaluators=[evaluator],
    )
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    calibrator = Calibrator(step_sizes={"quality": 0.5})

    with pytest.raises(ValueError, match="calibration requires at least two cases"):
        calibrator.run_evaluations(
            experiment,
            task,
            agent_runs=2,
            judge_replays=2,
        )

    assert task_calls == 0
    assert evaluator.calls == 0


def test_run_evaluations_rejects_task_failure_rows() -> None:
    experiment = _experiment(_CountingEvaluator(name="quality"))

    def task(case: Case[str, str]) -> str:
        if case.input == "bad":
            raise RuntimeError("task crashed")
        return case.input

    calibrator = Calibrator(step_sizes={"quality": 0.5})

    with pytest.raises(
        ValueError,
        match="calibration cannot use failed evaluation rows.*bad-case.*An error occurred: task crashed",
    ):
        calibrator.run_evaluations(
            experiment,
            task,
            agent_runs=2,
            judge_replays=2,
        )


def test_run_evaluations_rejects_evaluator_failure_rows() -> None:
    calibrator = Calibrator(step_sizes={"quality": 0.5})

    with pytest.raises(
        ValueError,
        match="calibration cannot use failed evaluation rows.*Evaluator error: judge crashed",
    ):
        calibrator.run_evaluations(
            _experiment(_FailingEvaluator(name="quality")),
            lambda case: case.input,
            agent_runs=2,
            judge_replays=2,
        )


def test_run_evaluations_rejects_changed_design_before_task_execution() -> None:
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    calibrator.fit(_repeated_report())
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    with pytest.raises(ValueError, match="evaluation design must match"):
        calibrator.run_evaluations(
            _experiment(_CountingEvaluator(name="quality")),
            task,
            agent_runs=3,
            judge_replays=2,
        )

    assert task_calls == 0


@pytest.mark.asyncio
async def test_run_evaluations_async_accepts_async_task() -> None:
    evaluator = _CountingEvaluator(name="quality")
    task_calls = 0

    async def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    calibrator = Calibrator(step_sizes={"quality": 0.5})
    result = await calibrator.run_evaluations_async(
        _experiment(evaluator),
        task,
        agent_runs=2,
        judge_replays=2,
        max_workers=2,
    )

    assert task_calls == 4
    assert evaluator.calls == 8
    assert result.score == pytest.approx(0.5)
    assert result.status == "uncalibrated"
