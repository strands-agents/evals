from collections import Counter
from threading import Lock

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.experimental.calibrator import (
    CalibrationResult,
    CalibrationRunError,
    Calibrator,
)
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


class _OutputRecordingEvaluator(_CountingEvaluator):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.outputs: list[tuple[str, str]] = []

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        assert isinstance(evaluation_case.actual_output, str)
        self.outputs.append((evaluation_case.input, evaluation_case.actual_output))
        return super().evaluate(evaluation_case)


class _NonCopyableOutput:
    def __init__(self) -> None:
        self.lock = Lock()


class _ExtendedCase(Case[str, str]):
    difficulty: str


def _experiment(
    *evaluators: Evaluator[str, str],
    cases: list[Case[str, str]] | None = None,
) -> Experiment[str, str]:
    return Experiment(
        cases=cases
        or [
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=list(evaluators),
    )


def _report(
    case_expected_outputs: dict[str, str] | None = None,
) -> EvaluationReport:
    cases: list[dict[str, str]] = [
        {"name": "good-case", "evaluator": "quality"},
        {"name": "bad-case", "evaluator": "quality"},
    ]
    if case_expected_outputs is not None:
        for case in cases:
            case["expected_output"] = case_expected_outputs[case["name"]]
    return EvaluationReport(
        overall_score=0.5,
        scores=[1.0, 0.0],
        cases=cases,
        test_passes=[True, False],
    )


def _repeated_report(
    report: EvaluationReport | None = None,
) -> list[list[EvaluationReport]]:
    report = report or _report()
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
    assert result.algorithm_identity == "equal_weighting:v1"
    assert result.model_parameters == {
        "weights": {"first": 0.5, "second": 0.5},
    }
    assert result.calibration_design == {
        "agent_runs": 2,
        "judge_replays": 3,
    }
    assert set(result.variance_components) == {"first", "second"}
    assert result.repeatabilities == {"first": 1.0, "second": 1.0}
    assert result.status == "calibrated"


def test_run_evaluations_replays_fixed_output_values() -> None:
    evaluator = _OutputRecordingEvaluator(name="quality")
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return f"{case.input}#{task_calls}"

    result = Calibrator(step_sizes={"quality": 0.5}).run_evaluations(
        _experiment(evaluator),
        task,
        agent_runs=2,
        judge_replays=3,
    )

    outputs_by_run: list[list[dict[str, str]]] = []
    for run_reports in result.reports_by_run:
        replay_outputs = [{case["name"]: case["actual_output"] for case in report.cases} for report in run_reports]
        assert replay_outputs == [replay_outputs[0]] * 3
        outputs_by_run.append(replay_outputs)

    assert outputs_by_run[0][0] != outputs_by_run[1][0]
    output_counts = Counter(output for _, output in evaluator.outputs)
    assert len(output_counts) == 4
    assert set(output_counts.values()) == {3}


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
    result.variance_components["quality"]["judge"] = 0.25

    assert result.score == original_score
    assert calibrator.weights == {"quality": 1.0}
    assert calibrator.variance_components["quality"]["judge"] == pytest.approx(0.0)


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


@pytest.mark.parametrize(
    "case_names",
    [
        (None, "bad-case"),
        ("", "bad-case"),
        ("duplicate", "duplicate"),
    ],
)
def test_run_evaluations_rejects_invalid_case_names_before_task_execution(
    case_names: tuple[str | None, str],
) -> None:
    evaluator = _CountingEvaluator(name="quality")
    experiment = _experiment(
        evaluator,
        cases=[
            Case(name=case_names[0], input="good"),
            Case(name=case_names[1], input="bad"),
        ],
    )
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    with pytest.raises(ValueError, match="unique, non-empty name"):
        Calibrator(step_sizes={"quality": 0.5}).run_evaluations(
            experiment,
            task,
            agent_runs=2,
            judge_replays=2,
        )

    assert task_calls == 0
    assert evaluator.calls == 0


def test_run_evaluations_preserves_reports_after_task_failure() -> None:
    experiment = _experiment(_CountingEvaluator(name="quality"))
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        if task_calls == 4:
            raise RuntimeError("task crashed")
        return case.input

    calibrator = Calibrator(step_sizes={"quality": 0.5})

    with pytest.raises(
        CalibrationRunError,
        match="calibration cannot use failed evaluation rows.*bad-case.*An error occurred: task crashed",
    ) as error:
        calibrator.run_evaluations(
            experiment,
            task,
            agent_runs=2,
            judge_replays=2,
        )

    assert task_calls == 4
    assert [len(reports) for reports in error.value.reports_by_run] == [2, 1]
    assert "task crashed" in error.value.reports_by_run[1][0].reasons[1]
    assert calibrator.is_fitted is False


def test_run_evaluations_rejects_evaluator_failure_rows() -> None:
    calibrator = Calibrator(step_sizes={"quality": 0.5})

    with pytest.raises(
        CalibrationRunError,
        match="calibration cannot use failed evaluation rows.*Evaluator error: judge crashed",
    ) as error:
        calibrator.run_evaluations(
            _experiment(_FailingEvaluator(name="quality")),
            lambda case: case.input,
            agent_runs=2,
            judge_replays=2,
        )

    assert [len(reports) for reports in error.value.reports_by_run] == [1]


def test_run_evaluations_names_non_copyable_task_output() -> None:
    def task(case: Case[str, str]) -> _NonCopyableOutput:
        return _NonCopyableOutput()

    with pytest.raises(CalibrationRunError, match="deep-copyable task output.*good-case") as error:
        Calibrator(step_sizes={"quality": 0.5}).run_evaluations(
            _experiment(_CountingEvaluator(name="quality")),
            task,  # type: ignore[arg-type]
            agent_runs=2,
            judge_replays=2,
        )

    assert [len(reports) for reports in error.value.reports_by_run] == [1]


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


def test_run_evaluations_rejects_changed_cases_before_task_execution() -> None:
    evaluator = _CountingEvaluator(name="quality")
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    calibrator.fit(_repeated_report())
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    with pytest.raises(
        ValueError,
        match=r"evaluation cases must match.*missing=\['bad-case'\].*unexpected=\['new-case'\]",
    ):
        calibrator.run_evaluations(
            _experiment(
                evaluator,
                cases=[
                    Case(name="good-case", input="good"),
                    Case(name="new-case", input="bad"),
                ],
            ),
            task,
            agent_runs=2,
            judge_replays=2,
        )

    assert task_calls == 0
    assert evaluator.calls == 0


def test_run_evaluations_rejects_changed_case_definition_before_task_execution() -> None:
    evaluator = _CountingEvaluator(name="quality")
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    calibrator.fit(
        _repeated_report(
            _report(
                {
                    "good-case": "expected good",
                    "bad-case": "expected bad",
                }
            )
        )
    )
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    with pytest.raises(ValueError, match="case definitions must match"):
        calibrator.run_evaluations(
            _experiment(
                evaluator,
                cases=[
                    Case(name="good-case", input="good", expected_output="changed"),
                    Case(name="bad-case", input="bad", expected_output="expected bad"),
                ],
            ),
            task,
            agent_runs=2,
            judge_replays=2,
        )

    assert task_calls == 0
    assert evaluator.calls == 0


def test_run_evaluations_accepts_unchanged_case_subclasses() -> None:
    evaluator = _CountingEvaluator(name="quality")
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    experiment = _experiment(
        evaluator,
        cases=[
            _ExtendedCase(name="good-case", input="good", difficulty="easy"),
            _ExtendedCase(name="bad-case", input="bad", difficulty="hard"),
        ],
    )

    for _ in range(2):
        result = calibrator.run_evaluations(
            experiment,
            lambda case: case.input,
            agent_runs=2,
            judge_replays=2,
        )
        assert result.status == "calibrated"


def test_run_evaluations_rejects_unsupported_case_metadata_before_task_execution() -> None:
    evaluator = _CountingEvaluator(name="quality")
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    with pytest.raises(ValueError, match="unsupported configuration value type"):
        Calibrator(step_sizes={"quality": 0.5}).run_evaluations(
            _experiment(
                evaluator,
                cases=[
                    Case(name="good-case", input="good", metadata={"value": object()}),
                    Case(name="bad-case", input="bad"),
                ],
            ),
            task,
            agent_runs=2,
            judge_replays=2,
        )

    assert task_calls == 0
    assert evaluator.calls == 0


def test_run_evaluations_rejects_evaluator_mismatch_before_task_execution() -> None:
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    with pytest.raises(ValueError, match="evaluator names do not match"):
        Calibrator(step_sizes={"expected": 0.5}).run_evaluations(
            _experiment(_CountingEvaluator(name="unexpected")),
            task,
            agent_runs=2,
            judge_replays=2,
        )

    assert task_calls == 0


def test_run_evaluations_rejects_invalid_max_workers_before_task_execution() -> None:
    evaluator = _CountingEvaluator(name="quality")
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        Calibrator(step_sizes={"quality": 0.5}).run_evaluations(
            _experiment(evaluator),
            task,
            agent_runs=2,
            judge_replays=2,
            max_workers=0,
        )

    assert task_calls == 0
    assert evaluator.calls == 0


@pytest.mark.parametrize(
    ("agent_runs", "judge_replays"),
    [
        (2.0, 2),
        (True, 2),
        (2, 2.5),
        (2, False),
    ],
)
def test_run_evaluations_rejects_non_integer_repetition_counts(
    agent_runs: object,
    judge_replays: object,
) -> None:
    evaluator = _CountingEvaluator(name="quality")
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    with pytest.raises(ValueError, match="must be >= 2"):
        Calibrator(step_sizes={"quality": 0.5}).run_evaluations(
            _experiment(evaluator),
            task,
            agent_runs=agent_runs,  # type: ignore[arg-type]
            judge_replays=judge_replays,  # type: ignore[arg-type]
        )

    assert task_calls == 0
    assert evaluator.calls == 0


def test_run_evaluations_rejects_async_task_before_execution() -> None:
    evaluator = _CountingEvaluator(name="quality")
    task_calls = 0

    async def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    with pytest.raises(ValueError, match="use run_evaluations_async"):
        Calibrator(step_sizes={"quality": 0.5}).run_evaluations(
            _experiment(evaluator),
            task,  # type: ignore[arg-type]
            agent_runs=2,
            judge_replays=2,
        )

    assert task_calls == 0
    assert evaluator.calls == 0


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
    assert result.status == "calibrated"
