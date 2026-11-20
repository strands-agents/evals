from collections.abc import Mapping

import pytest

from strands_evals.experimental import (
    CalibrationTarget,
    CalibrationTargetDefinition,
    Calibrator,
)
from strands_evals.experimental.calibrator import CalibrationData
from strands_evals.types.evaluation_report import EvaluationReport


def _report(
    scores: dict[str, dict[str, float]],
    *,
    case_inputs: dict[str, str] | None = None,
) -> EvaluationReport:
    report_scores: list[float] = []
    cases: list[dict[str, object]] = []
    passes: list[bool] = []
    for evaluator_name, case_scores in scores.items():
        for case_name, score in case_scores.items():
            report_scores.append(score)
            case: dict[str, object] = {
                "name": case_name,
                "evaluator": evaluator_name,
            }
            if case_inputs is not None:
                case["input"] = case_inputs[case_name]
            cases.append(case)
            passes.append(score >= 0.5)
    return EvaluationReport(
        overall_score=sum(report_scores) / len(report_scores),
        scores=report_scores,
        cases=cases,
        test_passes=passes,
    )


def _calibration_reports() -> list[list[EvaluationReport]]:
    reports: list[list[EvaluationReport]] = []
    for _ in range(2):
        run_reports = []
        for replay in range(2):
            run_reports.append(
                _report(
                    {
                        "stable": {"case-a": 0.0, "case-b": 1.0},
                        "noisy": {
                            "case-a": float(replay),
                            "case-b": float(replay),
                        },
                    }
                )
            )
        reports.append(run_reports)
    return reports


def _repeat_report(
    report: EvaluationReport,
    *,
    agent_runs: int = 2,
    judge_replays: int = 2,
) -> list[list[EvaluationReport]]:
    return [[report.model_copy(deep=True) for _ in range(judge_replays)] for _ in range(agent_runs)]


class _OpaqueConfiguration:
    pass


class _MaximumScoreModel:
    algorithm_id = "maximum_score"
    algorithm_version = 1
    weights = None

    def __init__(self, observation_count: int) -> None:
        self._observation_count = observation_count

    @property
    def parameters(self) -> dict[str, int | str]:
        return {
            "observation_count": self._observation_count,
            "reducer": "maximum",
        }

    def predict(self, evaluator_scores: Mapping[str, float]) -> float:
        return max(evaluator_scores.values())


class _MaximumScoreAlgorithm:
    algorithm_id = "maximum_score"
    algorithm_version = 1

    def fit(self, data: CalibrationData) -> _MaximumScoreModel:
        return _MaximumScoreModel(len(data.observations))


class _RecordingMaximumScoreAlgorithm(_MaximumScoreAlgorithm):
    def __init__(self) -> None:
        self.data: CalibrationData | None = None

    def fit(self, data: CalibrationData) -> _MaximumScoreModel:
        self.data = data
        return super().fit(data)


class _OpaqueParameterModel:
    algorithm_id = "opaque_parameter"
    algorithm_version = 1
    weights = None

    @property
    def parameters(self) -> dict[str, object]:
        return {"opaque": _OpaqueConfiguration()}

    def predict(self, evaluator_scores: Mapping[str, float]) -> float:
        return sum(evaluator_scores.values()) / len(evaluator_scores)


class _OpaqueParameterAlgorithm:
    algorithm_id = "opaque_parameter"
    algorithm_version = 1

    def fit(self, data: CalibrationData) -> _OpaqueParameterModel:
        return _OpaqueParameterModel()


def test_calibrator_starts_with_uncalibrated_equal_weights() -> None:
    calibrator = Calibrator(
        step_sizes={"first": 0.5, "second": 0.25},
        algorithm=_MaximumScoreAlgorithm(),
    )

    assert calibrator.weights == {"first": 0.5, "second": 0.5}
    assert calibrator.algorithm_id == "equal_weighting"
    assert calibrator.status == "uncalibrated"
    assert calibrator.is_fitted is False


def test_score_uses_equal_weights_before_calibration() -> None:
    calibrator = Calibrator(step_sizes={"first": 0.5, "second": 0.25})
    report = _report(
        {
            "first": {"case-a": 0.2},
            "second": {"case-a": 0.8},
        }
    )

    assert calibrator.score([[report]]) == pytest.approx(0.5)
    assert calibrator.status == "uncalibrated"


@pytest.mark.parametrize(
    "step_sizes",
    [
        {},
        {"quality": 0.0},
        {"quality": -0.5},
        {"quality": float("inf")},
        {"quality": float("nan")},
    ],
)
def test_calibrator_rejects_invalid_step_sizes(step_sizes: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="step_sizes"):
        Calibrator(step_sizes=step_sizes)


def test_fit_records_design_and_measurement_diagnostics() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})

    calibrator.fit(_calibration_reports())

    assert calibrator.calibration_design == {
        "agent_runs": 2,
        "judge_replays": 2,
    }
    assert calibrator.variance_components["stable"]["judge"] == pytest.approx(0.0)
    assert calibrator.variance_components["noisy"]["judge"] > 0.0
    assert calibrator.repeatabilities == pytest.approx(
        {
            "stable": 1.0,
            "noisy": 0.0,
        }
    )
    assert calibrator.status == "uncalibrated"
    assert calibrator.is_fitted is True


def test_repeatability_treats_agent_run_variation_as_output_signal() -> None:
    reports = [
        [
            _report({"quality": {"case-a": 0.0, "case-b": 0.0}}),
            _report({"quality": {"case-a": 0.0, "case-b": 0.0}}),
        ],
        [
            _report({"quality": {"case-a": 1.0, "case-b": 1.0}}),
            _report({"quality": {"case-a": 1.0, "case-b": 1.0}}),
        ],
    ]
    calibrator = Calibrator(step_sizes={"quality": 1.0})

    calibrator.fit(reports)

    assert calibrator.variance_components["quality"]["agent"] > 0.0
    assert calibrator.variance_components["quality"]["judge"] == pytest.approx(0.0)
    assert calibrator.repeatabilities["quality"] == pytest.approx(1.0)


def test_calibrator_can_fit_custom_non_weighting_algorithm() -> None:
    calibrator = Calibrator(
        step_sizes={"stable": 0.5, "noisy": 0.5},
        algorithm=_MaximumScoreAlgorithm(),
    )

    calibrator.fit(_calibration_reports())

    assert calibrator.algorithm_id == "maximum_score"
    assert calibrator.algorithm_version == 1
    assert calibrator.weights is None
    assert calibrator.model_parameters == {
        "observation_count": 16,
        "reducer": "maximum",
    }
    assert calibrator.score(_calibration_reports()) == pytest.approx(0.75)


def test_fit_passes_case_run_targets_to_algorithm() -> None:
    algorithm = _RecordingMaximumScoreAlgorithm()
    calibrator = Calibrator(
        step_sizes={"stable": 0.5, "noisy": 0.5},
        algorithm=algorithm,
    )
    definition = CalibrationTargetDefinition(name="tau3_reward", version="1")
    targets = [
        CalibrationTarget(case_name="case-b", agent_run=1, value=0),
        CalibrationTarget(case_name="case-a", agent_run=0, value=1),
        CalibrationTarget(case_name="case-b", agent_run=0, value=0),
        CalibrationTarget(case_name="case-a", agent_run=1, value=1),
    ]

    calibrator.fit(
        _calibration_reports(),
        targets=targets,
        target_definition=definition,
    )

    assert algorithm.data is not None
    assert algorithm.data.has_targets is True
    assert algorithm.data.target_definition == definition
    assert {(example.case_name, example.agent_run): example.target for example in algorithm.data.examples} == {
        ("case-a", 0): 1.0,
        ("case-b", 0): 0.0,
        ("case-a", 1): 1.0,
        ("case-b", 1): 0.0,
    }


@pytest.mark.parametrize(
    ("targets", "target_definition", "message"),
    [
        (
            [CalibrationTarget(case_name="case-a", agent_run=0, value=1.0)],
            None,
            "provided together",
        ),
        (
            None,
            CalibrationTargetDefinition(name="task_success", version="1"),
            "provided together",
        ),
        (
            [],
            CalibrationTargetDefinition(name="task_success", version="1"),
            "one value for every case",
        ),
    ],
)
def test_fit_requires_targets_and_definition_together(
    targets: list[CalibrationTarget] | None,
    target_definition: CalibrationTargetDefinition | None,
    message: str,
) -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})

    with pytest.raises(ValueError, match=message):
        calibrator.fit(
            _calibration_reports(),
            targets=targets,
            target_definition=target_definition,
        )


def test_fit_rejects_missing_target() -> None:
    targets = [
        CalibrationTarget(case_name="case-a", agent_run=0, value=1.0),
        CalibrationTarget(case_name="case-b", agent_run=0, value=0.0),
        CalibrationTarget(case_name="case-a", agent_run=1, value=1.0),
    ]
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})

    with pytest.raises(ValueError, match="missing=.*case-b.*1"):
        calibrator.fit(
            _calibration_reports(),
            targets=targets,
            target_definition=CalibrationTargetDefinition(name="task_success", version="1"),
        )


def test_fit_rejects_unexpected_target() -> None:
    targets = [
        CalibrationTarget(case_name=case_name, agent_run=agent_run, value=1.0)
        for agent_run in range(2)
        for case_name in ("case-a", "case-b")
    ]
    targets.append(CalibrationTarget(case_name="unknown-case", agent_run=0, value=0.0))
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})

    with pytest.raises(ValueError, match="unexpected=.*unknown-case"):
        calibrator.fit(
            _calibration_reports(),
            targets=targets,
            target_definition=CalibrationTargetDefinition(name="task_success", version="1"),
        )


def test_fit_rejects_duplicate_target() -> None:
    duplicate = CalibrationTarget(case_name="case-a", agent_run=0, value=1.0)
    targets = [
        duplicate,
        CalibrationTarget(case_name="case-b", agent_run=0, value=0.0),
        CalibrationTarget(case_name="case-a", agent_run=1, value=1.0),
        CalibrationTarget(case_name="case-b", agent_run=1, value=0.0),
        duplicate,
    ]
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})

    with pytest.raises(ValueError, match="duplicate case-run"):
        calibrator.fit(
            _calibration_reports(),
            targets=targets,
            target_definition=CalibrationTargetDefinition(name="task_success", version="1"),
        )


def test_fit_rejects_non_serializable_model_parameters() -> None:
    calibrator = Calibrator(
        step_sizes={"stable": 0.5, "noisy": 0.5},
        algorithm=_OpaqueParameterAlgorithm(),
    )

    with pytest.raises(ValueError, match="parameters must be deterministic and JSON-compatible"):
        calibrator.fit(_calibration_reports())


def test_fit_rejects_failed_evaluation_rows() -> None:
    report = _report({"quality": {"case-a": 0.0, "case-b": 1.0}})
    report.reasons = ["Evaluator error: judge crashed", ""]
    calibrator = Calibrator(step_sizes={"quality": 0.5})

    with pytest.raises(ValueError, match="calibration cannot use failed evaluation rows.*judge crashed"):
        calibrator.fit(_repeat_report(report))


def test_score_accepts_different_positive_repetition_counts() -> None:
    report = _report(
        {
            "first": {"case-a": 0.0, "case-b": 1.0},
            "second": {"case-a": 1.0, "case-b": 0.0},
        }
    )
    calibrator = Calibrator(step_sizes={"first": 0.5, "second": 0.5})
    calibrator.fit(_repeat_report(report))

    assert calibrator.score(_repeat_report(report, agent_runs=1, judge_replays=3)) == pytest.approx(0.5)


def test_score_accepts_new_cases_and_one_run_one_replay() -> None:
    calibrator = Calibrator(step_sizes={"first": 0.5, "second": 0.5})
    calibrator.fit(
        _repeat_report(
            _report(
                {
                    "first": {"case-a": 0.0, "case-b": 1.0},
                    "second": {"case-a": 0.0, "case-b": 1.0},
                }
            )
        )
    )
    scoring_report = _report(
        {
            "first": {"new-case": 0.4},
            "second": {"new-case": 0.8},
        }
    )

    assert calibrator.score([[scoring_report]]) == pytest.approx(0.6)


def test_fit_rejects_changed_repetition_design() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})
    calibrator.fit(_calibration_reports())

    with pytest.raises(ValueError, match="fit design must match"):
        calibrator.fit([run_reports + [run_reports[0].model_copy(deep=True)] for run_reports in _calibration_reports()])


def test_fit_rejects_changed_case_definition() -> None:
    original = _report(
        {"quality": {"case-a": 0.0, "case-b": 1.0}},
        case_inputs={"case-a": "first", "case-b": "second"},
    )
    changed = _report(
        {"quality": {"case-a": 0.0, "case-b": 1.0}},
        case_inputs={"case-a": "changed", "case-b": "second"},
    )
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    calibrator.fit(_repeat_report(original))

    with pytest.raises(ValueError, match="changed_components=\\['cases'\\]"):
        calibrator.fit(_repeat_report(changed))


def test_fit_rejects_mismatched_evaluator_names() -> None:
    calibrator = Calibrator(step_sizes={"expected": 0.5})
    reports = _repeat_report(_report({"unexpected": {"case-a": 0.0, "case-b": 1.0}}))

    with pytest.raises(ValueError, match="evaluator names do not match"):
        calibrator.fit(reports)


def test_fit_rejects_single_report() -> None:
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    report = _report({"quality": {"case-a": 0.0, "case-b": 1.0}})

    with pytest.raises(ValueError, match="reports_by_run must be a list"):
        calibrator.fit(report)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("algorithm", "message"),
    [
        (object(), "algorithm_id must be a non-empty string"),
        (type("MissingVersion", (), {"algorithm_id": "test", "fit": lambda self, data: None})(), "algorithm_version"),
        (
            type(
                "MissingFit",
                (),
                {"algorithm_id": "test", "algorithm_version": 1},
            )(),
            "callable fit",
        ),
    ],
)
def test_calibrator_rejects_invalid_algorithm(algorithm: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Calibrator(step_sizes={"quality": 0.5}, algorithm=algorithm)  # type: ignore[arg-type]
