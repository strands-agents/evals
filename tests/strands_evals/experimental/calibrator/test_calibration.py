from collections.abc import Mapping
from datetime import date

import pytest

from strands_evals.experimental.calibrator import (
    CalibrationAlgorithm,
    CalibrationData,
    CalibrationModel,
    Calibrator,
)
from strands_evals.types.evaluation_report import EvaluationReport


def _report(
    scores: dict[str, dict[str, float]],
    *,
    case_expected_outputs: dict[str, str] | None = None,
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
            if case_expected_outputs is not None:
                case["expected_output"] = case_expected_outputs[case_name]
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


class _MaximumScoreModel(CalibrationModel):
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


class _MaximumScoreAlgorithm(CalibrationAlgorithm):
    @property
    def identity(self) -> str:
        return "maximum_score:v1"

    def fit(self, data: CalibrationData) -> _MaximumScoreModel:
        return _MaximumScoreModel(len(data.observations))


class _OpaqueParameterModel(CalibrationModel):
    @property
    def parameters(self) -> dict[str, object]:
        return {"opaque": _OpaqueConfiguration()}

    def predict(self, evaluator_scores: Mapping[str, float]) -> float:
        return sum(evaluator_scores.values()) / len(evaluator_scores)


class _OpaqueParameterAlgorithm(CalibrationAlgorithm):
    @property
    def identity(self) -> str:
        return "opaque_parameter:v1"

    def fit(self, data: CalibrationData) -> _OpaqueParameterModel:
        return _OpaqueParameterModel()


def test_calibrator_starts_with_uncalibrated_equal_weights() -> None:
    calibrator = Calibrator(
        step_sizes={"first": 0.5, "second": 0.25},
        algorithm=_MaximumScoreAlgorithm(),
    )

    assert calibrator.weights == {"first": 0.5, "second": 0.5}
    assert calibrator.algorithm_identity == "equal_weighting:v1"
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


def test_score_clamps_floating_point_roundoff() -> None:
    evaluator_names = [f"evaluator-{index}" for index in range(9)]
    calibrator = Calibrator(step_sizes={name: 0.5 for name in evaluator_names})
    report = _report({name: {"case-a": 1.0} for name in evaluator_names})

    assert calibrator.score([[report]]) == 1.0


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
    assert calibrator.status == "calibrated"
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


def test_variance_estimators_match_hand_computed_grid() -> None:
    reports = [
        [
            _report({"quality": {"case-a": 0.5, "case-b": 0.25}}),
            _report({"quality": {"case-a": 1.0, "case-b": 0.75}}),
        ],
        [
            _report({"quality": {"case-a": 0.25, "case-b": 0.0}}),
            _report({"quality": {"case-a": 0.75, "case-b": 0.5}}),
        ],
    ]
    calibrator = Calibrator(step_sizes={"quality": 0.25})

    calibrator.fit(reports)

    assert calibrator.variance_components["quality"] == pytest.approx(
        {
            "judge": 0.125,
            "agent": 0.0,
            "case": 0.015625,
        }
    )
    assert calibrator.repeatabilities["quality"] == pytest.approx(0.2)


def test_calibrator_can_fit_custom_non_weighting_algorithm() -> None:
    calibrator = Calibrator(
        step_sizes={"stable": 0.5, "noisy": 0.5},
        algorithm=_MaximumScoreAlgorithm(),
    )

    calibrator.fit(_calibration_reports())

    assert calibrator.algorithm_identity == "maximum_score:v1"
    assert calibrator.weights is None
    assert calibrator.model_parameters == {
        "observation_count": 16,
        "reducer": "maximum",
    }
    assert calibrator.status == "calibrated"
    assert calibrator.score(_calibration_reports()) == pytest.approx(0.75)


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
        case_expected_outputs={"case-a": "first", "case-b": "second"},
    )
    changed = _report(
        {"quality": {"case-a": 0.0, "case-b": 1.0}},
        case_expected_outputs={"case-a": "changed", "case-b": "second"},
    )
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    calibrator.fit(_repeat_report(original))

    with pytest.raises(ValueError, match="changed_components=\\['cases'\\]"):
        calibrator.fit(_repeat_report(changed))


def test_fit_accepts_task_rewritten_inputs() -> None:
    reports = _calibration_reports()
    for run_index, run_reports in enumerate(reports):
        for replay_index, report in enumerate(run_reports):
            for case in report.cases:
                case["input"] = f"{case['name']}-run-{run_index}-replay-{replay_index}"
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})

    calibrator.fit(reports)

    assert calibrator.is_fitted is True


def test_fit_accepts_json_serializable_case_metadata() -> None:
    report = _report({"quality": {"case-a": 0.0, "case-b": 1.0}})
    for case in report.cases:
        case["metadata"] = {"created": date(2026, 11, 10)}
    calibrator = Calibrator(step_sizes={"quality": 0.5})

    calibrator.fit(_repeat_report(report))

    assert calibrator.is_fitted is True


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


def test_calibrator_rejects_invalid_algorithm() -> None:
    with pytest.raises(TypeError, match="algorithm must be a CalibrationAlgorithm"):
        Calibrator(step_sizes={"quality": 0.5}, algorithm=object())  # type: ignore[arg-type]
