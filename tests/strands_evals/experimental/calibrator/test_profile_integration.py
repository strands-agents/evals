from collections.abc import Mapping

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.experimental import (
    CalibrationApplicationResult,
    CalibrationProfile,
    Calibrator,
)
from strands_evals.experimental.calibrator import (
    CalibrationData,
    DiagnosticWeightingAlgorithm,
    EqualWeightingAlgorithm,
)
from strands_evals.experimental.calibrator._calibration import _build_calibration_context
from strands_evals.types import EvaluationData, EvaluationOutput
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


def _reports() -> list[list[EvaluationReport]]:
    report = _report(
        {
            "first": {"case-a": 0.0, "case-b": 1.0},
            "second": {"case-a": 0.5, "case-b": 0.5},
        },
        case_inputs={"case-a": "a", "case-b": "b"},
    )
    return [[report.model_copy(deep=True) for _ in range(2)] for _ in range(2)]


class _CountingEvaluator(Evaluator[str, str]):
    def __init__(self, name: str):
        super().__init__(name=name)

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        score = 1.0 if evaluation_case.input == "good" else 0.0
        return [EvaluationOutput(score=score, test_pass=score == 1.0)]


class _ConfiguredEvaluator(Evaluator[str, str]):
    def __init__(self, name: str, configuration_tag: str):
        super().__init__(name=name)
        self.configuration_tag = configuration_tag

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        return [EvaluationOutput(score=0.5, test_pass=True)]


class _MaximumScoreModel:
    algorithm_id = "maximum_score"
    algorithm_version = 1
    weights = None

    @property
    def parameters(self) -> dict[str, str]:
        return {"reducer": "maximum"}

    def predict(self, evaluator_scores: Mapping[str, float]) -> float:
        return max(evaluator_scores.values())


class _MaximumScoreAlgorithm:
    algorithm_id = "maximum_score"
    algorithm_version = 1

    def fit(self, data: CalibrationData) -> _MaximumScoreModel:
        return _MaximumScoreModel()

    def restore(self, parameters: Mapping[str, object]) -> _MaximumScoreModel:
        if parameters != {"reducer": "maximum"}:
            raise ValueError("unexpected saved parameters")
        return _MaximumScoreModel()


class _NewerMaximumScoreAlgorithm(_MaximumScoreAlgorithm):
    algorithm_version = 2


def test_context_free_profile_restores_equal_model_and_diagnostics() -> None:
    calibrator = Calibrator(step_sizes={"first": 0.5, "second": 0.5})
    calibrator.fit(_reports(), bootstrap_replicates=10, bootstrap_seed=42)
    profile = calibrator.profile

    assert profile is not None
    assert profile.calibration_context is None
    assert profile.fit_structure_fingerprint is not None

    restored = Calibrator.from_profile(profile)
    scoring_report = _report(
        {
            "first": {"new-case": 0.2},
            "second": {"new-case": 0.8},
        }
    )

    assert restored.score([[scoring_report]]) == pytest.approx(0.5)
    assert restored.profile == profile
    assert restored.variance_components == calibrator.variance_components
    assert restored.repeatability_intervals == calibrator.repeatability_intervals


def test_restored_context_free_profile_rejects_refit_on_changed_cases() -> None:
    calibrator = Calibrator(step_sizes={"first": 0.5, "second": 0.5})
    calibrator.fit(_reports())
    assert calibrator.profile is not None
    restored = Calibrator.from_profile(calibrator.profile)
    changed = _report(
        {
            "first": {"case-a": 0.0, "case-b": 1.0},
            "second": {"case-a": 0.5, "case-b": 0.5},
        },
        case_inputs={"case-a": "changed", "case-b": "b"},
    )

    with pytest.raises(ValueError, match="report structure is incompatible"):
        restored.fit([[changed.model_copy(deep=True) for _ in range(2)] for _ in range(2)])


def test_collection_creates_contextual_profile_that_restores() -> None:
    experiment = Experiment(
        cases=[
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=[
            _CountingEvaluator(name="first"),
            _CountingEvaluator(name="second"),
        ],
    )
    calibrator = Calibrator(step_sizes={"first": 0.5, "second": 0.5})

    result = calibrator.run_evaluations(
        experiment,
        lambda case: case.input,
        agent_runs=2,
        judge_replays=2,
    )
    profile = calibrator.profile

    assert profile is not None
    assert profile.calibration_context == result.calibration_context
    assert all(
        report.calibration_context == profile.calibration_context
        for run_reports in result.reports_by_run
        for report in run_reports
    )
    assert Calibrator.from_profile(profile).score(result.reports_by_run) == pytest.approx(result.score)


def test_contextual_profile_rejects_changed_evaluator_configuration() -> None:
    step_sizes = {"first": 0.5, "second": 0.5}
    cases = [
        Case(name="case-a", input="a"),
        Case(name="case-b", input="b"),
    ]
    original_context = _build_calibration_context(
        cases,
        [
            _ConfiguredEvaluator("first", "original"),
            _ConfiguredEvaluator("second", "original"),
        ],
        step_sizes,
    )
    reports = _reports()
    for run_reports in reports:
        for report in run_reports:
            report.calibration_context = original_context
    calibrator = Calibrator(step_sizes=step_sizes)
    calibrator.fit(reports)

    changed_report = _report(
        {
            "first": {"new-case": 0.2},
            "second": {"new-case": 0.8},
        }
    )
    changed_report.calibration_context = _build_calibration_context(
        [Case(name="new-case", input="new")],
        [
            _ConfiguredEvaluator("first", "changed"),
            _ConfiguredEvaluator("second", "original"),
        ],
        step_sizes,
    )

    with pytest.raises(ValueError, match="changed_components=\\['evaluators'\\]"):
        calibrator.apply_profile([[changed_report]])


def test_profile_application_preserves_historical_reports_and_provenance(tmp_path) -> None:
    calibrator = Calibrator(step_sizes={"first": 0.5, "second": 0.5})
    calibrator.fit(_reports())
    profile = calibrator.profile
    assert profile is not None
    profile_path = tmp_path / "equal-profile.json"
    profile.to_file(str(profile_path))
    restored = Calibrator.from_profile(CalibrationProfile.from_file(str(profile_path)))
    current_report = _report(
        {
            "first": {"new-case": 0.25},
            "second": {"new-case": 1.0},
        }
    )
    historical_report = current_report.model_copy(deep=True)
    historical_report.overall_score = 0.125
    for case in historical_report.cases:
        case["metadata"] = {"source": "historical"}
    original_historical = historical_report.model_copy(deep=True)

    current = restored.apply_profile([[current_report]])
    historical = restored.apply_profile([[historical_report]])

    assert isinstance(historical, CalibrationApplicationResult)
    assert historical.raw_evaluator_scores == pytest.approx(
        {
            "first": 0.25,
            "second": 1.0,
        }
    )
    assert historical.raw_equal_score == pytest.approx(0.625)
    assert historical.selected_score == pytest.approx(current.selected_score)
    assert historical.selected_score == pytest.approx(0.625)
    assert historical.calibrated_probability is None
    assert historical.selected_method == "equal_weighting"
    assert historical.algorithm_version == 1
    assert historical.weights == {
        "first": 0.5,
        "second": 0.5,
    }
    assert historical.profile_id == profile.profile_id
    assert historical.profile_schema_version == profile.schema_version
    assert historical.status == "uncalibrated"
    assert historical.reports_by_run[0][0] == original_historical
    assert historical_report == original_historical

    historical.reports_by_run[0][0].scores[0] = 0.0
    historical.reports_by_run[0][0].overall_score = 1.0

    assert historical_report == original_historical
    assert restored.profile == profile


def test_profile_application_requires_compatible_profile_version() -> None:
    calibrator = Calibrator(
        step_sizes={"first": 0.5, "second": 0.5},
        algorithm=_MaximumScoreAlgorithm(),
    )
    calibrator.fit(_reports())
    profile = calibrator.profile
    assert profile is not None

    with pytest.raises(ValueError, match="cannot restore stale calibration profile"):
        Calibrator.from_profile(profile, algorithm=_NewerMaximumScoreAlgorithm())

    with pytest.raises(ValueError, match="cannot restore incompatible calibration profile"):
        Calibrator.from_profile(profile, algorithm=EqualWeightingAlgorithm())


def test_profile_application_requires_a_fitted_or_restored_profile() -> None:
    calibrator = Calibrator(step_sizes={"first": 0.5, "second": 0.5})

    with pytest.raises(ValueError, match="fitted or restored calibration profile is required"):
        calibrator.apply_profile(_reports())


def test_diagnostic_profile_restores_without_a_custom_algorithm() -> None:
    calibrator = Calibrator(
        step_sizes={"first": 0.5, "second": 0.5},
        algorithm=DiagnosticWeightingAlgorithm("repeatability"),
    )
    calibrator.fit(_reports())
    assert calibrator.profile is not None

    restored = Calibrator.from_profile(calibrator.profile)

    assert restored.algorithm_id == "diagnostic_repeatability_weighting"
    assert restored.weights == calibrator.weights


def test_custom_profile_restores_through_supplied_algorithm() -> None:
    calibrator = Calibrator(
        step_sizes={"first": 0.5, "second": 0.5},
        algorithm=_MaximumScoreAlgorithm(),
    )
    calibrator.fit(_reports())
    profile = calibrator.profile
    assert profile is not None

    with pytest.raises(ValueError, match="custom calibration profiles require"):
        Calibrator.from_profile(profile)

    restored = Calibrator.from_profile(profile, algorithm=_MaximumScoreAlgorithm())
    application = restored.apply_profile(_reports())

    assert restored.weights is None
    assert restored.score(_reports()) == pytest.approx(0.75)
    assert application.raw_equal_score == pytest.approx(0.5)
    assert application.selected_score == pytest.approx(0.75)
    assert application.weights is None
    assert application.calibrated_probability is None
