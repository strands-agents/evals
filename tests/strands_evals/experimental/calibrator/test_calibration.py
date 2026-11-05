from collections.abc import Mapping

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.experimental import CalibrationResult, Calibrator
from strands_evals.experimental.calibrator import CalibrationData
from strands_evals.experimental.calibrator._calibration import (
    _build_calibration_context,
    _step_size_fingerprint,
)
from strands_evals.types import EvaluationData, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport


def _report(
    scores: dict[str, dict[str, float]],
    *,
    case_inputs: dict[str, str] | None = None,
) -> EvaluationReport:
    report_scores: list[float] = []
    cases: list[dict] = []
    passes: list[bool] = []
    for evaluator_name, case_scores in scores.items():
        for case_name, score in case_scores.items():
            report_scores.append(score)
            case = {"name": case_name, "evaluator": evaluator_name}
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


class _ConfiguredEvaluator(Evaluator[str, str]):
    def __init__(self, accepted_input: str, name: str):
        super().__init__(name=name)
        self.accepted_input = accepted_input
        self.calls = 0

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        self.calls += 1
        score = 1.0 if evaluation_case.input == self.accepted_input else 0.0
        return [EvaluationOutput(score=score, test_pass=score == 1.0)]


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


def test_calibration_context_rejects_unstored_evaluator_configuration() -> None:
    evaluator = _ConfiguredEvaluator(accepted_input="good", name="quality")
    del evaluator.accepted_input

    with pytest.raises(
        ValueError,
        match="cannot fingerprint evaluator 'quality'.*constructor parameter 'accepted_input' is not stored",
    ):
        _build_calibration_context(
            [Case(name="case", input="input")],
            [evaluator],
            {"quality": 0.5},
        )


def test_calibration_context_rejects_opaque_evaluator_configuration() -> None:
    evaluator = _ConfiguredEvaluator(accepted_input="good", name="quality")
    evaluator.accepted_input = _OpaqueConfiguration()

    with pytest.raises(
        ValueError,
        match="unsupported configuration value type",
    ):
        _build_calibration_context(
            [Case(name="case", input="input")],
            [evaluator],
            {"quality": 0.5},
        )


def test_calibrator_starts_with_equal_weights() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})

    assert calibrator.algorithm_id == "equal_weighting"
    assert calibrator.algorithm_version == 1
    assert calibrator.weighting_method == "equal"
    assert calibrator.weights == {"stable": 0.5, "noisy": 0.5}
    assert calibrator.is_fitted is False
    assert calibrator.status == "uncalibrated"
    assert calibrator.score(
        _repeat_report(
            _report(
                {
                    "stable": {"case-a": 0.0, "case-b": 1.0},
                    "noisy": {"case-a": 1.0, "case-b": 1.0},
                }
            )
        )
    ) == pytest.approx(0.75)


def test_fit_records_repetition_design() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})

    assert calibrator.calibration_design is None

    calibrator.fit(_calibration_reports())

    assert calibrator.calibration_design == {
        "agent_runs": 2,
        "judge_replays": 2,
    }


def test_fit_records_calibration_context() -> None:
    reports = _calibration_reports()
    context = {
        "version": "1",
        "dataset": "dataset-hash",
        "evaluators": "evaluator-hash",
        "step_sizes": _step_size_fingerprint({"stable": 0.25, "noisy": 0.25}),
        "overall": "overall-hash",
    }
    for run_reports in reports:
        for report in run_reports:
            report.calibration_context = dict(context)
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})

    calibrator.fit(reports)

    assert calibrator.calibration_context == context


def test_fit_rejects_context_from_different_step_sizes() -> None:
    reports = _calibration_reports()
    context = {
        "version": "1",
        "dataset": "dataset-hash",
        "evaluators": "evaluator-hash",
        "step_sizes": _step_size_fingerprint({"stable": 0.5, "noisy": 0.5}),
        "overall": "overall-hash",
    }
    for run_reports in reports:
        for report in run_reports:
            report.calibration_context = dict(context)
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})

    with pytest.raises(
        ValueError,
        match="does not match the configured evaluator step sizes",
    ):
        calibrator.fit(reports)


def test_fit_rejects_inconsistent_calibration_contexts() -> None:
    reports = _calibration_reports()
    for run_reports in reports:
        for report in run_reports:
            report.calibration_context = {
                "version": "1",
                "dataset": "dataset-a",
                "evaluators": "evaluators",
                "step_sizes": "steps",
                "overall": "overall-a",
            }
    reports[-1][-1].calibration_context = {
        "version": "1",
        "dataset": "dataset-b",
        "evaluators": "evaluators",
        "step_sizes": "steps",
        "overall": "overall-b",
    }
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})

    with pytest.raises(
        ValueError,
        match="incompatible calibration contexts",
    ):
        calibrator.fit(reports)


def test_different_calibrators_can_learn_different_repetition_designs() -> None:
    report = _report(
        {
            "stable": {"case-a": 0.0, "case-b": 1.0},
            "noisy": {"case-a": 1.0, "case-b": 1.0},
        }
    )
    first = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})
    second = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})

    first.fit(_repeat_report(report, agent_runs=2, judge_replays=3))
    second.fit(_repeat_report(report, agent_runs=4, judge_replays=2))

    assert first.calibration_design == {
        "agent_runs": 2,
        "judge_replays": 3,
    }
    assert second.calibration_design == {
        "agent_runs": 4,
        "judge_replays": 2,
    }


def test_score_rejects_single_report() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})

    with pytest.raises(ValueError, match="single EvaluationReport does not contain a repeated scoring design"):
        calibrator.score(  # type: ignore[arg-type]
            _report(
                {
                    "stable": {"case-a": 0.0, "case-b": 1.0},
                    "noisy": {"case-a": 1.0, "case-b": 0.0},
                }
            )
        )


def test_legacy_weighting_algorithm_can_be_selected_explicitly() -> None:
    calibrator = Calibrator(
        step_sizes={"stable": 0.25, "noisy": 0.25},
        weighting_method="inverse_sqrt_step",
    )

    returned = calibrator.fit(_calibration_reports())

    assert returned is calibrator
    assert calibrator.is_fitted is True
    assert calibrator.status == "uncalibrated"
    assert calibrator.algorithm_id == "inverse_sqrt_step_weighting"
    assert calibrator.weights["stable"] == pytest.approx(1.0)
    assert calibrator.weights["noisy"] == pytest.approx(0.0)


def test_fit_retains_variance_and_repeatability_diagnostics() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})

    calibrator.fit(_calibration_reports())

    assert calibrator.variance_components["stable"] == pytest.approx(
        {
            "judge": 0.0,
            "agent": 0.0,
            "case": 0.5,
        }
    )
    assert calibrator.variance_components["noisy"] == pytest.approx(
        {
            "judge": 0.5,
            "agent": 0.0,
            "case": 0.0,
        }
    )
    assert calibrator.repeatabilities == pytest.approx({"stable": 1.0, "noisy": 0.0})


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

    assert calibrator.variance_components["quality"] == pytest.approx(
        {
            "judge": 0.0,
            "agent": 0.5,
            "case": 0.0,
        }
    )
    assert calibrator.repeatabilities == {"quality": 1.0}


def test_fit_computes_all_candidate_weight_vectors() -> None:
    reports = [
        [
            _report(
                {
                    "fine": {"case-a": 0.0, "case-b": 1.0},
                    "coarse": {"case-a": 0.0, "case-b": 1.0},
                }
            )
        ]
        * 2,
    ] * 2
    calibrator = Calibrator(step_sizes={"fine": 0.25, "coarse": 1.0})

    calibrator.fit(reports)

    assert calibrator.candidate_weights["equal"] == pytest.approx({"fine": 0.5, "coarse": 0.5})
    assert calibrator.candidate_weights["repeatability"] == pytest.approx({"fine": 0.5, "coarse": 0.5})
    assert calibrator.candidate_weights["inverse_step"] == pytest.approx({"fine": 0.8, "coarse": 0.2})
    assert calibrator.candidate_weights["inverse_sqrt_step"] == pytest.approx({"fine": 2 / 3, "coarse": 1 / 3})


def test_calibrator_can_select_candidate_weighting_method() -> None:
    reports = [
        [
            _report(
                {
                    "fine": {"case-a": 0.0, "case-b": 1.0},
                    "coarse": {"case-a": 0.0, "case-b": 1.0},
                }
            )
        ]
        * 2,
    ] * 2
    calibrator = Calibrator(
        step_sizes={"fine": 0.25, "coarse": 1.0},
        weighting_method="inverse_step",
    )

    calibrator.fit(reports)

    assert calibrator.weighting_method == "inverse_step"
    assert calibrator.weights == pytest.approx({"fine": 0.8, "coarse": 0.2})


def test_calibrator_can_fit_custom_non_weighting_algorithm() -> None:
    calibrator = Calibrator(
        step_sizes={"stable": 0.25, "noisy": 0.25},
        algorithm=_MaximumScoreAlgorithm(),
    )

    calibrator.fit(_calibration_reports())
    score = calibrator.score(
        _repeat_report(
            _report(
                {
                    "stable": {"case-a": 0.0, "case-b": 1.0},
                    "noisy": {"case-a": 1.0, "case-b": 0.0},
                }
            )
        )
    )

    assert calibrator.algorithm_id == "maximum_score"
    assert calibrator.algorithm_version == 1
    assert calibrator.weighting_method is None
    assert calibrator.weights is None
    assert calibrator.model_parameters == {
        "observation_count": 16,
        "reducer": "maximum",
    }
    assert score == pytest.approx(1.0)
    assert calibrator.is_fitted is True
    assert calibrator.status == "uncalibrated"


def test_fit_rejects_non_serializable_model_parameters() -> None:
    calibrator = Calibrator(
        step_sizes={"stable": 0.25, "noisy": 0.25},
        algorithm=_OpaqueParameterAlgorithm(),
    )

    with pytest.raises(
        ValueError,
        match="fitted model parameters must be deterministic and JSON-compatible",
    ):
        calibrator.fit(_calibration_reports())


def test_score_candidates_compares_composite_sensitivity() -> None:
    reports = [
        [
            _report(
                {
                    "fine": {"case-a": 0.0, "case-b": 1.0},
                    "coarse": {"case-a": 0.0, "case-b": 1.0},
                }
            )
        ]
        * 2,
    ] * 2
    calibrator = Calibrator(step_sizes={"fine": 0.25, "coarse": 1.0})
    calibrator.fit(reports)

    scores = calibrator.score_candidates(
        _repeat_report(
            _report(
                {
                    "fine": {"case-a": 1.0, "case-b": 1.0},
                    "coarse": {"case-a": 0.0, "case-b": 0.0},
                }
            )
        )
    )

    assert scores == pytest.approx(
        {
            "equal": 0.5,
            "repeatability": 0.5,
            "inverse_step": 0.8,
            "inverse_sqrt_step": 2 / 3,
        }
    )


def test_fit_computes_reproducible_hierarchical_bootstrap_intervals() -> None:
    reports = [
        [
            _report(
                {
                    "fine": {"case-a": 0.0, "case-b": 1.0},
                    "coarse": {"case-a": 0.0, "case-b": 1.0},
                }
            )
        ]
        * 2,
    ] * 2
    first = Calibrator(step_sizes={"fine": 0.25, "coarse": 1.0})
    second = Calibrator(step_sizes={"fine": 0.25, "coarse": 1.0})

    first.fit(reports, bootstrap_replicates=200, bootstrap_seed=17)
    second.fit(reports, bootstrap_replicates=200, bootstrap_seed=17)

    assert first.bootstrap_replicates == 200
    assert first.bootstrap_seed == 17
    assert first.repeatability_intervals == second.repeatability_intervals
    assert first.candidate_weight_intervals == second.candidate_weight_intervals
    assert first.composite_score_intervals == second.composite_score_intervals

    fine_repeatability = first.repeatability_intervals["fine"]
    assert fine_repeatability["defined_replicates"] == 200
    assert fine_repeatability["undefined_replicates"] == 0
    assert fine_repeatability["lower_2_5"] == pytest.approx(0.0)
    assert fine_repeatability["upper_97_5"] == pytest.approx(1.0)

    equal_fine = first.candidate_weight_intervals["equal"]["fine"]
    assert equal_fine["mean"] == pytest.approx(0.5)
    assert equal_fine["defined_replicates"] == 200
    assert equal_fine["undefined_replicates"] == 0

    learned_fine = first.candidate_weight_intervals["inverse_sqrt_step"]["fine"]
    assert learned_fine["mean"] == pytest.approx(2 / 3)
    assert 0 < learned_fine["defined_replicates"] < 200
    assert learned_fine["undefined_replicates"] == 200 - learned_fine["defined_replicates"]

    learned_score = first.composite_score_intervals["inverse_sqrt_step"]
    assert learned_score["mean"] == pytest.approx(0.5)
    assert learned_score["defined_replicates"] == learned_fine["defined_replicates"]
    assert learned_score["undefined_replicates"] == learned_fine["undefined_replicates"]

    equal_score = first.composite_score_intervals["equal"]
    assert equal_score["defined_replicates"] == 200
    assert equal_score["lower_2_5"] == pytest.approx(0.0)
    assert equal_score["upper_97_5"] == pytest.approx(1.0)


def test_fit_without_bootstrap_clears_previous_intervals() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})
    reports = _calibration_reports()
    calibrator.fit(reports, bootstrap_replicates=20, bootstrap_seed=4)

    calibrator.fit(reports)

    assert calibrator.bootstrap_replicates == 0
    assert calibrator.bootstrap_seed is None
    assert calibrator.repeatability_intervals == {}
    assert calibrator.candidate_weight_intervals == {}
    assert calibrator.composite_score_intervals == {}


@pytest.mark.parametrize("replicates", [-1, 1])
def test_fit_rejects_invalid_bootstrap_replicates(replicates: int) -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})

    with pytest.raises(ValueError, match="bootstrap_replicates must be 0 or >= 2"):
        calibrator.fit(_calibration_reports(), bootstrap_replicates=replicates)


def test_fit_rejects_invalid_bootstrap_seed() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})

    with pytest.raises(ValueError, match="bootstrap_seed must be an integer"):
        calibrator.fit(
            _calibration_reports(),
            bootstrap_replicates=2,
            bootstrap_seed="invalid",  # type: ignore[arg-type]
        )


def test_score_uses_fitted_weights() -> None:
    calibrator = Calibrator(
        step_sizes={"stable": 0.25, "noisy": 0.25},
        weighting_method="inverse_sqrt_step",
    )
    calibrator.fit(_calibration_reports())

    score = calibrator.score(
        _repeat_report(
            _report(
                {
                    "stable": {"case-a": 0.0, "case-b": 1.0},
                    "noisy": {"case-a": 1.0, "case-b": 1.0},
                }
            )
        )
    )

    assert score == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("agent_runs", "judge_replays"),
    [
        (3, 2),
        (2, 3),
    ],
)
def test_score_accepts_a_different_repetition_design(
    agent_runs: int,
    judge_replays: int,
) -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})
    calibrator.fit(_calibration_reports())
    report = _report(
        {
            "stable": {"case-a": 0.0, "case-b": 1.0},
            "noisy": {"case-a": 1.0, "case-b": 1.0},
        }
    )

    assert calibrator.score(
        _repeat_report(
            report,
            agent_runs=agent_runs,
            judge_replays=judge_replays,
        )
    ) == pytest.approx(0.75)


def test_score_accepts_new_cases_without_calibration_context() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})
    calibrator.fit(_calibration_reports())

    changed_cases = _report(
        {
            "stable": {"case-x": 0.0, "case-y": 1.0},
            "noisy": {"case-x": 1.0, "case-y": 1.0},
        }
    )

    assert calibrator.score(_repeat_report(changed_cases)) == pytest.approx(0.75)


def test_score_accepts_new_case_definition_without_calibration_context() -> None:
    scores = {
        "stable": {"case-a": 0.0, "case-b": 1.0},
        "noisy": {"case-a": 1.0, "case-b": 1.0},
    }
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})
    calibrator.fit(
        _repeat_report(
            _report(
                scores,
                case_inputs={"case-a": "original-a", "case-b": "original-b"},
            )
        )
    )

    changed_definition = _report(
        scores,
        case_inputs={"case-a": "changed-a", "case-b": "original-b"},
    )

    assert calibrator.score(_repeat_report(changed_definition)) == pytest.approx(0.75)


def test_score_accepts_one_case_one_run_and_one_replay() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})
    calibrator.fit(_calibration_reports())

    report = _report(
        {
            "stable": {"new-case": 0.0},
            "noisy": {"new-case": 1.0},
        }
    )

    assert calibrator.score([[report]]) == pytest.approx(0.5)


def test_score_accepts_new_dataset_with_compatible_evaluator_context() -> None:
    reports = _calibration_reports()
    fitted_context = {
        "version": "1",
        "dataset": "calibration-dataset",
        "evaluators": "same-evaluators",
        "step_sizes": _step_size_fingerprint({"stable": 0.25, "noisy": 0.25}),
        "overall": "calibration-overall",
    }
    for run_reports in reports:
        for report in run_reports:
            report.calibration_context = dict(fitted_context)
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})
    calibrator.fit(reports)

    scoring_report = _report(
        {
            "stable": {"new-case": 0.0},
            "noisy": {"new-case": 1.0},
        }
    )
    scoring_report.calibration_context = {
        **fitted_context,
        "dataset": "new-dataset",
        "overall": "new-overall",
    }

    assert calibrator.score([[scoring_report]]) == pytest.approx(0.5)


def test_score_rejects_changed_evaluator_context() -> None:
    reports = _calibration_reports()
    fitted_context = {
        "version": "1",
        "dataset": "calibration-dataset",
        "evaluators": "original-evaluators",
        "step_sizes": _step_size_fingerprint({"stable": 0.25, "noisy": 0.25}),
        "overall": "calibration-overall",
    }
    for run_reports in reports:
        for report in run_reports:
            report.calibration_context = dict(fitted_context)
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})
    calibrator.fit(reports)

    scoring_report = _report(
        {
            "stable": {"new-case": 0.0},
            "noisy": {"new-case": 1.0},
        }
    )
    scoring_report.calibration_context = {
        **fitted_context,
        "dataset": "new-dataset",
        "evaluators": "changed-evaluators",
        "overall": "changed-overall",
    }

    with pytest.raises(
        ValueError,
        match="scoring calibration context is incompatible.*evaluators",
    ):
        calibrator.score([[scoring_report]])


@pytest.mark.parametrize(
    ("agent_runs", "judge_replays"),
    [
        (3, 2),
        (2, 3),
    ],
)
def test_fit_rejects_mismatched_repetition_design(
    agent_runs: int,
    judge_replays: int,
) -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.25, "noisy": 0.25})
    calibrator.fit(_calibration_reports())
    original_weights = calibrator.weights
    report = _report(
        {
            "stable": {"case-a": 0.0, "case-b": 1.0},
            "noisy": {"case-a": 1.0, "case-b": 1.0},
        }
    )

    with pytest.raises(
        ValueError,
        match="fit design must match the fitted calibration design",
    ):
        calibrator.fit(
            _repeat_report(
                report,
                agent_runs=agent_runs,
                judge_replays=judge_replays,
            )
        )

    assert calibrator.calibration_design == {
        "agent_runs": 2,
        "judge_replays": 2,
    }
    assert calibrator.weights == original_weights


def test_fit_keeps_equal_weights_when_no_case_signal_exists() -> None:
    reports = [
        [
            _report({"first": {"case-a": 0.5, "case-b": 0.5}, "second": {"case-a": 0.5, "case-b": 0.5}}),
            _report({"first": {"case-a": 0.5, "case-b": 0.5}, "second": {"case-a": 0.5, "case-b": 0.5}}),
        ],
        [
            _report({"first": {"case-a": 0.5, "case-b": 0.5}, "second": {"case-a": 0.5, "case-b": 0.5}}),
            _report({"first": {"case-a": 0.5, "case-b": 0.5}, "second": {"case-a": 0.5, "case-b": 0.5}}),
        ],
    ]
    calibrator = Calibrator(step_sizes={"first": 0.5, "second": 0.5})

    calibrator.fit(reports, bootstrap_replicates=20, bootstrap_seed=3)

    assert calibrator.weights == {"first": 0.5, "second": 0.5}
    assert calibrator.is_fitted is True
    assert calibrator.status == "uncalibrated"
    assert calibrator.candidate_weights == {
        "equal": {"first": 0.5, "second": 0.5},
        "repeatability": None,
        "inverse_step": None,
        "inverse_sqrt_step": None,
    }
    assert calibrator.candidate_weight_intervals["inverse_sqrt_step"]["first"] == {
        "mean": None,
        "lower_2_5": None,
        "median": None,
        "upper_97_5": None,
        "defined_replicates": 0,
        "undefined_replicates": 20,
    }
    assert calibrator.composite_score_intervals["inverse_sqrt_step"] == {
        "mean": None,
        "lower_2_5": None,
        "median": None,
        "upper_97_5": None,
        "defined_replicates": 0,
        "undefined_replicates": 20,
    }


def test_explicit_equal_weighting_is_not_reported_as_fallback() -> None:
    report = _report(
        {
            "first": {"case-a": 0.5, "case-b": 0.5},
            "second": {"case-a": 0.5, "case-b": 0.5},
        }
    )
    calibrator = Calibrator(
        step_sizes={"first": 0.5, "second": 0.5},
        weighting_method="equal",
    )

    calibrator.fit(_repeat_report(report))

    assert calibrator.weights == {"first": 0.5, "second": 0.5}
    assert calibrator.is_fitted is True
    assert calibrator.status == "uncalibrated"


def test_run_evaluations_reuses_each_agent_output_for_judge_replays() -> None:
    first = _CountingEvaluator(name="first")
    second = _CountingEvaluator(name="second")
    experiment = Experiment(
        cases=[
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=[first, second],
    )
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
        bootstrap_replicates=20,
        bootstrap_seed=9,
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
        "weighting_method": "equal",
        "weights": {"first": 0.5, "second": 0.5},
        "used_equal_fallback": False,
    }
    assert result.status == "uncalibrated"
    assert result.calibration_context == calibrator.calibration_context
    assert calibrator.weights == {"first": 0.5, "second": 0.5}
    assert calibrator.is_fitted is True
    assert calibrator.bootstrap_replicates == 20
    assert calibrator.bootstrap_seed == 9


def test_calibration_result_is_a_point_in_time_copy() -> None:
    evaluator = _CountingEvaluator(name="quality")
    experiment = Experiment(
        cases=[
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=[evaluator],
    )
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    result = calibrator.run_evaluations(
        experiment,
        lambda case: case.input,
        agent_runs=2,
        judge_replays=2,
    )
    original_score = result.score
    original_context = calibrator.calibration_context

    result.reports_by_run[0][0].scores[0] = 0.25
    result.weights["quality"] = 0.25
    assert result.calibration_context is not None
    result.calibration_context["dataset"] = "changed"

    assert result.score == original_score
    assert calibrator.weights == {"quality": 1.0}
    assert calibrator.calibration_context == original_context


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
    experiment = Experiment(
        cases=[
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=[_CountingEvaluator(name="quality")],
    )

    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        if case.input == "bad":
            raise RuntimeError("task crashed")
        return case.input

    calibrator = Calibrator(step_sizes={"quality": 0.5})

    with pytest.raises(
        ValueError,
        match="calibration cannot use failed evaluation rows.*bad-case.*quality.*An error occurred: task crashed",
    ):
        calibrator.run_evaluations(
            experiment,
            task,
            agent_runs=2,
            judge_replays=2,
        )

    assert task_calls == 2


def test_run_evaluations_rejects_evaluator_failure_rows() -> None:
    experiment = Experiment(
        cases=[
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=[_FailingEvaluator(name="quality")],
    )
    calibrator = Calibrator(step_sizes={"quality": 0.5})

    with pytest.raises(
        ValueError,
        match="calibration cannot use failed evaluation rows.*good-case.*quality.*Evaluator error: judge crashed",
    ):
        calibrator.run_evaluations(
            experiment,
            lambda case: case.input,
            agent_runs=2,
            judge_replays=2,
        )


def test_run_evaluations_validates_bootstrap_before_task_execution() -> None:
    experiment = Experiment(
        cases=[
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=[_CountingEvaluator(name="quality")],
    )
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    calibrator = Calibrator(step_sizes={"quality": 0.5})

    with pytest.raises(ValueError, match="bootstrap_replicates must be 0 or >= 2"):
        calibrator.run_evaluations(
            experiment,
            task,
            agent_runs=2,
            judge_replays=2,
            bootstrap_replicates=1,
        )

    assert task_calls == 0


def test_run_evaluations_rejects_mismatched_design_before_task_execution() -> None:
    experiment = Experiment(
        cases=[
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=[_CountingEvaluator(name="quality")],
    )
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    calibrator.fit(
        _repeat_report(
            _report(
                {
                    "quality": {
                        "good-case": 1.0,
                        "bad-case": 0.0,
                    }
                }
            ),
            agent_runs=2,
            judge_replays=2,
        )
    )
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    with pytest.raises(
        ValueError,
        match="evaluation design must match the fitted calibration design",
    ):
        calibrator.run_evaluations(
            experiment,
            task,
            agent_runs=3,
            judge_replays=2,
        )

    assert task_calls == 0


def test_run_evaluations_rejects_changed_dataset_before_task_execution() -> None:
    evaluator = _CountingEvaluator(name="quality")
    original = Experiment(
        cases=[
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=[evaluator],
    )
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    calibrator.run_evaluations(
        original,
        task,
        agent_runs=2,
        judge_replays=2,
    )
    assert calibrator.calibration_context is not None
    assert task_calls == 4

    calibrator.run_evaluations(
        original,
        task,
        agent_runs=2,
        judge_replays=2,
    )
    assert task_calls == 8

    changed = Experiment(
        cases=[
            Case(name="good-case", input="changed-good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=[_CountingEvaluator(name="quality")],
    )

    with pytest.raises(
        ValueError,
        match="evaluation calibration context is incompatible.*dataset",
    ):
        calibrator.run_evaluations(
            changed,
            task,
            agent_runs=2,
            judge_replays=2,
        )

    assert task_calls == 8


def test_run_evaluations_rejects_changed_evaluator_before_task_execution() -> None:
    cases = [
        Case(name="good-case", input="good"),
        Case(name="bad-case", input="bad"),
    ]
    original = Experiment(
        cases=cases,
        evaluators=[_ConfiguredEvaluator(accepted_input="good", name="quality")],
    )
    calibrator = Calibrator(step_sizes={"quality": 0.5})
    task_calls = 0

    def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    calibrator.run_evaluations(
        original,
        task,
        agent_runs=2,
        judge_replays=2,
    )
    assert task_calls == 4

    changed = Experiment(
        cases=cases,
        evaluators=[_ConfiguredEvaluator(accepted_input="bad", name="quality")],
    )

    with pytest.raises(
        ValueError,
        match="evaluation calibration context is incompatible.*evaluators",
    ):
        calibrator.run_evaluations(
            changed,
            task,
            agent_runs=2,
            judge_replays=2,
        )

    assert task_calls == 4


@pytest.mark.asyncio
async def test_run_evaluations_async_accepts_async_task() -> None:
    evaluator = _CountingEvaluator(name="quality")
    experiment = Experiment(
        cases=[
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=[evaluator],
    )
    task_calls = 0

    async def task(case: Case[str, str]) -> str:
        nonlocal task_calls
        task_calls += 1
        return case.input

    calibrator = Calibrator(step_sizes={"quality": 0.5})
    result = await calibrator.run_evaluations_async(
        experiment,
        task,
        agent_runs=2,
        judge_replays=2,
        max_workers=2,
    )

    assert task_calls == 4
    assert evaluator.calls == 8
    assert result.score == pytest.approx(0.5)
    assert len(result.reports_by_run) == 2
    assert result.status == "uncalibrated"
    assert calibrator.weights == {"quality": 1.0}
    assert calibrator.is_fitted is True


def test_fit_rejects_mismatched_evaluator_names() -> None:
    calibrator = Calibrator(step_sizes={"expected": 0.5})
    reports = [
        [_report({"unexpected": {"case-a": 0.0, "case-b": 1.0}})] * 2,
        [_report({"unexpected": {"case-a": 0.0, "case-b": 1.0}})] * 2,
    ]

    with pytest.raises(ValueError, match="evaluator names do not match"):
        calibrator.fit(reports)


def test_calibrator_rejects_unknown_weighting_method() -> None:
    with pytest.raises(ValueError, match="unknown weighting method"):
        Calibrator(
            step_sizes={"quality": 0.5},
            weighting_method="unknown",  # type: ignore[arg-type]
        )


def test_calibrator_rejects_algorithm_and_weighting_method() -> None:
    with pytest.raises(ValueError, match="algorithm and weighting_method are mutually exclusive"):
        Calibrator(
            step_sizes={"quality": 0.5},
            weighting_method="equal",
            algorithm=_MaximumScoreAlgorithm(),
        )
