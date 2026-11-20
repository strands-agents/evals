import random

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.experimental import Calibrator
from strands_evals.experimental.calibrator import DiagnosticWeightingAlgorithm
from strands_evals.experimental.calibrator._bootstrap import _bootstrap_sample
from strands_evals.experimental.calibrator._calibration import _CalibrationDataset
from strands_evals.types import EvaluationData, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport


def _report(scores: dict[str, dict[str, float]]) -> EvaluationReport:
    report_scores: list[float] = []
    cases: list[dict[str, str]] = []
    passes: list[bool] = []
    for evaluator_name, case_scores in scores.items():
        for case_name, score in case_scores.items():
            report_scores.append(score)
            cases.append({"name": case_name, "evaluator": evaluator_name})
            passes.append(score >= 0.5)
    return EvaluationReport(
        overall_score=sum(report_scores) / len(report_scores),
        scores=report_scores,
        cases=cases,
        test_passes=passes,
    )


def _reports() -> list[list[EvaluationReport]]:
    return [
        [
            _report(
                {
                    "stable": {"case-a": 0.0, "case-b": 1.0},
                    "noisy": {"case-a": 0.0, "case-b": 0.0},
                }
            ),
            _report(
                {
                    "stable": {"case-a": 0.0, "case-b": 1.0},
                    "noisy": {"case-a": 1.0, "case-b": 1.0},
                }
            ),
        ],
        [
            _report(
                {
                    "stable": {"case-a": 0.0, "case-b": 1.0},
                    "noisy": {"case-a": 0.0, "case-b": 0.0},
                }
            ),
            _report(
                {
                    "stable": {"case-a": 0.0, "case-b": 1.0},
                    "noisy": {"case-a": 1.0, "case-b": 1.0},
                }
            ),
        ],
    ]


class _CountingEvaluator(Evaluator[str, str]):
    def __init__(self) -> None:
        super().__init__(name="quality")
        self.calls = 0

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        self.calls += 1
        score = 1.0 if evaluation_case.input == "good" else 0.0
        return [EvaluationOutput(score=score, test_pass=score == 1.0)]


def test_fit_computes_candidate_diagnostic_weights() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})

    calibrator.fit(_reports())

    assert calibrator.candidate_weights["equal"] == {
        "stable": 0.5,
        "noisy": 0.5,
    }
    assert calibrator.candidate_weights["repeatability"] == pytest.approx(
        {
            "stable": 1.0,
            "noisy": 0.0,
        }
    )


def test_diagnostic_weighting_is_explicitly_selectable() -> None:
    calibrator = Calibrator(
        step_sizes={"stable": 0.5, "noisy": 0.5},
        algorithm=DiagnosticWeightingAlgorithm("inverse_sqrt_step"),
    )

    calibrator.fit(_reports())

    assert calibrator.algorithm_id == "diagnostic_inverse_sqrt_step_weighting"
    assert calibrator.weights == pytest.approx(
        {
            "stable": 1.0,
            "noisy": 0.0,
        }
    )
    assert calibrator.status == "uncalibrated"


def test_hierarchical_bootstrap_is_reproducible() -> None:
    first = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})
    second = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})

    first.fit(_reports(), bootstrap_replicates=20, bootstrap_seed=42)
    second.fit(_reports(), bootstrap_replicates=20, bootstrap_seed=42)

    assert first.repeatability_intervals == second.repeatability_intervals
    assert first.candidate_weight_intervals == second.candidate_weight_intervals
    assert first.composite_score_intervals == second.composite_score_intervals
    assert first.bootstrap_replicates == 20
    assert first.bootstrap_seed == 42


def test_fit_without_bootstrap_clears_previous_intervals() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})
    calibrator.fit(_reports(), bootstrap_replicates=20, bootstrap_seed=42)

    calibrator.fit(_reports())

    assert calibrator.bootstrap_replicates == 0
    assert calibrator.bootstrap_seed is None
    assert calibrator.repeatability_intervals == {}
    assert calibrator.candidate_weight_intervals == {}
    assert calibrator.composite_score_intervals == {}


@pytest.mark.parametrize("replicates", [-1, 1, True])
def test_fit_rejects_invalid_bootstrap_replicates(replicates: int) -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})

    with pytest.raises(ValueError, match="bootstrap_replicates must be 0 or >= 2"):
        calibrator.fit(_reports(), bootstrap_replicates=replicates)


def test_fit_rejects_invalid_bootstrap_seed() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})

    with pytest.raises(ValueError, match="bootstrap_seed must be an integer"):
        calibrator.fit(_reports(), bootstrap_replicates=20, bootstrap_seed=True)


def test_insufficient_signal_replicates_remain_visible() -> None:
    report = _report(
        {
            "first": {"case-a": 0.5, "case-b": 0.5},
            "second": {"case-a": 0.5, "case-b": 0.5},
        }
    )
    reports = [[report.model_copy(deep=True) for _ in range(2)] for _ in range(2)]
    calibrator = Calibrator(step_sizes={"first": 0.5, "second": 0.5})

    calibrator.fit(reports, bootstrap_replicates=20, bootstrap_seed=7)

    assert calibrator.candidate_weights["repeatability"] is None
    assert calibrator.candidate_weight_intervals["repeatability"]["first"] == {
        "mean": None,
        "lower_2_5": None,
        "median": None,
        "upper_97_5": None,
        "defined_replicates": 0,
        "undefined_replicates": 20,
    }


def test_score_candidates_reports_diagnostic_sensitivity() -> None:
    calibrator = Calibrator(step_sizes={"stable": 0.5, "noisy": 0.5})
    calibrator.fit(_reports())

    scores = calibrator.score_candidates(_reports())

    assert scores["equal"] == pytest.approx(0.5)
    assert scores["repeatability"] == pytest.approx(0.5)


def test_bootstrap_preserves_complete_evaluator_vectors() -> None:
    reports: list[list[EvaluationReport]] = []
    value = 0.1
    for _ in range(2):
        run_reports = []
        for _ in range(2):
            run_reports.append(
                _report(
                    {
                        "first": {"case-a": value, "case-b": value + 0.1},
                        "second": {
                            "case-a": 1.0 - value,
                            "case-b": 0.9 - value,
                        },
                    }
                )
            )
            value += 0.2
        reports.append(run_reports)
    dataset = _CalibrationDataset.from_reports(
        reports,
        {"first", "second"},
    )

    sampled = _bootstrap_sample(dataset, random.Random(9))

    for run_reports in sampled.parsed:
        for replay_scores in run_reports:
            for case_name in sampled.case_names:
                assert replay_scores["first"][case_name] + replay_scores["second"][case_name] == pytest.approx(1.0)


def test_run_evaluations_validates_bootstrap_before_task_execution() -> None:
    evaluator = _CountingEvaluator()
    experiment = Experiment(
        cases=[
            Case(name="good-case", input="good"),
            Case(name="bad-case", input="bad"),
        ],
        evaluators=[evaluator],
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
    assert evaluator.calls == 0
