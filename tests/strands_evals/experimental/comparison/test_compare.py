from statistics import fmean

import pytest

import strands_evals.experimental.comparison as comparison_module
from strands_evals import Case, Experiment
from strands_evals.evaluators import Equals
from strands_evals.experimental import (
    CalibrationProfile,
    CalibrationTarget,
    CalibrationTargetDefinition,
    Calibrator,
    EqualWeightingAlgorithm,
)
from strands_evals.experimental.calibrator import CalibratedEqualAlgorithm
from strands_evals.experimental.comparison import compare_calibrated, compare_paired
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport


def _report(evaluator_deltas: dict[str, list[float]], *, trial_offset: float = 0.0) -> EvaluationReport:
    cases: list[dict] = []
    scores: list[float] = []
    for evaluator, deltas in evaluator_deltas.items():
        for index, delta in enumerate(deltas):
            baseline_score = 0.2 + index * 0.1 + trial_offset
            cases.append(
                {
                    "name": f"case-{index}",
                    "input": f"input-{index}",
                    "evaluator": evaluator,
                }
            )
            scores.append(baseline_score + delta)

    return EvaluationReport(
        overall_score=sum(scores) / len(scores),
        scores=scores,
        cases=cases,
        test_passes=[True] * len(scores),
        detailed_results=[[] for _ in scores],
    )


def _score_report(scores: list[float], *, case_prefix: str = "case") -> EvaluationReport:
    return EvaluationReport(
        overall_score=sum(scores) / len(scores),
        scores=scores,
        cases=[
            {
                "name": f"{case_prefix}-{index}",
                "input": f"input-{index}",
                "evaluator": "quality",
            }
            for index in range(len(scores))
        ],
        test_passes=[score >= 0.5 for score in scores],
        detailed_results=[[] for _ in scores],
    )


def _report_runs(
    scores_by_run: list[list[float]],
    *,
    replay_offsets: tuple[float, ...] = (0.0,),
    case_prefix: str = "case",
) -> list[list[EvaluationReport]]:
    return [
        [_score_report([score + offset for score in scores], case_prefix=case_prefix) for offset in replay_offsets]
        for scores in scores_by_run
    ]


def _target_profile() -> CalibrationProfile:
    calibrator = Calibrator(
        step_sizes={"quality": 0.1},
        algorithm=CalibratedEqualAlgorithm(),
    )
    training_scores = [
        [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        [0.12, 0.22, 0.32, 0.72, 0.82, 0.92],
    ]
    calibrator.fit(
        _report_runs(training_scores, replay_offsets=(-0.01, 0.01)),
        targets=[
            CalibrationTarget(
                case_name=f"case-{case_index}",
                agent_run=agent_run,
                value=float(case_index >= 3),
            )
            for agent_run in range(2)
            for case_index in range(6)
        ],
        target_definition=CalibrationTargetDefinition(name="task_success", version="1"),
    )
    assert calibrator.profile is not None
    return calibrator.profile


def _case_probabilities(
    profile: CalibrationProfile,
    report_runs: list[list[EvaluationReport]],
) -> dict[str, float]:
    application = Calibrator.from_profile(profile).apply_profile(report_runs)
    case_names = {result.case_name for result in application.case_run_results}
    return {
        case_name: fmean(
            result.calibrated_probability
            for result in application.case_run_results
            if result.case_name == case_name and result.calibrated_probability is not None
        )
        for case_name in case_names
    }


def test_compares_single_reports():
    baseline = _report({"correctness": [0.0] * 6})
    candidate = _report({"correctness": [0.1] * 6})

    result = compare_paired(baseline=baseline, candidate=candidate)

    assert result.verdict == "improved"
    assert result.trial_count == 1
    assert result.corrupted_trial_indices == []
    assert len(result.pairs) == 6
    assert all(pair.baseline_trial_count == 1 for pair in result.pairs)
    assert all(pair.candidate_trial_count == 1 for pair in result.pairs)
    assert result.evaluator_summaries[0].verdict == "improved"


def test_compares_calibrated_target_probabilities_with_one_profile():
    profile = _target_profile()
    baseline = _report_runs(
        [
            [0.48, 0.485, 0.49, 0.495, 0.5, 0.505],
            [0.49, 0.495, 0.5, 0.505, 0.51, 0.515],
        ],
        replay_offsets=(-0.02, 0.02),
    )
    candidate = _report_runs(
        [
            [0.58, 0.585, 0.59, 0.595, 0.6, 0.605],
            [0.59, 0.595, 0.6, 0.605, 0.61, 0.615],
        ],
        replay_offsets=(-0.02, 0.02),
    )
    expected_baseline = _case_probabilities(profile, baseline)
    expected_candidate = _case_probabilities(profile, candidate)

    result = compare_calibrated(
        profile=profile,
        baseline=baseline,
        candidate=candidate,
    )

    assert result.profile_id == profile.profile_id
    assert result.target_definition == CalibrationTargetDefinition(name="task_success", version="1")
    assert result.verdict == "improved"
    assert result.trial_count == 2
    assert len(result.pairs) == 6
    assert all(pair.evaluator == "target:task_success@1" for pair in result.pairs)
    assert all(pair.baseline_trial_count == 2 for pair in result.pairs)
    assert all(pair.candidate_trial_count == 2 for pair in result.pairs)
    assert {pair.case_id: pair.baseline_score for pair in result.pairs} == pytest.approx(expected_baseline)
    assert {pair.case_id: pair.candidate_score for pair in result.pairs} == pytest.approx(expected_candidate)
    assert result.evaluator_summaries[0].mean_delta > 0.0


def test_calibrated_comparison_excludes_corrupted_runs_before_scoring():
    profile = _target_profile()
    baseline = _report_runs(
        [
            [0.95] * 6,
            [0.22, 0.32, 0.42, 0.52, 0.62, 0.72],
            [0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
        ],
        replay_offsets=(-0.01, 0.01),
    )
    candidate = _report_runs(
        [
            [0.05] * 6,
            [0.32, 0.42, 0.52, 0.62, 0.72, 0.82],
            [0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
        ],
        replay_offsets=(-0.01, 0.01),
    )
    expected_baseline = _case_probabilities(profile, baseline[1:])
    expected_candidate = _case_probabilities(profile, candidate[1:])

    result = compare_calibrated(
        profile=profile,
        baseline=baseline,
        candidate=candidate,
        corrupted_trial_indices={0},
    )

    assert result.trial_count == 2
    assert result.corrupted_trial_indices == [0]
    assert {pair.case_id: pair.baseline_score for pair in result.pairs} == pytest.approx(expected_baseline)
    assert {pair.case_id: pair.candidate_score for pair in result.pairs} == pytest.approx(expected_candidate)


def test_calibrated_comparison_requires_matching_replay_designs():
    profile = _target_profile()
    one_replay = _report_runs([[0.4] * 6, [0.5] * 6])
    two_replays = _report_runs(
        [[0.4] * 6, [0.5] * 6],
        replay_offsets=(-0.01, 0.01),
    )

    with pytest.raises(ValueError, match="same number of judge replays"):
        compare_calibrated(
            profile=profile,
            baseline=one_replay,
            candidate=two_replays,
        )

    with pytest.raises(ValueError, match="every usable agent run"):
        compare_calibrated(
            profile=profile,
            baseline=[two_replays[0], one_replay[1]],
            candidate=two_replays,
        )


def test_calibrated_comparison_errors_name_side_and_original_run():
    profile = _target_profile()
    baseline = _report_runs([[0.3] * 6, [0.4] * 6, [0.5] * 6], replay_offsets=(-0.01, 0.01))
    candidate = _report_runs([[0.4] * 6, [0.5] * 6, [0.6] * 6], replay_offsets=(-0.01, 0.01))
    candidate[2][1].cases[0]["name"] = "changed-case"

    with pytest.raises(ValueError, match=r"candidate: run 2 replay 1 .*different case set"):
        compare_calibrated(
            profile=profile,
            baseline=baseline,
            candidate=candidate,
            corrupted_trial_indices={0},
        )


def test_calibrated_comparison_forwards_statistical_settings():
    profile = _target_profile()
    baseline = _report_runs([[0.3] * 6, [0.35] * 6], replay_offsets=(-0.01, 0.01))
    candidate = _report_runs([[0.4] * 6, [0.45] * 6], replay_offsets=(-0.01, 0.01))

    result = compare_calibrated(
        profile=profile,
        baseline=baseline,
        candidate=candidate,
        confidence_level=0.9,
        minimum_effect=0.5,
    )

    summary = result.evaluator_summaries[0]
    assert summary.confidence_level == 0.9
    assert summary.minimum_effect == 0.5
    assert summary.verdict == "no_clear_change"


def test_calibrated_comparison_requires_target_profile_and_matching_cases():
    equal_calibrator = Calibrator(
        step_sizes={"quality": 0.1},
        algorithm=EqualWeightingAlgorithm(),
    )
    equal_calibrator.fit(
        _report_runs([[0.2, 0.3], [0.25, 0.35]], replay_offsets=(-0.01, 0.01)),
    )
    assert equal_calibrator.profile is not None

    with pytest.raises(ValueError, match="requires a calibrated-equal profile"):
        compare_calibrated(
            profile=equal_calibrator.profile,
            baseline=_report_runs([[0.2, 0.3]]),
            candidate=_report_runs([[0.3, 0.4]]),
        )

    with pytest.raises(ValueError, match="same calibrated case-runs"):
        compare_calibrated(
            profile=_target_profile(),
            baseline=_report_runs([[0.2] * 6]),
            candidate=_report_runs([[0.3] * 6], case_prefix="changed"),
        )


def test_reports_tradeoff_across_evaluators_and_trials():
    baseline = [
        _report({"correctness": [0.0] * 6, "helpfulness": [0.0] * 6}),
        _report({"correctness": [0.0] * 6, "helpfulness": [0.0] * 6}, trial_offset=0.02),
    ]
    candidate = [
        _report({"correctness": [0.1] * 6, "helpfulness": [-0.1] * 6}),
        _report({"correctness": [0.1] * 6, "helpfulness": [-0.1] * 6}, trial_offset=0.02),
    ]

    result = compare_paired(baseline=baseline, candidate=candidate)

    assert result.verdict == "tradeoff"
    assert result.trial_count == 2
    assert {summary.evaluator: summary.verdict for summary in result.evaluator_summaries} == {
        "correctness": "improved",
        "helpfulness": "regressed",
    }


def test_excludes_corrupted_trials_from_public_result():
    baseline = [
        _report({"correctness": [0.0] * 6}),
        _report({"correctness": [0.0] * 6}, trial_offset=0.02),
    ]
    candidate = [
        _report({"correctness": [0.1] * 6}),
        _report({"correctness": [-0.2] * 6}, trial_offset=0.02),
    ]
    candidate[1].cases[0]["name"] = "corrupted-case"

    result = compare_paired(
        baseline=baseline,
        candidate=candidate,
        corrupted_trial_indices={1},
    )

    assert result.verdict == "improved"
    assert result.trial_count == 1
    assert result.corrupted_trial_indices == [1]
    assert len(result.pairs) == 6


def test_excludes_corrupted_first_trial_before_validation():
    baseline = [
        _report({"correctness": [0.0] * 6}),
        _report({"correctness": [0.0] * 6}, trial_offset=0.02),
    ]
    candidate = [
        _report({"correctness": [-0.2] * 6}),
        _report({"correctness": [0.1] * 6}, trial_offset=0.02),
    ]
    candidate[0].cases[0]["name"] = "corrupted-case"

    result = compare_paired(
        baseline=baseline,
        candidate=candidate,
        corrupted_trial_indices={0},
    )

    assert result.verdict == "improved"
    assert result.trial_count == 1
    assert result.corrupted_trial_indices == [0]


def test_rejects_unknown_or_all_corrupted_trials():
    baseline = _report({"correctness": [0.0] * 6})
    candidate = _report({"correctness": [0.1] * 6})

    with pytest.raises(ValueError, match="unknown trials"):
        compare_paired(baseline=baseline, candidate=candidate, corrupted_trial_indices={1})
    with pytest.raises(ValueError, match="no usable trials"):
        compare_paired(baseline=baseline, candidate=candidate, corrupted_trial_indices={0})


@pytest.mark.parametrize(
    ("deltas", "expected_verdict"),
    [
        ([-0.1] * 6, "regressed"),
        ([0.1, -0.1] * 3, "no_clear_change"),
        ([0.1] * 5, "insufficient_data"),
    ],
)
def test_reports_overall_non_improvement_verdicts(
    deltas: list[float],
    expected_verdict: str,
):
    baseline = _report({"correctness": [0.0] * len(deltas)})
    candidate = _report({"correctness": deltas})

    result = compare_paired(baseline=baseline, candidate=candidate)

    assert result.verdict == expected_verdict


def test_applies_lower_is_better_polarity():
    baseline = _report({"attack_success": [0.0] * 6})
    candidate = _report({"attack_success": [-0.5] * 6})

    result = compare_paired(
        baseline=baseline,
        candidate=candidate,
        lower_is_better={"attack_success"},
    )

    assert result.verdict == "improved"
    assert result.evaluator_summaries[0].higher_is_better is False


def test_applies_lower_is_better_polarity_from_one_shot_iterable():
    baseline = _report({"attack_success": [0.0] * 6})
    candidate = _report({"attack_success": [-0.5] * 6})
    names = (name for name in ["attack_success"])

    result = compare_paired(
        baseline=baseline,
        candidate=candidate,
        lower_is_better=names,  # type: ignore[arg-type]
    )

    assert result.verdict == "improved"
    assert result.evaluator_summaries[0].higher_is_better is False


def test_rejects_bare_or_unknown_lower_is_better_names():
    baseline = _report({"attack_success": [0.0] * 6})
    candidate = _report({"attack_success": [-0.5] * 6})

    with pytest.raises(TypeError, match="not a string"):
        compare_paired(
            baseline=baseline,
            candidate=candidate,
            lower_is_better="attack_success",
        )
    with pytest.raises(ValueError, match="unknown=.*attack_sucess.*known=.*attack_success"):
        compare_paired(
            baseline=baseline,
            candidate=candidate,
            lower_is_better={"attack_sucess"},
        )
    with pytest.raises(TypeError, match="non-empty evaluator names"):
        compare_paired(
            baseline=baseline,
            candidate=candidate,
            lower_is_better={1},  # type: ignore[arg-type]
        )


def test_accepts_direction_for_evaluator_with_only_not_applicable_scores():
    not_applicable = EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)
    baseline = _report({"attack_success": [0.0]})
    candidate = _report({"attack_success": [0.0]})
    baseline.detailed_results = [[not_applicable]]
    candidate.detailed_results = [[not_applicable]]

    result = compare_paired(
        baseline=baseline,
        candidate=candidate,
        lower_is_better={"attack_success"},
    )

    assert result.pairs == []
    assert result.verdict == "insufficient_data"


def test_forwards_confidence_level_and_minimum_effect():
    baseline = _report({"correctness": [0.0] * 6})
    candidate = _report({"correctness": [0.1] * 6})

    result = compare_paired(
        baseline=baseline,
        candidate=candidate,
        confidence_level=0.9,
        minimum_effect=0.11,
    )

    summary = result.evaluator_summaries[0]
    assert summary.confidence_level == 0.9
    assert summary.minimum_effect == 0.11
    assert summary.verdict == "no_clear_change"


def test_insufficient_evaluator_prevents_overall_improved_claim():
    baseline = _report({"correctness": [0.0] * 7, "helpfulness": [0.0] * 5})
    candidate = _report({"correctness": [0.1] * 7, "helpfulness": [0.1] * 5})

    result = compare_paired(baseline=baseline, candidate=candidate)

    assert {summary.evaluator: summary.verdict for summary in result.evaluator_summaries} == {
        "correctness": "improved",
        "helpfulness": "insufficient_data",
    }
    assert result.verdict == "insufficient_data"


def test_regression_takes_precedence_over_insufficient_data():
    baseline = _report({"correctness": [0.0] * 7, "helpfulness": [0.0] * 5})
    candidate = _report({"correctness": [-0.1] * 7, "helpfulness": [0.1] * 5})

    result = compare_paired(baseline=baseline, candidate=candidate)

    assert {summary.evaluator: summary.verdict for summary in result.evaluator_summaries} == {
        "correctness": "regressed",
        "helpfulness": "insufficient_data",
    }
    assert result.verdict == "regressed"


def test_multiple_testing_can_make_regression_insufficient():
    baseline = _report({"correctness": [0.0] * 6, "unchanged": [0.0] * 6})
    candidate = _report({"correctness": [-0.1] * 6, "unchanged": [0.0] * 6})

    result = compare_paired(baseline=baseline, candidate=candidate)

    assert result.evaluator_summaries[0].adjusted_p_value == pytest.approx(0.0625)
    assert result.evaluator_summaries[0].verdict == "insufficient_data"
    assert result.verdict == "insufficient_data"


def test_no_comparable_scores_is_insufficient_data():
    not_applicable = EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)
    baseline = _report({"correctness": [0.0]})
    candidate = _report({"correctness": [0.0]})
    baseline.detailed_results = [[not_applicable]]
    candidate.detailed_results = [[not_applicable]]

    result = compare_paired(baseline=baseline, candidate=candidate)

    assert result.pairs == []
    assert result.evaluator_summaries == []
    assert result.verdict == "insufficient_data"


def test_compares_reports_produced_by_experiment():
    cases = [Case(name=f"case-{index}", input=f"question-{index}", expected_output="answer") for index in range(6)]
    experiment = Experiment(cases=cases, evaluators=[Equals(name="equals")])
    baseline = experiment.run_evaluations(lambda case: case.expected_output)
    candidate = experiment.run_evaluations(lambda _: "wrong")

    result = compare_paired(baseline=baseline, candidate=candidate)

    assert result.verdict == "regressed"
    assert result.evaluator_summaries[0].evaluator == "equals"
    assert result.evaluator_summaries[0].mean_delta == -1.0


def test_rejects_scipy_without_rng_permutation_api(monkeypatch: pytest.MonkeyPatch):
    class LegacyPermutationMethod:
        def __init__(self, *, random_state: int | None = None):
            self.random_state = random_state

    monkeypatch.setattr(comparison_module, "PermutationMethod", LegacyPermutationMethod)

    with pytest.raises(ImportError, match=r"scipy>=1\.15"):
        comparison_module._validate_scipy_api()
