"""Public orchestration for statistical evaluation-report comparison."""

from collections.abc import Collection, Sequence
from statistics import fmean
from typing import Literal

from pydantic import BaseModel

from ...types.evaluation_report import EvaluationReport
from ..calibrator import (
    CalibratedEqualAlgorithm,
    CalibrationCaseRunResult,
    CalibrationProfile,
    CalibrationTargetDefinition,
    Calibrator,
)
from .pairing import PairedScore, aggregate_trial_pairs, pair_report_trials
from .summary import EvaluatorSummary, summarize_pairs

PairedComparisonVerdict = Literal[
    "improved",
    "regressed",
    "tradeoff",
    "no_clear_change",
    "insufficient_data",
]


class PairedComparisonResult(BaseModel):
    """Statistical comparison of baseline and candidate evaluation reports.

    Attributes:
        pairs: Trial-averaged scores for each comparable case and evaluator.
        evaluator_summaries: Statistical results grouped by evaluator.
        verdict: Overall gate result. Conflicting directional evaluator
            verdicts produce `tradeoff`; otherwise any regression takes
            precedence, followed by insufficient data, improvement, and no
            clear change.
        trial_count: Number of usable trials represented in the result.
        corrupted_trial_indices: Explicitly excluded trial indices.
    """

    pairs: list[PairedScore]
    evaluator_summaries: list[EvaluatorSummary]
    verdict: PairedComparisonVerdict
    trial_count: int
    corrupted_trial_indices: list[int]


class CalibratedComparisonResult(PairedComparisonResult):
    """Paired comparison of probabilities produced by one calibration profile."""

    profile_id: str
    target_definition: CalibrationTargetDefinition


def compare_paired(
    *,
    baseline: EvaluationReport | Sequence[EvaluationReport],
    candidate: EvaluationReport | Sequence[EvaluationReport],
    corrupted_trial_indices: Collection[int] = (),
    confidence_level: float = 0.95,
    minimum_effect: float = 0.0,
    lower_is_better: Collection[str] = (),
) -> PairedComparisonResult:
    """Compare one or more baseline and candidate evaluation reports.

    The case is the unit of statistical evidence. Repeated trials reduce noise
    in each case's mean but do not increase the case count. Report sequences
    are aligned by position, and explicitly corrupted trials are removed from
    both sides before report validation. Pairing is strict about case/evaluator
    row identity. Evaluator applicability may vary independently across
    repeated baseline and candidate runs; each side is averaged over its own
    applicable trials, and a row is rejected only when one side has no usable
    score across all trials.

    Overall verdict precedence is conservative for regression gating:
    `tradeoff` when improvement and regression coexist, then `regressed` when
    any evaluator regresses, then `insufficient_data`, then `improved`, and
    finally `no_clear_change`. Consumers implementing an acceptance gate
    should treat `tradeoff`, `regressed`, and `insufficient_data` as
    non-passing unless their policy explicitly says otherwise.

    Task or evaluator failures that remain in a report are statistical
    evidence, including rows scored as zero. Exclude a trial only when the
    entire trial is known to be corrupted by infrastructure or collection
    failure.

    Args:
        baseline: One baseline report or an ordered sequence of trial reports.
        candidate: One candidate report or an ordered sequence of trial reports.
        corrupted_trial_indices: Trial indices to exclude from both sides.
        confidence_level: Desired confidence interval coverage. The
            corresponding `1 - confidence_level` value is also the significance
            threshold.
        minimum_effect: Smallest meaningful absolute score change. The full
            confidence interval must lie beyond this threshold for a
            directional evaluator verdict.
        lower_is_better: Evaluator names for which a lower score is beneficial.
            All other evaluators are treated as higher-is-better.

    Returns:
        Paired case scores, evaluator statistics, and an overall verdict.

    Raises:
        TypeError: If `lower_is_better` is passed as one bare string.
        ValueError: If the trial inputs cannot be paired or the comparison
            settings are invalid.
    """

    baseline_reports = _as_reports(baseline)
    candidate_reports = _as_reports(candidate)
    corrupted = set(corrupted_trial_indices)
    lower_is_better_names = _as_evaluator_names(lower_is_better)
    trial_pairs = pair_report_trials(
        baseline_reports=baseline_reports,
        candidate_reports=candidate_reports,
        corrupted_trial_indices=corrupted,
    )
    known_evaluators = {pair.evaluator for pair in trial_pairs}
    unknown_direction_names = lower_is_better_names - known_evaluators
    if unknown_direction_names:
        raise ValueError(
            "lower_is_better names unknown evaluators: "
            f"unknown={sorted(unknown_direction_names)}, known={sorted(known_evaluators)}"
        )

    pairs = aggregate_trial_pairs(trial_pairs)
    paired_evaluators = {pair.evaluator for pair in pairs}
    summaries = summarize_pairs(
        pairs,
        confidence_level=confidence_level,
        minimum_effect=minimum_effect,
        lower_is_better=lower_is_better_names & paired_evaluators,
    )
    return PairedComparisonResult(
        pairs=pairs,
        evaluator_summaries=summaries,
        verdict=_comparison_verdict(summaries),
        trial_count=len(baseline_reports) - len(corrupted),
        corrupted_trial_indices=sorted(corrupted),
    )


def compare_calibrated(
    *,
    profile: CalibrationProfile,
    baseline: Sequence[Sequence[EvaluationReport]],
    candidate: Sequence[Sequence[EvaluationReport]],
    corrupted_trial_indices: Collection[int] = (),
    confidence_level: float = 0.95,
    minimum_effect: float = 0.0,
) -> CalibratedComparisonResult:
    """Compare baseline and candidate target probabilities from one profile.

    The supplied profile is restored once and applied unchanged to both sides.
    Judge replays are averaged before prediction, agent runs are averaged
    within each case, and cases remain the independent statistical unit.
    `minimum_effect` is expressed in calibrated-probability points.

    Args:
        profile: Fitted calibrated-equal profile with an external target.
        baseline: Baseline reports grouped by agent run and judge replay.
        candidate: Candidate reports grouped by agent run and judge replay.
        corrupted_trial_indices: Agent-run indices to exclude from both sides.
        confidence_level: Desired confidence interval coverage.
        minimum_effect: Smallest meaningful probability change.

    Returns:
        A paired comparison carrying the profile and target provenance.

    Raises:
        ValueError: If the profile does not produce target probabilities or
            the baseline and candidate evidence cannot be compared.
    """
    if profile.algorithm_id != CalibratedEqualAlgorithm.algorithm_id or profile.target_definition is None:
        raise ValueError("compare_calibrated requires a calibrated-equal profile with an external target")

    baseline_runs, candidate_runs, corrupted = _usable_report_runs(
        baseline,
        candidate,
        corrupted_trial_indices,
    )
    calibrator = Calibrator.from_profile(profile)
    baseline_result = calibrator.apply_profile(baseline_runs)
    candidate_result = calibrator.apply_profile(candidate_runs)
    pairs = _pair_calibrated_results(
        baseline_result.case_run_results,
        candidate_result.case_run_results,
        target_definition=profile.target_definition,
    )
    summaries = summarize_pairs(
        pairs,
        confidence_level=confidence_level,
        minimum_effect=minimum_effect,
    )
    return CalibratedComparisonResult(
        pairs=pairs,
        evaluator_summaries=summaries,
        verdict=_comparison_verdict(summaries),
        trial_count=len(baseline_runs),
        corrupted_trial_indices=sorted(corrupted),
        profile_id=profile.profile_id,
        target_definition=profile.target_definition,
    )


def _as_reports(report_or_reports: EvaluationReport | Sequence[EvaluationReport]) -> tuple[EvaluationReport, ...]:
    if isinstance(report_or_reports, EvaluationReport):
        return (report_or_reports,)
    return tuple(report_or_reports)


def _as_evaluator_names(names: Collection[str]) -> set[str]:
    if isinstance(names, str):
        raise TypeError("lower_is_better must be a collection of evaluator names, not a string")
    materialized_names = list(names)
    if any(not isinstance(name, str) or not name.strip() for name in materialized_names):
        raise TypeError("lower_is_better must contain only non-empty evaluator names")
    return set(materialized_names)


def _usable_report_runs(
    baseline: Sequence[Sequence[EvaluationReport]],
    candidate: Sequence[Sequence[EvaluationReport]],
    corrupted_trial_indices: Collection[int],
) -> tuple[list[list[EvaluationReport]], list[list[EvaluationReport]], set[int]]:
    baseline_runs = [list(reports) for reports in baseline]
    candidate_runs = [list(reports) for reports in candidate]
    if not baseline_runs or not candidate_runs:
        raise ValueError("baseline and candidate must each contain at least one agent run")
    if len(baseline_runs) != len(candidate_runs):
        raise ValueError(
            "baseline and candidate must contain the same number of agent runs: "
            f"baseline={len(baseline_runs)}, candidate={len(candidate_runs)}"
        )

    trial_indices = set(range(len(baseline_runs)))
    corrupted = set(corrupted_trial_indices)
    unknown_corrupted = corrupted - trial_indices
    if unknown_corrupted:
        raise ValueError(
            "corrupted_trial_indices contain unknown trials: "
            f"unknown={sorted(unknown_corrupted)}, valid={sorted(trial_indices)}"
        )
    if corrupted == trial_indices:
        raise ValueError("no usable trials remain after removing corrupted trials")
    return (
        [reports for index, reports in enumerate(baseline_runs) if index not in corrupted],
        [reports for index, reports in enumerate(candidate_runs) if index not in corrupted],
        corrupted,
    )


def _pair_calibrated_results(
    baseline: Sequence[CalibrationCaseRunResult],
    candidate: Sequence[CalibrationCaseRunResult],
    *,
    target_definition: CalibrationTargetDefinition,
) -> list[PairedScore]:
    baseline_scores = {(result.agent_run, result.case_name): _required_probability(result) for result in baseline}
    candidate_scores = {(result.agent_run, result.case_name): _required_probability(result) for result in candidate}
    if set(baseline_scores) != set(candidate_scores):
        raise ValueError(
            "baseline and candidate do not contain the same calibrated case-runs: "
            f"baseline_only={sorted(set(baseline_scores) - set(candidate_scores))}, "
            f"candidate_only={sorted(set(candidate_scores) - set(baseline_scores))}"
        )

    evaluator = f"target:{target_definition.name}@{target_definition.version}"
    case_names = sorted({case_name for _, case_name in baseline_scores})
    return [
        PairedScore(
            case_id=case_name,
            evaluator=evaluator,
            baseline_score=fmean(score for (_, name), score in baseline_scores.items() if name == case_name),
            candidate_score=fmean(score for (_, name), score in candidate_scores.items() if name == case_name),
            baseline_trial_count=sum(name == case_name for _, name in baseline_scores),
            candidate_trial_count=sum(name == case_name for _, name in candidate_scores),
        )
        for case_name in case_names
    ]


def _required_probability(result: CalibrationCaseRunResult) -> float:
    if result.calibrated_probability is None:
        raise ValueError("the calibration profile did not produce target probabilities")
    return result.calibrated_probability


def _comparison_verdict(summaries: Sequence[EvaluatorSummary]) -> PairedComparisonVerdict:
    if not summaries:
        return "insufficient_data"
    verdicts = {summary.verdict for summary in summaries}
    if "improved" in verdicts and "regressed" in verdicts:
        return "tradeoff"
    if "regressed" in verdicts:
        return "regressed"
    if "insufficient_data" in verdicts:
        return "insufficient_data"
    if "improved" in verdicts:
        return "improved"
    return "no_clear_change"
