"""Per-evaluator statistics for paired evaluation scores."""

import math
from collections import defaultdict
from collections.abc import Collection, Sequence
from statistics import fmean, stdev
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy import stats
from scipy.stats import binomtest, false_discovery_control, rankdata, wilcoxon

from .pairing import PairedScore

_DELTA_DECIMAL_PLACES = 12
_EXACT_TIED_MAX_CHANGES = 500
EvaluatorVerdict = Literal["improved", "regressed", "no_clear_change", "insufficient_data"]


class EvaluatorSummary(BaseModel):
    """Summary of paired score changes for one evaluator.

    Attributes:
        evaluator: Name of the evaluator being summarized.
        case_count: Number of paired cases in the summary.
        changed_case_count: Number of cases with a non-zero rounded delta.
        higher_is_better: Whether a positive delta is considered beneficial.
        mean_delta: Mean candidate-minus-baseline score change.
        confidence_level: Coverage level used for the confidence interval;
            `1 - confidence_level` is also the significance threshold.
        minimum_effect: Smallest meaningful change; the full confidence
            interval must lie beyond this threshold for a directional verdict.
        confidence_lower: Lower bound of the paired-mean t interval, or
            `None` when fewer than two cases are available.
        confidence_upper: Upper bound of the paired-mean t interval, or
            `None` when fewer than two cases are available.
        wilcoxon_p_value: Two-sided paired Wilcoxon p-value, or `None` when
            fewer than two cases are available.
        adjusted_p_value: Wilcoxon p-value adjusted across evaluators using
            Benjamini-Hochberg false discovery rate control, or `None` when
            the Wilcoxon test is unavailable.
        verdict: `improved` or `regressed` when the adjusted p-value passes the
            significance threshold and the full confidence interval exceeds
            the minimum effect in the corresponding direction;
            `no_clear_change` when the evidence is testable but not
            directional, including all-zero changes; or `insufficient_data`
            when a directional test cannot reach the threshold.
    """

    evaluator: str
    case_count: int
    changed_case_count: int
    higher_is_better: bool
    mean_delta: float
    confidence_level: float
    minimum_effect: float
    confidence_lower: float | None
    confidence_upper: float | None
    wilcoxon_p_value: float | None
    adjusted_p_value: float | None
    verdict: EvaluatorVerdict


def summarize_pairs(
    pairs: Sequence[PairedScore],
    *,
    confidence_level: float = 0.95,
    minimum_effect: float = 0.0,
    lower_is_better: Collection[str] = (),
) -> list[EvaluatorSummary]:
    """Summarize paired case deltas separately for each evaluator.

    The confidence interval is a Student t interval for the mean change over
    complete paired cases. Each `(evaluator, case_id)` pair may appear only once
    so that duplicate rows are not treated as independent evidence. Deltas are
    rounded before analysis so mathematically equivalent score changes receive
    the same rank. Equal-magnitude changes use an exact sign test. Other tied
    changes use the exact Wilcoxon signed-rank distribution through 500 changes
    and the deterministic normal approximation for larger samples. The
    p-values control false discovery across every evaluator with a testable
    p-value in this call, so one evaluator's verdict can depend on the others.
    A workflow comparing several candidates must recompute the correction over
    the full set of candidate/evaluator tests before selecting a winner.
    Direction comes from the mean confidence interval, while the adjusted
    p-value supplies the significance check. `confidence_level` therefore also
    determines the significance threshold. A directional verdict is reported
    as `insufficient_data` when too few cases changed to reach that threshold,
    including after multiple-testing correction.

    Args:
        pairs: Aggregated paired case scores.
        confidence_level: Desired confidence interval coverage. Its complement
            is also used as the adjusted-p-value threshold.
        minimum_effect: Smallest meaningful absolute score change. A
            directional verdict requires the entire confidence interval to lie
            strictly beyond this threshold.
        lower_is_better: Evaluator names for which a lower score is beneficial.
            All other evaluators are treated as higher-is-better.

    Returns:
        One summary per evaluator, sorted by evaluator name.

    Raises:
        TypeError: If `lower_is_better` is passed as one bare string.
        ValueError: If the interval settings are invalid, a score direction
            names an unknown evaluator, or a case/evaluator pair is duplicated.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be greater than 0 and less than 1")
    if not math.isfinite(minimum_effect) or minimum_effect < 0.0:
        raise ValueError("minimum_effect must be finite and greater than or equal to 0")

    if isinstance(lower_is_better, str):
        raise TypeError("lower_is_better must be a collection of evaluator names, not a string")
    lower_is_better_names = set(lower_is_better)
    grouped: defaultdict[str, list[PairedScore]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        key = (pair.evaluator, pair.case_id)
        if key in seen:
            raise ValueError(
                f"pairs contain duplicate case/evaluator row: case_id={pair.case_id!r}, evaluator={pair.evaluator!r}"
            )
        seen.add(key)
        grouped[pair.evaluator].append(pair)

    unknown_direction_names = lower_is_better_names - grouped.keys()
    if unknown_direction_names:
        raise ValueError(
            "lower_is_better names unknown evaluators: "
            f"unknown={sorted(unknown_direction_names)}, known={sorted(grouped)}"
        )

    summaries: list[EvaluatorSummary] = []
    for evaluator in sorted(grouped):
        evaluator_pairs = sorted(grouped[evaluator], key=lambda pair: pair.case_id)
        raw_deltas = [pair.delta for pair in evaluator_pairs]
        if not all(math.isfinite(delta) for delta in raw_deltas):
            raise ValueError(f"pairs contain a non-finite score change for evaluator: evaluator={evaluator!r}")
        deltas = [round(delta, _DELTA_DECIMAL_PLACES) for delta in raw_deltas]
        changed_case_count = sum(delta != 0.0 for delta in deltas)
        mean_delta = fmean(deltas)
        confidence_interval = _mean_interval(deltas, confidence_level) if len(deltas) >= 2 else (None, None)
        summaries.append(
            EvaluatorSummary(
                evaluator=evaluator,
                case_count=len(deltas),
                changed_case_count=changed_case_count,
                higher_is_better=evaluator not in lower_is_better_names,
                mean_delta=mean_delta,
                confidence_level=confidence_level,
                minimum_effect=minimum_effect,
                confidence_lower=confidence_interval[0],
                confidence_upper=confidence_interval[1],
                wilcoxon_p_value=_wilcoxon_p_value(deltas) if len(deltas) >= 2 else None,
                adjusted_p_value=None,
                verdict="insufficient_data",
            )
        )

    tested_summaries = [summary for summary in summaries if summary.wilcoxon_p_value is not None]
    minimum_adjusted_p_values: dict[str, float] = {}
    if tested_summaries:
        adjusted_p_values = false_discovery_control(
            [summary.wilcoxon_p_value for summary in tested_summaries],
            method="bh",
        )
        for summary, adjusted_p_value in zip(tested_summaries, adjusted_p_values, strict=True):
            summary.adjusted_p_value = float(adjusted_p_value)
        minimum_adjusted_p_values = _minimum_adjusted_p_values(tested_summaries)

    for summary in summaries:
        summary.verdict = _evaluator_verdict(
            summary,
            minimum_adjusted_p_value=minimum_adjusted_p_values.get(summary.evaluator),
        )

    return summaries


def _evaluator_verdict(
    summary: EvaluatorSummary,
    *,
    minimum_adjusted_p_value: float | None,
) -> EvaluatorVerdict:
    significance_level = 1.0 - summary.confidence_level
    if summary.changed_case_count == 0:
        return "no_clear_change"
    if (
        summary.adjusted_p_value is None
        or summary.confidence_lower is None
        or summary.confidence_upper is None
        or minimum_adjusted_p_value is None
        or minimum_adjusted_p_value > significance_level
    ):
        return "insufficient_data"
    if summary.adjusted_p_value > significance_level:
        return "no_clear_change"

    if summary.higher_is_better:
        if summary.confidence_lower > summary.minimum_effect:
            return "improved"
        if summary.confidence_upper < -summary.minimum_effect:
            return "regressed"
    else:
        if summary.confidence_upper < -summary.minimum_effect:
            return "improved"
        if summary.confidence_lower > summary.minimum_effect:
            return "regressed"
    return "no_clear_change"


def _minimum_adjusted_p_values(summaries: Sequence[EvaluatorSummary]) -> dict[str, float]:
    observed_p_values = [summary.wilcoxon_p_value for summary in summaries]
    minimum_adjusted: dict[str, float] = {}
    for index, summary in enumerate(summaries):
        possible_p_values = list(observed_p_values)
        possible_p_values[index] = min(1.0, _minimum_wilcoxon_p_value(summary.changed_case_count))
        adjusted = false_discovery_control(possible_p_values, method="bh")
        minimum_adjusted[summary.evaluator] = float(adjusted[index])
    return minimum_adjusted


def _minimum_wilcoxon_p_value(changed_case_count: int) -> float:
    return 2.0 ** (1 - changed_case_count)


def _mean_interval(values: list[float], confidence_level: float) -> tuple[float, float]:
    mean = fmean(values)
    if min(values) == max(values):
        return values[0], values[0]
    half_width = (
        float(stats.t.ppf(0.5 + confidence_level / 2, len(values) - 1)) * stdev(values) / math.sqrt(len(values))
    )
    return mean - half_width, mean + half_width


def _wilcoxon_p_value(deltas: list[float]) -> float:
    changed = [delta for delta in deltas if delta != 0.0]
    if not changed:
        return 1.0

    magnitudes = [abs(delta) for delta in changed]
    if min(magnitudes) == max(magnitudes):
        positive_count = sum(delta > 0.0 for delta in changed)
        return float(binomtest(positive_count, len(changed), p=0.5, alternative="two-sided").pvalue)

    if len(set(magnitudes)) < len(magnitudes):
        if len(changed) <= _EXACT_TIED_MAX_CHANGES:
            return _exact_tied_wilcoxon_p_value(changed)
        return float(wilcoxon(changed, alternative="two-sided", method="approx").pvalue)
    return float(wilcoxon(changed, alternative="two-sided", method="auto").pvalue)


def _exact_tied_wilcoxon_p_value(changed: list[float]) -> float:
    doubled_ranks = [round(2 * rank) for rank in rankdata([abs(delta) for delta in changed])]
    total_rank = sum(doubled_ranks)
    distribution = np.zeros(total_rank + 1)
    distribution[0] = 1.0
    for rank in doubled_ranks:
        shifted = np.zeros_like(distribution)
        shifted[rank:] = distribution[:-rank]
        distribution = (distribution + shifted) / 2

    positive_rank = sum(rank for rank, delta in zip(doubled_ranks, changed, strict=True) if delta > 0.0)
    extreme = np.abs(2 * np.arange(total_rank + 1) - total_rank) >= abs(2 * positive_rank - total_rank)
    return float(min(1.0, distribution[extreme].sum()))
