"""Per-evaluator statistics for paired evaluation scores."""

import math
from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean, stdev

from pydantic import BaseModel
from scipy import stats
from scipy.stats import PermutationMethod, binomtest, false_discovery_control, wilcoxon

from .pairing import PairedScore

_DELTA_DECIMAL_PLACES = 12


class EvaluatorSummary(BaseModel):
    """Summary of paired score changes for one evaluator.

    Attributes:
        evaluator: Name of the evaluator being summarized.
        case_count: Number of paired cases in the summary.
        mean_delta: Mean candidate-minus-baseline score change.
        confidence_level: Coverage level used for the confidence interval.
        confidence_lower: Lower bound of the paired-mean t interval, or
            `None` when fewer than two cases are available.
        confidence_upper: Upper bound of the paired-mean t interval, or
            `None` when fewer than two cases are available.
        wilcoxon_p_value: Two-sided paired Wilcoxon p-value, or `None` when
            fewer than two cases are available.
        adjusted_p_value: Wilcoxon p-value adjusted across evaluators using
            Benjamini-Hochberg false discovery rate control, or `None` when
            the Wilcoxon test is unavailable.
    """

    evaluator: str
    case_count: int
    mean_delta: float
    confidence_level: float
    confidence_lower: float | None
    confidence_upper: float | None
    wilcoxon_p_value: float | None
    adjusted_p_value: float | None


def summarize_pairs(
    pairs: Sequence[PairedScore],
    *,
    confidence_level: float = 0.95,
) -> list[EvaluatorSummary]:
    """Summarize paired case deltas separately for each evaluator.

    The confidence interval is a Student t interval for the mean change over
    complete paired cases. Each `(evaluator, case_id)` pair may appear only once
    so that duplicate rows are not treated as independent evidence. Deltas are
    rounded before analysis so mathematically equivalent score changes receive
    the same rank. Equal-magnitude changes use an exact sign test, and other
    tied changes use a deterministic permutation Wilcoxon test. The adjusted
    p-values control false discovery across the evaluators in this call. A
    workflow comparing several candidates must recompute the correction over
    the full set of candidate/evaluator tests before selecting a winner.

    Args:
        pairs: Aggregated paired case scores.
        confidence_level: Desired confidence interval coverage.

    Returns:
        One summary per evaluator, sorted by evaluator name.

    Raises:
        ValueError: If the interval settings are invalid or a case/evaluator
            pair is duplicated.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be greater than 0 and less than 1")

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

    summaries: list[EvaluatorSummary] = []
    for evaluator in sorted(grouped):
        evaluator_pairs = sorted(grouped[evaluator], key=lambda pair: pair.case_id)
        raw_deltas = [pair.delta for pair in evaluator_pairs]
        if not all(math.isfinite(delta) for delta in raw_deltas):
            raise ValueError(f"pairs contain a non-finite score change for evaluator: evaluator={evaluator!r}")
        deltas = [round(delta, _DELTA_DECIMAL_PLACES) for delta in raw_deltas]
        mean_delta = fmean(deltas)
        confidence_interval = _mean_interval(deltas, confidence_level) if len(deltas) >= 2 else (None, None)
        summaries.append(
            EvaluatorSummary(
                evaluator=evaluator,
                case_count=len(deltas),
                mean_delta=mean_delta,
                confidence_level=confidence_level,
                confidence_lower=confidence_interval[0],
                confidence_upper=confidence_interval[1],
                wilcoxon_p_value=_wilcoxon_p_value(deltas) if len(deltas) >= 2 else None,
                adjusted_p_value=None,
            )
        )

    tested_summaries = [summary for summary in summaries if summary.wilcoxon_p_value is not None]
    if tested_summaries:
        adjusted_p_values = false_discovery_control(
            [summary.wilcoxon_p_value for summary in tested_summaries],
            method="bh",
        )
        for summary, adjusted_p_value in zip(tested_summaries, adjusted_p_values, strict=True):
            summary.adjusted_p_value = float(adjusted_p_value)

    return summaries


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

    method: str | PermutationMethod
    if len(set(magnitudes)) < len(magnitudes):
        method = PermutationMethod(n_resamples=99_999, rng=0)
    else:
        method = "auto"
    return float(wilcoxon(changed, alternative="two-sided", method=method).pvalue)
