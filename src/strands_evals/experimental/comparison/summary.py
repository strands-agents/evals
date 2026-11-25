"""Per-evaluator statistics for paired evaluation scores."""

import math
from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean, stdev

from pydantic import BaseModel
from scipy import stats

from .pairing import PairedScore


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
    """

    evaluator: str
    case_count: int
    mean_delta: float
    confidence_level: float
    confidence_lower: float | None
    confidence_upper: float | None


def summarize_pairs(
    pairs: Sequence[PairedScore],
    *,
    confidence_level: float = 0.95,
) -> list[EvaluatorSummary]:
    """Summarize paired case deltas separately for each evaluator.

    The confidence interval is a Student t interval for the mean change over
    complete paired cases. Each `(evaluator, case_id)` pair may appear only once
    so that duplicate rows are not treated as independent evidence.

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
        deltas = [pair.delta for pair in evaluator_pairs]
        if not all(math.isfinite(delta) for delta in deltas):
            raise ValueError(f"pairs contain a non-finite score change for evaluator: evaluator={evaluator!r}")
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
            )
        )

    return summaries


def _mean_interval(values: list[float], confidence_level: float) -> tuple[float, float]:
    mean = fmean(values)
    if min(values) == max(values):
        return mean, mean
    half_width = (
        float(stats.t.ppf(0.5 + confidence_level / 2, len(values) - 1)) * stdev(values) / math.sqrt(len(values))
    )
    return mean - half_width, mean + half_width
