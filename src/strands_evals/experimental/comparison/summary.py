"""Per-evaluator statistics for paired evaluation scores."""

import math
import random
from collections import defaultdict
from statistics import fmean
from typing import Sequence

from pydantic import BaseModel

from .pairing import PairedScore


class EvaluatorSummary(BaseModel):
    """Summary of paired score changes for one evaluator.

    Attributes:
        evaluator: Name of the evaluator being summarized.
        case_count: Number of paired cases in the summary.
        mean_delta: Mean candidate-minus-baseline score change.
        confidence_level: Coverage level used for the confidence interval.
        confidence_lower: Lower bound of the case-bootstrap interval, or
            `None` when fewer than two cases are available.
        confidence_upper: Upper bound of the case-bootstrap interval, or
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
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> list[EvaluatorSummary]:
    """Summarize paired case deltas separately for each evaluator.

    The confidence interval is a deterministic percentile bootstrap over
    complete paired cases. Each `(evaluator, case_id)` pair may appear only
    once so that duplicate rows are not treated as independent evidence.

    Args:
        pairs: Scores returned by `pair_reports`.
        confidence_level: Desired confidence interval coverage.
        bootstrap_samples: Number of case-bootstrap resamples.
        seed: Seed used to make bootstrap results reproducible.

    Returns:
        One summary per evaluator, sorted by evaluator name.

    Raises:
        ValueError: If the interval settings are invalid or a case/evaluator
            pair is duplicated.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be greater than 0 and less than 1")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1")

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

    random_generator = random.Random(seed)
    summaries: list[EvaluatorSummary] = []
    for evaluator in sorted(grouped):
        evaluator_pairs = sorted(grouped[evaluator], key=lambda pair: pair.case_id)
        deltas = [pair.delta for pair in evaluator_pairs]
        confidence_interval = (
            _bootstrap_mean_interval(
                deltas,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                random_generator=random_generator,
            )
            if len(deltas) >= 2
            else (None, None)
        )
        summaries.append(
            EvaluatorSummary(
                evaluator=evaluator,
                case_count=len(deltas),
                mean_delta=fmean(deltas),
                confidence_level=confidence_level,
                confidence_lower=confidence_interval[0],
                confidence_upper=confidence_interval[1],
            )
        )

    return summaries


def _bootstrap_mean_interval(
    values: list[float],
    *,
    confidence_level: float,
    bootstrap_samples: int,
    random_generator: random.Random,
) -> tuple[float, float]:
    sample_size = len(values)
    bootstrap_means = sorted(fmean(random_generator.choices(values, k=sample_size)) for _ in range(bootstrap_samples))
    tail_probability = (1.0 - confidence_level) / 2.0
    return (
        _percentile(bootstrap_means, tail_probability),
        _percentile(bootstrap_means, 1.0 - tail_probability),
    )


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * (position - lower_index)
