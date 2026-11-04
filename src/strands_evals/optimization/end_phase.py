"""The end phase decides, once: finalists from the search ranking, the pick on validation, the test report.

Imports only `scoring` and `score_model`, never `search` or `selection`: the search proposes, this decides.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..types.optimization import CandidateScores, PairedComparison, ValidationReport, Values
from .score_model.protocols import Fit
from .score_model.ranking import Mean, PickRule
from .scoring.objectives import Criteria

TEST_LEVEL = 0.90  # the test report's interval level (report only), matching the 90% effects
PARETO_MIN_FINALISTS = 5  # under Pareto, max(n_finalists, this) front members go to validation; the reserves use it too


@dataclass(frozen=True, kw_only=True)
class FinalPick:
    """The final pick: how many finalists go to validation, and how the pick values them.

    Only a candidate meeting every constraint ships (no `unmet="closest"`). `rule` is `Mean()` or
    `LowerBound(q)` to ship cautiously; the search's ranking always uses `Mean()`. Read back from
    `run.json` on resume (`from_config`), never passed again. Under a `Pareto` objective up to
    `max(n_finalists, PARETO_MIN_FINALISTS)` front members go to validation.
    """

    n_finalists: int = 3
    rule: PickRule = Mean()

    def __post_init__(self) -> None:
        """`n_finalists` >= 1, else `ValueError`."""
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """`{"n_finalists": n, "rule": rule.config()}`; stored in `run.json`."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> FinalPick:
        """Rebuild from `config()`'s output on resume; the rule is `Mean()` or `LowerBound(q)`.

        Raises:
            ValueError: An unknown rule name.
        """
        raise NotImplementedError


def choose_finalists(
    ranking: Sequence[str], baseline_id: str, n: int, *, front: Sequence[str] | None = None
) -> list[str]:
    """The first `n` of the search ranking, baseline excluded; fewer -> all of them; none -> `[]`.

    Takes the ranked ids, not a `SelectionContext`, so this module doesn't import `selection/`. Under a
    `Pareto` objective the first `n` of `front` instead; an empty front (nothing feasible on train) gives
    `[]`, so the baseline ships.

    Args:
        ranking: Every eligible candidate in ranking order (feasible first, then by objective).
        baseline_id: Left out; it is always measured on validation anyway.
        n: `FinalPick.n_finalists`, or `max(n_finalists, PARETO_MIN_FINALISTS)` under `Pareto`.
        front: Under `Pareto`, `pareto_front` over the search assessments, in ranking order; None otherwise.
    """
    raise NotImplementedError


def pick(
    *,
    baseline_id: str,
    finalist_ids: Sequence[str],
    validation: Mapping[str, CandidateScores],
    fit: Fit | None,
    values: Mapping[str, Values],
    trial_counts: Mapping[str, int],
    criteria: Criteria,
    rule: PickRule,
    problem: str | None = None,
) -> ValidationReport:
    """The champion: the first feasible candidate on validation, baseline included; else the baseline.

    Feasible = eligible on validation and P(met) >= `criteria.feasibility_threshold` for every
    constraint (measured means without a fit). Ranked by objective resolved to a tenth of its sd;
    ties -> most trials, then earlier.
    Under a `Pareto` objective the same champion (the validated front's highest-gain member) plus
    `ValidationReport.front`.

    Args:
        baseline_id: The baseline.
        finalist_ids: From `choose_finalists`.
        validation: Candidate id -> its validation scores.
        fit: A fresh fit on the validation runs of the baseline and finalists only (never the search fit);
            None (fewer than 2 eligible, or the score model raised) -> measured means.
        values: Candidate id -> values.
        trial_counts: Candidate id -> validation trials that count.
        criteria: The objective, constraints, and feasibility threshold.
        rule: `FinalPick.rule`, passed to `assess`.
        problem: Why `fit` is None, copied to `ValidationReport.problem`.
    """
    raise NotImplementedError


def paired_interval(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    level: float,
) -> PairedComparison | None:
    """A t-interval at `level` of the mean per-case difference, over the cases both measured.

    The case is the unit. Deterministic, no seed (scipy, from the `optimization` extra).

    Args:
        candidate: Case name -> the candidate's per-case mean.
        reference: Case name -> the reference's per-case mean.
        level: Two-sided interval level, `TEST_LEVEL`.

    Returns:
        The mean difference (candidate - reference), its interval, and the number of shared cases; no
        interval with one shared case; None with none.

    Raises:
        ValueError: A shared case's mean is NaN or infinite.
    """
    cases = _shared_cases(candidate, reference, caller="paired_interval")
    n = len(cases)
    if n == 0:
        return None
    differences = [candidate[c] - reference[c] for c in cases]
    mean = statistics.fmean(differences)
    if n == 1:
        return PairedComparison(mean_diff=mean, ci_low=None, ci_high=None, n_cases=n)
    half = _t_quantile(level, n - 1) * statistics.stdev(differences) / math.sqrt(n)
    return PairedComparison(mean_diff=mean, ci_low=mean - half, ci_high=mean + half, n_cases=n)


def paired_ratio(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    level: float,
) -> PairedComparison | None:
    """The ratio of means with a delta-method interval, over the cases both measured.

    `mean_diff` holds the ratio (candidate mean / reference mean). The caller leaves out metrics that can be
    < 0, whose ratio means nothing.

    Args:
        candidate: Case name -> the candidate's per-case mean.
        reference: Case name -> the reference's per-case mean.
        level: Two-sided interval level.

    Returns:
        The ratio and its interval (none with one shared case); None with no shared case or when the
        reference mean is 0.

    Raises:
        ValueError: A shared case's mean is NaN or infinite.
    """
    cases = _shared_cases(candidate, reference, caller="paired_ratio")
    n = len(cases)
    if n == 0:
        return None
    a = [candidate[c] for c in cases]
    b = [reference[c] for c in cases]
    mean_b = statistics.fmean(b)
    if mean_b == 0:
        return None
    ratio = statistics.fmean(a) / mean_b
    if n == 1:
        return PairedComparison(mean_diff=ratio, ci_low=None, ci_high=None, n_cases=n)
    spread = statistics.variance(a) + ratio * ratio * statistics.variance(b) - 2 * ratio * statistics.covariance(a, b)
    half = _t_quantile(level, n - 1) * math.sqrt(max(spread, 0.0) / (mean_b * mean_b * n))
    return PairedComparison(mean_diff=ratio, ci_low=ratio - half, ci_high=ratio + half, n_cases=n)


def _shared_cases(candidate: Mapping[str, float], reference: Mapping[str, float], *, caller: str) -> list[str]:
    """The cases both measured, sorted by name; `ValueError` naming `caller` when a shared mean isn't finite."""
    cases = sorted(set(candidate) & set(reference))
    not_finite = [c for c in cases if not (math.isfinite(candidate[c]) and math.isfinite(reference[c]))]
    if not_finite:
        raise ValueError(f"{caller}: per-case means must be finite; not for {not_finite}")
    return cases


def _t_quantile(level: float, df: int) -> float:
    """The two-sided `level` quantile of Student's t with `df` degrees of freedom."""
    # optional extra: optimization
    from scipy.stats import t

    return float(t.ppf(0.5 + level / 2, df))
