"""The end phase decides, once: finalists from the search ranking, the pick on validation, the test report.

Imports only `scoring` and `score_model`, never `search` or `selection`: the search proposes, this decides.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..types.optimization import CandidateScores, PairedComparison, ValidationReport, Values
from .score_model.protocols import Fit
from .score_model.ranking import _MEAN, PickRule
from .scoring.objectives import Criteria

TEST_LEVEL = 0.90  # the test report's interval level (report only), matching the 90% effects


@dataclass(frozen=True, kw_only=True)
class FinalPick:
    """The final pick: how many finalists go to validation, and how the pick values them.

    Only a candidate meeting every constraint ships (no `unmet="closest"`). `rule` is `Mean()` or
    `LowerBound(q)` to ship cautiously; the search's ranking always uses `Mean()`. Read back from
    `run.json` on resume (`from_config`), never passed again. Under a `Pareto` objective up to
    `max(finalists, 5)` front members go to validation.
    """

    finalists: int = 3
    rule: PickRule = _MEAN

    def __post_init__(self) -> None:
        """`finalists` >= 1, else `ValueError`."""
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """`{"finalists": n, "rule": rule.config()}`; stored in `run.json`."""
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
        n: `FinalPick.finalists`, or `max(finalists, 5)` under `Pareto`.
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
    rule: PickRule = _MEAN,
    problem: str | None = None,
) -> ValidationReport:
    """The champion: the first feasible candidate on validation, baseline included; else the baseline.

    Feasible = eligible on validation and P(met) >= `criteria.feasibility` for every constraint (measured means
    without a fit). Ranked by objective resolved to a tenth of its sd; ties -> most trials, then earlier.
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
) -> PairedComparison:
    """A t-interval at `level` of the mean per-case difference, over the cases both measured.

    The case is the unit. Deterministic, no seed (scipy, from the `optimization` extra). P1 may replace
    this with the core compare feature.

    Args:
        candidate: Case name -> the candidate's per-case mean.
        reference: Case name -> the reference's per-case mean.
        level: Two-sided interval level, `TEST_LEVEL`.

    Returns:
        The mean difference (candidate - reference), its interval, and the number of shared cases.
    """
    raise NotImplementedError


def paired_ratio(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    level: float,
) -> PairedComparison | None:
    """The ratio of means with a delta-method interval, over the cases both measured.

    Args:
        candidate: Case name -> the candidate's per-case mean.
        reference: Case name -> the reference's per-case mean.
        level: Two-sided interval level.

    Returns:
        The ratio and its interval; None when the reference mean is 0 or the metric can be < 0.
    """
    raise NotImplementedError
