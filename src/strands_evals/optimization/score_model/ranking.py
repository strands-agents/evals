"""The one ranking: assessing candidates against the objective and constraints, from a fit or measured means.

Shared by `TopRanked`, the finalists, inheritance fill, `Nested`'s round best, the order of
`ProposalRequest.candidates`, and the candidate label (`label_candidate`). Here, left of
`selection/` and `end_phase.py`, so neither imports the other. Without a fit, after a refit that raised
(until one succeeds), and while any metric is flagged, every candidate is assessed by measured means.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...types.optimization import CandidateLabel, CandidateScores, Estimate, Values
from ..scoring.objectives import Constraint, Criteria
from .protocols import Fit


def round_to_sd(value: float, sd: float) -> float:
    """`value` rounded to a tenth of `sd`'s leading digit.

    Candidates the model can't tell apart then tie, so the tie rules can fire on float estimates; `sd` 0
    (measured means) leaves `value` unchanged.
    """
    raise NotImplementedError


@dataclass(frozen=True)
class Assessment:
    """One candidate as the ranking sees it.

    Attributes:
        objective: `Criteria.objective_value` over each goal metric's resolved estimate, or the measured
            objective (`CandidateScores.objective`) when assessed by measured means.
        terms: `Criteria.terms` over the same values (oriented, the rule applied, resolved): what
            `pareto_front` compares and the `Pareto` tie-break reads; None when `objective` is None.
        feasible: Every constraint has P(met) >= `criteria.feasibility` under the fit, or is met on measured means.
        violations: Per constraint, <= 0 when met: the metric's quantile at `criteria.feasibility` against the limit,
            or the measured mean against the limit.
        violation: The sum of the positive violations (0 when feasible); what "least violation" ranks by.
        p_met: Per constraint; empty when assessed by measured means.
        estimates: Per goal metric, `fit.estimates(values, candidate)`; empty when assessed by measured means.
        by_fit: False when assessed by measured means (no fit, the last refit raised, any metric flagged, or
            the candidate not in the fit).
    """

    candidate_id: str
    objective: float | None
    terms: tuple[float, ...] | None
    feasible: bool
    violations: tuple[float, ...]
    violation: float
    p_met: dict[str, float]
    estimates: dict[str, Estimate]
    by_fit: bool


@dataclass(frozen=True)
class Mean:
    """An objective term's value is its posterior mean (the default)."""

    def config(self) -> dict[str, Any]:
        """`{"kind": "Mean"}`."""
        raise NotImplementedError


@dataclass(frozen=True)
class LowerBound:
    """Decide cautiously: each objective term's `q`-quantile on its worse side.

    `q` for a maximized metric, `1 - q` for a minimized one, so a candidate with few or noisy runs (wide sd)
    ranks lower than its mean says. `0 < q < 0.5`, else `ValueError`.
    """

    q: float = 0.1

    def __post_init__(self) -> None:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """`{"kind": "LowerBound", "q": q}`."""
        raise NotImplementedError


PickRule = Mean | LowerBound

_MEAN = Mean()  # the default rule, built once (rules are frozen)


def assess(
    ids: Sequence[str],
    *,
    fit: Fit | None,
    criteria: Criteria,
    values: Mapping[str, Values],
    measured: Mapping[str, CandidateScores],
    rule: PickRule = _MEAN,
) -> dict[str, Assessment]:
    """Assess each candidate from the fit, or from measured means without a usable one.

    `rule` turns each objective term's `Estimate` into one number before `round_to_sd()` and
    `Criteria.objective_value`; constraints are unaffected (P(met) >= `criteria.feasibility`), and measured
    means ignore it. The search always uses `Mean()`; only the final pick passes `FinalPick.rule`.

    Args:
        ids: The candidates to assess.
        fit: The fit, or None.
        criteria: The objective, constraints, and feasibility threshold.
        values: Candidate id -> values.
        measured: Candidate id -> its scores on the fitted split.
        rule: How an objective term's estimate becomes a number.
    """
    raise NotImplementedError


def rank(
    assessments: Mapping[str, Assessment],
    order: Sequence[str],
    *,
    trial_counts: Mapping[str, int] | None = None,
) -> list[str]:
    """Feasible first by objective (highest first), then the infeasible by least violation, then by objective.

    Ties (equal after resolving): with `trial_counts` (the final pick), most trials, then earlier in
    `order`, so the baseline wins when nothing beats it, under `Pareto` too; without (the search), under
    `Pareto` first the larger `terms` (compared in objective order), then earlier in `order`.

    Args:
        assessments: Candidate id -> its assessment.
        order: The candidates in scoring order.
        trial_counts: Candidate id -> counted trials; given only at the final pick.
    """
    raise NotImplementedError


def pareto_front(assessments: Mapping[str, Assessment], ids: Sequence[str]) -> list[str]:
    """The feasible candidates among `ids` that no other feasible one dominates, in `ids`' order.

    A dominates B when every one of A's `terms` is >= B's and they aren't all equal. Infeasible candidates are
    left out: constraints filter, they are never front dimensions. Used by the end phase only (the
    finalists from the search ranking, the validated front from the validation one).

    Args:
        assessments: Candidate id -> its assessment.
        ids: The candidates to consider, in ranking order.
    """
    raise NotImplementedError


def p_met(constraint: Constraint, value: Estimate) -> float:
    """P(constraint holds), 0 to 1, under the metric's estimate.

    Read by the ranking (feasible = P(met) >= `criteria.feasibility` for every constraint) and by the final pick
    under the validation fit.
    """
    raise NotImplementedError


def label_candidate(
    candidate: CandidateScores,
    assessment: Assessment | None,
    *,
    criteria: Criteria,
) -> CandidateLabel:
    """The candidate label: `feasible`, `infeasible`, or `ineligible`, as the step's ranking assessed it.

    A label recorded in the event log, never a filter; train data only.

    Feasibility is the ranking's own: P(met) >= `criteria.feasibility` under the fit,
    measured means without a fit or while any metric is flagged, so the label always matches how the loop
    treated the candidate. A snapshot: later refits can change how the ranking sees it. An infeasible
    candidate stays a possible parent, ranked after the feasible ones; an ineligible one is out of the fit
    and the parents until it is scored again and its unmeasured trials succeed.

    Args:
        candidate: Its train scores (`SplitScores.train`).
        assessment: Its `Assessment` from the step's ranking right after the refit; None when ineligible.
        criteria: Names the constraints in the reasons; holds the feasibility threshold.
    """
    raise NotImplementedError
