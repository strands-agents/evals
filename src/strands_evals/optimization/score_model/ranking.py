"""The one ranking: assessing candidates against the objective and constraints, from a fit or measured means.

Shared by `TopRanked`, the finalists, inheritance fill, `Nested`'s round best, the order of
`ProposalRequest.candidates`, and the candidate label (`label_candidate`). Here, left of
`selection/` and `end_phase.py`, so neither imports the other. Without a fit, after a refit that raised
(until one succeeds), and while any metric is flagged, every candidate is assessed by measured means.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...types.optimization import CandidateLabel, CandidateScores, Estimate, Values
from ..scoring.objectives import AtMost, Constraint, Criteria, Maximize, Minimize, near
from .protocols import Fit


def round_to_sd(value: float, sd: float) -> float:
    """`value` rounded to the decimal place of `sd / 10`'s first digit.

    Two estimates closer than the model's uncertainty then become equal, so the tie rules decide between
    them instead of float noise. An `sd` of 0 (measured means) or a value that isn't finite is returned
    unchanged.

    E.g. with `sd=0.3`, `sd / 10` is 0.03, so values round to 0.01: 1.23456 -> 1.23. With `sd=0.0007`,
    they round to 0.00001: 0.499996 -> 0.5.
    """
    if not sd > 0 or not math.isfinite(value):
        return value
    return round(value, -math.floor(math.log10(sd / 10)))


@dataclass(frozen=True)
class Assessment:
    """One candidate as the ranking sees it.

    Attributes:
        objective: `Criteria.objective_value` over each goal metric's resolved estimate, or the measured
            objective (`CandidateScores.objective`) when assessed by measured means.
        terms: `Criteria.terms` over the same values (oriented, the rule applied, resolved): what
            `pareto_front` compares and the `Pareto` tie-break reads; None when `objective` is None.
        is_feasible: Every constraint has P(met) >= `criteria.feasibility_threshold` under the fit, or is
            met on measured means.
        violations: Per constraint, <= 0 when met: the metric's quantile at
            `criteria.feasibility_threshold` against the limit, or the measured mean against the limit (0 when
            equal up to float noise, infinite when the metric was never measured).
        total_violation: The sum of the positive violations (0 when feasible); what "least violation" ranks by.
        p_met: Per constraint; empty when assessed by measured means.
        estimates: Per goal metric, `fit.estimates(values, candidate)`; empty when assessed by measured means.
        is_from_fit: False when assessed by measured means (no fit, the last refit raised, any metric flagged, or
            the candidate not in the fit).
    """

    candidate_id: str
    objective: float | None
    terms: tuple[float, ...] | None
    is_feasible: bool
    violations: tuple[float, ...]
    total_violation: float
    p_met: dict[str, float]
    estimates: dict[str, Estimate]
    is_from_fit: bool


@dataclass(frozen=True)
class Mean:
    """An objective term's value is its posterior mean (the default)."""

    def config(self) -> dict[str, Any]:
        """`{"kind": "Mean"}`."""
        return {"kind": "Mean"}


@dataclass(frozen=True)
class LowerBound:
    """Decide cautiously: each objective term's `q`-quantile on its worse side.

    `q` for a maximized metric, `1 - q` for a minimized one, so a candidate with few or noisy runs (wide sd)
    ranks lower than its mean says. `0 < q < 0.5`, else `ValueError`.
    """

    q: float = 0.1

    def __post_init__(self) -> None:
        if not 0 < self.q < 0.5:
            raise ValueError(f"LowerBound(q) needs 0 < q < 0.5, got {self.q!r}")

    def config(self) -> dict[str, Any]:
        """`{"kind": "LowerBound", "q": q}`."""
        return {"kind": "LowerBound", "q": self.q}


PickRule = Mean | LowerBound


def assess(
    ids: Sequence[str],
    *,
    fit: Fit | None,
    criteria: Criteria,
    values: Mapping[str, Values],
    measured: Mapping[str, CandidateScores],
    rule: PickRule,
) -> dict[str, Assessment]:
    """Assess each candidate from the fit, or from measured means without a usable one.

    `rule` turns each objective term's `Estimate` into one number before `round_to_sd()` and
    `Criteria.objective_value`; constraints are unaffected (P(met) >= `criteria.feasibility_threshold`),
    and measured means ignore it. Required: the search passes `Mean()`, the final pick `FinalPick.rule`.

    The fit is used for a candidate it was fitted on, unless any metric is flagged; every other candidate is
    assessed by its measured means (`is_from_fit` False).

    Args:
        ids: The candidates to assess.
        fit: The fit, or None.
        criteria: The objective, constraints, and feasibility threshold.
        values: Candidate id -> values.
        measured: Candidate id -> its scores on the fitted split.
        rule: How an objective term's estimate becomes a number.
    """
    objective = criteria.objective
    parts = (objective,) if isinstance(objective, (Maximize, Minimize)) else objective.objectives
    maximized = {o.metric: isinstance(o, Maximize) for o in parts}
    threshold = criteria.feasibility_threshold
    usable = fit is not None and not fit.flagged
    out: dict[str, Assessment] = {}
    for candidate_id in ids:
        if usable and fit is not None and candidate_id in fit.candidates:
            estimates = fit.estimates(values[candidate_id], candidate_id)
            point = {}
            for metric, is_max in maximized.items():
                estimate = estimates[metric]
                value = estimate.mean if isinstance(rule, Mean) else estimate.quantile(rule.q if is_max else 1 - rule.q)
                point[metric] = round_to_sd(value, estimate.sd)
            # Each constraint judged at the required probability: the value its metric stays below (AtMost) or
            # above (AtLeast) with probability `threshold`.
            judged: list[float | None] = [
                estimates[c.metric].quantile(threshold if isinstance(c, AtMost) else 1 - threshold)
                for c in criteria.constraints
            ]
            p = {str(c): p_met(c, estimates[c.metric]) for c in criteria.constraints}
            objective_value = criteria.objective_value(point)
            terms = criteria.terms(point)
            from_fit = True
        else:
            scores = measured[candidate_id]
            point = {
                metric: statistics.fmean(case_values)
                for metric in criteria.goal_metrics
                if (case_values := [case[metric] for case in scores.metrics_by_case.values() if metric in case])
            }
            judged = [point.get(c.metric) for c in criteria.constraints]
            p, estimates = {}, {}
            objective_value = scores.objective
            terms = criteria.terms(point)
            from_fit = False
        # As `Criteria.unmet_constraints`: a value at its bound up to float noise is met, a constraint never
        # measured is not.
        violations = tuple(
            math.inf
            if v is None
            else 0.0
            if near(v, c.bound)
            else (v - c.bound if isinstance(c, AtMost) else c.bound - v)
            for c, v in zip(criteria.constraints, judged, strict=True)
        )
        out[candidate_id] = Assessment(
            candidate_id=candidate_id,
            objective=objective_value,
            terms=terms if objective_value is not None else None,
            is_feasible=all(v <= 0 for v in violations),
            violations=violations,
            total_violation=sum((v for v in violations if v > 0), 0.0),
            p_met=p,
            estimates=estimates,
            is_from_fit=from_fit,
        )
    return out


def rank(
    assessments: Mapping[str, Assessment],
    order: Sequence[str],
    *,
    is_pareto: bool = False,
    trial_counts: Mapping[str, int] | None = None,
) -> list[str]:
    """Feasible first by objective (highest first), then the infeasible by least violation, then by objective.

    Ties (equal after resolving): with `trial_counts` (the final pick), most trials, then earlier in
    `order`, so the baseline wins when nothing beats it, under `Pareto` too; without (the search), under
    `Pareto` first the larger `terms` (compared in objective order), then earlier in `order`. Equal
    violations among infeasible candidates go by the higher objective, then the same tie rules.

    Args:
        assessments: Candidate id -> its assessment.
        order: The candidates in scoring order; only those in `assessments` are ranked.
        is_pareto: The objective is a `Pareto` (`Criteria.is_pareto`), so search ties go to the larger terms.
        trial_counts: Candidate id -> counted trials; given only at the final pick.
    """
    position = {candidate_id: i for i, candidate_id in enumerate(order)}

    def key(candidate_id: str) -> tuple[Any, ...]:
        a = assessments[candidate_id]
        objective = a.objective if a.objective is not None else -math.inf
        if trial_counts is not None:
            tie: tuple[float, ...] = (-trial_counts.get(candidate_id, 0),)
        elif is_pareto and a.terms is not None:
            tie = tuple(-t for t in a.terms)
        else:
            tie = ()
        violation = 0.0 if a.is_feasible else a.total_violation
        return (not a.is_feasible, violation, -objective, *tie, position[candidate_id])

    return sorted((c for c in order if c in assessments), key=key)


def pareto_front(assessments: Mapping[str, Assessment], ids: Sequence[str]) -> list[str]:
    """The feasible candidates among `ids` that no other feasible one dominates, in `ids`' order.

    A dominates B when every one of A's `terms` is >= B's and they aren't all equal. Infeasible candidates are
    left out: constraints filter, they are never front dimensions. Used by the end phase only (the
    finalists from the search ranking, the validated front from the validation one).

    Args:
        assessments: Candidate id -> its assessment.
        ids: The candidates to consider, in ranking order.
    """
    pool = {c: terms for c in ids if assessments[c].is_feasible and (terms := assessments[c].terms) is not None}

    def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        return all(x >= y for x, y in zip(a, b, strict=True)) and a != b

    return [c for c, terms in pool.items() if not any(dominates(other, terms) for other in pool.values())]


def p_met(constraint: Constraint, value: Estimate) -> float:
    """P(constraint holds), 0 to 1, under the metric's estimate.

    Read by the ranking (feasible = P(met) >= `criteria.feasibility_threshold` for every constraint)
    and by the final pick under the validation fit. With sd 0, 1 when the mean meets the bound (equal up to
    float noise counts), else 0.
    """
    at_most = isinstance(constraint, AtMost)
    if value.sd <= 0:
        met = value.mean <= constraint.bound if at_most else value.mean >= constraint.bound
        return 1.0 if met or near(value.mean, constraint.bound) else 0.0
    below = 0.5 * (1.0 + math.erf((constraint.bound - value.mean) / value.sd / math.sqrt(2.0)))
    return below if at_most else 1.0 - below


def label_candidate(
    candidate: CandidateScores,
    assessment: Assessment | None,
    *,
    criteria: Criteria,
) -> CandidateLabel:
    """The candidate label: `feasible`, `infeasible`, or `ineligible`, as the step's ranking assessed it.

    A label recorded in the event log, never a filter; train data only.

    Feasibility is the ranking's own: P(met) >= `criteria.feasibility_threshold` under the fit,
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
