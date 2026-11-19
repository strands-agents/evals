"""The one ranking: assessing candidates against the objective and constraints, from a fit or measured means.

Shared by `TopRanked`, the finalists, inheritance fill, `Nested`'s round best, the order of
`ProposalRequest.candidates`, and the candidate label (`label_candidate`). Here, left of
`selection/` and `end_phase.py`, so neither imports the other. Without a fit, after a refit that raised
(until one succeeds), and while any metric is flagged, every candidate is assessed by measured means,
adjusted for the cases where an evaluator found nothing to judge for a candidate.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...types.optimization import CandidateLabel, CandidateScores, Estimate, Values, near
from ..scoring.objectives import AtMost, Constraint, Criteria, Maximize, Minimize, measured_metrics
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
    # log10(sd) - 1, not log10(sd / 10): `sd / 10` underflows to 0 for an sd near the smallest float.
    return round(value, -math.floor(math.log10(sd) - 1))


@dataclass(frozen=True)
class Assessment:
    """One candidate as the ranking sees it.

    Attributes:
        objective: `Criteria.objective_value` over each goal metric's resolved estimate, or, when assessed by
            measured means, over each goal metric's adjusted mean (see `assess`); equal to
            `CandidateScores.objective` when no other assessed candidate has a value on a case this one lacks.
        terms: `Criteria.terms` over the same values (oriented, the rule applied, resolved): what
            `pareto_front` compares and the `Pareto` tie-break reads; None when `objective` is None or any
            objective's metric is unmeasured (`objective` can still be set then: `objective_value` drops a
            missing evaluator term).
        is_feasible: Every constraint has P(met) >= `criteria.feasibility_threshold` under the fit, or is
            met on measured means; a constraint whose metric the candidate has no value for on any case is
            unmet either way.
        violations: Per constraint, <= 0 when met: the metric's quantile at
            `criteria.feasibility_threshold` (`AtMost`) or at 1 minus it (`AtLeast`) against the limit, or the
            measured mean, as for `objective`, against the limit; 0 when equal up to float noise; infinite,
            under the fit as well, when the candidate has no value of the metric on any case.
        total_violation: The sum of the positive violations (0 when feasible); what "least violation" ranks by.
        p_met: Per constraint whose metric was measured; empty when assessed by measured means.
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
    assessed by its measured means (`is_from_fit` False). Either way, a constraint whose metric the candidate
    has no value for on any case of `measured` (an evaluator N/A on every case) is unmet: violation infinite,
    no `p_met`, whatever the fit predicts for it.

    A case where an evaluator found nothing to judge doesn't make a candidate ineligible, so plain means would
    compare candidates on different cases: one that leaves the evaluator nothing to judge on the hard cases
    could rank above one that was judged there. So each goal metric's measured mean is taken over the cases
    any candidate in `ids` has a value for (those assessed from the fit included); a case the candidate has no
    value on is filled with the other candidates' mean there, shifted by the candidate's mean difference from
    them on the cases they share, and clipped to the metric's range. A candidate with a value on every such
    case keeps its plain mean; one with gaps depends on who else is in `ids`. Constraints are judged on the
    same means.

    Args:
        ids: The candidates to assess.
        fit: The fit, or None.
        criteria: The objective, constraints, and feasibility threshold.
        values: Candidate id -> values.
        measured: Candidate id -> its scores on the fitted split, for every id in `ids`.
        rule: How an objective term's estimate becomes a number.
    """
    objective = criteria.objective
    parts = (objective,) if isinstance(objective, (Maximize, Minimize)) else objective.objectives
    maximized = {o.metric: isinstance(o, Maximize) for o in parts}
    threshold = criteria.feasibility_threshold
    usable = fit is not None and not fit.flagged
    adjusted: dict[str, dict[str, float]] | None = None
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
            # above (AtLeast) with probability `threshold`. A metric the candidate has no value for on any case
            # is never measured, as on measured means: the fit's estimate of it is a prediction from the
            # candidate's values alone.
            observed = {metric for case in measured[candidate_id].metrics_by_case.values() for metric in case}
            judged: list[float | None] = [
                estimates[c.metric].quantile(threshold if isinstance(c, AtMost) else 1 - threshold)
                if c.metric in observed
                else None
                for c in criteria.constraints
            ]
            p = {str(c): p_met(c, estimates[c.metric]) for c in criteria.constraints if c.metric in observed}
            objective_value = criteria.objective_value(point)
            terms = criteria.terms(point)
            from_fit = True
        else:
            if adjusted is None:
                adjusted = _adjusted_means(ids, measured, criteria)
            point = adjusted[candidate_id]
            judged = [point.get(c.metric) for c in criteria.constraints]
            p, estimates = {}, {}
            objective_value = criteria.objective_value(point)
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


def _adjusted_means(
    ids: Sequence[str], measured: Mapping[str, CandidateScores], criteria: Criteria
) -> dict[str, dict[str, float]]:
    """Each candidate's goal-metric means for the measured-means ranking, adjusted for the cases it has no value on.

    Per goal metric, over the cases any candidate in `ids` has a value for. A candidate with a value on every
    one of them gets its plain mean. A candidate missing some has each missing case filled with the other
    candidates' mean on that case, shifted by the candidate's mean difference from the others on the cases
    it shares with them (no shift when it shares none), and clipped to the metric's range. A metric the
    candidate has no value for on any case is left out, as with plain means.

    Args:
        ids: The pool; only those in `measured` count.
        measured: Candidate id -> its scores on one split.
        criteria: Gives the goal metrics and their ranges.

    Returns:
        Candidate id -> goal metric -> mean.
    """
    pool = {c: measured[c].metrics_by_case for c in dict.fromkeys(ids) if c in measured}
    out: dict[str, dict[str, float]] = {c: {} for c in pool}
    for metric in criteria.goal_metrics:
        values = {c: {case: m[metric] for case, m in cases.items() if metric in m} for c, cases in pool.items()}
        observed: dict[str, list[float]] = defaultdict(list)
        for own in values.values():
            for case, value in own.items():
                observed[case].append(value)
        # Per case, the pool's sum and mean. The mean is the first value plus the mean offset from it, so equal
        # values give exactly that value back (fsum / n doesn't, e.g. for 0.1 three times).
        sums = {case: math.fsum(case_values) for case, case_values in observed.items()}
        means = {case: vs[0] + (sums[case] - len(vs) * vs[0]) / len(vs) for case, vs in observed.items()}
        low, high = criteria.registry.range(metric)
        for c, own in values.items():
            if not own:
                continue
            missing = [case for case in observed if case not in own]
            if not missing:
                out[c][metric] = statistics.fmean(own.values())
                continue
            # Its difference from the others' mean on each case it shares with them: (n * own - sum) / (n - 1) for
            # n values on the case, exactly 0 when they are all equal.
            differences = [
                (len(observed[case]) * value - sums[case]) / (len(observed[case]) - 1)
                for case, value in own.items()
                if len(observed[case]) > 1
            ]
            shift = statistics.fmean(differences) if differences else 0.0
            # A case it has no value on: `means` there is the others' mean.
            filled = [_clip(means[case] + shift, low, high) for case in missing]
            out[c][metric] = statistics.fmean([*own.values(), *filled])
    return out


def _clip(value: float, low: float | None, high: float | None) -> float:
    """`value` within `[low, high]`; a None end is unbounded."""
    if low is not None and value < low:
        return low
    if high is not None and value > high:
        return high
    return value


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
    `Pareto` first the larger `terms` (compared in objective order; a candidate without `terms` after
    those with them), then earlier in `order`. Equal
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
            tie: tuple[Any, ...] = (-trial_counts.get(candidate_id, 0),)
        elif is_pareto:
            # One shape for every key: a candidate without terms (an objective evaluator N/A on every case)
            # loses the tie, so its position is never compared against another candidate's term.
            tie = (a.terms is None, tuple(-t for t in a.terms or ()))
        else:
            tie = ()
        violation = 0.0 if a.is_feasible else a.total_violation
        return (not a.is_feasible, violation, -objective, *tie, position[candidate_id])

    return sorted((c for c in order if c in assessments), key=key)


def pareto_front(assessments: Mapping[str, Assessment], ids: Sequence[str]) -> list[str]:
    """The feasible candidates among `ids` that no other feasible one dominates, in `ids`' order.

    A dominates B when every one of A's `terms` is >= B's and they aren't all equal. Infeasible candidates are
    left out: constraints filter, they are never front dimensions. Used for the finalists (from the search
    ranking), the validated front (from the validation one), and each step's `best_after`.

    Args:
        assessments: Candidate id -> its assessment.
        ids: The candidates to consider, in ranking order; only those in `assessments` are considered, as `rank`.
    """
    pool = {
        c: terms
        for c in ids
        if c in assessments and assessments[c].is_feasible and (terms := assessments[c].terms) is not None
    }

    def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        return all(x >= y for x, y in zip(a, b, strict=True)) and a != b

    return [c for c, terms in pool.items() if not any(dominates(other, terms) for other in pool.values())]


def p_met(constraint: Constraint, value: Estimate) -> float:
    """P(constraint holds), 0 to 1, under the metric's estimate.

    Reported in `Assessment.p_met` and the label's reasons; feasibility itself is `assess`'s quantile test
    (the same rule, without the float noise at the bound). With sd 0, 1 when the mean meets the bound (equal up to
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
    treated the candidate; a constraint whose metric was never measured reads "never measured" either way. A
    snapshot: later refits can change how the ranking sees it. An infeasible candidate stays a possible parent,
    ranked after the feasible ones; an ineligible one is out of the fit and the parents until it is scored again
    and its unmeasured trials succeed.

    Args:
        candidate: Its train scores (`SplitScores.train`).
        assessment: Its `Assessment` from the step's ranking right after the refit; None when ineligible.
        criteria: Names the constraints in the reasons; holds the feasibility threshold.
    """
    measured = measured_metrics(candidate, criteria.registry)
    if assessment is None or not candidate.is_eligible:
        unmeasured = sorted(c for c, runs in candidate.objective_by_run.items() if all(v is None for v in runs))
        reason = "a train case has no counted run for every goal metric"
        if unmeasured:
            reason += f": {', '.join(unmeasured)}"
        return CandidateLabel(
            outcome="ineligible",
            reason=reason,
            unmet_constraints=[],
            estimates={},
            p_met={},
            is_from_fit=False,
            measured=measured,
        )
    if assessment.is_from_fit:
        threshold = criteria.feasibility_threshold
        unmet = [
            # A constraint without P(met) is one whose metric the candidate has no value for.
            f"{c}: never measured"
            if str(c) not in assessment.p_met
            else f"{c}: P(met) {assessment.p_met[str(c)]:.2g} < {threshold:g}"
            for c, v in zip(criteria.constraints, assessment.violations, strict=True)
            if v > 0
        ]
    else:
        # The measured mean the ranking judged (the bound plus or minus its violation), in
        # `Criteria.unmet_constraints`' wording.
        unmet = [
            f"{c}: never measured"
            if math.isinf(v)
            else f"{c}: mean {c.bound + v if isinstance(c, AtMost) else c.bound - v:.4g} violates by {v:.4g}"
            for c, v in zip(criteria.constraints, assessment.violations, strict=True)
            if v > 0
        ]
    return CandidateLabel(
        outcome="infeasible" if unmet else "feasible",
        reason="; ".join(unmet) if unmet else "meets every constraint",
        unmet_constraints=unmet,
        estimates=dict(assessment.estimates),
        p_met=dict(assessment.p_met),
        is_from_fit=assessment.is_from_fit,
        measured=measured,
    )
