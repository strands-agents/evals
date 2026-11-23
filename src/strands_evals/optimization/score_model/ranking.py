"""The one ranking: assessing candidates against the objective and constraints, from a fit or measured means.

Shared by `TopRanked`, the finalists, inheritance fill, `Nested`'s round best, the order of
`ProposalRequest.candidates`, and the candidate label (`label_candidate`). Here, left of
`selection/` and `end_phase.py`, so neither imports the other. Without a fit, after a refit that raised
(until one succeeds), and while every goal metric is flagged, every candidate is assessed by measured means: its
adjusted means (`Assessment.adjusted_means`), the plain means with the cases where an evaluator found nothing
to judge for it filled in from the other candidates. While only some are, each flagged metric is judged by its
adjusted mean and the rest by the fit.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
        objective: `Criteria.objective_value` over each goal metric's resolved estimate (a flagged metric's
            adjusted mean instead), or, when assessed by measured means, over `adjusted_means` (see `assess`);
            equal to `CandidateScores.objective` when no other assessed candidate has a value on a case this one
            lacks.
        terms: `Criteria.terms` over the same values (oriented, the rule applied, resolved): what
            `pareto_front` compares and the `Pareto` tie-break reads; None when `objective` is None or any
            objective's metric is unmeasured (`objective` can still be set then: `objective_value` drops a
            missing evaluator term).
        is_feasible: Every constraint has P(met) >= `criteria.feasibility_threshold` under the fit, or is
            met on `adjusted_means`; a constraint whose metric the candidate has no value for on any case is
            unmet either way.
        violations: Per constraint, <= 0 when met: the metric's quantile at
            `criteria.feasibility_threshold` (`AtMost`) or at 1 minus it (`AtLeast`) against the limit, or, for a
            flagged metric or when assessed by measured means, the metric's adjusted mean
            (`adjusted_means`) against the limit; 0 when equal up to float noise; infinite, under the fit as
            well, when the candidate has no value of the metric on any case.
        total_violation: The sum of the positive violations (0 when feasible); what "least violation" ranks by.
        p_met: Per constraint the fit judged (its metric not flagged, and measured); empty when assessed by
            measured means.
        estimates: Per goal metric the fit judged, `fit.estimates(values, candidate)` without the flagged
            metrics; empty when assessed by measured means. A goal metric missing here was judged by its
            adjusted mean (`adjusted_means`), or never measured.
        adjusted_means: Per goal metric the candidate has a value of, the mean `objective`, `terms`, and
            `violations` were computed from when assessed by measured means (see `assess`): its plain mean
            (`CandidateScores.metrics_by_case` averaged) when no other assessed candidate has a value on a case
            it lacks, else that mean with those cases filled in. Holds every goal metric the fit didn't judge (the
            flagged ones under a fit, all of them on measured means), so it and `estimates` never share a metric;
            empty when the fit judged every goal metric.
        is_from_fit: True when the fit judged at least one goal metric (`estimates` is not empty); False when
            assessed by measured means alone (no fit, the last refit raised, every goal metric flagged, a
            flagged name that isn't a goal metric, or the candidate not in the fit).
        adjusted_by_case: Case -> goal metric -> the value each of `adjusted_means` averages: the candidate's
            own values and the ones filled in; only the metrics in `adjusted_means`.
    """

    candidate_id: str
    objective: float | None
    terms: tuple[float, ...] | None
    is_feasible: bool
    violations: tuple[float, ...]
    total_violation: float
    p_met: dict[str, float]
    estimates: dict[str, Estimate]
    adjusted_means: dict[str, float]
    is_from_fit: bool
    adjusted_by_case: dict[str, dict[str, float]] = field(default_factory=dict)


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

    The fit is used for a candidate it was fitted on, metric by metric: a flagged metric is judged by its
    adjusted mean (below), in the objective and in the constraints alike, and the others by the fit. A
    flagged metric's mean is not resolved by `round_to_sd()`, and its constraint is judged at the mean, without
    the `feasibility_threshold` margin. Every other candidate is assessed by its measured means (`is_from_fit`
    False), and so is every candidate when every goal metric is flagged or a flagged name isn't a goal metric.
    Either way, a constraint whose metric the candidate has no value for on any case of `measured` (an
    evaluator N/A on every case) is unmet: violation infinite, no `p_met`, whatever the fit predicts for it.

    A case where an evaluator found nothing to judge doesn't make a candidate ineligible, so plain means would
    compare candidates on different cases: one that leaves the evaluator nothing to judge on the hard cases
    could rank above one that was judged there. So instead of its plain mean, each goal metric gets an
    adjusted mean (`Assessment.adjusted_means`) over the cases any candidate in `ids` has a value for (those
    assessed from the fit included): the candidate's own values, plus each case it has no value on filled
    with the mean, over the other candidates with a value there, of that value plus the candidate's mean
    difference from that candidate on the cases the two share, weighed by the number of those cases, and
    clipped to the metric's range widened to the values measured on that case (the values averaged are in
    `Assessment.adjusted_by_case`). A candidate with a value on every such case gets its plain mean; one
    with gaps depends on who else is in `ids`. Constraints are judged on the same adjusted means.
    The plain means stay in `CandidateScores` and `measured_metrics`.

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
    # The goal metrics the fit judges: the unflagged ones; none when a flagged name isn't a goal metric, since
    # that flag can't be traced to the metric it concerns.
    by_fit: list[str] = []
    if fit is not None and set(fit.flagged) <= set(criteria.goal_metrics):
        by_fit = [m for m in criteria.goal_metrics if m not in fit.flagged]
    adjusted: tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]] | None = None
    out: dict[str, Assessment] = {}
    for candidate_id in ids:
        estimates: dict[str, Estimate] = {}
        if by_fit and fit is not None and candidate_id in fit.candidates:
            fitted = fit.estimates(values[candidate_id], candidate_id)
            estimates = {m: fitted[m] for m in by_fit}
        # Every goal metric the fit doesn't judge is judged by its adjusted mean, in the objective and the constraints
        # alike: all of them without a usable fit, the flagged ones with one.
        adjusted_means: dict[str, float] = {}
        adjusted_by_case: dict[str, dict[str, float]] = {}
        if any(m not in estimates for m in criteria.goal_metrics):
            if adjusted is None:
                adjusted = _adjusted_means(ids, measured, criteria)
            adjusted_means = {m: v for m, v in adjusted[0][candidate_id].items() if m not in estimates}
            adjusted_by_case = {
                case: kept
                for case, by_metric in adjusted[1][candidate_id].items()
                if (kept := {m: v for m, v in by_metric.items() if m not in estimates})
            }
        point = dict(adjusted_means)
        for metric, is_max in maximized.items():
            if metric in estimates:
                estimate = estimates[metric]
                value = estimate.mean if isinstance(rule, Mean) else estimate.quantile(rule.q if is_max else 1 - rule.q)
                point[metric] = round_to_sd(value, estimate.sd)
        # A constraint the fit judges is judged at the required probability: the value its metric stays below
        # (AtMost) or above (AtLeast) with probability `threshold`. A metric the candidate has no value for on any
        # case is never measured, as on measured means: the fit's estimate of it is a prediction from the
        # candidate's values alone.
        observed = {metric for case in measured[candidate_id].metrics_by_case.values() for metric in case}
        judged: list[float | None] = [
            (
                estimates[c.metric].quantile(threshold if isinstance(c, AtMost) else 1 - threshold)
                if c.metric in observed
                else None
            )
            if c.metric in estimates
            else adjusted_means.get(c.metric)
            for c in criteria.constraints
        ]
        p = {
            str(c): p_met(c, estimates[c.metric])
            for c in criteria.constraints
            if c.metric in estimates and c.metric in observed
        }
        objective_value = criteria.objective_value(point)
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
            terms=criteria.terms(point) if objective_value is not None else None,
            is_feasible=all(v <= 0 for v in violations),
            violations=violations,
            total_violation=sum((v for v in violations if v > 0), 0.0),
            p_met=p,
            estimates=estimates,
            adjusted_means=adjusted_means,
            is_from_fit=bool(estimates),
            adjusted_by_case=adjusted_by_case,
        )
    return out


def _adjusted_means(
    ids: Sequence[str], measured: Mapping[str, CandidateScores], criteria: Criteria
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    """Each candidate's goal-metric means for the measured-means ranking, adjusted for the cases it has no value on.

    Per goal metric, over the cases any candidate in `ids` has a value for. A candidate with a value on every
    one of them gets its plain mean. A candidate missing some has each missing case filled with the mean, over
    the other candidates with a value there, of that value plus the candidate's mean difference from that
    candidate on the cases the two share, weighed by the number of those cases (a candidate sharing none counts
    only when none shares any, unshifted), and clipped to the metric's range widened to the values measured on
    that case.
    So each filled value is shifted against the candidates it comes from: when two candidates share a gap,
    neither one's values move the other's fill. A metric the candidate has no value for on any case is left
    out, as with plain means.

    Args:
        ids: The pool; only those in `measured` count.
        measured: Candidate id -> its scores on one split.
        criteria: Gives the goal metrics and their ranges.

    Returns:
        Candidate id -> goal metric -> mean, and candidate id -> case -> goal metric -> the value each mean
        averages (its own and the filled ones).
    """
    pool = {c: measured[c].metrics_by_case for c in dict.fromkeys(ids) if c in measured}
    means: dict[str, dict[str, float]] = {c: {} for c in pool}
    by_case: dict[str, dict[str, dict[str, float]]] = {c: defaultdict(dict) for c in pool}
    for metric in criteria.goal_metrics:
        values = {c: {case: m[metric] for case, m in cases.items() if metric in m} for c, cases in pool.items()}
        cases = list(dict.fromkeys(case for own in values.values() for case in own))
        low, high = criteria.registry.range(metric)
        for c, own in values.items():
            if not own:
                continue
            missing = [case for case in cases if case not in own]
            filled: dict[str, float] = {}
            if missing:
                # Its mean difference from each other candidate on the cases the two share (0 when they share
                # none), so a case is filled from the candidates judged there, each shifted by its own difference.
                # Its mean difference from each other candidate on the cases the two share, and how many
                # cases that rests on; 0 and 0 when they share none.
                offsets: dict[str, float] = {}
                weights: dict[str, int] = {}
                for d, theirs in values.items():
                    if d != c and theirs:
                        shared = [case for case in own if case in theirs]
                        offsets[d] = statistics.fmean([own[k] - theirs[k] for k in shared]) if shared else 0.0
                        weights[d] = len(shared)
                for case in missing:
                    # Some other candidate has a value on `case`, so `judged` isn't empty. Each one judged there gives
                    # its value plus the difference, weighed by the cases the difference rests on; its value alone
                    # only when none shares a case with this candidate.
                    judged = [d for d in offsets if case in values[d]]
                    donors = [d for d in judged if weights[d]] or judged
                    terms = [values[d][case] + offsets[d] for d in donors]
                    ws = [weights[d] or 1 for d in donors]
                    # The first term plus the weighted mean offset from it, so equal terms give exactly that value
                    # back (fsum / n doesn't, e.g. for 0.1 three times).
                    mean = terms[0] + math.fsum(w * (t - terms[0]) for w, t in zip(ws, terms, strict=True)) / sum(ws)
                    # Within the range, widened to the values measured on the case: a fill leaves the range only as
                    # far as a measured value there does.
                    there = [values[d][case] for d in judged]
                    filled[case] = _clip(
                        mean, None if low is None else min(low, *there), None if high is None else max(high, *there)
                    )
            for case, value in {**own, **filled}.items():
                by_case[c][case][metric] = value
            means[c][metric] = statistics.fmean([*own.values(), *filled.values()])
    return means, {c: dict(cases) for c, cases in by_case.items()}


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

    Feasibility is the ranking's own, per constraint: P(met) >= `criteria.feasibility_threshold` under the
    fit, the adjusted mean for a flagged metric or without a fit, so the label always matches how the loop
    treated the candidate; a constraint whose metric was never measured reads "never measured" either way. A
    snapshot: later refits can change how the ranking sees it. An infeasible candidate stays a possible parent,
    ranked after the feasible ones; an ineligible one is out of the fit and the parents until it is scored again
    and its unmeasured trials succeed (one with no trial left to re-run and no failed evaluator never is).

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
    threshold = criteria.feasibility_threshold
    unmet = [
        f"{c}: P(met) {assessment.p_met[str(c)]:.2g} < {threshold:g}"
        if str(c) in assessment.p_met
        else _unmet_by_mean(c, assessment.adjusted_means.get(c.metric), v)
        for c, v in zip(criteria.constraints, assessment.violations, strict=True)
        if v > 0
    ]
    return CandidateLabel(
        outcome="infeasible" if unmet else "feasible",
        reason="; ".join(unmet) if unmet else "meets every constraint",
        unmet_constraints=unmet,
        estimates=dict(assessment.estimates),
        p_met=dict(assessment.p_met),
        adjusted_means=dict(assessment.adjusted_means),
        is_from_fit=assessment.is_from_fit,
        measured=measured,
    )


def _unmet_by_mean(constraint: Constraint, mean: float | None, violation: float) -> str:
    """The sentence `Criteria.unmet_constraints` gives for a constraint whose adjusted `mean` misses by `violation`.

    The mean is the one judged, not the bound plus or minus the violation, which can differ in the last digit.
    """
    if mean is None or math.isinf(violation):
        return f"{constraint}: never measured"
    return f"{constraint}: mean {mean:.4g} violates by {violation:.4g}"
