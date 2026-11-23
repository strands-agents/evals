"""The end phase decides, once: finalists from the search ranking, the pick on validation, the test report.

Imports only `scoring` and `score_model`, never `search` or `selection`: the search proposes, this decides.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scipy import stats

from ..types.optimization import (
    CandidateScores,
    Estimate,
    PairedComparison,
    ValidationReport,
    Values,
    near,
)
from .score_model.protocols import Fit
from .score_model.ranking import Assessment, LowerBound, Mean, PickRule, assess, label_candidate, pareto_front, rank
from .scoring.objectives import AtMost, Criteria, measured_metrics

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
        """`n_finalists` an int (else `TypeError`) >= 1 (else `ValueError`); `rule` `Mean()` or `LowerBound(q)`."""
        if isinstance(self.n_finalists, bool) or not isinstance(self.n_finalists, int):
            raise TypeError(f"FinalPick(n_finalists=) must be an int, got {self.n_finalists!r}")
        if self.n_finalists < 1:
            raise ValueError(f"FinalPick(n_finalists=) must be >= 1, got {self.n_finalists!r}")
        if not isinstance(self.rule, (Mean, LowerBound)):
            raise TypeError(f"FinalPick(rule=) must be Mean() or LowerBound(q), got {type(self.rule).__name__}")

    def config(self) -> dict[str, Any]:
        """`{"n_finalists": n, "rule": rule.config()}`; stored in `run.json`."""
        return {"n_finalists": self.n_finalists, "rule": self.rule.config()}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> FinalPick:
        """Rebuild from `config()`'s output on resume; the rule is `Mean()` or `LowerBound(q)`.

        Raises:
            ValueError: An unknown rule name.
        """
        rule = config["rule"]
        if rule["kind"] == "Mean":
            return cls(n_finalists=config["n_finalists"], rule=Mean())
        if rule["kind"] == "LowerBound":
            return cls(n_finalists=config["n_finalists"], rule=LowerBound(q=rule["q"]))
        raise ValueError(f"unknown FinalPick rule {rule['kind']!r} in run.json")


def choose_finalists(
    ranking: Sequence[str],
    baseline_id: str,
    n: int,
    *,
    assessments: Mapping[str, Assessment],
    train: Mapping[str, CandidateScores],
    criteria: Criteria,
    n_validation_cases: int,
    front: Sequence[str] | None = None,
) -> list[str]:
    """The first `n` of the search ranking, baseline excluded, those holding every constraint with margin first.

    A feasible candidate holds a constraint with margin when it still meets it after allowing for how far
    its mean over the validation cases can differ from its mean over the train cases (`_holds_with_margin`),
    so a candidate whose train mean is close to a bound, or whose train cases are spread out, comes after
    those with margin. Each group keeps the ranking order. Without constraints, or when no candidate has
    margin, this is the ranking's first `n`. Fewer -> all of them; none -> `[]`.

    Takes the ranked ids, not a `SelectionContext`, so this module doesn't import `selection/`. Under a
    `Pareto` objective the pool is `front` instead, ordered the same way; an empty front (nothing feasible
    on train) gives `[]`, so the baseline ships.

    Args:
        ranking: Every eligible candidate in ranking order (feasible first, then by objective).
        baseline_id: Left out; it is always measured on validation anyway.
        n: `FinalPick.n_finalists`, or `max(n_finalists, PARETO_MIN_FINALISTS)` under `Pareto`.
        assessments: Candidate id -> its assessment in the search ranking; every pool member has one.
        train: Candidate id -> its train scores; every pool member has them.
        criteria: The constraints and the feasibility threshold.
        n_validation_cases: The number of validation cases the finalists will be measured on.
        front: Under `Pareto`, `pareto_front` over the search assessments, in ranking order; None otherwise.
    """
    pool = [c for c in (ranking if front is None else front) if c != baseline_id]
    margin = {c for c in pool if _holds_with_margin(assessments[c], train[c], criteria, n_validation_cases)}
    return [*(c for c in pool if c in margin), *(c for c in pool if c not in margin)][:n]


def _holds_with_margin(
    assessment: Assessment, scores: CandidateScores, criteria: Criteria, n_validation_cases: int
) -> bool:
    """Feasible on train, and every constraint met at `criteria.feasibility_threshold` with room for new cases.

    Each constraint is judged as `assess` judges it under a fit (`ranking.py`): at the threshold's quantile
    against the bound, a value at the bound up to float noise met. The estimate judged has the ranking's
    mean of the metric (the fit's mean, else, for a flagged metric or without a fit, the adjusted train mean,
    `Assessment.adjusted_means`, so a
    candidate N/A on some cases isn't judged on its other cases alone) and the sd of the difference between
    a mean over the train cases and one over the validation cases: the stdev of the candidate's per-case
    train means times `sqrt(1 / measured train cases + 1 / expected measured validation cases)`. For a
    metric judged by its adjusted mean the stdev is the larger of the measured cases' and that over those plus
    the cases the adjusted mean filled in (`Assessment.adjusted_by_case`): a candidate N/A on the hard cases
    doesn't look steady on its easy ones, and filled values near its own level don't hide the spread of its
    measured ones. The filled values are estimated from the measured ones, so both counts cover only the cases
    the metric has a value on: the means compared are means over measured cases, and the expected validation
    count assumes the metric is measured there at the train rate (measured train cases / train cases). Fewer
    than 2 measured cases show no spread, so the constraint has no margin.
    """
    if not assessment.is_feasible:
        return False
    threshold = criteria.feasibility_threshold
    for constraint in criteria.constraints:
        metric = constraint.metric
        case_means = [case[metric] for case in scores.metrics_by_case.values() if metric in case]
        if len(case_means) < 2:
            return False
        spread = statistics.stdev(case_means)
        if metric in assessment.estimates:
            mean = assessment.estimates[metric].mean
        else:
            mean = assessment.adjusted_means[metric]
            filled = [case[metric] for case in assessment.adjusted_by_case.values() if metric in case]
            spread = max(spread, statistics.stdev(filled))
        # the validation mean also covers only measured cases: expect the train measurement rate there
        measured_validation = n_validation_cases * len(case_means) / len(scores.metrics_by_case)
        sd = spread * math.sqrt(1 / len(case_means) + 1 / measured_validation)
        at_most = isinstance(constraint, AtMost)
        judged = Estimate(mean=mean, sd=sd).quantile(threshold if at_most else 1 - threshold)
        met = judged <= constraint.bound if at_most else judged >= constraint.bound
        if not (met or near(judged, constraint.bound)):
            return False
    return True


def pick(
    *,
    baseline_id: str,
    finalist_ids: Sequence[str],
    validation: Mapping[str, CandidateScores],
    fit: Fit | None,
    values: Mapping[str, Values],
    trial_counts: Mapping[str, int],
    completed: Mapping[str, int],
    criteria: Criteria,
    rule: PickRule,
    problem: str | None = None,
) -> ValidationReport:
    """The champion: the first feasible candidate on validation, baseline included; else the baseline.

    Feasible = eligible on validation, at least one completed validation trial, and P(met) >=
    `criteria.feasibility_threshold` for every constraint (measured means without a fit, and for a flagged
    metric); a constraint whose metric has no value on any validation case is unmet (`assess`). Ranked by
    objective resolved to a tenth of its sd; ties -> most trials, then earlier.
    Under a `Pareto` objective the same champion (the validated front's highest-gain member) plus
    `ValidationReport.front`.

    Args:
        baseline_id: The baseline.
        finalist_ids: From `choose_finalists`.
        validation: Candidate id -> its validation scores.
        fit: A fresh fit on the validation runs of the baseline and finalists only (never the search fit),
            leaving out those with no completed validation trial; None (fewer than 2 eligible, or the score
            model raised) -> measured means.
        values: Candidate id -> values.
        trial_counts: Candidate id -> validation trials that count.
        completed: Candidate id -> validation trials that completed (a missing id counts as 0). A candidate
            with none is left out of the assessment, so it ships only as the fallback baseline: an
            `agent_failure` scored by `agent_failure_score` counts, but with no completed trial its scores are
            that failure value alone (and a crash's usage).
        criteria: The objective, constraints, and feasibility threshold.
        rule: `FinalPick.rule`, passed to `assess`.
        problem: Why `fit` wasn't used for every goal metric, copied to `ValidationReport.problem`.
    """
    ids = list(dict.fromkeys([baseline_id, *finalist_ids]))
    eligible = [c for c in ids if c in validation and validation[c].is_eligible and completed.get(c, 0) > 0]
    assessments = assess(eligible, fit=fit, criteria=criteria, values=values, measured=validation, rule=rule)
    # The baseline comes first in `ids`, so with equal objectives and trial counts it wins the tie.
    ranked = rank(assessments, ids, trial_counts=trial_counts)
    feasible = [c for c in ranked if assessments[c].is_feasible]
    champion = feasible[0] if feasible else baseline_id
    infeasible: dict[str, str] = {}
    for c in ids:
        if c not in validation or not validation[c].is_eligible:
            infeasible[c] = "not eligible on validation: a case has no counted run for every goal metric"
        elif c not in assessments:
            infeasible[c] = "every validation trial failed, so nothing was measured"
        elif not assessments[c].is_feasible:
            infeasible[c] = label_candidate(validation[c], assessments[c], criteria=criteria).reason
    objectives = {c: assessments[c].objective if c in assessments else None for c in ids}
    report = ValidationReport(
        champion_id=champion,
        front=pareto_front(assessments, ranked) if criteria.is_pareto else [],
        objectives=objectives,
        estimates={c: dict(assessments[c].estimates) for c in ids if c in assessments},
        adjusted_means={c: dict(assessments[c].adjusted_means) for c in ids if c in assessments},
        infeasible=infeasible,
        measured={c: measured_metrics(validation[c], criteria.registry) for c in ids if c in validation},
        p_met={c: dict(assessments[c].p_met) for c in ids if c in assessments},
        problem=problem,
        rule=rule.config(),
        reason="",
    )
    return report.model_copy(update={"reason": _pick_reason(report, feasible)})


def _pick_reason(report: ValidationReport, feasible: Sequence[str]) -> str:
    """One sentence on why `report.champion_id` ships, for each `report.outcome`; `feasible` in ranking order."""
    baseline = next(iter(report.objectives))

    def shown(c: str) -> str:
        value = report.objectives.get(c)
        return "unmeasured" if value is None else f"{value:.4g}"

    outcome = report.outcome
    if outcome == "no_finalists":
        return "no finalist from the search; the baseline ships"
    if outcome == "top_finalist":
        champion = report.champion_id
        return f"{champion} highest on validation: {shown(champion)} vs baseline {shown(baseline)}"
    if outcome == "no_feasible_finalist":
        return "no finalist meets every constraint on validation; the baseline ships"
    best = next(c for c in feasible if c != baseline)
    return f"the baseline is highest on validation: {shown(baseline)} vs best finalist {shown(best)}"


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
    means = {
        "candidate_mean": statistics.fmean(candidate[c] for c in cases),
        "reference_mean": statistics.fmean(reference[c] for c in cases),
    }
    if n == 1:
        return PairedComparison(mean_diff=mean, ci_low=None, ci_high=None, n_cases=n, **means)
    half = _t_quantile(level, n - 1) * statistics.stdev(differences) / math.sqrt(n)
    return PairedComparison(mean_diff=mean, ci_low=mean - half, ci_high=mean + half, n_cases=n, **means)


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
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    if mean_b == 0:
        return None
    ratio = mean_a / mean_b
    means = {"candidate_mean": mean_a, "reference_mean": mean_b}
    if n == 1:
        return PairedComparison(mean_diff=ratio, ci_low=None, ci_high=None, n_cases=n, **means)
    spread = statistics.variance(a) + ratio * ratio * statistics.variance(b) - 2 * ratio * statistics.covariance(a, b)
    half = _t_quantile(level, n - 1) * math.sqrt(max(spread, 0.0) / (mean_b * mean_b * n))
    return PairedComparison(mean_diff=ratio, ci_low=ratio - half, ci_high=ratio + half, n_cases=n, **means)


def _shared_cases(candidate: Mapping[str, float], reference: Mapping[str, float], *, caller: str) -> list[str]:
    """The cases both measured, sorted by name; `ValueError` naming `caller` when a shared mean isn't finite."""
    cases = sorted(set(candidate) & set(reference))
    not_finite = [c for c in cases if not (math.isfinite(candidate[c]) and math.isfinite(reference[c]))]
    if not_finite:
        raise ValueError(f"{caller}: per-case means must be finite; not for {not_finite}")
    return cases


def _t_quantile(level: float, df: int) -> float:
    """The two-sided `level` quantile of Student's t with `df` degrees of freedom."""
    return float(stats.t.ppf(0.5 + level / 2, df))
