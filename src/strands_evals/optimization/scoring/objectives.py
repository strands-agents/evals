"""Objectives, constraints, and turning trials into scores."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from scipy import stats

from ...evaluators.evaluator import Evaluator
from ...types.evaluation_report import EvaluationReport
from ...types.optimization import (
    CandidateScores,
    CaseSplit,
    Measured,
    Range,
    SplitName,
    SplitScores,
    TrialRecord,
    TrialStatus,
    Usage,
    near,
)
from .metrics import USAGE_METRICS, MetricRegistry

MetricTarget = str | Evaluator | type[Evaluator]


@dataclass(frozen=True, init=False)
class _Direction:
    """One metric to maximize or minimize; `weight` and `scale` are read only inside `WeightedSum` or `Pareto`."""

    metric: str
    weight: float
    scale: float | None

    def __init__(self, target: MetricTarget, /, *, weight: float = 1.0, scale: float | None = None):
        """Name the metric; an evaluator instance resolves via `get_name()`, a class via `get_type_name()`.

        Names are checked against the run's evaluators when the `Optimizer` is built.

        Args:
            target: A metric name, an evaluator class, or an evaluator instance.
            weight: Relative importance inside `WeightedSum` or `Pareto`.
            scale: The size of a meaningful change, so metrics in different units combine; None means the
                metric's range width (1.0 for a 0-1 evaluator). An unbounded metric (every usage metric)
                inside `WeightedSum` or `Pareto` needs it.

        Raises:
            TypeError: `weight` or `scale` is not a number.
            ValueError: `weight` or `scale` is not a finite number > 0.
        """
        object.__setattr__(self, "metric", _metric_name(target))
        object.__setattr__(self, "weight", _positive("weight", weight))
        object.__setattr__(self, "scale", None if scale is None else _positive("scale", scale))

    def __str__(self) -> str:
        """E.g. `"Minimize(tokens, scale=1000)"`; stored in `run.json`."""
        settings = [self.metric]
        if self.weight != 1.0:
            settings.append(f"weight={_number(self.weight)}")
        if self.scale is not None:
            settings.append(f"scale={_number(self.scale)}")
        return f"{type(self).__name__}({', '.join(settings)})"


class Maximize(_Direction):
    """Maximize a metric, e.g. `Maximize("Helpfulness")`."""


class Minimize(_Direction):
    """Minimize a metric, e.g. `Minimize("tokens")`."""


@dataclass(frozen=True, init=False)
class _Combination:
    """Two or more objectives combined; `WeightedSum` and `Pareto` differ only in how the end phase uses them."""

    objectives: tuple[Maximize | Minimize, ...]

    def __init__(self, *objectives: Maximize | Minimize):
        """Combine objectives; each objective's `weight` sets its share.

        Args:
            *objectives: At least one `Maximize` or `Minimize`.

        Raises:
            TypeError: No objectives, or one that isn't a `Maximize` / `Minimize`.
            ValueError: The same target string twice, or a usage metric name without `scale=`. `Criteria`
                repeats both checks on resolved names and ranges (an evaluator with an unbounded
                `score_range`, `"Helpfulness"` next to `HelpfulnessEvaluator`).
        """
        # A combination of one objective is that objective, so it needs no scale to combine with others.
        _check_terms(type(self).__name__, objectives, need_scale=len(objectives) > 1)
        object.__setattr__(self, "objectives", tuple(objectives))

    def __str__(self) -> str:
        """E.g. `"WeightedSum(Maximize(Helpfulness), Minimize(tokens, scale=1000))"`; stored in `run.json` and
        compared on resume."""
        return f"{type(self).__name__}({', '.join(str(o) for o in self.objectives)})"


class WeightedSum(_Combination):
    """Several objectives as one number: per case, the weighted mean of each term's distance from its worst end.

    Each term is (oriented value - worst end) / scale: a bounded term goes from 0 at its worst end to 1 at
    its best with the default scale, whatever its range or direction; a term with no worst end (e.g.
    `Minimize("tokens")`) is its oriented value / scale.
    `WeightedSum(Maximize("Helpfulness"), Minimize("tokens", scale=1000))` makes 1,000 tokens worth
    as much as the whole 0-1 Helpfulness range. Per case, N/A evaluator terms drop and the remaining
    evaluator weights are rescaled to the same total; usage terms are never rescaled. All evaluator
    terms N/A, or any usage term unmeasured, leaves the case unmeasured. A `WeightedSum` of one
    objective is that objective.
    """


class Pareto(_Combination):
    """Several objectives optimized jointly; one champion ships, and the front is reported.

    `Pareto(Maximize("Helpfulness", weight=2), Minimize("tokens", scale=1000))`. The front is the candidates
    no other one beats on every objective. Candidates are ranked by the weighted gain, `WeightedSum`'s value
    with each objective's own `weight` and `scale`, and the champion is the validated front's highest-gain
    member (the baseline when nothing feasible beats it); every front member is tested against the
    baseline. Constraints only filter candidates; they are never front dimensions. A `Pareto` of one
    objective is that objective.
    """


Objective = Maximize | Minimize | WeightedSum | Pareto


@dataclass(frozen=True, init=False)
class _Bound:
    """A bound on one metric's mean; during the search it only ranks, at the final pick it decides."""

    metric: str
    bound: float

    def __init__(self, target: MetricTarget, /, bound: float):
        """Name the metric as in `Maximize`.

        Args:
            target: A metric name, an evaluator class, or an evaluator instance.
            bound: The limit on the mean; by position or keyword (`AtMost("tokens", bound=4000)`).

        Raises:
            TypeError: `bound` is not a finite number.
        """
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound):
            raise TypeError(f"{type(self).__name__} bound must be a finite number, got {bound!r}")
        object.__setattr__(self, "metric", _metric_name(target))
        object.__setattr__(self, "bound", float(bound))

    def __str__(self) -> str:
        """E.g. `"tokens <= 4000"`; stored in `run.json`."""
        operator = "<=" if isinstance(self, AtMost) else ">="
        return f"{self.metric} {operator} {_number(self.bound)}"


class AtMost(_Bound):
    """The metric's mean must not exceed `bound`, e.g. `AtMost("tokens", 4000)`."""


class AtLeast(_Bound):
    """The metric's mean must not fall below `bound`, e.g. `AtLeast("Correctness", 0.8)`."""


Constraint = AtMost | AtLeast


@dataclass(frozen=True)
class _Term:
    """One objective term, resolved: its metric, direction, weight in the combination, scale, and offset.

    `is_usage` marks a usage metric (tokens, seconds, ...): when another term is missing on a case, its
    weight is never rescaled, and when it is missing itself, the case is left unmeasured.

    `offset` is the term's oriented worst end (the range's low end under `Maximize`, minus its high end
    under `Minimize`), subtracted before dividing by `scale`, so every bounded term of a combination
    spans 0 at worst to 1 at best (with the default scale). 0 for a lone objective and for a term whose
    worst end is unbounded (usage metrics under `Minimize`).
    """

    metric: str
    maximize: bool
    weight: float
    scale: float
    is_usage: bool
    offset: float = 0.0


class Criteria:
    """The objective plus the constraints; names resolve through the registry at construction.

    Attributes:
        objective: What to optimize, the default filled in, every metric name resolved.
        constraints: The constraints, metric names resolved.
        registry: The run's metric registry.
        agent_failure_score: What an `agent_failure` run's evaluator metrics count as.
        feasibility_threshold: The P(met) threshold a constraint needs to count as met (not a fraction of cases);
            read by the ranking, the candidate label, and the final pick.
        goal_metrics: The metrics the score model fits, one model each: the objective's metrics, then the
            constraints' metrics, each once, in that order.
        is_pareto: The objective is a `Pareto` of two or more objectives, so the end phase works from the front.
    """

    objective: Objective
    constraints: tuple[Constraint, ...]
    registry: MetricRegistry
    agent_failure_score: Literal["worst"] | float | None
    feasibility_threshold: float
    goal_metrics: tuple[str, ...]
    is_pareto: bool

    def __init__(
        self,
        objective: Objective | None,
        constraints: Sequence[Constraint],
        registry: MetricRegistry,
        *,
        agent_failure_score: Literal["worst"] | float | None = "worst",
        feasibility_threshold: float = 0.9,
    ):
        """`objective=None` means maximize the mean of every evaluator's score.

        `agent_failure_score` is what an `agent_failure` run's evaluator metrics count
        as: "worst" = each evaluator metric's worst end for the goals that use it (0 for a 0-1 score
        maximized), a number = that score, None = missing data. Infrastructure and evaluator failures are
        always missing. Usage metrics have no worst end, so a crashed run's usage counts as measured.

        Args:
            objective: What to optimize; None means the mean of every evaluator's score.
            constraints: Bounds every shipped candidate must meet.
            registry: Resolves the metric names and gives their ranges.
            agent_failure_score: What an `agent_failure` run's evaluator metrics count as.
            feasibility_threshold: The P(met) threshold a constraint needs
                (`Optimizer(feasibility_threshold=)`), 0 < x < 1.

        Raises:
            ValueError: A metric name is unknown or ambiguous; inside a `WeightedSum` or `Pareto`, the same
                metric twice after resolution, a metric with an unbounded range and no `scale=`, or an
                evaluator with no end on its worse side (a low end under `Maximize`, a high end under
                `Minimize`), since dropping its term on an N/A case would score the case above a perfect one; a lone
                `Maximize` / `Minimize` with a non-default `weight` or `scale` (read only inside a
                `WeightedSum` or `Pareto`, so it would be silently ignored); `feasibility_threshold` outside (0, 1);
                no objective and no evaluators to default to; `agent_failure_score` that isn't "worst", a finite
                number, or None, or a number outside the range of an evaluator a goal uses.
            TypeError: `objective` or a constraint of an unknown type.

        A `WeightedSum` or `Pareto` of one objective is unwrapped to that objective, so it follows the lone
        objective's rule (no `weight=` / `scale=`).
        """
        if not 0 < feasibility_threshold < 1:
            raise ValueError(f"feasibility_threshold must lie in (0, 1), got {feasibility_threshold!r}")
        if not (
            agent_failure_score is None
            or agent_failure_score == "worst"
            or (
                isinstance(agent_failure_score, (int, float))
                and not isinstance(agent_failure_score, bool)
                and math.isfinite(agent_failure_score)
            )
        ):
            raise ValueError(
                f'agent_failure_score must be "worst", a finite number, or None, got {agent_failure_score!r}'
            )
        self.registry = registry
        self.agent_failure_score = agent_failure_score
        self.feasibility_threshold = feasibility_threshold
        self.objective = self._resolve_objective(objective)
        self.constraints = tuple(self._resolve_constraint(c) for c in constraints)
        self.is_pareto = isinstance(self.objective, Pareto)
        self._terms = self._build_terms(self.objective)
        # Fixed by the objective; computed once, since objective_value runs for every run, case, and fit estimate.
        self._evaluator_terms = tuple(t for t in self._terms if not t.is_usage)
        self._usage_terms = tuple(t for t in self._terms if t.is_usage)
        self._total_weight = sum(t.weight for t in self._terms)
        self._evaluator_weight = sum(t.weight for t in self._evaluator_terms)
        self.goal_metrics = tuple(dict.fromkeys([t.metric for t in self._terms] + [c.metric for c in self.constraints]))
        if isinstance(agent_failure_score, (int, float)):
            for name in self.goal_metrics:
                if not registry.is_usage(name) and not _in_range(agent_failure_score, registry.range(name)):
                    low, high = registry.range(name)
                    raise ValueError(
                        f"agent_failure_score={agent_failure_score!r} is outside {name}'s range [{low}, {high}]; "
                        'use agent_failure_score="worst" to score each evaluator at its own worst end'
                    )
        # Computed once: read for every evaluator of every crashed run.
        self._agent_failure_values = {name: self._compute_agent_failure_value(name) for name in registry.evaluators}

    def agent_failure_value(self, metric: str) -> float | None:
        """What an `agent_failure` run counts as for one evaluator metric.

        Under `agent_failure_score="worst"`: the range's low end when every goal on the metric is a `Maximize` or
        `AtLeast`, the high end when every one is a `Minimize` or `AtMost`; None when that end is unbounded,
        the goals disagree, or no goal uses the metric. A number gives that number, or None for a metric no
        goal uses whose range doesn't contain it (outside a goal metric's range, `Criteria` raises); None
        gives None. Usage metrics always give None (a crash's usage is measured, not scored). A crashed run
        can't tell which evaluators would have found nothing to judge, so it scores an evaluator that
        another run of the case found N/A too.

        Raises:
            KeyError: `metric` is not a resolved metric.
        """
        self.registry.range(metric)  # raises KeyError for an unknown metric
        return self._agent_failure_values.get(metric)  # usage metrics aren't in it: None

    def _compute_agent_failure_value(self, metric: str) -> float | None:
        """`agent_failure_value` for one evaluator metric, computed once in `__init__`."""
        low, high = self.registry.range(metric)
        if self.agent_failure_score is None:
            return None
        if self.agent_failure_score != "worst":
            score = float(self.agent_failure_score)
            return score if _in_range(score, (low, high)) else None
        wants = {"max" if t.maximize else "min" for t in self._terms if t.metric == metric}
        wants |= {"max" if isinstance(c, AtLeast) else "min" for c in self.constraints if c.metric == metric}
        if wants == {"max"}:
            return low
        if wants == {"min"}:
            return high
        return None

    def objective_value(self, metrics: Mapping[str, float]) -> float | None:
        """The objective from per-metric values, always higher-is-better (`Minimize` negates).

        A lone `Maximize` / `Minimize` gives the metric's value. `WeightedSum` and `Pareto` give the weighted
        mean (each objective's `weight`) of each term's oriented value minus its worst end, over its scale:
        0 to 1 for a bounded term with the default scale, whatever its range or direction, so an N/A term
        dropped on a case counts as scoring like the measured ones instead of moving the level. The worst end
        is fixed by the range, not the baseline, so every case, run, and split is on one scale. Applied to
        measured values (per case or per run, measured means, the test report) and, by the ranking, to each
        goal metric's estimate from a fit, never to a fitted scalar.
        None when the values leave it unmeasured.
        """
        # A missing evaluator term drops, its weight spread over the measured evaluator terms; a missing usage
        # term leaves the value unmeasured, since rescaling a cost up would swamp the scores.
        if any(t.metric not in metrics for t in self._usage_terms):
            return None
        # Weights are > 0, so 0 means no evaluator term is measured.
        measured_weight = sum(t.weight for t in self._evaluator_terms if t.metric in metrics)
        if self._evaluator_terms and measured_weight == 0:
            return None
        rescale = self._evaluator_weight / measured_weight if measured_weight else 1.0
        total = 0.0
        for term in self._terms:
            if term.metric not in metrics:
                continue
            weight = term.weight if term.is_usage else term.weight * rescale
            total += weight * (_oriented(term, metrics[term.metric]) - term.offset) / term.scale
        return total / self._total_weight

    def terms(self, metrics: Mapping[str, float]) -> tuple[float, ...] | None:
        """Each objective term's value, oriented higher-is-better, in objective order; None when one is unmeasured.

        What Pareto dominance compares; one value for a lone objective.
        """
        if any(t.metric not in metrics for t in self._terms):
            return None
        return tuple(_oriented(t, metrics[t.metric]) for t in self._terms)

    def unmet_constraints(self, metrics_by_case: Mapping[str, Mapping[str, float]]) -> list[str]:
        """One sentence per constraint not met on measured means; a constraint never measured counts as unmet.

        Sentences, as `CandidateLabel.unmet_constraints`; the per-constraint margins are `Assessment.violations`.

        `metrics_by_case` maps case name -> metric name -> value, as `CandidateScores.metrics_by_case`.
        """
        sentences = []
        for constraint in self.constraints:
            values = [case[constraint.metric] for case in metrics_by_case.values() if constraint.metric in case]
            if not values:
                sentences.append(f"{constraint}: never measured")
                continue
            mean = statistics.fmean(values)
            excess = mean - constraint.bound if isinstance(constraint, AtMost) else constraint.bound - mean
            # A mean equal to the bound up to float noise (0.1 + 0.2 against 0.3) is met.
            if excess > 0 and not near(mean, constraint.bound):
                sentences.append(f"{constraint}: mean {mean:.4g} violates by {excess:.4g}")
        return sentences

    def _resolve_objective(self, objective: Objective | None) -> Objective:
        """The objective as the run uses it: the default filled in, metric names resolved, and checked.

        None becomes the mean of every evaluator's score; a `WeightedSum` or `Pareto` of one objective becomes that
        objective; each name becomes the registry's (`"Helpfulness"` -> `"HelpfulnessEvaluator"`). Raises
        the `ValueError`s listed in `__init__` (a lone objective with `weight=` / `scale=`, a metric twice
        after resolution, an unbounded metric without `scale=`).
        """
        defaulted = objective is None
        if objective is None:
            if not self.registry.evaluators:
                raise ValueError("nothing to optimize: give an objective, or evaluators to maximize")
            objective = WeightedSum(*(Maximize(name) for name in self.registry.evaluators))
        if isinstance(objective, _Combination) and len(objective.objectives) == 1:
            objective = objective.objectives[0]
        if isinstance(objective, (Maximize, Minimize)):
            if objective.weight != 1.0 or objective.scale is not None:
                raise ValueError(
                    f"{objective}: weight= and scale= are read only when two or more objectives are combined in a "
                    "WeightedSum or Pareto, so a single objective would ignore them; drop them"
                )
            return type(objective)(self.registry.resolve(objective.metric))
        if isinstance(objective, _Combination):
            resolved = [
                type(o)(self.registry.resolve(o.metric), weight=o.weight, scale=o.scale) for o in objective.objectives
            ]
            combined = type(objective)(*resolved)  # repeats the same-metric check on resolved names
            name = type(objective).__name__
            for o in resolved:
                if o.scale is None and _width(self.registry.range(o.metric)) is None:
                    if defaulted:
                        raise ValueError(
                            f"{o.metric} has no bounded range, so the default objective (the mean of every "
                            "evaluator's score) can't include it; give objective=, e.g. WeightedSum(..., "
                            f"Maximize({o.metric!r}, scale=...))"
                        )
                    raise ValueError(
                        f"{o.metric} has no bounded range: give it a scale= (the size of a meaningful "
                        f"change) so it can be combined in the {name}"
                    )
                low, high = self.registry.range(o.metric)
                worst, side = (low, "low") if isinstance(o, Maximize) else (high, "high")
                if worst is None and not self.registry.is_usage(o.metric):
                    # A term is measured from its worst end; with none, dropping it on a case where the evaluator
                    # is N/A would score that case as if the evaluator had done better than its best.
                    raise ValueError(
                        f"{o}: the evaluator {o.metric} has no {side} end in its score_range ({low}, {high}), so "
                        f"it can't be combined in the {name} (a case it finds N/A would score above a perfect "
                        f"one); give the evaluator a score_range with a {side} end"
                    )
            return combined
        raise TypeError(f"objective must be Maximize, Minimize, WeightedSum, or Pareto, got {type(objective).__name__}")

    def _resolve_constraint(self, constraint: Constraint) -> Constraint:
        if not isinstance(constraint, (AtMost, AtLeast)):
            raise TypeError(f"constraints must be AtMost or AtLeast, got {type(constraint).__name__}")
        return type(constraint)(self.registry.resolve(constraint.metric), constraint.bound)

    def _build_terms(self, objective: Objective) -> tuple[_Term, ...]:
        if isinstance(objective, (Maximize, Minimize)):
            # A lone term isn't combined, so its weight and scale never matter.
            return (
                _Term(
                    objective.metric,
                    isinstance(objective, Maximize),
                    1.0,
                    1.0,
                    self.registry.is_usage(objective.metric),
                ),
            )
        terms = []
        for o in objective.objectives:
            low, high = self.registry.range(o.metric)
            scale = o.scale if o.scale is not None else _width((low, high))
            assert scale is not None  # an unbounded metric without scale= was rejected when resolved
            maximize = isinstance(o, Maximize)
            # Anchored at the worst end, so dropping an N/A term and rescaling the others keeps the level; only a
            # usage metric (never N/A) may lack that end.
            worst = low if maximize else (None if high is None else -high)
            offset = 0.0 if worst is None else worst
            terms.append(_Term(o.metric, maximize, o.weight, scale, self.registry.is_usage(o.metric), offset))
        return tuple(terms)


def _case_metrics(
    records: Sequence[TrialRecord], criteria: Criteria
) -> tuple[dict[str, dict[str, list[float | None]]], dict[str, dict[int, dict[str, str]]], dict[str, set[str]]]:
    """Each case's metric values, one per run (None when the run has none), the scores taken out and why, and
    the evaluators each case found not applicable.

    Evaluator metrics: completed runs as measured, `agent_failure` runs as `criteria.agent_failure_value(metric)`
    (left out when None, and applied even to an evaluator another run of the case found N/A), N/A rows
    dropped, an evaluator that raised left out. Usage metrics: completed and `agent_failure` runs as measured
    (a crash's tokens are real cost). `infrastructure_invalid` and `evaluator_invalid` runs are left out.
    An evaluator value outside its range (beyond float noise at an end) is taken out with the reason "give
    the evaluator a score_range = (low, high)", never clipped, and a NaN or infinite evaluator or usage value
    with "is not a finite number"; the trial's saved status is left as is.

    Args:
        records: One candidate's trial records on one split.
        criteria: Supplies each metric's range, kind, and agent-failure value.

    Returns:
        Case -> metric -> per-run values; what was taken out: case -> run index -> metric -> reason (kept as
        `CandidateScores.unavailable`); and case -> the evaluators a completed run of it found N/A (these don't
        count against eligibility).
    """
    registry = criteria.registry
    runs = 1 + max((record.run_index for record in records), default=-1)
    by_case: dict[str, dict[str, list[float | None]]] = {}
    unavailable: dict[str, dict[int, dict[str, str]]] = {}
    not_applicable: dict[str, set[str]] = defaultdict(set)
    for record in records:
        values = by_case.setdefault(
            record.case_name, {metric: [None] * runs for metric in (*registry.evaluators, *USAGE_METRICS)}
        )
        if record.status not in (TrialStatus.COMPLETED, TrialStatus.AGENT_FAILURE):
            continue
        for metric in USAGE_METRICS:
            usage = _usage_value(record.usage, metric)
            if usage is not None and not math.isfinite(usage):
                not_finite = f"{metric}={usage!r} is not a finite number"
                unavailable.setdefault(record.case_name, {}).setdefault(record.run_index, {})[metric] = not_finite
                continue
            values[metric][record.run_index] = usage
        for metric in registry.evaluators:
            if record.status is TrialStatus.AGENT_FAILURE:
                values[metric][record.run_index] = criteria.agent_failure_value(metric)
                continue
            output = record.evaluations.get(metric)
            if output is None:
                continue
            if _not_applicable(record, metric):
                not_applicable[record.case_name].add(metric)
                continue
            low, high = registry.range(metric)
            reason = None
            if not math.isfinite(output.score):
                # NaN passes every range comparison and would poison the means and the ranking.
                reason = f"{metric}={output.score!r} is not a finite number"
            elif not _in_range(output.score, (low, high)):
                reason = (
                    f"{metric}={output.score!r} is outside its range [{low}, {high}]; give the evaluator a "
                    "score_range = (low, high) (None for unbounded)"
                )
            if reason is not None:
                unavailable.setdefault(record.case_name, {}).setdefault(record.run_index, {})[metric] = reason
                continue
            values[metric][record.run_index] = float(output.score)
    return by_case, unavailable, not_applicable


def score(candidate_id: str, records: Sequence[TrialRecord], criteria: Criteria, splits: CaseSplit) -> SplitScores:
    """Group records by split and average per case.

    An `agent_failure` run is measured with `criteria`'s `agent_failure_score`; infrastructure and evaluator
    failures are missing. Consumers take one split: the search (fit, selection, labels, observe,
    Reflection) `.train`, the end phase's pick `.validation`, the test report `.test`.

    Args:
        candidate_id: The candidate.
        records: Its trial records on any splits.
        criteria: Computes each run's objective and supplies `agent_failure_score`.
        splits: The run's split.

    Returns:
        One `CandidateScores` per split scored; validation and test are None when it wasn't scored on them.

    Raises:
        ValueError: A record's case isn't in `splits` under its own `split`.
    """
    by_split: dict[SplitName, list[TrialRecord]] = defaultdict(list)
    for record in records:
        by_split[record.split].append(record)
    return SplitScores(
        train=_split_scores(candidate_id, by_split["train"], criteria, splits.train),
        validation=(
            _split_scores(candidate_id, by_split["validation"], criteria, splits.validation)
            if by_split["validation"]
            else None
        ),
        test=_split_scores(candidate_id, by_split["test"], criteria, splits.test) if by_split["test"] else None,
    )


def measured_metrics(scores: CandidateScores, registry: MetricRegistry) -> dict[str, Measured]:
    """Every metric scored, with its mean and a 95% t-interval clipped to the metric's range.

    The unit is the case (its mean over runs, `scores.metrics_by_case`); for a metric only one case has a
    value for (a single-case split, or the other cases unmeasured), the unit is that case's runs, so it still
    gets an interval. Chosen per metric. A metric with no value is left out. For reporting only
    (`CandidateLabel`, `ValidationReport`, `TestReport`, `CandidateSummary`); the ranking keeps using
    `CandidateScores.objective` and the fit.

    Args:
        scores: One candidate's scores on one split.
        registry: Gives each metric's range, which the interval is clipped to.
    """
    metrics = dict.fromkeys(metric for runs in scores.metrics_by_run.values() for metric in runs)
    by_metric: dict[str, list[float]] = {}
    for metric in metrics:
        case_means = [case[metric] for case in scores.metrics_by_case.values() if metric in case]
        if len(case_means) >= 2:
            by_metric[metric] = case_means
        else:  # at most one case has a value: its runs (the other cases' slots are all None)
            by_metric[metric] = [
                v for runs in scores.metrics_by_run.values() for v in runs.get(metric, []) if v is not None
            ]
    result = {}
    for metric, values in by_metric.items():
        if not values:
            continue
        mean = statistics.fmean(values)
        ci95 = None
        if len(values) >= 2:
            half = float(stats.t.ppf(0.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
            low, high = registry.range(metric)
            ci95 = (
                mean - half if low is None else max(low, mean - half),
                mean + half if high is None else min(high, mean + half),
            )
        result[metric] = Measured(mean=mean, ci95=ci95, n=len(values))
    return result


def _split_scores(
    candidate_id: str, records: Sequence[TrialRecord], criteria: Criteria, cases: Sequence[str]
) -> CandidateScores:
    """One split's `CandidateScores`; `cases` are the split's cases, so an unscored one makes it ineligible.

    Raises:
        ValueError: A record's case isn't one of `cases` (its `split` disagrees with the run's `CaseSplit`), or
            two records share a (case, run): `_case_metrics` would blend them field by field, e.g. one
            attempt's score with another's tokens. The record layer keeps the last per key (the replaced one
            in `previous_attempts`), so a duplicate here is a caller's bug.
    """
    strays = sorted({record.case_name for record in records} - set(cases))
    if strays:
        raise ValueError(
            f"{candidate_id}: records for cases {strays} say split {records[0].split!r}, "
            "but the split doesn't hold them"
        )
    counts = Counter((record.case_name, record.run_index) for record in records)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{candidate_id}: more than one record for (case, run) {duplicates}; pass only the last record per key"
        )
    metrics_by_run, unavailable, not_applicable = _case_metrics(records, criteria)
    objective_by_run: dict[str, list[float | None]] = {}
    objective_by_case: dict[str, float] = {}
    metrics_by_case: dict[str, dict[str, float]] = {}
    for case, values in metrics_by_run.items():
        runs = len(values[USAGE_METRICS[0]])  # every metric's list has one slot per run
        per_run = [
            criteria.objective_value({m: value for m, v in values.items() if (value := v[run]) is not None})
            for run in range(runs)
        ]
        objective_by_run[case] = per_run
        counted = [v for v in per_run if v is not None]
        if counted:
            objective_by_case[case] = statistics.fmean(counted)
        metrics_by_case[case] = {
            metric: statistics.fmean(counted_values)
            for metric, run_values in values.items()
            if (counted_values := [v for v in run_values if v is not None])
        }
    goal_means = {
        metric: statistics.fmean(case_values)
        for metric in criteria.goal_metrics
        if (case_values := [case[metric] for case in metrics_by_case.values() if metric in case])
    }
    objective = criteria.objective_value(goal_means)
    # Eligible candidates are fitted and ranked by `objective` (and under Pareto by every term), so it must be
    # measured: a goal evaluator N/A on every case passes the per-case check below but can leave it unmeasured.
    eligible = (
        objective is not None
        and (not criteria.is_pareto or criteria.terms(goal_means) is not None)
        and bool(cases)
        and all(
            metric in metrics_by_case.get(case, {}) or metric in not_applicable.get(case, set())
            for case in cases
            for metric in criteria.goal_metrics
        )
    )
    return CandidateScores(
        candidate_id=candidate_id,
        objective_by_case=objective_by_case,
        metrics_by_case=metrics_by_case,
        unavailable=unavailable,
        objective_by_run=objective_by_run,
        metrics_by_run=metrics_by_run,
        objective=objective,
        is_eligible=eligible,
        failures=sum(1 for record in records if record.status is TrialStatus.AGENT_FAILURE),
    )


def _not_applicable(record: TrialRecord, evaluator: str) -> bool:
    """Whether the evaluator found nothing to judge in this trial, by the report's own N/A rule."""
    rows = record.details.get(evaluator) or [record.evaluations[evaluator]]
    return not EvaluationReport.is_applicable(rows)


def _usage_value(usage: Usage, metric: str) -> float | None:
    value = getattr(usage, metric)
    return None if value is None else float(value)


def _in_range(value: float, range_: Range) -> bool:
    """Whether `value` lies in the inclusive range, up to float noise at an end; a None end is unbounded.

    `1 - 0.3 - 0.2 - 0.5` (-5.55e-17) is in (0, 1), as `unmet_constraints` treats a mean at its bound.
    """
    low, high = range_
    return (low is None or value >= low or near(value, low)) and (high is None or value <= high or near(value, high))


def _width(range_: Range) -> float | None:
    """A bounded range's width, the default scale; None when unbounded (`MetricRegistry` rejects empty ones)."""
    low, high = range_
    return None if low is None or high is None else high - low


def _oriented(term: _Term, value: float) -> float:
    return value if term.maximize else -value


def _metric_name(target: MetricTarget) -> str:
    """A metric name from a name, an evaluator instance (`get_name()`), or an evaluator class (`get_type_name()`)."""
    if isinstance(target, str):
        return target
    if isinstance(target, Evaluator):
        return target.get_name()
    if isinstance(target, type) and issubclass(target, Evaluator):
        return target.get_type_name()
    raise TypeError(f"a metric is a name, an evaluator, or an evaluator class, got {target!r}")


def _check_terms(kind: str, objectives: Sequence[Maximize | Minimize], *, need_scale: bool = True) -> None:
    """The checks a combination can make alone; `Criteria` repeats them on resolved names and ranges."""
    if not objectives:
        raise TypeError(f"{kind} needs at least one objective")
    for o in objectives:
        if not isinstance(o, (Maximize, Minimize)):
            raise TypeError(f"{kind} takes Maximize and Minimize objectives, got {type(o).__name__}")
    names = [o.metric for o in objectives]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{kind} names {repeated} more than once")
    if need_scale:
        unscaled = [o.metric for o in objectives if o.scale is None and o.metric in USAGE_METRICS]
        if unscaled:
            raise ValueError(
                f"{kind}: usage metrics {unscaled} have no bounded range; give each a scale= (the size of a "
                "meaningful change, e.g. Minimize('tokens', scale=1000))"
            )


def _positive(name: str, value: float) -> float:
    """`value` as a float when it is a finite number > 0 (a `weight=` or `scale=`), else raises.

    Raises:
        TypeError: `value` is not a number (a bool counts as not a number).
        ValueError: `value` is NaN, infinite, or <= 0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return float(value)


def _number(value: float) -> str:
    """`1000.0` -> `"1000"`, `0.5` -> `"0.5"`: the shortest form that reads back as the same float."""
    return str(int(value)) if value.is_integer() else repr(value)
