"""Objectives, constraints, and turning trials into scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ...evaluators.evaluator import Evaluator
from ...types.optimization import CandidateScores, CaseSplit, Measured, SplitScores, TrialRecord
from .metrics import MetricRegistry

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
            ValueError: `weight` or `scale` is not > 0.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """E.g. `"Minimize(tokens, scale=1000)"`; stored in `run.json`."""
        raise NotImplementedError


class Maximize(_Direction):
    """Maximize a metric, e.g. `Maximize("Helpfulness")`."""


class Minimize(_Direction):
    """Minimize a metric, e.g. `Minimize("tokens")`."""


@dataclass(frozen=True, init=False)
class WeightedSum:
    """Several objectives as one number: per case, the weighted mean of each oriented value / scale.

    `WeightedSum(Maximize("Helpfulness"), Minimize("tokens", scale=1000))` makes 1,000 tokens worth
    as much as the whole 0-1 Helpfulness range. Per case, N/A evaluator terms drop and the remaining
    evaluator weights are rescaled to the same total; usage terms are never rescaled. All evaluator
    terms N/A, or any usage term unmeasured, leaves the case unmeasured.
    """

    objectives: tuple[Maximize | Minimize, ...]

    def __init__(self, *objectives: Maximize | Minimize):
        """Combine objectives.

        Args:
            *objectives: At least one `Maximize` or `Minimize`.

        Raises:
            TypeError: No objectives, or one that isn't a `Maximize` / `Minimize`.
            ValueError: The same target string twice, or a usage metric name without `scale=`. `Criteria`
                repeats both checks on resolved names and ranges (an evaluator with an unbounded
                `score_range`, `"Helpfulness"` next to `HelpfulnessEvaluator`).
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """E.g. `"WeightedSum(Maximize(Helpfulness), Minimize(tokens, scale=1000))"`; stored in `run.json`."""
        raise NotImplementedError


@dataclass(frozen=True, init=False)
class Pareto:
    """Several objectives optimized jointly; one champion ships, and the front is reported.

    `Pareto(Maximize("Helpfulness"), Minimize("tokens", scale=1000))`. The ranking (parents, inheritance,
    `Nested`'s round best, the pick) uses the weighted gain, computed as `WeightedSum`'s value with these
    weights: the highest-gain candidate of a pool is never dominated, so this matches taking the front's
    highest-gain member again and again. `TPE` is told each objective's value (a multi-objective study);
    `Thompson` weighs the objectives with fresh random weights per draw. In the end phase the finalists are
    the feasible train front, the champion is the validated front's highest-gain member (the baseline when
    nothing feasible beats it), and every validated front member is tested against the baseline.
    Constraints only filter candidates; they are never front dimensions. A `Pareto` of one objective is
    that objective.
    """

    objectives: tuple[Maximize | Minimize, ...]
    weights: tuple[float, ...] | None

    def __init__(self, *objectives: Maximize | Minimize, weights: Sequence[float] | None = None):
        """Combine objectives jointly.

        Args:
            *objectives: At least one `Maximize` or `Minimize`.
            weights: One per objective, each > 0, for the gain; None means each objective's own `weight`.

        Raises:
            TypeError: No objectives, or one that isn't a `Maximize` / `Minimize`.
            ValueError: `WeightedSum`'s checks; `weights` of another length than `objectives`, or a weight
                <= 0; `weights` next to an objective with a non-default `weight=` (one would be ignored).
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """E.g. `"Pareto(Maximize(Helpfulness), Minimize(tokens, scale=1000))"`, plus `", weights=[2.0, 1.0]"`
        when given; stored in `run.json` and compared on resume."""
        raise NotImplementedError


Objective = Maximize | Minimize | WeightedSum | Pareto


@dataclass(frozen=True, init=False)
class _Bound:
    """A bound on one metric's mean; during the search it only ranks, at the final pick it decides."""

    metric: str
    bound: float

    def __init__(self, target: MetricTarget, bound: float, /):
        """Name the metric as in `Maximize`.

        Args:
            target: A metric name, an evaluator class, or an evaluator instance.
            bound: The limit on the mean.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """E.g. `"tokens <= 4000"`; stored in `run.json`."""
        raise NotImplementedError


class AtMost(_Bound):
    """The metric's mean must not exceed `bound`, e.g. `AtMost("tokens", 4000)`."""


class AtLeast(_Bound):
    """The metric's mean must not fall below `bound`, e.g. `AtLeast("Harmfulness", 0.95)`."""


Constraint = AtMost | AtLeast


class Criteria:
    """The objective plus the constraints; names resolve through the registry at construction.

    Attributes:
        objective: What to optimize, the default filled in, every metric name resolved.
        constraints: The constraints, metric names resolved.
        registry: The run's metric registry.
        crash_score: What an `agent_failure` run's evaluator metrics count as.
        feasibility: The P(met) threshold a constraint needs to count as met (not a fraction of cases);
            read by the ranking, the candidate label, and the final pick.
        goal_metrics: The metrics the score model fits, one model each: the objective's metrics, then the
            constraints' metrics, each once, in that order.
        is_pareto: The objective is a `Pareto` of two or more objectives, so the end phase works from the front.
    """

    objective: Objective
    constraints: tuple[Constraint, ...]
    registry: MetricRegistry
    crash_score: Literal["worst"] | float | None
    feasibility: float
    goal_metrics: tuple[str, ...]
    is_pareto: bool

    def __init__(
        self,
        objective: Objective | None,
        constraints: Sequence[Constraint],
        registry: MetricRegistry,
        *,
        crash_score: Literal["worst"] | float | None = "worst",
        feasibility: float = 0.9,
    ):
        """`objective=None` means maximize the mean of every evaluator's score.

        `crash_score` is what an `agent_failure` run's evaluator metrics count
        as: "worst" = each evaluator metric's worst end for the goals that use it (0 for a 0-1 score
        maximized), a number = that score, None = missing data. Infrastructure and verifier failures are
        always missing. Usage metrics have no worst end, so a crashed run's usage counts as measured.

        Args:
            objective: What to optimize; None means the mean of every evaluator's score.
            constraints: Bounds every shipped candidate must meet.
            registry: Resolves the metric names and gives their ranges.
            crash_score: What an `agent_failure` run's evaluator metrics count as.
            feasibility: The P(met) threshold a constraint needs (`Optimizer(feasibility=)`), 0 < x < 1.

        Raises:
            ValueError: A metric name is unknown or ambiguous; inside a `WeightedSum` or `Pareto`, the same
                metric twice after resolution, or a metric with an unbounded range and no `scale=`; a lone
                `Maximize` / `Minimize` with a non-default `weight` or `scale` (read only inside a
                `WeightedSum` or `Pareto`, so it would be silently ignored); `feasibility` outside (0, 1).

        A `Pareto` of one objective is unwrapped to that objective.
        """
        raise NotImplementedError

    def crash_value(self, metric: str) -> float | None:
        """What an `agent_failure` run counts as for one evaluator metric.

        Under `crash_score="worst"`: the range's low end when every goal on the metric is a `Maximize` or
        `AtLeast`, the high end when every one is a `Minimize` or `AtMost`; None when that end is unbounded,
        the goals disagree, or no goal uses the metric. A number gives that number; None gives None.
        Usage metrics always give None (a crash's usage is measured, not scored).
        """
        raise NotImplementedError

    def objective_value(self, metrics: Mapping[str, float]) -> float | None:
        """The objective from per-metric values, always higher-is-better (`Minimize` negates).

        A lone `Maximize` / `Minimize` gives the metric's value; `WeightedSum` and `Pareto` the weighted mean
        (a `Pareto`'s `weights=` when given), with no baseline subtraction. Applied to measured values (per
        case or per run, measured means, the test report) and, by the ranking, to each goal metric's estimate
        from a fit, never to a fitted scalar.
        None when the values leave it unmeasured.
        """
        raise NotImplementedError

    def terms(self, metrics: Mapping[str, float]) -> tuple[float, ...] | None:
        """Each objective term's value, oriented higher-is-better, in objective order; None when one is unmeasured.

        What Pareto dominance compares; one value for a lone objective.
        """
        raise NotImplementedError

    def unmet(self, metrics_by_case: Mapping[str, Mapping[str, float]]) -> list[str]:
        """One sentence per constraint not met on measured means; a constraint never measured counts as unmet.

        Sentences, as `CandidateLabel.unmet`; the per-constraint margins are `Assessment.violations`.

        `metrics_by_case` maps case name -> metric name -> value, as `CandidateScores.metrics_by_case`.
        """
        raise NotImplementedError


def _case_metrics(
    records: Sequence[TrialRecord], criteria: Criteria
) -> tuple[dict[str, dict[str, list[float | None]]], dict[str, dict[int, dict[str, str]]]]:
    """Per-run metric values per case, and the values taken out.

    Evaluator metrics: completed runs as measured, `agent_failure` runs as `criteria.crash_value(metric)`
    (left out when None), N/A rows dropped. Usage metrics: completed and `agent_failure` runs as measured
    (a crash's tokens are real cost). `infrastructure_invalid` and `verifier_invalid` runs are left out.
    An evaluator value outside its range is taken out with the reason "give the evaluator a
    score_range = (low, high)", never clipped; the trial's saved status is left as is.

    Args:
        records: One candidate's trial records on one split.
        criteria: Supplies each metric's range, kind, and crash value.

    Returns:
        Case -> metric -> per-run values, and what was taken out: case -> run index -> metric -> reason
        (kept as `CandidateScores.unavailable`).
    """
    raise NotImplementedError


def score(candidate_id: str, records: Sequence[TrialRecord], criteria: Criteria, splits: CaseSplit) -> SplitScores:
    """Group records by split and average per case.

    An `agent_failure` run is measured with `criteria`'s `crash_score`; infrastructure and verifier
    failures are missing. Consumers take one split: the search (fit, selection, labels, observe,
    Reflection) `.train`, the end phase's pick `.validation`, the test report `.test`.

    Args:
        candidate_id: The candidate.
        records: Its trial records on any splits.
        criteria: Computes each run's objective and supplies `crash_score`.
        splits: The run's split.

    Returns:
        One `CandidateScores` per split scored; validation and test are None when it wasn't scored on them.
    """
    raise NotImplementedError


def measured(scores: CandidateScores) -> dict[str, Measured]:
    """Every metric in `scores.metrics_by_case`, with a 90% t-interval over the case means.

    A metric with no value on any case is left out. For reporting only (`CandidateLabel`, `ValidationReport`,
    `TestReport`); the ranking keeps using `CandidateScores.objective` and the fit.
    """
    raise NotImplementedError
