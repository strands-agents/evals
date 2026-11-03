"""Objectives and per-split scores by measured means, plus the report intervals; there is no score model yet."""

import importlib.util
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ...optimization.scoring.metrics import USAGE_METRICS
from ...types.evaluation_report import EvaluationReport
from ...types.optimization import CandidateScores, Measured, PairedComparison, Range, TrialRecord, TrialStatus, Usage

CrashScore = Literal["worst"] | float | None
Direction = Literal["Maximize", "Minimize"]

REPORT_LEVEL = 0.90
MEASURED_USAGE_METRICS = ("seconds",)
INSTALL_HINT = "pip install 'strands-agents-evals[optimization]'"
_COUNTED = (TrialStatus.COMPLETED, TrialStatus.AGENT_FAILURE)


def require_optimization_extra() -> None:
    """Fail fast, naming the install command, when the `optimization` extra is missing."""
    if importlib.util.find_spec("scipy") is None:
        raise RuntimeError(f"optimization needs the optimization extra: {INSTALL_HINT}")


def zero_usage() -> Usage:
    """No usage: every count 0, cache tokens unmeasured, no models."""
    return Usage(
        tokens=0,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=None,
        cache_write_tokens=None,
        model_calls=0,
        seconds=0.0,
        models=[],
    )


def add_usage(left: Usage, right: Usage) -> Usage:
    """Field by field; cache tokens stay None only when both sides are None; models join in first-use order."""

    def cache(a: int | None, b: int | None) -> int | None:
        return None if a is None and b is None else (a or 0) + (b or 0)

    return Usage(
        tokens=left.tokens + right.tokens,
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cache_read_tokens=cache(left.cache_read_tokens, right.cache_read_tokens),
        cache_write_tokens=cache(left.cache_write_tokens, right.cache_write_tokens),
        model_calls=left.model_calls + right.model_calls,
        seconds=left.seconds + right.seconds,
        models=list(dict.fromkeys([*left.models, *right.models])),
    )


@dataclass(frozen=True)
class Term:
    """One `Maximize` or `Minimize` term; `weight` and `scale` count only inside a `WeightedSum`."""

    direction: Direction
    metric: str
    weight: float = 1.0
    scale: float | None = None

    def __str__(self) -> str:
        extras = [f"weight={self.weight:g}"] if self.weight != 1.0 else []
        extras += [f"scale={self.scale:g}"] if self.scale is not None else []
        return f"{self.direction}({', '.join([self.metric, *extras])})"


@dataclass(frozen=True)
class Objective:
    """A lone term, or a `WeightedSum` of terms when `weighted` is True."""

    terms: tuple[Term, ...]
    weighted: bool

    def __str__(self) -> str:
        if not self.weighted:
            return str(self.terms[0])
        return f"WeightedSum({', '.join(str(term) for term in self.terms)})"


class MetricRegistry:
    """Evaluator names with their ranges, plus the usage metrics that project mode measures."""

    def __init__(self, evaluators: Mapping[str, Range]):
        """Register the run's evaluators by `get_name()`.

        Raises:
            ValueError: An evaluator has the name of a usage metric.
        """
        clash = sorted(set(evaluators) & set(USAGE_METRICS))
        if clash:
            raise ValueError(f"evaluator names clash with usage metrics: {', '.join(clash)}")
        self.evaluators: tuple[str, ...] = tuple(evaluators)
        self._ranges = dict(evaluators)

    def is_usage(self, metric: str) -> bool:
        """Whether the metric is a usage metric."""
        return metric in USAGE_METRICS

    def range(self, metric: str) -> Range:
        """The metric's range; usage metrics are `(0, None)`."""
        return (0.0, None) if self.is_usage(metric) else self._ranges[metric]

    def resolve(self, name: str) -> str:
        """An exact evaluator or usage metric name, then `"Output"` -> `"OutputEvaluator"`.

        Raises:
            ValueError: The name is unknown, or a usage metric that project mode cannot measure yet.
        """
        if name in self._ranges or name in MEASURED_USAGE_METRICS:
            return name
        if name in USAGE_METRICS:
            raise ValueError(f"metric {name!r} is not measured in project mode yet: no chat spans are captured")
        if f"{name}Evaluator" in self._ranges:
            return f"{name}Evaluator"
        known = ", ".join([*self.evaluators, *MEASURED_USAGE_METRICS])
        raise ValueError(f"unknown metric {name!r}; known metrics: {known}")


def parse_term(spec: str) -> Term:
    """Parse `maximize:METRIC[:weight=W][:scale=S]` or the same with `minimize`.

    Raises:
        ValueError: The spec does not match that form, or a weight or scale is not a number > 0.
    """
    direction, _, rest = spec.partition(":")
    metric, *options = rest.split(":") if rest else [""]
    if direction.lower() not in ("maximize", "minimize") or not metric:
        raise ValueError(f"--objective must be maximize:METRIC or minimize:METRIC, got {spec!r}")
    values: dict[str, float] = {}
    for option in options:
        key, _, raw = option.partition("=")
        if key not in ("weight", "scale") or key in values:
            raise ValueError(f"--objective option must be weight=N or scale=N, once each: {spec!r}")
        try:
            values[key] = float(raw)
        except ValueError as exc:
            raise ValueError(f"--objective {key} must be a number: {spec!r}") from exc
        if not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError(f"--objective {key} must be > 0: {spec!r}")
    return Term(
        direction="Maximize" if direction.lower() == "maximize" else "Minimize",
        metric=metric,
        weight=values.get("weight", 1.0),
        scale=values.get("scale"),
    )


def build_objective(specs: Sequence[str], registry: MetricRegistry) -> Objective:
    """The planned objective; no specs maximizes the mean of every evaluator's score.

    Raises:
        ValueError: An unknown metric, a metric used twice, a lone term with a weight or scale, or a
            `WeightedSum` term on an unbounded metric without `scale=`.
    """
    parsed = [parse_term(spec) for spec in specs] or [Term("Maximize", name) for name in registry.evaluators]
    terms = tuple(Term(term.direction, registry.resolve(term.metric), term.weight, term.scale) for term in parsed)
    metrics = [term.metric for term in terms]
    if len(set(metrics)) != len(metrics):
        raise ValueError(f"each metric may appear once in the objective: {', '.join(metrics)}")
    weighted = len(terms) > 1
    for term in terms:
        if not weighted and (term.weight != 1.0 or term.scale is not None):
            raise ValueError("weight= and scale= apply only when the objective has two or more terms")
        if weighted and term.scale is None and registry.range(term.metric)[1] is None:
            raise ValueError(f"{term.metric} is unbounded: give it scale= inside a weighted objective")
    return Objective(terms=terms, weighted=weighted)


@dataclass(frozen=True)
class Criteria:
    """The objective plus how a crashed run's evaluator metrics count."""

    objective: Objective
    registry: MetricRegistry
    crash_score: CrashScore = "worst"

    @property
    def goal_metrics(self) -> tuple[str, ...]:
        """The objective's metrics, in objective order."""
        return tuple(term.metric for term in self.objective.terms)

    @property
    def reported_metrics(self) -> tuple[str, ...]:
        """Every metric that project mode measures: the goals, the other evaluators, then `seconds`."""
        return tuple(dict.fromkeys([*self.goal_metrics, *self.registry.evaluators, *MEASURED_USAGE_METRICS]))

    def crash_value(self, metric: str) -> float | None:
        """What an `agent_failure` run counts as for one evaluator metric; None means missing."""
        if self.registry.is_usage(metric) or self.crash_score is None:
            return None
        if not isinstance(self.crash_score, str):
            return float(self.crash_score)
        directions = {term.direction for term in self.objective.terms if term.metric == metric}
        low, high = self.registry.range(metric)
        if directions == {"Maximize"}:
            return low
        return high if directions == {"Minimize"} else None

    def objective_value(self, metrics: Mapping[str, float]) -> float | None:
        """The objective from per-metric values, higher is better; None when the values leave it unmeasured."""
        terms = self.objective.terms
        if not self.objective.weighted:
            value = metrics.get(terms[0].metric)
            return None if value is None else _oriented(terms[0], value)
        usage = [term for term in terms if self.registry.is_usage(term.metric)]
        evaluator_terms = [term for term in terms if not self.registry.is_usage(term.metric)]
        present = [term for term in evaluator_terms if term.metric in metrics]
        if any(term.metric not in metrics for term in usage) or (evaluator_terms and not present):
            return None
        rescale = sum(t.weight for t in evaluator_terms) / sum(t.weight for t in present) if present else 0.0
        total = sum(t.weight * rescale * _oriented(t, metrics[t.metric]) / self._scale(t) for t in present)
        total += sum(t.weight * _oriented(t, metrics[t.metric]) / self._scale(t) for t in usage)
        return total / sum(term.weight for term in terms)

    def _scale(self, term: Term) -> float:
        if term.scale is not None:
            return term.scale
        low, high = self.registry.range(term.metric)
        return (high - low) if low is not None and high is not None and high > low else 1.0


def run_metrics(record: TrialRecord, criteria: Criteria) -> tuple[dict[str, float], dict[str, str]] | None:
    """One run's metric values and the values taken out; None when the run does not count."""
    if record.status not in _COUNTED:
        return None
    values: dict[str, float] = {}
    taken_out: dict[str, str] = {}
    for metric in criteria.reported_metrics:
        if criteria.registry.is_usage(metric):
            values[metric] = record.usage.seconds
            continue
        if record.status is TrialStatus.AGENT_FAILURE:
            crash = criteria.crash_value(metric)
            if crash is not None:
                values[metric] = crash
            continue
        output = record.evaluations.get(metric)
        if output is None or not EvaluationReport.is_applicable(record.details.get(metric, [])):
            continue
        low, high = criteria.registry.range(metric)
        if (low is not None and output.score < low) or (high is not None and output.score > high):
            taken_out[metric] = "outside the evaluator range: give the evaluator a score_range = (low, high)"
            continue
        values[metric] = output.score
    return values, taken_out


def score_split(
    candidate_id: str, records: Sequence[TrialRecord], case_names: Sequence[str], criteria: Criteria
) -> CandidateScores:
    """One candidate's `CandidateScores` on one split, by measured means."""
    by_case: dict[str, list[TrialRecord]] = {name: [] for name in case_names}
    for record in records:
        by_case.setdefault(record.case_name, []).append(record)
    objective_by_run: dict[str, list[float | None]] = {}
    metrics_by_run: dict[str, dict[str, list[float | None]]] = {}
    unavailable: dict[str, dict[int, dict[str, str]]] = {}
    for name, runs in by_case.items():
        runs.sort(key=lambda record: record.run_index)
        measured_runs = [run_metrics(record, criteria) for record in runs]
        objective_by_run[name] = [None if m is None else criteria.objective_value(m[0]) for m in measured_runs]
        metrics_by_run[name] = {
            metric: [None if m is None else m[0].get(metric) for m in measured_runs]
            for metric in criteria.reported_metrics
        }
        removed = {run.run_index: m[1] for run, m in zip(runs, measured_runs, strict=True) if m is not None and m[1]}
        if removed:
            unavailable[name] = removed
    objective_by_case = {name: mean for name, runs in objective_by_run.items() if (mean := _mean(runs)) is not None}
    metrics_by_case = {
        name: {metric: mean for metric, runs in metrics.items() if (mean := _mean(runs)) is not None}
        for name, metrics in metrics_by_run.items()
    }
    metric_means = {
        metric: mean
        for metric in criteria.goal_metrics
        if (mean := _mean([case.get(metric) for case in metrics_by_case.values()])) is not None
    }
    return CandidateScores(
        candidate_id=candidate_id,
        objective_by_case=objective_by_case,
        metrics_by_case=metrics_by_case,
        unavailable=unavailable,
        objective_by_run=objective_by_run,
        metrics_by_run=metrics_by_run,
        objective=criteria.objective_value(metric_means) if metric_means else None,
        eligible=all(name in objective_by_case for name in case_names),
        failures=sum(record.status is TrialStatus.AGENT_FAILURE for record in records),
    )


def measured(scores: CandidateScores) -> dict[str, Measured]:
    """Every metric with a value, as the mean of case means with a 90% t-interval over the cases."""
    metrics = dict.fromkeys(metric for case in scores.metrics_by_case.values() for metric in case)
    result = {}
    for metric in metrics:
        values = [case[metric] for case in scores.metrics_by_case.values() if metric in case]
        result[metric] = Measured(mean=sum(values) / len(values), ci90=_interval(values), n=len(values))
    return result


def paired(candidate: Mapping[str, float], reference: Mapping[str, float]) -> PairedComparison:
    """A 90% paired t-interval of the mean per-case difference (candidate - reference) over shared cases."""
    shared = [name for name in candidate if name in reference]
    differences = [candidate[name] - reference[name] for name in shared]
    if not differences:
        return PairedComparison(mean_diff=0.0, ci_low=0.0, ci_high=0.0, n_cases=0)
    mean_diff = sum(differences) / len(differences)
    low, high = _interval(differences) or (mean_diff, mean_diff)
    return PairedComparison(mean_diff=mean_diff, ci_low=low, ci_high=high, n_cases=len(differences))


def paired_ratio(
    candidate: Mapping[str, float], reference: Mapping[str, float], *, can_be_negative: bool
) -> PairedComparison | None:
    """The ratio of means with a delta-method 90% interval; None when the reference mean is 0 or values can be < 0."""
    shared = [name for name in candidate if name in reference]
    if can_be_negative or not shared:
        return None
    c_values, r_values = [candidate[name] for name in shared], [reference[name] for name in shared]
    c_mean, r_mean, count = sum(c_values) / len(shared), sum(r_values) / len(shared), len(shared)
    if r_mean == 0:
        return None
    ratio = c_mean / r_mean
    if count < 2:
        return PairedComparison(mean_diff=ratio, ci_low=ratio, ci_high=ratio, n_cases=count)
    c_var, r_var = _variance(c_values, c_mean), _variance(r_values, r_mean)
    covariance = sum((c - c_mean) * (r - r_mean) for c, r in zip(c_values, r_values, strict=True)) / (count - 1)
    variance = (c_var / r_mean**2 - 2 * c_mean * covariance / r_mean**3 + c_mean**2 * r_var / r_mean**4) / count
    half = _t_quantile(count - 1) * math.sqrt(max(variance, 0.0))
    return PairedComparison(mean_diff=ratio, ci_low=ratio - half, ci_high=ratio + half, n_cases=count)


def _oriented(term: Term, value: float) -> float:
    return value if term.direction == "Maximize" else -value


def _mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _variance(values: Sequence[float], mean: float) -> float:
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _interval(values: Sequence[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    half = _t_quantile(len(values) - 1) * math.sqrt(_variance(values, mean) / len(values))
    return mean - half, mean + half


def _t_quantile(degrees_of_freedom: int) -> float:
    # optional extra: optimization (scipy)
    from scipy import stats

    return float(stats.t.ppf(0.5 + REPORT_LEVEL / 2, degrees_of_freedom))
