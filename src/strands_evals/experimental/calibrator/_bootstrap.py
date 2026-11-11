"""Hierarchical bootstrap diagnostics for evaluator calibration."""

import copy
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from ._calibration import (
    _EPSILON,
    _CalibrationDataset,
    _estimate_statistics,
    _mean,
)
from .algorithm import (
    WEIGHTING_METHODS,
    VarianceSource,
    WeightingMethod,
    _compute_weight_vector,
)

_VARIANCE_SOURCES: tuple[VarianceSource, ...] = ("judge", "agent", "case")


@dataclass(frozen=True, kw_only=True)
class BootstrapSummary:
    """Percentile interval and replicate counts for one bootstrap quantity."""

    mean: float | None
    lower_2_5: float | None
    median: float | None
    upper_97_5: float | None
    defined_replicates: int
    undefined_replicates: int

    def to_dict(self) -> dict[str, float | int | None]:
        """Return a JSON-compatible summary."""
        return {
            "mean": self.mean,
            "lower_2_5": self.lower_2_5,
            "median": self.median,
            "upper_97_5": self.upper_97_5,
            "defined_replicates": self.defined_replicates,
            "undefined_replicates": self.undefined_replicates,
        }


@dataclass(frozen=True, kw_only=True)
class BootstrapDiagnostics:
    """Summarized hierarchical bootstrap output."""

    replicates: int
    seed: int
    repeatability_intervals: Mapping[str, BootstrapSummary]
    variance_component_intervals: Mapping[str, Mapping[VarianceSource, BootstrapSummary]]
    candidate_weight_intervals: Mapping[WeightingMethod, Mapping[str, BootstrapSummary]]
    composite_score_intervals: Mapping[WeightingMethod, BootstrapSummary]

    def __post_init__(self) -> None:
        """Copy interval mappings into immutable views."""
        object.__setattr__(
            self,
            "repeatability_intervals",
            MappingProxyType(dict(self.repeatability_intervals)),
        )
        object.__setattr__(
            self,
            "variance_component_intervals",
            MappingProxyType(
                {
                    evaluator_name: MappingProxyType(dict(intervals))
                    for evaluator_name, intervals in self.variance_component_intervals.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "candidate_weight_intervals",
            MappingProxyType(
                {
                    method: MappingProxyType(dict(intervals))
                    for method, intervals in self.candidate_weight_intervals.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "composite_score_intervals",
            MappingProxyType(dict(self.composite_score_intervals)),
        )

    def __deepcopy__(self, memo: dict[int, object]) -> "BootstrapDiagnostics":
        """Return this immutable value for deep-copy operations."""
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Rebuild this value from pickle-compatible mappings."""
        return (
            _restore_bootstrap_diagnostics,
            (
                self.replicates,
                self.seed,
                dict(self.repeatability_intervals),
                {
                    evaluator_name: dict(intervals)
                    for evaluator_name, intervals in self.variance_component_intervals.items()
                },
                {method: dict(intervals) for method, intervals in self.candidate_weight_intervals.items()},
                dict(self.composite_score_intervals),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible bootstrap diagnostics."""
        return {
            "replicates": self.replicates,
            "seed": self.seed,
            "repeatability_intervals": {
                name: summary.to_dict() for name, summary in self.repeatability_intervals.items()
            },
            "variance_component_intervals": {
                name: {source: summary.to_dict() for source, summary in intervals.items()}
                for name, intervals in self.variance_component_intervals.items()
            },
            "candidate_weight_intervals": {
                method: {name: summary.to_dict() for name, summary in intervals.items()}
                for method, intervals in self.candidate_weight_intervals.items()
            },
            "composite_score_intervals": {
                method: summary.to_dict() for method, summary in self.composite_score_intervals.items()
            },
        }


def _restore_bootstrap_diagnostics(
    replicates: int,
    seed: int,
    repeatability_intervals: dict[str, BootstrapSummary],
    variance_component_intervals: dict[str, dict[VarianceSource, BootstrapSummary]],
    candidate_weight_intervals: dict[WeightingMethod, dict[str, BootstrapSummary]],
    composite_score_intervals: dict[WeightingMethod, BootstrapSummary],
) -> BootstrapDiagnostics:
    """Rebuild immutable diagnostics from pickle-compatible mappings."""
    return BootstrapDiagnostics(
        replicates=replicates,
        seed=seed,
        repeatability_intervals=repeatability_intervals,
        variance_component_intervals=variance_component_intervals,
        candidate_weight_intervals=candidate_weight_intervals,
        composite_score_intervals=composite_score_intervals,
    )


def compute_candidate_weights(
    dataset: _CalibrationDataset,
    step_sizes: dict[str, float],
    repeatabilities: dict[str, float],
) -> dict[WeightingMethod, dict[str, float] | None]:
    """Compute every diagnostic candidate vector for one fitted dataset."""
    return {
        method: _compute_weight_vector(
            method,
            dataset.evaluator_names,
            repeatabilities,
            step_sizes,
        )
        for method in WEIGHTING_METHODS
    }


def run_bootstrap(
    dataset: _CalibrationDataset,
    step_sizes: dict[str, float],
    replicates: int,
    seed: int,
) -> BootstrapDiagnostics:
    """Run a conservative case-run-replay bootstrap and summarize it.

    Each replicate resamples cases, agent runs within each selected case, and
    judge replays within each selected run so uncertainty from every measured
    level is reflected in the intervals.
    """
    rng = random.Random(seed)
    repeatability_distributions: dict[str, list[float | None]] = {name: [] for name in dataset.evaluator_names}
    variance_distributions: dict[str, dict[VarianceSource, list[float]]] = {
        name: {source: [] for source in _VARIANCE_SOURCES} for name in dataset.evaluator_names
    }
    weight_distributions: dict[WeightingMethod, dict[str, list[float | None]]] = {
        method: {name: [] for name in dataset.evaluator_names} for method in WEIGHTING_METHODS
    }
    score_distributions: dict[WeightingMethod, list[float | None]] = {method: [] for method in WEIGHTING_METHODS}

    for _ in range(replicates):
        sampled = _bootstrap_sample(dataset, rng)
        sampled_repeatabilities: dict[str, float] = {}
        for evaluator_name in dataset.evaluator_names:
            statistics = _estimate_statistics(sampled, evaluator_name)
            sampled_repeatabilities[evaluator_name] = statistics.repeatability
            has_signal = any(value > _EPSILON for value in statistics.variance_components.values())
            repeatability_distributions[evaluator_name].append(statistics.repeatability if has_signal else None)
            for source in _VARIANCE_SOURCES:
                variance_distributions[evaluator_name][source].append(statistics.variance_components[source])

        evaluator_scores = sampled.evaluator_scores()
        for method in WEIGHTING_METHODS:
            weights = _compute_weight_vector(
                method,
                dataset.evaluator_names,
                sampled_repeatabilities,
                step_sizes,
            )
            for evaluator_name in dataset.evaluator_names:
                weight_distributions[method][evaluator_name].append(
                    None if weights is None else weights[evaluator_name]
                )
            score_distributions[method].append(None if weights is None else _weighted_score(evaluator_scores, weights))

    return BootstrapDiagnostics(
        replicates=replicates,
        seed=seed,
        repeatability_intervals={
            name: _summarize_distribution(values) for name, values in repeatability_distributions.items()
        },
        variance_component_intervals={
            name: {
                source: _summarize_distribution(variance_distributions[name][source]) for source in _VARIANCE_SOURCES
            }
            for name in dataset.evaluator_names
        },
        candidate_weight_intervals={
            method: {name: _summarize_distribution(values) for name, values in weight_distributions[method].items()}
            for method in WEIGHTING_METHODS
        },
        composite_score_intervals={
            method: _summarize_distribution(values) for method, values in score_distributions.items()
        },
    )


def validate_bootstrap_options(replicates: int, seed: int) -> None:
    """Validate hierarchical bootstrap configuration."""
    if not isinstance(replicates, int) or isinstance(replicates, bool):
        raise ValueError(f"bootstrap_replicates must be an integer, got {replicates!r}")
    if replicates < 0 or replicates == 1:
        raise ValueError(f"bootstrap_replicates must be 0 or >= 2, got {replicates}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"bootstrap_seed must be an integer, got {seed!r}")


def _bootstrap_sample(
    dataset: _CalibrationDataset,
    rng: random.Random,
) -> _CalibrationDataset:
    """Resample cases, runs within cases, and replays within fixed outputs."""
    sampled: list[list[dict[str, dict[str, float]]]] = [
        [{name: {} for name in dataset.evaluator_names} for _ in range(dataset.replay_count)]
        for _ in range(dataset.run_count)
    ]
    sampled_case_names: list[str] = []
    sampled_case_signatures: list[tuple[str, str]] = []
    signature_by_case = dict(dataset.case_signatures)

    for case_slot in range(len(dataset.case_names)):
        source_case = rng.choice(dataset.case_names)
        sampled_case = f"{source_case}#bootstrap-case-{case_slot}"
        sampled_case_names.append(sampled_case)
        sampled_case_signatures.append((sampled_case, signature_by_case[source_case]))
        for run_slot in range(dataset.run_count):
            source_run = rng.randrange(dataset.run_count)
            for replay_slot in range(dataset.replay_count):
                source_replay = rng.randrange(dataset.replay_count)
                source_scores = dataset.parsed[source_run][source_replay]
                for evaluator_name in dataset.evaluator_names:
                    sampled[run_slot][replay_slot][evaluator_name][sampled_case] = source_scores[evaluator_name][
                        source_case
                    ]

    return _CalibrationDataset(
        parsed=sampled,
        evaluator_names=dataset.evaluator_names,
        case_names=tuple(sampled_case_names),
        case_signatures=tuple(sampled_case_signatures),
        run_count=dataset.run_count,
        replay_count=dataset.replay_count,
        calibration_context=copy.deepcopy(dataset.calibration_context),
    )


def _weighted_score(
    evaluator_scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Combine evaluator-level scores with a normalized weight vector."""
    score = sum(weights[name] * evaluator_scores[name] for name in evaluator_scores)
    return min(max(score, 0.0), 1.0)


def _summarize_distribution(values: Sequence[float | None]) -> BootstrapSummary:
    """Produce percentile-bootstrap summaries and definition counts."""
    defined = sorted(value for value in values if value is not None and math.isfinite(value))
    if not defined:
        return BootstrapSummary(
            mean=None,
            lower_2_5=None,
            median=None,
            upper_97_5=None,
            defined_replicates=0,
            undefined_replicates=len(values),
        )

    return BootstrapSummary(
        mean=_mean(defined),
        lower_2_5=_percentile(defined, 0.025),
        median=_percentile(defined, 0.5),
        upper_97_5=_percentile(defined, 0.975),
        defined_replicates=len(defined),
        undefined_replicates=len(values) - len(defined),
    )


def _percentile(values: list[float], probability: float) -> float:
    """Interpolate a percentile from sorted values."""
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return values[lower_index]
    fraction = position - lower_index
    return values[lower_index] + fraction * (values[upper_index] - values[lower_index])
