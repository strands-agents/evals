"""Pluggable algorithms for experimental evaluator calibration."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from typing_extensions import Any, Literal

_EPSILON = 1e-15

WeightingMethod = Literal["equal", "repeatability", "inverse_step", "inverse_sqrt_step"]
WEIGHTING_METHODS: tuple[WeightingMethod, ...] = (
    "equal",
    "repeatability",
    "inverse_step",
    "inverse_sqrt_step",
)


@dataclass(frozen=True)
class CalibrationObservation:
    """One evaluator score in a repeated calibration experiment."""

    evaluator_name: str
    case_name: str
    agent_run: int
    judge_replay: int
    score: float


@dataclass(frozen=True)
class CalibrationExample:
    """Replay-aggregated evaluator features for one case and agent run."""

    case_name: str
    agent_run: int
    evaluator_scores: Mapping[str, float]


@dataclass(frozen=True)
class CalibrationData:
    """Validated observations and measurement diagnostics supplied to an algorithm."""

    observations: tuple[CalibrationObservation, ...]
    examples: tuple[CalibrationExample, ...]
    evaluator_names: tuple[str, ...]
    case_names: tuple[str, ...]
    agent_runs: int
    judge_replays: int
    step_sizes: Mapping[str, float]
    variance_components: Mapping[str, Mapping[str, float]]
    repeatabilities: Mapping[str, float]

    def evaluator_means(self) -> dict[str, float]:
        """Return the mean replay-aggregated score for each evaluator."""
        values: dict[str, list[float]] = {name: [] for name in self.evaluator_names}
        for example in self.examples:
            for evaluator_name, score in example.evaluator_scores.items():
                values[evaluator_name].append(score)
        return {name: sum(scores) / len(scores) for name, scores in values.items()}


@runtime_checkable
class FittedCalibrationModel(Protocol):
    """Scoring model returned by a `CalibrationAlgorithm`."""

    @property
    def algorithm_id(self) -> str:
        """Return the stable algorithm identifier."""
        ...

    @property
    def algorithm_version(self) -> int:
        """Return the fitted algorithm version."""
        ...

    @property
    def weights(self) -> Mapping[str, float] | None:
        """Return evaluator weights when the model has a linear aggregation."""
        ...

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return serializable fitted parameters for inspection and persistence."""
        ...

    def predict(self, evaluator_scores: Mapping[str, float]) -> float:
        """Predict one case-run outcome from its evaluator features."""
        ...


@runtime_checkable
class CalibrationAlgorithm(Protocol):
    """Stateless fitting strategy used by `Calibrator`."""

    @property
    def algorithm_id(self) -> str:
        """Return the stable algorithm identifier."""
        ...

    @property
    def algorithm_version(self) -> int:
        """Return the algorithm implementation version."""
        ...

    def fit(self, data: CalibrationData) -> FittedCalibrationModel:
        """Fit and return an immutable scoring model."""
        ...


@dataclass(frozen=True)
class WeightedCalibrationModel:
    """Fitted normalized weighted-average scoring model."""

    algorithm_id: str
    algorithm_version: int
    weighting_method: WeightingMethod
    evaluator_weights: Mapping[str, float]
    used_equal_fallback: bool = False

    @property
    def weights(self) -> Mapping[str, float]:
        """Return the normalized evaluator weights."""
        return self.evaluator_weights

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the fitted weighting parameters."""
        return {
            "weighting_method": self.weighting_method,
            "weights": dict(self.evaluator_weights),
            "used_equal_fallback": self.used_equal_fallback,
        }

    def predict(self, evaluator_scores: Mapping[str, float]) -> float:
        """Return the normalized weighted average of evaluator scores."""
        if set(evaluator_scores) != set(self.evaluator_weights):
            missing = sorted(set(self.evaluator_weights) - set(evaluator_scores))
            unexpected = sorted(set(evaluator_scores) - set(self.evaluator_weights))
            raise ValueError(
                f"evaluator scores do not match the fitted model; missing={missing}, unexpected={unexpected}"
            )
        return sum(self.evaluator_weights[name] * evaluator_scores[name] for name in evaluator_scores)


class WeightingAlgorithm:
    """Built-in adapter for experimental evaluator weighting diagnostics."""

    algorithm_version = 1

    def __init__(self, method: WeightingMethod) -> None:
        """Initialize a built-in weighting algorithm.

        Args:
            method: Weighting rule applied to the measurement diagnostics.

        Raises:
            ValueError: If the weighting method is unknown.
        """
        if method not in WEIGHTING_METHODS:
            raise ValueError(f"unknown weighting method: {method!r}")
        self.method = method
        self.algorithm_id = f"{method}_weighting"

    def fit(self, data: CalibrationData) -> WeightedCalibrationModel:
        """Fit a normalized weighted-average model."""
        weights = _compute_weight_vector(
            self.method,
            data.evaluator_names,
            data.repeatabilities,
            data.step_sizes,
        )
        used_equal_fallback = weights is None
        if weights is None:
            weights = _equal_weights(data.evaluator_names)
        return WeightedCalibrationModel(
            algorithm_id=self.algorithm_id,
            algorithm_version=self.algorithm_version,
            weighting_method=self.method,
            evaluator_weights=MappingProxyType(dict(weights)),
            used_equal_fallback=used_equal_fallback,
        )


def _compute_weight_vector(
    method: WeightingMethod,
    evaluator_names: tuple[str, ...] | list[str],
    repeatabilities: Mapping[str, float],
    step_sizes: Mapping[str, float],
) -> dict[str, float] | None:
    """Compute and normalize one evaluator-weight vector."""
    if method == "equal":
        return _equal_weights(evaluator_names)

    unnormalized: dict[str, float] = {}
    for evaluator_name in evaluator_names:
        repeatability = repeatabilities[evaluator_name]
        step = step_sizes[evaluator_name]
        if method == "repeatability":
            value = repeatability
        elif method == "inverse_step":
            value = repeatability / step
        else:
            value = repeatability / math.sqrt(step)
        unnormalized[evaluator_name] = value

    total = sum(unnormalized.values())
    if total <= _EPSILON:
        return None
    return {name: value / total for name, value in unnormalized.items()}


def _equal_weights(evaluator_names: tuple[str, ...] | list[str]) -> dict[str, float]:
    """Return an equal-weight vector for the supplied evaluators."""
    weight = 1.0 / len(evaluator_names)
    return {name: weight for name in evaluator_names}
