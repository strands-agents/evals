"""Pluggable algorithms for experimental evaluator calibration."""

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean
from types import MappingProxyType
from typing import Any, Literal

VarianceSource = Literal["judge", "agent", "case"]

_EPSILON = 1e-15

WeightingMethod = Literal["equal", "repeatability", "inverse_step", "inverse_sqrt_step"]
WEIGHTING_METHODS: tuple[WeightingMethod, ...] = (
    "equal",
    "repeatability",
    "inverse_step",
    "inverse_sqrt_step",
)
DiagnosticWeightingMethod = Literal["repeatability", "inverse_step", "inverse_sqrt_step"]
DIAGNOSTIC_WEIGHTING_METHODS: tuple[DiagnosticWeightingMethod, ...] = (
    "repeatability",
    "inverse_step",
    "inverse_sqrt_step",
)


@dataclass(frozen=True, kw_only=True)
class CalibrationObservation:
    """One evaluator score in a repeated calibration experiment."""

    evaluator_name: str
    case_name: str
    agent_run: int
    judge_replay: int
    score: float


@dataclass(frozen=True, kw_only=True)
class CalibrationSample:
    """Replay-aggregated evaluator scores for one case and agent run."""

    case_name: str
    agent_run: int
    evaluator_scores: Mapping[str, float]

    def __post_init__(self) -> None:
        """Copy evaluator scores into an immutable mapping."""
        object.__setattr__(
            self,
            "evaluator_scores",
            MappingProxyType(dict(self.evaluator_scores)),
        )


@dataclass(frozen=True, kw_only=True)
class CalibrationData:
    """Immutable observations and measurement diagnostics supplied to an algorithm."""

    observations: tuple[CalibrationObservation, ...]
    evaluator_names: tuple[str, ...]
    case_names: tuple[str, ...]
    agent_runs: int
    judge_replays: int
    step_sizes: Mapping[str, float]
    variance_components: Mapping[str, Mapping[VarianceSource, float]]
    repeatabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        """Copy mutable inputs and validate evaluator names."""
        if not self.evaluator_names:
            raise ValueError("calibration data must contain at least one evaluator")
        if len(set(self.evaluator_names)) != len(self.evaluator_names):
            raise ValueError("calibration evaluator names must be unique")

        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "evaluator_names", tuple(self.evaluator_names))
        object.__setattr__(self, "case_names", tuple(self.case_names))
        object.__setattr__(
            self,
            "step_sizes",
            MappingProxyType(dict(self.step_sizes)),
        )
        object.__setattr__(
            self,
            "variance_components",
            MappingProxyType(
                {
                    evaluator_name: MappingProxyType(dict(components))
                    for evaluator_name, components in self.variance_components.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "repeatabilities",
            MappingProxyType(dict(self.repeatabilities)),
        )

    @property
    def samples(self) -> tuple[CalibrationSample, ...]:
        """Return replay-aggregated samples grouped by case and agent run."""
        grouped_scores: defaultdict[tuple[int, str], defaultdict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for observation in self.observations:
            grouped_scores[observation.agent_run, observation.case_name][observation.evaluator_name].append(
                observation.score
            )

        return tuple(
            CalibrationSample(
                case_name=case_name,
                agent_run=agent_run,
                evaluator_scores={name: fmean(scores) for name, scores in scores_by_evaluator.items()},
            )
            for (agent_run, case_name), scores_by_evaluator in sorted(grouped_scores.items())
        )


class CalibrationModel(ABC):
    """Scoring model returned by a `CalibrationAlgorithm`."""

    @property
    def weights(self) -> Mapping[str, float] | None:
        """Return evaluator weights when the model has a linear aggregation."""
        return None

    @property
    @abstractmethod
    def parameters(self) -> Mapping[str, Any]:
        """Return serializable fitted parameters for inspection and persistence."""
        ...

    @abstractmethod
    def predict(self, evaluator_scores: Mapping[str, float]) -> float:
        """Predict one case-run outcome from its evaluator features."""
        ...


class CalibrationAlgorithm(ABC):
    """Stateless fitting strategy used by the calibration facade."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return the stable, versioned identity used by persisted profiles."""
        ...

    @abstractmethod
    def fit(self, data: CalibrationData) -> CalibrationModel:
        """Fit and return a scoring model."""
        ...

    def restore(self, parameters: Mapping[str, Any]) -> CalibrationModel:
        """Restore a fitted model from saved parameters when supported."""
        raise NotImplementedError("this calibration algorithm does not support profile restoration")


@dataclass(frozen=True, kw_only=True)
class DiagnosticWeightingModel(CalibrationModel):
    """Fitted weighting model retained for stability diagnostics."""

    weighting_method: DiagnosticWeightingMethod
    evaluator_weights: Mapping[str, float]
    used_equal_fallback: bool = False

    def __post_init__(self) -> None:
        """Copy fitted weights into an immutable mapping."""
        object.__setattr__(
            self,
            "evaluator_weights",
            MappingProxyType(dict(self.evaluator_weights)),
        )

    @property
    def weights(self) -> Mapping[str, float]:
        """Return the normalized evaluator weights."""
        return self.evaluator_weights

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the fitted diagnostic weighting parameters."""
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
        score = sum(self.evaluator_weights[name] * evaluator_scores[name] for name in evaluator_scores)
        return min(max(score, 0.0), 1.0)


class DiagnosticWeightingAlgorithm(CalibrationAlgorithm):
    """Explicit opt-in adapter for repeatability and score-step diagnostics.

    `repeatability` weights directly by measurement repeatability.
    `inverse_step` divides repeatability by the evaluator's meaningful score
    increment, while `inverse_sqrt_step` applies the milder square-root
    adjustment. These rules measure stability and score resolution; they do
    not establish semantic relevance or correctness against a quality target.
    """

    def __init__(self, method: DiagnosticWeightingMethod) -> None:
        """Initialize one diagnostic weighting rule.

        Args:
            method: Diagnostic weighting rule applied to measurement results.

        Raises:
            ValueError: If the method is not a supported diagnostic rule.
        """
        if method not in DIAGNOSTIC_WEIGHTING_METHODS:
            raise ValueError(
                f"unknown diagnostic weighting method {method!r}; expected one of {list(DIAGNOSTIC_WEIGHTING_METHODS)}"
            )
        self.method = method

    @property
    def identity(self) -> str:
        """Return the stable identity for this diagnostic rule."""
        return f"diagnostic_{self.method}_weighting:v1"

    def fit(self, data: CalibrationData) -> DiagnosticWeightingModel:
        """Fit a normalized diagnostic weighted-average model."""
        weights = _compute_weight_vector(
            self.method,
            data.evaluator_names,
            data.repeatabilities,
            data.step_sizes,
        )
        used_equal_fallback = weights is None
        if weights is None:
            weights = _equal_weights(data.evaluator_names)
        return DiagnosticWeightingModel(
            weighting_method=self.method,
            evaluator_weights=weights,
            used_equal_fallback=used_equal_fallback,
        )

    def restore(self, parameters: Mapping[str, Any]) -> DiagnosticWeightingModel:
        """Restore a diagnostic model from saved profile parameters."""
        if parameters.get("weighting_method") != self.method:
            raise ValueError("saved weighting method does not match the configured algorithm")
        used_equal_fallback = parameters.get("used_equal_fallback", False)
        if not isinstance(used_equal_fallback, bool):
            raise ValueError("saved used_equal_fallback must be a boolean")
        return DiagnosticWeightingModel(
            weighting_method=self.method,
            evaluator_weights=_restore_weights(parameters),
            used_equal_fallback=used_equal_fallback,
        )


def _compute_weight_vector(
    method: WeightingMethod,
    evaluator_names: tuple[str, ...] | list[str],
    repeatabilities: Mapping[str, float],
    step_sizes: Mapping[str, float],
) -> dict[str, float] | None:
    """Compute and normalize one candidate evaluator-weight vector."""
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
        elif method == "inverse_sqrt_step":
            value = repeatability / math.sqrt(step)
        else:
            raise ValueError(f"unknown weighting method: {method!r}")
        unnormalized[evaluator_name] = value

    total = sum(unnormalized.values())
    if total <= _EPSILON:
        return None
    return {name: value / total for name, value in unnormalized.items()}


def _equal_weights(evaluator_names: tuple[str, ...] | list[str]) -> dict[str, float]:
    """Return an equal-weight vector for the supplied evaluators."""
    weight = 1.0 / len(evaluator_names)
    return {name: weight for name in evaluator_names}


def _restore_weights(parameters: Mapping[str, Any]) -> dict[str, float]:
    """Validate and copy a saved evaluator-weight mapping."""
    raw_weights = parameters.get("weights")
    if not isinstance(raw_weights, Mapping):
        raise ValueError("saved model parameters must contain a weights mapping")
    weights: dict[str, float] = {}
    for evaluator_name, weight in raw_weights.items():
        if not isinstance(evaluator_name, str):
            raise ValueError("saved weight names must be strings")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValueError("saved weights must be numeric")
        weights[evaluator_name] = float(weight)
    return weights
