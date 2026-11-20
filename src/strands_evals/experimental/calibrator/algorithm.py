"""Pluggable algorithms for experimental evaluator calibration."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

_EPSILON = 1e-15
_LOGISTIC_L2 = 1e-8
_LOGISTIC_MAX_ITERATIONS = 100
_LOGISTIC_TOLERANCE = 1e-10

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


@dataclass(frozen=True)
class CalibrationObservation:
    """One evaluator score in a repeated calibration experiment."""

    evaluator_name: str
    case_name: str
    agent_run: int
    judge_replay: int
    score: float


@dataclass(frozen=True)
class CalibrationTargetDefinition:
    """Stable identity of an external quality target."""

    name: str
    version: str

    def __post_init__(self) -> None:
        """Validate the target identity."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("target definition name must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("target definition version must be a non-empty string")


@dataclass(frozen=True)
class CalibrationTarget:
    """One external target value for a case and agent run."""

    case_name: str
    agent_run: int
    value: float

    def __post_init__(self) -> None:
        """Validate one target observation."""
        if not isinstance(self.case_name, str) or not self.case_name:
            raise ValueError("target case_name must be a non-empty string")
        if not isinstance(self.agent_run, int) or isinstance(self.agent_run, bool) or self.agent_run < 0:
            raise ValueError("target agent_run must be a non-negative integer")
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool) or not math.isfinite(self.value):
            raise ValueError("target value must be a finite number")
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class CalibrationExample:
    """Replay-aggregated evaluator features for one case and agent run."""

    case_name: str
    agent_run: int
    evaluator_scores: Mapping[str, float]
    target: float | None = None


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
    target_definition: CalibrationTargetDefinition | None = None

    def __post_init__(self) -> None:
        """Require targets to be either complete or absent."""
        targeted_examples = sum(example.target is not None for example in self.examples)
        if self.target_definition is None and targeted_examples:
            raise ValueError("target_definition is required when examples contain targets")
        if self.target_definition is not None and targeted_examples != len(self.examples):
            raise ValueError("every calibration example must contain a target")

    def evaluator_means(self) -> dict[str, float]:
        """Return the mean replay-aggregated score for each evaluator."""
        values: dict[str, list[float]] = {name: [] for name in self.evaluator_names}
        for example in self.examples:
            for evaluator_name, score in example.evaluator_scores.items():
                values[evaluator_name].append(score)
        return {name: sum(scores) / len(scores) for name, scores in values.items()}

    @property
    def has_targets(self) -> bool:
        """Return whether complete external targets are available."""
        return self.target_definition is not None


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
    """Stateless fitting strategy used by the calibration facade."""

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


@runtime_checkable
class RestorableCalibrationAlgorithm(CalibrationAlgorithm, Protocol):
    """Calibration algorithm that can rebuild a model from saved parameters."""

    def restore(self, parameters: Mapping[str, Any]) -> FittedCalibrationModel:
        """Restore a fitted model from deterministic profile parameters."""
        ...


@dataclass(frozen=True)
class EqualWeightingModel:
    """Fitted equal-weight average."""

    evaluator_weights: Mapping[str, float]
    algorithm_id: str = "equal_weighting"
    algorithm_version: int = 1

    @property
    def weights(self) -> Mapping[str, float]:
        """Return the normalized evaluator weights."""
        return self.evaluator_weights

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the fitted equal-weight parameters."""
        return {"weights": dict(self.evaluator_weights)}

    def predict(self, evaluator_scores: Mapping[str, float]) -> float:
        """Return the equal-weight average of evaluator scores."""
        if set(evaluator_scores) != set(self.evaluator_weights):
            missing = sorted(set(self.evaluator_weights) - set(evaluator_scores))
            unexpected = sorted(set(evaluator_scores) - set(self.evaluator_weights))
            raise ValueError(
                f"evaluator scores do not match the fitted model; missing={missing}, unexpected={unexpected}"
            )
        return sum(self.evaluator_weights[name] * evaluator_scores[name] for name in evaluator_scores)


class EqualWeightingAlgorithm:
    """Transparent equal-weight baseline used when no target is available."""

    algorithm_id = "equal_weighting"
    algorithm_version = 1

    def fit(self, data: CalibrationData) -> EqualWeightingModel:
        """Fit an equal-weight model over the configured evaluators."""
        return EqualWeightingModel(
            evaluator_weights=MappingProxyType(_equal_weights(data.evaluator_names)),
        )

    def restore(self, parameters: Mapping[str, Any]) -> EqualWeightingModel:
        """Restore an equal-weight model from saved profile parameters."""
        return EqualWeightingModel(
            evaluator_weights=MappingProxyType(_restore_weights(parameters)),
        )


@dataclass(frozen=True)
class CalibratedEqualModel:
    """Equal-weight average mapped to an external-target probability."""

    evaluator_weights: Mapping[str, float]
    intercept: float
    scale: float
    algorithm_id: str = "calibrated_equal"
    algorithm_version: int = 1

    @property
    def weights(self) -> Mapping[str, float]:
        """Return the equal evaluator weights."""
        return self.evaluator_weights

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the fitted probability-mapping parameters."""
        return {
            "weights": dict(self.evaluator_weights),
            "intercept": self.intercept,
            "scale": self.scale,
        }

    def predict(self, evaluator_scores: Mapping[str, float]) -> float:
        """Map one equal-weight average to a calibrated probability."""
        raw_average = _weighted_average(
            evaluator_scores,
            self.evaluator_weights,
        )
        return _sigmoid(self.intercept + self.scale * raw_average)


class CalibratedEqualAlgorithm:
    """Fit a logistic probability mapping over an equal-weight average."""

    algorithm_id = "calibrated_equal"
    algorithm_version = 1

    def fit(self, data: CalibrationData) -> CalibratedEqualModel:
        """Fit an intercept and scale against complete binary targets."""
        if not data.has_targets:
            raise ValueError("calibrated equal fitting requires complete external targets")

        evaluator_weights = _equal_weights(data.evaluator_names)
        raw_averages: list[float] = []
        targets: list[float] = []
        for example in data.examples:
            if example.target is None:
                raise ValueError("calibrated equal fitting requires complete external targets")
            if example.target not in {0.0, 1.0}:
                raise ValueError("calibrated equal fitting requires binary targets containing only 0 or 1")
            raw_averages.append(
                _weighted_average(
                    example.evaluator_scores,
                    evaluator_weights,
                )
            )
            targets.append(example.target)

        if set(targets) != {0.0, 1.0}:
            raise ValueError("calibrated equal fitting requires both binary target classes")

        intercept, scale = _fit_logistic_mapping(raw_averages, targets)
        return CalibratedEqualModel(
            evaluator_weights=MappingProxyType(evaluator_weights),
            intercept=intercept,
            scale=scale,
        )

    def restore(self, parameters: Mapping[str, Any]) -> CalibratedEqualModel:
        """Restore a calibrated-equal model from saved profile parameters."""
        evaluator_weights = _restore_weights(parameters)
        _validate_equal_weights(evaluator_weights)
        return CalibratedEqualModel(
            evaluator_weights=MappingProxyType(evaluator_weights),
            intercept=_restore_finite_parameter(parameters, "intercept"),
            scale=_restore_finite_parameter(parameters, "scale"),
        )


@dataclass(frozen=True)
class DiagnosticWeightingModel:
    """Fitted weighting model retained for stability diagnostics."""

    algorithm_id: str
    algorithm_version: int
    weighting_method: DiagnosticWeightingMethod
    evaluator_weights: Mapping[str, float]
    used_equal_fallback: bool = False

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
        return sum(self.evaluator_weights[name] * evaluator_scores[name] for name in evaluator_scores)


class DiagnosticWeightingAlgorithm:
    """Explicit opt-in adapter for repeatability and score-step diagnostics.

    These rules measure stability and score resolution. They do not establish
    semantic relevance or correctness against an external quality target.
    """

    algorithm_version = 1

    def __init__(self, method: DiagnosticWeightingMethod) -> None:
        """Initialize one diagnostic weighting rule.

        Args:
            method: Diagnostic weighting rule applied to measurement results.

        Raises:
            ValueError: If the method is not a supported diagnostic rule.
        """
        if method not in DIAGNOSTIC_WEIGHTING_METHODS:
            raise ValueError(f"unknown diagnostic weighting method: {method!r}")
        self.method = method
        self.algorithm_id = f"diagnostic_{method}_weighting"

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
            algorithm_id=self.algorithm_id,
            algorithm_version=self.algorithm_version,
            weighting_method=self.method,
            evaluator_weights=MappingProxyType(dict(weights)),
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
            algorithm_id=self.algorithm_id,
            algorithm_version=self.algorithm_version,
            weighting_method=self.method,
            evaluator_weights=MappingProxyType(_restore_weights(parameters)),
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


def _weighted_average(
    evaluator_scores: Mapping[str, float],
    evaluator_weights: Mapping[str, float],
) -> float:
    """Return a weighted average after validating the evaluator set."""
    if set(evaluator_scores) != set(evaluator_weights):
        missing = sorted(set(evaluator_weights) - set(evaluator_scores))
        unexpected = sorted(set(evaluator_scores) - set(evaluator_weights))
        raise ValueError(f"evaluator scores do not match the fitted model; missing={missing}, unexpected={unexpected}")
    return sum(evaluator_weights[name] * evaluator_scores[name] for name in evaluator_scores)


def _fit_logistic_mapping(
    raw_averages: list[float],
    targets: list[float],
) -> tuple[float, float]:
    """Fit a deterministic regularized two-parameter logistic mapping."""
    base_rate = sum(targets) / len(targets)
    intercept = math.log(base_rate / (1.0 - base_rate))
    if max(raw_averages) - min(raw_averages) <= _LOGISTIC_TOLERANCE:
        return intercept, 0.0
    scale = 0.0
    loss = _logistic_loss(raw_averages, targets, intercept, scale)

    for _ in range(_LOGISTIC_MAX_ITERATIONS):
        gradient_intercept = _LOGISTIC_L2 * intercept
        gradient_scale = _LOGISTIC_L2 * scale
        hessian_intercept = _LOGISTIC_L2
        hessian_cross = 0.0
        hessian_scale = _LOGISTIC_L2

        for raw_average, target in zip(raw_averages, targets, strict=True):
            probability = _sigmoid(intercept + scale * raw_average)
            residual = probability - target
            curvature = probability * (1.0 - probability)
            gradient_intercept += residual
            gradient_scale += residual * raw_average
            hessian_intercept += curvature
            hessian_cross += curvature * raw_average
            hessian_scale += curvature * raw_average * raw_average

        if max(abs(gradient_intercept), abs(gradient_scale)) <= _LOGISTIC_TOLERANCE:
            return intercept, scale

        determinant = hessian_intercept * hessian_scale - hessian_cross * hessian_cross
        if determinant <= _EPSILON:
            raise ValueError("calibrated equal logistic fit is numerically singular")

        intercept_step = (hessian_scale * gradient_intercept - hessian_cross * gradient_scale) / determinant
        scale_step = (hessian_intercept * gradient_scale - hessian_cross * gradient_intercept) / determinant

        step_size = 1.0
        while step_size >= _LOGISTIC_TOLERANCE:
            candidate_intercept = intercept - step_size * intercept_step
            candidate_scale = scale - step_size * scale_step
            candidate_loss = _logistic_loss(
                raw_averages,
                targets,
                candidate_intercept,
                candidate_scale,
            )
            if candidate_loss <= loss:
                intercept = candidate_intercept
                scale = candidate_scale
                loss = candidate_loss
                break
            step_size /= 2.0
        else:
            raise ValueError("calibrated equal logistic fit did not converge")

        if max(abs(step_size * intercept_step), abs(step_size * scale_step)) <= _LOGISTIC_TOLERANCE:
            return intercept, scale

    raise ValueError("calibrated equal logistic fit did not converge")


def _logistic_loss(
    raw_averages: list[float],
    targets: list[float],
    intercept: float,
    scale: float,
) -> float:
    """Return regularized binary log loss without exponential overflow."""
    loss = 0.5 * _LOGISTIC_L2 * (intercept * intercept + scale * scale)
    for raw_average, target in zip(raw_averages, targets, strict=True):
        linear_value = intercept + scale * raw_average
        if linear_value >= 0.0:
            loss += linear_value + math.log1p(math.exp(-linear_value))
        else:
            loss += math.log1p(math.exp(linear_value))
        loss -= target * linear_value
    return loss


def _sigmoid(value: float) -> float:
    """Return a numerically stable logistic sigmoid."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _restore_finite_parameter(
    parameters: Mapping[str, Any],
    name: str,
) -> float:
    """Validate and return one finite saved model parameter."""
    value = parameters.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"saved {name} must be a finite number")
    return float(value)


def _validate_equal_weights(weights: Mapping[str, float]) -> None:
    """Require a restored calibrated-equal model to retain equal weights."""
    if not weights:
        raise ValueError("saved calibrated-equal weights must not be empty")
    expected = 1.0 / len(weights)
    if any(not math.isclose(weight, expected, abs_tol=1e-12) for weight in weights.values()):
        raise ValueError("saved calibrated-equal weights must be equal")


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
