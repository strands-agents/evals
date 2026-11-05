"""Pluggable algorithms for experimental evaluator calibration."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


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


def _equal_weights(evaluator_names: tuple[str, ...] | list[str]) -> dict[str, float]:
    """Return an equal-weight vector for the supplied evaluators."""
    weight = 1.0 / len(evaluator_names)
    return {name: weight for name in evaluator_names}
