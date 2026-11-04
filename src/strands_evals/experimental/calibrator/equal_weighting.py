"""Equal-weight calibration algorithm."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .algorithm import CalibrationAlgorithm, CalibrationData, CalibrationModel


@dataclass(frozen=True)
class EqualWeightingModel(CalibrationModel):
    """Fitted equal-weight average."""

    evaluator_weights: dict[str, float]

    @property
    def weights(self) -> Mapping[str, float]:
        """Return the normalized evaluator weights."""
        return dict(self.evaluator_weights)

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


class EqualWeightingAlgorithm(CalibrationAlgorithm):
    """Transparent equal-weight baseline used when no target is available."""

    @property
    def identity(self) -> str:
        """Return the stable equal-weighting identity."""
        return "equal_weighting:v1"

    def fit(self, data: CalibrationData) -> EqualWeightingModel:
        """Fit an equal-weight model over the configured evaluators."""
        weight = 1.0 / len(data.evaluator_names)
        return EqualWeightingModel(
            evaluator_weights={name: weight for name in data.evaluator_names},
        )
