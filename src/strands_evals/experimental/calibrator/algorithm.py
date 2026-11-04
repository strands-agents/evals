"""Pluggable algorithms for experimental evaluator calibration."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

VarianceSource = Literal["judge", "agent", "case"]


@dataclass(frozen=True)
class CalibrationObservation:
    """One evaluator score in a repeated calibration experiment."""

    evaluator_name: str
    case_name: str
    agent_run: int
    judge_replay: int
    score: float


@dataclass(frozen=True)
class CalibrationSample:
    """Replay-aggregated evaluator scores for one case and agent run."""

    case_name: str
    agent_run: int
    evaluator_scores: Mapping[str, float]


@dataclass(frozen=True)
class CalibrationData:
    """Validated observations and measurement diagnostics supplied to an algorithm."""

    observations: tuple[CalibrationObservation, ...]
    evaluator_names: tuple[str, ...]
    case_names: tuple[str, ...]
    agent_runs: int
    judge_replays: int
    step_sizes: Mapping[str, float]
    variance_components: Mapping[str, Mapping[VarianceSource, float]]
    repeatabilities: Mapping[str, float]

    @property
    def samples(self) -> tuple[CalibrationSample, ...]:
        """Return replay-aggregated samples grouped by case and agent run."""
        grouped_scores: dict[tuple[str, int], dict[str, list[float]]] = {}
        for observation in self.observations:
            sample_scores = grouped_scores.setdefault((observation.case_name, observation.agent_run), {})
            sample_scores.setdefault(observation.evaluator_name, []).append(observation.score)

        samples = []
        ordered_groups = sorted(
            grouped_scores.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )
        for (case_name, agent_run), evaluator_scores in ordered_groups:
            samples.append(
                CalibrationSample(
                    case_name=case_name,
                    agent_run=agent_run,
                    evaluator_scores={
                        evaluator_name: sum(scores) / len(scores) for evaluator_name, scores in evaluator_scores.items()
                    },
                )
            )
        return tuple(samples)


class CalibrationModel(ABC):
    """Scoring model returned by a `CalibrationAlgorithm`."""

    @property
    @abstractmethod
    def weights(self) -> Mapping[str, float] | None:
        """Return evaluator weights when the model has a linear aggregation."""
        ...

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
        """Return the stable algorithm identity."""
        ...

    @abstractmethod
    def fit(self, data: CalibrationData) -> CalibrationModel:
        """Fit and return a scoring model."""
        ...
