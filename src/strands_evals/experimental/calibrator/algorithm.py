"""Pluggable algorithms for experimental evaluator calibration."""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean
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
