"""Pluggable algorithms for experimental evaluator calibration."""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean
from types import MappingProxyType
from typing import Any, Literal

VarianceSource = Literal["judge", "agent", "case"]


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
