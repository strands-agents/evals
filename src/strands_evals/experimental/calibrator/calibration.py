"""Experimental evaluator calibration facade."""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from typing_extensions import Self

from ...types.evaluation_report import EvaluationReport
from ._calibration import (
    CalibrationStatus,
    _CalibrationDataset,
    _CalibrationState,
    _fit_dataset,
    _initial_calibration_state,
    _normalize_configuration_value,
    _prepare_calibration_dataset,
)
from .algorithm import (
    CalibrationAlgorithm,
    CalibrationData,
    EqualWeightingAlgorithm,
    EqualWeightingModel,
    FittedCalibrationModel,
    _equal_weights,
)


class Calibrator:
    """Fit a pluggable scoring algorithm from repeated evaluation reports.

    This API is experimental and may change in a minor release.

    Reports passed to `fit` are grouped first by independent agent run and then
    by judge replay:

    ```python
    calibrator.fit(
        [
            [run_0_replay_0, run_0_replay_1],
            [run_1_replay_0, run_1_replay_1],
        ]
    )
    ```

    Fitting requires at least two cases, two agent runs, and two judge replays.
    Scoring may use new cases and different positive run or replay counts as
    long as the evaluator set matches the fitted model.
    """

    def __init__(
        self,
        *,
        step_sizes: dict[str, float],
        algorithm: CalibrationAlgorithm | None = None,
    ) -> None:
        """Initialize a calibrator with a scoring algorithm.

        Args:
            step_sizes: Mapping from evaluator instance name to its smallest
                meaningful score increment. The core stores these values for
                algorithms and later uncertainty extensions; equal weighting
                does not reinterpret them as semantic importance.
            algorithm: Custom fitting strategy. Defaults to equal weighting.

        Raises:
            ValueError: If no evaluators are provided, a step is invalid, or
                the algorithm configuration is invalid.
        """
        if not step_sizes:
            raise ValueError("step_sizes must contain at least one evaluator")
        invalid = {
            name: step
            for name, step in step_sizes.items()
            if not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0
        }
        if invalid:
            raise ValueError(f"step_sizes must be finite and greater than zero: {invalid}")

        self._step_sizes = {name: float(step) for name, step in step_sizes.items()}
        self._algorithm = _validate_algorithm(
            EqualWeightingAlgorithm() if algorithm is None else algorithm,
        )
        self._model: FittedCalibrationModel = EqualWeightingModel(
            evaluator_weights=MappingProxyType(_equal_weights(tuple(self._step_sizes))),
        )
        self._state = _initial_calibration_state()

    @property
    def algorithm_id(self) -> str:
        """Return the active scoring model's algorithm identifier."""
        return self._model.algorithm_id

    @property
    def algorithm_version(self) -> int:
        """Return the active scoring model's algorithm version."""
        return self._model.algorithm_version

    @property
    def weights(self) -> dict[str, float] | None:
        """Return fitted evaluator weights when the model exposes them."""
        return None if self._model.weights is None else dict(self._model.weights)

    @property
    def model_parameters(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible fitted model parameters."""
        normalized = _normalize_configuration_value(dict(self._model.parameters))
        if not isinstance(normalized, dict):
            raise ValueError("fitted model parameters must normalize to a mapping")
        return normalized

    @property
    def calibration_design(self) -> dict[str, int] | None:
        """Return the fitted agent-run and judge-replay counts."""
        if self._state.calibration_design is None:
            return None
        agent_runs, judge_replays = self._state.calibration_design
        return {
            "agent_runs": agent_runs,
            "judge_replays": judge_replays,
        }

    @property
    def variance_components(self) -> dict[str, dict[str, float]]:
        """Return corrected non-negative variance estimates by evaluator."""
        return {name: dict(components) for name, components in self._state.variance_components.items()}

    @property
    def repeatabilities(self) -> dict[str, float]:
        """Return judge-repeatability estimates for fixed agent outputs."""
        return dict(self._state.repeatabilities)

    @property
    def is_fitted(self) -> bool:
        """Whether measurement data has been fitted to the configured algorithm."""
        return self._state.calibration_design is not None

    @property
    def status(self) -> CalibrationStatus:
        """Return the current calibration readiness status."""
        return self._state.status

    def fit(self, reports_by_run: list[list[EvaluationReport]]) -> Self:
        """Fit the configured algorithm from repeated evaluation reports.

        Args:
            reports_by_run: Reports grouped by agent run and judge replay.

        Returns:
            This calibrator with its fitted scoring model updated.

        Raises:
            ValueError: If reports are incomplete or structurally incompatible.
        """
        dataset = _prepare_calibration_dataset(
            reports_by_run,
            configured_evaluators=set(self._step_sizes),
            expected_design=self._state.calibration_design,
            expected_structure=self._state.calibration_structure,
            operation="fit",
        )
        fitted_state = _fit_dataset(dataset)
        data = _build_algorithm_data(dataset, fitted_state, self._step_sizes)
        fitted_model = _validate_fitted_model(
            self._algorithm.fit(data),
            self._algorithm,
            set(self._step_sizes),
        )
        _score_dataset(fitted_model, dataset)
        self._state = fitted_state
        self._model = fitted_model
        return self

    def score(self, reports_by_run: list[list[EvaluationReport]]) -> float:
        """Apply the active model to structurally compatible reports.

        Before fitting, this returns the transparent equal-weight score with an
        `uncalibrated` status. After fitting, one or more cases, runs, and
        replays are accepted as long as the evaluator set remains compatible.

        Args:
            reports_by_run: Reports grouped by agent run and judge replay.

        Returns:
            Predicted composite score.

        Raises:
            ValueError: If reports are internally unbalanced or use a different
                evaluator set.
        """
        dataset = _prepare_calibration_dataset(
            reports_by_run,
            configured_evaluators=set(self._step_sizes),
            expected_design=self._state.calibration_design,
            expected_structure=self._state.calibration_structure,
            operation="scoring",
            scoring=True,
        )
        return _score_dataset(self._model, dataset)


def _build_algorithm_data(
    dataset: _CalibrationDataset,
    state: _CalibrationState,
    step_sizes: dict[str, float],
) -> CalibrationData:
    """Build an immutable public view of one validated calibration dataset."""
    variance_components = {
        name: MappingProxyType(dict(components)) for name, components in state.variance_components.items()
    }
    return CalibrationData(
        observations=dataset.observations(),
        examples=dataset.examples(),
        evaluator_names=dataset.evaluator_names,
        case_names=dataset.case_names,
        agent_runs=dataset.run_count,
        judge_replays=dataset.replay_count,
        step_sizes=MappingProxyType(dict(step_sizes)),
        variance_components=MappingProxyType(variance_components),
        repeatabilities=MappingProxyType(dict(state.repeatabilities)),
    )


def _validate_algorithm(algorithm: object) -> CalibrationAlgorithm:
    """Validate the stable identity and fitting surface of an algorithm."""
    algorithm_id = getattr(algorithm, "algorithm_id", None)
    algorithm_version = getattr(algorithm, "algorithm_version", None)
    if not isinstance(algorithm_id, str) or not algorithm_id:
        raise ValueError("algorithm_id must be a non-empty string")
    if not isinstance(algorithm_version, int) or isinstance(algorithm_version, bool) or algorithm_version < 1:
        raise ValueError("algorithm_version must be an integer greater than or equal to 1")
    if not callable(getattr(algorithm, "fit", None)):
        raise ValueError("algorithm must define a callable fit method")
    return cast(CalibrationAlgorithm, algorithm)


def _validate_fitted_model(
    model: object,
    algorithm: CalibrationAlgorithm,
    evaluator_names: set[str],
) -> FittedCalibrationModel:
    """Validate the identity and optional weights of a fitted model."""
    if getattr(model, "algorithm_id", None) != algorithm.algorithm_id:
        raise ValueError("fitted model algorithm_id must match the configured algorithm")
    if getattr(model, "algorithm_version", None) != algorithm.algorithm_version:
        raise ValueError("fitted model algorithm_version must match the configured algorithm")
    if not callable(getattr(model, "predict", None)):
        raise ValueError("fitted model must define a callable predict method")
    parameters = getattr(model, "parameters", None)
    if not isinstance(parameters, Mapping):
        raise ValueError("fitted model parameters must be a mapping")
    try:
        normalized_parameters = _normalize_configuration_value(dict(parameters))
    except ValueError as error:
        raise ValueError("fitted model parameters must be deterministic and JSON-compatible") from error
    if not isinstance(normalized_parameters, dict):
        raise ValueError("fitted model parameters must normalize to a mapping")
    weights = getattr(model, "weights", None)
    if weights is None:
        return cast(FittedCalibrationModel, model)
    if not isinstance(weights, Mapping):
        raise ValueError("fitted model weights must be a mapping or None")
    if set(weights) != evaluator_names:
        raise ValueError("fitted model weights must match the configured evaluators")
    if any(
        not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight) or weight < 0.0
        for weight in weights.values()
    ):
        raise ValueError("fitted model weights must be finite and non-negative")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("fitted model weights must sum to 1")
    return cast(FittedCalibrationModel, model)


def _predict_model(
    model: FittedCalibrationModel,
    evaluator_scores: dict[str, float],
) -> float:
    """Predict and validate a score from a fitted calibration model."""
    score = model.predict(MappingProxyType(dict(evaluator_scores)))
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
        raise ValueError("fitted model prediction must be a finite number")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"fitted model prediction must be between 0 and 1, got {score}")
    return float(score)


def _score_dataset(
    model: FittedCalibrationModel,
    dataset: _CalibrationDataset,
) -> float:
    """Average model predictions across case-run examples."""
    predictions = [_predict_model(model, dict(example.evaluator_scores)) for example in dataset.examples()]
    return sum(predictions) / len(predictions)
