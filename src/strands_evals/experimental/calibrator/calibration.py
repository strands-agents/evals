"""Experimental evaluator calibration orchestration."""

import asyncio
import copy
import inspect
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict
from types import MappingProxyType
from typing import cast

from pydantic import BaseModel
from typing_extensions import Any, Generic, Self

from ...case import Case
from ...experiment import Experiment
from ...types.evaluation import InputT, OutputT
from ...types.evaluation_report import EvaluationReport
from ._calibration import (
    CalibrationStatus,
    _BootstrapResult,
    _build_calibration_context,
    _CalibrationDataset,
    _CalibrationState,
    _collect_repeated_reports,
    _fit_dataset,
    _initial_calibration_state,
    _normalize_configuration_value,
    _prepare_calibration_dataset,
    _validate_bootstrap_options,
    _validate_calibration_compatibility,
    _validate_evaluator_names,
    _validate_repetition_counts,
    _weighted_score,
)
from .algorithm import (
    WEIGHTING_METHODS,
    CalibrationAlgorithm,
    CalibrationData,
    FittedCalibrationModel,
    RestorableCalibrationAlgorithm,
    WeightedCalibrationModel,
    WeightingAlgorithm,
    WeightingMethod,
    _compute_weight_vector,
    _equal_weights,
)
from .calibration_profile import CalibrationProfile, ProfileCompatibility


class CalibrationResult(BaseModel):
    """Point-in-time calibration result for in-memory inspection.

    Mutating the returned observations does not recompute `score` or update the
    fitted calibrator. This model is not a stable persistence format, and custom
    `EvaluationReport` subclasses are not guaranteed to survive serialization
    and deserialization.
    """

    score: float
    reports_by_run: list[list[EvaluationReport]]
    weights: dict[str, float] | None
    algorithm_id: str
    algorithm_version: int
    model_parameters: dict[str, Any]
    status: CalibrationStatus
    calibration_context: dict[str, Any] | None


class Calibrator(Generic[InputT, OutputT]):
    """Fit and apply a pluggable evaluator scoring algorithm.

    This API is experimental and may change in a minor release.

    The calibrator is the public facade for collecting repeated evaluations,
    estimating measurement diagnostics, fitting one configured algorithm, and
    scoring reports. Algorithms receive a validated, immutable view of the
    observations and return a fitted model with a common `predict` interface.

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

    Reports collected through `run_evaluations` carry a full experiment
    fingerprint. Later fits must match that calibration design. Scoring may use
    new cases and different positive run or replay counts, but the evaluator
    configuration must remain compatible with the fitted model.
    """

    def __init__(
        self,
        *,
        step_sizes: dict[str, float],
        weighting_method: WeightingMethod | None = None,
        algorithm: CalibrationAlgorithm | None = None,
    ) -> None:
        """Initialize a calibrator with a scoring algorithm.

        Args:
            step_sizes: Mapping from evaluator instance name to its smallest
                meaningful score increment.
            weighting_method: Backward-compatible built-in weighting rule.
                Defaults to equal weighting. Mutually exclusive with
                `algorithm`.
            algorithm: Custom fitting strategy. The algorithm must be stateless
                and return a fitted model implementing `predict`.

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
        if algorithm is not None and weighting_method is not None:
            raise ValueError("algorithm and weighting_method are mutually exclusive")

        self._step_sizes = {name: float(step) for name, step in step_sizes.items()}
        selected_algorithm = (
            WeightingAlgorithm("equal" if weighting_method is None else weighting_method)
            if algorithm is None
            else algorithm
        )
        self._algorithm = _validate_algorithm(selected_algorithm)
        initial_weights = _equal_weights(tuple(self._step_sizes))
        self._model: FittedCalibrationModel = WeightedCalibrationModel(
            algorithm_id="equal_weighting",
            algorithm_version=1,
            weighting_method="equal",
            evaluator_weights=MappingProxyType(initial_weights),
        )
        self._state = _initial_calibration_state(self._step_sizes)
        self._profile: CalibrationProfile | None = None

    @classmethod
    def from_profile(
        cls,
        profile: CalibrationProfile,
        *,
        algorithm: RestorableCalibrationAlgorithm | None = None,
    ) -> Self:
        """Restore a fitted calibrator from a compatible profile.

        Args:
            profile: Validated calibration profile to restore.
            algorithm: Algorithm implementation used to restore a custom
                fitted model. Built-in weighting profiles restore
                automatically.

        Returns:
            A calibrator with the profile's weights and diagnostics.

        Raises:
            ValueError: If the profile is stale, incompatible, or requires a
                custom algorithm that cannot restore saved parameters.
        """
        validated_profile = CalibrationProfile.from_dict(profile.to_dict())
        if algorithm is None:
            if validated_profile.weighting_method is None:
                raise ValueError("custom calibration profiles require a restorable algorithm")
            selected_algorithm: CalibrationAlgorithm = WeightingAlgorithm(validated_profile.weighting_method)
        else:
            selected_algorithm = algorithm
        selected_algorithm = _validate_algorithm(selected_algorithm)
        compatibility = validated_profile.algorithm_compatibility(
            selected_algorithm.algorithm_id,
            selected_algorithm.algorithm_version,
        )
        if compatibility != ProfileCompatibility.COMPATIBLE:
            raise ValueError(f"cannot restore {compatibility.value} calibration profile")

        restore = getattr(selected_algorithm, "restore", None)
        if not callable(restore):
            raise ValueError("profile restoration requires an algorithm with a callable restore method")
        fitted_model = _validate_fitted_model(
            restore(MappingProxyType(copy.deepcopy(validated_profile.model_parameters))),
            selected_algorithm,
            set(validated_profile.step_sizes),
        )
        if _normalized_model_parameters(fitted_model) != validated_profile.model_parameters:
            raise ValueError("restored model parameters do not match the calibration profile")
        restored_weights = None if fitted_model.weights is None else dict(fitted_model.weights)
        if restored_weights != validated_profile.weights:
            raise ValueError("restored model weights do not match the calibration profile")

        calibrator = cls(
            step_sizes=dict(validated_profile.step_sizes),
            algorithm=selected_algorithm,
        )
        calibrator._state = _state_from_profile(validated_profile)
        calibrator._model = fitted_model
        calibrator._profile = validated_profile.model_copy(deep=True)
        return calibrator

    @property
    def algorithm_id(self) -> str:
        """Return the configured algorithm identifier."""
        return self._algorithm.algorithm_id

    @property
    def algorithm_version(self) -> int:
        """Return the configured algorithm version."""
        return self._algorithm.algorithm_version

    @property
    def weighting_method(self) -> WeightingMethod | None:
        """Return the built-in weighting rule, if one is configured."""
        if isinstance(self._algorithm, WeightingAlgorithm):
            return self._algorithm.method
        return None

    @property
    def weights(self) -> dict[str, float] | None:
        """Return fitted evaluator weights when the model exposes them."""
        return None if self._model.weights is None else dict(self._model.weights)

    @property
    def model_parameters(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible fitted model parameters."""
        return _normalized_model_parameters(self._model)

    @property
    def profile(self) -> CalibrationProfile | None:
        """Return a copy of the fitted calibration profile, if available."""
        return None if self._profile is None else self._profile.model_copy(deep=True)

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
    def calibration_context(self) -> dict[str, Any] | None:
        """Return the fitted experiment compatibility fingerprint."""
        if self._state.calibration_context is None:
            return None
        return copy.deepcopy(self._state.calibration_context)

    @property
    def variance_components(self) -> dict[str, dict[str, float]]:
        """Return corrected non-negative variance estimates by evaluator."""
        return {name: dict(components) for name, components in self._state.variance_components.items()}

    @property
    def repeatabilities(self) -> dict[str, float]:
        """Return judge-repeatability estimates for fixed agent outputs."""
        return dict(self._state.repeatabilities)

    @property
    def candidate_weights(self) -> dict[WeightingMethod, dict[str, float] | None]:
        """Return weight vectors for every candidate weighting rule."""
        return {
            method: None if weights is None else dict(weights)
            for method, weights in self._state.candidate_weights.items()
        }

    @property
    def bootstrap_replicates(self) -> int:
        """Return the number of bootstrap replicates from the latest fit."""
        return 0 if self._state.bootstrap_result is None else self._state.bootstrap_result.replicates

    @property
    def bootstrap_seed(self) -> int | None:
        """Return the random seed from the latest bootstrap run."""
        return None if self._state.bootstrap_result is None else self._state.bootstrap_result.seed

    @property
    def repeatability_intervals(self) -> dict[str, dict[str, float | int | None]]:
        """Return bootstrap summaries for evaluator repeatability."""
        if self._state.bootstrap_result is None:
            return {}
        return {name: dict(summary) for name, summary in self._state.bootstrap_result.repeatability_intervals.items()}

    @property
    def candidate_weight_intervals(
        self,
    ) -> dict[WeightingMethod, dict[str, dict[str, float | int | None]]]:
        """Return bootstrap summaries for every candidate weight vector."""
        if self._state.bootstrap_result is None:
            return {}
        return {
            method: {name: dict(summary) for name, summary in intervals.items()}
            for method, intervals in self._state.bootstrap_result.candidate_weight_intervals.items()
        }

    @property
    def composite_score_intervals(self) -> dict[WeightingMethod, dict[str, float | int | None]]:
        """Return bootstrap summaries for calibration-suite composite scores."""
        if self._state.bootstrap_result is None:
            return {}
        return {
            method: dict(summary) for method, summary in self._state.bootstrap_result.composite_score_intervals.items()
        }

    @property
    def is_fitted(self) -> bool:
        """Whether measurement data has been fitted to the configured algorithm."""
        return self._state.calibration_design is not None

    @property
    def status(self) -> CalibrationStatus:
        """Return the current calibration readiness status."""
        return self._state.status

    def run_evaluations(
        self,
        experiment: Experiment[InputT, OutputT],
        task: Callable[[Case[InputT, OutputT]], OutputT | dict[str, Any]],
        *,
        agent_runs: int,
        judge_replays: int,
        max_workers: int = 1,
        bootstrap_replicates: int = 0,
        bootstrap_seed: int = 0,
    ) -> CalibrationResult:
        """Run repeated evaluations, fit the algorithm, and return the result.

        Args:
            experiment: Existing experiment containing cases and evaluators.
            task: Task function evaluated by the experiment.
            agent_runs: Number of independent task executions per case.
            judge_replays: Number of evaluator passes over each fixed output.
            max_workers: Maximum parallel case workers used by the experiment.
            bootstrap_replicates: Number of hierarchical bootstrap samples.
                Zero disables bootstrap uncertainty estimates.
            bootstrap_seed: Random seed used for reproducible resampling.

        Returns:
            The predicted score, raw reports, and fitted-model snapshot.

        Raises:
            ValueError: If `task` is async or repetition counts are invalid.
        """
        if inspect.iscoroutinefunction(task):
            raise ValueError("Async task is not supported. Please use run_evaluations_async instead.")

        return asyncio.run(
            self.run_evaluations_async(
                experiment,
                task,
                agent_runs=agent_runs,
                judge_replays=judge_replays,
                max_workers=max_workers,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
        )

    async def run_evaluations_async(
        self,
        experiment: Experiment[InputT, OutputT],
        task: Callable[
            [Case[InputT, OutputT]],
            OutputT | dict[str, Any] | Awaitable[OutputT | dict[str, Any]],
        ],
        *,
        agent_runs: int,
        judge_replays: int,
        max_workers: int = 10,
        bootstrap_replicates: int = 0,
        bootstrap_seed: int = 0,
    ) -> CalibrationResult:
        """Asynchronously collect reports, fit weights, and return the result.

        Args:
            experiment: Existing experiment containing cases and evaluators.
            task: Synchronous or asynchronous task function.
            agent_runs: Number of independent task executions per case.
            judge_replays: Number of evaluator passes over each fixed output.
            max_workers: Maximum parallel case workers used by the experiment.
            bootstrap_replicates: Number of hierarchical bootstrap samples.
                Zero disables bootstrap uncertainty estimates.
            bootstrap_seed: Random seed used for reproducible resampling.

        Returns:
            The weighted score, raw reports, and calibration snapshot.
        """
        _validate_bootstrap_options(bootstrap_replicates, bootstrap_seed)
        _validate_repetition_counts(agent_runs, judge_replays)
        if len(experiment.cases) < 2:
            raise ValueError("calibration requires at least two cases")
        calibration_context = _build_calibration_context(
            experiment.cases,
            experiment.evaluators,
            self._step_sizes,
        )
        _validate_calibration_compatibility(
            expected_design=self._state.calibration_design,
            expected_context=self._state.calibration_context,
            actual_design=(agent_runs, judge_replays),
            actual_context=calibration_context,
            step_sizes=self._step_sizes,
            operation="evaluation",
        )
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        _validate_evaluator_names(
            {evaluator.get_name() for evaluator in experiment.evaluators},
            set(self._step_sizes),
        )

        reports_by_run = await _collect_repeated_reports(
            experiment,
            task,
            agent_runs=agent_runs,
            judge_replays=judge_replays,
            max_workers=max_workers,
            calibration_context=calibration_context,
        )

        self.fit(
            reports_by_run,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        return CalibrationResult(
            score=self.score(reports_by_run),
            reports_by_run=[
                [report.model_copy(deep=True) for report in replay_reports] for replay_reports in reports_by_run
            ],
            weights=self.weights,
            algorithm_id=self.algorithm_id,
            algorithm_version=self.algorithm_version,
            model_parameters=self.model_parameters,
            status=self.status,
            calibration_context=self.calibration_context,
        )

    def fit(
        self,
        reports_by_run: list[list[EvaluationReport]],
        *,
        bootstrap_replicates: int = 0,
        bootstrap_seed: int = 0,
    ) -> Self:
        """Fit the configured algorithm from repeated evaluation reports.

        Args:
            reports_by_run: Reports grouped by agent run and judge replay.
            bootstrap_replicates: Number of hierarchical bootstrap samples.
                Zero disables bootstrap uncertainty estimates.
            bootstrap_seed: Random seed used for reproducible resampling.

        Returns:
            This calibrator with its fitted scoring model updated.

        Raises:
            ValueError: If reports are incomplete or incompatible, or bootstrap
                options are invalid. Reports without a calibration context are
                compared using evaluator names and their available stable case
                fields.
        """
        _validate_bootstrap_options(bootstrap_replicates, bootstrap_seed)
        dataset = _prepare_calibration_dataset(
            reports_by_run,
            configured_evaluators=set(self._step_sizes),
            step_sizes=self._step_sizes,
            expected_design=self._state.calibration_design,
            expected_context=self._state.calibration_context,
            expected_structure=self._state.calibration_structure,
            expected_structure_fingerprint=self._state.fit_structure_fingerprint,
            operation="fit",
        )
        fitted_state = _fit_dataset(
            dataset,
            self._step_sizes,
            bootstrap_replicates,
            bootstrap_seed,
        )
        data = _build_algorithm_data(dataset, fitted_state, self._step_sizes)
        fitted_model = _validate_fitted_model(
            self._algorithm.fit(data),
            self._algorithm,
            set(self._step_sizes),
        )
        _score_dataset(fitted_model, dataset)
        self._state = fitted_state
        self._model = fitted_model
        self._profile = _profile_from_state(
            fitted_state,
            fitted_model,
            self._step_sizes,
            self.weighting_method,
        )
        return self

    def score(self, reports_by_run: list[list[EvaluationReport]]) -> float:
        """Apply the fitted model to compatible evaluation reports.

        Args:
            reports_by_run: Reports grouped by agent run and judge replay. One
                or more cases, runs, and replays are accepted.

        Returns:
            Predicted composite score.

        Raises:
            ValueError: If reports are internally unbalanced or their evaluator
                set or configuration is incompatible with the fitted model.
        """
        dataset = self._scoring_dataset(reports_by_run)
        return _score_dataset(self._model, dataset)

    def score_candidates(
        self,
        reports_by_run: list[list[EvaluationReport]],
    ) -> dict[WeightingMethod, float | None]:
        """Score compatible reports under every candidate rule.

        Args:
            reports_by_run: Reports grouped by agent run and judge replay. One
                or more cases, runs, and replays are accepted.

        Returns:
            Composite score for each candidate weighting rule.

        Raises:
            ValueError: If reports are internally unbalanced or their evaluator
                set or configuration is incompatible with the fitted model.
        """
        dataset = self._scoring_dataset(reports_by_run)
        evaluator_scores = dataset.evaluator_scores()
        return {
            method: None if weights is None else _weighted_score(evaluator_scores, weights)
            for method, weights in self._state.candidate_weights.items()
        }

    def _scoring_dataset(
        self,
        reports_by_run: list[list[EvaluationReport]],
    ) -> _CalibrationDataset:
        return _prepare_calibration_dataset(
            reports_by_run,
            configured_evaluators=set(self._step_sizes),
            step_sizes=self._step_sizes,
            expected_design=self._state.calibration_design,
            expected_context=self._state.calibration_context,
            expected_structure=self._state.calibration_structure,
            expected_structure_fingerprint=self._state.fit_structure_fingerprint,
            operation="scoring",
            scoring=True,
        )


def _profile_from_state(
    state: _CalibrationState,
    model: FittedCalibrationModel,
    step_sizes: dict[str, float],
    weighting_method: WeightingMethod | None,
) -> CalibrationProfile:
    """Convert fitted runtime state and model parameters into a profile."""
    if state.calibration_design is None or state.case_count is None:
        raise ValueError("calibrator must be fitted before creating a calibration profile")
    agent_runs, judge_replays = state.calibration_design
    return CalibrationProfile.create(
        algorithm_id=model.algorithm_id,
        algorithm_version=model.algorithm_version,
        model_parameters=_normalized_model_parameters(model),
        weighting_method=weighting_method,
        weights=None if model.weights is None else dict(model.weights),
        status=state.status,
        step_sizes=dict(step_sizes),
        variance_components={name: dict(components) for name, components in state.variance_components.items()},
        repeatabilities=dict(state.repeatabilities),
        bootstrap=None if state.bootstrap_result is None else asdict(state.bootstrap_result),
        case_count=state.case_count,
        agent_runs=agent_runs,
        judge_replays=judge_replays,
        calibration_context=copy.deepcopy(state.calibration_context),
        fit_structure_fingerprint=state.fit_structure_fingerprint,
    )


def _state_from_profile(profile: CalibrationProfile) -> _CalibrationState:
    """Restore measurement diagnostics and compatibility state from a profile."""
    bootstrap_result = None if profile.bootstrap is None else _BootstrapResult(**profile.bootstrap.model_dump())
    candidate_weights = {
        method: _compute_weight_vector(
            method,
            list(profile.step_sizes),
            profile.repeatabilities,
            profile.step_sizes,
        )
        for method in WEIGHTING_METHODS
    }
    return _CalibrationState(
        variance_components={name: dict(components) for name, components in profile.variance_components.items()},
        repeatabilities=dict(profile.repeatabilities),
        candidate_weights=candidate_weights,
        bootstrap_result=bootstrap_result,
        calibration_design=(profile.agent_runs, profile.judge_replays),
        calibration_context=copy.deepcopy(profile.calibration_context),
        case_count=profile.case_count,
        calibration_structure=None,
        fit_structure_fingerprint=profile.fit_structure_fingerprint,
        status=profile.status,
    )


def _normalized_model_parameters(model: FittedCalibrationModel) -> dict[str, Any]:
    """Return deterministic JSON-compatible parameters for a fitted model."""
    normalized = _normalize_configuration_value(dict(model.parameters))
    if not isinstance(normalized, dict):
        raise ValueError("fitted model parameters must normalize to a mapping")
    return normalized


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
