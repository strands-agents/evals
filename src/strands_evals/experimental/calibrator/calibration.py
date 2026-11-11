"""Experimental evaluator calibration facade."""

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, field_serializer
from typing_extensions import Self

from ...case import Case
from ...experiment import Experiment
from ...types.evaluation import EvaluationData, InputT, OutputT
from ...types.evaluation_report import EvaluationReport
from ._bootstrap import (
    BootstrapDiagnostics,
    compute_candidate_weights,
    run_bootstrap,
    validate_bootstrap_options,
)
from ._calibration import (
    CalibrationStatus,
    _CalibrationDataset,
    _CalibrationState,
    _extract_case_signatures_from_cases,
    _fit_dataset,
    _initial_calibration_state,
    _normalize_configuration_value,
    _prepare_calibration_dataset,
    _validate_calibration_compatibility,
    _validate_evaluator_names,
    _validate_repetition_counts,
)
from ._collection import CalibrationRunError, _collect_repeated_reports
from .algorithm import (
    CalibrationAlgorithm,
    CalibrationData,
    CalibrationModel,
    VarianceSource,
    WeightingMethod,
    _equal_weights,
)
from .equal_weighting import EqualWeightingAlgorithm, EqualWeightingModel


class CalibrationResult(BaseModel):
    """Point-in-time calibration result for in-memory inspection.

    Mutating `reports_by_run` does not recompute `score` or update the
    fitted calibrator. This model is not a stable persistence format, and custom
    `EvaluationReport` subclasses are not guaranteed to survive serialization
    and deserialization.
    """

    score: float
    reports_by_run: list[list[EvaluationReport]]
    weights: dict[str, float] | None
    algorithm_identity: str
    model_parameters: dict[str, Any]
    calibration_design: dict[str, int] | None
    variance_components: dict[str, dict[VarianceSource, float]]
    repeatabilities: dict[str, float]
    bootstrap: BootstrapDiagnostics | None
    status: CalibrationStatus

    @field_serializer("bootstrap")
    def _serialize_bootstrap(self, bootstrap: BootstrapDiagnostics | None) -> dict[str, object] | None:
        """Serialize immutable bootstrap mappings without Pydantic warnings."""
        return None if bootstrap is None else bootstrap.to_dict()


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
    Refitting the same instance requires the original cases and repetition
    design. Within each agent run, every replay must judge the same fixed
    output.

    Direct calls to `score` may use new cases and different positive run or
    replay counts as long as the evaluator set matches the fitted model.
    `run_evaluations` collects and fits, so reusing it after fitting requires
    the original cases and repetition design.
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
            TypeError: If `algorithm` does not implement `CalibrationAlgorithm`.
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
        default_algorithm = EqualWeightingAlgorithm()
        self._algorithm = _validate_algorithm(default_algorithm if algorithm is None else algorithm)
        weight = 1.0 / len(self._step_sizes)
        self._model: CalibrationModel = EqualWeightingModel(
            evaluator_weights={name: weight for name in self._step_sizes},
        )
        self._algorithm_identity = default_algorithm.identity
        self._state = _initial_calibration_state()
        self._candidate_weights: dict[WeightingMethod, dict[str, float] | None] = {
            "equal": _equal_weights(tuple(self._step_sizes)),
            "repeatability": None,
            "inverse_step": None,
            "inverse_sqrt_step": None,
        }
        self._bootstrap: BootstrapDiagnostics | None = None

    @property
    def algorithm_identity(self) -> str:
        """Return the active scoring model's stable, versioned identity."""
        return self._algorithm_identity

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
    def variance_components(self) -> dict[str, dict[VarianceSource, float]]:
        """Return corrected non-negative variance estimates by evaluator."""
        return {name: dict(components) for name, components in self._state.variance_components.items()}

    @property
    def repeatabilities(self) -> dict[str, float]:
        """Return replay-mean reliability at the fitted judge replay count.

        The estimate depends on the number of judge replays. A dataset with no
        observed signal or noise reports `0.0` conservatively.
        """
        return dict(self._state.repeatabilities)

    @property
    def candidate_weights(self) -> dict[WeightingMethod, dict[str, float] | None]:
        """Return weight vectors for every diagnostic candidate rule."""
        return {
            method: None if weights is None else dict(weights) for method, weights in self._candidate_weights.items()
        }

    @property
    def bootstrap(self) -> BootstrapDiagnostics | None:
        """Return diagnostics from the latest bootstrap fit, if requested."""
        return self._bootstrap

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
            task: Task function evaluated by the experiment. Returned values
                must support deep copying so judge replays remain isolated.
            agent_runs: Number of independent task executions per case.
            judge_replays: Number of evaluator passes over each fixed output.
            max_workers: Maximum parallel case workers used by the experiment.
            bootstrap_replicates: Number of hierarchical bootstrap samples.
                Zero disables bootstrap diagnostics.
            bootstrap_seed: Random seed used for reproducible resampling.

        Returns:
            The composite score, raw reports, and fitted calibration snapshot.

        Raises:
            CalibrationRunError: If collection or fitting fails. The error
                retains every report collected before the failure.
            ValueError: If pre-flight validation fails or `task` is async.
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
        """Collect repeated reports, fit the algorithm, and return a snapshot.

        Args:
            experiment: Existing experiment containing cases and evaluators.
            task: Synchronous or asynchronous task function. Returned values
                must support deep copying so judge replays remain isolated.
            agent_runs: Number of independent task executions per case.
            judge_replays: Number of evaluator passes over each fixed output.
            max_workers: Maximum parallel case workers used by the experiment.
            bootstrap_replicates: Number of hierarchical bootstrap samples.
                Zero disables bootstrap diagnostics.
            bootstrap_seed: Random seed used for reproducible resampling.

        Returns:
            The composite score, raw reports, and fitted calibration snapshot.

        Raises:
            CalibrationRunError: If collection or fitting fails. The error
                retains every report collected before the failure.
            ValueError: If pre-flight validation fails.
        """
        validate_bootstrap_options(bootstrap_replicates, bootstrap_seed)
        _validate_repetition_counts(agent_runs, judge_replays)
        cases = experiment.cases
        if len(cases) < 2:
            raise ValueError("calibration requires at least two cases")
        case_names: list[str] = []
        for case in cases:
            if not isinstance(case.name, str) or not case.name:
                raise ValueError("calibration requires every case to have a unique, non-empty name")
            case_names.append(case.name)
        if len(set(case_names)) != len(case_names):
            raise ValueError("calibration requires every case to have a unique, non-empty name")
        if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        evaluator_names = {evaluator.get_name() for evaluator in experiment.evaluators}
        _validate_evaluator_names(evaluator_names, set(self._step_sizes))
        report_case_fields = EvaluationData.model_fields.keys()
        case_signatures = _extract_case_signatures_from_cases(
            {key: value for key, value in case.model_dump().items() if key in report_case_fields} for case in cases
        )
        _validate_calibration_compatibility(
            expected_design=self._state.calibration_design,
            actual_design=(agent_runs, judge_replays),
            operation="evaluation",
        )
        if self._state.calibration_structure is not None:
            fitted_cases = {name for name, _ in self._state.calibration_structure[1]}
            actual_cases = set(case_names)
            if actual_cases != fitted_cases:
                raise ValueError(
                    "evaluation cases must match the fitted calibration cases; "
                    f"missing={sorted(fitted_cases - actual_cases)}, "
                    f"unexpected={sorted(actual_cases - fitted_cases)}"
                )
            if case_signatures != self._state.calibration_structure[1]:
                raise ValueError("evaluation case definitions must match the fitted calibration cases")

        reports_by_run = await _collect_repeated_reports(
            experiment,
            task,
            agent_runs=agent_runs,
            judge_replays=judge_replays,
            max_workers=max_workers,
        )
        try:
            self.fit(
                reports_by_run,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
            score = self.score(reports_by_run)
        except Exception as error:
            raise CalibrationRunError(
                f"calibration fitting failed: {error}",
                reports_by_run,
            ) from error
        return CalibrationResult(
            score=score,
            reports_by_run=reports_by_run,
            weights=self.weights,
            algorithm_identity=self.algorithm_identity,
            model_parameters=self.model_parameters,
            calibration_design=self.calibration_design,
            variance_components=self.variance_components,
            repeatabilities=self.repeatabilities,
            bootstrap=self.bootstrap,
            status=self.status,
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
                Zero disables bootstrap diagnostics.
            bootstrap_seed: Random seed used for reproducible resampling.

        Returns:
            This calibrator with its fitted scoring model updated.

        Raises:
            ValueError: If reports are incomplete or structurally incompatible.
        """
        validate_bootstrap_options(bootstrap_replicates, bootstrap_seed)
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
            set(self._step_sizes),
        )
        _score_dataset(fitted_model, dataset)
        candidate_weights = compute_candidate_weights(
            dataset,
            self._step_sizes,
            fitted_state.repeatabilities,
        )
        bootstrap = (
            run_bootstrap(
                dataset,
                self._step_sizes,
                bootstrap_replicates,
                bootstrap_seed,
            )
            if bootstrap_replicates
            else None
        )
        self._state = fitted_state
        self._model = fitted_model
        self._algorithm_identity = self._algorithm.identity
        self._candidate_weights = candidate_weights
        self._bootstrap = bootstrap
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

    def score_candidates(
        self,
        reports_by_run: list[list[EvaluationReport]],
    ) -> dict[WeightingMethod, float | None]:
        """Score reports under every diagnostic candidate weighting rule."""
        dataset = _prepare_calibration_dataset(
            reports_by_run,
            configured_evaluators=set(self._step_sizes),
            expected_design=self._state.calibration_design,
            expected_structure=self._state.calibration_structure,
            operation="scoring",
            scoring=True,
        )
        evaluator_scores = dataset.evaluator_scores()
        return {
            method: (
                None
                if weights is None
                else min(
                    max(sum(weights[name] * evaluator_scores[name] for name in evaluator_scores), 0.0),
                    1.0,
                )
            )
            for method, weights in self._candidate_weights.items()
        }


def _build_algorithm_data(
    dataset: _CalibrationDataset,
    state: _CalibrationState,
    step_sizes: dict[str, float],
) -> CalibrationData:
    """Build an immutable public view of one validated calibration dataset."""
    return CalibrationData(
        observations=dataset.observations(),
        evaluator_names=dataset.evaluator_names,
        case_names=dataset.case_names,
        agent_runs=dataset.run_count,
        judge_replays=dataset.replay_count,
        step_sizes=step_sizes,
        variance_components=state.variance_components,
        repeatabilities=state.repeatabilities,
    )


def _validate_algorithm(algorithm: object) -> CalibrationAlgorithm:
    """Validate the stable identity and fitting surface of an algorithm."""
    if not isinstance(algorithm, CalibrationAlgorithm):
        raise TypeError("algorithm must be a CalibrationAlgorithm")
    if not isinstance(algorithm.identity, str) or not algorithm.identity:
        raise ValueError("algorithm identity must be a non-empty string")
    return algorithm


def _validate_fitted_model(
    model: object,
    evaluator_names: set[str],
) -> CalibrationModel:
    """Validate a fitted model's parameters and optional weights."""
    if not isinstance(model, CalibrationModel):
        raise TypeError("fitted model must be a CalibrationModel")
    parameters = model.parameters
    if not isinstance(parameters, Mapping):
        raise ValueError("fitted model parameters must be a mapping")
    try:
        normalized_parameters = _normalize_configuration_value(dict(parameters))
    except ValueError as error:
        raise ValueError("fitted model parameters must be deterministic and JSON-compatible") from error
    if not isinstance(normalized_parameters, dict):
        raise ValueError("fitted model parameters must normalize to a mapping")
    weights = model.weights
    if weights is None:
        return model
    if not isinstance(weights, Mapping):
        raise ValueError("fitted model weights must be a mapping or None")
    if set(weights) != evaluator_names:
        raise ValueError("fitted model weights must match the configured evaluators")
    if any(
        not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight)
        for weight in weights.values()
    ):
        raise ValueError("fitted model weights must be finite numbers")
    return model


def _predict_model(
    model: CalibrationModel,
    evaluator_scores: dict[str, float],
) -> float:
    """Predict and validate a score from a fitted calibration model."""
    score = model.predict(MappingProxyType(dict(evaluator_scores)))
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
        raise ValueError("fitted model prediction must be a finite number")
    if not -1e-9 <= score <= 1.0 + 1e-9:
        raise ValueError(f"fitted model prediction must be between 0 and 1, got {score}")
    return min(max(float(score), 0.0), 1.0)


def _score_dataset(
    model: CalibrationModel,
    dataset: _CalibrationDataset,
) -> float:
    """Average model predictions across case-run samples."""
    predictions = [_predict_model(model, dict(sample.evaluator_scores)) for sample in dataset.samples()]
    return sum(predictions) / len(predictions)
