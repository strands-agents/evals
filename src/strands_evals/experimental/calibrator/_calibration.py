"""Private data and statistical helpers for evaluator calibration."""

import hashlib
import inspect
import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from pydantic import BaseModel
from strands.models.model import Model
from typing_extensions import Any, Literal

from ...case import Case
from ...evaluators.evaluator import Evaluator
from ...experiment import Experiment
from ...types.evaluation import EvaluationData
from ...types.evaluation_report import EvaluationReport
from .algorithm import (
    WEIGHTING_METHODS,
    CalibrationExample,
    CalibrationObservation,
    WeightingMethod,
    _compute_weight_vector,
    _equal_weights,
)

_EPSILON = 1e-15
_ERROR_REASON_PREFIXES = ("An error occurred:", "Evaluator error:")
_CALIBRATION_CONTEXT_VERSION = "1"
_RUNTIME_CASE_FIELDS = {
    "actual_environment_state",
    "actual_interactions",
    "actual_output",
    "actual_trajectory",
    "evaluator",
    "evaluator_type",
    "session_id",
}

CalibrationStatus = Literal["uncalibrated", "equal_weight_fallback", "calibrated"]

_BootstrapSummary = dict[str, float | int | None]
_ParsedReports = list[list[dict[str, dict[str, float]]]]
_CaseSignatures = tuple[tuple[str, str], ...]
_CalibrationStructure = tuple[tuple[str, ...], _CaseSignatures]


@dataclass(frozen=True)
class _EvaluatorStatistics:
    """Variance components and judge repeatability for one evaluator."""

    variance_components: dict[str, float]
    repeatability: float


@dataclass(frozen=True)
class _BootstrapResult:
    """Summarized hierarchical bootstrap output."""

    replicates: int
    seed: int
    repeatability_intervals: dict[str, _BootstrapSummary]
    candidate_weight_intervals: dict[WeightingMethod, dict[str, _BootstrapSummary]]
    composite_score_intervals: dict[WeightingMethod, _BootstrapSummary]


@dataclass(frozen=True)
class _CalibrationState:
    """Complete immutable-by-replacement state for one calibrator."""

    variance_components: dict[str, dict[str, float]]
    repeatabilities: dict[str, float]
    candidate_weights: dict[WeightingMethod, dict[str, float] | None]
    bootstrap_result: _BootstrapResult | None
    calibration_design: tuple[int, int] | None
    calibration_context: dict[str, str] | None
    calibration_structure: _CalibrationStructure | None
    status: CalibrationStatus


@dataclass(frozen=True)
class _CalibrationDataset:
    """Balanced case-by-run-by-replay evaluator observations.

    `parsed` uses the following shape:

    `parsed[agent_run][judge_replay][evaluator_name][case_name] -> score`
    """

    parsed: _ParsedReports
    evaluator_names: tuple[str, ...]
    case_names: tuple[str, ...]
    case_signatures: _CaseSignatures
    run_count: int
    replay_count: int
    calibration_context: dict[str, str] | None

    @property
    def structure(self) -> _CalibrationStructure:
        """Return the evaluator names and stable case definitions."""
        return self.evaluator_names, self.case_signatures

    @classmethod
    def from_reports(
        cls,
        reports_by_run: list[list[EvaluationReport]],
        configured_evaluators: set[str],
        *,
        scoring: bool = False,
    ) -> "_CalibrationDataset":
        """Parse and validate reports as a balanced fit or scoring dataset."""
        if not isinstance(reports_by_run, list):
            raise ValueError(
                "reports_by_run must be a list of agent-run report lists; "
                "a single EvaluationReport does not contain a repeated scoring design"
            )
        if not reports_by_run:
            raise ValueError("reports_by_run must contain at least one agent run")
        if any(not isinstance(reports, list) for reports in reports_by_run):
            raise ValueError("each agent run must be represented by a list of judge replay reports")
        if not reports_by_run[0]:
            raise ValueError("each agent run must contain at least one judge replay")
        if any(not isinstance(report, EvaluationReport) for reports in reports_by_run for report in reports):
            raise ValueError("every judge replay must be an EvaluationReport")

        contexts = [report.calibration_context for reports in reports_by_run for report in reports]
        present_contexts = [context for context in contexts if context is not None]
        if present_contexts and len(present_contexts) != len(contexts):
            raise ValueError("calibration reports must either all include a calibration context or all omit it")
        if present_contexts and any(context != present_contexts[0] for context in present_contexts[1:]):
            raise ValueError("calibration reports contain incompatible calibration contexts")
        calibration_context = None if not present_contexts else dict(present_contexts[0])

        replay_count = len(reports_by_run[0])
        if any(len(reports) != replay_count for reports in reports_by_run):
            raise ValueError("every agent run must contain the same number of judge replays")

        parsed = [[_extract_report_scores(report) for report in reports] for reports in reports_by_run]
        first = parsed[0][0]
        evaluator_names = tuple(sorted(first))
        _validate_evaluator_names(set(evaluator_names), configured_evaluators)
        case_names = tuple(sorted(next(iter(first.values()))))
        case_signatures = _extract_case_signatures(reports_by_run[0][0])

        for run_index, replay_reports in enumerate(parsed):
            for replay_index, report_scores in enumerate(replay_reports):
                if set(report_scores) != set(evaluator_names):
                    raise ValueError(f"run {run_index} replay {replay_index} has a different evaluator set")
                for evaluator_name in evaluator_names:
                    if set(report_scores[evaluator_name]) != set(case_names):
                        raise ValueError(
                            f"run {run_index} replay {replay_index} evaluator "
                            f"{evaluator_name!r} has a different case set"
                        )
                if _extract_case_signatures(reports_by_run[run_index][replay_index]) != case_signatures:
                    raise ValueError(f"run {run_index} replay {replay_index} has different stable case definitions")

        run_count = len(parsed)
        if not scoring:
            _validate_repetition_counts(run_count, replay_count)
            if len(case_names) < 2:
                raise ValueError("calibration requires at least two cases")

        return cls(
            parsed=parsed,
            evaluator_names=evaluator_names,
            case_names=case_names,
            case_signatures=case_signatures,
            run_count=run_count,
            replay_count=replay_count,
            calibration_context=calibration_context,
        )

    def evaluator_scores(self) -> dict[str, float]:
        """Collapse the hierarchy to one suite score per evaluator."""
        evaluator_scores: dict[str, float] = {}
        for evaluator_name in self.evaluator_names:
            case_scores: list[float] = []
            for case_name in self.case_names:
                values = [
                    report_scores[evaluator_name][case_name]
                    for run_reports in self.parsed
                    for report_scores in run_reports
                ]
                case_scores.append(_mean(values))
            evaluator_scores[evaluator_name] = _mean(case_scores)
        return evaluator_scores

    def observations(self) -> tuple[CalibrationObservation, ...]:
        """Flatten the balanced hierarchy for a pluggable algorithm."""
        return tuple(
            CalibrationObservation(
                evaluator_name=evaluator_name,
                case_name=case_name,
                agent_run=run_index,
                judge_replay=replay_index,
                score=report_scores[evaluator_name][case_name],
            )
            for run_index, run_reports in enumerate(self.parsed)
            for replay_index, report_scores in enumerate(run_reports)
            for evaluator_name in self.evaluator_names
            for case_name in self.case_names
        )

    def examples(self) -> tuple[CalibrationExample, ...]:
        """Aggregate judge replays to one feature vector per case and agent run."""
        return tuple(
            CalibrationExample(
                case_name=case_name,
                agent_run=run_index,
                evaluator_scores=MappingProxyType(
                    {
                        evaluator_name: _mean(
                            [report_scores[evaluator_name][case_name] for report_scores in run_reports]
                        )
                        for evaluator_name in self.evaluator_names
                    }
                ),
            )
            for run_index, run_reports in enumerate(self.parsed)
            for case_name in self.case_names
        )

    def bootstrap_sample(self, rng: random.Random) -> "_CalibrationDataset":
        """Resample cases, runs, and replays while preserving evaluator vectors."""
        sampled: _ParsedReports = [
            [{name: {} for name in self.evaluator_names} for _ in range(self.replay_count)]
            for _ in range(self.run_count)
        ]
        sampled_case_names: list[str] = []
        sampled_case_signatures: list[tuple[str, str]] = []
        signature_by_case = dict(self.case_signatures)

        for case_slot in range(len(self.case_names)):
            source_case = rng.choice(self.case_names)
            sampled_case = f"{source_case}#bootstrap-case-{case_slot}"
            sampled_case_names.append(sampled_case)
            sampled_case_signatures.append((sampled_case, signature_by_case[source_case]))
            for run_slot in range(self.run_count):
                source_run = rng.randrange(self.run_count)
                for replay_slot in range(self.replay_count):
                    source_replay = rng.randrange(self.replay_count)
                    source_scores = self.parsed[source_run][source_replay]
                    for evaluator_name in self.evaluator_names:
                        sampled[run_slot][replay_slot][evaluator_name][sampled_case] = source_scores[evaluator_name][
                            source_case
                        ]

        return _CalibrationDataset(
            parsed=sampled,
            evaluator_names=self.evaluator_names,
            case_names=tuple(sampled_case_names),
            case_signatures=tuple(sampled_case_signatures),
            run_count=self.run_count,
            replay_count=self.replay_count,
            calibration_context=(None if self.calibration_context is None else dict(self.calibration_context)),
        )


class _InMemoryEvaluationDataStore:
    """Task-output cache used to replay evaluators on fixed outputs."""

    def __init__(self) -> None:
        self._results: dict[str, EvaluationData] = {}

    def load(self, case_name: str) -> EvaluationData | None:
        result = self._results.get(case_name)
        return result.model_copy(deep=True) if result is not None else None

    def save(self, case_name: str, result: EvaluationData) -> None:
        self._results[case_name] = result.model_copy(deep=True)


async def _collect_repeated_reports(
    experiment: Experiment[Any, Any],
    task: Callable[..., Any],
    *,
    agent_runs: int,
    judge_replays: int,
    max_workers: int,
    calibration_context: dict[str, str],
) -> list[list[EvaluationReport]]:
    """Run agent attempts and replay evaluators against each fixed output."""
    reports_by_run: list[list[EvaluationReport]] = []
    for _ in range(agent_runs):
        task_store = _InMemoryEvaluationDataStore()
        replay_reports: list[EvaluationReport] = []
        for _ in range(judge_replays):
            report = await experiment.run_evaluations_async(
                task,
                max_workers=max_workers,
                evaluation_data_store=task_store,
            )
            report.calibration_context = dict(calibration_context)
            _extract_report_scores(report)
            replay_reports.append(report)
        reports_by_run.append(replay_reports)
    return reports_by_run


def _prepare_calibration_dataset(
    reports_by_run: list[list[EvaluationReport]],
    *,
    configured_evaluators: set[str],
    step_sizes: dict[str, float],
    expected_design: tuple[int, int] | None,
    expected_context: dict[str, str] | None,
    expected_structure: _CalibrationStructure | None,
    operation: str,
    scoring: bool = False,
) -> _CalibrationDataset:
    """Parse reports and enforce the calibrator's compatibility invariants."""
    dataset = _CalibrationDataset.from_reports(
        reports_by_run,
        configured_evaluators,
        scoring=scoring,
    )
    _validate_calibration_compatibility(
        expected_design=expected_design,
        expected_context=expected_context,
        actual_design=(dataset.run_count, dataset.replay_count),
        actual_context=dataset.calibration_context,
        step_sizes=step_sizes,
        operation=operation,
        expected_structure=expected_structure,
        actual_structure=dataset.structure,
        scoring=scoring,
    )
    return dataset


def _build_calibration_context(
    cases: list[Case],
    evaluators: list[Evaluator],
    step_sizes: dict[str, float],
) -> dict[str, str]:
    """Fingerprint the static experiment inputs that calibration depends on."""
    case_payloads = [
        _normalize_configuration_value(
            case.model_dump(exclude={"session_id"}),
        )
        for case in cases
    ]
    case_payloads.sort(key=_canonical_json)

    evaluator_payloads = [_evaluator_configuration(evaluator) for evaluator in evaluators]
    evaluator_payloads.sort(key=_canonical_json)

    components = {
        "dataset": _fingerprint(case_payloads),
        "evaluators": _fingerprint(evaluator_payloads),
        "step_sizes": _step_size_fingerprint(step_sizes),
    }
    return {
        "version": _CALIBRATION_CONTEXT_VERSION,
        **components,
        "overall": _fingerprint(components),
    }


def _evaluator_configuration(evaluator: Evaluator) -> dict[str, Any]:
    """Extract constructor-backed evaluator configuration without runtime state."""
    parameters: dict[str, Any] = {}
    signature = inspect.signature(evaluator.__class__.__init__)
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if name == "name":
            parameters[name] = evaluator.get_name()
            continue
        if hasattr(evaluator, name):
            value = getattr(evaluator, name)
        elif hasattr(evaluator, f"_{name}"):
            value = getattr(evaluator, f"_{name}")
        else:
            raise ValueError(
                f"cannot fingerprint evaluator {evaluator.get_name()!r}: "
                f"constructor parameter {name!r} is not stored as {name!r} or _{name!r}"
            )

        if name == "model":
            model_config: dict[str, Any] = {
                "model_id": evaluator._get_model_id(value),
            }
            if isinstance(value, Model):
                model_config["config"] = _normalize_configuration_value(value.config)
            parameters[name] = model_config
        else:
            parameters[name] = _normalize_configuration_value(value)

    return {
        "implementation": (f"{evaluator.__class__.__module__}.{evaluator.__class__.__qualname__}"),
        "name": evaluator.get_name(),
        "parameters": parameters,
    }


def _normalize_configuration_value(value: Any) -> Any:
    """Convert configuration values to deterministic JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"float": repr(value)}
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Enum):
        return _normalize_configuration_value(value.value)
    if isinstance(value, BaseModel):
        return _normalize_configuration_value(value.model_dump())
    if isinstance(value, dict):
        return {
            str(key): _normalize_configuration_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_configuration_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_configuration_value(item) for item in value]
        return sorted(normalized, key=_canonical_json)
    if callable(value):
        return {
            "callable": (
                f"{getattr(value, '__module__', value.__class__.__module__)}."
                f"{getattr(value, '__qualname__', value.__class__.__qualname__)}"
            )
        }
    value_type = f"{value.__class__.__module__}.{value.__class__.__qualname__}"
    raise ValueError(
        f"unsupported configuration value type {value_type!r}; use JSON-compatible values or a Pydantic model"
    )


def _fingerprint(value: Any) -> str:
    """Hash normalized configuration data."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _step_size_fingerprint(step_sizes: dict[str, float]) -> str:
    """Fingerprint a normalized evaluator step-size registry."""
    return _fingerprint({name: step_sizes[name] for name in sorted(step_sizes)})


def _validate_calibration_compatibility(
    *,
    expected_design: tuple[int, int] | None,
    expected_context: dict[str, str] | None,
    actual_design: tuple[int, int],
    actual_context: dict[str, str] | None,
    step_sizes: dict[str, float],
    operation: str,
    expected_structure: _CalibrationStructure | None = None,
    actual_structure: _CalibrationStructure | None = None,
    scoring: bool = False,
) -> None:
    """Require compatible evaluators while allowing new scoring samples."""
    if not scoring and expected_design is not None and actual_design != expected_design:
        expected_runs, expected_replays = expected_design
        actual_runs, actual_replays = actual_design
        raise ValueError(
            f"{operation} design must match the fitted calibration design; "
            f"expected agent_runs={expected_runs}, judge_replays={expected_replays}; "
            f"got agent_runs={actual_runs}, judge_replays={actual_replays}"
        )

    if expected_structure is not None and actual_structure is not None:
        changed_components: list[str] = []
        if actual_structure[0] != expected_structure[0]:
            changed_components.append("evaluators")
        if not scoring and actual_structure[1] != expected_structure[1]:
            changed_components.append("cases")
        if changed_components:
            raise ValueError(
                f"{operation} report structure is incompatible with the fitted calibration; "
                f"changed_components={changed_components}"
            )

    if actual_context is not None:
        if actual_context.get("version") != _CALIBRATION_CONTEXT_VERSION:
            raise ValueError(f"{operation} calibration context has an unsupported version")
        expected_step_sizes = _step_size_fingerprint(step_sizes)
        if actual_context.get("step_sizes") != expected_step_sizes:
            raise ValueError(f"{operation} calibration context does not match the configured evaluator step sizes")

    if expected_design is None:
        return
    if expected_context is None and actual_context is None:
        return
    if expected_context is None or actual_context is None:
        raise ValueError(
            f"{operation} calibration context must match the fitted context; "
            "cannot mix fingerprinted reports with legacy reports"
        )
    compared_components = {"evaluators", "step_sizes"} if scoring else {"dataset", "evaluators", "step_sizes"}
    changed_components = sorted(
        component
        for component in compared_components
        if actual_context.get(component) != expected_context.get(component)
    )
    if changed_components:
        raise ValueError(
            f"{operation} calibration context is incompatible with the fitted context; "
            f"changed_components={changed_components}"
        )


def _canonical_json(value: Any) -> str:
    """Serialize normalized values deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _extract_report_scores(report: EvaluationReport) -> dict[str, dict[str, float]]:
    """Pivot one row-oriented report into evaluator-by-case scores."""
    if len(report.cases) != len(report.scores):
        raise ValueError("report cases and scores must have equal lengths")

    scores: dict[str, dict[str, float]] = {}
    for index, (case, raw_score) in enumerate(zip(report.cases, report.scores, strict=True)):
        evaluator_name = case.get("evaluator")
        case_name = case.get("name")
        if not isinstance(evaluator_name, str) or not evaluator_name:
            raise ValueError(f"report row {index} is missing an evaluator name")
        if not isinstance(case_name, str) or not case_name:
            raise ValueError(f"report row {index} is missing a case name")
        if not math.isfinite(raw_score) or not 0.0 <= raw_score <= 1.0:
            raise ValueError(f"report row {index} score must be between 0 and 1")

        reason = report.reasons[index] if index < len(report.reasons) else ""
        if reason.startswith(_ERROR_REASON_PREFIXES):
            raise ValueError(
                f"calibration cannot use failed evaluation rows: "
                f"case={case_name!r}, evaluator={evaluator_name!r}, reason={reason!r}"
            )

        outputs = report.detailed_results[index] if index < len(report.detailed_results) else []
        if index < len(report.detailed_results) and not EvaluationReport.is_applicable(outputs):
            raise ValueError(
                f"calibration does not yet support not-applicable rows: "
                f"case={case_name!r}, evaluator={evaluator_name!r}"
            )

        evaluator_scores = scores.setdefault(evaluator_name, {})
        if case_name in evaluator_scores:
            raise ValueError(
                f"report contains duplicate case/evaluator row: case={case_name!r}, evaluator={evaluator_name!r}"
            )
        evaluator_scores[case_name] = float(raw_score)

    if not scores:
        raise ValueError("report must contain at least one score")

    case_sets = {frozenset(evaluator_scores) for evaluator_scores in scores.values()}
    if len(case_sets) != 1:
        raise ValueError("every evaluator in a report must score the same cases")
    return scores


def _extract_case_signatures(report: EvaluationReport) -> _CaseSignatures:
    """Fingerprint stable case fields while ignoring run and evaluator output."""
    signatures: dict[str, str] = {}
    for index, case in enumerate(report.cases):
        case_name = case.get("name")
        if not isinstance(case_name, str) or not case_name:
            raise ValueError(f"report row {index} is missing a case name")

        stable_case = {key: value for key, value in case.items() if key not in _RUNTIME_CASE_FIELDS}
        signature = _fingerprint(_normalize_configuration_value(stable_case))
        existing = signatures.setdefault(case_name, signature)
        if existing != signature:
            raise ValueError(f"report contains inconsistent definitions for case {case_name!r}")

    return tuple(sorted(signatures.items()))


def _estimate_statistics(dataset: _CalibrationDataset, evaluator_name: str) -> _EvaluatorStatistics:
    """Estimate variance components and repeatability across fixed outputs."""
    replay_variances: list[float] = []
    run_means_by_case: dict[str, list[float]] = {case_name: [] for case_name in dataset.case_names}

    for case_name in dataset.case_names:
        for run_reports in dataset.parsed:
            replay_scores = [report_scores[evaluator_name][case_name] for report_scores in run_reports]
            replay_variances.append(_sample_variance(replay_scores))
            run_means_by_case[case_name].append(_mean(replay_scores))

    judge_variance = _mean(replay_variances)
    observed_run_variance = _mean([_sample_variance(run_means) for run_means in run_means_by_case.values()])
    agent_variance_raw = observed_run_variance - judge_variance / dataset.replay_count
    agent_variance = _truncate_variance(agent_variance_raw)

    case_means = [_mean(run_means_by_case[case_name]) for case_name in dataset.case_names]
    observed_case_variance = _sample_variance(case_means)
    case_variance_raw = (
        observed_case_variance
        - agent_variance_raw / dataset.run_count
        - judge_variance / (dataset.run_count * dataset.replay_count)
    )
    case_variance = _truncate_variance(case_variance_raw)

    signal_variance = case_variance + agent_variance
    denominator = signal_variance + judge_variance / dataset.replay_count
    repeatability = 0.0 if denominator <= _EPSILON else signal_variance / denominator
    return _EvaluatorStatistics(
        variance_components={
            "judge": judge_variance,
            "agent": agent_variance,
            "case": case_variance,
        },
        repeatability=repeatability,
    )


def _initial_calibration_state(step_sizes: dict[str, float]) -> _CalibrationState:
    """Return the equal-weight state used before calibration."""
    equal_weights = _equal_weights(tuple(step_sizes))
    return _CalibrationState(
        variance_components={},
        repeatabilities={},
        candidate_weights={
            "equal": dict(equal_weights),
            "repeatability": None,
            "inverse_step": None,
            "inverse_sqrt_step": None,
        },
        bootstrap_result=None,
        calibration_design=None,
        calibration_context=None,
        calibration_structure=None,
        status="uncalibrated",
    )


def _fit_dataset(
    dataset: _CalibrationDataset,
    step_sizes: dict[str, float],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> _CalibrationState:
    """Estimate measurement diagnostics and return a replacement fitted state."""
    variance_components: dict[str, dict[str, float]] = {}
    repeatabilities: dict[str, float] = {}
    for evaluator_name in dataset.evaluator_names:
        statistics = _estimate_statistics(dataset, evaluator_name)
        variance_components[evaluator_name] = statistics.variance_components
        repeatabilities[evaluator_name] = statistics.repeatability

    candidate_weights = {
        method: _compute_weight_vector(
            method,
            dataset.evaluator_names,
            repeatabilities,
            step_sizes,
        )
        for method in WEIGHTING_METHODS
    }
    bootstrap_result = (
        _run_bootstrap(
            dataset,
            step_sizes,
            bootstrap_replicates,
            bootstrap_seed,
        )
        if bootstrap_replicates
        else None
    )
    return _CalibrationState(
        variance_components=variance_components,
        repeatabilities=repeatabilities,
        candidate_weights=candidate_weights,
        bootstrap_result=bootstrap_result,
        calibration_design=(dataset.run_count, dataset.replay_count),
        calibration_context=(None if dataset.calibration_context is None else dict(dataset.calibration_context)),
        calibration_structure=dataset.structure,
        status="uncalibrated",
    )


def _run_bootstrap(
    dataset: _CalibrationDataset,
    step_sizes: dict[str, float],
    replicates: int,
    seed: int,
) -> _BootstrapResult:
    """Run the hierarchical bootstrap and summarize its distributions."""
    rng = random.Random(seed)
    repeatability_distributions: dict[str, list[float | None]] = {name: [] for name in dataset.evaluator_names}
    weight_distributions: dict[WeightingMethod, dict[str, list[float | None]]] = {
        method: {name: [] for name in dataset.evaluator_names} for method in WEIGHTING_METHODS
    }
    score_distributions: dict[WeightingMethod, list[float | None]] = {method: [] for method in WEIGHTING_METHODS}

    for _ in range(replicates):
        sampled = dataset.bootstrap_sample(rng)
        sampled_repeatabilities: dict[str, float] = {}
        for evaluator_name in dataset.evaluator_names:
            statistics = _estimate_statistics(sampled, evaluator_name)
            sampled_repeatabilities[evaluator_name] = statistics.repeatability
            repeatability_distributions[evaluator_name].append(statistics.repeatability)

        evaluator_scores = sampled.evaluator_scores()
        for method in WEIGHTING_METHODS:
            weights = _compute_weight_vector(
                method,
                dataset.evaluator_names,
                sampled_repeatabilities,
                step_sizes,
            )
            for evaluator_name in dataset.evaluator_names:
                weight_distributions[method][evaluator_name].append(
                    None if weights is None else weights[evaluator_name]
                )
            score_distributions[method].append(None if weights is None else _weighted_score(evaluator_scores, weights))

    return _BootstrapResult(
        replicates=replicates,
        seed=seed,
        repeatability_intervals={
            name: _summarize_distribution(values) for name, values in repeatability_distributions.items()
        },
        candidate_weight_intervals={
            method: {name: _summarize_distribution(values) for name, values in weight_distributions[method].items()}
            for method in WEIGHTING_METHODS
        },
        composite_score_intervals={
            method: _summarize_distribution(values) for method, values in score_distributions.items()
        },
    )


def _weighted_score(evaluator_scores: dict[str, float], weights: dict[str, float]) -> float:
    """Combine evaluator-level scores with a normalized weight vector."""
    return sum(weights[name] * evaluator_scores[name] for name in evaluator_scores)


def _validate_bootstrap_options(replicates: int, seed: int) -> None:
    """Validate bootstrap configuration."""
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 0 or replicates == 1:
        raise ValueError(f"bootstrap_replicates must be 0 or >= 2, got {replicates}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"bootstrap_seed must be an integer, got {seed!r}")


def _validate_repetition_counts(agent_runs: int, judge_replays: int) -> None:
    """Validate the balanced design has enough observations for variance."""
    if agent_runs < 2:
        raise ValueError(f"agent_runs must be >= 2, got {agent_runs}")
    if judge_replays < 2:
        raise ValueError(f"judge_replays must be >= 2, got {judge_replays}")


def _validate_evaluator_names(observed: set[str], configured: set[str]) -> None:
    """Require observations to match the configured evaluator registry."""
    if observed == configured:
        return
    missing = sorted(configured - observed)
    unexpected = sorted(observed - configured)
    raise ValueError(f"evaluator names do not match step_sizes; missing={missing}, unexpected={unexpected}")


def _summarize_distribution(values: list[float | None]) -> _BootstrapSummary:
    """Produce percentile-bootstrap summaries and definition counts."""
    defined = sorted(value for value in values if value is not None and math.isfinite(value))
    if not defined:
        return {
            "mean": None,
            "lower_2_5": None,
            "median": None,
            "upper_97_5": None,
            "defined_replicates": 0,
            "undefined_replicates": len(values),
        }

    return {
        "mean": _mean(defined),
        "lower_2_5": _percentile(defined, 0.025),
        "median": _percentile(defined, 0.5),
        "upper_97_5": _percentile(defined, 0.975),
        "defined_replicates": len(defined),
        "undefined_replicates": len(values) - len(defined),
    }


def _percentile(values: list[float], probability: float) -> float:
    """Interpolate a percentile from sorted values."""
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return values[lower_index]
    fraction = position - lower_index
    return values[lower_index] + fraction * (values[upper_index] - values[lower_index])


def _sample_variance(values: list[float]) -> float:
    """Return Bessel-corrected sample variance."""
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean."""
    return sum(values) / len(values)


def _truncate_variance(value: float) -> float:
    """Enforce nonnegative reported variance components."""
    return 0.0 if value <= _EPSILON else value
