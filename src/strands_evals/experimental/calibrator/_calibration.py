"""Private data and statistical helpers for evaluator calibration."""

import copy
import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel
from strands.models.model import Model

from ...case import Case
from ...evaluators.evaluator import Evaluator
from ...types.evaluation_report import EvaluationReport
from .algorithm import (
    CalibrationExample,
    CalibrationObservation,
    CalibrationTarget,
    CalibrationTargetDefinition,
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

_ParsedReports = list[list[dict[str, dict[str, float]]]]
_CaseSignatures = tuple[tuple[str, str], ...]
_CalibrationStructure = tuple[tuple[str, ...], _CaseSignatures]


@dataclass(frozen=True)
class _EvaluatorStatistics:
    """Variance components and judge repeatability for one evaluator."""

    variance_components: dict[str, float]
    repeatability: float


@dataclass(frozen=True)
class _CalibrationState:
    """Complete immutable-by-replacement state for one calibrator."""

    variance_components: dict[str, dict[str, float]]
    repeatabilities: dict[str, float]
    calibration_design: tuple[int, int] | None
    calibration_context: dict[str, Any] | None
    case_count: int | None
    calibration_structure: _CalibrationStructure | None
    fit_structure_fingerprint: str | None
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
    calibration_context: dict[str, Any] | None

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
        calibration_context = None if not present_contexts else copy.deepcopy(present_contexts[0])

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

    def examples(
        self,
        targets: dict[tuple[str, int], float] | None = None,
    ) -> tuple[CalibrationExample, ...]:
        """Aggregate judge replays to one feature vector per case and agent run."""
        target_values = {} if targets is None else targets
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
                target=target_values.get((case_name, run_index)),
            )
            for run_index, run_reports in enumerate(self.parsed)
            for case_name in self.case_names
        )


def _prepare_calibration_dataset(
    reports_by_run: list[list[EvaluationReport]],
    *,
    configured_evaluators: set[str],
    step_sizes: dict[str, float],
    expected_design: tuple[int, int] | None,
    expected_context: dict[str, Any] | None,
    expected_structure: _CalibrationStructure | None,
    expected_structure_fingerprint: str | None,
    operation: str,
    scoring: bool = False,
) -> _CalibrationDataset:
    """Parse reports and enforce the calibrator's structural invariants."""
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
        expected_structure_fingerprint=expected_structure_fingerprint,
        actual_structure=dataset.structure,
        scoring=scoring,
    )
    return dataset


def _prepare_calibration_targets(
    case_names: tuple[str, ...],
    agent_runs: int,
    targets: list[CalibrationTarget] | None,
    target_definition: CalibrationTargetDefinition | None,
) -> tuple[CalibrationTargetDefinition | None, dict[tuple[str, int], float]]:
    """Validate and index optional case-run targets."""
    if targets is None and target_definition is None:
        return None, {}
    if targets is None or target_definition is None:
        raise ValueError("targets and target_definition must be provided together")
    if not isinstance(targets, list):
        raise ValueError("targets must be a list of CalibrationTarget values")
    if not targets:
        raise ValueError("targets must contain one value for every case and agent run")
    if not isinstance(target_definition, CalibrationTargetDefinition):
        raise ValueError("target_definition must be a CalibrationTargetDefinition")

    target_values: dict[tuple[str, int], float] = {}
    for index, target in enumerate(targets):
        if not isinstance(target, CalibrationTarget):
            raise ValueError(f"target at index {index} must be a CalibrationTarget")
        key = (target.case_name, target.agent_run)
        if key in target_values:
            raise ValueError(
                f"targets contain a duplicate case-run value: case={target.case_name!r}, agent_run={target.agent_run}"
            )
        target_values[key] = target.value

    expected = {(case_name, run_index) for run_index in range(agent_runs) for case_name in case_names}
    observed = set(target_values)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(f"targets do not match the calibration case-runs; missing={missing}, unexpected={unexpected}")
    return target_definition, target_values


def _build_calibration_context(
    cases: list[Case],
    evaluators: list[Evaluator],
    step_sizes: dict[str, float],
) -> dict[str, Any]:
    """Fingerprint experiment inputs and retain safe audit metadata."""
    case_payloads = [
        _normalize_configuration_value(
            case.model_dump(exclude={"session_id"}),
        )
        for case in cases
    ]
    case_payloads.sort(key=_canonical_json)

    evaluator_payloads = [_evaluator_configuration(evaluator) for evaluator in evaluators]
    evaluator_payloads.sort(key=_canonical_json)
    evaluator_metadata = [_evaluator_profile_metadata(payload) for payload in evaluator_payloads]

    components = {
        "dataset": _fingerprint(case_payloads),
        "evaluators": _fingerprint(evaluator_payloads),
        "step_sizes": _step_size_fingerprint(step_sizes),
    }
    return {
        "version": _CALIBRATION_CONTEXT_VERSION,
        **components,
        "overall": _fingerprint(components),
        "metadata": {
            "dataset": {
                "fingerprint": components["dataset"],
                "case_count": len(cases),
                "named_case_count": sum(case.name is not None for case in cases),
            },
            "evaluators": evaluator_metadata,
            "step_sizes": {name: step_sizes[name] for name in sorted(step_sizes)},
        },
    }


def _evaluator_profile_metadata(configuration: dict[str, Any]) -> dict[str, Any]:
    """Reduce evaluator configuration to safe, inspectable profile metadata."""
    parameters = configuration["parameters"]
    metadata: dict[str, Any] = {
        "implementation": configuration["implementation"],
        "name": configuration["name"],
        "configuration_fingerprint": _fingerprint(configuration),
    }

    model = parameters.get("model")
    if isinstance(model, dict):
        metadata["judge_model"] = {"model_id": model.get("model_id", "")}

    for rubric_field in ("rubric", "system_prompt"):
        rubric = parameters.get(rubric_field)
        if isinstance(rubric, str):
            metadata[rubric_field] = {
                "sha256": hashlib.sha256(rubric.encode("utf-8")).hexdigest(),
                "character_count": len(rubric),
            }
    return metadata


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


def _validate_calibration_compatibility(
    *,
    expected_design: tuple[int, int] | None,
    expected_context: dict[str, Any] | None,
    actual_design: tuple[int, int],
    actual_context: dict[str, Any] | None,
    step_sizes: dict[str, float],
    operation: str,
    expected_structure: _CalibrationStructure | None = None,
    expected_structure_fingerprint: str | None = None,
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

    if (
        not scoring
        and expected_structure_fingerprint is not None
        and (
            actual_structure is None
            or _report_structure_fingerprint(actual_structure) != expected_structure_fingerprint
        )
    ):
        raise ValueError(f"{operation} report structure is incompatible with the fitted calibration")

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
            "cannot mix fingerprinted reports with context-free reports"
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


def _report_structure_fingerprint(structure: _CalibrationStructure) -> str:
    """Fingerprint evaluator names and stable case definitions."""
    return _fingerprint(structure)


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
        signature = hashlib.sha256(
            _canonical_json(_normalize_configuration_value(stable_case)).encode("utf-8")
        ).hexdigest()
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


def _initial_calibration_state() -> _CalibrationState:
    """Return the state used before measurement data is fitted."""
    return _CalibrationState(
        variance_components={},
        repeatabilities={},
        calibration_design=None,
        calibration_context=None,
        case_count=None,
        calibration_structure=None,
        fit_structure_fingerprint=None,
        status="uncalibrated",
    )


def _fit_dataset(dataset: _CalibrationDataset) -> _CalibrationState:
    """Estimate measurement diagnostics and return a replacement fitted state."""
    variance_components: dict[str, dict[str, float]] = {}
    repeatabilities: dict[str, float] = {}
    for evaluator_name in dataset.evaluator_names:
        statistics = _estimate_statistics(dataset, evaluator_name)
        variance_components[evaluator_name] = statistics.variance_components
        repeatabilities[evaluator_name] = statistics.repeatability

    return _CalibrationState(
        variance_components=variance_components,
        repeatabilities=repeatabilities,
        calibration_design=(dataset.run_count, dataset.replay_count),
        calibration_context=copy.deepcopy(dataset.calibration_context),
        case_count=len(dataset.case_names),
        calibration_structure=dataset.structure,
        fit_structure_fingerprint=(
            _report_structure_fingerprint(dataset.structure) if dataset.calibration_context is None else None
        ),
        status="uncalibrated",
    )


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


def _canonical_json(value: Any) -> str:
    """Serialize normalized values deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


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
