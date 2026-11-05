"""Private data and statistical helpers for evaluator calibration."""

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel

from ...types.evaluation_report import EvaluationReport
from .algorithm import CalibrationExample, CalibrationObservation

_EPSILON = 1e-15
_ERROR_REASON_PREFIXES = ("An error occurred:", "Evaluator error:")
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


def _prepare_calibration_dataset(
    reports_by_run: list[list[EvaluationReport]],
    *,
    configured_evaluators: set[str],
    expected_design: tuple[int, int] | None,
    expected_structure: _CalibrationStructure | None,
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
        actual_design=(dataset.run_count, dataset.replay_count),
        operation=operation,
        expected_structure=expected_structure,
        actual_structure=dataset.structure,
        scoring=scoring,
    )
    return dataset


def _validate_calibration_compatibility(
    *,
    expected_design: tuple[int, int] | None,
    actual_design: tuple[int, int],
    operation: str,
    expected_structure: _CalibrationStructure | None = None,
    actual_structure: _CalibrationStructure | None = None,
    scoring: bool = False,
) -> None:
    """Require the fitted structure while allowing new scoring samples."""
    if not scoring and expected_design is not None and actual_design != expected_design:
        expected_runs, expected_replays = expected_design
        actual_runs, actual_replays = actual_design
        raise ValueError(
            f"{operation} design must match the fitted calibration design; "
            f"expected agent_runs={expected_runs}, judge_replays={expected_replays}; "
            f"got agent_runs={actual_runs}, judge_replays={actual_replays}"
        )

    if expected_structure is None or actual_structure is None:
        return

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
        calibration_structure=None,
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
        calibration_structure=dataset.structure,
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
