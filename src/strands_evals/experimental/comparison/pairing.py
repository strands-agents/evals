"""Deterministic pairing for baseline and candidate evaluation reports."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, TypeAdapter

from ...types.evaluation_report import EvaluationReport

_JSON_VALUE_ADAPTER = TypeAdapter(Any)


class PairedScore(BaseModel):
    """One evaluator score matched between baseline and candidate reports.

    Attributes:
        case_id: Stable identifier derived from the case name or input.
        evaluator: Name of the evaluator that produced both scores.
        baseline_index: Position of the row in the baseline report.
        candidate_index: Position of the row in the candidate report.
        baseline_score: Score produced by the baseline.
        candidate_score: Score produced by the candidate.
        delta: Candidate score minus baseline score.
    """

    case_id: str
    evaluator: str
    baseline_index: int
    candidate_index: int
    baseline_score: float
    candidate_score: float
    delta: float


class PairedReport(BaseModel):
    """Matched evaluator scores from a baseline and candidate report.

    Attributes:
        pairs: Comparable rows ordered by evaluator and case identity.
    """

    pairs: list[PairedScore]


@dataclass(frozen=True)
class _IndexedRow:
    index: int
    case_id: str
    evaluator: str
    input_signature: str
    score: float
    applicable: bool


def pair_reports(baseline: EvaluationReport, candidate: EvaluationReport) -> PairedReport:
    """Match comparable evaluator scores from two reports.

    Cases are paired by exact case name when one is present. An unnamed case is
    paired by a deterministic hash of its input. Evaluator names must also
    match. Row order, generated session IDs, and task outputs do not affect
    pairing.

    Rows that are not applicable in both reports are omitted. A mismatch in
    applicability is rejected because the two scores do not represent the same
    measurement.

    Args:
        baseline: Evaluation report produced by the existing configuration.
        candidate: Evaluation report produced by the changed configuration.

    Returns:
        The matched score rows and their candidate-minus-baseline deltas.

    Raises:
        ValueError: If either report is malformed, contains ambiguous rows, or
            cannot be paired exactly with the other report.
    """

    baseline_rows = _index_report(baseline, "baseline")
    candidate_rows = _index_report(candidate, "candidate")

    baseline_keys = set(baseline_rows)
    candidate_keys = set(candidate_rows)
    if baseline_keys != candidate_keys:
        baseline_only = sorted(_display_key(key) for key in baseline_keys - candidate_keys)
        candidate_only = sorted(_display_key(key) for key in candidate_keys - baseline_keys)
        raise ValueError(
            "reports do not contain the same case/evaluator rows: "
            f"baseline_only={baseline_only}, candidate_only={candidate_only}"
        )

    pairs = []
    for key in sorted(baseline_keys):
        baseline_row = baseline_rows[key]
        candidate_row = candidate_rows[key]

        if baseline_row.input_signature != candidate_row.input_signature:
            raise ValueError(
                "case inputs differ between reports: "
                f"case_id={baseline_row.case_id!r}, evaluator={baseline_row.evaluator!r}"
            )
        if baseline_row.applicable != candidate_row.applicable:
            raise ValueError(
                "case applicability differs between reports: "
                f"case_id={baseline_row.case_id!r}, evaluator={baseline_row.evaluator!r}"
            )
        if not baseline_row.applicable:
            continue

        pairs.append(
            PairedScore(
                case_id=baseline_row.case_id,
                evaluator=baseline_row.evaluator,
                baseline_index=baseline_row.index,
                candidate_index=candidate_row.index,
                baseline_score=baseline_row.score,
                candidate_score=candidate_row.score,
                delta=candidate_row.score - baseline_row.score,
            )
        )

    return PairedReport(pairs=pairs)


def _index_report(report: EvaluationReport, report_name: str) -> dict[tuple[str, str, str], _IndexedRow]:
    if len(report.cases) != len(report.scores):
        raise ValueError(f"{report_name} report has {len(report.cases)} cases but {len(report.scores)} scores")

    rows = {}
    for index, (case, score) in enumerate(zip(report.cases, report.scores, strict=True)):
        evaluator = case.get("evaluator")
        if not isinstance(evaluator, str) or not evaluator.strip():
            raise ValueError(f"{report_name} report row {index} is missing a non-empty evaluator name")

        if "input" not in case:
            raise ValueError(f"{report_name} report row {index} is missing its case input")

        input_signature = _canonical_json(case["input"], f"{report_name} report row {index} input")
        identity_kind, identity_value, case_id = _case_identity(case, input_signature, report_name, index)
        key = (evaluator, identity_kind, identity_value)
        if key in rows:
            raise ValueError(
                f"{report_name} report contains duplicate case/evaluator row: "
                f"case_id={case_id!r}, evaluator={evaluator!r}"
            )

        detailed_results = report.detailed_results[index] if index < len(report.detailed_results) else []
        rows[key] = _IndexedRow(
            index=index,
            case_id=case_id,
            evaluator=evaluator,
            input_signature=input_signature,
            score=score,
            applicable=EvaluationReport.is_applicable(detailed_results),
        )

    return rows


def _case_identity(
    case: dict,
    input_signature: str,
    report_name: str,
    index: int,
) -> tuple[str, str, str]:
    name = case.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"{report_name} report row {index} has a non-string case name")
    if isinstance(name, str) and name.strip():
        normalized_name = name.strip()
        return "name", normalized_name, normalized_name

    digest = hashlib.sha256(input_signature.encode("utf-8")).hexdigest()
    return "input", digest, f"input:{digest[:12]}"


def _canonical_json(value: Any, context: str) -> str:
    try:
        jsonable_value = _JSON_VALUE_ADAPTER.dump_python(value, mode="json")
        return json.dumps(jsonable_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be JSON-serializable") from error


def _display_key(key: tuple[str, str, str]) -> str:
    evaluator, identity_kind, identity_value = key
    if identity_kind == "input":
        identity_value = f"{identity_value[:12]}..."
    return f"{evaluator}:{identity_kind}:{identity_value}"
