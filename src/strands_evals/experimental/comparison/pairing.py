"""Deterministic pairing for baseline and candidate evaluation reports."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from ...types.evaluation_report import EvaluationReport


class PairedScore(BaseModel):
    """One evaluator score matched between baseline and candidate reports.

    Attributes:
        case_id: Stable identifier derived from the case name or input.
        evaluator: Name of the evaluator that produced both scores.
        baseline_score: Score produced by the baseline.
        candidate_score: Score produced by the candidate.
        delta: Candidate score minus baseline score.
    """

    case_id: str
    evaluator: str
    baseline_score: float
    candidate_score: float
    delta: float


def pair_reports(baseline: EvaluationReport, candidate: EvaluationReport) -> list[PairedScore]:
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
        raise ValueError(
            "reports do not contain the same case/evaluator rows: "
            f"baseline_only={sorted(baseline_keys - candidate_keys)}, "
            f"candidate_only={sorted(candidate_keys - baseline_keys)}"
        )

    pairs: list[PairedScore] = []
    for key in sorted(baseline_keys):
        baseline_index = baseline_rows[key]
        candidate_index = candidate_rows[key]
        baseline_case = baseline.cases[baseline_index]
        candidate_case = candidate.cases[candidate_index]
        evaluator, case_id = key

        if _canonical_json(baseline_case["input"]) != _canonical_json(candidate_case["input"]):
            raise ValueError(f"case inputs differ between reports: case_id={case_id!r}, evaluator={evaluator!r}")

        baseline_details = (
            baseline.detailed_results[baseline_index] if baseline_index < len(baseline.detailed_results) else []
        )
        candidate_details = (
            candidate.detailed_results[candidate_index] if candidate_index < len(candidate.detailed_results) else []
        )
        baseline_applicable = EvaluationReport.is_applicable(baseline_details)
        candidate_applicable = EvaluationReport.is_applicable(candidate_details)
        if baseline_applicable != candidate_applicable:
            raise ValueError(
                f"case applicability differs between reports: case_id={case_id!r}, evaluator={evaluator!r}"
            )
        if not baseline_applicable:
            continue

        baseline_score = baseline.scores[baseline_index]
        candidate_score = candidate.scores[candidate_index]
        pairs.append(
            PairedScore(
                case_id=case_id,
                evaluator=evaluator,
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                delta=candidate_score - baseline_score,
            )
        )

    return pairs


def _index_report(report: EvaluationReport, report_name: str) -> dict[tuple[str, str], int]:
    if len(report.cases) != len(report.scores):
        raise ValueError(f"{report_name} report has {len(report.cases)} cases but {len(report.scores)} scores")

    rows: dict[tuple[str, str], int] = {}
    for index, case in enumerate(report.cases):
        evaluator = case.get("evaluator")
        if not isinstance(evaluator, str) or not evaluator.strip():
            raise ValueError(f"{report_name} report row {index} is missing a non-empty evaluator name")

        if "input" not in case:
            raise ValueError(f"{report_name} report row {index} is missing its case input")

        case_id = _case_id(case, report_name, index)
        key = (evaluator, case_id)
        if key in rows:
            raise ValueError(
                f"{report_name} report contains duplicate case/evaluator row: "
                f"case_id={case_id!r}, evaluator={evaluator!r}"
            )

        rows[key] = index

    return rows


def _case_id(case: dict, report_name: str, index: int) -> str:
    name = case.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"{report_name} report row {index} has a non-string case name")
    if isinstance(name, str) and name.strip():
        return name.strip()

    input_signature = _canonical_json(case["input"])
    digest = hashlib.sha256(input_signature.encode("utf-8")).hexdigest()
    return f"input:{digest}"


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError("case input must be JSON-serializable") from error
