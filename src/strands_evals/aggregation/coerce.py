"""Coerce ``EvaluationReport`` objects into a tabular ``pandas.DataFrame``.

This is the single touch point where the structured evaluation report data
model becomes a data frame. Once in data-frame form, all aggregation,
filtering, and description piggybacks on pandas rather than being
reimplemented here.

Column schema (one row per case):

    case        the case key (metadata["case_key"], falling back to the
                case name)
    evaluator   the report's evaluator_name
    score       per-case score
    passed      per-case pass/fail flag
    reason      per-case free-text reason
    metadata.*  one column per metadata key, named ``metadata.<key>``

``flatten`` concatenates the frames of many reports into one, taking the
union of metadata columns. This is how results spanning multiple evaluators
or runs are combined before grouping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import pandas as pd

if TYPE_CHECKING:  # avoid a hard import cycle / keep import-time cost low
    from ..types.evaluation_report import EvaluationReport

# Stable column names other modules can reference instead of bare strings.
CASE = "case"
EVALUATOR = "evaluator"
SCORE = "score"
PASSED = "passed"
REASON = "reason"
METADATA_PREFIX = "metadata."

_BASE_COLUMNS = [CASE, EVALUATOR, SCORE, PASSED, REASON]


def _case_key(case: dict, index: int) -> str:
    """Resolve a case's grouping key: metadata case_key, else name, else index."""
    metadata = case.get("metadata") or {}
    return (
        metadata.get("case_key")
        or case.get("name")
        or f"case#{index}"
    )


def to_dataframe(report: "EvaluationReport") -> pd.DataFrame:
    """Coerce one ``EvaluationReport`` into a one-row-per-case data frame.

    The report's parallel ``scores`` / ``test_passes`` / ``reasons`` lists are
    zipped with ``cases``. Metadata keys are spread into ``metadata.<key>``
    columns. A report with no cases yields an empty frame with the base
    columns present.
    """
    cases = report.cases or []
    scores = report.scores or []
    passes = report.test_passes or []
    reasons = report.reasons or []

    rows: list[dict] = []
    for i, case in enumerate(cases):
        metadata = case.get("metadata") or {}
        row = {
            CASE: _case_key(case, i),
            EVALUATOR: report.evaluator_name or "Unknown",
            SCORE: float(scores[i]) if i < len(scores) else float("nan"),
            PASSED: bool(passes[i]) if i < len(passes) else False,
            REASON: reasons[i] if i < len(reasons) else "",
        }
        for key, value in metadata.items():
            # case_key is already promoted to the `case` column; don't duplicate it.
            if key == "case_key":
                continue
            row[f"{METADATA_PREFIX}{key}"] = value
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_BASE_COLUMNS)

    frame = pd.DataFrame(rows)
    # Keep the base columns first, metadata columns after, for stable display.
    meta_cols = [c for c in frame.columns if c.startswith(METADATA_PREFIX)]
    ordered = [c for c in _BASE_COLUMNS if c in frame.columns] + sorted(meta_cols)
    return frame[ordered]


def flatten(reports: Iterable["EvaluationReport"]) -> pd.DataFrame:
    """Concatenate many reports into a single data frame.

    Reports may carry different metadata keys; the union of columns is taken
    and missing cells are filled with ``NaN`` (pandas default). An empty
    iterable yields an empty frame with the base columns present.
    """
    frames = [to_dataframe(r) for r in reports]
    if not frames:
        return pd.DataFrame(columns=_BASE_COLUMNS)
    return pd.concat(frames, ignore_index=True)
