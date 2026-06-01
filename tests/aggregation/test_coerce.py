"""Tests for aggregation/coerce.py — report -> DataFrame coercion and flatten."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from strands_evals.aggregation import coerce, flatten, to_dataframe
from strands_evals.types.evaluation_report import EvaluationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_report(evaluator_name: str, rows: list[dict[str, Any]]) -> EvaluationReport:
    """Construct an EvaluationReport from row dicts.

    Each row supports: case_key, score, passed, reason, plus arbitrary extra
    keys that flow through into ``metadata``.
    """
    cases, scores, passes, reasons = [], [], [], []
    for i, r in enumerate(rows):
        meta = {k: v for k, v in r.items() if k not in {"score", "passed", "reason"}}
        case_name = r.get("name") or f"{r.get('case_key', 'case')}#{i}"
        cases.append({"name": case_name, "metadata": meta})
        scores.append(r["score"])
        passes.append(r["passed"])
        reasons.append(r.get("reason", ""))
    overall = sum(scores) / len(scores) if scores else 0.0
    return EvaluationReport(
        evaluator_name=evaluator_name,
        overall_score=overall,
        scores=scores,
        cases=cases,
        test_passes=passes,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------


def test_to_dataframe_one_row_per_case():
    report = _build_report(
        "Eval1",
        [
            {"case_key": "foo", "score": 0.9, "passed": True, "reason": "ok"},
            {"case_key": "bar", "score": 0.4, "passed": False, "reason": "bad"},
        ],
    )
    df = to_dataframe(report)
    assert len(df) == 2
    assert list(df[coerce.CASE]) == ["foo", "bar"]
    assert list(df[coerce.EVALUATOR]) == ["Eval1", "Eval1"]
    assert list(df[coerce.SCORE]) == [0.9, 0.4]
    assert list(df[coerce.PASSED]) == [True, False]
    assert list(df[coerce.REASON]) == ["ok", "bad"]


def test_to_dataframe_base_columns_present():
    report = _build_report("Eval1", [{"case_key": "foo", "score": 1.0, "passed": True}])
    df = to_dataframe(report)
    for col in (coerce.CASE, coerce.EVALUATOR, coerce.SCORE, coerce.PASSED, coerce.REASON):
        assert col in df.columns


def test_to_dataframe_spreads_metadata_columns():
    report = _build_report(
        "Eval1",
        [{"case_key": "foo", "score": 1.0, "passed": True, "scenario": "baseline"}],
    )
    df = to_dataframe(report)
    assert "metadata.scenario" in df.columns
    assert df["metadata.scenario"].iloc[0] == "baseline"


def test_to_dataframe_falls_back_to_case_name():
    report = _build_report("Eval1", [{"name": "fallback", "score": 0.5, "passed": True}])
    df = to_dataframe(report)
    # case_key absent -> uses the case name
    assert df[coerce.CASE].iloc[0] == "fallback"


def test_to_dataframe_empty_report_has_base_columns():
    report = EvaluationReport(
        evaluator_name="Eval1",
        overall_score=0.0,
        scores=[],
        cases=[],
        test_passes=[],
        reasons=[],
    )
    df = to_dataframe(report)
    assert df.empty
    assert list(df.columns) == [
        coerce.CASE,
        coerce.EVALUATOR,
        coerce.SCORE,
        coerce.PASSED,
        coerce.REASON,
    ]


def test_to_dataframe_handles_missing_evaluator_name():
    report = EvaluationReport(
        evaluator_name="",
        overall_score=0.5,
        scores=[0.5],
        cases=[{"name": "c", "metadata": {}}],
        test_passes=[True],
        reasons=[""],
    )
    df = to_dataframe(report)
    assert df[coerce.EVALUATOR].iloc[0] == "Unknown"


def test_to_dataframe_pads_short_parallel_lists():
    # cases longer than scores/passes/reasons -> pad rather than crash
    report = EvaluationReport(
        evaluator_name="Eval1",
        overall_score=0.5,
        scores=[0.5],
        cases=[{"name": "c0", "metadata": {}}, {"name": "c1", "metadata": {}}],
        test_passes=[True],
        reasons=["only-one"],
    )
    df = to_dataframe(report)
    assert len(df) == 2
    assert math.isnan(df[coerce.SCORE].iloc[1])
    assert df[coerce.PASSED].iloc[1] == False  # noqa: E712
    assert df[coerce.REASON].iloc[1] == ""


def test_to_dataframe_base_columns_precede_metadata():
    report = _build_report(
        "Eval1",
        [{"case_key": "foo", "score": 1.0, "passed": True, "zzz": 1, "aaa": 2}],
    )
    df = to_dataframe(report)
    cols = list(df.columns)
    assert cols[:5] == [coerce.CASE, coerce.EVALUATOR, coerce.SCORE, coerce.PASSED, coerce.REASON]
    # metadata columns sorted after the base columns
    assert cols[5:] == ["metadata.aaa", "metadata.zzz"]


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


def test_flatten_concatenates_reports():
    a = _build_report("EvalA", [{"case_key": "foo", "score": 1.0, "passed": True}])
    b = _build_report("EvalB", [{"case_key": "foo", "score": 0.0, "passed": False}])
    df = flatten([a, b])
    assert len(df) == 2
    assert set(df[coerce.EVALUATOR]) == {"EvalA", "EvalB"}


def test_flatten_takes_union_of_metadata_columns():
    a = _build_report("EvalA", [{"case_key": "foo", "score": 1.0, "passed": True, "scenario": "x"}])
    b = _build_report("EvalB", [{"case_key": "foo", "score": 0.0, "passed": False, "tool": "y"}])
    df = flatten([a, b])
    assert "metadata.scenario" in df.columns
    assert "metadata.tool" in df.columns
    # missing cells filled with NaN
    assert df["metadata.tool"].isna().sum() == 1
    assert df["metadata.scenario"].isna().sum() == 1


def test_flatten_empty_iterable_has_base_columns():
    df = flatten([])
    assert df.empty
    assert coerce.CASE in df.columns


def test_flatten_resets_index():
    a = _build_report("EvalA", [{"case_key": "foo", "score": 1.0, "passed": True}])
    b = _build_report("EvalB", [{"case_key": "bar", "score": 0.0, "passed": False}])
    df = flatten([a, b])
    assert list(df.index) == [0, 1]


def test_flatten_returns_plain_dataframe():
    a = _build_report("EvalA", [{"case_key": "foo", "score": 1.0, "passed": True}])
    assert isinstance(flatten([a]), pd.DataFrame)
