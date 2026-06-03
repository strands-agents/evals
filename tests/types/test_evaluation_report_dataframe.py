"""Tests for the pandas-style surface on ``EvaluationReport``.

Consolidates what were previously test_coerce / test_frame / test_mixin, now
that the aggregation package is dissolved onto the report itself. Exercises:
- ``report.to_dataframe()`` (one row per case, metadata spread, evaluator)
- ``report.group_by(...).agg(...)`` / filter / describe
- ``EvaluationReport.flatten(reports).group_by("evaluator")`` preserving
  per-report evaluator identity (the multi-report path)
- ``ReportFrame`` wrapper behavior + round-trip
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from strands_evals.types.evaluation_report import (
    CASE,
    EVALUATOR,
    PASSED,
    REASON,
    SCORE,
    EvaluationReport,
    ReportFrame,
    ReportGroupBy,
)


def _build_report(evaluator_name: str, rows: list[dict[str, Any]]) -> EvaluationReport:
    """Build a report from row dicts: case_key/score/passed/reason + extra meta keys."""
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
    df = report.to_dataframe()
    assert len(df) == 2
    assert list(df[CASE]) == ["foo", "bar"]
    assert list(df[EVALUATOR]) == ["Eval1", "Eval1"]
    assert list(df[SCORE]) == [0.9, 0.4]
    assert list(df[PASSED]) == [True, False]
    assert list(df[REASON]) == ["ok", "bad"]


def test_to_dataframe_base_columns_present():
    report = _build_report("Eval1", [{"case_key": "foo", "score": 1.0, "passed": True}])
    df = report.to_dataframe()
    for col in (CASE, EVALUATOR, SCORE, PASSED, REASON):
        assert col in df.columns


def test_to_dataframe_spreads_metadata_columns():
    report = _build_report(
        "Eval1",
        [{"case_key": "foo", "score": 1.0, "passed": True, "scenario": "baseline"}],
    )
    df = report.to_dataframe()
    assert "metadata.scenario" in df.columns
    assert df["metadata.scenario"].iloc[0] == "baseline"


def test_to_dataframe_does_not_duplicate_case_key_in_metadata():
    report = EvaluationReport(
        evaluator_name="Eval1",
        overall_score=1.0,
        scores=[1.0],
        cases=[{"name": "n", "metadata": {"case_key": "foo", "scenario": "x"}}],
        test_passes=[True],
        reasons=[""],
    )
    df = report.to_dataframe()
    assert df[CASE].iloc[0] == "foo"
    assert "metadata.case_key" not in df.columns
    assert "metadata.scenario" in df.columns


def test_to_dataframe_falls_back_to_case_name():
    report = _build_report("Eval1", [{"name": "fallback", "score": 0.5, "passed": True}])
    df = report.to_dataframe()
    assert df[CASE].iloc[0] == "fallback"


def test_to_dataframe_empty_report_has_base_columns():
    report = EvaluationReport(
        evaluator_name="Eval1",
        overall_score=0.0,
        scores=[],
        cases=[],
        test_passes=[],
        reasons=[],
    )
    df = report.to_dataframe()
    assert df.empty
    assert list(df.columns) == [CASE, EVALUATOR, SCORE, PASSED, REASON]


def test_to_dataframe_handles_missing_evaluator_name():
    report = EvaluationReport(
        evaluator_name="",
        overall_score=0.5,
        scores=[0.5],
        cases=[{"name": "c", "metadata": {}}],
        test_passes=[True],
        reasons=[""],
    )
    df = report.to_dataframe()
    assert df[EVALUATOR].iloc[0] == "Unknown"


def test_to_dataframe_pads_short_parallel_lists():
    report = EvaluationReport(
        evaluator_name="Eval1",
        overall_score=0.5,
        scores=[0.5],
        cases=[{"name": "c0", "metadata": {}}, {"name": "c1", "metadata": {}}],
        test_passes=[True],
        reasons=["only-one"],
    )
    df = report.to_dataframe()
    assert len(df) == 2
    assert math.isnan(df[SCORE].iloc[1])
    assert df[PASSED].iloc[1] == False  # noqa: E712
    assert df[REASON].iloc[1] == ""


def test_to_dataframe_base_columns_precede_metadata():
    report = _build_report(
        "Eval1",
        [{"case_key": "foo", "score": 1.0, "passed": True, "zzz": 1, "aaa": 2}],
    )
    cols = list(report.to_dataframe().columns)
    assert cols[:5] == [CASE, EVALUATOR, SCORE, PASSED, REASON]
    assert cols[5:] == ["metadata.aaa", "metadata.zzz"]


# ---------------------------------------------------------------------------
# single-report surface
# ---------------------------------------------------------------------------


def test_report_frame_returns_report_frame():
    report = _build_report("Eval1", [{"case_key": "foo", "score": 1.0, "passed": True}])
    assert isinstance(report.frame(), ReportFrame)


def test_report_group_by_within_one_evaluator():
    report = _build_report(
        "GoalSuccess",
        [
            {"case_key": "foo", "score": 0.9, "passed": True},
            {"case_key": "bar", "score": 0.6, "passed": True},
        ],
    )
    result = report.group_by("evaluator").agg(
        mean_score=("score", "mean"),
        pass_rate=("passed", "mean"),
        n=("score", "count"),
    )
    assert isinstance(result, ReportFrame)
    assert len(result.df) == 1
    row = result.df.iloc[0]
    assert row["evaluator"] == "GoalSuccess"
    assert row["mean_score"] == pytest.approx(0.75)
    assert row["n"] == 2


def test_report_group_by_metadata_column():
    report = _build_report(
        "Eval1",
        [
            {"case_key": "foo", "score": 1.0, "passed": True, "category": "A"},
            {"case_key": "bar", "score": 0.0, "passed": False, "category": "B"},
        ],
    )
    result = report.group_by("metadata.category").agg(mean_score=("score", "mean"))
    assert set(result.df["metadata.category"]) == {"A", "B"}


def test_report_group_by_returns_report_groupby():
    report = _build_report("Eval1", [{"case_key": "foo", "score": 1.0, "passed": True}])
    assert isinstance(report.group_by("evaluator"), ReportGroupBy)


def test_report_filter():
    report = _build_report(
        "Eval1",
        [
            {"case_key": "foo", "score": 0.9, "passed": True},
            {"case_key": "bar", "score": 0.2, "passed": False},
        ],
    )
    result = report.filter(lambda d: d["passed"])
    assert len(result) == 1
    assert result.df["case"].iloc[0] == "foo"


def test_report_describe():
    report = _build_report(
        "Eval1",
        [
            {"case_key": "foo", "score": 0.9, "passed": True},
            {"case_key": "bar", "score": 0.6, "passed": True},
        ],
    )
    assert isinstance(report.describe(), ReportFrame)


def test_report_agg_whole_frame():
    report = _build_report(
        "Eval1",
        [
            {"case_key": "foo", "score": 0.9, "passed": True},
            {"case_key": "bar", "score": 0.7, "passed": True},
        ],
    )
    result = report.agg({"score": "mean"})
    assert isinstance(result, ReportFrame)
    assert result.df["score"].iloc[0] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# multi-report surface: flatten then group (the 5/27 API)
# ---------------------------------------------------------------------------


def test_flatten_returns_combined_report():
    a = _build_report("EvalA", [{"case_key": "foo", "score": 1.0, "passed": True}])
    b = _build_report("EvalB", [{"case_key": "foo", "score": 0.0, "passed": False}])
    combined = EvaluationReport.flatten([a, b])
    assert isinstance(combined, EvaluationReport)
    assert combined.evaluator_name == "Combined"
    assert len(combined.cases) == 2


def test_flatten_to_dataframe_preserves_per_report_evaluator():
    # The crux: the combined report says "Combined", but the frame must keep
    # the real per-row evaluators (read from the case-level evaluator stamp).
    a = _build_report("EvalA", [{"case_key": "foo", "score": 1.0, "passed": True}])
    b = _build_report("EvalB", [{"case_key": "foo", "score": 0.0, "passed": False}])
    df = EvaluationReport.flatten([a, b]).to_dataframe()
    assert set(df[EVALUATOR]) == {"EvalA", "EvalB"}


def test_flatten_then_group_by_evaluator():
    reports = [
        _build_report(
            "GoalSuccess",
            [
                {"case_key": "foo", "score": 0.9, "passed": True},
                {"case_key": "bar", "score": 0.6, "passed": True},
            ],
        ),
        _build_report(
            "ToolAccuracy",
            [
                {"case_key": "foo", "score": 0.8, "passed": True},
                {"case_key": "bar", "score": 0.4, "passed": False},
            ],
        ),
    ]
    result = EvaluationReport.flatten(reports).group_by("evaluator").agg(
        mean_score=("score", "mean"),
        pass_rate=("passed", "mean"),
        n=("score", "count"),
    )
    df = result.df
    assert set(df["evaluator"]) == {"GoalSuccess", "ToolAccuracy"}
    tool = df[df["evaluator"] == "ToolAccuracy"].iloc[0]
    assert tool["mean_score"] == pytest.approx(0.6)
    assert tool["pass_rate"] == pytest.approx(0.5)


def test_flatten_then_group_by_multiple_dimensions():
    reports = [
        _build_report(
            "GoalSuccess",
            [
                {"case_key": "foo", "score": 0.9, "passed": True, "scenario": "baseline"},
                {"case_key": "foo", "score": 0.5, "passed": False, "scenario": "chaos"},
            ],
        ),
    ]
    result = EvaluationReport.flatten(reports).group_by(
        ["evaluator", "metadata.scenario"]
    ).agg(mean_score=("score", "mean"))
    assert len(result.df) == 2
    assert set(result.df["metadata.scenario"]) == {"baseline", "chaos"}


def test_flatten_empty_returns_empty_combined_report():
    combined = EvaluationReport.flatten([])
    assert isinstance(combined, EvaluationReport)
    assert combined.to_dataframe().empty


# ---------------------------------------------------------------------------
# ReportFrame wrapper
# ---------------------------------------------------------------------------


def test_report_frame_len_and_df():
    report = _build_report(
        "Eval1",
        [
            {"case_key": "foo", "score": 0.9, "passed": True},
            {"case_key": "bar", "score": 0.6, "passed": True},
        ],
    )
    rf = report.frame()
    assert len(rf) == 2
    assert isinstance(rf.df, pd.DataFrame)


def test_report_frame_passthrough_rewraps_dataframe():
    report = _build_report(
        "Eval1",
        [
            {"case_key": "foo", "score": 0.9, "passed": True},
            {"case_key": "bar", "score": 0.6, "passed": True},
        ],
    )
    # head() is not defined on ReportFrame -> delegates to pandas, re-wrapped
    head = report.frame().head(1)
    assert isinstance(head, ReportFrame)
    assert len(head) == 1


def test_report_frame_to_file_roundtrip(tmp_path):
    report = _build_report(
        "Eval1",
        [
            {"case_key": "foo", "score": 0.9, "passed": True},
            {"case_key": "bar", "score": 0.6, "passed": False},
        ],
    )
    path = str(tmp_path / "frame.json")
    report.frame().to_file(path)
    loaded = ReportFrame.from_file(path)
    assert len(loaded) == 2
    assert set(loaded.df[CASE]) == {"foo", "bar"}
