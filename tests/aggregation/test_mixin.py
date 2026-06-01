"""Tests for aggregation/mixin.py — the EvaluationReport surface.

These exercise the methods exactly as they appear in user code and in the
meeting notes: ``report.group_by(...)`` for one report and
``ReportFrame.from_reports(reports).group_by(...)`` across reports.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_evals.aggregation import ReportFrame, ReportGroupBy
from strands_evals.types.evaluation_report import EvaluationReport


def _build_report(evaluator_name: str, rows: list[dict[str, Any]]) -> EvaluationReport:
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
# single-report surface
# ---------------------------------------------------------------------------


def test_report_to_dataframe():
    report = _build_report("Eval1", [{"case_key": "foo", "score": 1.0, "passed": True}])
    df = report.to_dataframe()
    assert len(df) == 1
    assert df["evaluator"].iloc[0] == "Eval1"


def test_report_frame_returns_report_frame():
    report = _build_report("Eval1", [{"case_key": "foo", "score": 1.0, "passed": True}])
    assert isinstance(report.frame(), ReportFrame)


def test_report_group_by_within_one_evaluator():
    # collapse the case dimension within one report -> 1 row per evaluator
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
    result = report.describe()
    assert isinstance(result, ReportFrame)


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
# multi-report surface: flatten then group
# ---------------------------------------------------------------------------


def test_from_reports_returns_report_frame():
    a = _build_report("EvalA", [{"case_key": "foo", "score": 1.0, "passed": True}])
    b = _build_report("EvalB", [{"case_key": "foo", "score": 0.0, "passed": False}])
    assert isinstance(ReportFrame.from_reports([a, b]), ReportFrame)


def test_from_reports_then_group_by_evaluator():
    # the canonical multi-report path from the meeting notes
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
    result = ReportFrame.from_reports(reports).group_by("evaluator").agg(
        mean_score=("score", "mean"),
        pass_rate=("passed", "mean"),
        n=("score", "count"),
    )
    df = result.df
    assert set(df["evaluator"]) == {"GoalSuccess", "ToolAccuracy"}
    tool = df[df["evaluator"] == "ToolAccuracy"].iloc[0]
    assert tool["mean_score"] == pytest.approx(0.6)
    assert tool["pass_rate"] == pytest.approx(0.5)


def test_from_reports_then_group_by_multiple_dimensions():
    reports = [
        _build_report(
            "GoalSuccess",
            [
                {"case_key": "foo", "score": 0.9, "passed": True, "scenario": "baseline"},
                {"case_key": "foo", "score": 0.5, "passed": False, "scenario": "chaos"},
            ],
        ),
    ]
    result = ReportFrame.from_reports(reports).group_by(
        ["evaluator", "metadata.scenario"]
    ).agg(mean_score=("score", "mean"))
    assert len(result.df) == 2
    assert set(result.df["metadata.scenario"]) == {"baseline", "chaos"}
