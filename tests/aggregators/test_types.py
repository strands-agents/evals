"""Tests for aggregators/types.py.

Covers ``AggregationResult`` field defaults and ``AggregationReport``
JSON serialization round-trips.
"""

from __future__ import annotations

import json
import os

import pytest

from strands_evals.aggregators.types import (
    AggregationReport,
    AggregationResult,
)


# ---------------------------------------------------------------------------
# AggregationResult
# ---------------------------------------------------------------------------


def _make_result(**overrides) -> AggregationResult:
    """Build a default AggregationResult with sensible test values."""
    base = dict(
        group_key="case_a",
        evaluator_name="Eval1",
        mean_score=0.8,
        min_score=0.6,
        max_score=1.0,
        pass_rate=0.75,
        num_results=4,
        num_passed=3,
        num_failed=1,
    )
    base.update(overrides)
    return AggregationResult(**base)


def test_aggregation_result_base_fields():
    r = _make_result()
    assert r.group_key == "case_a"
    assert r.evaluator_name == "Eval1"
    assert r.mean_score == 0.8
    assert r.pass_rate == 0.75
    assert r.num_results == 4
    assert r.num_passed == 3
    assert r.num_failed == 1


def test_aggregation_result_defaults():
    r = _make_result()
    assert r.reasons == []
    assert r.summary == ""
    assert r.extra == {}


def test_aggregation_result_with_reasons_and_summary():
    r = _make_result(reasons=["good run", "bad run"], summary="Mixed performance.")
    assert r.reasons == ["good run", "bad run"]
    assert r.summary == "Mixed performance."


def test_aggregation_result_extra_for_subclass_fields():
    r = _make_result(extra={"win_rate": 0.72, "ci_low": 0.58})
    assert r.extra["win_rate"] == 0.72
    assert r.extra["ci_low"] == 0.58


# ---------------------------------------------------------------------------
# AggregationReport: dict/JSON round-trip
# ---------------------------------------------------------------------------


def test_report_roundtrip_via_dict():
    report = AggregationReport(aggregations=[_make_result()])
    data = report.model_dump()
    restored = AggregationReport.model_validate(data)
    assert restored == report


def test_report_handles_empty_aggregations():
    report = AggregationReport()
    assert report.aggregations == []
    data = report.model_dump()
    restored = AggregationReport.model_validate(data)
    assert restored.aggregations == []


# ---------------------------------------------------------------------------
# AggregationReport: to_file / from_file
# ---------------------------------------------------------------------------


def test_report_to_file_writes_json(tmp_path):
    report = AggregationReport(aggregations=[_make_result()])
    out = tmp_path / "report.json"
    report.to_file(str(out))

    assert out.exists()
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["aggregations"]) == 1
    assert data["aggregations"][0]["group_key"] == "case_a"


def test_report_to_file_appends_extension_when_missing(tmp_path):
    report = AggregationReport(aggregations=[_make_result()])
    base = tmp_path / "report"
    report.to_file(str(base))

    expected = tmp_path / "report.json"
    assert expected.exists()


def test_report_to_file_rejects_non_json_extension(tmp_path):
    report = AggregationReport(aggregations=[_make_result()])
    with pytest.raises(ValueError, match="Only .json"):
        report.to_file(str(tmp_path / "report.yaml"))


def test_report_from_file_rejects_non_json_extension(tmp_path):
    path = tmp_path / "report.yaml"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Only .json"):
        AggregationReport.from_file(str(path))


def test_report_from_file_roundtrip(tmp_path):
    original = AggregationReport(aggregations=[_make_result(group_key="x")])
    path = tmp_path / "r.json"
    original.to_file(str(path))
    loaded = AggregationReport.from_file(str(path))
    assert loaded == original


def test_report_to_file_creates_parent_directory(tmp_path):
    report = AggregationReport(aggregations=[_make_result()])
    nested = tmp_path / "nested" / "deep" / "report.json"
    report.to_file(str(nested))
    assert nested.exists()
