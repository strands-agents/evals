"""Tests for aggregators/types.py."""

import json
from pathlib import Path

import pytest

from strands_evals.aggregators.types import (
    AggregationReport,
    AggregationResult,
    EfficiencyStats,
    PairedComparisonStats,
)


# ---------------------------------------------------------------------------
# PairedComparisonStats
# ---------------------------------------------------------------------------


def test_paired_comparison_stats_construction():
    ps = PairedComparisonStats(
        metric_name="tokens_in",
        baseline_label="baseline",
        variant_label="variant",
        baseline_mean=1000.0,
        variant_mean=900.0,
        delta=-100.0,
        delta_pct=-0.1,
        test_used="wilcoxon",
        p_value=0.003,
        ci_low=-150.0,
        ci_high=-50.0,
        n_used=30,
        n_corrupted=2,
    )
    assert ps.metric_name == "tokens_in"
    assert ps.delta == -100.0
    assert ps.test_used == "wilcoxon"


def test_paired_comparison_stats_defaults():
    ps = PairedComparisonStats(
        metric_name="wall_clock_s",
        baseline_label="baseline",
        variant_label="variant",
        baseline_mean=2.0,
        variant_mean=2.5,
        delta=0.5,
        test_used="paired_t",
        p_value=0.04,
        ci_low=0.1,
        ci_high=0.9,
        n_used=20,
    )
    assert ps.delta_pct is None
    assert ps.n_corrupted == 0


# ---------------------------------------------------------------------------
# EfficiencyStats
# ---------------------------------------------------------------------------


def test_efficiency_stats_all_optional():
    e = EfficiencyStats()
    assert e.mean_tokens_in is None
    assert e.mean_cost_usd is None
    assert e.n_samples == 0


def test_efficiency_stats_partial():
    e = EfficiencyStats(mean_tokens_in=500.0, n_samples=10)
    assert e.mean_tokens_in == 500.0
    assert e.mean_tokens_out is None
    assert e.n_samples == 10


# ---------------------------------------------------------------------------
# AggregationResult
# ---------------------------------------------------------------------------


def _make_result(**overrides) -> AggregationResult:
    base = {
        "group_key": "order_flow",
        "evaluator_name": "GoalSuccessRateEvaluator",
        "mean_score": 0.85,
        "min_score": 0.5,
        "max_score": 1.0,
        "pass_rate": 0.9,
        "num_results": 30,
        "num_passed": 27,
        "num_failed": 3,
    }
    base.update(overrides)
    return AggregationResult(**base)


def test_aggregation_result_base_fields():
    r = _make_result()
    assert r.group_key == "order_flow"
    assert r.pass_rate == 0.9
    # Optional fields default to empty/None.
    assert r.paired_stats == []
    assert r.raw_values == {}
    assert r.trajectory_pointers == []
    assert r.efficiency is None
    assert r.summary == ""
    assert r.metadata == {}
    assert r.n_total == 0
    assert r.n_used == 0


def test_aggregation_result_with_paired_stats():
    ps = PairedComparisonStats(
        metric_name="pass_rate",
        baseline_label="baseline",
        variant_label="variant",
        baseline_mean=0.7,
        variant_mean=0.88,
        delta=0.18,
        test_used="mcnemar",
        p_value=0.003,
        ci_low=0.08,
        ci_high=0.28,
        n_used=60,
    )
    r = _make_result(
        paired_stats=[ps],
        n_total=64,
        n_corrupted=4,
        n_used=60,
        raw_values={"pass_rate": [1.0, 0.0, 1.0]},
        trajectory_pointers=["s1", "s2", "s3"],
        efficiency=EfficiencyStats(mean_tokens_in=1200.0, n_samples=60),
    )
    assert len(r.paired_stats) == 1
    assert r.paired_stats[0].metric_name == "pass_rate"
    assert r.n_total == 64
    assert r.efficiency.mean_tokens_in == 1200.0
    assert r.raw_values["pass_rate"] == [1.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# AggregationReport — serialization
# ---------------------------------------------------------------------------


def test_report_roundtrip_via_dict():
    r = _make_result(
        paired_stats=[
            PairedComparisonStats(
                metric_name="tokens_in",
                baseline_label="baseline",
                variant_label="variant",
                baseline_mean=1000.0,
                variant_mean=950.0,
                delta=-50.0,
                test_used="wilcoxon",
                p_value=0.04,
                ci_low=-90.0,
                ci_high=-10.0,
                n_used=28,
                n_corrupted=2,
            )
        ],
        n_total=30,
        n_corrupted=2,
        n_used=28,
    )
    report = AggregationReport(aggregations=[r])
    data = report.model_dump()
    restored = AggregationReport.model_validate(data)
    assert len(restored.aggregations) == 1
    assert restored.aggregations[0].group_key == "order_flow"
    assert restored.aggregations[0].paired_stats[0].metric_name == "tokens_in"
    assert restored.aggregations[0].paired_stats[0].delta == -50.0


def test_report_to_file_writes_json(tmp_path: Path):
    report = AggregationReport(aggregations=[_make_result()])
    target = tmp_path / "report.json"
    report.to_file(str(target))
    assert target.exists()
    data = json.loads(target.read_text())
    assert data["aggregations"][0]["group_key"] == "order_flow"


def test_report_to_file_appends_extension_when_missing(tmp_path: Path):
    report = AggregationReport(aggregations=[_make_result()])
    target = tmp_path / "report_no_ext"
    report.to_file(str(target))
    assert (tmp_path / "report_no_ext.json").exists()


def test_report_to_file_rejects_non_json_extension(tmp_path: Path):
    report = AggregationReport(aggregations=[_make_result()])
    with pytest.raises(ValueError, match="Only .json format"):
        report.to_file(str(tmp_path / "report.yaml"))


def test_report_from_file_rejects_non_json_extension(tmp_path: Path):
    target = tmp_path / "report.yaml"
    target.write_text("{}")
    with pytest.raises(ValueError, match="Only .json format"):
        AggregationReport.from_file(str(target))


def test_report_from_file_roundtrip(tmp_path: Path):
    r = _make_result(
        paired_stats=[
            PairedComparisonStats(
                metric_name="wall_clock_s",
                baseline_label="baseline",
                variant_label="variant",
                baseline_mean=2.0,
                variant_mean=1.5,
                delta=-0.5,
                test_used="paired_t",
                p_value=0.02,
                ci_low=-0.8,
                ci_high=-0.2,
                n_used=25,
            )
        ]
    )
    report = AggregationReport(aggregations=[r])
    target = tmp_path / "report.json"
    report.to_file(str(target))
    restored = AggregationReport.from_file(str(target))
    assert restored.aggregations[0].paired_stats[0].metric_name == "wall_clock_s"
    assert restored.aggregations[0].paired_stats[0].test_used == "paired_t"


def test_report_handles_empty_aggregations(tmp_path: Path):
    report = AggregationReport(aggregations=[])
    target = tmp_path / "empty.json"
    report.to_file(str(target))
    restored = AggregationReport.from_file(str(target))
    assert restored.aggregations == []
