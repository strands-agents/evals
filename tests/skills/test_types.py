"""Tests for skills/aggregator_types.py."""

import json
from pathlib import Path

import pytest

from strands_evals.skills.aggregator_types import (
    PairedComparisonStats,
    SkillEvalAggregation,
    SkillEvalAggregationReport,
)


# ---------------------------------------------------------------------------
# PairedComparisonStats
# ---------------------------------------------------------------------------


def test_paired_comparison_stats_construction():
    ps = PairedComparisonStats(
        metric_name="tokens",
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
    assert ps.metric_name == "tokens"
    assert ps.delta == -100.0
    assert ps.test_used == "wilcoxon"


def test_paired_comparison_stats_defaults():
    """delta_pct and n_corrupted default to sensible values."""
    ps = PairedComparisonStats(
        metric_name="latency_s",
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
# SkillEvalAggregation
# ---------------------------------------------------------------------------


def _make_aggregation(**overrides) -> SkillEvalAggregation:
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
    return SkillEvalAggregation(**base)


def test_skill_aggregation_inherits_base_fields():
    """SkillEvalAggregation must carry all AggregationResult fields."""
    agg = _make_aggregation()
    assert agg.group_key == "order_flow"
    assert agg.mean_score == 0.85
    assert agg.pass_rate == 0.9
    # And its own fields with defaults.
    assert agg.paired_stats == []
    assert agg.raw_baseline_values == {}
    assert agg.n_total == 0
    assert agg.n_used == 0


def test_skill_aggregation_with_paired_stats():
    ps = PairedComparisonStats(
        metric_name="pass_rate",
        baseline_mean=0.7,
        variant_mean=0.88,
        delta=0.18,
        test_used="mcnemar",
        p_value=0.003,
        ci_low=0.08,
        ci_high=0.28,
        n_used=60,
        n_corrupted=4,
    )
    agg = _make_aggregation(
        paired_stats=[ps],
        n_total=64,
        n_corrupted=4,
        n_used=60,
        raw_baseline_values={"pass_rate": [1.0, 0.0, 1.0]},
        raw_variant_values={"pass_rate": [1.0, 1.0, 1.0]},
        trajectory_pointers_baseline=["s1", "s2", "s3"],
        trajectory_pointers_variant=["s4", "s5", "s6"],
    )
    assert len(agg.paired_stats) == 1
    assert agg.paired_stats[0].metric_name == "pass_rate"
    assert agg.n_total == 64
    assert agg.raw_baseline_values["pass_rate"] == [1.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# SkillEvalAggregationReport — serialization
# ---------------------------------------------------------------------------


def test_report_roundtrip_via_dict():
    """model_dump → model_validate produces an equivalent report."""
    agg = _make_aggregation(
        paired_stats=[
            PairedComparisonStats(
                metric_name="tokens",
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
    report = SkillEvalAggregationReport(aggregations=[agg])
    data = report.model_dump()
    restored = SkillEvalAggregationReport.model_validate(data)
    assert len(restored.aggregations) == 1
    assert restored.aggregations[0].group_key == "order_flow"
    assert restored.aggregations[0].paired_stats[0].metric_name == "tokens"
    assert restored.aggregations[0].paired_stats[0].delta == -50.0


def test_report_to_file_writes_json(tmp_path: Path):
    agg = _make_aggregation()
    report = SkillEvalAggregationReport(aggregations=[agg])
    target = tmp_path / "report.json"
    report.to_file(str(target))
    assert target.exists()
    data = json.loads(target.read_text())
    assert data["aggregations"][0]["group_key"] == "order_flow"


def test_report_to_file_appends_extension_when_missing(tmp_path: Path):
    agg = _make_aggregation()
    report = SkillEvalAggregationReport(aggregations=[agg])
    target = tmp_path / "report_no_ext"
    report.to_file(str(target))
    assert (tmp_path / "report_no_ext.json").exists()


def test_report_to_file_rejects_non_json_extension(tmp_path: Path):
    agg = _make_aggregation()
    report = SkillEvalAggregationReport(aggregations=[agg])
    with pytest.raises(ValueError, match="Only .json format"):
        report.to_file(str(tmp_path / "report.yaml"))


def test_report_from_file_rejects_non_json_extension(tmp_path: Path):
    target = tmp_path / "report.yaml"
    target.write_text("{}")
    with pytest.raises(ValueError, match="Only .json format"):
        SkillEvalAggregationReport.from_file(str(target))


def test_report_from_file_roundtrip(tmp_path: Path):
    agg = _make_aggregation(
        paired_stats=[
            PairedComparisonStats(
                metric_name="latency_s",
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
    report = SkillEvalAggregationReport(aggregations=[agg])
    target = tmp_path / "report.json"
    report.to_file(str(target))

    restored = SkillEvalAggregationReport.from_file(str(target))
    assert restored.aggregations[0].paired_stats[0].metric_name == "latency_s"
    assert restored.aggregations[0].paired_stats[0].test_used == "paired_t"


def test_report_handles_empty_aggregations(tmp_path: Path):
    report = SkillEvalAggregationReport(aggregations=[])
    target = tmp_path / "empty.json"
    report.to_file(str(target))
    restored = SkillEvalAggregationReport.from_file(str(target))
    assert restored.aggregations == []
