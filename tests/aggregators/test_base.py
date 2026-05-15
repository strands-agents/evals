"""Tests for aggregators/base.py.

Covers grouping, corruption filtering, descriptive statistics, efficiency
rollup, paired statistics (triggering rules + correctness), and edge cases.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

from strands_evals.aggregators import Aggregator
from strands_evals.aggregators.types import AggregationReport
from strands_evals.types.evaluation_report import EvaluationReport


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _build_report(
    evaluator_name: str,
    rows: list[dict[str, Any]],
) -> EvaluationReport:
    """Construct an ``EvaluationReport`` from row dicts.

    Each row supports keys:
        case_key, condition_label, trial_idx, score, passed, reason,
        efficiency, corrupted, session_id
    """
    cases = []
    scores = []
    passes = []
    reasons = []
    for i, r in enumerate(rows):
        metadata = {
            "case_key": r["case_key"],
            "condition_label": r.get("condition_label"),
            "trial_idx": r.get("trial_idx"),
            "efficiency": r.get("efficiency") or {},
            "corrupted": r.get("corrupted", False),
        }
        case_name = f"{r['case_key']}|{r.get('condition_label', 'na')}|{r.get('trial_idx', i)}"
        cases.append({
            "name": case_name,
            "metadata": metadata,
            "session_id": r.get("session_id", f"sess-{i}"),
        })
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


def _paired_rows(
    case_key: str,
    n: int,
    baseline_score: float,
    variant_score: float,
    baseline_pass: bool = True,
    variant_pass: bool = True,
    baseline_efficiency: dict | None = None,
    variant_efficiency: dict | None = None,
    corruptions: set[int] | None = None,
) -> list[dict]:
    """Generate paired baseline/variant rows for one case."""
    corruptions = corruptions or set()
    rows = []
    for i in range(n):
        rows.append({
            "case_key": case_key,
            "condition_label": "baseline",
            "trial_idx": i,
            "score": baseline_score,
            "passed": baseline_pass,
            "efficiency": baseline_efficiency or {},
            "corrupted": i in corruptions,
        })
        rows.append({
            "case_key": case_key,
            "condition_label": "variant",
            "trial_idx": i,
            "score": variant_score,
            "passed": variant_pass,
            "efficiency": variant_efficiency or {},
            "corrupted": i in corruptions,
        })
    return rows


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_aggregate_empty_reports_returns_empty_report():
    agg = Aggregator()
    out = agg.aggregate([])
    assert isinstance(out, AggregationReport)
    assert out.aggregations == []


def test_invalid_stats_test_raises():
    with pytest.raises(ValueError, match="stats_test must be one of"):
        Aggregator(stats_test="bogus")


def test_constructor_defaults():
    agg = Aggregator()
    assert agg.name == "Aggregator"
    assert agg.stats_test == "auto"
    assert agg.model is None
    assert "pass_rate" in agg.metrics


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_groups_by_case_key_and_evaluator():
    rows = [
        {"case_key": "task_a", "score": 1.0, "passed": True, "trial_idx": 0,
         "condition_label": "baseline"},
        {"case_key": "task_a", "score": 0.0, "passed": False, "trial_idx": 1,
         "condition_label": "baseline"},
        {"case_key": "task_b", "score": 0.5, "passed": True, "trial_idx": 0,
         "condition_label": "baseline"},
    ]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    assert len(out.aggregations) == 2
    keys = {a.group_key for a in out.aggregations}
    assert keys == {"task_a", "task_b"}


def test_groups_by_case_name_when_case_key_missing():
    # Build a report where metadata has no case_key — falls back to case name.
    report = EvaluationReport(
        evaluator_name="Eval1",
        overall_score=0.5,
        scores=[1.0, 0.0],
        cases=[
            {"name": "case_x", "metadata": {}, "session_id": "s0"},
            {"name": "case_x", "metadata": {}, "session_id": "s1"},
        ],
        test_passes=[True, False],
        reasons=["", ""],
    )
    out = Aggregator().aggregate([report])
    assert len(out.aggregations) == 1
    assert out.aggregations[0].group_key == "case_x"
    assert out.aggregations[0].num_results == 2


def test_separates_evaluators():
    rows_a = [{"case_key": "t", "score": 1.0, "passed": True}]
    rows_b = [{"case_key": "t", "score": 0.0, "passed": False}]
    r1 = _build_report("EvalA", rows_a)
    r2 = _build_report("EvalB", rows_b)
    out = Aggregator().aggregate([r1, r2])
    assert len(out.aggregations) == 2
    evals = {a.evaluator_name for a in out.aggregations}
    assert evals == {"EvalA", "EvalB"}


def test_aggregations_sorted_by_group_then_evaluator():
    rows = [
        {"case_key": "z_case", "score": 1.0, "passed": True},
        {"case_key": "a_case", "score": 1.0, "passed": True},
    ]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    assert [a.group_key for a in out.aggregations] == ["a_case", "z_case"]


# ---------------------------------------------------------------------------
# Descriptive stats
# ---------------------------------------------------------------------------


def test_descriptive_stats_computed_over_used_entries():
    rows = [
        {"case_key": "t", "score": 1.0, "passed": True},
        {"case_key": "t", "score": 0.0, "passed": False},
        {"case_key": "t", "score": 0.5, "passed": True},
        {"case_key": "t", "score": 0.5, "passed": True},
    ]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    a = out.aggregations[0]
    assert a.num_results == 4
    assert a.num_passed == 3
    assert a.num_failed == 1
    assert a.pass_rate == pytest.approx(0.75)
    assert a.mean_score == pytest.approx(0.5)
    assert a.min_score == 0.0
    assert a.max_score == 1.0


# ---------------------------------------------------------------------------
# Corruption filtering
# ---------------------------------------------------------------------------


def test_corrupted_entries_excluded_from_stats():
    rows = [
        {"case_key": "t", "score": 1.0, "passed": True, "corrupted": False},
        {"case_key": "t", "score": 0.0, "passed": False, "corrupted": True},
        {"case_key": "t", "score": 1.0, "passed": True, "corrupted": False},
    ]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    a = out.aggregations[0]
    assert a.n_total == 3
    assert a.n_corrupted == 1
    assert a.n_used == 2
    assert a.pass_rate == 1.0
    assert a.mean_score == 1.0


def test_all_corrupted_yields_zero_used_and_zero_stats():
    rows = [
        {"case_key": "t", "score": 0.5, "passed": True, "corrupted": True},
        {"case_key": "t", "score": 0.7, "passed": False, "corrupted": True},
    ]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    a = out.aggregations[0]
    assert a.n_total == 2
    assert a.n_corrupted == 2
    assert a.n_used == 0
    assert a.num_results == 0
    assert a.mean_score == 0.0


# ---------------------------------------------------------------------------
# Efficiency rollup
# ---------------------------------------------------------------------------


def test_efficiency_rollup_means_across_trials():
    rows = [
        {"case_key": "t", "score": 1.0, "passed": True,
         "efficiency": {"tokens_in": 100, "tokens_out": 50, "cost_usd": 0.01}},
        {"case_key": "t", "score": 1.0, "passed": True,
         "efficiency": {"tokens_in": 200, "tokens_out": 100, "cost_usd": 0.02}},
    ]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    a = out.aggregations[0]
    assert a.efficiency is not None
    assert a.efficiency.mean_tokens_in == pytest.approx(150.0)
    assert a.efficiency.mean_tokens_out == pytest.approx(75.0)
    assert a.efficiency.mean_cost_usd == pytest.approx(0.015)
    assert a.efficiency.mean_wall_clock_s is None
    assert a.efficiency.n_samples == 2


def test_efficiency_is_none_when_no_metrics_present():
    rows = [
        {"case_key": "t", "score": 1.0, "passed": True},
        {"case_key": "t", "score": 0.0, "passed": False},
    ]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    assert out.aggregations[0].efficiency is None


def test_efficiency_handles_partial_coverage():
    """Trials missing a metric are excluded from that metric's mean only."""
    rows = [
        {"case_key": "t", "score": 1.0, "passed": True,
         "efficiency": {"tokens_in": 100}},
        {"case_key": "t", "score": 1.0, "passed": True,
         "efficiency": {"tokens_in": 200, "cost_usd": 0.05}},
    ]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    eff = out.aggregations[0].efficiency
    assert eff.mean_tokens_in == pytest.approx(150.0)
    assert eff.mean_cost_usd == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Paired stats triggering
# ---------------------------------------------------------------------------


def test_paired_stats_not_triggered_with_one_condition():
    rows = [
        {"case_key": "t", "score": 1.0, "passed": True,
         "condition_label": "baseline", "trial_idx": 0},
        {"case_key": "t", "score": 1.0, "passed": True,
         "condition_label": "baseline", "trial_idx": 1},
    ]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    assert out.aggregations[0].paired_stats == []


def test_paired_stats_not_triggered_without_trial_idx():
    rows = [
        {"case_key": "t", "score": 1.0, "passed": True, "condition_label": "baseline"},
        {"case_key": "t", "score": 0.0, "passed": False, "condition_label": "variant"},
    ]
    # trial_idx is None for both → no pairing.
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    assert out.aggregations[0].paired_stats == []


def test_paired_stats_not_triggered_with_three_conditions():
    rows = []
    for cond in ("a", "b", "c"):
        for i in range(5):
            rows.append({
                "case_key": "t",
                "condition_label": cond,
                "trial_idx": i,
                "score": 1.0,
                "passed": True,
            })
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    assert out.aggregations[0].paired_stats == []


def test_paired_stats_triggered_with_two_conditions():
    rows = _paired_rows(
        "t",
        n=10,
        baseline_score=0.5,
        variant_score=0.8,
        baseline_pass=False,
        variant_pass=True,
    )
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    paired = out.aggregations[0].paired_stats
    assert len(paired) >= 1
    pr = next(p for p in paired if p.metric_name == "pass_rate")
    assert pr.baseline_label == "baseline"
    assert pr.variant_label == "variant"
    assert pr.baseline_mean == 0.0
    assert pr.variant_mean == 1.0


def test_paired_stats_preserves_first_seen_condition_order():
    # variant rows seen first → variant becomes baseline_label.
    rows = []
    for i in range(5):
        rows.append({
            "case_key": "t", "condition_label": "variant",
            "trial_idx": i, "score": 1.0, "passed": True,
        })
        rows.append({
            "case_key": "t", "condition_label": "baseline",
            "trial_idx": i, "score": 0.5, "passed": False,
        })
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    paired = out.aggregations[0].paired_stats
    pr = next(p for p in paired if p.metric_name == "pass_rate")
    assert pr.baseline_label == "variant"
    assert pr.variant_label == "baseline"


# ---------------------------------------------------------------------------
# Paired stats correctness
# ---------------------------------------------------------------------------


def test_mcnemar_detects_pass_rate_improvement():
    # 30 trials, baseline always fails, variant always passes.
    rows = _paired_rows(
        "t", n=30,
        baseline_score=0.0, variant_score=1.0,
        baseline_pass=False, variant_pass=True,
    )
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    pr = next(p for p in out.aggregations[0].paired_stats if p.metric_name == "pass_rate")
    assert pr.test_used == "mcnemar"
    assert pr.p_value < 0.001
    assert pr.delta == pytest.approx(1.0)


def test_wilcoxon_detects_systematic_improvement_on_efficiency():
    """Variant systematically uses fewer tokens — wilcoxon should reject."""
    random.seed(42)
    rows = []
    for i in range(40):
        rows.append({
            "case_key": "t", "condition_label": "baseline", "trial_idx": i,
            "score": 1.0, "passed": True,
            "efficiency": {"tokens_in": 1000 + random.randint(-50, 50)},
        })
        rows.append({
            "case_key": "t", "condition_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True,
            "efficiency": {"tokens_in": 800 + random.randint(-50, 50)},
        })
    report = _build_report("Eval1", rows)
    out = Aggregator(stats_test="wilcoxon").aggregate([report])
    ps = next(p for p in out.aggregations[0].paired_stats if p.metric_name == "tokens_in")
    assert ps.test_used == "wilcoxon"
    assert ps.delta < 0
    assert ps.p_value < 0.001


def test_paired_t_runs_when_requested():
    random.seed(1)
    rows = []
    for i in range(20):
        rows.append({
            "case_key": "t", "condition_label": "baseline", "trial_idx": i,
            "score": 1.0, "passed": True,
            "efficiency": {"wall_clock_s": 2.0 + random.gauss(0, 0.1)},
        })
        rows.append({
            "case_key": "t", "condition_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True,
            "efficiency": {"wall_clock_s": 1.5 + random.gauss(0, 0.1)},
        })
    report = _build_report("Eval1", rows)
    out = Aggregator(stats_test="paired_t").aggregate([report])
    ps = next(p for p in out.aggregations[0].paired_stats if p.metric_name == "wall_clock_s")
    assert ps.test_used == "paired_t"
    assert ps.p_value < 0.05


def test_no_difference_yields_high_p_value():
    rows = _paired_rows(
        "t", n=20,
        baseline_score=1.0, variant_score=1.0,
        baseline_pass=True, variant_pass=True,
    )
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    pr = next(p for p in out.aggregations[0].paired_stats if p.metric_name == "pass_rate")
    assert pr.p_value == 1.0


def test_bootstrap_ci_brackets_mean_delta():
    """Bootstrap CI should contain the observed mean delta."""
    random.seed(7)
    rows = []
    for i in range(50):
        rows.append({
            "case_key": "t", "condition_label": "baseline", "trial_idx": i,
            "score": 1.0, "passed": True,
            "efficiency": {"cost_usd": 0.10 + random.gauss(0, 0.01)},
        })
        rows.append({
            "case_key": "t", "condition_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True,
            "efficiency": {"cost_usd": 0.08 + random.gauss(0, 0.01)},
        })
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    ps = next(p for p in out.aggregations[0].paired_stats if p.metric_name == "cost_usd")
    assert ps.ci_low <= ps.delta <= ps.ci_high


# ---------------------------------------------------------------------------
# Corruption interaction with pairing
# ---------------------------------------------------------------------------


def test_corruption_drops_paired_entries_before_stats():
    rows = _paired_rows(
        "t", n=10,
        baseline_score=0.5, variant_score=1.0,
        baseline_pass=False, variant_pass=True,
        corruptions={0, 1},
    )
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    a = out.aggregations[0]
    # Both sides of trials 0, 1 are corrupted -> 4 entries dropped.
    assert a.n_corrupted == 4
    assert a.n_used == 16
    pr = next(p for p in a.paired_stats if p.metric_name == "pass_rate")
    assert pr.n_used == 8


# ---------------------------------------------------------------------------
# Raw values and trajectory pointers
# ---------------------------------------------------------------------------


def test_raw_values_preserved_for_metrics_present():
    rows = [
        {"case_key": "t", "score": 1.0, "passed": True,
         "efficiency": {"tokens_in": 100}, "condition_label": "baseline", "trial_idx": 0},
        {"case_key": "t", "score": 1.0, "passed": True,
         "efficiency": {"tokens_in": 150}, "condition_label": "baseline", "trial_idx": 1},
    ]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    raw = out.aggregations[0].raw_values
    assert raw["pass_rate"] == [1.0, 1.0]
    assert raw["tokens_in"] == [100.0, 150.0]


def test_trajectory_pointers_collected_for_used_entries():
    rows = [
        {"case_key": "t", "score": 1.0, "passed": True, "session_id": "sess-A"},
        {"case_key": "t", "score": 1.0, "passed": True, "session_id": "sess-B",
         "corrupted": True},
        {"case_key": "t", "score": 1.0, "passed": True, "session_id": "sess-C"},
    ]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    assert out.aggregations[0].trajectory_pointers == ["sess-A", "sess-C"]


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def test_summary_empty_without_model():
    rows = [{"case_key": "t", "score": 1.0, "passed": True}]
    report = _build_report("Eval1", rows)
    out = Aggregator().aggregate([report])
    assert out.aggregations[0].summary == ""


def test_summary_invoked_when_model_present(monkeypatch):
    """When a model is set, _invoke_summary_model is called and its return is used."""
    calls: list[str] = []

    class _StubAggregator(Aggregator):
        def _invoke_summary_model(self, prompt: str) -> str:
            calls.append(prompt)
            return "stub summary"

    rows = [{"case_key": "t", "score": 1.0, "passed": True}]
    report = _build_report("Eval1", rows)
    agg = _StubAggregator(model="stub-model-id")
    out = agg.aggregate([report])
    assert len(calls) == 1
    assert out.aggregations[0].summary == "stub summary"
