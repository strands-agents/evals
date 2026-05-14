"""Tests for skills/aggregator.py.

Covers paired statistics correctness against synthetic data, corruption
filtering, raw-value preservation, baseline pairing, and edge cases.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pytest

from strands_evals.skills.aggregator import (
    SkillEvalAggregator,
)
from strands_evals.skills.aggregator_types import (
    SkillEvalAggregationReport,
)
from strands_evals.types.evaluation_report import EvaluationReport


# ---------------------------------------------------------------------------
# Helpers — build synthetic EvaluationReport input
# ---------------------------------------------------------------------------


def _build_report(
    evaluator_name: str,
    rows: list[dict[str, Any]],
) -> EvaluationReport:
    """Build a single EvaluationReport from row dicts.

    Each row is:
        {
            "case_name": str,             # original case name
            "variant_label": str,         # "baseline" | "variant"
            "trial_idx": int,
            "score": float,
            "passed": bool,
            "reason": str,
            "efficiency": dict[str, float] | None,
            "corrupted": bool | None,
            "session_id": str | None,
        }
    """
    cases = []
    scores = []
    passes = []
    reasons = []
    for r in rows:
        metadata = {
            "original_case_name": r["case_name"],
            "variant_label": r["variant_label"],
            "trial_idx": r["trial_idx"],
            "efficiency": r.get("efficiency") or {},
            "corrupted": r.get("corrupted", False),
        }
        cases.append({
            "name": f"{r['case_name']}|{r['variant_label']}|{r['trial_idx']}",
            "metadata": metadata,
            "session_id": r.get("session_id", f"sess-{len(cases)}"),
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


def _make_paired_rows(
    case_name: str,
    n: int,
    baseline_efficiency: dict[str, float],
    variant_efficiency: dict[str, float],
    baseline_passed: bool = True,
    variant_passed: bool = True,
    baseline_score: float = 1.0,
    variant_score: float = 1.0,
    corrupted_trial_indices: set[int] | None = None,
    corruption_side: str = "both",  # "baseline" | "variant" | "both"
) -> list[dict]:
    """Build 2N rows for a single case: N baseline + N variant, all paired."""
    rows = []
    corrupted_trial_indices = corrupted_trial_indices or set()
    for trial_idx in range(n):
        b_corrupted = (
            trial_idx in corrupted_trial_indices and corruption_side in ("baseline", "both")
        )
        v_corrupted = (
            trial_idx in corrupted_trial_indices and corruption_side in ("variant", "both")
        )
        rows.append({
            "case_name": case_name,
            "variant_label": "baseline",
            "trial_idx": trial_idx,
            "score": baseline_score,
            "passed": baseline_passed,
            "efficiency": dict(baseline_efficiency),
            "corrupted": b_corrupted,
        })
        rows.append({
            "case_name": case_name,
            "variant_label": "variant",
            "trial_idx": trial_idx,
            "score": variant_score,
            "passed": variant_passed,
            "efficiency": dict(variant_efficiency),
            "corrupted": v_corrupted,
        })
    return rows


# ---------------------------------------------------------------------------
# Empty input / edge cases
# ---------------------------------------------------------------------------


def test_aggregate_empty_reports_returns_empty_report():
    agg = SkillEvalAggregator()
    report = agg.aggregate([])
    assert isinstance(report, SkillEvalAggregationReport)
    assert report.aggregations == []


def test_aggregate_missing_metadata_skips_cases():
    """Cases without variant_label / trial_idx metadata are skipped."""
    report = EvaluationReport(
        evaluator_name="GoalSuccessRateEvaluator",
        overall_score=0.5,
        scores=[0.5],
        cases=[{"name": "untagged", "metadata": {}}],
        test_passes=[False],
        reasons=[""],
    )
    agg = SkillEvalAggregator()
    out = agg.aggregate([report])
    assert out.aggregations == []


def test_aggregate_no_pairs_when_only_baseline_runs():
    """If every trial is baseline-only, there are zero paired observations."""
    rows = [
        {
            "case_name": "task1",
            "variant_label": "baseline",
            "trial_idx": i,
            "score": 1.0,
            "passed": True,
            "efficiency": {"tokens": 100.0},
        }
        for i in range(5)
    ]
    report = _build_report("GoalSuccessRateEvaluator", rows)
    agg = SkillEvalAggregator()
    out = agg.aggregate([report])
    assert len(out.aggregations) == 1
    assert out.aggregations[0].n_total == 0
    assert out.aggregations[0].n_used == 0
    assert out.aggregations[0].paired_stats == []


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_aggregate_pairs_by_trial_idx():
    """Trials present on both sides are paired; orphans are dropped."""
    rows = []
    # 5 baseline trials, indices 0-4
    for i in range(5):
        rows.append({
            "case_name": "task1",
            "variant_label": "baseline",
            "trial_idx": i,
            "score": 1.0,
            "passed": True,
            "efficiency": {"tokens": 100.0},
        })
    # 4 variant trials, indices 0-3 (one orphan baseline at idx=4)
    for i in range(4):
        rows.append({
            "case_name": "task1",
            "variant_label": "variant",
            "trial_idx": i,
            "score": 1.0,
            "passed": True,
            "efficiency": {"tokens": 90.0},
        })

    report = _build_report("GoalSuccessRateEvaluator", rows)
    agg = SkillEvalAggregator()
    out = agg.aggregate([report])

    assert len(out.aggregations) == 1
    a = out.aggregations[0]
    assert a.n_total == 4  # 4 paired trials, orphan dropped
    assert a.n_used == 4


# ---------------------------------------------------------------------------
# Corruption filtering
# ---------------------------------------------------------------------------


def test_corrupted_pair_dropped_when_baseline_corrupted():
    rows = _make_paired_rows(
        "task1", n=10,
        baseline_efficiency={"tokens": 100.0},
        variant_efficiency={"tokens": 90.0},
        corrupted_trial_indices={2, 7},
        corruption_side="baseline",
    )
    report = _build_report("GoalSuccessRateEvaluator", rows)
    agg = SkillEvalAggregator()
    out = agg.aggregate([report])
    a = out.aggregations[0]
    assert a.n_total == 10
    assert a.n_corrupted == 2
    assert a.n_used == 8


def test_corrupted_pair_dropped_when_variant_corrupted():
    rows = _make_paired_rows(
        "task1", n=10,
        baseline_efficiency={"tokens": 100.0},
        variant_efficiency={"tokens": 90.0},
        corrupted_trial_indices={1, 4, 9},
        corruption_side="variant",
    )
    report = _build_report("GoalSuccessRateEvaluator", rows)
    agg = SkillEvalAggregator()
    out = agg.aggregate([report])
    a = out.aggregations[0]
    assert a.n_corrupted == 3
    assert a.n_used == 7


def test_all_corrupted_yields_no_pairs():
    rows = _make_paired_rows(
        "task1", n=5,
        baseline_efficiency={"tokens": 100.0},
        variant_efficiency={"tokens": 90.0},
        corrupted_trial_indices={0, 1, 2, 3, 4},
    )
    report = _build_report("GoalSuccessRateEvaluator", rows)
    agg = SkillEvalAggregator()
    out = agg.aggregate([report])
    a = out.aggregations[0]
    assert a.n_total == 5
    assert a.n_corrupted == 5
    assert a.n_used == 0
    # No stats can be computed.
    assert a.paired_stats == []


# ---------------------------------------------------------------------------
# Raw values preservation
# ---------------------------------------------------------------------------


def test_raw_values_preserved_in_trial_idx_order():
    """raw_*_values should hold per-trial values in trial_idx order, post-filter."""
    rows = []
    # baseline tokens: [100, 110, 120, 130, 140]
    # variant tokens:  [ 90, 100, 105, 115, 130]
    baseline_tokens = [100.0, 110.0, 120.0, 130.0, 140.0]
    variant_tokens = [90.0, 100.0, 105.0, 115.0, 130.0]
    for i in range(5):
        rows.append({
            "case_name": "task1", "variant_label": "baseline", "trial_idx": i,
            "score": 1.0, "passed": True,
            "efficiency": {"tokens": baseline_tokens[i]},
        })
        rows.append({
            "case_name": "task1", "variant_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True,
            "efficiency": {"tokens": variant_tokens[i]},
        })

    report = _build_report("GoalSuccessRateEvaluator", rows)
    agg = SkillEvalAggregator(metrics=["pass_rate", "tokens"])
    out = agg.aggregate([report])
    a = out.aggregations[0]
    assert a.raw_baseline_values["tokens"] == baseline_tokens
    assert a.raw_variant_values["tokens"] == variant_tokens


def test_missing_efficiency_metric_skips_that_pair():
    """If a trial is missing a metric, that pair is excluded for that metric."""
    rows = [
        {"case_name": "t", "variant_label": "baseline", "trial_idx": 0,
         "score": 1.0, "passed": True, "efficiency": {"tokens": 100.0}},
        {"case_name": "t", "variant_label": "variant", "trial_idx": 0,
         "score": 1.0, "passed": True, "efficiency": {}},  # missing tokens
        {"case_name": "t", "variant_label": "baseline", "trial_idx": 1,
         "score": 1.0, "passed": True, "efficiency": {"tokens": 110.0}},
        {"case_name": "t", "variant_label": "variant", "trial_idx": 1,
         "score": 1.0, "passed": True, "efficiency": {"tokens": 105.0}},
    ]
    report = _build_report("GoalSuccessRateEvaluator", rows)
    agg = SkillEvalAggregator(metrics=["tokens"])
    out = agg.aggregate([report])
    a = out.aggregations[0]
    # Only one valid pair for tokens.
    assert len(a.raw_baseline_values["tokens"]) == 1
    # But with only 1 pair, no stat is computed (needs n >= 2).
    assert a.paired_stats == []


# ---------------------------------------------------------------------------
# Paired statistics correctness
# ---------------------------------------------------------------------------


def test_wilcoxon_detects_systematic_improvement():
    """Variant systematically better — Wilcoxon should reject H0."""
    rng = random.Random(42)
    n = 30
    rows = []
    for i in range(n):
        b = 1000.0 + rng.gauss(0, 50)
        v = b - 100 + rng.gauss(0, 20)  # consistently lower
        rows.append({
            "case_name": "t", "variant_label": "baseline", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"tokens": b},
        })
        rows.append({
            "case_name": "t", "variant_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"tokens": v},
        })
    report = _build_report("Eval", rows)
    agg = SkillEvalAggregator(metrics=["tokens"], stats_test="wilcoxon")
    out = agg.aggregate([report])
    ps = next(p for p in out.aggregations[0].paired_stats if p.metric_name == "tokens")
    assert ps.test_used == "wilcoxon"
    assert ps.p_value < 0.01
    assert ps.delta < 0  # variant uses fewer tokens


def test_paired_t_detects_systematic_improvement():
    rng = random.Random(42)
    n = 30
    rows = []
    for i in range(n):
        b = 2.0 + rng.gauss(0, 0.2)
        v = b - 0.3 + rng.gauss(0, 0.05)
        rows.append({
            "case_name": "t", "variant_label": "baseline", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"latency_s": b},
        })
        rows.append({
            "case_name": "t", "variant_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"latency_s": v},
        })
    report = _build_report("Eval", rows)
    agg = SkillEvalAggregator(metrics=["latency_s"], stats_test="paired_t")
    out = agg.aggregate([report])
    ps = next(p for p in out.aggregations[0].paired_stats if p.metric_name == "latency_s")
    assert ps.test_used == "paired_t"
    assert ps.p_value < 0.01
    assert ps.delta < 0


def test_mcnemar_used_for_pass_rate_regardless_of_setting():
    """pass_rate always uses mcnemar even when stats_test='wilcoxon'."""
    rows = []
    # Variant flips 10 failures to passes; baseline has 10 failures, variant has 0.
    for i in range(20):
        rows.append({
            "case_name": "t", "variant_label": "baseline", "trial_idx": i,
            "score": 0.0 if i < 10 else 1.0,
            "passed": i >= 10,
            "efficiency": {},
        })
        rows.append({
            "case_name": "t", "variant_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {},
        })
    report = _build_report("Eval", rows)
    agg = SkillEvalAggregator(metrics=["pass_rate"], stats_test="wilcoxon")
    out = agg.aggregate([report])
    ps = out.aggregations[0].paired_stats[0]
    assert ps.metric_name == "pass_rate"
    assert ps.test_used == "mcnemar"
    assert ps.p_value < 0.01


def test_mcnemar_no_discordant_pairs_returns_p_1():
    """When every pair agrees, McNemar p-value is 1.0."""
    rows = []
    for i in range(10):
        rows.append({
            "case_name": "t", "variant_label": "baseline", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {},
        })
        rows.append({
            "case_name": "t", "variant_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {},
        })
    report = _build_report("Eval", rows)
    agg = SkillEvalAggregator(metrics=["pass_rate"])
    out = agg.aggregate([report])
    ps = out.aggregations[0].paired_stats[0]
    assert ps.test_used == "mcnemar"
    assert ps.p_value == 1.0
    assert ps.delta == 0.0


def test_no_meaningful_difference_yields_high_p():
    """When variant ≈ baseline, p should not reject H0."""
    rng = random.Random(7)
    rows = []
    for i in range(30):
        b = 1000.0 + rng.gauss(0, 50)
        v = 1000.0 + rng.gauss(0, 50)
        rows.append({
            "case_name": "t", "variant_label": "baseline", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"tokens": b},
        })
        rows.append({
            "case_name": "t", "variant_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"tokens": v},
        })
    report = _build_report("Eval", rows)
    agg = SkillEvalAggregator(metrics=["tokens"], stats_test="wilcoxon")
    out = agg.aggregate([report])
    ps = out.aggregations[0].paired_stats[0]
    assert ps.p_value > 0.05


def test_auto_test_picks_paired_t_for_normal_deltas():
    """When deltas look normal, auto should select paired_t."""
    rng = np.random.default_rng(0)
    n = 30
    deltas = rng.normal(loc=-0.3, scale=0.1, size=n)
    rows = []
    for i in range(n):
        b = 2.0
        v = float(b + deltas[i])
        rows.append({
            "case_name": "t", "variant_label": "baseline", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"latency_s": b},
        })
        rows.append({
            "case_name": "t", "variant_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"latency_s": v},
        })
    report = _build_report("Eval", rows)
    agg = SkillEvalAggregator(metrics=["latency_s"], stats_test="auto")
    out = agg.aggregate([report])
    ps = out.aggregations[0].paired_stats[0]
    assert ps.test_used in {"paired_t", "wilcoxon"}
    # With clearly-Gaussian deltas at this n, auto should usually pick paired_t.
    # We don't enforce that hard because shapiro is noisy.


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_delta():
    """CI should contain the observed delta (mean of deltas)."""
    rng = random.Random(99)
    n = 50
    rows = []
    for i in range(n):
        b = 100.0 + rng.gauss(0, 5)
        v = b - 10 + rng.gauss(0, 2)
        rows.append({
            "case_name": "t", "variant_label": "baseline", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"tokens": b},
        })
        rows.append({
            "case_name": "t", "variant_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"tokens": v},
        })
    report = _build_report("Eval", rows)
    agg = SkillEvalAggregator(metrics=["tokens"])
    out = agg.aggregate([report])
    ps = out.aggregations[0].paired_stats[0]
    # CI should bracket the delta (which is roughly -10).
    assert ps.ci_low < ps.delta < ps.ci_high
    # CI should be on the negative side (variant clearly better).
    assert ps.ci_high < 0


# ---------------------------------------------------------------------------
# Multi-case / multi-evaluator
# ---------------------------------------------------------------------------


def test_multiple_cases_yield_separate_aggregations():
    rows = (
        _make_paired_rows("case_a", n=5,
                          baseline_efficiency={"tokens": 100.0},
                          variant_efficiency={"tokens": 90.0})
        + _make_paired_rows("case_b", n=5,
                            baseline_efficiency={"tokens": 200.0},
                            variant_efficiency={"tokens": 250.0})
    )
    report = _build_report("Eval", rows)
    agg = SkillEvalAggregator(metrics=["tokens"])
    out = agg.aggregate([report])
    assert len(out.aggregations) == 2
    keys = {a.group_key for a in out.aggregations}
    assert keys == {"case_a", "case_b"}


def test_multiple_evaluators_yield_separate_aggregations():
    rows_a = _make_paired_rows("task1", n=5,
                               baseline_efficiency={"tokens": 100.0},
                               variant_efficiency={"tokens": 90.0})
    report1 = _build_report("EvalA", rows_a)
    report2 = _build_report("EvalB", rows_a)
    agg = SkillEvalAggregator(metrics=["tokens"])
    out = agg.aggregate([report1, report2])
    assert len(out.aggregations) == 2
    evaluators = {a.evaluator_name for a in out.aggregations}
    assert evaluators == {"EvalA", "EvalB"}


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_stats_test_raises():
    with pytest.raises(ValueError, match="stats_test must be one of"):
        SkillEvalAggregator(stats_test="bogus")


def test_summarize_reasons_returns_empty_when_no_model():
    agg = SkillEvalAggregator()  # model is None
    assert agg.summarize_reasons(["a", "b"]) == ""


# ---------------------------------------------------------------------------
# Trajectory pointers
# ---------------------------------------------------------------------------


def test_trajectory_pointers_collected_in_pair_order():
    rows = []
    for i in range(3):
        rows.append({
            "case_name": "task", "variant_label": "baseline", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"tokens": 100.0},
            "session_id": f"b{i}",
        })
        rows.append({
            "case_name": "task", "variant_label": "variant", "trial_idx": i,
            "score": 1.0, "passed": True, "efficiency": {"tokens": 90.0},
            "session_id": f"v{i}",
        })
    report = _build_report("Eval", rows)
    agg = SkillEvalAggregator(metrics=["tokens"])
    out = agg.aggregate([report])
    a = out.aggregations[0]
    assert a.trajectory_pointers_baseline == ["b0", "b1", "b2"]
    assert a.trajectory_pointers_variant == ["v0", "v1", "v2"]
