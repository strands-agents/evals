"""Tests for aggregators/base.py.

Covers grouping by (case_key, evaluator_name), descriptive stats, reason
aggregation, LLM summary hook, and the subclass extension points
(``_group_key``, ``_filter_entry``, ``_build_result``).
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_evals.aggregators import Aggregator
from strands_evals.aggregators.types import AggregationReport, AggregationResult
from strands_evals.types.evaluation_report import EvaluationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_report(
    evaluator_name: str,
    rows: list[dict[str, Any]],
) -> EvaluationReport:
    """Construct an EvaluationReport from row dicts.

    Each row supports: case_key, score, passed, reason, plus arbitrary
    extra keys that flow through into ``metadata``.
    """
    cases = []
    scores = []
    passes = []
    reasons = []
    for i, r in enumerate(rows):
        meta_passthrough = {
            k: v for k, v in r.items() if k not in {"score", "passed", "reason"}
        }
        metadata = dict(meta_passthrough)
        case_name = r.get("name") or f"{r.get('case_key', 'case')}#{i}"
        cases.append({"name": case_name, "metadata": metadata})
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
# Public API: empty inputs
# ---------------------------------------------------------------------------


def test_aggregate_empty_reports_returns_empty_report():
    report = Aggregator().aggregate([])
    assert isinstance(report, AggregationReport)
    assert report.aggregations == []


def test_constructor_defaults():
    agg = Aggregator()
    assert agg.name == "Aggregator"
    assert agg.model is None
    assert agg.system_prompt  # default prompt is non-empty


def test_constructor_accepts_custom_system_prompt():
    agg = Aggregator(system_prompt="custom")
    assert agg.system_prompt == "custom"


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_groups_by_case_key_and_evaluator():
    rows = [
        {"case_key": "order_flow", "score": 0.9, "passed": True},
        {"case_key": "order_flow", "score": 0.8, "passed": True},
        {"case_key": "stock_check", "score": 0.7, "passed": False},
    ]
    report = Aggregator().aggregate([_build_report("Eval1", rows)])
    keys = {(a.group_key, a.evaluator_name) for a in report.aggregations}
    assert keys == {("order_flow", "Eval1"), ("stock_check", "Eval1")}


def test_groups_by_case_name_when_case_key_missing():
    rows = [
        {"name": "fallback_case", "score": 0.5, "passed": True},
    ]
    report = Aggregator().aggregate([_build_report("Eval1", rows)])
    assert len(report.aggregations) == 1
    assert report.aggregations[0].group_key == "fallback_case"


def test_separates_evaluators():
    rows_a = [{"case_key": "c1", "score": 1.0, "passed": True}]
    rows_b = [{"case_key": "c1", "score": 0.0, "passed": False}]
    report = Aggregator().aggregate(
        [_build_report("EvalA", rows_a), _build_report("EvalB", rows_b)]
    )
    keys = {(a.group_key, a.evaluator_name) for a in report.aggregations}
    assert keys == {("c1", "EvalA"), ("c1", "EvalB")}


def test_aggregations_sorted_by_group_then_evaluator():
    rows = [
        {"case_key": "z_case", "score": 1.0, "passed": True},
        {"case_key": "a_case", "score": 1.0, "passed": True},
        {"case_key": "m_case", "score": 1.0, "passed": True},
    ]
    report = Aggregator().aggregate([_build_report("Eval1", rows)])
    order = [a.group_key for a in report.aggregations]
    assert order == ["a_case", "m_case", "z_case"]


def test_entries_without_case_key_or_name_are_skipped():
    """A case with neither metadata.case_key nor name is dropped."""
    report = EvaluationReport(
        evaluator_name="Eval1",
        overall_score=0.5,
        scores=[0.5],
        cases=[{"name": "", "metadata": {}}],
        test_passes=[True],
        reasons=[""],
    )
    out = Aggregator().aggregate([report])
    assert out.aggregations == []


# ---------------------------------------------------------------------------
# Descriptive stats
# ---------------------------------------------------------------------------


def test_descriptive_stats_computed():
    rows = [
        {"case_key": "c1", "score": 0.5, "passed": True},
        {"case_key": "c1", "score": 0.7, "passed": False},
        {"case_key": "c1", "score": 0.9, "passed": True},
    ]
    report = Aggregator().aggregate([_build_report("Eval1", rows)])
    a = report.aggregations[0]
    assert a.num_results == 3
    assert a.num_passed == 2
    assert a.num_failed == 1
    assert a.mean_score == pytest.approx(0.7)
    assert a.min_score == pytest.approx(0.5)
    assert a.max_score == pytest.approx(0.9)
    assert a.pass_rate == pytest.approx(2 / 3)


def test_single_trial_group():
    rows = [{"case_key": "c1", "score": 0.42, "passed": False}]
    report = Aggregator().aggregate([_build_report("Eval1", rows)])
    a = report.aggregations[0]
    assert a.num_results == 1
    assert a.mean_score == 0.42
    assert a.min_score == 0.42
    assert a.max_score == 0.42
    assert a.pass_rate == 0.0


# ---------------------------------------------------------------------------
# Reasons aggregation
# ---------------------------------------------------------------------------


def test_reasons_aggregated_when_present():
    rows = [
        {"case_key": "c1", "score": 1.0, "passed": True, "reason": "looks good"},
        {"case_key": "c1", "score": 0.0, "passed": False, "reason": "wrong tool"},
    ]
    report = Aggregator().aggregate([_build_report("Eval1", rows)])
    assert report.aggregations[0].reasons == ["looks good", "wrong tool"]


def test_empty_reasons_are_filtered():
    rows = [
        {"case_key": "c1", "score": 1.0, "passed": True, "reason": ""},
        {"case_key": "c1", "score": 0.5, "passed": True, "reason": "ok"},
    ]
    report = Aggregator().aggregate([_build_report("Eval1", rows)])
    assert report.aggregations[0].reasons == ["ok"]


# ---------------------------------------------------------------------------
# LLM summary hook
# ---------------------------------------------------------------------------


def test_summary_empty_without_model():
    rows = [{"case_key": "c1", "score": 1.0, "passed": True}]
    report = Aggregator().aggregate([_build_report("Eval1", rows)])
    assert report.aggregations[0].summary == ""


def test_summary_invoked_when_model_present():
    """When model is set, _invoke_summary_model is called and its output lands on the result."""

    class StubAgg(Aggregator):
        def _invoke_summary_model(self, prompt: str) -> str:
            self.last_prompt = prompt
            return "summary text"

    agg = StubAgg(model="stub-model")
    rows = [{"case_key": "c1", "score": 1.0, "passed": True, "reason": "r"}]
    report = agg.aggregate([_build_report("Eval1", rows)])
    assert report.aggregations[0].summary == "summary text"
    assert "c1" in agg.last_prompt
    assert "Eval1" in agg.last_prompt


# ---------------------------------------------------------------------------
# Extension hooks (subclass overrides)
# ---------------------------------------------------------------------------


def test_group_key_override_changes_grouping():
    """A subclass overriding _group_key can group by an extra metadata field."""

    class ByConditionAggregator(Aggregator):
        def _group_key(self, entry):
            base = super()._group_key(entry)
            if base is None:
                return None
            condition = entry["metadata"].get("condition_label")
            if condition is None:
                return base
            return (base[0], condition, base[1])

        @staticmethod
        def _format_group_key(group_key):
            # Render as "case_key / condition" for display.
            return f"{group_key[0]} / {group_key[1]}"

    rows = [
        {"case_key": "c1", "condition_label": "A", "score": 0.9, "passed": True},
        {"case_key": "c1", "condition_label": "A", "score": 0.8, "passed": True},
        {"case_key": "c1", "condition_label": "B", "score": 0.4, "passed": False},
        {"case_key": "c1", "condition_label": "B", "score": 0.5, "passed": True},
    ]
    report = ByConditionAggregator().aggregate([_build_report("Eval1", rows)])
    group_keys = {a.group_key for a in report.aggregations}
    assert group_keys == {"c1 / A", "c1 / B"}

    a_result = next(a for a in report.aggregations if a.group_key == "c1 / A")
    b_result = next(a for a in report.aggregations if a.group_key == "c1 / B")
    assert a_result.num_results == 2
    assert b_result.num_results == 2
    assert a_result.mean_score == pytest.approx(0.85)


def test_filter_entry_override_drops_corrupted():
    """A subclass overriding _filter_entry can drop corrupted trials before aggregation."""

    class CorruptionFilterAggregator(Aggregator):
        def _filter_entry(self, entry) -> bool:
            return not entry["metadata"].get("corrupted", False)

    rows = [
        {"case_key": "c1", "score": 0.9, "passed": True, "corrupted": False},
        {"case_key": "c1", "score": 0.0, "passed": False, "corrupted": True},
        {"case_key": "c1", "score": 0.8, "passed": True, "corrupted": False},
    ]
    report = CorruptionFilterAggregator().aggregate([_build_report("Eval1", rows)])
    a = report.aggregations[0]
    assert a.num_results == 2
    assert a.mean_score == pytest.approx(0.85)


def test_filter_entry_dropping_all_yields_no_aggregation():
    """When _filter_entry drops every entry in a group, that group is omitted."""

    class DropAllAggregator(Aggregator):
        def _filter_entry(self, entry) -> bool:
            return False

    rows = [{"case_key": "c1", "score": 0.9, "passed": True}]
    report = DropAllAggregator().aggregate([_build_report("Eval1", rows)])
    assert report.aggregations == []


def test_build_result_override_attaches_extra_fields():
    """A subclass overriding _build_result can attach project-specific fields via extra."""

    class WinRateAggregator(Aggregator):
        def _build_result(self, group_key, entries):
            result = super()._build_result(group_key, entries)
            wins = sum(1 for e in entries if e["passed"])
            result.extra["win_rate"] = wins / len(entries)
            return result

    rows = [
        {"case_key": "c1", "score": 1.0, "passed": True},
        {"case_key": "c1", "score": 0.0, "passed": False},
        {"case_key": "c1", "score": 1.0, "passed": True},
    ]
    report = WinRateAggregator().aggregate([_build_report("Eval1", rows)])
    a = report.aggregations[0]
    assert a.extra["win_rate"] == pytest.approx(2 / 3)
    # Base stats should still be populated.
    assert a.num_results == 3


# ---------------------------------------------------------------------------
# Multiple reports
# ---------------------------------------------------------------------------


def test_aggregates_across_multiple_reports_for_same_group():
    """Two reports for the same evaluator and case should merge into one group."""
    rows_1 = [{"case_key": "c1", "score": 0.4, "passed": False}]
    rows_2 = [{"case_key": "c1", "score": 0.8, "passed": True}]
    report = Aggregator().aggregate(
        [_build_report("Eval1", rows_1), _build_report("Eval1", rows_2)]
    )
    assert len(report.aggregations) == 1
    a = report.aggregations[0]
    assert a.num_results == 2
    assert a.mean_score == pytest.approx(0.6)
