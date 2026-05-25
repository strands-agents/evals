"""Tests for skills/experiment.py."""

import pytest

from strands_evals.case import Case
from strands_evals.skills.aggregator import SkillEvalAggregator
from strands_evals.skills.experiment import SkillEvalExperiment


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_invalid_num_trials_raises():
    with pytest.raises(ValueError, match="num_trials must be >= 1"):
        SkillEvalExperiment(cases=[], num_trials=0)


def test_insufficient_variant_labels_raises():
    with pytest.raises(ValueError, match="at least two labels"):
        SkillEvalExperiment(
            cases=[Case[str, str](name="t", input="x")],
            variant_labels=["baseline_only"],
        )


# ---------------------------------------------------------------------------
# Case expansion
# ---------------------------------------------------------------------------


def test_case_expansion_count_matches_cases_times_variants_times_trials():
    cases = [
        Case[str, str](name="task_a", input="x"),
        Case[str, str](name="task_b", input="y"),
    ]
    experiment = SkillEvalExperiment(
        cases=cases,
        variant_labels=["baseline", "variant"],
        num_trials=5,
    )
    # 2 cases × 2 variants × 5 trials = 20
    assert len(experiment._expanded_cases) == 20


def test_expanded_cases_carry_correct_metadata():
    cases = [Case[str, str](name="task_a", input="x")]
    experiment = SkillEvalExperiment(
        cases=cases,
        variant_labels=["baseline", "variant"],
        num_trials=3,
    )
    metas = [c.metadata for c in experiment._expanded_cases]
    # All carry original_case_name = "task_a"
    assert all(m["original_case_name"] == "task_a" for m in metas)
    # variant_label values come from variant_labels list
    variant_labels = sorted({m["variant_label"] for m in metas})
    assert variant_labels == ["baseline", "variant"]
    # trial_idx covers 0..num_trials-1 for each variant
    by_variant: dict[str, list[int]] = {"baseline": [], "variant": []}
    for m in metas:
        by_variant[m["variant_label"]].append(m["trial_idx"])
    assert sorted(by_variant["baseline"]) == [0, 1, 2]
    assert sorted(by_variant["variant"]) == [0, 1, 2]


def test_expanded_case_names_include_variant_and_trial():
    cases = [Case[str, str](name="task_a", input="x")]
    experiment = SkillEvalExperiment(
        cases=cases,
        variant_labels=["baseline", "variant"],
        num_trials=2,
    )
    names = sorted(c.name for c in experiment._expanded_cases)
    assert names == [
        "task_a|baseline|0",
        "task_a|baseline|1",
        "task_a|variant|0",
        "task_a|variant|1",
    ]


def test_expanded_cases_get_unique_session_ids():
    cases = [Case[str, str](name="task_a", input="x")]
    experiment = SkillEvalExperiment(
        cases=cases,
        variant_labels=["baseline", "variant"],
        num_trials=4,
    )
    session_ids = [c.session_id for c in experiment._expanded_cases]
    assert len(set(session_ids)) == len(session_ids)


def test_user_metadata_preserved_in_expansion():
    """Pre-existing case.metadata fields survive expansion."""
    cases = [
        Case[str, str](
            name="task_a", input="x", metadata={"category": "math", "weight": 2.0}
        )
    ]
    experiment = SkillEvalExperiment(
        cases=cases,
        variant_labels=["baseline", "variant"],
        num_trials=1,
    )
    for c in experiment._expanded_cases:
        assert c.metadata["category"] == "math"
        assert c.metadata["weight"] == 2.0
        # And the expansion fields are added.
        assert "variant_label" in c.metadata
        assert "trial_idx" in c.metadata


def test_unnamed_case_expansion_uses_variant_trial_only():
    cases = [Case[str, str](input="x")]
    experiment = SkillEvalExperiment(
        cases=cases,
        variant_labels=["baseline", "variant"],
        num_trials=1,
    )
    names = sorted(c.name for c in experiment._expanded_cases)
    assert names == ["baseline|0", "variant|0"]


# ---------------------------------------------------------------------------
# Aggregation hookup
# ---------------------------------------------------------------------------


def test_aggregate_evaluations_without_aggregator_raises():
    experiment = SkillEvalExperiment(
        cases=[Case[str, str](name="t", input="x")],
        num_trials=1,
    )
    with pytest.raises(RuntimeError, match="No aggregator configured"):
        experiment.aggregate_evaluations()


def test_aggregate_evaluations_without_run_raises():
    experiment = SkillEvalExperiment(
        cases=[Case[str, str](name="t", input="x")],
        num_trials=1,
        aggregator=SkillEvalAggregator(),
    )
    with pytest.raises(RuntimeError, match="No evaluation reports available"):
        experiment.aggregate_evaluations()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_properties_expose_constructor_values():
    cases = [Case[str, str](name="t", input="x")]
    experiment = SkillEvalExperiment(
        cases=cases,
        variant_labels=["a", "b", "c"],
        num_trials=7,
    )
    assert experiment.cases == cases
    assert experiment.variant_labels == ["a", "b", "c"]
    assert experiment.num_trials == 7


def test_variant_labels_returns_a_copy():
    """Mutating the returned list must not affect the experiment."""
    experiment = SkillEvalExperiment(
        cases=[Case[str, str](name="t", input="x")],
        variant_labels=["a", "b"],
        num_trials=1,
    )
    labels = experiment.variant_labels
    labels.append("c")
    assert experiment.variant_labels == ["a", "b"]
