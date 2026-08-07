"""Unit tests for trajectory serialization in judge prompts."""

from strands_evals.evaluators.prompt_templates.trajectory_prompt_template import serialize_trajectory


def test_serialize_trajectory_truncates_oversized_runs():
    """Real runs can exceed any judge context window, so the middle is dropped."""
    huge = [{"role": "user", "content": [{"text": "x" * 900_000}]}]

    serialized = serialize_trajectory(huge)

    assert len(serialized) < 900_000
    assert "characters omitted" in serialized


def test_serialize_trajectory_leaves_normal_runs_intact():
    small = [{"role": "user", "content": [{"text": "do pdf"}]}]

    assert "omitted" not in serialize_trajectory(small)


def test_serialize_trajectory_reports_a_missing_trajectory():
    assert serialize_trajectory(None) == "(no trajectory)"
