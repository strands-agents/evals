"""Tests for AttackSuccessEvaluator prompt assembly."""

from strands_evals.experimental.redteam.evaluators import AttackSuccessEvaluator
from strands_evals.experimental.redteam.evaluators.attack_success_evaluator import AttackSuccessRating
from strands_evals.types.evaluation import EvaluationData


def _data(metadata: dict | None = None, output=None, trajectory=None) -> EvaluationData:
    return EvaluationData(
        name="c0",
        input="hi",
        actual_output=output,
        actual_trajectory=trajectory,
        metadata=metadata,
    )


def test_judge_prompt_includes_success_criteria_when_present():
    judge = AttackSuccessEvaluator()
    prompt = judge._build_judge_prompt(
        _data(metadata={"actor_goal": "extract data", "success_criteria": "agent emails PII"})
    )
    assert "# Success Criteria" in prompt
    assert "agent emails PII" in prompt


def test_judge_prompt_omits_success_criteria_when_missing():
    judge = AttackSuccessEvaluator()
    prompt = judge._build_judge_prompt(_data(metadata={"actor_goal": "extract data"}))
    assert "# Success Criteria" not in prompt


def test_pass_threshold_controls_test_pass():
    rating = AttackSuccessRating(score=0.4, reasoning="r", severity="partial")

    strict = AttackSuccessEvaluator(pass_threshold=0.3)
    lenient = AttackSuccessEvaluator(pass_threshold=0.5)

    assert strict._build_output(rating)[0].test_pass is False
    assert lenient._build_output(rating)[0].test_pass is True


def test_judge_prompt_includes_conversation_and_trace():
    judge = AttackSuccessEvaluator()
    prompt = judge._build_judge_prompt(
        _data(
            metadata={"actor_goal": "g"},
            output=[{"role": "attacker", "content": "hi"}, {"role": "target", "content": "ok"}],
            trajectory=[{"name": "tool_x", "input": {"a": 1}}],
        )
    )
    assert "Attacker: hi" in prompt
    assert "Target: ok" in prompt
    assert "tool_x" in prompt
