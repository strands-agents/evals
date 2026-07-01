"""Tests for AttackSuccessEvaluator prompt assembly."""

from strands_evals.experimental.redteam.evaluators import AttackSuccessEvaluator
from strands_evals.experimental.redteam.evaluators.attack_success_evaluator import AttackSuccessRating
from strands_evals.experimental.redteam.evaluators.prompt_templates.attack_success import (
    DEFAULT_VERSION,
    attack_success_v0,
)
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


def test_judge_prompt_includes_risk_category_when_present():
    judge = AttackSuccessEvaluator()
    prompt = judge._build_judge_prompt(_data(metadata={"actor_goal": "g", "risk_category": "sensitive_data"}))
    assert "# Risk Category" in prompt
    assert "sensitive_data" in prompt


def test_judge_prompt_omits_risk_category_when_missing():
    judge = AttackSuccessEvaluator()
    prompt = judge._build_judge_prompt(_data(metadata={"actor_goal": "g"}))
    assert "# Risk Category" not in prompt


def test_rubric_has_four_anchors():
    """The rubric scores on the four severity anchors (refused/partial/substantial/full)."""
    anchors = [ln for ln in attack_success_v0.RUBRIC.splitlines() if ln.startswith(("- 0.", "- 1."))]
    assert len(anchors) == 4


def test_defines_success_for_all_risk_categories():
    rubric = attack_success_v0.RUBRIC
    assert "by Risk Category" in rubric
    for category in (
        "prompt_injection",
        "excessive_agency",
        "sensitive_data",
        "data_exfiltration",
        "social_engineering",
        "privilege_escalation",
        "business_rule_violation",
    ):
        assert category in rubric


def test_disclosure_has_echo_exception():
    """The prompt_injection definition must guard against scoring an echo of the
    attacker's own token as a leak (the canary's refusal-gate, in prose)."""
    rubric = attack_success_v0.RUBRIC.lower()
    assert "echoing" in rubric or "echo" in rubric
    assert "string the attacker supplied" in rubric


def test_default_version_carries_per_category_block():
    """Pin the user-facing default: the default evaluator scores under v0, whose system
    prompt carries the per-category success block."""
    assert DEFAULT_VERSION == "v0"
    judge = AttackSuccessEvaluator()
    assert judge.version == "v0"
    assert "by Risk Category" in judge.system_prompt
