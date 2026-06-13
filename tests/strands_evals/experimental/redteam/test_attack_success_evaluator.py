"""Tests for AttackSuccessEvaluator prompt assembly."""

from strands_evals.experimental.redteam.evaluators import AttackSuccessEvaluator
from strands_evals.experimental.redteam.evaluators.attack_success_evaluator import AttackSuccessRating
from strands_evals.experimental.redteam.evaluators.prompt_templates.attack_success import (
    DEFAULT_VERSION,
    attack_success_v0,
    attack_success_v1,
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
    prompt = judge._build_judge_prompt(_data(metadata={"actor_goal": "g", "risk_category": "system_prompt_leak"}))
    assert "# Risk Category" in prompt
    assert "system_prompt_leak" in prompt


def test_judge_prompt_omits_risk_category_when_missing():
    judge = AttackSuccessEvaluator()
    prompt = judge._build_judge_prompt(_data(metadata={"actor_goal": "g"}))
    assert "# Risk Category" not in prompt


def test_v1_anchors_are_verbatim_from_v0():
    """v1 must only ADD the per-category success block; the four score anchors and the
    system prompt stay byte-identical to v0 so v0-vs-v1 is a clean A/B."""
    v0_anchors = [ln for ln in attack_success_v0.RUBRIC.splitlines() if ln.startswith(("- 0.", "- 1."))]
    v1_anchors = [ln for ln in attack_success_v1.RUBRIC.splitlines() if ln.startswith(("- 0.", "- 1."))]
    assert v1_anchors == v0_anchors
    assert attack_success_v1.SYSTEM_PROMPT == attack_success_v0.SYSTEM_PROMPT


def test_v1_defines_success_for_all_five_risk_categories():
    rubric = attack_success_v1.RUBRIC
    assert "by Risk Category" in rubric
    for category in (
        "system_prompt_leak",
        "data_exfiltration",
        "harmful_content",
        "excessive_agency",
        "guideline_bypass",
    ):
        assert category in rubric


def test_v1_disclosure_has_echo_exception():
    """The system_prompt_leak definition must guard against scoring an echo of the
    attacker's own token as a leak (the canary's refusal-gate, in prose)."""
    rubric = attack_success_v1.RUBRIC.lower()
    assert "echoing" in rubric or "echo" in rubric
    assert "attacker already" in rubric or "attacker did not already" in rubric


def test_evaluator_accepts_v1_version():
    judge = AttackSuccessEvaluator(version="v1")
    assert "by Risk Category" in judge.system_prompt


def test_default_version_is_v1():
    """Pin the user-facing default: a silent revert of DEFAULT_VERSION to "v0" (which the
    rest of the suite would still pass) must fail here. The default evaluator scores under
    v1, so its system prompt carries the per-category success block."""
    assert DEFAULT_VERSION == "v1"
    judge = AttackSuccessEvaluator()
    assert judge.version == "v1"
    assert "by Risk Category" in judge.system_prompt
