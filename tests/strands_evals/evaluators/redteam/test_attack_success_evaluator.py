"""Tests for AttackSuccessEvaluator prompt assembly."""

from unittest.mock import MagicMock, patch

import pytest

from strands_evals.evaluators.redteam import AttackSuccessEvaluator
from strands_evals.evaluators.redteam.attack_success_evaluator import AttackSuccessRating
from strands_evals.evaluators.redteam.prompt_templates.attack_success import (
    DEFAULT_VERSION,
    attack_success_v0,
)
from strands_evals.types.evaluation import NOT_APPLICABLE, EnvironmentState, EvaluationData
from strands_evals.types.redteam import ATTACKER_REFUSED, DEFAULT_SEVERITY, RISK_CATEGORIES, RUN_RESULTS

_EVALUATOR = "strands_evals.evaluators.redteam.attack_success_evaluator"


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
    for category in RISK_CATEGORIES:
        assert f"`{category}`" in rubric
    assert set(DEFAULT_SEVERITY) == set(RISK_CATEGORIES)


def test_disclosure_has_echo_exception():
    """The sensitive_data definition must guard against scoring an echo of the
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


def _run_results_state(state) -> list[EnvironmentState]:
    return [EnvironmentState(name=RUN_RESULTS, state=state)]


def _judge_returning(score: float, severity: str) -> MagicMock:
    judge = MagicMock()
    judge.return_value.structured_output = AttackSuccessRating(score=score, reasoning="r", severity=severity)
    return judge


_SENT_THEN_REFUSED = [
    {"role": "attacker", "content": "hi"},
    {"role": "target", "content": "ok"},
    {"role": ATTACKER_REFUSED, "content": "I won't write that"},
]


def test_attacker_refusal_without_breach_is_not_applicable():
    """A clean result from an attack its attacker cut short is inconclusive: NOT_APPLICABLE, never defended."""
    data = _data(metadata={"actor_goal": "g"}, output=_SENT_THEN_REFUSED)
    data.actual_environment_state = _run_results_state({ATTACKER_REFUSED: True})
    judge = _judge_returning(0.1, "refused")
    with patch(f"{_EVALUATOR}.Agent", return_value=judge):
        outputs = AttackSuccessEvaluator().evaluate(data)
    judge.assert_called_once()
    assert len(outputs) == 1
    assert outputs[0].label == NOT_APPLICABLE
    assert outputs[0].test_pass is False
    assert "declined its role" in outputs[0].reason


def test_attacker_refusal_on_the_first_message_skips_the_judge():
    """With nothing sent to the target there is nothing to judge, so even a scoring judge can't make a breach."""
    data = _data(metadata={"actor_goal": "g"}, output=[{"role": ATTACKER_REFUSED, "content": "I won't"}])
    data.actual_environment_state = _run_results_state({ATTACKER_REFUSED: True})
    judge = _judge_returning(0.6, "substantial")
    with patch(f"{_EVALUATOR}.Agent", return_value=judge):
        outputs = AttackSuccessEvaluator().evaluate(data)
    judge.assert_not_called()
    assert outputs[0].label == NOT_APPLICABLE
    assert outputs[0].test_pass is False


def test_attacker_refusal_keeps_a_breach_in_the_turns_sent():
    """A breach before the attacker refused is real, so the judge's verdict stands."""
    data = _data(metadata={"actor_goal": "g"}, output=_SENT_THEN_REFUSED)
    data.actual_environment_state = _run_results_state({ATTACKER_REFUSED: True})
    with patch(f"{_EVALUATOR}.Agent", return_value=_judge_returning(0.6, "substantial")):
        outputs = AttackSuccessEvaluator().evaluate(data)
    assert outputs[0].label == "substantial"
    assert outputs[0].score == 0.6
    assert outputs[0].test_pass is False


def test_judge_prompt_omits_the_refused_attacker_turn():
    """The refused message was never sent, so the judge sees only the turns the target answered."""
    prompt = AttackSuccessEvaluator()._build_judge_prompt(
        _data(metadata={"actor_goal": "g"}, output=_SENT_THEN_REFUSED)
    )
    assert "Attacker: hi" in prompt
    assert "I won't write that" not in prompt


@pytest.mark.parametrize(
    "environment_state",
    [
        None,
        [EnvironmentState(name="other", state={ATTACKER_REFUSED: True})],
        _run_results_state({ATTACKER_REFUSED: False}),
        _run_results_state(["not", "a", "dict"]),
    ],
)
def test_judge_runs_unless_run_results_report_a_refusal(environment_state):
    data = _data(metadata={"actor_goal": "g"})
    data.actual_environment_state = environment_state
    judge = MagicMock()
    judge.return_value.structured_output = AttackSuccessRating(score=0.1, reasoning="r", severity="refused")
    with patch(f"{_EVALUATOR}.Agent", return_value=judge):
        outputs = AttackSuccessEvaluator().evaluate(data)
    judge.assert_called_once()
    assert outputs[0].label == "refused"


def test_attacker_refusal_read_from_plain_dict_environment_state():
    """A custom task may return `environment_state` as plain dicts; the refusal flag is still read."""
    data = _data(metadata={"actor_goal": "g"}, output=_SENT_THEN_REFUSED)
    data.actual_environment_state = [{"name": RUN_RESULTS, "state": {ATTACKER_REFUSED: True}}]  # type: ignore[list-item]
    with patch(f"{_EVALUATOR}.Agent", return_value=_judge_returning(0.1, "refused")):
        outputs = AttackSuccessEvaluator().evaluate(data)
    assert outputs[0].label == NOT_APPLICABLE


@pytest.mark.parametrize("output", ["a plain-string transcript", ["a", "list", "of", "strings"]])
def test_attacker_refusal_with_non_dict_output_still_judges_the_sent_turns(output):
    """Custom output shapes neither crash the check nor skip the judge: a breach in them still stands."""
    data = _data(metadata={"actor_goal": "g"}, output=output)
    data.actual_environment_state = _run_results_state({ATTACKER_REFUSED: True})
    judge = _judge_returning(0.6, "substantial")
    with patch(f"{_EVALUATOR}.Agent", return_value=judge):
        outputs = AttackSuccessEvaluator().evaluate(data)
    judge.assert_called_once()
    assert outputs[0].label == "substantial"


def test_judge_prompt_skips_non_dict_turns():
    prompt = AttackSuccessEvaluator()._build_judge_prompt(
        _data(metadata={"actor_goal": "g"}, output=["stray", {"role": "attacker", "content": "hi"}])
    )
    assert "Attacker: hi" in prompt
    assert "stray" not in prompt
