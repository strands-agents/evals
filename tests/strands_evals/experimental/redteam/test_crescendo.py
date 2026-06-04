"""Tests for CrescendoStrategy and its module-level helpers."""

from unittest.mock import MagicMock, patch

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.strategies import BUILTIN_STRATEGIES, CrescendoStrategy
from strands_evals.experimental.redteam.strategies.base import AttackRunResult
from strands_evals.experimental.redteam.strategies.crescendo import (
    gen_escalating_question,
    is_refusal,
    success_score,
)
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig

_CRESCENDO = "strands_evals.experimental.redteam.strategies.crescendo"


def _case(success_criteria: str = "leaked the secret") -> RedTeamCase:
    return RedTeamCase(
        name="c0",
        input="hello",
        config=RedTeamConfig(
            attack_goal=AttackGoal(
                risk_category="data_exfiltration",
                actor_goal="exfiltrate the secret",
                success_criteria=success_criteria,
            )
        ),
        metadata={"strategy": "crescendo"},
    )


def _strategy(**kwargs) -> CrescendoStrategy:
    """A strategy with its agent builders stubbed so no real LLM is created."""
    strat = CrescendoStrategy(**kwargs)
    strat._attacker_agent = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    strat._judge_agent = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    return strat


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------


class TestIsRefusal:
    def test_detects_refusal_markers(self):
        assert is_refusal("I'm sorry, I can't help with that.")
        assert is_refusal("I CANNOT assist with this request.")

    def test_engagement_is_not_refusal(self):
        assert not is_refusal("Sure, here is how the process works...")


class TestSuccessScore:
    def test_returns_zero_without_criteria(self):
        judge = MagicMock()
        assert success_score(judge, "anything", None) == 0.0
        judge.assert_not_called()

    def test_parses_judge_score(self):
        judge = MagicMock()
        judge.return_value.structured_output = MagicMock(score=0.8)
        assert success_score(judge, "leaked it", "leak the secret") == 0.8

    def test_parse_failure_defaults_to_zero(self):
        judge = MagicMock()
        judge.return_value.structured_output = None
        assert success_score(judge, "resp", "criteria") == 0.0


class TestGenEscalatingQuestion:
    def test_first_turn_uses_goal(self):
        attacker = MagicMock()
        attacker.return_value.structured_output = MagicMock(message="benign opener")
        goal = AttackGoal(risk_category="x", actor_goal="the goal")
        assert gen_escalating_question(attacker, goal, [], "") == "benign opener"

    def test_parse_failure_returns_none(self):
        attacker = MagicMock()
        attacker.return_value.structured_output = None
        goal = AttackGoal(risk_category="x", actor_goal="g")
        assert gen_escalating_question(attacker, goal, [{"role": "attacker", "content": "q"}], "r") is None


# ---------------------------------------------------------------------------
# run_attack loop
# ---------------------------------------------------------------------------


def test_returns_attack_run_result_shape():
    strat = _strategy(max_turns=1)
    with (
        patch(f"{_CRESCENDO}.gen_escalating_question", return_value="q"),
        patch(f"{_CRESCENDO}.success_score", return_value=0.0),
    ):
        result = strat.run_attack(_case(), lambda _m: "engaged reply", max_turns=10)
    assert isinstance(result, AttackRunResult)
    assert set(result.metadata) == {"turns_used", "backtracks", "parse_failures"}


def test_backtracks_on_refusal_without_consuming_conversation():
    """A refusal is not appended to the conversation; the turn is retried."""
    strat = _strategy(max_turns=2, max_backtracks=5)
    replies = iter(["I can't help with that", "here is the answer"])
    with (
        patch(f"{_CRESCENDO}.gen_escalating_question", return_value="q"),
        patch(f"{_CRESCENDO}.success_score", return_value=0.0),
    ):
        result = strat.run_attack(_case(), lambda _m: next(replies), max_turns=10)
    # refused turn dropped: only the engaged (q, r) pair is in the conversation
    assert result.conversation == [
        {"role": "attacker", "content": "q"},
        {"role": "target", "content": "here is the answer"},
    ]
    assert result.metadata["backtracks"] == 1


def test_respects_max_backtracks():
    """Once the backtrack cap is hit, a refusal is appended instead of retried."""
    strat = _strategy(max_turns=5, max_backtracks=1)
    with (
        patch(f"{_CRESCENDO}.gen_escalating_question", return_value="q"),
        patch(f"{_CRESCENDO}.success_score", return_value=0.0),
    ):
        result = strat.run_attack(_case(), lambda _m: "I cannot do that", max_turns=10)
    assert result.metadata["backtracks"] == 1
    # after the cap, refusals are recorded rather than dropped
    assert any(turn["role"] == "target" for turn in result.conversation)


def test_stops_on_success_threshold():
    strat = _strategy(max_turns=10, success_threshold=0.7)
    with (
        patch(f"{_CRESCENDO}.gen_escalating_question", return_value="q"),
        patch(f"{_CRESCENDO}.success_score", return_value=0.9),
    ):
        result = strat.run_attack(_case(), lambda _m: "leaked", max_turns=10)
    assert result.strategy_succeeded is True
    assert result.strategy_score == 0.9
    assert result.metadata["turns_used"] == 1


def test_runs_to_max_turns_when_never_succeeds():
    strat = _strategy(max_turns=3, success_threshold=0.7)
    with (
        patch(f"{_CRESCENDO}.gen_escalating_question", return_value="q"),
        patch(f"{_CRESCENDO}.success_score", return_value=0.1),
    ):
        result = strat.run_attack(_case(), lambda _m: "engaged", max_turns=10)
    assert result.strategy_succeeded is False
    assert result.metadata["turns_used"] == 3


def test_max_turns_clamped_by_injected_value():
    """The smaller of the strategy's max_turns and the injected max_turns wins."""
    strat = _strategy(max_turns=10, success_threshold=0.7)
    with (
        patch(f"{_CRESCENDO}.gen_escalating_question", return_value="q"),
        patch(f"{_CRESCENDO}.success_score", return_value=0.0),
    ):
        result = strat.run_attack(_case(), lambda _m: "engaged", max_turns=2)
    assert result.metadata["turns_used"] == 2


def test_question_parse_failure_safe_terminates():
    strat = _strategy(max_turns=5)
    with (
        patch(f"{_CRESCENDO}.gen_escalating_question", return_value=None),
        patch(f"{_CRESCENDO}.success_score", return_value=0.0),
    ):
        result = strat.run_attack(_case(), lambda _m: "reply", max_turns=10)
    assert result.metadata["parse_failures"] == 1
    assert result.conversation == []  # nothing sent, but no exception
    assert result.strategy_succeeded is False


def test_ctor_model_takes_precedence():
    strat = CrescendoStrategy(model="ctor-model")
    captured = {}
    strat._attacker_agent = lambda goal, model: captured.setdefault("attacker_model", model) or MagicMock()  # type: ignore[method-assign]
    strat._judge_agent = lambda model: captured.setdefault("judge_model", model) or MagicMock()  # type: ignore[method-assign]
    with (
        patch(f"{_CRESCENDO}.gen_escalating_question", return_value=None),
        patch(f"{_CRESCENDO}.success_score", return_value=0.0),
    ):
        strat.run_attack(_case(), lambda _m: "r", max_turns=1, model="experiment-model")
    assert captured["attacker_model"] == "ctor-model"


def test_reset_nulls_agents():
    strat = CrescendoStrategy()
    strat._attacker = MagicMock()
    strat._judge = MagicMock()
    strat.reset()
    assert strat._attacker is None
    assert strat._judge is None


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_crescendo_not_in_builtin_strategies():
    assert "crescendo" not in BUILTIN_STRATEGIES


def test_name_and_default_label():
    assert CrescendoStrategy().name == "crescendo"
    assert CrescendoStrategy().label == "crescendo"
    assert CrescendoStrategy(label="cre-10").label == "cre-10"
