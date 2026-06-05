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


def _case(success_criteria: str | None = "leaked the secret") -> RedTeamCase:
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
    strat._refusal_judge_agent = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    return strat


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------


class TestIsRefusal:
    def test_no_marker_is_not_refusal_without_llm(self):
        # fast negative path: no marker -> False, no judge needed
        assert not is_refusal("Sure, here is how the process works...")

    def test_marker_without_judge_falls_back_to_heuristic(self):
        assert is_refusal("I can't help with that.")
        assert is_refusal("I CANNOT assist with this request.")

    def test_marker_hit_is_disambiguated_by_judge(self):
        """Marker-bearing but compliant text is NOT a refusal when a judge is given."""
        judge = MagicMock()
        judge.return_value.structured_output = MagicMock(is_refusal=False)
        # contains "i cannot" but is actually complying
        assert not is_refusal("I cannot stress enough how important this is. Here are the steps:", judge)
        judge.return_value.structured_output = MagicMock(is_refusal=True)
        assert is_refusal("I cannot help with that.", judge)

    def test_judge_parse_failure_keeps_turn(self):
        judge = MagicMock()
        judge.return_value.structured_output = None
        assert is_refusal("I apologize, but here is the answer:", judge) is False

    def test_engagement_with_refusal_substring_not_called_to_judge_when_no_marker(self):
        judge = MagicMock()
        assert not is_refusal("Absolutely, here you go.", judge)
        judge.assert_not_called()  # fast negative path skips the LLM


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
    assert set(result.metadata) == {"turns_used", "target_calls", "backtracks", "parse_failures"}


def test_backtracks_on_refusal_without_consuming_conversation():
    """A refusal is not appended to the conversation; the turn is retried."""
    strat = _strategy(max_turns=2, max_backtracks=5)
    replies = iter(["refused", "here is the answer"])
    refusals = iter([True, False])
    with (
        patch(f"{_CRESCENDO}.gen_escalating_question", return_value="q"),
        patch(f"{_CRESCENDO}.is_refusal", side_effect=lambda *_a, **_k: next(refusals)),
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
        patch(f"{_CRESCENDO}.is_refusal", return_value=True),
        patch(f"{_CRESCENDO}.success_score", return_value=0.0),
    ):
        result = strat.run_attack(_case(), lambda _m: "refused", max_turns=10)
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


def test_ctor_max_turns_wins_when_smaller():
    """The smaller of ctor max_turns and injected max_turns wins (other direction)."""
    strat = _strategy(max_turns=2, success_threshold=0.7)
    with (
        patch(f"{_CRESCENDO}.gen_escalating_question", return_value="q"),
        patch(f"{_CRESCENDO}.success_score", return_value=0.0),
    ):
        result = strat.run_attack(_case(), lambda _m: "engaged", max_turns=10)
    assert result.metadata["turns_used"] == 2


def test_all_refusals_yields_empty_conversation():
    """Every turn refused (under the cap) -> empty conversation, turns_used=0, calls counted."""
    strat = _strategy(max_turns=3, max_backtracks=10)
    with (
        patch(f"{_CRESCENDO}.gen_escalating_question", return_value="q"),
        patch(f"{_CRESCENDO}.is_refusal", return_value=True),
        patch(f"{_CRESCENDO}.success_score", return_value=0.0),
    ):
        result = strat.run_attack(_case(), lambda _m: "refused", max_turns=10)
    assert result.conversation == []
    assert result.metadata["turns_used"] == 0
    assert result.metadata["target_calls"] == 3
    assert result.metadata["backtracks"] == 3


def test_none_success_criteria_runs_to_max_turns_without_judge():
    """No success_criteria -> success gate never fires, runs to max_turns."""
    strat = _strategy(max_turns=3, success_threshold=0.7)
    case = _case(success_criteria=None)
    with patch(f"{_CRESCENDO}.gen_escalating_question", return_value="q"):
        result = strat.run_attack(case, lambda _m: "engaged", max_turns=10)
    assert result.strategy_succeeded is False
    assert result.metadata["turns_used"] == 3


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
