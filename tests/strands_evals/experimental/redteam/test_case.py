"""Tests for RedTeamCase."""

import threading

import pytest

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.strategies.base import AttackRunResult, AttackStrategy
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig


class _StubStrategy(AttackStrategy):
    def __init__(self, *, label=None, model=None):
        super().__init__(label=label)
        self._model = model

    @property
    def name(self) -> str:
        return "stub"

    def run_attack(self, case, target_session, *, max_turns, model=None, **kwargs) -> AttackRunResult:
        return AttackRunResult(conversation=[])


def _case() -> RedTeamCase:
    return RedTeamCase(
        name="c0",
        input="hello",
        config=RedTeamConfig(attack_goal=AttackGoal(risk_category="guideline_bypass", actor_goal="goal")),
    )


def test_strategy_round_trips_through_the_setter():
    strategy = _StubStrategy()
    case = _case()

    case.strategy = strategy

    assert case.strategy is strategy


def test_missing_strategy_raises_clear_error():
    with pytest.raises(AttributeError, match="'c0' has no strategy"):
        _ = _case().strategy


def test_missing_strategy_is_probeable():
    """A custom task can probe for the strategy on a mixed case list."""
    case = _case()
    assert not hasattr(case, "strategy")
    assert getattr(case, "strategy", None) is None


def test_strategy_is_not_serialized():
    case = _case()
    case.strategy = _StubStrategy()

    dumped = case.model_dump()

    assert "strategy" not in dumped
    assert "_strategy" not in dumped
    with pytest.raises(AttributeError, match="has no strategy"):
        _ = RedTeamCase.model_validate(dumped).strategy


def test_deep_copy_shares_the_strategy():
    """A deep copy keeps the same instance, even when the strategy holds an object deepcopy can't handle."""
    strategy = _StubStrategy(model=threading.Lock())
    case = _case()
    case.strategy = strategy

    copied = case.model_copy(deep=True)

    assert copied.strategy is strategy
