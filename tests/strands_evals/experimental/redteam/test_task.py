"""Tests for _build_attacker_task."""

from unittest.mock import MagicMock

import pytest

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.strategies.base import AttackRunResult, AttackStrategy
from strands_evals.experimental.redteam.task import (
    MAX_ALLOWED_TURNS,
    _build_attacker_task,
    _wrap_agent_with_trace,
)
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig


class _StubStrategy(AttackStrategy):
    """Records the injected call_target and returns a canned result."""

    def __init__(self, label="stub", *, result=None, calls=None):
        super().__init__(label=label)
        self._result = result
        self.reset_count = 0
        self.calls = calls if calls is not None else []
        self.received_max_turns: int | None = None

    @property
    def name(self) -> str:
        return "stub"

    def run_attack(self, case, call_target, *, max_turns, model=None, **kwargs) -> AttackRunResult:
        self.received_max_turns = max_turns
        # exercise the injected call_target so trace/output wiring is covered
        reply = call_target("ping")
        self.calls.append(reply)
        return self._result or AttackRunResult(
            conversation=[{"role": "attacker", "content": "ping"}, {"role": "target", "content": reply}],
            metadata={"turns_used": 1},
        )

    def reset(self) -> None:
        self.reset_count += 1


def _case(name: str = "c0", label: str = "stub") -> RedTeamCase:
    return RedTeamCase(
        name=name,
        input="hello",
        config=RedTeamConfig(attack_goal=AttackGoal(risk_category="guideline_bypass", actor_goal="goal")),
        metadata={"strategy": label},
    )


def _by_label(*strategies: AttackStrategy) -> dict[str, AttackStrategy]:
    return {s.label: s for s in strategies}


def test_run_attack_receives_max_allowed_turns_ceiling():
    """task_fn passes MAX_ALLOWED_TURNS as the hard ceiling; the strategy owns its budget."""
    from strands_evals.experimental.redteam.task import MAX_ALLOWED_TURNS

    strat = _StubStrategy()
    _build_attacker_task(lambda _msg: "ok", _by_label(strat))(_case())
    assert strat.received_max_turns == MAX_ALLOWED_TURNS


def test_task_fn_calls_run_attack_and_maps_result():
    strat = _StubStrategy()
    task = _build_attacker_task(lambda _msg: "target reply", _by_label(strat))

    result = task(_case())

    assert "output" in result
    assert result["output"][1]["content"] == "target reply"
    assert result["metadata"]["turns_used"] == 1
    assert strat.reset_count == 1


def test_task_fn_resets_strategy_each_case():
    strat = _StubStrategy()
    task = _build_attacker_task(lambda _msg: "ok", _by_label(strat))
    task(_case("c0"))
    task(_case("c1"))
    assert strat.reset_count == 2


def test_task_fn_missing_strategy_label_raises():
    task = _build_attacker_task(lambda _msg: "ok", _by_label(_StubStrategy()))
    case = _case()
    case.metadata = {}  # no strategy label
    with pytest.raises(ValueError, match="strategy"):
        task(case)


def test_task_fn_unknown_strategy_label_raises():
    task = _build_attacker_task(lambda _msg: "ok", _by_label(_StubStrategy(label="stub")))
    with pytest.raises(ValueError, match="missing"):
        task(_case(label="missing"))


def test_task_fn_dict_target_extends_trace_and_extracts_output():
    """A callable target returning {"output", "trace"} feeds the trajectory and the conversation."""

    def dict_target(_msg):
        return {"output": "dict reply", "trace": [{"name": "lookup", "input": {"id": "1"}}]}

    task = _build_attacker_task(dict_target, _by_label(_StubStrategy()))
    result = task(_case())

    assert result["output"][1]["content"] == "dict reply"
    assert result["trajectory"] == [{"name": "lookup", "input": {"id": "1"}}]


def test_task_fn_dict_target_without_trace_key():
    """A dict target lacking a "trace" key still extracts output and emits an empty trajectory."""
    task = _build_attacker_task(lambda _msg: {"output": "no trace here"}, _by_label(_StubStrategy()))
    result = task(_case())

    assert result["output"][1]["content"] == "no trace here"
    assert result["trajectory"] == []


def test_task_fn_clears_agent_messages_per_case():
    """Agent target's message history is cleared at the start of each case."""
    from strands import Agent

    agent = MagicMock(spec=Agent)
    agent.messages = MagicMock()
    agent.messages.__len__.return_value = 0
    agent.return_value = "ok"
    task = _build_attacker_task(agent, _by_label(_StubStrategy()))

    task(_case("c0"))
    task(_case("c1"))

    assert agent.messages.clear.call_count == 2


def test_max_allowed_turns_constant_present():
    assert MAX_ALLOWED_TURNS >= 50


# ---------------------------------------------------------------------------
# _wrap_agent_with_trace (inlined helper)
# ---------------------------------------------------------------------------


def _make_agent_for_wrap(messages=None):
    agent = MagicMock()
    agent.messages = messages if messages is not None else []
    return agent


class TestWrapAgentWithTrace:
    def test_returns_string_response(self):
        agent = _make_agent_for_wrap()
        agent.return_value = "hello back"
        wrapped = _wrap_agent_with_trace(agent, [])
        assert wrapped("hi") == "hello back"

    def test_appends_tool_uses_to_trace(self):
        agent = _make_agent_for_wrap(messages=[])
        new_msg = {"content": [{"toolUse": {"name": "search", "input": {"q": "x"}}}]}

        def _call_side_effect(_msg):
            agent.messages.append(new_msg)
            return "ok"

        agent.side_effect = _call_side_effect
        trace: list[dict] = []
        _wrap_agent_with_trace(agent, trace)("hi")
        assert trace == [{"name": "search", "input": {"q": "x"}}]

    def test_only_new_messages_are_scanned(self):
        old_msg = {"content": [{"toolUse": {"name": "old", "input": {}}}]}
        agent = _make_agent_for_wrap(messages=[old_msg])
        agent.return_value = "ok"
        trace: list[dict] = []
        _wrap_agent_with_trace(agent, trace)("hi")
        assert trace == []

    def test_swallows_message_format_errors(self):
        agent = _make_agent_for_wrap(messages=[])
        agent.side_effect = lambda _msg: agent.messages.append("not a dict") or "ok"
        trace: list[dict] = []
        _wrap_agent_with_trace(agent, trace)("hi")
        assert trace == []
