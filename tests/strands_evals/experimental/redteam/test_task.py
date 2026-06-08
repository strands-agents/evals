"""Tests for _build_attacker_task."""

from unittest.mock import MagicMock

import pytest

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.strategies.base import AttackRunResult, AttackStrategy
from strands_evals.experimental.redteam.task import (
    MAX_ALLOWED_TURNS,
    _build_attacker_task,
)
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig


class _StubStrategy(AttackStrategy):
    """Records the injected target session and returns a canned result."""

    def __init__(self, label="stub", *, result=None, calls=None):
        super().__init__(label=label)
        self._result = result
        self.reset_count = 0
        self.calls = calls if calls is not None else []
        self.received_max_turns: int | None = None

    @property
    def name(self) -> str:
        return "stub"

    def run_attack(self, case, target_session, *, max_turns, model=None, **kwargs) -> AttackRunResult:
        self.received_max_turns = max_turns
        # exercise the injected session so trace/output wiring is covered
        reply = target_session.invoke("ping")
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

    # Only the keys the base Experiment reads are returned; run stats flow via run_meta.
    assert set(result) == {"output", "trajectory"}
    assert result["output"][1]["content"] == "target reply"
    assert strat.reset_count == 1


def test_task_fn_records_run_stats_into_run_meta():
    """Strategy metadata reaches the experiment via run_meta, not the returned dict."""
    strat = _StubStrategy()
    run_meta: dict[str, dict] = {}
    task = _build_attacker_task(lambda _msg: "target reply", _by_label(strat), run_meta=run_meta)

    task(_case("c0"))

    assert run_meta["c0"]["turns_used"] == 1


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
