"""Tests for _build_attacker_task."""

from unittest.mock import MagicMock, patch

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.task import (
    MAX_ALLOWED_TURNS,
    _build_attacker_task,
    _wrap_agent_with_trace,
)
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig


def _case(name: str = "c0") -> RedTeamCase:
    return RedTeamCase(
        name=name,
        input="hello",
        config=RedTeamConfig(
            attack_goal=AttackGoal(risk_category="guideline_bypass", actor_goal="goal"),
            system_prompt_template="prompt {actor_profile} {max_turns}",
            strategy="gradual_escalation",
        ),
    )


def test_max_turns_clamped_above_ceiling(caplog):
    with caplog.at_level("WARNING"):
        _build_attacker_task(target=lambda msg: "ok", max_turns=999)
    assert any("clamping" in rec.message for rec in caplog.records)


@patch("strands_evals.experimental.redteam.task.ActorSimulator")
def test_task_fn_drives_conversation_loop(mock_simulator_cls):
    mock_sim = MagicMock()
    # 2 turns then exit
    mock_sim.has_next.side_effect = [True, True, False]
    act_result = MagicMock()
    act_result.structured_output.message = "follow-up attack"
    mock_sim.act.return_value = act_result
    mock_simulator_cls.return_value = mock_sim

    target = MagicMock(return_value="target reply")
    task = _build_attacker_task(target=target)

    result = task(_case())

    assert "output" in result
    conversation = result["output"]
    assert len(conversation) == 4
    assert conversation[0]["role"] == "attacker"
    assert conversation[1]["role"] == "target"
    assert "trajectory" not in result


@patch("strands_evals.experimental.redteam.task.ActorSimulator")
def test_task_fn_substitutes_max_turns_placeholder(mock_simulator_cls):
    mock_sim = MagicMock()
    mock_sim.has_next.return_value = False
    mock_simulator_cls.return_value = mock_sim

    task = _build_attacker_task(target=lambda _msg: "ok", max_turns=7)
    task(_case())

    kwargs = mock_simulator_cls.call_args.kwargs
    template = kwargs["system_prompt_template"]
    assert "{max_turns}" not in template
    assert "7" in template
    assert "{actor_profile}" in template


@patch("strands_evals.experimental.redteam.task.ActorSimulator")
def test_task_fn_handles_target_exception(mock_simulator_cls):
    mock_sim = MagicMock()
    mock_sim.has_next.side_effect = [True, False]
    mock_simulator_cls.return_value = mock_sim

    def broken_target(msg):
        raise RuntimeError("boom")

    task = _build_attacker_task(target=broken_target)
    result = task(_case())

    assert "[Error: boom]" in result["output"][1]["content"]


@patch("strands_evals.experimental.redteam.task.ActorSimulator")
def test_task_fn_dict_target_extends_trace_and_extracts_output(mock_simulator_cls):
    """A callable target returning {"output", "trace"} feeds the trajectory and the conversation."""
    mock_sim = MagicMock()
    mock_sim.has_next.side_effect = [True, False]
    mock_simulator_cls.return_value = mock_sim

    def dict_target(_msg):
        return {"output": "dict reply", "trace": [{"name": "lookup", "input": {"id": "1"}}]}

    task = _build_attacker_task(target=dict_target)
    result = task(_case())

    # output extracted from the dict and placed in the conversation
    assert result["output"][1]["content"] == "dict reply"
    # trace from the dict surfaced as the trajectory
    assert result["trajectory"] == [{"name": "lookup", "input": {"id": "1"}}]


@patch("strands_evals.experimental.redteam.task.ActorSimulator")
def test_task_fn_dict_target_without_trace_key(mock_simulator_cls):
    """A dict target lacking a "trace" key still extracts output and emits no trajectory."""
    mock_sim = MagicMock()
    mock_sim.has_next.side_effect = [True, False]
    mock_simulator_cls.return_value = mock_sim

    task = _build_attacker_task(target=lambda _msg: {"output": "no trace here"})
    result = task(_case())

    assert result["output"][1]["content"] == "no trace here"
    assert "trajectory" not in result


@patch("strands_evals.experimental.redteam.task.ActorSimulator")
def test_task_fn_clears_agent_messages_per_case(mock_simulator_cls):
    """Agent target's message history is cleared at the start of each case."""
    from strands import Agent

    mock_sim = MagicMock()
    mock_sim.has_next.return_value = False
    mock_simulator_cls.return_value = mock_sim

    agent = MagicMock(spec=Agent)
    agent.messages = MagicMock()
    agent.return_value = "ok"
    task = _build_attacker_task(target=agent)

    task(_case("c0"))
    task(_case("c1"))

    assert agent.messages.clear.call_count == 2


@patch("strands_evals.experimental.redteam.task.ActorSimulator")
def test_task_fn_invokes_strategy_hooks(mock_simulator_cls):
    """task_fn calls strategy.reset() per case and strategy.enhance() per turn."""
    from strands_evals.experimental.redteam.task import BUILTIN_STRATEGIES

    mock_sim = MagicMock()
    mock_sim.has_next.side_effect = [True, True, False]
    act_result = MagicMock()
    act_result.structured_output.message = "follow-up"
    mock_sim.act.return_value = act_result
    mock_simulator_cls.return_value = mock_sim

    strategy = BUILTIN_STRATEGIES["gradual_escalation"]
    with (
        patch.object(strategy, "reset") as mock_reset,
        patch.object(strategy, "enhance", side_effect=lambda p, **_: p) as mock_enhance,
    ):
        task = _build_attacker_task(target=lambda _msg: "ok")
        task(_case())

    mock_reset.assert_called_once()
    mock_enhance.assert_called()
    kwargs = mock_enhance.call_args.kwargs
    assert "target_response" in kwargs
    assert "conversation" in kwargs
    assert "attack_goal" in kwargs


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
        wrapped = _wrap_agent_with_trace(agent)
        assert wrapped("hi") == "hello back"

    def test_appends_tool_uses_to_trace(self):
        agent = _make_agent_for_wrap(messages=[])
        new_msg = {"content": [{"toolUse": {"name": "search", "input": {"q": "x"}}}]}

        def _call_side_effect(_msg):
            agent.messages.append(new_msg)
            return "ok"

        agent.side_effect = _call_side_effect
        wrapped = _wrap_agent_with_trace(agent)
        trace: list[dict] = []
        wrapped("hi", trace)
        assert trace == [{"name": "search", "input": {"q": "x"}}]

    def test_only_new_messages_are_scanned(self):
        old_msg = {"content": [{"toolUse": {"name": "old", "input": {}}}]}
        agent = _make_agent_for_wrap(messages=[old_msg])
        agent.return_value = "ok"
        wrapped = _wrap_agent_with_trace(agent)
        trace: list[dict] = []
        wrapped("hi", trace)
        assert trace == []

    def test_trace_none_skips_capture(self):
        agent = _make_agent_for_wrap(messages=[])

        def _call_side_effect(_msg):
            agent.messages.append({"content": [{"toolUse": {"name": "x", "input": {}}}]})
            return "ok"

        agent.side_effect = _call_side_effect
        _wrap_agent_with_trace(agent)("hi", None)  # must not raise

    def test_swallows_message_format_errors(self):
        agent = _make_agent_for_wrap(messages=[])
        agent.side_effect = lambda _msg: agent.messages.append("not a dict") or "ok"
        wrapped = _wrap_agent_with_trace(agent)
        trace: list[dict] = []
        wrapped("hi", trace)
        assert trace == []
