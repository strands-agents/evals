"""Tests for TargetSession protocol implementations."""

from unittest.mock import MagicMock

import pytest
from strands import Agent

from strands_evals.experimental.redteam.strategies.target_session import (
    AgentTargetSession,
    CallableTargetSession,
)


def _mock_agent(messages=None):
    agent = MagicMock()
    agent.messages = messages if messages is not None else []
    return agent


# ---------------------------------------------------------------------------
# AgentTargetSession — invoke + trace
# ---------------------------------------------------------------------------


class TestAgentTargetSessionInvoke:
    def test_returns_string_response(self):
        agent = _mock_agent()
        agent.return_value = "hello back"
        assert AgentTargetSession(agent).invoke("hi") == "hello back"

    def test_appends_tool_uses_to_trace(self):
        agent = _mock_agent(messages=[])
        new_msg = {"content": [{"toolUse": {"name": "search", "input": {"q": "x"}}}]}
        agent.side_effect = lambda _m: (agent.messages.append(new_msg), "ok")[1]
        session = AgentTargetSession(agent)
        session.invoke("hi")
        assert session.trace == [{"name": "search", "input": {"q": "x"}}]

    def test_only_new_messages_are_scanned(self):
        old_msg = {"content": [{"toolUse": {"name": "old", "input": {}}}]}
        agent = _mock_agent(messages=[old_msg])
        agent.return_value = "ok"
        session = AgentTargetSession(agent)
        session.invoke("hi")
        assert session.trace == []

    def test_swallows_message_format_errors(self):
        agent = _mock_agent(messages=[])
        agent.side_effect = lambda _m: (agent.messages.append("not a dict"), "ok")[1]
        session = AgentTargetSession(agent)
        session.invoke("hi")  # must not raise
        assert session.trace == []

    def test_supports_rewind_is_true(self):
        assert AgentTargetSession(_mock_agent()).supports_rewind is True


# ---------------------------------------------------------------------------
# AgentTargetSession — snapshot / restore (real Agent, the proof test)
# ---------------------------------------------------------------------------


class TestAgentTargetSessionRewind:
    def _real_agent(self):
        # A real Agent (no model needed) so take_snapshot/load_snapshot exercise the
        # SDK's actual deepcopy rollback rather than a mock.
        return Agent(
            model=None,
            messages=[{"role": "user", "content": [{"text": "seed"}]}],
            callback_handler=None,
        )

    def test_restore_rolls_back_agent_messages(self):
        session = AgentTargetSession(self._real_agent())
        snap = session.snapshot()
        # mutate the agent's history past the snapshot
        session._agent.messages.append({"role": "user", "content": [{"text": "refused turn"}]})
        assert len(session._agent.messages) == 2
        session.restore(snap)
        assert len(session._agent.messages) == 1
        assert session._agent.messages[0]["content"][0]["text"] == "seed"

    def test_restore_truncates_trace(self):
        session = AgentTargetSession(self._real_agent())
        session._trace.append({"name": "before", "input": {}})
        snap = session.snapshot()
        session._trace.append({"name": "after_snapshot", "input": {}})
        session.restore(snap)
        # only the pre-snapshot trace entry survives; the ghost tool-call is gone
        assert session.trace == [{"name": "before", "input": {}}]

    def test_reset_clears_agent_history_and_trace(self):
        session = AgentTargetSession(self._real_agent())
        session._trace.append({"name": "leftover", "input": {}})
        assert len(session._agent.messages) == 1  # seeded
        session.reset()
        assert session._agent.messages == []
        assert session.trace == []


# ---------------------------------------------------------------------------
# CallableTargetSession
# ---------------------------------------------------------------------------


class TestCallableTargetSession:
    def test_invoke_plain_string(self):
        session = CallableTargetSession(lambda m: f"echo:{m}")
        assert session.invoke("hi") == "echo:hi"

    def test_invoke_dict_merges_trace_and_returns_output(self):
        payload = {"output": "done", "trace": [{"name": "t", "input": {}}]}
        session = CallableTargetSession(lambda _m: payload)
        assert session.invoke("hi") == "done"
        assert session.trace == [{"name": "t", "input": {}}]

    def test_supports_rewind_is_false(self):
        assert CallableTargetSession(lambda _m: "x").supports_rewind is False

    def test_snapshot_raises(self):
        with pytest.raises(NotImplementedError):
            CallableTargetSession(lambda _m: "x").snapshot()

    def test_restore_raises(self):
        session = CallableTargetSession(lambda _m: "x")
        with pytest.raises(NotImplementedError):
            session.restore(MagicMock())

    def test_reset_clears_trace(self):
        session = CallableTargetSession(lambda _m: "x")
        session._trace.append({"name": "leftover", "input": {}})
        session.reset()
        assert session.trace == []
