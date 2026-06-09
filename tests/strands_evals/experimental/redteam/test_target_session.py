"""Tests for TargetSession protocol implementations."""

from unittest.mock import MagicMock

from strands import Agent

from strands_evals.experimental.redteam.strategies.target_session import (
    MALFORMED_TOOL_NAME,
    StrandsAgentSession,
    _tool_uses_in,
)


def _mock_agent(messages=None):
    agent = MagicMock()
    agent.messages = messages if messages is not None else []
    return agent


# ---------------------------------------------------------------------------
# _tool_uses_in — the single place that knows the SDK message schema (J7)
# ---------------------------------------------------------------------------


class TestToolUsesIn:
    def test_extracts_tool_use_blocks(self):
        messages = [{"content": [{"toolUse": {"name": "search", "input": {"q": "x"}}}]}]
        assert _tool_uses_in(messages) == [{"name": "search", "input": {"q": "x"}}]

    def test_ignores_non_tool_use_blocks(self):
        messages = [{"content": [{"text": "just text"}]}]
        assert _tool_uses_in(messages) == []

    def test_defaults_missing_name_and_input(self):
        messages = [{"content": [{"toolUse": {}}]}]
        assert _tool_uses_in(messages) == [{"name": "", "input": {}}]

    def test_multiple_blocks_across_messages_in_order(self):
        messages = [
            {"content": [{"toolUse": {"name": "a", "input": {}}}]},
            {"content": [{"text": "x"}, {"toolUse": {"name": "b", "input": {}}}]},
        ]
        assert _tool_uses_in(messages) == [{"name": "a", "input": {}}, {"name": "b", "input": {}}]

    def test_malformed_tool_use_block_recorded_as_placeholder(self):
        # A toolUse that isn't the expected mapping is recorded as a distinguishable
        # placeholder (with a warning), not skipped or raised: it must not discard
        # the case's trace, and the trace length must stay honest so a backtracking
        # strategy still sees that a tool call happened this turn.
        messages = [
            {"content": [{"toolUse": "not a dict"}]},
            {"content": [{"toolUse": {"name": "good", "input": {}}}]},
        ]
        result = _tool_uses_in(messages)
        # length stays honest (2, not 1) -- this is what the J4 tool-call gate keys on
        assert len(result) == 2
        assert result == [{"name": MALFORMED_TOOL_NAME, "input": {}}, {"name": "good", "input": {}}]

    def test_missing_content_key_is_skipped(self):
        assert _tool_uses_in([{"role": "user"}]) == []

    def test_does_not_raise_on_malformed_message_or_block(self):
        # Schema drift in the message/block shape (not just the toolUse value) must
        # be tolerated, never raised: a raise would propagate to the experiment's
        # per-case catch and mislabel a possible breach as defended.
        assert _tool_uses_in(["not a dict"]) == []  # non-dict message
        assert _tool_uses_in([{"content": [None]}]) == []  # non-dict block
        assert _tool_uses_in([{"content": None}]) == []  # content is None


# ---------------------------------------------------------------------------
# StrandsAgentSession — invoke + trace
# ---------------------------------------------------------------------------


class TestStrandsAgentSessionInvoke:
    def test_returns_string_response(self):
        agent = _mock_agent()
        agent.return_value = "hello back"
        assert StrandsAgentSession(agent).invoke("hi") == "hello back"

    def test_appends_tool_uses_to_trace(self):
        agent = _mock_agent(messages=[])
        new_msg = {"content": [{"toolUse": {"name": "search", "input": {"q": "x"}}}]}
        agent.side_effect = lambda _m: (agent.messages.append(new_msg), "ok")[1]
        session = StrandsAgentSession(agent)
        session.invoke("hi")
        assert session.trace == [{"name": "search", "input": {"q": "x"}}]

    def test_only_new_messages_are_scanned(self):
        old_msg = {"content": [{"toolUse": {"name": "old", "input": {}}}]}
        agent = _mock_agent(messages=[old_msg])
        agent.return_value = "ok"
        session = StrandsAgentSession(agent)
        session.invoke("hi")
        assert session.trace == []


# ---------------------------------------------------------------------------
# StrandsAgentSession — snapshot / restore (real Agent, the proof test)
# ---------------------------------------------------------------------------


class TestStrandsAgentSessionRewind:
    def _real_agent(self):
        # A real Agent (no model needed) so take_snapshot/load_snapshot exercise the
        # SDK's actual deepcopy rollback rather than a mock.
        return Agent(
            model=None,
            messages=[{"role": "user", "content": [{"text": "seed"}]}],
            callback_handler=None,
        )

    def test_restore_rolls_back_agent_messages(self):
        session = StrandsAgentSession(self._real_agent())
        snap = session.snapshot()
        # mutate the agent's history past the snapshot
        session._agent.messages.append({"role": "user", "content": [{"text": "refused turn"}]})
        assert len(session._agent.messages) == 2
        session.restore(snap)
        assert len(session._agent.messages) == 1
        assert session._agent.messages[0]["content"][0]["text"] == "seed"

    def test_restore_truncates_trace(self):
        session = StrandsAgentSession(self._real_agent())
        session.trace.append({"name": "before", "input": {}})
        snap = session.snapshot()
        session.trace.append({"name": "after_snapshot", "input": {}})
        session.restore(snap)
        # only the pre-snapshot trace entry survives; the ghost tool-call is gone
        assert session.trace == [{"name": "before", "input": {}}]

    def test_reset_clears_agent_history_and_trace(self):
        session = StrandsAgentSession(self._real_agent())
        session.trace.append({"name": "leftover", "input": {}})
        assert len(session._agent.messages) == 1  # seeded
        session.reset()
        assert session._agent.messages == []
        assert session.trace == []
