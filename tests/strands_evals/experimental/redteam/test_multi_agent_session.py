"""Tests for StrandsMultiAgentSession."""

from dataclasses import dataclass, field
from typing import Any

from strands import Agent
from strands.multiagent.base import MultiAgentBase

from strands_evals.experimental.redteam.strategies.target_session import (
    StrandsMultiAgentSession,
    TargetCheckpoint,
    _MultiAgentSnapshot,
)

# ---------------------------------------------------------------------------
# Fakes -- minimal MultiAgentBase the session can drive without a real LLM
# ---------------------------------------------------------------------------


@dataclass
class _FakeNode:
    node_id: str
    executor: Any


class _FakeOrchestrator(MultiAgentBase):
    """A MultiAgentBase that exposes the surface StrandsMultiAgentSession uses.

    The real Graph/Swarm classes do far more, but the session only touches
    `nodes`, `__call__`, `serialize_state`, and `deserialize_state` --
    so a fake with those is enough to exercise every code path without booting
    a model. `__call__` walks the leaves and feeds each one the message, which
    is all that's needed to drive the session's tail-diff trace capture.
    """

    id: str = "root"

    def __init__(
        self,
        nodes: dict[str, Any],
        *,
        result_text: str = "ok",
        invoke_message: str | None = None,
    ) -> None:
        self.nodes = {nid: _FakeNode(nid, ex) for nid, ex in nodes.items()}
        self._result_text = result_text
        # Optional override for what the leaves are sent (defaults to forwarding).
        self._invoke_message = invoke_message
        # Seedable orchestrator-level state so serialize/deserialize can be
        # exercised end-to-end.
        self._orch_state: dict[str, Any] = {"version": 1, "scratch": None}

    async def invoke_async(self, task, invocation_state=None, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def __call__(self, task, invocation_state=None, **kwargs):
        msg = self._invoke_message if self._invoke_message is not None else task
        for node in self.nodes.values():
            ex = node.executor
            if isinstance(ex, _FakeOrchestrator):
                ex(msg)
            elif isinstance(ex, Agent):
                ex(msg)
        return _FakeMultiAgentResult(self._result_text)

    def serialize_state(self) -> dict[str, Any]:
        return {"type": "fake", "id": self.id, "orch_state": dict(self._orch_state)}

    def deserialize_state(self, payload: dict[str, Any]) -> None:
        self._orch_state = dict(payload.get("orch_state", {}))


@dataclass
class _FakeMultiAgentResult:
    """The minimum shape `_multi_agent_result_text` falls back to via `str()`."""

    text: str
    results: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


class _FakeNonAgentExecutor:
    """An executor that is neither Agent nor MultiAgentBase -- must be skipped."""


def _real_agent(seed_text: str | None = None) -> Agent:
    """Build a real Agent (no model needed) so take_snapshot/load_snapshot
    exercise the SDK path rather than mocks. Mirrors the helper in
    test_target_session.py."""
    messages = [{"role": "user", "content": [{"text": seed_text}]}] if seed_text else []
    return Agent(model=None, messages=messages, callback_handler=None)


def _append_to(agent: Agent, text: str) -> None:
    """Helper that pushes a turn the same way Agent.__call__ would, without
    actually invoking a model."""
    agent.messages.append({"role": "user", "content": [{"text": text}]})


# ---------------------------------------------------------------------------
# Tree indexing
# ---------------------------------------------------------------------------


class TestIndexTree:
    def test_indexes_top_level_agents(self):
        a = _real_agent()
        b = _real_agent()
        root = _FakeOrchestrator({"a": a, "b": b})
        s = StrandsMultiAgentSession(root)
        assert set(s._agent_index.keys()) == {("a",), ("b",)}
        assert s._agent_index[("a",)] is a
        assert set(s._orch_index.keys()) == {()}

    def test_recurses_into_nested_orchestrators(self):
        # Root contains a nested orchestrator with two leaves -- both must be indexed
        # at their full path so a node id repeated under different parents stays
        # distinguishable.
        leaf_a = _real_agent()
        leaf_b = _real_agent()
        nested = _FakeOrchestrator({"x": leaf_a, "y": leaf_b})
        root = _FakeOrchestrator({"sub": nested})
        s = StrandsMultiAgentSession(root)
        assert set(s._agent_index.keys()) == {("sub", "x"), ("sub", "y")}
        assert set(s._orch_index.keys()) == {(), ("sub",)}

    def test_skips_unsupported_executors(self):
        # A custom AgentBase subclass without take_snapshot can't round-trip; the
        # session must not silently include it (would lie to a backtracking strategy
        # about coverage) but must not raise either (other leaves are still valid).
        a = _real_agent()
        root = _FakeOrchestrator({"good": a, "bad": _FakeNonAgentExecutor()})
        s = StrandsMultiAgentSession(root)
        assert set(s._agent_index.keys()) == {("good",)}


# ---------------------------------------------------------------------------
# invoke + trace capture
# ---------------------------------------------------------------------------


class TestInvoke:
    def test_returns_string_response(self):
        root = _FakeOrchestrator({"a": _real_agent()}, result_text="hello back")
        assert StrandsMultiAgentSession(root).invoke("hi") == "hello back"

    def test_diffs_each_leaf_for_tool_uses(self):
        # The fake __call__ doesn't run a real model, so we synthesize the message
        # the agent would have appended (a tool-use block) before invoke captures
        # the tail. Two leaves, each contributing one tool use; both must surface.
        leaf_a = _real_agent()
        leaf_b = _real_agent()

        class _Recording(_FakeOrchestrator):
            def __call__(self, task, invocation_state=None, **kwargs):
                leaf_a.messages.append({"content": [{"toolUse": {"name": "search_a", "input": {}}}]})
                leaf_b.messages.append({"content": [{"toolUse": {"name": "search_b", "input": {}}}]})
                return _FakeMultiAgentResult("ok")

        root = _Recording({"a": leaf_a, "b": leaf_b})
        s = StrandsMultiAgentSession(root)
        s.invoke("hi")
        names = sorted(t["name"] for t in s.trace)
        assert names == ["search_a", "search_b"]

    def test_only_new_messages_are_scanned(self):
        # A leaf that already has prior tool uses must NOT have them re-counted on
        # the next invoke -- only the tail appended during this call. Use a no-op
        # orchestrator so the prior (deliberately minimal) tool-use block doesn't
        # have to satisfy the SDK's full message schema.
        leaf = _real_agent()
        leaf.messages.append({"content": [{"toolUse": {"name": "old", "input": {}}}]})

        class _NoOp(_FakeOrchestrator):
            def __call__(self, task, invocation_state=None, **kwargs):
                return _FakeMultiAgentResult("ok")

        root = _NoOp({"a": leaf})
        s = StrandsMultiAgentSession(root)
        s.invoke("hi")
        assert s.trace == []


# ---------------------------------------------------------------------------
# snapshot / restore
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_snapshot_captures_every_leaf_and_orchestrator(self):
        leaf_a = _real_agent()
        leaf_b = _real_agent()
        nested = _FakeOrchestrator({"y": leaf_b})
        root = _FakeOrchestrator({"a": leaf_a, "sub": nested})
        s = StrandsMultiAgentSession(root)

        ck = s.snapshot()
        assert isinstance(ck, TargetCheckpoint)
        assert isinstance(ck.agent_snapshot, _MultiAgentSnapshot)
        assert set(ck.agent_snapshot.agents.keys()) == {("a",), ("sub", "y")}
        assert set(ck.agent_snapshot.orchestrators.keys()) == {(), ("sub",)}

    def test_restore_rolls_back_each_leaf_messages(self):
        leaf_a = _real_agent("seed_a")
        leaf_b = _real_agent("seed_b")
        root = _FakeOrchestrator({"a": leaf_a, "b": leaf_b})
        s = StrandsMultiAgentSession(root)

        ck = s.snapshot()
        # mutate every leaf past the snapshot
        _append_to(leaf_a, "case turn a")
        _append_to(leaf_b, "case turn b")
        assert len(leaf_a.messages) == 2
        assert len(leaf_b.messages) == 2

        s.restore(ck)
        assert len(leaf_a.messages) == 1
        assert leaf_a.messages[0]["content"][0]["text"] == "seed_a"
        assert len(leaf_b.messages) == 1
        assert leaf_b.messages[0]["content"][0]["text"] == "seed_b"

    def test_restore_rolls_back_orchestrator_state(self):
        # serialize_state / deserialize_state are the orchestrator-bookkeeping
        # channel; mutating it post-snapshot and then restoring must revert it.
        root = _FakeOrchestrator({"a": _real_agent()})
        s = StrandsMultiAgentSession(root)
        ck = s.snapshot()
        root._orch_state["scratch"] = "dirty"
        s.restore(ck)
        assert root._orch_state["scratch"] is None

    def test_restore_truncates_trace(self):
        root = _FakeOrchestrator({"a": _real_agent()})
        s = StrandsMultiAgentSession(root)
        s.trace.append({"name": "before", "input": {}})
        ck = s.snapshot()
        s.trace.append({"name": "after_snapshot", "input": {}})
        s.restore(ck)
        assert s.trace == [{"name": "before", "input": {}}]

    def test_restore_rejects_wrong_checkpoint_payload(self):
        # An opaque agent_snapshot that isn't a _MultiAgentSnapshot is a programmer
        # error (someone mixed checkpoints across session types). Fail loud rather
        # than silently passing through deepcopy and exploding deeper.
        root = _FakeOrchestrator({"a": _real_agent()})
        s = StrandsMultiAgentSession(root)
        bogus = TargetCheckpoint(agent_snapshot="not a multi-agent snapshot", trace_len=0)
        try:
            s.restore(bogus)
        except TypeError as e:
            assert "_MultiAgentSnapshot" in str(e)
        else:  # pragma: no cover - guard
            raise AssertionError("expected TypeError")


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_with_baseline_restores_as_constructed_state(self):
        leaf = _real_agent("seed")
        root = _FakeOrchestrator({"a": leaf})
        baseline = StrandsMultiAgentSession(root).snapshot().agent_snapshot
        s = StrandsMultiAgentSession(root, baseline=baseline)

        # simulate a case mutating the target
        _append_to(leaf, "case turn")
        leaf.state.set("exfiltrated", True)
        root._orch_state["scratch"] = "dirty"
        s.trace.append({"name": "leftover", "input": {}})

        s.reset()

        assert leaf.messages == [{"role": "user", "content": [{"text": "seed"}]}]
        # full snapshot fields reset, not just messages -- agent state too
        assert leaf.state.get("exfiltrated") is None
        # orchestrator-level state rolled back through deserialize_state
        assert root._orch_state["scratch"] is None
        assert s.trace == []

    def test_baseline_survives_repeated_resets(self):
        # The baseline is captured once and replayed every case; load_snapshot must
        # copy out of it so a case mutation can't poison subsequent resets.
        leaf = _real_agent("seed")
        root = _FakeOrchestrator({"a": leaf})
        baseline = StrandsMultiAgentSession(root).snapshot().agent_snapshot
        s = StrandsMultiAgentSession(root, baseline=baseline)
        for i in range(3):
            _append_to(leaf, f"case{i}")
            leaf.state.set("leak", i)
            root._orch_state["scratch"] = f"dirty{i}"
            s.reset()
            assert leaf.messages == [{"role": "user", "content": [{"text": "seed"}]}]
            assert leaf.state.get("leak") is None
            assert root._orch_state["scratch"] is None

    def test_reset_without_baseline_clears_leaf_messages(self):
        # No baseline: best-effort fallback clears each leaf's messages, mirroring
        # StrandsAgentSession's no-baseline behavior. Orchestrator state is left as
        # the SDK has it (next invoke refreshes it).
        leaf_a = _real_agent("seed")
        leaf_b = _real_agent()
        _append_to(leaf_b, "added")
        root = _FakeOrchestrator({"a": leaf_a, "b": leaf_b})
        s = StrandsMultiAgentSession(root)
        s.trace.append({"name": "leftover", "input": {}})
        s.reset()
        assert leaf_a.messages == []
        assert leaf_b.messages == []
        assert s.trace == []
