"""Tests for StrandsMultiAgentSession."""

import copy
from dataclasses import dataclass, field
from typing import Any

import pytest
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
        # No leaves needed — this test only checks that the orchestrator's result
        # text is returned. Driving a real Agent here would call Bedrock.
        root = _FakeOrchestrator({}, result_text="hello back")
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
        with pytest.raises(TypeError, match="_MultiAgentSnapshot"):
            s.restore(bogus)


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


# ---------------------------------------------------------------------------
# restore ordering vs Graph/Swarm.deserialize_state
# ---------------------------------------------------------------------------


class _GraphLikeOrchestrator(_FakeOrchestrator):
    """A `_FakeOrchestrator` whose `deserialize_state` mimics
    `Graph`/`Swarm.deserialize_state` for completed payloads.

    The real SDK resets every leaf's `messages` / `state` / `_model_state` to the
    values captured at `GraphBuilder.build()` time (`GraphNode.__post_init__`)
    whenever the payload has no `next_nodes_to_execute` -- the normal between-turn
    state for a `PENDING` or `COMPLETED` orchestrator. Without this fake,
    `_FakeOrchestrator.deserialize_state` only round-trips an opaque dict and the
    leaf-vs-orchestrator restore order is invisible.

    The fake snapshots each leaf's `messages` and `state` at the point the
    orchestrator was wrapped (~ `GraphBuilder.build()` time) and replays them on
    completed-payload restores.
    """

    def __init__(self, nodes: dict[str, Any]) -> None:
        super().__init__(nodes)
        # Capture per-leaf build-time messages + state the same way
        # GraphNode.__post_init__ does. We deep-copy each leaf's `state` object
        # whole and replay it on reset; this keeps us agnostic to whether the
        # SDK exposes state as `AgentState` or `JSONSerializableDict`.
        self._initial_messages: dict[str, Any] = {}
        self._initial_state: dict[str, Any] = {}
        for node_id, node in self.nodes.items():
            ex = node.executor
            if hasattr(ex, "messages"):
                self._initial_messages[node_id] = copy.deepcopy(ex.messages)
            if hasattr(ex, "state"):
                self._initial_state[node_id] = copy.deepcopy(ex.state)

    def serialize_state(self) -> dict[str, Any]:
        # Match Graph: a settled orchestrator returns a payload with no
        # `next_nodes_to_execute`, which is exactly the case that triggers leaf
        # reset on the way back in.
        return {"type": "graph-like", "next_nodes_to_execute": [], "orch_state": dict(self._orch_state)}

    def deserialize_state(self, payload: dict[str, Any]) -> None:
        if not payload.get("next_nodes_to_execute"):
            for node_id, node in self.nodes.items():
                ex = node.executor
                if node_id in self._initial_messages and hasattr(ex, "messages"):
                    ex.messages = copy.deepcopy(self._initial_messages[node_id])
                if node_id in self._initial_state and hasattr(ex, "state"):
                    ex.state = copy.deepcopy(self._initial_state[node_id])
        self._orch_state = dict(payload.get("orch_state", {}))


class TestRestoreOrderVsGraphReset:
    """Regression: leaves must outlive `deserialize_state`'s build-time reset.

    With the loops in the wrong order, every backtrack against a real
    `Graph`/`Swarm` target restarts the conversation from build-time state,
    silently breaking Crescendo's "escalate from accumulated context" guarantee.
    """

    def test_restore_preserves_post_build_messages_on_completed_payload(self):
        # Build-time leaf has just the seed message; the case appended a turn
        # AFTER build, then we snapshot. Restoring must end with snapshot-time
        # state (seed + 1 turn), NOT build-time state (seed only) -- the latter
        # is what the buggy ordering produces.
        leaf = _real_agent("seed")
        root = _GraphLikeOrchestrator({"a": leaf})
        _append_to(leaf, "case turn")  # post-build mutation
        assert len(leaf.messages) == 2

        ck = StrandsMultiAgentSession(root).snapshot()
        _append_to(leaf, "to be rolled back")  # past the snapshot
        assert len(leaf.messages) == 3

        StrandsMultiAgentSession(root).restore(ck)

        # 2 messages = snapshot-time state; 1 would mean deserialize_state ran
        # AFTER load_snapshot and reset the leaf to build-time.
        assert len(leaf.messages) == 2
        assert leaf.messages[0]["content"][0]["text"] == "seed"
        assert leaf.messages[1]["content"][0]["text"] == "case turn"

    def test_baseline_reset_preserves_seeded_post_build_state(self):
        # The task runner captures the baseline AFTER the harness wraps the
        # target -- so any leaf history seeded after `GraphBuilder.build()` lives
        # in the baseline, and `reset()` must restore to it. With the loops in
        # the wrong order, `deserialize_state` would wipe that seeded state and
        # every case would start from build-time.
        leaf = _real_agent("seed")
        root = _GraphLikeOrchestrator({"a": leaf})
        _append_to(leaf, "post-build seed")  # seeded after the orchestrator was built
        assert len(leaf.messages) == 2

        baseline = StrandsMultiAgentSession(root).snapshot().agent_snapshot
        s = StrandsMultiAgentSession(root, baseline=baseline)

        _append_to(leaf, "case mutation")
        leaf.state.set("exfiltrated", True)
        s.reset()

        assert len(leaf.messages) == 2
        assert leaf.messages[1]["content"][0]["text"] == "post-build seed"
        assert leaf.state.get("exfiltrated") is None


# ---------------------------------------------------------------------------
# Swarm-shaped resume-flag handling
# ---------------------------------------------------------------------------


class _SwarmLikeOrchestrator(_FakeOrchestrator):
    """A `_FakeOrchestrator` whose `(de)serialize_state` mimics `Swarm`.

    Two SDK details matter for `_restore` correctness and aren't exercised by
    the Graph-shaped fake:

    1. `Swarm.serialize_state` ALWAYS emits `next_nodes_to_execute` (empty list
       for a settled swarm with no current_node / no handoff).
    2. `Swarm.deserialize_state` checks key PRESENCE (`"key" in payload`), not
       truthiness, so a settled-but-round-tripped swarm always takes the
       resume branch -- setting `_resume_from_session=True` while leaving
       `current_node=None`. The next invoke then dereferences `current_node.node_id`
       and crashes; the experiment's per-case try/except records score=0 and
       silently mislabels every Swarm case as defended.

    This fake reproduces both behaviors so the session's "force fresh invoke
    when payload is settled" guard can be regression-tested without booting a
    real swarm.
    """

    def __init__(self, nodes: dict[str, Any], *, status: str = "pending") -> None:
        super().__init__(nodes)
        self._status = status
        self._resume_from_session = False
        self.current_node: str | None = "a"  # set by a fresh invoke
        self.invoked_with: str | None = None

    def __call__(self, task, invocation_state=None, **kwargs):
        # Mirror Swarm.invoke: when _resume_from_session is True, skip the
        # current_node re-init, then dereference it -- crashes if None. We
        # don't iterate leaves here (the base class would call them as real
        # Agents with no model) -- the guard's effect is observable from the
        # current_node behavior alone.
        if not self._resume_from_session:
            self.current_node = "a"
        # The crash the real Swarm hits at swarm.py:419: AttributeError on a
        # None current_node. The guard is what stops it.
        _ = self.current_node.upper()  # type: ignore[union-attr]
        self.invoked_with = task
        return _FakeMultiAgentResult(self._result_text)

    def serialize_state(self) -> dict[str, Any]:
        # Always emit `next_nodes_to_execute` (empty for a settled swarm), and
        # carry a `status` so the session's resume-flag guard can read it.
        return {
            "type": "swarm-like",
            "status": self._status,
            "next_nodes_to_execute": [],
            "orch_state": dict(self._orch_state),
        }

    def deserialize_state(self, payload: dict[str, Any]) -> None:
        # Membership check, not truthiness -- the bug's root cause on the SDK side.
        self._resume_from_session = "next_nodes_to_execute" in payload
        if self._resume_from_session:
            # _from_dict equivalent: empty next_node_ids leaves current_node alone,
            # which is None on a freshly-restored fake.
            self.current_node = None
        self._orch_state = dict(payload.get("orch_state", {}))


class TestSwarmShapedResumeFlagGuard:
    """Regression: a settled-payload restore must leave the orchestrator able
    to start a fresh invoke, not stuck in a half-resumed state.

    Without the guard, `Swarm.deserialize_state` always takes the resume
    branch (key presence, not truthiness), `current_node` is None, and the
    next invoke crashes -- which the experiment's per-case try/except buries
    as score=0 / "defended", silently mislabeling every Swarm case safe.
    """

    def test_invoke_after_restore_does_not_crash(self):
        # Settled status (COMPLETED) + always-emit-next-nodes is exactly the
        # Swarm-after-a-turn shape. Without _force_fresh_invoke_if_settled,
        # this invoke raises AttributeError.
        leaf = _real_agent("seed")
        root = _SwarmLikeOrchestrator({"a": leaf}, status="completed")
        s = StrandsMultiAgentSession(root)

        ck = s.snapshot()
        s.restore(ck)

        # The next invoke must succeed; the guard must have cleared
        # _resume_from_session so the orchestrator re-initializes current_node.
        result = s.invoke("next attack turn")
        assert result == "ok"
        assert root.invoked_with == "next attack turn"
        assert root._resume_from_session is False

    def test_reset_with_baseline_does_not_crash(self):
        # The task runner calls session.reset() before every case -- so the
        # bug surfaces on case 1 turn 1, not just on backtracks. Same guard,
        # same fix; this test asserts the case-startup path specifically.
        leaf = _real_agent("seed")
        root = _SwarmLikeOrchestrator({"a": leaf}, status="completed")
        baseline = StrandsMultiAgentSession(root).snapshot().agent_snapshot
        s = StrandsMultiAgentSession(root, baseline=baseline)

        s.reset()
        result = s.invoke("case 1 turn 1")
        assert result == "ok"
        assert root.invoked_with == "case 1 turn 1"

    def test_pending_and_failed_payloads_also_force_fresh_invoke(self):
        # PENDING and FAILED are the other settled statuses; both must clear
        # the resume flag too so a fresh invoke re-initializes cleanly.
        for status in ("pending", "failed"):
            leaf = _real_agent("seed")
            root = _SwarmLikeOrchestrator({"a": leaf}, status=status)
            s = StrandsMultiAgentSession(root)
            s.restore(s.snapshot())
            assert root._resume_from_session is False, f"status={status} left resume flag set"

    def test_interrupted_payload_keeps_resume_flag(self):
        # An INTERRUPTED payload genuinely needs to resume mid-execution; the
        # guard must NOT clear the flag in that case (the orchestrator owns
        # current_node for an interrupted resume).
        leaf = _real_agent("seed")
        root = _SwarmLikeOrchestrator({"a": leaf}, status="interrupted")
        s = StrandsMultiAgentSession(root)
        s.restore(s.snapshot())
        # _resume_from_session was set by deserialize_state and the guard left
        # it alone because the payload status isn't settled.
        assert root._resume_from_session is True

    def test_settled_payload_without_resume_attr_logs_warning(self, caplog):
        # The guard reaches into a private SDK attribute. If a future SDK
        # rename drops `_resume_from_session`, the guard would silently no-op
        # and the original bug (score=0 cases looking "defended") would come
        # back unnoticed. The warning is the tripwire that catches the rename.
        class _NoResumeAttr(_FakeOrchestrator):
            # serialize_state emits a settled `status` so the guard runs;
            # deserialize_state never sets `_resume_from_session`, so the
            # attribute is genuinely absent at the moment the guard checks.
            def serialize_state(self) -> dict[str, Any]:
                return {"type": "no-resume", "status": "completed"}

            def deserialize_state(self, payload: dict[str, Any]) -> None:
                pass

        leaf = _real_agent("seed")
        root = _NoResumeAttr({"a": leaf})
        s = StrandsMultiAgentSession(root)
        with caplog.at_level(
            "WARNING", logger="strands_evals.experimental.redteam.strategies.target_session"
        ):
            s.restore(s.snapshot())
        assert any("_resume_from_session" in rec.message for rec in caplog.records), (
            "missing-attribute path must log a rename warning"
        )

    def test_non_settled_payload_without_resume_attr_does_not_warn(self, caplog):
        # A custom MultiAgentBase that doesn't carry `_resume_from_session` AND
        # doesn't return a settled status (or returns no status at all) should
        # not trip the warning -- the guard has no business firing there.
        leaf = _real_agent("seed")
        root = _FakeOrchestrator({"a": leaf})  # serialize_state emits no `status`
        s = StrandsMultiAgentSession(root)
        with caplog.at_level(
            "WARNING", logger="strands_evals.experimental.redteam.strategies.target_session"
        ):
            s.restore(s.snapshot())
        assert not any("_resume_from_session" in rec.message for rec in caplog.records)
