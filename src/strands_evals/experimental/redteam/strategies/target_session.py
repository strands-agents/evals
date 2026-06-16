"""Target session protocol and implementations.

A `TargetSession` is the handle a strategy uses to talk to the target: it sends messages and exposes
snapshot/restore for backtracking strategies. The task runner wraps an `Agent` in `StrandsAgentSession` and
a `MultiAgentBase` in `StrandsMultiAgentSession`; custom targets can implement the `Protocol` directly.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from strands import Agent, Snapshot
from strands.multiagent.base import MultiAgentBase, Status
from strands.types.content import Message

logger = logging.getLogger(__name__)


# Distinct from a real empty-named tool call so the evaluator can tell corruption from a genuine no-name
# tool use while the entry still counts toward trace length.
MALFORMED_TOOL_NAME = "<malformed>"


class ToolUseEntry(TypedDict):
    """One tool invocation captured into a session's trace."""

    name: str
    input: dict[str, Any]


def _tool_uses_in(messages: list[Message]) -> list[ToolUseEntry]:
    """Extract tool-use entries from Strands messages, never raising on schema drift.

    A malformed `toolUse` block is logged and recorded as a `MALFORMED_TOOL_NAME` placeholder so the trace
    length stays honest (a backtracking strategy decides whether a turn drove a tool call by whether the
    trace grew).
    """
    tool_uses: list[ToolUseEntry] = []
    for message in messages:
        if not isinstance(message, dict):
            logger.warning("shape=<%s> | unexpected message, skipping", type(message).__name__)
            continue
        for block in message.get("content") or []:
            if not isinstance(block, dict) or "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            if not isinstance(tool_use, dict):
                logger.warning("shape=<%s> | unexpected toolUse block, recording placeholder", type(tool_use).__name__)
                tool_uses.append({"name": MALFORMED_TOOL_NAME, "input": {}})
                continue
            tool_uses.append({"name": tool_use.get("name", ""), "input": tool_use.get("input", {})})
    return tool_uses


@dataclass
class TargetCheckpoint:
    """An opaque rewind point returned by `TargetSession.snapshot`.

    Strategies treat it as opaque — take one, pass it back to `restore`.

    Attributes:
        agent_snapshot: Saved target state, opaque to strategies.
        trace_len: `len(session.trace)` when the checkpoint was taken.
    """

    agent_snapshot: Any
    trace_len: int


class TargetSession(Protocol):
    """Contract a strategy uses to interact with the target under test."""

    trace: list[ToolUseEntry]
    """Tool-use entries captured across this session's `invoke` calls."""

    def invoke(self, message: str) -> str:
        """Send `message` to the target, return the response, and append tool uses to `trace`."""
        ...

    def reset(self) -> None:
        """Clear per-case state so the session starts a fresh conversation."""
        ...

    def snapshot(self) -> TargetCheckpoint:
        """Capture the target's current state as an opaque checkpoint."""
        ...

    def restore(self, checkpoint: TargetCheckpoint) -> None:
        """Roll the target back to `checkpoint` and truncate `trace` to its length then."""
        ...


class StrandsAgentSession:
    """A `TargetSession` backed by a `strands.Agent`, rewindable via the SDK snapshot API."""

    def __init__(self, agent: Agent, *, baseline: Snapshot | None = None) -> None:
        """Initialize the session.

        Args:
            agent: The target agent to drive and snapshot.
            baseline: Clean snapshot to roll back to in `reset`. When `None`, `reset` falls back to clearing
                messages only.
        """
        self._agent = agent
        self._baseline = baseline
        self.trace: list[ToolUseEntry] = []

    def invoke(self, message: str) -> str:
        result, new_messages = self._send(message)
        self.trace.extend(_tool_uses_in(new_messages))
        return str(result)

    def _send(self, message: str) -> tuple[Any, list[Message]]:
        """Return the agent's result plus the messages it appended."""
        messages_before = len(self._agent.messages)
        result = self._agent(message)
        return result, self._agent.messages[messages_before:]

    def reset(self) -> None:
        # Use load_snapshot so reset covers every field snapshot() captures, not just messages.
        if self._baseline is not None:
            self._agent.load_snapshot(self._baseline)
        else:
            self._agent.messages.clear()
        self.trace.clear()

    def snapshot(self) -> TargetCheckpoint:
        return TargetCheckpoint(
            agent_snapshot=self._agent.take_snapshot(preset="session"),
            trace_len=len(self.trace),
        )

    def restore(self, checkpoint: TargetCheckpoint) -> None:
        self._agent.load_snapshot(checkpoint.agent_snapshot)
        del self.trace[checkpoint.trace_len :]


# Path of node ids from the root orchestrator to a leaf. Tuple (not str) so the
# same node id under different parents stays distinguishable.
_AgentPath = tuple[str, ...]


@dataclass
class _MultiAgentSnapshot:
    """Composite snapshot: per-leaf `Snapshot` + per-orchestrator `serialize_state` dict."""

    agents: dict[_AgentPath, Snapshot]
    orchestrators: dict[_AgentPath, dict[str, Any]]


class StrandsMultiAgentSession:
    """A `TargetSession` backed by a `strands.multiagent.MultiAgentBase`.

    Walks the tree once at init to index every leaf `Agent` and orchestrator; `snapshot`/`restore` round-trip
    through `Agent.take_snapshot` and `MultiAgentBase.serialize_state`/`deserialize_state`. Tool-use trace is
    captured by diffing each leaf's `messages` tail across an `invoke` call.
    """

    def __init__(self, root: MultiAgentBase, *, baseline: _MultiAgentSnapshot | None = None) -> None:
        """Initialize the session.

        Args:
            root: The target orchestrator; topology must not change after construction.
            baseline: Clean composite snapshot to roll back to in `reset`. When `None`, `reset` falls back to
                clearing each leaf's `messages`.
        """
        self._root = root
        self._agent_index: dict[_AgentPath, Agent] = {}
        self._orch_index: dict[_AgentPath, MultiAgentBase] = {}
        self._index_tree(root, ())
        self._baseline = baseline
        self.trace: list[ToolUseEntry] = []

    def _index_tree(self, orch: MultiAgentBase, path: _AgentPath) -> None:
        """Recursively populate the path indexes for every leaf and orchestrator."""
        self._orch_index[path] = orch
        for node_id, node in orch.nodes.items():
            child_path = path + (node_id,)
            executor = node.executor
            if isinstance(executor, MultiAgentBase):
                self._index_tree(executor, child_path)
            elif isinstance(executor, Agent):
                self._agent_index[child_path] = executor
            else:
                logger.warning(
                    "path=<%s>, type=<%s> | executor is not Agent or MultiAgentBase, skipping snapshot coverage",
                    "/".join(child_path),
                    type(executor).__name__,
                )

    def invoke(self, message: str) -> str:
        """Send `message` to the root and capture tool uses from every leaf."""
        before = {path: len(agent.messages) for path, agent in self._agent_index.items()}
        result = self._root(message)
        for path, agent in self._agent_index.items():
            self.trace.extend(_tool_uses_in(agent.messages[before[path] :]))
        return _multi_agent_result_text(result)

    def reset(self) -> None:
        """Roll the tree back to the baseline (or clear messages if no baseline)."""
        if self._baseline is not None:
            self._restore(self._baseline)
        else:
            # No baseline: best-effort clear; orchestrator bookkeeping refreshes on the next invoke.
            for agent in self._agent_index.values():
                agent.messages.clear()
        self.trace.clear()

    def snapshot(self) -> TargetCheckpoint:
        """Capture a composite snapshot of every leaf and every orchestrator."""
        return TargetCheckpoint(
            agent_snapshot=_MultiAgentSnapshot(
                agents={path: agent.take_snapshot(preset="session") for path, agent in self._agent_index.items()},
                orchestrators={path: orch.serialize_state() for path, orch in self._orch_index.items()},
            ),
            trace_len=len(self.trace),
        )

    def restore(self, checkpoint: TargetCheckpoint) -> None:
        """Roll the tree back to `checkpoint` and truncate the trace."""
        if not isinstance(checkpoint.agent_snapshot, _MultiAgentSnapshot):
            raise TypeError(
                f"StrandsMultiAgentSession.restore: expected _MultiAgentSnapshot, "
                f"got {type(checkpoint.agent_snapshot).__name__}"
            )
        self._restore(checkpoint.agent_snapshot)
        del self.trace[checkpoint.trace_len :]

    def _restore(self, snapshot: _MultiAgentSnapshot) -> None:
        """Push a composite snapshot back into the tree.

        Orchestrators are restored FIRST, leaves LAST: `deserialize_state` resets leaf executor state to
        build-time values for settled payloads, so leaves must be the final writers.
        """
        # Deep-copy at the boundary so a baseline can be replayed across cases without being mutated.
        clone = copy.deepcopy(snapshot)
        for path, orch_state in clone.orchestrators.items():
            orch = self._orch_index.get(path)
            if orch is None:
                logger.warning("path=<%s> | orchestrator path missing at restore, skipping", "/".join(path))
                continue
            orch.deserialize_state(orch_state)
            self._force_fresh_invoke_if_settled(orch, orch_state)
        for path, agent_snap in clone.agents.items():
            agent = self._agent_index.get(path)
            if agent is None:
                logger.warning("path=<%s> | agent path missing at restore, skipping", "/".join(path))
                continue
            agent.load_snapshot(agent_snap)

    @staticmethod
    def _force_fresh_invoke_if_settled(orch: MultiAgentBase, orch_state: dict[str, Any]) -> None:
        """Clear `_resume_from_session` for a settled-status payload.

        Required for `Swarm`: its `deserialize_state` always takes the resume branch, leaving
        `_resume_from_session=True` with `current_node=None`, which crashes the next invoke.
        """
        status = orch_state.get("status")
        if status not in {Status.PENDING.value, Status.COMPLETED.value, Status.FAILED.value}:
            return
        if hasattr(orch, "_resume_from_session"):
            orch._resume_from_session = False
            return
        logger.warning(
            "orchestrator=<%s> | settled-status payload but no `_resume_from_session` attribute; "
            "set `_resume_from_session = False` on custom MultiAgentBase subclasses to opt out",
            type(orch).__name__,
        )


def _multi_agent_result_text(result: Any) -> str:
    """Flatten a `MultiAgentResult` into a string by joining each `AgentResult`."""
    results_dict = getattr(result, "results", None)
    if isinstance(results_dict, dict):
        parts: list[str] = []
        for node_result in results_dict.values():
            get_results = getattr(node_result, "get_agent_results", None)
            if callable(get_results):
                parts.extend(str(r) for r in get_results())
        if parts:
            return "\n".join(parts)
    return str(result)


__all__ = [
    "MALFORMED_TOOL_NAME",
    "StrandsAgentSession",
    "StrandsMultiAgentSession",
    "TargetCheckpoint",
    "TargetSession",
    "ToolUseEntry",
]
