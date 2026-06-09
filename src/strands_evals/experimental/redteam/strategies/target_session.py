"""Target session protocol and implementations.

A :class:`TargetSession` is the handle a strategy uses to talk to the system
under test. It replaces the older opaque ``call_target: Callable[[str], str]``:
besides sending a message, a session exposes snapshot/restore so a strategy can
roll the target back to an earlier state (e.g. Crescendo backtracking past a
refusal).

The task runner builds a :class:`StrandsAgentSession` (which wraps a
``strands.Agent`` and is rewindable via the SDK snapshot API) from the agent the
experiment was given. A future multi-agent session would be a second
implementation; because the contract is a ``Protocol``, a customer can also
supply their own without subclassing anything here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from strands import Agent, Snapshot
from strands.types.content import Message

logger = logging.getLogger(__name__)


# Placeholder name for a tool-use block whose shape we couldn't parse. Distinct
# from a real (empty-named) tool call so the evaluator/report can tell corruption
# from a genuine no-name tool use, while the entry still counts toward trace length.
MALFORMED_TOOL_NAME = "<malformed>"


class ToolUseEntry(TypedDict):
    """A single tool invocation captured into a session's trace.

    The shape every session implementation produces and every consumer (the
    report, the ``AttackSuccessEvaluator``) reads, declared once here. Named
    ``ToolUseEntry`` rather than ``ToolUse`` to avoid colliding with the SDK's
    ``strands.types.tools.ToolUse`` content block, which this is derived from but
    is not the same shape.
    """

    name: str
    input: dict[str, Any]


def _tool_uses_in(messages: list[Message]) -> list[ToolUseEntry]:
    """Extract tool-use entries from Strands messages, tolerating schema drift.

    The single place that knows the Strands ``message -> content[] -> toolUse``
    schema, so the brittle dict-walking lives in exactly one tested function. It
    never raises on a malformed shape: a non-mapping message or content block is
    skipped, and a ``toolUse`` block whose value isn't a mapping is recorded as a
    ``MALFORMED_TOOL_NAME`` placeholder. Two reasons not to abort:

    - One malformed turn must not discard the whole case's trace. ``invoke`` runs
      inside the experiment's per-case ``try/except``, which would turn a raise
      into ``score=0`` -- silently mislabeling a possible breach as defended.
    - The trace length must stay honest. A backtracking strategy decides whether a
      turn drove a tool call by whether the trace grew, so a real-but-malformed
      tool block must still grow the trace (hence the placeholder) or the breach
      could be backtracked away.

    A ``logger.warning`` surfaces the drift without making it case-fatal.

    Args:
        messages: Strands messages to scan (typically the tail appended by one
            ``Agent`` call).

    Returns:
        One :class:`ToolUseEntry` per ``toolUse`` block found, in order; a block
        with a non-mapping ``toolUse`` yields a ``MALFORMED_TOOL_NAME`` placeholder.
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
    """An opaque rewind point returned by :meth:`TargetSession.snapshot`.

    Bundles the target's saved state with the session's trace length at capture
    time, so :meth:`TargetSession.restore` can roll back both the target's
    conversation and the tool trace together. Strategies treat it as opaque —
    take one, pass it back to ``restore``.

    ``agent_snapshot`` is deliberately typed ``Any``: each session decides what it
    stores there (a single ``strands.Snapshot`` for :class:`StrandsAgentSession`, a
    composite of per-agent snapshots + orchestrator state for a future multi-agent
    session). The checkpoint is opaque to strategies either way.

    Restore is sound for the snapshot/restore-the-immediately-preceding-turn
    pattern Crescendo uses (one live checkpoint at a time). Non-monotonic restores
    across multiple outstanding checkpoints (e.g. a future PAIR/TAP tree search)
    would need the checkpoint to carry the trace slice rather than just its length;
    revisit ``trace_len`` then.

    Attributes:
        agent_snapshot: The session's saved target state, opaque to strategies.
        trace_len: ``len(session.trace)`` when the checkpoint was taken.
    """

    agent_snapshot: Any
    trace_len: int


class TargetSession(Protocol):
    """Contract a strategy uses to interact with the target under test.

    A strategy receives a ``TargetSession`` in ``run_attack`` and drives the
    conversation through :meth:`invoke`. Strategies that backtrack (e.g.
    Crescendo) use :meth:`snapshot` / :meth:`restore` to roll the target back;
    strategies that only converse forward (e.g. GOAT, Bad Likert Judge) use
    :meth:`invoke` alone.
    """

    trace: list[ToolUseEntry]
    """Tool-use entries captured across this session's :meth:`invoke` calls."""

    def invoke(self, message: str) -> str:
        """Send ``message`` to the target and return its text response.

        Appends any tool uses observed during the call to :attr:`trace`.

        Args:
            message: The attacker message to send.

        Returns:
            The target's response as text.
        """
        ...

    def reset(self) -> None:
        """Clear per-case state so the session starts a fresh conversation."""
        ...

    def snapshot(self) -> TargetCheckpoint:
        """Capture the target's current state for a later :meth:`restore`.

        Returns:
            An opaque checkpoint to pass back to :meth:`restore`.
        """
        ...

    def restore(self, checkpoint: TargetCheckpoint) -> None:
        """Roll the target back to a previously captured ``checkpoint``.

        Also rolls :attr:`trace` back to its length at checkpoint time, so tool
        uses from rolled-back turns do not linger in the trajectory. The session
        owns this truncation; callers pass a checkpoint and never edit
        :attr:`trace` directly.

        Args:
            checkpoint: A checkpoint previously returned by :meth:`snapshot`.
        """
        ...


class StrandsAgentSession:
    """A :class:`TargetSession` backed by a ``strands.Agent``.

    Rewindable: :meth:`snapshot` / :meth:`restore` delegate to the SDK's
    ``Agent.take_snapshot`` / ``Agent.load_snapshot``, which restore the agent's
    message history to its snapshot-time state; :meth:`restore` additionally
    truncates :attr:`trace` back to its snapshot-time length.
    """

    def __init__(self, agent: Agent, *, baseline: Snapshot | None = None) -> None:
        """Initialize the session.

        Args:
            agent: The target agent to drive and snapshot.
            baseline: A clean snapshot of the agent to roll back to in :meth:`reset`.
                The task runner captures this once, before the first case, so every
                case starts from the same as-constructed target state (system prompt
                plus any seeded history). When ``None`` (e.g. a directly-constructed
                session), :meth:`reset` falls back to clearing messages only.
        """
        self._agent = agent
        self._baseline = baseline
        self.trace: list[ToolUseEntry] = []

    def invoke(self, message: str) -> str:
        result, new_messages = self._send(message)
        self.trace.extend(_tool_uses_in(new_messages))
        return str(result)

    def _send(self, message: str) -> tuple[Any, list[Message]]:
        """Send ``message``; return the agent's result and the messages it appended."""
        messages_before = len(self._agent.messages)
        result = self._agent(message)
        return result, self._agent.messages[messages_before:]

    def reset(self) -> None:
        # Roll back through the same load_snapshot path restore() uses, so reset
        # covers every field snapshot() captures (messages, state,
        # conversation_manager_state, interrupt_state) rather than just messages --
        # otherwise stale agent state leaks across cases. Falls back to clearing
        # messages when no baseline was supplied.
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


__all__ = ["MALFORMED_TOOL_NAME", "StrandsAgentSession", "TargetCheckpoint", "TargetSession", "ToolUseEntry"]
