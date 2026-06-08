"""Target session protocol and implementations.

A :class:`TargetSession` is the handle a strategy uses to talk to the system
under test. It replaces the older opaque ``call_target: Callable[[str], str]``:
besides sending a message, a session exposes snapshot/restore so a strategy can
roll the target back to an earlier state (e.g. Crescendo backtracking past a
refusal). Targets that cannot expose their internal state (a plain callable)
report ``supports_rewind = False`` and a strategy degrades gracefully.

Two concrete sessions are built by the task runner from what the experiment was
given: :class:`AgentTargetSession` wraps a ``strands.Agent`` (rewindable via the
SDK snapshot API), and :class:`CallableTargetSession` wraps an opaque
``Callable[[str], Any]`` (not rewindable). A future multi-agent session would be
a third implementation; because the contract is a ``Protocol``, a customer can
also supply their own without subclassing anything here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from strands import Agent, Snapshot

logger = logging.getLogger(__name__)


@dataclass
class TargetCheckpoint:
    """An opaque rewind point returned by :meth:`TargetSession.snapshot`.

    Bundles the SDK agent snapshot with the session's trace length at capture
    time, so :meth:`TargetSession.restore` can roll back both the target's
    conversation and the tool trace together. Strategies treat it as opaque —
    take one, pass it back to ``restore``.

    Restore is sound for the snapshot/restore-the-immediately-preceding-turn
    pattern Crescendo uses (one live checkpoint at a time). Non-monotonic restores
    across multiple outstanding checkpoints (e.g. a future PAIR/TAP tree search)
    would need the checkpoint to carry the trace slice rather than just its length;
    revisit ``trace_len`` then.

    Attributes:
        agent_snapshot: The SDK ``Snapshot`` of the target agent's state.
        trace_len: ``len(session.trace)`` when the checkpoint was taken.
    """

    agent_snapshot: Snapshot
    trace_len: int


class TargetSession(Protocol):
    """Contract a strategy uses to interact with the target under test.

    A strategy receives a ``TargetSession`` in ``run_attack`` and drives the
    conversation through :meth:`invoke`. Strategies that backtrack (e.g.
    Crescendo) guard on :attr:`supports_rewind` and use :meth:`snapshot` /
    :meth:`restore` to roll the target back; strategies that only converse
    forward (e.g. GOAT, Bad Likert Judge) use :meth:`invoke` alone.
    """

    @property
    def supports_rewind(self) -> bool:
        """Whether this session can snapshot and restore its target's state.

        ``True`` for sessions backed by a ``strands.Agent`` (whose history we
        own); ``False`` for opaque callables, whose internal state we cannot
        see. A strategy must check this before calling :meth:`snapshot` /
        :meth:`restore`.
        """
        ...

    @property
    def trace(self) -> list[dict[str, Any]]:
        """Tool-use entries captured across this session's :meth:`invoke` calls."""
        ...

    def invoke(self, message: str) -> str:
        """Send ``message`` to the target and return its text response.

        Appends any tool uses observed during the call to :attr:`trace`.

        Args:
            message: The attacker message to send.

        Returns:
            The target's response as text.
        """
        ...

    def trim_trace(self, length: int) -> None:
        """Drop trace entries past ``length``, keeping the first ``length``.

        Lets a strategy roll the tool trace back to a known point when it drops a
        turn from the scored conversation (e.g. a non-rewindable backtrack), so the
        dropped turn's tool uses do not linger in the trajectory. The session owns
        the truncation; callers do not mutate :attr:`trace` directly. A no-op when
        ``length`` is at or beyond the current trace length.

        Args:
            length: The trace length to keep.
        """
        ...

    def reset(self) -> None:
        """Clear per-case state so the session starts a fresh conversation."""
        ...

    def snapshot(self) -> TargetCheckpoint:
        """Capture the target's current state for a later :meth:`restore`.

        Returns:
            An opaque checkpoint to pass back to :meth:`restore`.

        Raises:
            NotImplementedError: If this session is not rewindable
                (:attr:`supports_rewind` is ``False``).
        """
        ...

    def restore(self, checkpoint: TargetCheckpoint) -> None:
        """Roll the target back to a previously captured ``checkpoint``.

        Also rolls :attr:`trace` back to its length at checkpoint time, so tool
        uses from rolled-back turns do not linger in the trajectory.

        Args:
            checkpoint: A checkpoint previously returned by :meth:`snapshot`.

        Raises:
            NotImplementedError: If this session is not rewindable
                (:attr:`supports_rewind` is ``False``).
        """
        ...


class AgentTargetSession:
    """A :class:`TargetSession` backed by a ``strands.Agent``.

    Rewindable: :meth:`snapshot` / :meth:`restore` delegate to the SDK's
    ``Agent.take_snapshot`` / ``Agent.load_snapshot`` (which deep-copy the
    agent's messages), and :meth:`restore` additionally truncates :attr:`trace`
    back to its snapshot-time length.
    """

    def __init__(self, agent: Agent) -> None:
        """Initialize the session.

        Args:
            agent: The target agent to drive and snapshot.
        """
        self._agent = agent
        self._trace: list[dict[str, Any]] = []

    @property
    def supports_rewind(self) -> bool:
        return True

    @property
    def trace(self) -> list[dict[str, Any]]:
        return self._trace

    def invoke(self, message: str) -> str:
        messages_before = len(self._agent.messages)
        result = self._agent(message)

        try:
            for msg in self._agent.messages[messages_before:]:
                for block in msg.get("content", []):
                    if "toolUse" in block:
                        tool_use = block["toolUse"]
                        self._trace.append(
                            {
                                "name": tool_use.get("name", ""),
                                "input": tool_use.get("input", {}),
                            }
                        )
        except (AttributeError, KeyError, TypeError) as e:
            logger.debug("error=<%s> | failed to extract tool trace", e)

        return str(result)

    def trim_trace(self, length: int) -> None:
        del self._trace[length:]

    def reset(self) -> None:
        self._agent.messages.clear()
        self._trace.clear()

    def snapshot(self) -> TargetCheckpoint:
        return TargetCheckpoint(
            agent_snapshot=self._agent.take_snapshot(preset="session"),
            trace_len=len(self._trace),
        )

    def restore(self, checkpoint: TargetCheckpoint) -> None:
        self._agent.load_snapshot(checkpoint.agent_snapshot)
        self.trim_trace(checkpoint.trace_len)


class CallableTargetSession:
    """A :class:`TargetSession` backed by an opaque ``Callable[[str], Any]``.

    Not rewindable: the callable's internal state is invisible to us, so
    :meth:`snapshot` / :meth:`restore` raise. A backtracking strategy guards on
    :attr:`supports_rewind` and degrades to report-scope backtracking (dropping
    the refused turn from the reported conversation) for these targets.

    The callable may return either a plain string or a
    ``{"output": str, "trace": list[dict[str, Any]]}`` mapping; in the latter case the
    trace entries are merged into :attr:`trace`.
    """

    def __init__(self, target: Callable[[str], Any]) -> None:
        """Initialize the session.

        Args:
            target: The callable target mapping a message to a response.
        """
        self._target = target
        self._trace: list[dict[str, Any]] = []

    @property
    def supports_rewind(self) -> bool:
        return False

    @property
    def trace(self) -> list[dict[str, Any]]:
        return self._trace

    def invoke(self, message: str) -> str:
        raw = self._target(message)
        if isinstance(raw, dict):
            if "trace" in raw:
                self._trace.extend(raw["trace"])
            return str(raw.get("output", ""))
        return str(raw)

    def trim_trace(self, length: int) -> None:
        del self._trace[length:]

    def reset(self) -> None:
        self._trace.clear()

    def snapshot(self) -> TargetCheckpoint:
        raise NotImplementedError("callable target is not rewindable; guard on supports_rewind")

    def restore(self, checkpoint: TargetCheckpoint) -> None:
        raise NotImplementedError("callable target is not rewindable; guard on supports_rewind")


__all__ = ["AgentTargetSession", "CallableTargetSession", "TargetCheckpoint", "TargetSession"]
