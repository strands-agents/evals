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

import logging
from collections.abc import Callable
from typing import Any, Protocol

from strands import Agent, Snapshot

logger = logging.getLogger(__name__)


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
    def trace(self) -> list[dict]:
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

    def snapshot(self) -> Snapshot:
        """Capture the target's current state for a later :meth:`restore`.

        Returns:
            A snapshot of the target's conversation/state.

        Raises:
            NotImplementedError: If this session is not rewindable
                (:attr:`supports_rewind` is ``False``).
        """
        ...

    def restore(self, snapshot: Snapshot) -> None:
        """Roll the target back to a previously captured ``snapshot``.

        Also rolls :attr:`trace` back to its length at snapshot time, so tool
        uses from rolled-back turns do not linger in the trajectory.

        Args:
            snapshot: A snapshot previously returned by :meth:`snapshot`.

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
        self._trace: list[dict] = []

    @property
    def supports_rewind(self) -> bool:
        return True

    @property
    def trace(self) -> list[dict]:
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

    def snapshot(self) -> Snapshot:
        snap = self._agent.take_snapshot(preset="session")
        # Stash the trace length alongside the SDK snapshot so restore() can trim
        # tool uses recorded after this point (the SDK snapshot covers agent
        # messages but not our separate trace list).
        snap.app_data["_trace_len"] = len(self._trace)
        return snap

    def restore(self, snapshot: Snapshot) -> None:
        self._agent.load_snapshot(snapshot)
        trace_len = snapshot.app_data.get("_trace_len")
        if trace_len is not None:
            del self._trace[trace_len:]


class CallableTargetSession:
    """A :class:`TargetSession` backed by an opaque ``Callable[[str], Any]``.

    Not rewindable: the callable's internal state is invisible to us, so
    :meth:`snapshot` / :meth:`restore` raise. A backtracking strategy guards on
    :attr:`supports_rewind` and degrades to report-scope backtracking (dropping
    the refused turn from the reported conversation) for these targets.

    The callable may return either a plain string or a
    ``{"output": str, "trace": list[dict]}`` mapping; in the latter case the
    trace entries are merged into :attr:`trace`.
    """

    def __init__(self, target: Callable[[str], Any]) -> None:
        """Initialize the session.

        Args:
            target: The callable target mapping a message to a response.
        """
        self._target = target
        self._trace: list[dict] = []

    @property
    def supports_rewind(self) -> bool:
        return False

    @property
    def trace(self) -> list[dict]:
        return self._trace

    def invoke(self, message: str) -> str:
        raw = self._target(message)
        if isinstance(raw, dict):
            if "trace" in raw:
                self._trace.extend(raw["trace"])
            return str(raw.get("output", ""))
        return str(raw)

    def snapshot(self) -> Snapshot:
        raise NotImplementedError("callable target is not rewindable; guard on supports_rewind")

    def restore(self, snapshot: Snapshot) -> None:
        raise NotImplementedError("callable target is not rewindable; guard on supports_rewind")


__all__ = ["AgentTargetSession", "CallableTargetSession", "TargetSession"]
