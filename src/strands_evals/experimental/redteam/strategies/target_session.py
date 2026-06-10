"""Target session protocol and implementations.

A :class:`TargetSession` is the handle a strategy uses to talk to the system
under test. It replaces the older opaque `call_target: Callable[[str], str]`:
besides sending a message, a session exposes snapshot/restore so a strategy can
roll the target back to an earlier state (e.g. Crescendo backtracking past a
refusal).

The task runner builds a :class:`StrandsAgentSession` (which wraps a
`strands.Agent` and is rewindable via the SDK snapshot API) from the agent
the experiment was given, or a :class:`StrandsMultiAgentSession` (which wraps a
`strands.multiagent.MultiAgentBase` such as a Graph or Swarm). Because the
contract is a `Protocol`, a customer can also supply their own without
subclassing anything here.
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


# Placeholder name for a tool-use block whose shape we couldn't parse. Distinct
# from a real (empty-named) tool call so the evaluator/report can tell corruption
# from a genuine no-name tool use, while the entry still counts toward trace length.
MALFORMED_TOOL_NAME = "<malformed>"


class ToolUseEntry(TypedDict):
    """A single tool invocation captured into a session's trace.

    The shape every session implementation produces and every consumer (the
    report, the `AttackSuccessEvaluator`) reads, declared once here. Named
    `ToolUseEntry` rather than `ToolUse` to avoid colliding with the SDK's
    `strands.types.tools.ToolUse` content block, which this is derived from but
    is not the same shape.
    """

    name: str
    input: dict[str, Any]


def _tool_uses_in(messages: list[Message]) -> list[ToolUseEntry]:
    """Extract tool-use entries from Strands messages, tolerating schema drift.

    The single place that knows the Strands `message -> content[] -> toolUse`
    schema, so the brittle dict-walking lives in exactly one tested function. It
    never raises on a malformed shape: a non-mapping message or content block is
    skipped, and a `toolUse` block whose value isn't a mapping is recorded as a
    `MALFORMED_TOOL_NAME` placeholder. Two reasons not to abort:

    - One malformed turn must not discard the whole case's trace. `invoke` runs
      inside the experiment's per-case `try/except`, which would turn a raise
      into `score=0` -- silently mislabeling a possible breach as defended.
    - The trace length must stay honest. A backtracking strategy decides whether a
      turn drove a tool call by whether the trace grew, so a real-but-malformed
      tool block must still grow the trace (hence the placeholder) or the breach
      could be backtracked away.

    A `logger.warning` surfaces the drift without making it case-fatal.

    Args:
        messages: Strands messages to scan (typically the tail appended by one
            `Agent` call).

    Returns:
        One :class:`ToolUseEntry` per `toolUse` block found, in order; a block
        with a non-mapping `toolUse` yields a `MALFORMED_TOOL_NAME` placeholder.
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
    take one, pass it back to `restore`.

    `agent_snapshot` is deliberately typed `Any`: each session decides what it
    stores there (a single `strands.Snapshot` for :class:`StrandsAgentSession`, a
    composite of per-agent snapshots + orchestrator state for a future multi-agent
    session). The checkpoint is opaque to strategies either way.

    Restore is sound for the snapshot/restore-the-immediately-preceding-turn
    pattern Crescendo uses (one live checkpoint at a time). Non-monotonic restores
    across multiple outstanding checkpoints (e.g. a future PAIR/TAP tree search)
    would need the checkpoint to carry the trace slice rather than just its length;
    revisit `trace_len` then.

    Attributes:
        agent_snapshot: The session's saved target state, opaque to strategies.
        trace_len: `len(session.trace)` when the checkpoint was taken.
    """

    agent_snapshot: Any
    trace_len: int


class TargetSession(Protocol):
    """Contract a strategy uses to interact with the target under test.

    A strategy receives a `TargetSession` in `run_attack` and drives the
    conversation through :meth:`invoke`. Strategies that backtrack (e.g.
    Crescendo) use :meth:`snapshot` / :meth:`restore` to roll the target back;
    strategies that only converse forward (e.g. GOAT, Bad Likert Judge) use
    :meth:`invoke` alone.
    """

    trace: list[ToolUseEntry]
    """Tool-use entries captured across this session's :meth:`invoke` calls."""

    def invoke(self, message: str) -> str:
        """Send `message` to the target and return its text response.

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
        """Roll the target back to a previously captured `checkpoint`.

        Also rolls :attr:`trace` back to its length at checkpoint time, so tool
        uses from rolled-back turns do not linger in the trajectory. The session
        owns this truncation; callers pass a checkpoint and never edit
        :attr:`trace` directly.

        Args:
            checkpoint: A checkpoint previously returned by :meth:`snapshot`.
        """
        ...


class StrandsAgentSession:
    """A :class:`TargetSession` backed by a `strands.Agent`.

    Rewindable: :meth:`snapshot` / :meth:`restore` delegate to the SDK's
    `Agent.take_snapshot` / `Agent.load_snapshot`, which restore the agent's
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
                plus any seeded history). When `None` (e.g. a directly-constructed
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
        """Send `message`; return the agent's result and the messages it appended."""
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


# Path of node ids from the root orchestrator to a leaf Agent. Tuple (not str) so
# the same node id repeated under different parents stays distinguishable.
_AgentPath = tuple[str, ...]


@dataclass
class _MultiAgentSnapshot:
    """Composite snapshot for a multi-agent target.

    `agents` holds one `Snapshot` per leaf `Agent` keyed by its path in the
    tree; `orchestrators` holds one `serialize_state` dict per
    `MultiAgentBase` (root and every nested orchestrator). Stored opaquely
    inside :class:`TargetCheckpoint.agent_snapshot`; strategies never inspect it.
    """

    agents: dict[_AgentPath, Snapshot]
    orchestrators: dict[_AgentPath, dict[str, Any]]


class StrandsMultiAgentSession:
    """A :class:`TargetSession` backed by a `strands.multiagent.MultiAgentBase`.

    Wraps a Graph, Swarm, or any other `MultiAgentBase` (including nested
    orchestrators). At init we walk the tree once to build a path index of every
    leaf `Agent`; :meth:`snapshot` then captures one `Agent.take_snapshot`
    per leaf plus a `serialize_state` dict per orchestrator, and :meth:`restore`
    pushes them back through `Agent.load_snapshot` and
    `MultiAgentBase.deserialize_state` respectively. The composite is opaque to
    strategies (it lives in :attr:`TargetCheckpoint.agent_snapshot`).

    Trace capture diffs each leaf agent's `messages` tail across an
    :meth:`invoke` call -- the same approach :class:`StrandsAgentSession` uses,
    extended to every leaf so tool uses anywhere in the tree are recorded.
    """

    def __init__(self, root: MultiAgentBase, *, baseline: _MultiAgentSnapshot | None = None) -> None:
        """Initialize the session.

        Args:
            root: The target orchestrator. Its tree is walked once and indexed;
                topology must not change after construction (Strands graphs and
                swarms are static after build, so this is the normal case).
            baseline: A clean composite snapshot to roll back to in :meth:`reset`.
                The task runner captures this once, before the first case, so
                every case starts from the same as-constructed target state.
                When `None`, :meth:`reset` falls back to clearing each leaf
                agent's `messages` (matching :class:`StrandsAgentSession`).
        """
        self._root = root
        self._agent_index: dict[_AgentPath, Agent] = {}
        self._orch_index: dict[_AgentPath, MultiAgentBase] = {}
        self._index_tree(root, ())
        self._baseline = baseline
        self.trace: list[ToolUseEntry] = []

    def _index_tree(self, orch: MultiAgentBase, path: _AgentPath) -> None:
        """Walk the orchestrator tree once and populate the path indexes.

        Recurses into nested `MultiAgentBase` executors so a Graph-of-Graphs or
        a Graph containing a Swarm is fully covered. Non-Agent, non-MultiAgentBase
        executors (custom `AgentBase` subclasses without `take_snapshot`) are
        skipped silently with a warning -- their state cannot round-trip through
        the SDK snapshot API and including them would mislead a backtracking
        strategy into thinking it captured the whole tree.
        """
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
        """Send `message` to the root and capture tool uses from every leaf.

        Diffs each indexed agent's `messages` tail before/after the call, in
        the deterministic order leaves were registered during :meth:`_index_tree`
        (insertion order of each `orch.nodes` dict, recursed depth-first). A
        single agent invoked multiple times during one orchestrator call (e.g. a
        revisited graph node) contributes every new message in order. The exact
        across-leaves order is implementation-defined when leaves run
        concurrently, but each leaf's own tool-use order is preserved.

        Out of scope: a single `Agent` instance reused as the executor for
        multiple distinct node paths. The diff is keyed by path, so a shared
        instance's tail would be scanned once per path and its tool uses
        double-counted. `Graph` and `Swarm` give each node its own executor, so
        this only affects hand-built `MultiAgentBase` subclasses that
        deliberately share instances; rebuild distinct executors per node if
        accurate trace counts matter there.
        """
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
            # No baseline: best-effort clear of every leaf's messages, mirroring
            # StrandsAgentSession's no-baseline fallback. Orchestrator bookkeeping
            # (completed_nodes, results, ...) is left to the SDK to refresh on
            # the next invoke; without a baseline we don't have a serialize_state
            # payload to restore from.
            for agent in self._agent_index.values():
                agent.messages.clear()
        self.trace.clear()

    def snapshot(self) -> TargetCheckpoint:
        """Capture a composite snapshot of every leaf agent and every orchestrator."""
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

        Deep-copies the composite once at the boundary so a stored or baseline
        snapshot can be replayed across cases without being mutated by the load.
        Cloning the whole composite once shares the deepcopy cost across leaves
        and orchestrators rather than copying piecewise.

        Orchestrators are restored FIRST, leaves LAST. `Graph.deserialize_state`
        and `Swarm.deserialize_state` reset every node's executor state to
        graph-build-time values when the payload has no ``next_nodes_to_execute``
        (the common case between attack turns: the orchestrator is PENDING or
        COMPLETED). Running the orchestrator load AFTER the leaf loads would wipe
        the leaves we just restored back to build-time, silently breaking
        backtracking. Doing orchestrators first lets the per-leaf snapshots be
        the final writers; for interrupted payloads the order is irrelevant
        because `deserialize_state` does not touch leaves.

        After each `deserialize_state`, settled-status payloads get the
        orchestrator's resume bookkeeping forced back to a fresh-invoke state.
        Settled covers PENDING/COMPLETED/FAILED; in practice you'll see
        COMPLETED between attack turns and FAILED on a target that errored
        mid-attack, with PENDING included for completeness (a snapshot of a
        never-invoked orchestrator). `Swarm.deserialize_state` always takes
        the resume branch because `Swarm.serialize_state` always emits the
        `next_nodes_to_execute` key (empty list for a settled swarm) and the
        deserialize side checks key presence, not truthiness; without this
        forcing, `_resume_from_session` stays True with `current_node=None`
        and the next invoke crashes with `AttributeError`. The same forcing
        is a no-op-or-better for `Graph` (which already takes the reset branch
        for empty next-nodes) and any other `MultiAgentBase`.
        """
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
                # Topology shifted out from under us (caller mutated the tree
                # post-construction). Drop the orphaned entry; the user should
                # rebuild the session if they reshape the target.
                logger.warning("path=<%s> | agent path missing at restore, skipping", "/".join(path))
                continue
            agent.load_snapshot(agent_snap)

    @staticmethod
    def _force_fresh_invoke_if_settled(orch: MultiAgentBase, orch_state: dict[str, Any]) -> None:
        """Clear `_resume_from_session` for a settled-status payload.

        Settled = PENDING/COMPLETED/FAILED in the just-restored payload. For a
        Swarm this is the load-bearing fix (see :meth:`_restore`); for a Graph
        it's redundant with `deserialize_state`'s own reset path but harmless;
        for any third-party `MultiAgentBase` it provides the same between-turn
        guarantee.

        `_resume_from_session` is a private SDK attribute we deliberately reach
        into. If it's missing on an orchestrator that returned a settled-status
        payload, the SDK has likely renamed/restructured it -- log a warning
        rather than silently no-op'ing, because the original Swarm bug
        (silent score=0 / "defended" mislabels) is exactly what comes back when
        this guard goes stale unnoticed. Custom `MultiAgentBase` subclasses
        without the attribute trigger the same warning once per restore; tag
        them with `_resume_from_session = False` to opt out.
        """
        status = orch_state.get("status")
        if status not in {Status.PENDING.value, Status.COMPLETED.value, Status.FAILED.value}:
            return
        if hasattr(orch, "_resume_from_session"):
            orch._resume_from_session = False
            return
        logger.warning(
            "orchestrator=<%s> | settled-status payload but no `_resume_from_session` attribute; "
            "SDK may have renamed it -- update StrandsMultiAgentSession or strategy backtracks may "
            "silently mislabel cases as defended",
            type(orch).__name__,
        )


def _multi_agent_result_text(result: Any) -> str:
    """Best-effort extract a textual response from a `MultiAgentResult`.

    The orchestrator returns a `MultiAgentResult` (a tree of `NodeResult`s),
    not a single string. Strategies expect text, so we flatten the underlying
    `AgentResult`s in node-result iteration order via `NodeResult.get_agent_results`
    and concatenate their string forms. Falls back to `str(result)` when no
    agent results surface (orchestrator failure, custom `MultiAgentBase`
    returning a different shape) -- we hand the strategy a string rather than
    raise inside the per-case `try/except` and silently mislabel a turn as
    defended.
    """
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
