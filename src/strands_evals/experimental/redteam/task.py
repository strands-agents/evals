"""Task builder for red team experiments."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from strands import Agent, Snapshot
from strands.models.model import Model
from strands.multiagent.base import MultiAgentBase

from .case import RedTeamCase
from .strategies import AttackStrategy
from .strategies.target_session import StrandsAgentSession, StrandsMultiAgentSession, TargetSession

logger = logging.getLogger(__name__)

MAX_ALLOWED_TURNS = 50


def _build_attacker_task(
    agent: Agent | MultiAgentBase | TargetSession,
    by_label: dict[str, AttackStrategy],
    *,
    model: Model | str | None = None,
    run_meta: dict[str, dict[str, Any]] | None = None,
) -> Callable[[RedTeamCase], dict]:
    """Build a red team task function for `Experiment.run_evaluations`.

    Internal helper used by :class:`RedTeamExperiment`. Returns a `task(case)
    -> {"output": conversation, "trajectory": tool_uses}` that looks up the
    case's strategy (by `metadata["strategy"]`) and delegates the multi-turn
    loop to `strategy.run_attack`, injecting a `TargetSession` that handles
    target invocation, tool-trace capture, and per-case isolation. An `Agent`
    is wrapped in a `StrandsAgentSession`; a `MultiAgentBase` (Graph, Swarm,
    nested orchestrator) is wrapped in a `StrandsMultiAgentSession`; a
    `TargetSession` is used as-is. Either way the session is reset between
    cases.

    Each strategy owns its own turn budget; `MAX_ALLOWED_TURNS` is passed as a
    hard ceiling so no strategy can run unbounded.

    The strategy's run metadata (turns_used, backtracks, ...) is recorded into
    `run_meta` keyed by case name; the experiment owns that dict and joins it
    onto the report (the base `Experiment` copies `metadata` into a fresh
    `EvaluationData`, so the strategy can't reach the report through it).
    """

    # Capture the target's clean state ONCE, before the first case, so every case
    # resets to the same as-constructed baseline. Must be here (build time), not in
    # the session __init__: the target is shared and reused, so by the time case N's
    # session is built the target already carries case N-1's conversation --
    # snapshotting then would bake a dirty baseline. Only an Agent or
    # MultiAgentBase is rewindable this way; a passed-in TargetSession owns its
    # own reset.
    initial_snapshot: Any
    if isinstance(agent, Agent):
        initial_snapshot = agent.take_snapshot(preset="session")
    elif isinstance(agent, MultiAgentBase):
        # The composite snapshot shape is internal to StrandsMultiAgentSession;
        # capture it through the session's public snapshot() so this layer never
        # has to import _MultiAgentSnapshot. The throwaway session shares the
        # same indexes the per-case session will rebuild, so the baseline is
        # apples-to-apples.
        initial_snapshot = StrandsMultiAgentSession(agent).snapshot().agent_snapshot
    else:
        initial_snapshot = None

    def task_fn(case: RedTeamCase) -> dict:
        strategy = _resolve_case_strategy(case, by_label)
        strategy.reset()

        session = _build_session(agent, baseline=initial_snapshot)
        session.reset()  # roll the target back to the clean baseline before this case

        result = strategy.run_attack(case, session, max_turns=MAX_ALLOWED_TURNS, model=model)
        if run_meta is not None and case.name is not None:
            # run stats reach the report via run_meta (see below); pruned_branches
            # rides the same channel so defended-turn evidence survives to display().
            run_meta[case.name] = {**result.metadata, "pruned_branches": result.pruned_branches}
        # Assemble only the keys the base Experiment reads: the strategy's conversation
        # becomes the output, and the trace captured by the session (which the task owns)
        # becomes the trajectory. The strategy's own success/score/metadata are NOT
        # returned here -- the base copies metadata into a fresh EvaluationData and drops
        # the rest, so run stats reach the report via run_meta instead.
        return {
            "output": result.conversation,
            # Copy: a passed-in TargetSession is reused across cases, and the next
            # case's session.reset() clears this same list in place. The base engine
            # holds the trajectory by reference until it model_dumps it, so hand it a
            # snapshot rather than the live trace.
            "trajectory": list(session.trace),
        }

    return task_fn


def _build_session(
    agent: Agent | MultiAgentBase | TargetSession,
    *,
    baseline: Snapshot | Any | None = None,
) -> TargetSession:
    """Wrap an `Agent` or `MultiAgentBase`, or pass a `TargetSession` through.

    Args:
        agent: An `Agent` to wrap in :class:`StrandsAgentSession`, a
            `MultiAgentBase` (Graph, Swarm, ...) to wrap in
            :class:`StrandsMultiAgentSession`, or a ready `TargetSession` to
            use as-is.
        baseline: A clean snapshot the wrapped session resets to between cases.
            For an `Agent` this is a `Snapshot`; for a `MultiAgentBase`
            it's the opaque composite returned by
            :meth:`StrandsMultiAgentSession.snapshot`. Ignored for a passed-in
            `TargetSession`, which owns its own reset.

    Raises:
        TypeError: If `agent` is none of the supported types (e.g. a bare
            callable -- a strategy needs snapshot/restore to manage the target's
            state, which an opaque callable cannot provide).
    """
    if isinstance(agent, Agent):
        return StrandsAgentSession(agent, baseline=baseline)
    if isinstance(agent, MultiAgentBase):
        return StrandsMultiAgentSession(agent, baseline=baseline)
    # Structural (not isinstance) check: TargetSession is a Protocol, and the method
    # set is checked by hand. The `trace` check is separate and load-bearing -- it's
    # the one member the task runner dereferences directly (it becomes the
    # trajectory), and a @runtime_checkable isinstance would only verify method
    # presence, not that `trace` is a list. Without it a session missing `trace`
    # would pass and later die with an AttributeError the experiment swallows into a
    # misleading "defended" verdict.
    has_methods = all(callable(getattr(agent, method, None)) for method in ("invoke", "reset", "snapshot", "restore"))
    if has_methods and isinstance(getattr(agent, "trace", None), list):
        return agent
    raise TypeError(
        f"agent must be a strands.Agent, strands.multiagent.MultiAgentBase, or a TargetSession, "
        f"got {type(agent).__name__!r}; wrap a custom target in a TargetSession so the strategy "
        "can snapshot/restore its state."
    )


def _resolve_case_strategy(case: RedTeamCase, by_label: dict[str, AttackStrategy]) -> AttackStrategy:
    """Look up the strategy instance for a case from its `metadata["strategy"]` label."""
    metadata = case.metadata or {}
    label = metadata.get("strategy")
    if label is None:
        raise ValueError(f"RedTeamCase {case.name!r}: metadata is missing the 'strategy' label.")
    strategy = by_label.get(label)
    if strategy is None:
        raise ValueError(f"RedTeamCase {case.name!r}: no strategy registered for label {label!r}.")
    return strategy
