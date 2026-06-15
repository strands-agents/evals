"""Task builder for red team experiments."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from strands import Agent
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
    """Build a `task(case) -> {"output": conversation, "trajectory": tool_uses}` callable.

    Looks up each case's strategy by `metadata["strategy"]` and delegates the multi-turn loop to
    `strategy.run_attack`, injecting a `TargetSession`. `MAX_ALLOWED_TURNS` is the hard ceiling. Run metadata
    is recorded into `run_meta` keyed by case name.
    """

    # Capture the clean baseline ONCE, before any case runs, so every per-case reset
    # rolls back to the same as-constructed state rather than to case N-1's history.
    initial_snapshot: Any
    if isinstance(agent, Agent):
        initial_snapshot = agent.take_snapshot(preset="session")
    elif isinstance(agent, MultiAgentBase):
        initial_snapshot = StrandsMultiAgentSession(agent).snapshot().agent_snapshot
    else:
        initial_snapshot = None

    def task_fn(case: RedTeamCase) -> dict:
        strategy = _resolve_case_strategy(case, by_label)
        strategy.reset()

        session = _build_session(agent, baseline=initial_snapshot)
        session.reset()

        result = strategy.run_attack(case, session, max_turns=MAX_ALLOWED_TURNS, model=model)
        if run_meta is not None and case.name is not None:
            run_meta[case.name] = {**result.metadata, "pruned_branches": result.pruned_branches}
        return {
            "output": result.conversation,
            # Snapshot of the trace; the next case's session.reset() clears this list in place.
            "trajectory": list(session.trace),
        }

    return task_fn


def _build_session(
    agent: Agent | MultiAgentBase | TargetSession,
    *,
    baseline: Any = None,
) -> TargetSession:
    """Wrap an `Agent` / `MultiAgentBase`, or pass a `TargetSession` through.

    Args:
        agent: The target to wrap, or a ready `TargetSession`.
        baseline: Clean snapshot the wrapped session resets to between cases. Ignored for a passed-in
            `TargetSession`. Typed `Any` because the two session types use different opaque baseline shapes.

    Raises:
        TypeError: If `agent` is not an `Agent`, `MultiAgentBase`, or a structural `TargetSession` (must expose
            `invoke`/`reset`/`snapshot`/`restore` and a `trace: list`).
    """
    if isinstance(agent, Agent):
        return StrandsAgentSession(agent, baseline=baseline)
    if isinstance(agent, MultiAgentBase):
        return StrandsMultiAgentSession(agent, baseline=baseline)
    # Structural check: TargetSession is a Protocol. The `trace: list` check is
    # load-bearing because the task runner dereferences `.trace` directly.
    has_methods = all(callable(getattr(agent, method, None)) for method in ("invoke", "reset", "snapshot", "restore"))
    if has_methods and isinstance(getattr(agent, "trace", None), list):
        return agent
    raise TypeError(
        f"agent must be a strands.Agent, strands.multiagent.MultiAgentBase, or a TargetSession, "
        f"got {type(agent).__name__!r}; wrap a custom target in a TargetSession so the strategy "
        "can snapshot/restore its state."
    )


def _resolve_case_strategy(case: RedTeamCase, by_label: dict[str, AttackStrategy]) -> AttackStrategy:
    """Look up the strategy for `case` from its `metadata["strategy"]` label."""
    metadata = case.metadata or {}
    label = metadata.get("strategy")
    if label is None:
        raise ValueError(f"RedTeamCase {case.name!r}: metadata is missing the 'strategy' label.")
    strategy = by_label.get(label)
    if strategy is None:
        raise ValueError(f"RedTeamCase {case.name!r}: no strategy registered for label {label!r}.")
    return strategy
