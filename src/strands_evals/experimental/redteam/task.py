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
    agent: Agent | MultiAgentBase | TargetSession | None,
    by_label: dict[str, AttackStrategy],
    *,
    agent_factory: Callable[[], Agent | MultiAgentBase | TargetSession] | None = None,
    model: Model | str | None = None,
    run_meta: dict[str, dict[str, Any]] | None = None,
    parallel: bool = False,
) -> Callable[[RedTeamCase], dict]:
    """Build a `task(case) -> {"output": conversation, "trajectory": tool_uses}` callable.

    Looks up each case's strategy by `metadata["strategy"]` and delegates the multi-turn loop to
    `strategy.run_attack`, injecting a `TargetSession`. `MAX_ALLOWED_TURNS` is the hard ceiling. Run metadata
    is recorded into `run_meta` keyed by case name.

    Args:
        agent: The shared target for sequential runs. Required when `agent_factory` is None.
        by_label: Strategy registry keyed by `metadata["strategy"]` label.
        agent_factory: Zero-arg callable returning a fresh target for each case. Required for parallel
            runs (`parallel=True`); takes precedence over `agent` when both are set.
        model: Model passed through to `strategy.run_attack` for strategy-internal LLM calls.
        run_meta: Per-case strategy metadata sink, keyed by `case.name`.
        parallel: When True, every case is built from `agent_factory` so concurrent cases never share
            mutable state. When False, all cases share one target and rewind to a once-captured baseline
            between cases.
    """
    if agent is None and agent_factory is None:
        raise ValueError("must provide either `agent` or `agent_factory`.")

    # Per-case construction handles both parallel runs (fresh target per worker) and the case where
    # the user passes a factory without an `agent` to share. Snapshot-based reset only fits when the
    # same target is reused sequentially across cases.
    if parallel or agent_factory is not None:
        return _build_per_case_task_fn(
            agent=agent,
            agent_factory=agent_factory,
            by_label=by_label,
            model=model,
            run_meta=run_meta,
        )

    # Sequential path: `agent` is guaranteed non-None here -- the upfront check rejects (None,
    # None), and `agent_factory is not None` would have routed us to the per-case builder above.
    return _build_shared_target_task_fn(
        agent=agent,  # type: ignore[arg-type]
        by_label=by_label,
        model=model,
        run_meta=run_meta,
    )


def _build_shared_target_task_fn(
    *,
    agent: Agent | MultiAgentBase | TargetSession,
    by_label: dict[str, AttackStrategy],
    model: Model | str | None,
    run_meta: dict[str, dict[str, Any]] | None,
) -> Callable[[RedTeamCase], dict]:
    """Build a task fn that drives one shared target across cases, rewinding to a fixed baseline.

    Captures the clean baseline ONCE, before any case runs, so every per-case reset rolls back to
    the same as-constructed state rather than to case N-1's history.
    """
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


def _build_per_case_task_fn(
    *,
    agent: Agent | MultiAgentBase | TargetSession | None,
    agent_factory: Callable[[], Agent | MultiAgentBase | TargetSession] | None,
    by_label: dict[str, AttackStrategy],
    model: Model | str | None,
    run_meta: dict[str, dict[str, Any]] | None,
) -> Callable[[RedTeamCase], dict]:
    """Build a per-case task fn that constructs its own target every case.

    Resolves the target source once at build time into a single zero-arg `make_target` callable so
    the per-case path has no runtime branch. Callers (parallel runs against a custom
    `TargetSession` with no factory) get a `TypeError` here, before any case runs.
    """
    make_target = _resolve_target_source(agent=agent, agent_factory=agent_factory)

    def task_fn(case: RedTeamCase) -> dict:
        strategy = _resolve_case_strategy(case, by_label)
        strategy.reset()

        # No baseline: each case starts from a freshly built target, and `session.reset()` only
        # needs to clear the per-case trace.
        session = _build_session(make_target(), baseline=None)
        session.reset()

        result = strategy.run_attack(case, session, max_turns=MAX_ALLOWED_TURNS, model=model)
        if run_meta is not None and case.name is not None:
            # CPython dict assignment for a single distinct key is atomic, and case names are unique
            # per cross-product expansion, so concurrent writers never target the same key.
            run_meta[case.name] = {**result.metadata, "pruned_branches": result.pruned_branches}
        return {
            "output": result.conversation,
            "trajectory": list(session.trace),
        }

    return task_fn


def _resolve_target_source(
    *,
    agent: Agent | MultiAgentBase | TargetSession | None,
    agent_factory: Callable[[], Agent | MultiAgentBase | TargetSession] | None,
) -> Callable[[], Agent | MultiAgentBase | TargetSession]:
    """Require an `agent_factory` for the per-case path, rejecting config errors up front.

    Per-case construction always goes through a user-supplied factory. The deep-copy fallback
    against a `strands.Agent` was removed because real Strands agents are not deepcopyable -- the
    default `BedrockModel` carries an httplib pool with thread locks that `copy.deepcopy` chokes
    on. Surfacing the requirement at config time beats crashing mid-run with a `cannot pickle
    '_thread.lock'` deep in `copy.py`.
    """
    if agent_factory is not None:
        return agent_factory
    raise TypeError(
        "Parallel red team runs require `agent_factory=...`. Strands `Agent` and `MultiAgentBase` "
        "instances cannot be deep-copied per worker (their default model client holds non-pickleable "
        f"state); got agent={type(agent).__name__!r}, agent_factory=None. Pass a zero-arg factory "
        "that returns a fresh target for each case."
    )


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
