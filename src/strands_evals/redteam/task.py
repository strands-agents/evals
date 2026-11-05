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
from .strategies.base import MAX_ALLOWED_TURNS as MAX_ALLOWED_TURNS  # re-exported for the pre-move import path
from .strategies.target_session import StrandsMultiAgentSession, TargetSession, _build_session, as_target_session

logger = logging.getLogger(__name__)


def _build_attacker_task(
    agent: Agent | MultiAgentBase | TargetSession | None,
    *,
    agent_factory: Callable[[], Agent | MultiAgentBase | TargetSession] | None = None,
    model: Model | str | None = None,
    parallel: bool = False,
) -> Callable[[RedTeamCase], dict]:
    """Build a `task(case) -> {"output", "trajectory", "environment_state"}` callable.

    Reads each case's strategy from `case.strategy` (set by `RedTeamExperiment`) and delegates the multi-turn
    loop to `strategy.run_attack`, injecting a `TargetSession`. `MAX_ALLOWED_TURNS` is the hard ceiling.

    The returned dict holds:
        output: The attacker/target conversation.
        trajectory: The target's tool uses.
        environment_state: `[result.to_environment_state()]`, the run stats `RedTeamReport` reads.

    Args:
        agent: The shared target for sequential runs. Required when `agent_factory` is None.
        agent_factory: Zero-arg callable returning a fresh target for each case. Required for parallel
            runs (`parallel=True`); takes precedence over `agent` when both are set.
        model: Model passed through to `strategy.run_attack` for strategy-internal LLM calls.
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
            model=model,
        )

    # Sequential path: `agent` is guaranteed non-None here -- the upfront check rejects (None,
    # None), and `agent_factory is not None` would have routed us to the per-case builder above.
    return _build_shared_target_task_fn(
        agent=agent,  # type: ignore[arg-type]
        model=model,
    )


def _build_shared_target_task_fn(
    *,
    agent: Agent | MultiAgentBase | TargetSession,
    model: Model | str | None,
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
        strategy = case.strategy
        strategy.reset()

        session = _build_session(agent, baseline=initial_snapshot)
        session.reset()

        return _run_attack(strategy, case, session, model=model)

    return task_fn


def _build_per_case_task_fn(
    *,
    agent: Agent | MultiAgentBase | TargetSession | None,
    agent_factory: Callable[[], Agent | MultiAgentBase | TargetSession] | None,
    model: Model | str | None,
) -> Callable[[RedTeamCase], dict]:
    """Build a per-case task fn that constructs its own target every case.

    Resolves the target source once at build time into a single zero-arg `make_target` callable so
    the per-case path has no runtime branch. Callers (parallel runs against a custom
    `TargetSession` with no factory) get a `TypeError` here, before any case runs.
    """
    make_target = _resolve_target_source(agent=agent, agent_factory=agent_factory)

    def task_fn(case: RedTeamCase) -> dict:
        strategy = case.strategy
        strategy.reset()

        # No baseline: each case starts from a freshly built target, and `session.reset()` only
        # needs to clear the per-case trace.
        session = as_target_session(make_target())
        session.reset()

        return _run_attack(strategy, case, session, model=model)

    return task_fn


def _run_attack(
    strategy: AttackStrategy,
    case: RedTeamCase,
    session: TargetSession,
    *,
    model: Model | str | None,
) -> dict:
    """Run one attack and return its conversation, tool uses and run stats.

    Errors propagate: the base `Experiment` retries throttling, records any other failure as an
    error reason (which the report classifies as errored), and skips caching the failed case.
    """
    # Passed explicitly: custom strategies written against the old contract declare `max_turns` without a default.
    result = strategy.run_attack(case, session, max_turns=MAX_ALLOWED_TURNS, model=model)
    return {
        "output": result.conversation,
        # Snapshot of the trace; the next case's session.reset() clears this list in place.
        "trajectory": list(session.trace),
        "environment_state": [result.to_environment_state()],
    }


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
