"""Task builder for red team experiments."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from strands import Agent
from strands.models.model import Model

from .case import RedTeamCase
from .strategies import AttackStrategy
from .strategies.target_session import AgentTargetSession, CallableTargetSession

logger = logging.getLogger(__name__)

MAX_ALLOWED_TURNS = 50


def _build_attacker_task(
    agent: Agent | Callable[[str], Any],
    by_label: dict[str, AttackStrategy],
    *,
    model: Model | str | None = None,
    run_meta: dict[str, dict[str, Any]] | None = None,
) -> Callable[[RedTeamCase], dict]:
    """Build a red team task function for ``Experiment.run_evaluations``.

    Internal helper used by :class:`RedTeamExperiment`. Returns a ``task(case)
    -> {"output": conversation, "trajectory": tool_uses}`` that looks up the
    case's strategy (by ``metadata["strategy"]``) and delegates the multi-turn
    loop to ``strategy.run_attack``, injecting a ``TargetSession`` that handles
    target invocation, tool-trace capture, and per-case isolation. When
    ``agent`` is an ``Agent``, its message history is reset between cases.

    Each strategy owns its own turn budget; ``MAX_ALLOWED_TURNS`` is passed as a
    hard ceiling so no strategy can run unbounded.

    The strategy's run metadata (turns_used, backtracks, ...) is recorded into
    ``run_meta`` keyed by case name; the experiment owns that dict and joins it
    onto the report (the base ``Experiment`` copies ``metadata`` into a fresh
    ``EvaluationData``, so the strategy can't reach the report through it).
    """

    def task_fn(case: RedTeamCase) -> dict:
        strategy = _resolve_case_strategy(case, by_label)
        strategy.reset()

        session = AgentTargetSession(agent) if isinstance(agent, Agent) else CallableTargetSession(agent)
        session.reset()  # clear any prior-case state (e.g. a reused Agent's message history)

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
            "trajectory": session.trace,
        }

    return task_fn


def _resolve_case_strategy(case: RedTeamCase, by_label: dict[str, AttackStrategy]) -> AttackStrategy:
    """Look up the strategy instance for a case from its ``metadata["strategy"]`` label."""
    metadata = case.metadata or {}
    label = metadata.get("strategy")
    if label is None:
        raise ValueError(f"RedTeamCase {case.name!r}: metadata is missing the 'strategy' label.")
    strategy = by_label.get(label)
    if strategy is None:
        raise ValueError(f"RedTeamCase {case.name!r}: no strategy registered for label {label!r}.")
    return strategy
