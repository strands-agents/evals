"""Task builder for red team experiments."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from strands import Agent
from strands.models.model import Model

from ...simulation.actor_simulator import ActorSimulator
from ...types.simulation import ActorProfile
from .case import RedTeamCase
from .strategies import BUILTIN_STRATEGIES, AttackStrategy

logger = logging.getLogger(__name__)

MAX_ALLOWED_TURNS = 50


def _wrap_agent_with_trace(agent: Agent) -> Callable[[str, list[dict] | None], str]:
    """Wrap an Agent as ``(message, trace) -> response``; appends tool uses to ``trace``."""

    def _call(message: str, trace: list[dict] | None = None) -> str:
        messages_before = len(agent.messages)
        result = agent(message)

        if trace is not None:
            try:
                for msg in agent.messages[messages_before:]:
                    for block in msg.get("content", []):
                        if "toolUse" in block:
                            tool_use = block["toolUse"]
                            trace.append(
                                {
                                    "name": tool_use.get("name", ""),
                                    "input": tool_use.get("input", {}),
                                }
                            )
            except (AttributeError, KeyError, TypeError) as e:
                logger.debug("Failed to extract tool trace: %s", e)

        return str(result)

    return _call


def _build_attacker_task(
    target: Agent | Callable[[str], Any],
    *,
    max_turns: int = 10,
    model: Model | str | None = None,
) -> Callable[[RedTeamCase], dict]:
    """Build a multi-turn red team task function for ``Experiment.run_evaluations``.

    Internal helper used by :class:`RedTeamExperiment` when no explicit task is
    supplied. Returns a ``task(case) -> {"output": conversation, "trajectory":
    tool_uses}`` that drives the attacker/target conversation. When ``target``
    is an ``Agent``, its message history is reset between cases for isolation.
    Callable targets are assumed stateless; if your callable wraps a stateful
    object, reset that state inside the callable yourself.
    """
    if max_turns > MAX_ALLOWED_TURNS:
        logger.warning(
            "max_turns=%d exceeds recommended ceiling %d; clamping.",
            max_turns,
            MAX_ALLOWED_TURNS,
        )
        max_turns = MAX_ALLOWED_TURNS

    agent_fn = _wrap_agent_with_trace(target) if isinstance(target, Agent) else None

    def _call_target(message: str, trace: list[dict]) -> str:
        if agent_fn is not None:
            return agent_fn(message, trace)
        raw = target(message)  # type: ignore[operator]
        if isinstance(raw, dict):
            if "trace" in raw:
                trace.extend(raw["trace"])
            return str(raw.get("output", ""))
        return str(raw)

    def task_fn(case: RedTeamCase) -> dict:
        if isinstance(target, Agent):
            target.messages.clear()
        config = case.config
        goal = config.attack_goal
        strategy: AttackStrategy | None = BUILTIN_STRATEGIES.get(config.strategy) if config.strategy else None
        if strategy is not None:
            strategy.reset()
        actor_profile = ActorProfile(
            traits=config.traits,
            context=goal.context,
            actor_goal=goal.actor_goal,
        )
        if not config.system_prompt_template:
            raise ValueError(f"RedTeamCase {case.name!r}: config.system_prompt_template is empty.")
        system_prompt_template = config.system_prompt_template.replace("{max_turns}", str(max_turns))

        initial_query = str(case.input)
        simulator = ActorSimulator(
            actor_profile=actor_profile,
            initial_query=initial_query,
            system_prompt_template=system_prompt_template,
            model=model,  # type: ignore[arg-type]
            max_turns=max_turns,
        )

        trace: list[dict] = []
        conversation: list[dict[str, str]] = []
        attacker_message: str = initial_query

        while simulator.has_next():
            try:
                target_response = _call_target(attacker_message, trace)
            except Exception as e:
                logger.warning("Target agent error on turn %d: %s", len(conversation), e)
                target_response = f"[Error: {e}]"

            conversation.append({"role": "attacker", "content": attacker_message})
            conversation.append({"role": "target", "content": target_response})

            try:
                attacker_result = simulator.act(target_response)
            except Exception as e:
                logger.warning("Attacker simulator error on turn %d: %s", len(conversation), e)
                break
            structured = attacker_result.structured_output
            attacker_message = str(getattr(structured, "message", "")) if structured else ""
            if not attacker_message.strip():
                logger.warning("Attacker produced an empty message; ending case early.")
                break
            if strategy is not None:
                attacker_message = strategy.enhance(
                    attacker_message,
                    target_response=target_response,
                    conversation=conversation,
                    attack_goal=goal,
                )

        result: dict = {"output": conversation}
        if trace:
            result["trajectory"] = trace
        return result

    return task_fn
