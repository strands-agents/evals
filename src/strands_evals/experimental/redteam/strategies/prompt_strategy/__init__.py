"""Prompt-based attack strategies."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from strands.models.model import Model

from .....simulation.actor_simulator import ActorSimulator
from .....types.simulation import ActorProfile
from ..base import AttackRunResult, AttackStrategy

if TYPE_CHECKING:
    from ...case import RedTeamCase

logger = logging.getLogger(__name__)


class PromptStrategy(AttackStrategy):
    """AttackStrategy that drives the attack via a system prompt template.

    Runs an :class:`ActorSimulator` that escalates over the conversation using
    a static system prompt. The simulator decides when to stop; each turn the
    attacker message is sent through the injected ``call_target``.
    """

    def __init__(
        self, strategy_name: str, system_prompt_template: str, *, max_turns: int = 10, label: str | None = None
    ) -> None:
        super().__init__(label=label)
        self._name = strategy_name
        self._system_prompt_template = system_prompt_template
        self._max_turns = max_turns

    @property
    def name(self) -> str:
        return self._name

    @property
    def system_prompt_template(self) -> str:
        return self._system_prompt_template

    def run_attack(
        self,
        case: RedTeamCase,
        call_target: Callable[[str], str],
        *,
        max_turns: int,
        model: Model | str | None = None,
        **kwargs: Any,
    ) -> AttackRunResult:
        goal = case.config.attack_goal
        turn_cap = min(self._max_turns, max_turns)
        actor_profile = ActorProfile(
            traits=case.config.traits,
            context=goal.context,
            actor_goal=goal.actor_goal,
        )
        system_prompt_template = self._system_prompt_template.replace("{max_turns}", str(turn_cap))

        initial_query = str(case.input)
        simulator = ActorSimulator(
            actor_profile=actor_profile,
            initial_query=initial_query,
            system_prompt_template=system_prompt_template,
            model=model,  # type: ignore[arg-type]
            max_turns=turn_cap,
        )

        conversation: list[dict] = []
        attacker_message: str = initial_query

        while simulator.has_next():
            try:
                target_response = call_target(attacker_message)
            except Exception as e:
                logger.warning("turn=<%d>, error=<%s> | target agent failed", len(conversation), e)
                target_response = f"[Error: {e}]"

            conversation.append({"role": "attacker", "content": attacker_message})
            conversation.append({"role": "target", "content": target_response})

            try:
                attacker_result = simulator.act(target_response)
            except Exception as e:
                logger.warning("turn=<%d>, error=<%s> | attacker simulator failed", len(conversation), e)
                break
            structured = attacker_result.structured_output
            attacker_message = str(getattr(structured, "message", "")) if structured else ""
            if not attacker_message.strip():
                logger.warning("reason=<empty_message> | attacker produced empty message | ending case early")
                break

        return AttackRunResult(conversation=conversation, metadata={"turns_used": len(conversation) // 2})


__all__ = ["PromptStrategy"]
