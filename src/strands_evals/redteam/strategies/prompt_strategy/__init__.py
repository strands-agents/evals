"""Prompt-based attack strategies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from strands.models.model import Model

from ....simulation.actor_simulator import ActorSimulator
from ....types.simulation import ActorProfile
from ..base import ATTACKER_REFUSED, MAX_ALLOWED_TURNS, AttackRunResult, AttackStrategy

if TYPE_CHECKING:
    from ...case import RedTeamCase
    from ..target_session import TargetSession

logger = logging.getLogger(__name__)


class PromptStrategy(AttackStrategy):
    """`AttackStrategy` that drives the attack via an `ActorSimulator` system-prompt template."""

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
        target_session: TargetSession,
        *,
        max_turns: int = MAX_ALLOWED_TURNS,
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
        # The opening message is the case input; only the simulator's own messages need the refusal check.
        attacker_message: str = initial_query
        refusal_stats = self._new_refusal_stats()

        while simulator.has_next():
            # Checked only once it is about to be sent: the message produced alongside the stop signal never is,
            # so it can't flag the case as refused.
            if conversation and not attacker_message.strip():
                refusal_stats[ATTACKER_REFUSED] = True
                logger.warning("reason=<empty_message> | attacker produced empty message | ending case early")
                break
            if conversation and self._attacker_refused_turn(attacker_message, model, conversation, refusal_stats):
                break
            target_response = target_session.invoke(attacker_message)

            conversation.append({"role": "attacker", "content": attacker_message})
            conversation.append({"role": "target", "content": target_response})

            try:
                attacker_result = simulator.act(target_response)
            except AttributeError as e:
                # `act` reads `.stop` off the structured output, so a reply with none (e.g. content-filtered)
                # raises here: an attacker that gave no message. Any other error propagates, as in other strategies.
                refusal_stats[ATTACKER_REFUSED] = True
                logger.warning("turn=<%d>, error=<%s> | attacker produced no structured output", len(conversation), e)
                break
            structured = attacker_result.structured_output
            attacker_message = str(getattr(structured, "message", "")) if structured else ""

        return AttackRunResult(
            conversation=conversation,
            metadata={"turns_used": len(conversation) // 2, **refusal_stats},
        )

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out.update(
            strategy_name=self._name,
            system_prompt_template=self._system_prompt_template,
            max_turns=self._max_turns,
        )
        return out


__all__ = ["PromptStrategy"]
