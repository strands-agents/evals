from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActorProfile(BaseModel):
    """
    Profile for actor simulation.

    Attributes:
        traits: Dictionary of actor characteristics and attributes.
        context: Supplementary background information about the actor.
        actor_goal: What the actor ultimately wants to achieve in the interaction.
    """

    traits: dict[str, Any] = Field(..., description="Actor traits for simulation")
    context: str = Field(..., description="Supplementary actor background details")
    actor_goal: str = Field(
        ...,
        description="What the actor ultimately wants to achieve in this interaction - "
        "should be specific, actionable, and written from the actor's perspective",
    )


class ActorResponse(BaseModel):
    """
    Structured response from an actor.

    Used by :meth:`ActorSimulator.act` as the LLM's structured-output schema. The
    simulator keeps ``act()`` on this legacy shape so existing callers continue
    to see ``message: str`` as a required field. New code should prefer
    :meth:`ActorSimulator.act_structured` and :class:`SimulatorResult`.

    Attributes:
        reasoning: Internal reasoning process for the response.
        message: The actual message content from the actor.
    """

    reasoning: str = Field(..., description="Reasoning for the actor's response")
    message: str = Field(..., description="Message from the actor")


class SimulatorResult(BaseModel):
    """
    Typed return value from :meth:`ActorSimulator.act_structured`.

    Used in two roles:

    - **As the LLM structured-output schema.** ``reasoning``, ``stop``, and
      ``message`` are produced by the LLM via Strands' tool-use contract.
      ``stop_reason`` is not part of the schema — the simulator fills it in
      after the call, but it is kept on this model so the public return type
      stays a single class.

      When ``ActorSimulator`` is given an ``input_type``, a dynamic subclass of
      this model is built that narrows ``message`` to ``input_type | None`` so
      the LLM's tool-use schema enforces the caller's agent-input shape.

    - **As the caller-facing result.** Callers of ``act_structured()`` receive
      an instance of this class (or its dynamic subclass) with all four fields
      populated.

    Attributes:
        message: The actor's next message. An ``input_type`` instance when the
            simulator was constructed with ``input_type``; a plain string or
            ``None`` otherwise. ``None`` is expected when ``stop=True``.
        reasoning: The actor's internal reasoning for this response.
        stop: ``True`` when the actor signals the conversation should end
            (either the goal was completed or ``max_turns`` was reached).
        stop_reason: Why the conversation ended: ``"goal_completed"``,
            ``"max_turns"``, or ``None`` while the conversation is still ongoing.
            Populated by the simulator after the LLM call; not part of the
            LLM-facing schema semantics even though the field exists on the
            model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reasoning: str = Field(..., description="Reasoning for the actor's response")
    stop: bool = Field(
        False,
        description=(
            "Set to true when the conversation goal is met or the "
            "conversation should end."
        ),
    )
    message: Any = Field(
        None,
        description=(
            "The actor's next message to the agent. Provide when stop=false; "
            "set to null when stop=true."
        ),
    )
    stop_reason: str | None = Field(
        None,
        description=(
            "Populated by the simulator after the call. One of "
            '"goal_completed", "max_turns", or None.'
        ),
    )
