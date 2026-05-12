from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActorProfile(BaseModel):
    """Profile for actor simulation.

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
    """Default structured response from the actor simulator.

    Used as the LLM structured-output schema for `ActorSimulator.act` when no
    custom `structured_output_model` is provided. The LLM fills `reasoning`,
    `stop`, and `message`. The simulator fills `stop_reason` after the LLM call.

    Attributes:
        reasoning: Internal reasoning process for the response.
        stop: `True` when the actor signals the conversation should end.
        message: The actual message content from the actor. `None` when `stop=True`.
        stop_reason: Why the conversation ended. One of `"goal_completed"`,
            `"max_turns"`, or `None` while ongoing. Populated by the simulator
            after the LLM call.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reasoning: str = Field(..., description="Reasoning for the actor's response")
    stop: bool = Field(
        False,
        description="Set to true when the conversation goal is met or the conversation should end.",
    )
    message: str | None = Field(
        None,
        description="The actor's next message to the agent. Provide when stop=false; set to null when stop=true.",
    )
    stop_reason: str | None = Field(
        None,
        description='Populated by the simulator after the call. One of "goal_completed", "max_turns", or None.',
    )
