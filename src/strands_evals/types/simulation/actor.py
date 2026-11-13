from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from strands.types.content import ContentBlock


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
    message: str | list[ContentBlock] | None = Field(
        None,
        description="The actor's next message to the agent. Provide when stop=false; set to null when stop=true. "
        "Typed as `str | list[ContentBlock] | None` for forward compatibility, but v1 only supports the `str | None` "
        "variants — `list[ContentBlock]` raises until multimodal simulator output is supported.",
    )
    stop_reason: str | None = Field(
        None,
        description='Populated by the simulator after the call. One of "goal_completed", "max_turns", or None.',
    )

    @field_validator("message")
    @classmethod
    def _v1_only_str_message(cls, value: str | list[ContentBlock] | None) -> str | None:
        if isinstance(value, list):
            # Raise `ValueError` (not `TypeError`) so Pydantic wraps it into `ValidationError`,
            # keeping the structured-output retry path intact.
            raise ValueError(
                "ActorResponse.message must be a str or None in v1; got list[ContentBlock]. "
                "The annotation includes list[ContentBlock] for forward compatibility, but multimodal "
                "simulator output is not yet supported."
            )
        return value
