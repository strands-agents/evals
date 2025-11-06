from pydantic import BaseModel, Field
from typing_extensions import Any


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

    Attributes:
        reasoning: Internal reasoning process for the response.
        message: The actual message content from the actor.
    """

    reasoning: str = Field(..., description="Reasoning for the actor's response")
    message: str = Field(..., description="Message from the actor")
