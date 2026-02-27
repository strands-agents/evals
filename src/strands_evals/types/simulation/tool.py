from typing import Callable

from pydantic import BaseModel, Field


class RegisteredTool(BaseModel):
    """
    Represents a registered function tool in the simulator.

    Attributes:
        name: Name of the tool for identification and registration.
        function: Function callable (excluded from serialization).
        output_schema: Pydantic BaseModel for output schema (excluded from serialization).
        initial_state_description: Initial state description for the tool's context.
        share_state_id: Optional shared state ID for sharing state between tools.
    """

    name: str = Field(..., description="Name of the tool")
    function: Callable | None = Field(default=None, description="Function callable", exclude=True)
    output_schema: type[BaseModel] | None = Field(
        default=None, description="Pydantic BaseModel for output schema", exclude=True
    )
    initial_state_description: str | None = Field(
        default=None, description="Initial state description for the tool's context"
    )
    share_state_id: str | None = Field(
        default=None, description="Optional shared state ID for sharing state between tools"
    )

    model_config = {"arbitrary_types_allowed": True}


class DefaultToolResponse(BaseModel):
    response: str = Field(default="", description="The tool's response message or result")
