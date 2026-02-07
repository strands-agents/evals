from typing import Any, Callable

from pydantic import BaseModel, Field


class RegisteredTool(BaseModel):
    """
    Represents a registered function tool in the simulator.

    Attributes:
        name: Name of the tool for identification and registration.
        function: Function callable (excluded from serialization).
        output_schema: Pydantic BaseModel for output schema (excluded from serialization).
        initial_state_description: Initial state description for the tool's context.
        simulator_kwargs: Additional simulator configuration parameters.
    """

    name: str = Field(..., description="Name of the tool")
    function: Callable | None = Field(default=None, description="Function callable", exclude=True)
    output_schema: type[BaseModel] | None = Field(
        default=None, description="Pydantic BaseModel for output schema", exclude=True
    )
    initial_state_description: str | None = Field(
        default=None, description="Initial state description for the tool's context"
    )
    simulator_kwargs: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional simulator configuration"
    )

    model_config = {"arbitrary_types_allowed": True}
