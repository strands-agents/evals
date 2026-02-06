from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field


class ToolType(Enum):
    """
    Enumeration of supported tool types for simulation.

    Attributes:
        FUNCTION: Python function tools that can be called directly.
        MCP: Model Context Protocol tools with structured schemas.
        API: REST API endpoints with HTTP methods and paths.
    """

    FUNCTION = "function"
    MCP = "mcp"
    API = "api"


class RegisteredTool(BaseModel):
    """
    Represents a registered tool in the simulator.

    Attributes:
        name: Name of the tool for identification and registration.
        tool_type: Type of the tool (FUNCTION, MCP, or API).
        function: Function callable for FUNCTION type tools (excluded from serialization).
        output_schema: Pydantic BaseModel for output schema (excluded from serialization).
        prompt_template: Custom prompt template override.
        mcp_schema: MCP tool schema dictionary for MCP type tools.
        api_path: API endpoint path for API type tools.
        api_method: HTTP method for API type tools (GET, POST, etc.).
        initial_state_description: Initial state description for the tool's context.
        simulator_kwargs: Additional simulator configuration parameters.
    """

    name: str = Field(..., description="Name of the tool")
    tool_type: ToolType = Field(..., description="Type of the tool")
    function: Callable | None = Field(default=None, description="Function callable", exclude=True)
    output_schema: type[BaseModel] | None = Field(
        default=None, description="Pydantic BaseModel for output schema", exclude=True
    )
    prompt_template: str | None = Field(default=None, description="Custom prompt template override")
    mcp_schema: dict[str, Any] | None = Field(default=None, description="MCP tool schema")
    api_path: str | None = Field(default=None, description="API endpoint path")
    api_method: str | None = Field(default=None, description="HTTP method")
    initial_state_description: str | None = Field(
        default=None, description="Initial state description for the tool's context"
    )
    simulator_kwargs: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional simulator configuration"
    )

    model_config = {"arbitrary_types_allowed": True}


class ContentBlock(BaseModel):
    """Individual content item in MCP response following official MCP specification."""

    type: str = Field(..., description="Type of content (text, resource, etc.)")
    text: str | None = Field(default=None, description="Text content")
    resource: dict[str, Any] | None = Field(default=None, description="Resource information")


class MCPToolResponse(BaseModel):
    """
    Response model for MCP tool simulation following official MCP specification.

    Matches the official MCP ToolResultContent format for consistency with MCP ecosystem.
    """

    tool_use_id: str = Field(..., description="Unique identifier corresponding to tool call's id")
    content: list[ContentBlock] = Field(
        default_factory=list, description="List of content objects representing tool result"
    )
    structured_content: dict[str, Any] | None = Field(
        default=None, description="Structured tool output matching outputSchema"
    )
    is_error: bool | None = Field(default=False, description="Whether tool execution resulted in error")
    meta: dict[str, Any] | None = Field(default=None, description="Metadata following MCP specification")


class APIErrorDetail(BaseModel):
    """Error detail structure for API responses."""

    type: str = Field(..., description="Error type identifier")
    title: str = Field(..., description="Human-readable error title")
    detail: str = Field(..., description="Detailed error description")


class APIToolResponse(BaseModel):
    """
    Response model for API tool simulation using structured output.

    Follows HTTP response format with status code and optional data or error.
    """

    status: int = Field(..., description="HTTP status code")
    data: Any | None = Field(default=None, description="Response data for successful requests")
    error: APIErrorDetail | None = Field(default=None, description="Error details for failed requests")

    # Allow additional fields for flexibility
    model_config = {"extra": "allow"}
