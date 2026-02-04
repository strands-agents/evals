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


class ToolSimulationMode(Enum):
    """
    Enumeration of supported simulation modes.

    Attributes:
        DYNAMIC: Generate responses using LLM based on tool context and history.
        STATIC: Return predefined static responses.
        MOCK: Call custom mock functions for controlled behavior.
    """

    DYNAMIC = "dynamic"
    STATIC = "static"
    MOCK = "mock"


class RegisteredTool(BaseModel):
    """
    Represents a registered tool in the simulator.

    Attributes:
        name: Name of the tool for identification and registration.
        tool_type: Type of the tool (FUNCTION, MCP, or API).
        function: Function callable for FUNCTION type tools (excluded from serialization).
        mcp_schema: MCP tool schema dictionary for MCP type tools.
        api_path: API endpoint path for API type tools.
        api_method: HTTP method for API type tools (GET, POST, etc.).
        initial_state_description: Initial state description for the tool's context.
        simulator_kwargs: Additional simulator configuration parameters.
    """

    name: str = Field(..., description="Name of the tool")
    tool_type: ToolType = Field(..., description="Type of the tool")
    function: Callable | None = Field(default=None, description="Function callable", exclude=True)
    mcp_schema: dict[str, Any] | None = Field(default=None, description="MCP tool schema")
    api_path: str | None = Field(default=None, description="API endpoint path")
    api_method: str | None = Field(default=None, description="HTTP method")
    initial_state_description: str | None = Field(
        default=None, description="Initial state description for the tool's context"
    )
    simulator_kwargs: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional simulator configuration"
    )
    mode: ToolSimulationMode = Field(
        default=ToolSimulationMode.DYNAMIC, description="Simulation mode: dynamic, static, mock"
    )
    static_response: dict[str, Any] | None = Field(default=None, description="Static response for static mode")
    mock_function: Callable | None = Field(default=None, description="Mock function for mock mode", exclude=True)

    model_config = {"arbitrary_types_allowed": True}


class MCPContentItem(BaseModel):
    """Individual content item in MCP response."""

    type: str = Field(..., description="Type of content (text, resource, etc.)")
    text: str | None = Field(default=None, description="Text content")
    resource: dict[str, Any] | None = Field(default=None, description="Resource information")


class MCPToolResponse(BaseModel):
    """
    Response model for MCP tool simulation using structured output.

    Follows the MCP response format with content array and optional error flag.
    """

    content: list[MCPContentItem] = Field(..., description="Array of content items")
    isError: bool | None = Field(default=False, description="Whether this response represents an error")


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
