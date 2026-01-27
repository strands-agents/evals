from enum import Enum
from typing import Any, Callable, Dict, Optional

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
        mcp_schema: MCP tool schema dictionary for MCP type tools.
        api_path: API endpoint path for API type tools.
        api_method: HTTP method for API type tools (GET, POST, etc.).
        initial_state_description: Initial state description for the tool's context.
        simulator_kwargs: Additional simulator configuration parameters.
    """
    name: str = Field(..., description="Name of the tool")
    tool_type: ToolType = Field(..., description="Type of the tool")
    function: Optional[Callable] = Field(default=None, description="Function callable", exclude=True)
    mcp_schema: Optional[Dict[str, Any]] = Field(default=None, description="MCP tool schema")
    api_path: Optional[str] = Field(default=None, description="API endpoint path")
    api_method: Optional[str] = Field(default=None, description="HTTP method")
    initial_state_description: Optional[str] = Field(default=None, description="Initial state description for the tool's context")
    simulator_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional simulator configuration")
    mode: str = Field(default="dynamic", description="Simulation mode: dynamic, static, mock")
    static_response: Optional[Dict[str, Any]] = Field(default=None, description="Static response for static mode")
    mock_function: Optional[Callable] = Field(default=None, description="Mock function for mock mode", exclude=True)

    class Config:
        arbitrary_types_allowed = True
