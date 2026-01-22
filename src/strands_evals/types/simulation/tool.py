from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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


class FailureConditions(BaseModel):
    """
    Configuration for failure simulation conditions.

    Attributes:
        enabled: Whether failure simulation is enabled for the tool.
        error_rate: Error rate between 0.0 and 1.0 for random failure injection.
        error_type: Type of error to simulate when failures occur.
        error_message: Optional custom error message for simulated failures.
    """
    
    enabled: bool = Field(default=False, description="Whether failure simulation is enabled")
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Error rate between 0.0 and 1.0")
    error_type: str = Field(default="execution_error", description="Type of error to simulate")
    error_message: Optional[str] = None
    
    @field_validator("error_rate")
    @classmethod
    def validate_error_rate(cls, v: float) -> float:
        """Validate error rate is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Error rate must be between 0.0 and 1.0")
        return v
    
    @model_validator(mode='after')
    def validate_enabled_state(self) -> 'FailureConditions':
        """Validate that if enabled is True, error_rate is > 0."""
        if self.enabled and self.error_rate == 0.0:
            raise ValueError("If failure conditions are enabled, error_rate must be greater than 0")
        return self


class ToolOverrideConfig(BaseModel):
    """
    Configuration for tool override behavior.

    Attributes:
        failure_conditions: Configuration for failure simulation conditions.
    """
    failure_conditions: FailureConditions = Field(default_factory=FailureConditions, description="Configuration for failure simulation")


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
        simulator_kwargs: Additional simulator configuration parameters.
    """
    name: str = Field(..., description="Name of the tool")
    tool_type: ToolType = Field(..., description="Type of the tool")
    function: Optional[Callable] = Field(default=None, description="Function callable", exclude=True)
    mcp_schema: Optional[Dict[str, Any]] = Field(default=None, description="MCP tool schema")
    api_path: Optional[str] = Field(default=None, description="API endpoint path")
    api_method: Optional[str] = Field(default=None, description="HTTP method")
    simulator_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional simulator configuration")

    class Config:
        arbitrary_types_allowed = True


class StateRegistry:
    """
    Simple state registry for maintaining tool state across calls.

    Attributes:
        _states: Internal dictionary mapping state keys to recorded call history.
    """
    
    def __init__(self):
        """
        Initialize state registry.

        Creates an empty state dictionary to track tool calls and responses
        across different simulation sessions.
        """
        self._states: Dict[str, Dict[str, Any]] = {}
    
    def get_state(self, key: str) -> Dict[str, Any]:
        """
        Get state for a given key.

        Args:
            key: State key to retrieve recorded calls for.

        Returns:
            Dictionary containing recorded call history for the key, empty if not found.
        """
        return self._states.get(key, {})
    
    def record_function_call(self, tool_name: str, state_key: str, parameters: Dict[str, Any], response_data: Any):
        """
        Record a function call in state.

        Args:
            tool_name: Name of the function tool that was called.
            state_key: State key to record the call under.
            parameters: Parameters passed to the function.
            response_data: Response data returned from the function.
        """
        if state_key not in self._states:
            self._states[state_key] = {"function_calls": []}
        
        call_record = {
            "tool_name": tool_name,
            "parameters": parameters,
            "response": response_data,
            "timestamp": datetime.now().isoformat()
        }
        self._states[state_key]["function_calls"].append(call_record)
    
    def record_mcp_tool_call(self, tool_name: str, state_key: str, input_mcp_payload: Dict[str, Any], response_data: Any):
        """
        Record an MCP tool call in state.

        Args:
            tool_name: Name of the MCP tool that was called.
            state_key: State key to record the call under.
            input_mcp_payload: Input payload sent to the MCP tool.
            response_data: Response data returned from the MCP tool.
        """
        if state_key not in self._states:
            self._states[state_key] = {"mcp_calls": []}
        
        call_record = {
            "tool_name": tool_name,
            "input": input_mcp_payload,
            "response": response_data,
            "timestamp": datetime.now().isoformat()
        }
        self._states[state_key]["mcp_calls"].append(call_record)
    
    def record_api_call(self, tool_name: str, state_key: str, path: str, method: str, input_data: Dict[str, Any], response: Dict[str, Any]):
        """
        Record an API call in state.

        Args:
            tool_name: Name of the API tool that was called.
            state_key: State key to record the call under.
            path: API endpoint path that was called.
            method: HTTP method used for the API call.
            input_data: Input data sent to the API endpoint.
            response: Response data returned from the API endpoint.
        """
        if state_key not in self._states:
            self._states[state_key] = {"api_calls": []}
        
        call_record = {
            "tool_name": tool_name,
            "path": path,
            "method": method,
            "input": input_data,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self._states[state_key]["api_calls"].append(call_record)
