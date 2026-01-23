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

    class Config:
        arbitrary_types_allowed = True


class StateRegistry:
    """
    State registry for managing shared state between tool simulators.
    Organized by state_key to isolate state between different tools or shared state groups.
    """
    
    def __init__(self):
        """
        Initialize state registry.

        Creates an empty state dictionary to track tool calls and responses
        across different simulation sessions.
        """
        self._states: Dict[str, Dict[str, Any]] = {}
    
    def initialize_state_via_description(self, initial_state_description: str, state_key: str) -> None:
        """
        Initialize state based on the provided description.

        This method pre-seeds the state with an initial description that will be
        included in all subsequent LLM prompts, allowing the simulator to have
        context about pre-existing data or system state.

        Args:
            initial_state_description: Description of the initial state (e.g., existing
                database records, system configuration, etc.).
            state_key: Key for the state in the registry (typically tool_name or share_state_id).
        """
        if state_key not in self._states:
            self._states[state_key] = {
                "initial_state": initial_state_description,
                "previous_calls": [],
                "user_context": {},
            }

    def get_state(self, state_key: str) -> Dict[str, Any]:
        """
        Get state for a specific tool or shared state group.

        Args:
            state_key: Key for the state (tool_name or share_state_id).

        Returns:
            State dictionary containing previous_calls and user_context.
        """
        if state_key is None:
            raise ValueError("Value of state_key is required.")

        if state_key not in self._states:
            self._states[state_key] = {
                "previous_calls": [],
                "user_context": {},
            }

        return dict(self._states[state_key])

    def record_function_call(
        self,
        tool_name: str,
        state_key: str,
        parameters: Dict[str, Any],
        response_data: Any,
    ) -> Dict[str, Any]:
        """
        Record a function call in the tool's state history.

        Args:
            tool_name: Name of the function being called.
            state_key: Key for the state (tool_name or share_state_id).
            parameters: Parameters passed to the function.
            response_data: Response from the function call.

        Returns:
            Updated state dictionary.
        """
        state = self.get_state(state_key)
        date_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        state["previous_calls"].append({
            'tool_name': tool_name,
            'tool_type': 'function',
            'parameters': parameters,
            'response': response_data,
            'timestamp': date_timestamp
        })

        # Keep history manageable
        if len(state["previous_calls"]) > 20:
            state["previous_calls"] = state["previous_calls"][-20:]

        # Update the stored state
        self._states[state_key] = state

        return state

    def record_mcp_tool_call(
        self,
        tool_name: str,
        state_key: str,
        input_mcp_payload: Dict[str, Any],
        response_data: Any,
    ) -> Dict[str, Any]:
        """
        Record an MCP tool call in the tool's state history.

        Args:
            tool_name: Name of the MCP tool being called.
            state_key: Key for the state (tool_name or share_state_id).
            input_mcp_payload: Input payload for the MCP tool call.
            response_data: Response from the MCP tool call.

        Returns:
            Updated state dictionary.
        """
        state = self.get_state(state_key)
        date_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        state["previous_calls"].append({
            'tool_name': tool_name,
            'tool_type': 'mcp',
            'input_mcp_payload': input_mcp_payload,
            'response': response_data,
            'timestamp': date_timestamp
        })

        # Keep history manageable
        if len(state["previous_calls"]) > 20:
            state["previous_calls"] = state["previous_calls"][-20:]

        # Update the stored state
        self._states[state_key] = state

        return state

    def record_api_call(
        self,
        tool_name: str,
        state_key: str,
        path: str,
        method: str,
        input_data: Any,
        response: Any,
    ) -> Dict[str, Any]:
        """
        Record an API call in the tool's state history.

        Args:
            tool_name: Name of the API tool being called.
            state_key: Key for the state (tool_name or share_state_id).
            path: API endpoint path.
            method: HTTP method.
            input_data: Input data for the API call.
            response: Response from the API call.

        Returns:
            Updated state dictionary.
        """
        state = self.get_state(state_key)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        state["previous_calls"].append({
            'tool_name': tool_name,
            'tool_type': 'api',
            'path': path,
            'method': method,
            'input': input_data,
            'response': response,
            'timestamp': timestamp
        })

        # Keep history manageable
        if len(state["previous_calls"]) > 20:
            state["previous_calls"] = state["previous_calls"][-20:]

        # Update the stored state
        self._states[state_key] = state

        return state

    def set_user_context(self, state_key: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set user context for a state.

        Args:
            state_key: Key for the state (tool_name or share_state_id).
            user_context: User context dictionary to store.

        Returns:
            Updated state dictionary.
        """
        state = self.get_state(state_key)
        state["user_context"] = user_context
        self._states[state_key] = state
        return state

    def clear_state(self, state_key: str) -> None:
        """
        Clear state for a specific tool or shared state group.

        Args:
            state_key: Key for the state to clear.
        """
        if state_key in self._states:
            del self._states[state_key]
