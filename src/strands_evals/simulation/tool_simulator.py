import inspect
import json
import logging
import warnings
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Callable

from strands import Agent
from strands.agent import AgentResult
from strands.models.model import Model

from strands_evals.types.simulation.tool import (
    APIToolResponse,
    MCPToolResponse,
    RegisteredTool,
    ToolType,
)

from .prompt_templates.tool_response_generation import (
    API_TOOL_RESPONSE_GENERATION_PROMPT,
    FUNCTION_TOOL_RESPONSE_GENERATION_PROMPT,
    MCP_TOOL_RESPONSE_GENERATION_PROMPT,
)

logger = logging.getLogger(__name__)


class StateRegistry:
    """
    State registry for managing shared state between tool simulators.
    Organized by state_key to isolate state between different tools or shared state groups.

    The registry automatically maintains a bounded cache of tool calls per state key.
    The maximum number of tool calls stored is configurable via max_tool_call_cache_size parameter.
    """

    def __init__(self, max_tool_call_cache_size: int = 20):
        """
        Initialize state registry.

        Creates an empty state dictionary to track tool calls and responses
        across different simulation sessions. Tool call cache is automatically
        bounded to prevent excessive memory usage.

        Args:
            max_tool_call_cache_size: Maximum number of tool calls to store per state key.
                                      Older calls are automatically evicted when limit is exceeded.
                                      Default is 20.
        """
        self._max_tool_call_cache_size = max_tool_call_cache_size
        self._states: defaultdict[str, dict[str, Any]] = defaultdict(
            lambda: {"previous_calls": deque(maxlen=self._max_tool_call_cache_size)}
        )

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
                "previous_calls": deque(maxlen=self._max_tool_call_cache_size),
            }
        else:
            warnings.warn(
                f"State with key '{state_key}' already initialized. Skipping re-initialization.", stacklevel=2
            )

    def get_state(self, state_key: str) -> dict[str, Any]:
        """
        Get state for a specific tool or shared state group.

        Args:
            state_key: Key for the state (tool_name or share_state_id).

        Returns:
            State dictionary containing previous_calls.
        """
        if state_key is None:
            raise ValueError("Value of state_key is required.")

        # Access will create the default state automatically due to defaultdict
        state = self._states[state_key]

        # Convert deque to list for JSON serialization compatibility
        return {key: list(value) if isinstance(value, deque) else value for key, value in state.items()}

    def cache_tool_call(
        self,
        tool_name: str,
        state_key: str,
        tool_type: ToolType,
        response_data: Any,
        **call_data: Any,
    ) -> dict[str, Any]:
        """
        Cache a tool call in the tool's state key.

        Args:
            tool_name: Name of the tool being called.
            state_key: Key for the state (tool_name or share_state_id).
            tool_type: Type of the tool (FUNCTION, MCP, or API).
            response_data: Response from the tool call.
            **call_data: Tool-specific call data (parameters, input_mcp_payload, path, method, input_data, etc.).

        Returns:
            Updated state dictionary.
        """
        # Access the actual state storage (not converted copy)
        state = self._states[state_key]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Build call record based on tool type
        call_record = {
            "tool_name": tool_name,
            "tool_type": tool_type.value,
            "response": response_data,
            "timestamp": timestamp,
        }

        # Add tool-specific fields
        if tool_type == ToolType.FUNCTION:
            call_record["parameters"] = call_data.get("parameters", {})
        elif tool_type == ToolType.MCP:
            call_record["input_mcp_payload"] = call_data.get("input_mcp_payload", {})
        elif tool_type == ToolType.API:
            call_record.update(
                {
                    "path": call_data.get("path", ""),
                    "method": call_data.get("method", "GET"),
                    "input": call_data.get("input_data", {}),
                }
            )

        # Append to deque with automatic FIFO eviction when cache is full
        state["previous_calls"].append(call_record)

        # Return converted state for external use
        return self.get_state(state_key)

    def clear_state(self, state_key: str) -> None:
        """
        Clear state for a specific tool or shared state group.

        Args:
            state_key: Key for the state to clear.
        """
        if state_key in self._states:
            del self._states[state_key]


class ToolSimulator:
    """
    Simulates tool behavior with decorator-based registration system for agent evaluation.

    ToolSimulator provides decorator functions for different tool types and maintains
    a registry of all registered tools. It can be configured to override tool
    behavior for simulation purposes, enabling controlled testing scenarios.

    The simulator automatically maintains a bounded cache of tool calls for context.
    The maximum number of tool calls stored per state key is configurable via
    max_tool_call_cache_size parameter (default: 20).

    Attributes:
        model: Provider for running inference or model identifier for Bedrock.
        _registered_tools: Class-level registry for all registered tools.
        _state_registry: Registry for maintaining tool state across calls.
    """

    # Class-level registry for all registered tools
    _registered_tools: dict[str, RegisteredTool] = {}
    _state_registry: StateRegistry

    def __init__(
        self,
        state_registry: StateRegistry | None = None,
        function_tool_prompt: str | None = None,
        mcp_tool_prompt: str | None = None,
        api_tool_prompt: str | None = None,
        model: Model | str | None = None,
        framework: str = "strands",
        max_tool_call_cache_size: int = 20,
    ):
        """
        Initialize a ToolSimulator instance.

        Args:
            state_registry: Registry for maintaining tool state. If not provided,
                           a new StateRegistry will be created with max_tool_call_cache_size.
            function_tool_prompt: Optional custom prompt for function tool response generation
            mcp_tool_prompt: Optional custom prompt for MCP tool response generation
            api_tool_prompt: Optional custom prompt for API tool response generation
            model: Provider for running inference or a string representing the model-id for Bedrock to use
            framework: Agent framework to use (default: "strands")
            max_tool_call_cache_size: Maximum number of tool calls to store per state key.
                                     Only used when creating a new StateRegistry (ignored if state_registry
                                     is provided). Older calls are automatically evicted when limit is exceeded.
                                     Default is 20.
        """
        # Store framework selection
        self.framework = framework
        # Store model configuration for creating internal agents
        self.model = model

        # Set custom prompts or use defaults
        self.function_tool_prompt = function_tool_prompt or FUNCTION_TOOL_RESPONSE_GENERATION_PROMPT
        self.mcp_tool_prompt = mcp_tool_prompt or MCP_TOOL_RESPONSE_GENERATION_PROMPT
        self.api_tool_prompt = api_tool_prompt or API_TOOL_RESPONSE_GENERATION_PROMPT

        # Set up state registry
        self._state_registry = state_registry or StateRegistry(max_tool_call_cache_size=max_tool_call_cache_size)

        # Initialize shared states from registered tools
        self._initialize_shared_states()

    def _initialize_shared_states(self):
        """Initialize shared states from registered tools' initial descriptions."""
        for tool_name, registered_tool in self._registered_tools.items():
            if registered_tool.initial_state_description:
                # Determine state key from tool name or simulator kwargs
                state_key = (
                    registered_tool.simulator_kwargs.get("share_state_id", registered_tool.name)
                    if registered_tool.simulator_kwargs
                    else registered_tool.name
                )

                # Initialize state with description
                self._state_registry.initialize_state_via_description(
                    registered_tool.initial_state_description, state_key
                )
                logger.info(f"Initialized state for tool '{tool_name}' with key '{state_key}'")

    def __getattr__(self, name: str) -> Any:
        """
        Allow direct access to registered tools as attributes.

        Args:
            name: Tool name

        Returns:
            Tool callable

        Raises:
            AttributeError: If tool not found
        """
        registered_tool = self._registered_tools.get(name)
        if registered_tool:
            return self._create_tool_wrapper(registered_tool)

        raise AttributeError(f"Tool '{name}' not found in registered tools")

    def _create_tool_wrapper(self, registered_tool: RegisteredTool):
        """Create a framework-compatible tool wrapper."""

        def wrapper(*args, **kwargs):
            # Determine state key
            state_key = (
                registered_tool.simulator_kwargs.get("share_state_id", registered_tool.name)
                if registered_tool.simulator_kwargs
                else registered_tool.name
            )

            # Build input data based on tool type
            if registered_tool.tool_type == ToolType.FUNCTION:
                parameters_string = (
                    json.dumps({"args": args, "kwargs": kwargs}, indent=2) if args else json.dumps(kwargs, indent=2)
                )

                input_data = {
                    "tool_name": registered_tool.name,
                    "parameters": parameters_string,
                }

            elif registered_tool.tool_type == ToolType.MCP:
                input_data = {
                    "tool_name": registered_tool.name,
                    "input_mcp_payload": kwargs,
                }

            elif registered_tool.tool_type == ToolType.API:
                input_data = {
                    "tool_name": registered_tool.name,
                    "user_input_api_payload": kwargs,
                    "path": registered_tool.api_path or "",
                    "method": registered_tool.api_method or "GET",
                }

            else:
                raise ValueError(f"Unsupported tool type: {registered_tool.tool_type}")

            return self._call(registered_tool, input_data, state_key)

        # Copy function metadata
        if registered_tool.function:
            wrapper.__name__ = registered_tool.function.__name__
            wrapper.__doc__ = registered_tool.function.__doc__
        else:
            wrapper.__name__ = registered_tool.name
            wrapper.__doc__ = f"Simulated {registered_tool.name} tool"

        # Use framework-specific method to create the tool wrapper
        if self.framework == "strands":
            return self._create_strands_tool_wrapper(registered_tool, wrapper)
        else:
            raise ValueError(f"Framework '{self.framework}' is not supported. Only 'strands' is currently supported.")

    def _create_strands_tool_wrapper(self, registered_tool: RegisteredTool, wrapper: Callable):
        """Create a Strands-specific DecoratedFunctionTool wrapper."""
        from strands.tools.decorator import DecoratedFunctionTool, FunctionToolMetadata

        # Create tool spec based on function signature and docstring
        tool_description = wrapper.__doc__ or f"Simulated {registered_tool.name} tool"

        # Build input schema from function signature
        input_schema: dict[str, Any] = {"type": "object", "properties": {}}
        if registered_tool.function:
            try:
                sig = inspect.signature(registered_tool.function)
                for param_name, param in sig.parameters.items():
                    if param.annotation != inspect.Parameter.empty:
                        param_type = (
                            str(param.annotation).replace("<class '", "").replace("'>", "").replace("typing.", "")
                        )
                        if "str" in param_type.lower():
                            input_schema["properties"][param_name] = {"type": "string"}
                        elif "int" in param_type.lower():
                            input_schema["properties"][param_name] = {"type": "integer"}
                        elif "float" in param_type.lower():
                            input_schema["properties"][param_name] = {"type": "number"}
                        elif "bool" in param_type.lower():
                            input_schema["properties"][param_name] = {"type": "boolean"}
                        else:
                            input_schema["properties"][param_name] = {"type": "object"}
                    else:
                        input_schema["properties"][param_name] = {"type": "string"}  # default
            except Exception:
                pass  # fallback to empty schema

        # Create Strands tool's FunctionToolMetadata object and DecoratedFunctionTool instance
        metadata = FunctionToolMetadata(registered_tool.function or wrapper)

        # Extract tool_spec from metadata; override with our custom description if needed
        extracted_tool_spec = metadata.extract_metadata()
        if tool_description != extracted_tool_spec.get("description"):
            extracted_tool_spec["description"] = tool_description
        extracted_tool_spec["name"] = registered_tool.name

        decorated_tool = DecoratedFunctionTool(
            tool_name=registered_tool.name,
            tool_spec=extracted_tool_spec,
            tool_func=wrapper,  # Always use wrapper to ensure simulation logic is executed
            metadata=metadata,
        )

        return decorated_tool

    def _simulate_tool_call(self, prompt: str, structured_output_model=None) -> Any:
        """Tool simulation agent creation and response generation."""
        agent = Agent(
            system_prompt=prompt,
            tools=[],
            model=self.model,
            callback_handler=None,
        )
        return agent(prompt, structured_output_model=structured_output_model)

    def _parse_simulated_response(self, result: AgentResult) -> dict[str, Any]:
        """Parse tool simulation agent response, trying to extract JSON first, falling back to wrapping in result."""
        response_text = str(result) or "No response"
        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError:
            response_data = {"result": response_text}
        return response_data

    def _call(self, registered_tool: RegisteredTool, input_data: dict[str, Any], state_key: str) -> Any:
        """Simulate a tool invocation and return the response."""
        if registered_tool.tool_type == ToolType.FUNCTION:
            return self._handle_function_tool(registered_tool.name, input_data, state_key)
        elif registered_tool.tool_type == ToolType.MCP:
            return self._handle_mcp_tool(registered_tool.name, input_data, state_key)
        elif registered_tool.tool_type == ToolType.API:
            return self._handle_api_tool(registered_tool.name, input_data, state_key)
        else:
            raise ValueError(f"Unsupported tool type: {registered_tool.tool_type}")

    def _handle_function_tool(self, tool_name: str, input_data: dict[str, Any], state_key: str) -> dict[str, Any]:
        """Handle function tool simulation."""
        parameters = input_data.get("parameters", {})

        current_state = self._state_registry.get_state(state_key)

        prompt = self.function_tool_prompt.format(
            tool_name=tool_name,
            parameters=json.dumps(parameters, indent=2),
            initial_state_description=current_state.get("initial_state", "No initial state provided."),
            previous_responses=json.dumps(current_state.get("previous_calls", []), indent=2),
        )

        result = self._simulate_tool_call(prompt, structured_output_model=None)

        response_data = self._parse_simulated_response(result)

        self._state_registry.cache_tool_call(
            tool_name, state_key, ToolType.FUNCTION, response_data, parameters=parameters
        )

        return response_data

    def _handle_mcp_tool(self, tool_name: str, input_data: dict[str, Any], state_key: str) -> dict[str, Any]:
        """Handle MCP tool simulation."""
        input_mcp_payload = input_data.get("input_mcp_payload", {})

        current_state = self._state_registry.get_state(state_key)

        prompt = self.mcp_tool_prompt.format(
            tool_name=tool_name,
            mcp_payload=json.dumps(input_mcp_payload, indent=2),
            initial_state_description=current_state.get("initial_state", "No initial state provided."),
            previous_responses=json.dumps(current_state.get("previous_calls", []), indent=2),
        )

        result = self._simulate_tool_call(prompt, structured_output_model=MCPToolResponse)

        response_data = self._parse_simulated_response(result)

        self._state_registry.cache_tool_call(
            tool_name, state_key, ToolType.MCP, response_data, input_mcp_payload=input_mcp_payload
        )

        return response_data

    def _handle_api_tool(self, tool_name: str, input_data: dict[str, Any], state_key: str) -> dict[str, Any]:
        """Handle API tool simulation."""
        user_input_api_payload = input_data.get("user_input_api_payload", {})
        path = input_data.get("path", "")
        method = input_data.get("method", "GET").upper()  # Normalize HTTP method to uppercase

        current_state = self._state_registry.get_state(state_key)

        prompt = self.api_tool_prompt.format(
            tool_name=tool_name,
            path=path,
            method=method,
            api_payload=json.dumps(user_input_api_payload, indent=2) if user_input_api_payload else "{}",
            initial_state_description=current_state.get("initial_state", "No initial state provided."),
            previous_responses=json.dumps(current_state.get("previous_calls", []), indent=2),
        )

        result = self._simulate_tool_call(prompt, structured_output_model=APIToolResponse)

        response_data = self._parse_simulated_response(result)

        self._state_registry.cache_tool_call(
            tool_name,
            state_key,
            ToolType.API,
            response_data,
            path=path,
            method=method,
            input_data=user_input_api_payload,
        )

        return response_data

    @classmethod
    def clear_registry(cls):
        """Clear all registered tools. Useful for testing."""
        cls._registered_tools.clear()
        logger.info("Cleared tool registry")

    def get_state(self, state_key: str) -> dict[str, Any]:
        """
        Get state for a specific tool or shared state group.

        Args:
            state_key: Key for the state (tool_name or share_state_id).

        Returns:
            State dictionary containing previous_calls.
        """
        return self._state_registry.get_state(state_key)

    @classmethod
    def function_tool(
        cls,
        name: str | None = None,
        initial_state_description: str | None = None,
        **simulator_kwargs,
    ) -> Callable:
        """
        Decorator for registering Python function tools.

        Args:
            name: Optional name for the tool. If None, uses function.__name__
            initial_state_description: Optional initial state description for the tool's context
            **simulator_kwargs: Additional simulator configuration

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            try:
                tool_name = name or func.__name__

                # Register tool
                registered_tool = RegisteredTool(
                    name=tool_name,
                    tool_type=ToolType.FUNCTION,
                    function=func,
                    initial_state_description=initial_state_description,
                    simulator_kwargs=simulator_kwargs,
                )
                cls._registered_tools[tool_name] = registered_tool

                logger.info(f"Registered function tool: {tool_name}")

            except Exception as e:
                raise RuntimeError(f"Error registering function tool {name or func.__name__}: {e}") from e

            return func

        return decorator

    @classmethod
    def mcp_tool(
        cls,
        name: str | None = None,
        schema: dict[str, Any] | None = None,
        initial_state_description: str | None = None,
        **simulator_kwargs,
    ) -> Callable:
        """
        Decorator for registering MCP (Model Context Protocol) tools.

        Args:
            name: Optional name for the tool. If None, uses function.__name__
            schema: MCP tool schema dictionary
            initial_state_description: Optional initial state description for the tool's context
            **simulator_kwargs: Additional simulator configuration

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__

            if schema is None:
                raise ValueError(f"MCP schema is required for tool {tool_name}")

            # Register tool
            registered_tool = RegisteredTool(
                name=tool_name,
                tool_type=ToolType.MCP,
                function=func,
                mcp_schema=schema,
                initial_state_description=initial_state_description,
                simulator_kwargs=simulator_kwargs,
            )
            cls._registered_tools[tool_name] = registered_tool

            logger.info(f"Registered MCP tool: {tool_name}")
            return func

        return decorator

    @classmethod
    def api_tool(
        cls,
        name: str | None = None,
        path: str | None = None,
        method: str | None = None,
        schema: dict[str, Any] | None = None,
        initial_state_description: str | None = None,
        **simulator_kwargs,
    ) -> Callable:
        """
        Decorator for registering API tools.

        Args:
            name: Optional name for the tool. If None, uses function.__name__
            path: API endpoint path
            method: HTTP method (GET, POST, etc.)
            schema: API tool schema dictionary
            initial_state_description: Optional initial state description for the tool's context
            **simulator_kwargs: Additional simulator configuration

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__

            if path is None:
                raise ValueError("API path is required")
            if method is None:
                raise ValueError("HTTP method is required")

            # Register tool
            registered_tool = RegisteredTool(
                name=tool_name,
                tool_type=ToolType.API,
                function=func,
                api_path=path,
                api_method=method,
                initial_state_description=initial_state_description,
                simulator_kwargs=simulator_kwargs,
            )
            cls._registered_tools[tool_name] = registered_tool

            logger.info(f"Registered API tool: {tool_name}")
            return func

        return decorator

    def get_tool(self, tool_name: str) -> Callable | None:
        """
        Get a tool by name and create a simulation wrapper.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Tool callable wrapper if found, None otherwise
        """
        registered_tool = self._registered_tools.get(tool_name)
        if not registered_tool:
            return None

        return self._create_tool_wrapper(registered_tool)

    def list_tools(self) -> list[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._registered_tools.keys())
