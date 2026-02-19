import json
import logging
import warnings
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel
from strands import Agent
from strands.agent import AgentResult
from strands.models.model import Model
from strands.tools.decorator import DecoratedFunctionTool

from strands_evals.types.simulation.tool import RegisteredTool

from .prompt_templates.tool_response_generation import TOOL_RESPONSE_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class StateRegistry:
    """
    State registry for managing shared state between tool simulators.
    Organized by state_key to isolate state between different tools or shared state groups.

    The registry automatically maintains a bounded cache of tool calls per state key.
    The maximum number of tool calls stored is configurable via max_tool_call_cache_size parameter.

    Attributes:
        max_tool_call_cache_size: Maximum number of tool calls to store per state key.
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
        self.max_tool_call_cache_size = max_tool_call_cache_size
        self._states: defaultdict[str, dict[str, Any]] = defaultdict(
            lambda: {"previous_calls": deque(maxlen=self.max_tool_call_cache_size)}
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
                "previous_calls": deque(maxlen=self.max_tool_call_cache_size),
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
        response_data: Any,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Cache a tool call in the tool's state key.

        Args:
            tool_name: Name of the tool being called.
            state_key: Key for the state (tool_name or share_state_id).
            response_data: Response from the tool call.
            parameters: Function parameters.

        Returns:
            Updated state dictionary.
        """
        # Access the actual state storage (not converted copy)
        state = self._states[state_key]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        call_record = {
            "tool_name": tool_name,
            "response": response_data,
            "timestamp": timestamp,
            "parameters": parameters,
        }

        # Append to deque with automatic FIFO eviction when cache is full
        state["previous_calls"].append(call_record)
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

    ToolSimulator provides a decorator for tools and maintains a registry of all
    registered tools. It can be configured to override tool behavior for simulation purposes,
    enabling controlled testing scenarios.

    IMPORTANT: This simulator expects functions to be decorated with Strands' @tool decorator first.

    Example usage:
        simulator = ToolSimulator()

        @simulator.tool()
        @tool
        def my_tool(param: str) -> dict:
            '''Tool description'''
            pass

    The simulator automatically maintains a bounded cache of tool calls for context.
    The maximum number of tool calls stored per state key is configurable via
    max_tool_call_cache_size parameter (default: 20).

    Attributes:
        state_registry: Registry for maintaining tool state across calls.
        model: Provider for running inference or model identifier for Bedrock.
        max_tool_call_cache_size: Maximum number of tool calls to store per state key.
    """

    def __init__(
        self,
        state_registry: StateRegistry | None = None,
        model: Model | str | None = None,
        max_tool_call_cache_size: int = 20,
    ):
        """
        Initialize a ToolSimulator instance.

        Args:
            state_registry: Registry for maintaining tool state. If not provided,
                           a new StateRegistry will be created with max_tool_call_cache_size.
            model: Provider for running inference or a string representing the model-id for Bedrock to use
            max_tool_call_cache_size: Maximum number of tool calls to store per state key.
                                     Only used when creating a new StateRegistry (ignored if state_registry
                                     is provided). Older calls are automatically evicted when limit is exceeded.
                                     Default is 20.
        """
        self.model = model
        self.tool_prompt = TOOL_RESPONSE_GENERATION_PROMPT
        self.state_registry = state_registry or StateRegistry(max_tool_call_cache_size=max_tool_call_cache_size)
        self._registered_tools: dict[str, RegisteredTool] = {}
        self._initialize_shared_states()

    def _initialize_shared_states(self):
        """Initialize shared states from registered tools' initial descriptions."""
        for tool_name, registered_tool in self._registered_tools.items():
            if registered_tool.initial_state_description:
                state_key = registered_tool.share_state_id or registered_tool.name
                self.state_registry.initialize_state_via_description(
                    registered_tool.initial_state_description, state_key
                )
                logger.info(f"Initialized state for tool '{tool_name}' with key '{state_key}'")

    def _create_tool_wrapper(self, registered_tool: RegisteredTool):
        """
        Create a Strands tool wrapper for simulation.

        Since the registered function is already a DecoratedFunctionTool (from @tool decorator),
        we reuse its existing metadata and spec, but replace the tool_func with our simulation wrapper.
        """
        original_tool = registered_tool.function

        if not isinstance(original_tool, DecoratedFunctionTool):
            raise TypeError(
                f"Expected DecoratedFunctionTool, got {type(original_tool).__name__}. "
                f"Ensure your function is decorated with @tool first."
            )

        def wrapper(*args, **kwargs):
            state_key = registered_tool.share_state_id or registered_tool.name

            parameters_string = (
                json.dumps({"args": args, "kwargs": kwargs}, indent=2) if args else json.dumps(kwargs, indent=2)
            )

            return self._call_tool(registered_tool, parameters_string, state_key)

        if registered_tool.function:
            wrapper.__name__ = registered_tool.function.__name__
            wrapper.__doc__ = registered_tool.function.__doc__
        else:
            wrapper.__name__ = registered_tool.name
            wrapper.__doc__ = f"Simulated {registered_tool.name} tool"

        tool_spec = original_tool.tool_spec.copy()
        tool_spec["name"] = registered_tool.name

        simulated_tool = DecoratedFunctionTool(
            tool_name=registered_tool.name,
            tool_spec=tool_spec,
            tool_func=wrapper,  # Use our simulation wrapper instead of original function
            metadata=original_tool._metadata,  # Reuse existing metadata
        )

        return simulated_tool

    def _simulate_tool_call(self, prompt: str, structured_output_model=None) -> Any:
        """Tool simulation agent creation and response generation."""
        agent = Agent(
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

    def _call_tool(self, registered_tool: RegisteredTool, parameters_string: str, state_key: str) -> dict[str, Any]:
        """Simulate a tool invocation and return the response."""
        # Get input schema from Strands tool decorator
        input_schema_dict = registered_tool.function.tool_spec.get("inputSchema", {}).get("json", {})
        input_schema = json.dumps(input_schema_dict, indent=2)

        # Get output schema as JSON string using Pydantic's model_json_schema (output_schema is mandatory)
        output_schema = registered_tool.output_schema.model_json_schema()
        output_schema_string = json.dumps(output_schema, indent=2)

        current_state = self.state_registry.get_state(state_key)

        prompt = self.tool_prompt.format(
            tool_name=registered_tool.name,
            input_schema=input_schema,
            output_schema=output_schema_string,
            user_payload=parameters_string,
            state_context=json.dumps(current_state, indent=2),
        )

        result = self._simulate_tool_call(prompt, structured_output_model=registered_tool.output_schema)

        response_data = self._parse_simulated_response(result)

        self.state_registry.cache_tool_call(
            registered_tool.name, state_key, response_data, parameters=json.loads(parameters_string)
        )
        return response_data

    def tool(
        self,
        output_schema: type[BaseModel],
        name: str | None = None,
        share_state_id: str | None = None,
        initial_state_description: str | None = None,
    ) -> Callable:
        """
        Decorator for registering tools with flexible output schemas.

        IMPORTANT: This decorator expects the function to already be decorated with @tool
        from strands.tools.decorator.

        Args:
            output_schema: Required pydantic BaseModel for tool's output schema.
            name: Optional name for the tool. If None, uses DecoratedFunctionTool.tool_name
            share_state_id: Optional shared state ID for sharing state between tools
            initial_state_description: Optional initial state description for the tool's context

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            try:
                if not isinstance(func, DecoratedFunctionTool):
                    raise TypeError(
                        f"Expected DecoratedFunctionTool (from @tool decorator), got {type(func).__name__}. "
                        f"Please ensure your function is decorated with @tool first, then @simulator.tool()."
                    )

                tool_name = name or func.tool_name

                registered_tool = RegisteredTool(
                    name=tool_name,
                    function=func,
                    output_schema=output_schema,
                    initial_state_description=initial_state_description,
                    share_state_id=share_state_id,
                )
                self._registered_tools[tool_name] = registered_tool

                if initial_state_description:
                    state_key = share_state_id or tool_name
                    self.state_registry.initialize_state_via_description(initial_state_description, state_key)
                    logger.info(f"Initialized state for tool '{tool_name}' with key '{state_key}'")

                logger.info(f"Registered tool: {tool_name}")

            except Exception as e:
                raise RuntimeError(f"Error registering tool {name or getattr(func, '__name__', 'unknown')}: {e}") from e

            return func

        return decorator

    def __getattr__(self, name: str) -> Any:
        """
        Allow direct access to registered tools as attributes.

        Args:
            name: Tool name

        Returns:
            Tool callable wrapper

        Raises:
            AttributeError: If tool not found
        """
        registered_tool = self._registered_tools.get(name)
        if registered_tool:
            return self._create_tool_wrapper(registered_tool)

        raise AttributeError(f"Tool '{name}' not found in registered tools")

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

    def clear_tools(self):
        """Clear all registered tools for this simulator instance."""
        self._registered_tools.clear()
        logger.info("Cleared tool registry for this simulator instance")

    def get_state(self, state_key: str) -> dict[str, Any]:
        """
        Get state for a specific tool or shared state group.

        Args:
            state_key: Key for the state (tool_name or share_state_id).

        Returns:
            State dictionary containing previous_calls.
        """
        return self.state_registry.get_state(state_key)
