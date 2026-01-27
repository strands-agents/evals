import inspect
import json
import logging
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from strands.models.bedrock import BedrockModel
from strands.models.model import Model

from strands_evals.types.simulation.tool import (
    RegisteredTool, 
    ToolType,
)

logger = logging.getLogger(__name__)


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
        else:
            warnings.warn(f"State with key '{state_key}' already initialized. Skipping re-initialization.")

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


class ToolSimulator:
    """
    Simulates tool behavior with decorator-based registration system for agent evaluation.

    ToolSimulator provides decorator functions for different tool types and maintains
    a registry of all registered tools. It can be configured to override tool
    behavior for simulation purposes, enabling controlled testing scenarios.

    Attributes:
        tool_overrides: Dictionary mapping tool names to override configurations.
        system_prompt_template: Template string for system prompts.
        model: Provider for running inference or model identifier for Bedrock.
        _registered_tools: Class-level registry for all registered tools.
        _state_registry: Registry for maintaining tool state across calls.
    """

    # Class-level registry for all registered tools
    _registered_tools: Dict[str, RegisteredTool] = {}
    _state_registry: Optional[StateRegistry] = None

    def __init__(
        self,
        state_registry: Optional[StateRegistry] = None,
        system_prompt_template: Optional[str] = None,
        function_tool_prompt: Optional[str] = None,
        mcp_tool_prompt: Optional[str] = None,
        api_tool_prompt: Optional[str] = None,
        model: Model | str | None = None,
    ):
        """
        Initialize a ToolSimulator instance.

        Args:
            state_registry: Registry for maintaining tool state
            system_prompt_template: Template for system prompts
            function_tool_prompt: Optional custom prompt for function tool response generation
            mcp_tool_prompt: Optional custom prompt for MCP tool response generation
            api_tool_prompt: Optional custom prompt for API tool response generation
            model: Provider for running inference or a string representing the model-id for Bedrock to use
        """
        self.system_prompt_template = system_prompt_template
        
        # Set custom prompts or use defaults
        if function_tool_prompt is None:
            from .prompt_templates.tool_response_generation import FUNCTION_TOOL_RESPONSE_GENERATION_PROMPT
            self.function_tool_prompt = FUNCTION_TOOL_RESPONSE_GENERATION_PROMPT
        else:
            self.function_tool_prompt = function_tool_prompt
            
        if mcp_tool_prompt is None:
            from .prompt_templates.tool_response_generation import MCP_TOOL_RESPONSE_GENERATION_PROMPT
            self.mcp_tool_prompt = MCP_TOOL_RESPONSE_GENERATION_PROMPT
        else:
            self.mcp_tool_prompt = mcp_tool_prompt
            
        if api_tool_prompt is None:
            from .prompt_templates.tool_response_generation import API_TOOL_RESPONSE_GENERATION_PROMPT
            self.api_tool_prompt = API_TOOL_RESPONSE_GENERATION_PROMPT
        else:
            self.api_tool_prompt = api_tool_prompt
        
        # Initialize model following Agent pattern
        self.model = BedrockModel() if not model else BedrockModel(model_id=model) if isinstance(model, str) else model

        # Set up state registry
        if state_registry:
            self._state_registry = state_registry
        elif self._state_registry is None:
            self._state_registry = StateRegistry()

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
                    registered_tool.initial_state_description, 
                    state_key
                )
                logger.info(f"Initialized state for tool '{tool_name}' with key '{state_key}'")


    def _simulate_tool_call(self, tool_type: ToolType, state_key: str, input_data: Dict[str, Any]) -> Any:
        """Simulate a tool invocation and return the response."""
        tool_name = input_data.get("tool_name", "")
        registered_tool = self._registered_tools.get(tool_name)
        
        if not registered_tool:
            return self._create_error_response("tool_not_found", f"Tool '{tool_name}' not found")
        
        # Handle different simulation modes
        if registered_tool.mode == "static":
            return self._handle_static_mode(registered_tool, tool_type)
        elif registered_tool.mode == "mock":
            return self._handle_mock_mode(registered_tool, input_data, state_key, tool_type)
        elif registered_tool.mode == "dynamic":
            # Route to appropriate handler based on tool type
            if tool_type == ToolType.FUNCTION:
                return self._handle_function_tool(input_data, state_key)
            elif tool_type == ToolType.MCP:
                return self._handle_mcp_tool(input_data, state_key)
            elif tool_type == ToolType.API:
                return self._handle_api_tool(input_data, state_key)
            else:
                return self._create_error_response("unsupported_tool_type", f"Tool type '{tool_type}' not supported")
        else:
            return self._create_error_response("unsupported_mode", f"Simulation mode '{registered_tool.mode}' not supported")
    
    def _handle_function_tool(self, input_data: Dict[str, Any], state_key: str) -> Dict[str, Any]:
        """Handle function tool simulation."""
        tool_name = input_data.get("tool_name", "")
        parameters = input_data.get("parameters", {})
        
        if not tool_name:
            return {"status": "error", "error_type": "missing_tool_name", "message": "Tool name is required"}
        
        # Generate response using LLM
        try:
            # Get initial state description from state registry
            current_state = self._state_registry.get_state(state_key)
            initial_state_description = current_state.get("initial_state", "No initial state provided.")
            
            prompt = self.function_tool_prompt.format(
                tool_name=tool_name,
                parameters=json.dumps(parameters, indent=2) if parameters else "{}",
                initial_state_description=initial_state_description,
                previous_responses=json.dumps(current_state, indent=2) or "{}"
            )
            
            llm_response = self._generate_llm_response(prompt)
            response_data = self._parse_llm_response(llm_response)
            
            # Record the call
            self._state_registry.record_function_call(tool_name, state_key, parameters, response_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating function response: {e}")
            return {"status": "error", "error_type": "generation_error", "message": str(e)}
    
    def _handle_mcp_tool(self, input_data: Dict[str, Any], state_key: str) -> Dict[str, Any]:
        """Handle MCP tool simulation."""
        tool_name = input_data.get("tool_name", "")
        input_mcp_payload = input_data.get("input_mcp_payload", {})
        
        if not tool_name:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "Tool name is required"}]
            }
        
        try:
            # Get initial state description from state registry
            current_state = self._state_registry.get_state(state_key)
            initial_state_description = current_state.get("initial_state", "No initial state provided.")
            
            prompt = self.mcp_tool_prompt.format(
                tool_name=tool_name,
                mcp_payload=json.dumps(input_mcp_payload, indent=2) if input_mcp_payload else "{}",
                initial_state_description=initial_state_description,
                previous_responses=json.dumps(current_state, indent=2) or "{}"
            )
            
            llm_response = self._generate_llm_response(prompt)
            response_data = self._parse_llm_response(llm_response)
            
            # Record the call
            self._state_registry.record_mcp_tool_call(tool_name, state_key, input_mcp_payload, response_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating MCP response: {e}")
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Error generating response: {str(e)}"}]
            }
    
    def _handle_api_tool(self, input_data: Dict[str, Any], state_key: str) -> Dict[str, Any]:
        """Handle API tool simulation."""
        tool_name = input_data.get("tool_name", "")
        user_input_api_payload = input_data.get("user_input_api_payload", {})
        path = input_data.get("path", "")
        method = input_data.get("method", "GET")
        
        if not tool_name:
            return self._create_error_response("missing_tool_name", "Tool name is required", 400)
        
        try:
            # Get initial state description from state registry
            current_state = self._state_registry.get_state(state_key)
            initial_state_description = current_state.get("initial_state", "No initial state provided.")
            
            prompt = self.api_tool_prompt.format(
                tool_name=tool_name,
                path=path,
                method=method,
                api_payload=json.dumps(user_input_api_payload, indent=2) if user_input_api_payload else "{}",
                initial_state_description=initial_state_description,
                previous_responses=json.dumps(current_state, indent=2) or "{}"
            )
            
            llm_response = self._generate_llm_response(prompt)
            response_data = self._parse_llm_response(llm_response)
            
            # Ensure proper API response format
            if "status" not in response_data:
                response_data = {"status": 200, "data": response_data}
            
            # Record the call
            self._state_registry.record_api_call(tool_name, state_key, path, method, user_input_api_payload, response_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating API response: {e}")
            return self._create_error_response("generation_error", str(e), 500)
    
    def _generate_llm_response(self, prompt: str) -> str:
        """
        Generate LLM response using the model for a given prompt.
        
        Args:
            prompt: The prompt string to send to the LLM
            
        Returns:
            Raw LLM response text
            
        Raises:
            Exception: If LLM generation fails
        """
        try:
            # Create message for model inference
            messages = [{"role": "user", "content": [{"text": prompt}]}]
            
            # Generate response
            llm_response = ""
            for event in self.model.structured_output(str, messages, system_prompt=self.system_prompt_template):
                if hasattr(event, 'get') and event.get("contentBlockDelta"):
                    delta = event["contentBlockDelta"]
                    if "text" in delta:
                        llm_response += delta["text"]
                elif hasattr(event, 'get') and event.get("message"):
                    # Handle final message
                    content = event["message"].get("content", [])
                    for block in content:
                        if "text" in block:
                            llm_response += block["text"]
                elif hasattr(event, 'get') and event.get("output"):
                    # Handle structured output result
                    return str(event["output"])
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise

    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response with fallback handling."""
        try:
            return json.loads(llm_response)
        except json.JSONDecodeError:
            # Try to extract JSON from code blocks
            import re
            json_matches = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', llm_response)
            
            for json_str in json_matches:
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
            
            # Fallback to simple text response
            return {"result": llm_response}
    
    def _create_error_response(self, error_type: str, error_message: str, status_code: int = 400) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "status": status_code,
            "error": {
                "type": error_type,
                "title": self._get_error_title(status_code),
                "detail": error_message
            }
        }
    
    def _get_error_title(self, status_code: int) -> str:
        """Get error title based on status code."""
        error_titles = {
            400: 'Bad Request',
            401: 'Unauthorized',
            403: 'Forbidden',
            404: 'Not Found',
            429: 'Too Many Requests',
            500: 'Internal Server Error',
            503: 'Service Unavailable'
        }
        return error_titles.get(status_code, 'Error')

    def _handle_static_mode(self, registered_tool: RegisteredTool, tool_type: ToolType) -> Dict[str, Any]:
        """Handle static mode simulation - returns predefined static response."""
        if registered_tool.static_response is not None:
            return registered_tool.static_response
        
        # Default static responses for different tool types
        if tool_type == ToolType.FUNCTION:
            return {"status": "success", "result": f"Static response from {registered_tool.name}"}
        elif tool_type == ToolType.MCP:
            return {
                "isError": False,
                "content": [{"type": "text", "text": f"Static response from {registered_tool.name}"}]
            }
        elif tool_type == ToolType.API:
            return {"status": 200, "data": {"message": f"Static response from {registered_tool.name}"}}
        else:
            return {"status": "error", "message": "Unsupported tool type for static mode"}

    def _handle_mock_mode(self, registered_tool: RegisteredTool, input_data: Dict[str, Any], state_key: str, tool_type: ToolType) -> Dict[str, Any]:
        """Handle mock mode simulation - calls custom mock function."""
        if registered_tool.mock_function is not None:
            try:
                # Extract parameters based on tool type
                if tool_type == ToolType.FUNCTION:
                    parameters = input_data.get("parameters", {})
                    if isinstance(parameters, str):
                        parameters = json.loads(parameters)
                    
                    # Call mock function with extracted parameters
                    if "kwargs" in parameters:
                        result = registered_tool.mock_function(**parameters["kwargs"])
                    elif "args" in parameters:
                        result = registered_tool.mock_function(*parameters["args"])
                    else:
                        result = registered_tool.mock_function(**parameters)
                    
                elif tool_type == ToolType.MCP:
                    input_mcp_payload = input_data.get("input_mcp_payload", {})
                    result = registered_tool.mock_function(**input_mcp_payload)
                    
                elif tool_type == ToolType.API:
                    user_input_api_payload = input_data.get("user_input_api_payload", {})
                    result = registered_tool.mock_function(**user_input_api_payload)
                    
                else:
                    return {"status": "error", "message": "Unsupported tool type for mock mode"}
                
                # Record the call in state registry
                tool_name = registered_tool.name
                if tool_type == ToolType.FUNCTION:
                    self._state_registry.record_function_call(tool_name, state_key, parameters, result)
                elif tool_type == ToolType.MCP:
                    self._state_registry.record_mcp_tool_call(tool_name, state_key, input_mcp_payload, result)
                elif tool_type == ToolType.API:
                    path = input_data.get("path", "")
                    method = input_data.get("method", "GET")
                    self._state_registry.record_api_call(tool_name, state_key, path, method, user_input_api_payload, result)
                
                return result
                
            except Exception as e:
                logger.error(f"Error calling mock function for {registered_tool.name}: {e}")
                if tool_type == ToolType.API:
                    return self._create_error_response("mock_error", str(e), 500)
                else:
                    return {"status": "error", "error_type": "mock_error", "message": str(e)}
        
        # Fallback to static mode if no mock function provided
        logger.warning(f"No mock function provided for {registered_tool.name}, falling back to static mode")
        return self._handle_static_mode(registered_tool, tool_type)

    @classmethod
    def function_tool(
        cls, 
        name: Optional[str] = None, 
        initial_state_description: Optional[str] = None, 
        mode: str = "dynamic",
        static_response: Optional[Dict[str, Any]] = None,
        mock_function: Optional[Callable] = None,
        **simulator_kwargs
    ) -> Callable:
        """
        Decorator for registering Python function tools.

        Args:
            name: Optional name for the tool. If None, uses function.__name__
            initial_state_description: Optional initial state description for the tool's context
            mode: Simulation mode - "dynamic", "static", or "mock"
            static_response: Static response dict for static mode
            mock_function: Custom callable for mock mode
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
                    mode=mode,
                    static_response=static_response,
                    mock_function=mock_function,
                )
                cls._registered_tools[tool_name] = registered_tool

                logger.info(f"Registered function tool: {tool_name}")

            except Exception as e:
                logger.error(f"Error registering function tool {name or func.__name__}: {e}")
                raise

            return func

        return decorator

    @classmethod
    def mcp_tool(
        cls, 
        name: Optional[str] = None, 
        schema: Optional[Dict[str, Any]] = None, 
        initial_state_description: Optional[str] = None,
        mode: str = "dynamic",
        static_response: Optional[Dict[str, Any]] = None,
        mock_function: Optional[Callable] = None,
        **simulator_kwargs
    ) -> Callable:
        """
        Decorator for registering MCP (Model Context Protocol) tools.

        Args:
            name: Optional name for the tool. If None, uses function.__name__
            schema: MCP tool schema dictionary
            initial_state_description: Optional initial state description for the tool's context
            mode: Simulation mode - "dynamic", "static", or "mock"
            static_response: Static response dict for static mode
            mock_function: Custom callable for mock mode
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
                mode=mode,
                static_response=static_response,
                mock_function=mock_function,
            )
            cls._registered_tools[tool_name] = registered_tool

            logger.info(f"Registered MCP tool: {tool_name}")
            return func

        return decorator

    @classmethod
    def api_tool(
        cls,
        name: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        initial_state_description: Optional[str] = None,
        mode: str = "dynamic",
        static_response: Optional[Dict[str, Any]] = None,
        mock_function: Optional[Callable] = None,
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
            mode: Simulation mode - "dynamic", "static", or "mock"
            static_response: Static response dict for static mode
            mock_function: Custom callable for mock mode
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
                mode=mode,
                static_response=static_response,
                mock_function=mock_function,
            )
            cls._registered_tools[tool_name] = registered_tool

            logger.info(f"Registered API tool: {tool_name}")
            return func

        return decorator


    def get_tool(self, tool_name: str) -> Optional[Callable]:
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

    def _create_tool_wrapper(self, registered_tool: RegisteredTool) -> Callable:
        """Create a wrapper function for direct tool access."""
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
                    json.dumps({"args": args, "kwargs": kwargs}, indent=2)
                    if args
                    else json.dumps(kwargs, indent=2)
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
            
            return self._simulate_tool_call(registered_tool.tool_type, state_key, input_data)
        
        # Copy function metadata
        if registered_tool.function:
            wrapper.__name__ = registered_tool.function.__name__
            try:
                wrapper.__signature__ = inspect.signature(registered_tool.function)  # type: ignore
            except (ValueError, TypeError):
                pass
            wrapper.__doc__ = registered_tool.function.__doc__
        else:
            wrapper.__name__ = registered_tool.name
            
        return wrapper

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._registered_tools.keys())

    @classmethod
    def clear_registry(cls):
        """Clear all registered tools. Useful for testing."""
        cls._registered_tools.clear()
        cls._state_registry = None
        logger.info("Cleared tool registry")

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
