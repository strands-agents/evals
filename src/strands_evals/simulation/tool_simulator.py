import inspect
import json
import logging
import random
from typing import Any, Callable, Dict, List, Optional

from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.models.model import Model

from strands_evals.case import Case
from strands_evals.types.simulation.tool import (
    FailureConditions,
    RegisteredTool, 
    StateRegistry,
    ToolOverrideConfig,
    ToolType,
)

logger = logging.getLogger(__name__)


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
        tool_overrides: Optional[Dict[str, ToolOverrideConfig]] = None,
        state_registry: Optional[StateRegistry] = None,
        system_prompt_template: Optional[str] = None,
        model: Model | str | None = None,
    ):
        """
        Initialize a ToolSimulator instance.

        Args:
            tool_overrides: Dictionary mapping tool names to ToolOverrideConfig instances
            state_registry: Registry for maintaining tool state
            system_prompt_template: Template for system prompts
            model: Provider for running inference or a string representing the model-id for Bedrock to use
        """
        self.tool_overrides = tool_overrides or {}
        self.system_prompt_template = system_prompt_template
        
        # Initialize model following Agent pattern
        self.model = BedrockModel() if not model else BedrockModel(model_id=model) if isinstance(model, str) else model

        # Set up state registry
        if state_registry:
            self._state_registry = state_registry
        elif self._state_registry is None:
            self._state_registry = StateRegistry()

        # Initialize shared states from registered tools
        self._initialize_shared_states()

    def _function_has_implementation(self, func: Callable) -> bool:
        """Check if a function has actual implementation or is just an empty stub."""
        try:
            import dis
            # Get function bytecode
            bytecode = list(dis.get_instructions(func))

            # Check if function only contains simple return patterns
            if len(bytecode) <= 3:
                load_const_none_count = sum(
                    1 for instr in bytecode if instr.opname == "LOAD_CONST" and instr.argval is None
                )
                return_count = sum(1 for instr in bytecode if instr.opname == "RETURN_VALUE")

                if load_const_none_count >= 1 and return_count == 1 and len(bytecode) <= 3:
                    return False

            return True
        except Exception:
            # If we can't analyze bytecode, assume it's implemented
            return True

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
        
        # Handle tool behavior configuration
        tool_override = input_data.get("tool_override", {})
        
        # Check for failure conditions
        failure_conditions = tool_override.get("failure_conditions", {})
        if failure_conditions and failure_conditions.get("enabled", False):
            error_rate = failure_conditions.get("error_rate", 0.0)
            if random.random() < error_rate:
                error_type = failure_conditions.get("error_type", "execution_error")
                error_message = failure_conditions.get("error_message", "An error occurred")
                
                if tool_type == ToolType.API:
                    return self._create_error_response(error_type, error_message)
                elif tool_type in [ToolType.FUNCTION, ToolType.MCP]:
                    return {
                        "status": "error",
                        "error_type": error_type,
                        "message": error_message
                    }
        
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
            from .prompt_templates.tool_response_generation import FUNCTION_TOOL_RESPONSE_GENERATION_PROMPT
            
            # Get initial state description from state registry
            current_state = self._state_registry.get_state(state_key)
            initial_state_description = current_state.get("initial_state", "No initial state provided.")
            
            prompt = FUNCTION_TOOL_RESPONSE_GENERATION_PROMPT.format(
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
            from .prompt_templates.tool_response_generation import MCP_TOOL_RESPONSE_GENERATION_PROMPT
            
            # Get initial state description from state registry
            current_state = self._state_registry.get_state(state_key)
            initial_state_description = current_state.get("initial_state", "No initial state provided.")
            
            prompt = MCP_TOOL_RESPONSE_GENERATION_PROMPT.format(
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
            from .prompt_templates.tool_response_generation import API_TOOL_RESPONSE_GENERATION_PROMPT
            
            # Get initial state description from state registry
            current_state = self._state_registry.get_state(state_key)
            initial_state_description = current_state.get("initial_state", "No initial state provided.")
            
            prompt = API_TOOL_RESPONSE_GENERATION_PROMPT.format(
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

    @classmethod
    def from_case_for_tool_simulator(
        cls,
        case: Case,
        system_prompt_template: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> "ToolSimulator":
        """
        Create a ToolSimulator instance configured for a specific case.

        Args:
            case: Case object containing test case information and metadata
            system_prompt_template: Template for system prompts
            model: Model identifier for LLM-based simulation
            **kwargs: Additional configuration options

        Returns:
            Configured ToolSimulator instance
        """
        tool_overrides = cls._generate_override_from_case(case)
        return cls(
            tool_overrides=tool_overrides,
            system_prompt_template=system_prompt_template,
            model=model,
            **kwargs,
        )

    @staticmethod
    def _generate_override_from_case(case: Case) -> Dict[str, ToolOverrideConfig]:
        """Generate tool override configuration from a case using LLM."""
        # Extract scenario description from case
        scenario_description = f"Test case: {case.name or 'unnamed'}. Input: {case.input}"
        if case.metadata:
            scenario_description += f". Metadata: {case.metadata}"

        # Create tools list from registered tools
        tools_list = []
        for tool_name, registered_tool in ToolSimulator._registered_tools.items():
            tool_info = {
                "name": tool_name,
                "type": registered_tool.tool_type.value,
                "description": (
                    getattr(registered_tool.function, "__doc__", "")
                    if registered_tool.function
                    else ""
                ),
            }

            # Add schema information based on tool type
            if registered_tool.tool_type == ToolType.FUNCTION and registered_tool.function:
                sig = inspect.signature(registered_tool.function)
                parameters = {}
                for param_name, param in sig.parameters.items():
                    param_type = "string"
                    if param.annotation != inspect.Parameter.empty:
                        type_map = {
                            int: "integer",
                            float: "number",
                            bool: "boolean",
                            list: "array",
                            dict: "object",
                            str: "string",
                        }
                        param_type = type_map.get(param.annotation, "string")

                    parameters[param_name] = {
                        "type": param_type,
                        "required": param.default == inspect.Parameter.empty,
                    }

                tool_info["parameters"] = parameters

            elif registered_tool.tool_type == ToolType.MCP and registered_tool.mcp_schema:
                tool_info["schema"] = registered_tool.mcp_schema

            elif registered_tool.tool_type == ToolType.API:
                tool_info["path"] = registered_tool.api_path
                tool_info["method"] = registered_tool.api_method

            tools_list.append(tool_info)

        # If no registered tools, return empty override
        if not tools_list:
            logger.warning("No registered tools found for override generation")
            return {}

        # Generate overrides using LLM prompt
        try:
            tools_json = json.dumps(tools_list, indent=2)

            # Use the tool override generation prompt
            from .prompt_templates.tool_override_generation import TOOL_OVERRIDE_GENERATION_PROMPT
            
            prompt = TOOL_OVERRIDE_GENERATION_PROMPT.format(
                scenario=scenario_description,
                tools_json=tools_json,
            )

            # Generate response
            agent = Agent(callback_handler=None)
            result = agent(prompt)
            llm_response = str(result)

            # Parse LLM response
            try:
                response_data = json.loads(llm_response.strip())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw LLM response: {llm_response}")
                return {}

            # Convert LLM response to ToolOverrideConfig instances
            tool_configs: Dict[str, ToolOverrideConfig] = {}
            tool_overrides = response_data.get("tool_overrides", [])

            for override in tool_overrides:
                tool_name = override.get("tool_name")

                if not tool_name:
                    continue

                # Add failure conditions using new schema format
                failure_conditions = override.get("failure_conditions", {})
                failure_conditions = {
                    "enabled": failure_conditions.get("enabled", False),
                    "error_rate": failure_conditions.get("error_rate", 0.0),
                    "error_type": failure_conditions.get("error_type", "execution_error"),
                    "error_message": failure_conditions.get("error_message"),
                }

                try:
                    # Create FailureConditions instance
                    failure_conditions_instance = FailureConditions(**failure_conditions)

                    # Create ToolOverrideConfig instance
                    tool_configs[tool_name] = ToolOverrideConfig(
                        failure_conditions=failure_conditions_instance,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create ToolOverrideConfig for {tool_name}: {e}")
                    continue

            logger.info(f"Generated overrides for {len(tool_configs)} tools using LLM")
            return tool_configs

        except Exception as e:
            logger.error(f"Error generating overrides using LLM: {e}")
            logger.warning("Falling back to empty override configuration")
            return {}

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
                
                # Get tool override configuration
                tool_override_config = {}
                if registered_tool.name in self.tool_overrides:
                    override_config = self.tool_overrides[registered_tool.name]
                    if override_config.failure_conditions:
                        tool_override_config["failure_conditions"] = override_config.failure_conditions.model_dump()
                else:
                    tool_override_config["failure_conditions"] = {"enabled": False}
                
                input_data = {
                    "tool_name": registered_tool.name,
                    "parameters": parameters_string,
                    "tool_override": tool_override_config,
                }
                
            elif registered_tool.tool_type == ToolType.MCP:
                # Get tool override configuration
                tool_override_config = {}
                if registered_tool.name in self.tool_overrides:
                    override_config = self.tool_overrides[registered_tool.name]
                    if override_config.failure_conditions:
                        tool_override_config["failure_conditions"] = override_config.failure_conditions.model_dump()
                else:
                    tool_override_config["failure_conditions"] = {"enabled": False}
                
                input_data = {
                    "tool_name": registered_tool.name,
                    "input_mcp_payload": kwargs,
                    "tool_override": tool_override_config,
                }
                
            elif registered_tool.tool_type == ToolType.API:
                # Get tool override configuration
                tool_override_config = {}
                if registered_tool.name in self.tool_overrides:
                    override_config = self.tool_overrides[registered_tool.name]
                    if override_config.failure_conditions:
                        tool_override_config["failure_conditions"] = override_config.failure_conditions.model_dump()
                else:
                    tool_override_config["failure_conditions"] = {"enabled": False}
                
                input_data = {
                    "tool_name": registered_tool.name,
                    "user_input_api_payload": kwargs,
                    "path": registered_tool.api_path or "",
                    "method": registered_tool.api_method or "GET",
                    "tool_override": tool_override_config,
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
