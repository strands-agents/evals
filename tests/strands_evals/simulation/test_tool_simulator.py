"""Tests for ToolSimulator class."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands_evals.simulation.tool_simulator import ToolSimulator
from strands_evals.types.simulation.tool import (
    FailureConditions,
    RegisteredTool,
    StateRegistry,
    ToolOverrideConfig,
    ToolType,
)


@pytest.fixture
def sample_failure_conditions():
    """Fixture providing sample failure conditions."""
    return FailureConditions(
        enabled=True,
        error_rate=0.5,
        error_type="timeout_error",
        error_message="Operation timed out",
    )


@pytest.fixture
def sample_tool_override_config(sample_failure_conditions):
    """Fixture providing sample tool override configuration."""
    return ToolOverrideConfig(
        failure_conditions=sample_failure_conditions,
        scenario_config={"test_key": "test_value"},
    )


@pytest.fixture
def sample_scenario():
    """Fixture providing a sample scenario dictionary."""
    return {
        "name": "Banking Simulation",
        "description": "Test scenario for banking operations with account balance checks",
        "metadata": {"domain": "finance", "complexity": "medium"},
    }


@pytest.fixture
def mock_model():
    """Fixture providing a mock model for testing."""
    mock = MagicMock()
    
    # Mock the async generator for model.generate()
    async def mock_generate(messages, system_prompt=None):
        # Simulate streaming response
        yield {
            "contentBlockDelta": {
                "text": '{"result": "mocked response"}'
            }
        }
    
    mock.generate = mock_generate
    return mock


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear tool registry before each test."""
    ToolSimulator.clear_registry()
    yield
    ToolSimulator.clear_registry()


class TestToolSimulatorInitialization:
    """Test cases for ToolSimulator initialization."""

    def test_init_with_defaults(self):
        """Test ToolSimulator initialization with default parameters."""
        simulator = ToolSimulator()
        
        assert simulator.tool_overrides == {}
        assert simulator.simulator_config == {}
        assert simulator.system_prompt_template is None
        assert simulator.model is not None
        assert simulator._state_registry is not None
        assert simulator._active_simulators == {}

    def test_init_with_model_string(self):
        """Test ToolSimulator initialization with model string."""
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        simulator = ToolSimulator(model=model_id)
        
        assert simulator.model is not None
        # The model should be configured with the provided model_id

    def test_init_with_model_object(self, mock_model):
        """Test ToolSimulator initialization with model object."""
        simulator = ToolSimulator(model=mock_model)
        
        assert simulator.model is mock_model

    def test_init_with_tool_overrides(self, sample_tool_override_config):
        """Test ToolSimulator initialization with tool overrides."""
        tool_overrides = {"test_tool": sample_tool_override_config}
        simulator = ToolSimulator(tool_overrides=tool_overrides)
        
        assert simulator.tool_overrides == tool_overrides

    def test_init_with_custom_state_registry(self):
        """Test ToolSimulator initialization with custom state registry."""
        custom_registry = StateRegistry()
        simulator = ToolSimulator(state_registry=custom_registry)
        
        assert simulator._state_registry is custom_registry

    def test_init_with_system_prompt_template(self):
        """Test ToolSimulator initialization with system prompt template."""
        template = "You are a helpful assistant simulating tools."
        simulator = ToolSimulator(system_prompt_template=template)
        
        assert simulator.system_prompt_template == template


class TestToolDecorators:
    """Test cases for tool decorator registration."""

    def test_function_tool_decorator(self):
        """Test function tool decorator registration."""
        @ToolSimulator.function_tool("test_function")
        def sample_function(x: int, y: str) -> dict:
            """A sample function for testing."""
            return {"x": x, "y": y}

        assert "test_function" in ToolSimulator._registered_tools
        registered_tool = ToolSimulator._registered_tools["test_function"]
        assert registered_tool.name == "test_function"
        assert registered_tool.tool_type == ToolType.FUNCTION
        assert registered_tool.function == sample_function

    def test_function_tool_decorator_without_name(self):
        """Test function tool decorator uses function name when no name provided."""
        @ToolSimulator.function_tool()
        def my_test_function():
            """Test function."""
            pass

        assert "my_test_function" in ToolSimulator._registered_tools
        registered_tool = ToolSimulator._registered_tools["my_test_function"]
        assert registered_tool.name == "my_test_function"

    def test_mcp_tool_decorator(self):
        """Test MCP tool decorator registration."""
        schema = {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"]
        }

        @ToolSimulator.mcp_tool("test_mcp", schema=schema)
        def sample_mcp_tool(**params):
            """A sample MCP tool for testing."""
            return {"content": [{"type": "text", "text": f"Result: {params}"}]}

        assert "test_mcp" in ToolSimulator._registered_tools
        registered_tool = ToolSimulator._registered_tools["test_mcp"]
        assert registered_tool.name == "test_mcp"
        assert registered_tool.tool_type == ToolType.MCP
        assert registered_tool.mcp_schema == schema

    def test_mcp_tool_decorator_requires_schema(self):
        """Test MCP tool decorator requires schema parameter."""
        with pytest.raises(ValueError, match="MCP schema is required"):
            @ToolSimulator.mcp_tool("test_mcp")
            def sample_mcp_tool(**params):
                pass

    def test_api_tool_decorator(self):
        """Test API tool decorator registration."""
        @ToolSimulator.api_tool("test_api", path="/test", method="POST")
        def sample_api_tool(**kwargs):
            """A sample API tool for testing."""
            return {"status": 200, "data": kwargs}

        assert "test_api" in ToolSimulator._registered_tools
        registered_tool = ToolSimulator._registered_tools["test_api"]
        assert registered_tool.name == "test_api"
        assert registered_tool.tool_type == ToolType.API
        assert registered_tool.api_path == "/test"
        assert registered_tool.api_method == "POST"

    def test_api_tool_decorator_requires_path(self):
        """Test API tool decorator requires path parameter."""
        with pytest.raises(ValueError, match="API path is required"):
            @ToolSimulator.api_tool("test_api", method="GET")
            def sample_api_tool(**kwargs):
                pass

    def test_api_tool_decorator_requires_method(self):
        """Test API tool decorator requires method parameter."""
        with pytest.raises(ValueError, match="HTTP method is required"):
            @ToolSimulator.api_tool("test_api", path="/test")
            def sample_api_tool(**kwargs):
                pass

    def test_function_tool_with_simulator_kwargs(self):
        """Test function tool decorator with simulator kwargs."""
        @ToolSimulator.function_tool("test_function", share_state_id="shared_state")
        def sample_function():
            pass

        registered_tool = ToolSimulator._registered_tools["test_function"]
        assert registered_tool.simulator_kwargs == {"share_state_id": "shared_state"}


class TestToolSimulation:
    """Test cases for tool simulation functionality."""

    @patch('strands._async.run_async')
    def test_function_tool_simulation(self, mock_run_async):
        """Test function tool simulation."""
        # Setup mock response
        mock_run_async.return_value = '{"result": "simulated response"}'
        
        # Register and create simulator
        @ToolSimulator.function_tool("test_function")
        def test_func(message: str) -> dict:
            """Test function that should be simulated."""
            pass
        
        simulator = ToolSimulator()
        
        # Execute simulated function
        result = simulator.test_function("Hello, world!")
        
        assert result == {"result": "simulated response"}
        mock_run_async.assert_called_once()

    @patch('strands._async.run_async')
    def test_mcp_tool_simulation(self, mock_run_async):
        """Test MCP tool simulation."""
        # Setup mock response
        mock_run_async.return_value = '{"content": [{"type": "text", "text": "MCP response"}]}'
        
        # Register and create simulator
        schema = {"type": "object", "properties": {"param": {"type": "string"}}}
        @ToolSimulator.mcp_tool("test_mcp", schema=schema)
        def test_mcp(**params):
            """Test MCP tool that should be simulated."""
            pass
        
        simulator = ToolSimulator()
        
        # Execute simulated MCP tool
        result = simulator.test_mcp(param="test_value")
        
        assert result == {"content": [{"type": "text", "text": "MCP response"}]}
        mock_run_async.assert_called_once()

    @patch('strands._async.run_async')
    def test_api_tool_simulation(self, mock_run_async):
        """Test API tool simulation."""
        # Setup mock response
        mock_run_async.return_value = '{"status": 200, "data": {"key": "value"}}'
        
        # Register and create simulator
        @ToolSimulator.api_tool("test_api", path="/test", method="POST")
        def test_api(**kwargs):
            """Test API tool that should be simulated."""
            pass
        
        simulator = ToolSimulator()
        
        # Execute simulated API tool
        result = simulator.test_api(key="value")
        
        assert result == {"status": 200, "data": {"key": "value"}}
        mock_run_async.assert_called_once()

    def test_implemented_function_uses_real_implementation(self):
        """Test that functions with real implementations are not simulated."""
        @ToolSimulator.function_tool("implemented_function")
        def real_function(x: int) -> dict:
            """A function with real implementation."""
            return {"doubled": x * 2}
        
        simulator = ToolSimulator()
        result = simulator.implemented_function(5)
        
        assert result == {"doubled": 10}

    def test_failure_conditions_trigger_error(self):
        """Test that failure conditions trigger errors as expected."""
        # Register tool
        @ToolSimulator.function_tool("failing_function")
        def test_func():
            pass
        
        # Create failure conditions with 100% error rate
        failure_conditions = FailureConditions(
            enabled=True,
            error_rate=1.0,
            error_type="timeout_error",
            error_message="Simulated timeout"
        )
        tool_overrides = {
            "failing_function": ToolOverrideConfig(failure_conditions=failure_conditions)
        }
        
        simulator = ToolSimulator(tool_overrides=tool_overrides)
        
        # Function should return error due to failure conditions
        result = simulator.failing_function()
        
        assert result["status"] == "error"
        assert result["error_type"] == "timeout_error"
        assert result["message"] == "Simulated timeout"


class TestToolRetrieval:
    """Test cases for tool retrieval and listing."""

    def test_list_tools(self):
        """Test listing registered tools."""
        @ToolSimulator.function_tool("func1")
        def func1():
            pass

        @ToolSimulator.function_tool("func2")  
        def func2():
            pass

        simulator = ToolSimulator()
        tools = simulator.list_tools()
        
        assert set(tools) == {"func1", "func2"}

    def test_get_tool(self):
        """Test getting tool by name."""
        @ToolSimulator.function_tool("test_function")
        def test_func():
            return {"test": "result"}
        
        simulator = ToolSimulator()
        tool = simulator.get_tool("test_function")
        
        assert tool is not None
        assert callable(tool)

    def test_get_nonexistent_tool(self):
        """Test getting non-existent tool returns None."""
        simulator = ToolSimulator()
        tool = simulator.get_tool("nonexistent_tool")
        
        assert tool is None

    def test_tool_attribute_access(self):
        """Test accessing tools as attributes."""
        @ToolSimulator.function_tool("test_function")
        def test_func():
            return {"test": "result"}
        
        simulator = ToolSimulator()
        
        # Should be able to access as attribute
        assert hasattr(simulator, "test_function")
        tool = simulator.test_function
        assert callable(tool)

    def test_nonexistent_tool_attribute_raises_error(self):
        """Test accessing non-existent tool as attribute raises AttributeError."""
        simulator = ToolSimulator()
        
        with pytest.raises(AttributeError, match="Tool 'nonexistent' not found"):
            _ = simulator.nonexistent


class TestFactoryMethods:
    """Test cases for factory methods."""

    @patch('strands._async.run_async')
    def test_from_scenario_for_tool_simulator(self, mock_run_async, sample_scenario):
        """Test factory method creates simulator from scenario."""
        # Register a test tool first
        @ToolSimulator.function_tool("account_balance_check")
        def check_balance(account_id: str) -> dict:
            """Check account balance."""
            pass
        
        # Mock LLM response for override generation
        mock_override_response = {
            "tool_overrides": [{
                "tool_name": "account_balance_check",
                "should_simulate": True,
                "failure_conditions": {
                    "enabled": False,
                    "error_rate": 0.0
                }
            }]
        }
        mock_run_async.return_value = json.dumps(mock_override_response)
        
        simulator = ToolSimulator.from_scenario_for_tool_simulator(
            scenario_dict=sample_scenario,
            system_prompt_template="Test template",
            model="test-model"
        )
        
        assert simulator is not None
        assert simulator.system_prompt_template == "Test template"
        mock_run_async.assert_called_once()

    @patch('strands._async.run_async')
    def test_generate_override_from_scenario(self, mock_run_async, sample_scenario):
        """Test override generation from scenario."""
        # Register test tools
        @ToolSimulator.function_tool("test_function")
        def test_func(param: str) -> dict:
            """Test function."""
            pass
        
        # Mock LLM response
        mock_response = {
            "tool_overrides": [{
                "tool_name": "test_function",
                "should_simulate": True,
                "failure_conditions": {
                    "enabled": True,
                    "error_rate": 0.1,
                    "error_type": "network_error",
                    "error_message": "Network timeout"
                }
            }]
        }
        mock_run_async.return_value = json.dumps(mock_response)
        
        overrides = ToolSimulator._generate_override_from_scenario(sample_scenario)
        
        assert "test_function" in overrides
        override = overrides["test_function"]
        assert override.failure_conditions.enabled is True
        assert override.failure_conditions.error_rate == 0.1
        assert override.failure_conditions.error_type == "network_error"
        mock_run_async.assert_called_once()

    def test_generate_override_with_no_tools(self, sample_scenario):
        """Test override generation with no registered tools."""
        # Clear registry to ensure no tools
        ToolSimulator.clear_registry()
        
        overrides = ToolSimulator._generate_override_from_scenario(sample_scenario)
        
        assert overrides == {}

    @patch('strands._async.run_async')
    def test_generate_override_handles_llm_error(self, mock_run_async, sample_scenario):
        """Test override generation handles LLM errors gracefully."""
        # Register a test tool
        @ToolSimulator.function_tool("test_function")
        def test_func():
            pass
        
        # Mock LLM to return invalid JSON
        mock_run_async.return_value = "invalid json response"
        
        overrides = ToolSimulator._generate_override_from_scenario(sample_scenario)
        
        # Should return empty dict on error
        assert overrides == {}


class TestStateRegistry:
    """Test cases for StateRegistry functionality."""

    def test_state_registry_creation(self):
        """Test StateRegistry is created properly."""
        registry = StateRegistry()
        
        assert registry is not None
        assert registry._states == {}

    def test_record_function_call(self):
        """Test recording function call in state registry."""
        registry = StateRegistry()
        
        registry.record_function_call(
            tool_name="test_tool",
            state_key="test_state",
            parameters={"param": "value"},
            response_data={"result": "success"}
        )
        
        state = registry.get_state("test_state")
        assert "function_calls" in state
        assert len(state["function_calls"]) == 1
        call = state["function_calls"][0]
        assert call["tool_name"] == "test_tool"
        assert call["parameters"] == {"param": "value"}
        assert call["response"] == {"result": "success"}

    def test_record_mcp_tool_call(self):
        """Test recording MCP tool call in state registry."""
        registry = StateRegistry()
        
        registry.record_mcp_tool_call(
            tool_name="mcp_tool",
            state_key="mcp_state",
            input_mcp_payload={"input": "data"},
            response_data={"content": [{"type": "text", "text": "result"}]}
        )
        
        state = registry.get_state("mcp_state")
        assert "mcp_calls" in state
        assert len(state["mcp_calls"]) == 1
        call = state["mcp_calls"][0]
        assert call["tool_name"] == "mcp_tool"
        assert call["input"] == {"input": "data"}

    def test_record_api_call(self):
        """Test recording API call in state registry."""
        registry = StateRegistry()
        
        registry.record_api_call(
            tool_name="api_tool",
            state_key="api_state",
            path="/test",
            method="POST",
            input_data={"data": "test"},
            response={"status": 200}
        )
        
        state = registry.get_state("api_state")
        assert "api_calls" in state
        assert len(state["api_calls"]) == 1
        call = state["api_calls"][0]
        assert call["tool_name"] == "api_tool"
        assert call["path"] == "/test"
        assert call["method"] == "POST"


class TestErrorHandling:
    """Test cases for error handling."""

    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response."""
        simulator = ToolSimulator()
        
        response = simulator._parse_llm_response('{"key": "value"}')
        
        assert response == {"key": "value"}

    def test_parse_llm_response_json_in_code_block(self):
        """Test parsing JSON from code blocks."""
        simulator = ToolSimulator()
        
        llm_text = '```json\n{"key": "value"}\n```'
        response = simulator._parse_llm_response(llm_text)
        
        assert response == {"key": "value"}

    def test_parse_llm_response_invalid_json_fallback(self):
        """Test fallback for invalid JSON."""
        simulator = ToolSimulator()
        
        response = simulator._parse_llm_response("This is not JSON")
        
        assert response == {"result": "This is not JSON"}

    def test_create_error_response(self):
        """Test error response creation."""
        simulator = ToolSimulator()
        
        error = simulator._create_error_response("test_error", "Test message", 400)
        
        assert error["status"] == 400
        assert error["error"]["type"] == "test_error"
        assert error["error"]["detail"] == "Test message"
        assert error["error"]["title"] == "Bad Request"

    def test_get_error_title(self):
        """Test error title mapping."""
        simulator = ToolSimulator()
        
        assert simulator._get_error_title(400) == "Bad Request"
        assert simulator._get_error_title(404) == "Not Found"
        assert simulator._get_error_title(500) == "Internal Server Error"
        assert simulator._get_error_title(999) == "Error"  # Unknown status code


class TestRegistryManagement:
    """Test cases for registry management."""

    def test_clear_registry(self):
        """Test clearing tool registry."""
        @ToolSimulator.function_tool("test_function")
        def test_func():
            pass
        
        assert len(ToolSimulator._registered_tools) == 1
        
        ToolSimulator.clear_registry()
        
        assert len(ToolSimulator._registered_tools) == 0
        assert ToolSimulator._state_registry is None

    def test_function_has_implementation_detection(self):
        """Test detection of function implementation."""
        simulator = ToolSimulator()
        
        # Empty function should be detected as not implemented
        def empty_func():
            pass
        
        def implemented_func():
            return {"result": "value"}
        
        assert not simulator._function_has_implementation(empty_func)
        assert simulator._function_has_implementation(implemented_func)

    def test_function_has_implementation_error_handling(self):
        """Test function implementation detection handles errors."""
        simulator = ToolSimulator()
        
        # Create a mock function that will cause dis.get_instructions to fail
        mock_func = MagicMock()
        mock_func.__code__ = None
        
        # Should assume implemented on error
        result = simulator._function_has_implementation(mock_func)
        assert result is True
