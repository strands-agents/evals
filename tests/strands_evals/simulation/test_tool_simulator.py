"""Tests for ToolSimulator class."""

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent, tool

from strands_evals.case import Case
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
def sample_case():
    """Fixture providing a sample test case."""
    return Case(
        input="I want to test tool simulation",
        metadata={"task_description": "Complete tool simulation test"},
    )


@pytest.fixture
def mock_model():
    """Fixture providing a mock model for testing."""
    mock = MagicMock()
    
    # Mock the structured_output method
    def mock_structured_output(output_type, messages, system_prompt=None):
        # Simulate streaming response
        yield {
            "contentBlockDelta": {
                "text": '{"result": "mocked response"}'
            }
        }
    
    mock.structured_output = mock_structured_output
    return mock


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear tool registry before each test."""
    ToolSimulator.clear_registry()
    yield
    ToolSimulator.clear_registry()


def test_tool_simulator_init(sample_tool_override_config):
    """Test ToolSimulator initialization with all parameters."""
    custom_registry = StateRegistry()
    tool_overrides = {"test_tool": sample_tool_override_config}
    template = "You are a helpful assistant simulating tools."
    
    simulator = ToolSimulator(
        tool_overrides=tool_overrides,
        state_registry=custom_registry,
        system_prompt_template=template,
        model=None,
    )
    
    assert simulator.tool_overrides == tool_overrides
    assert simulator._state_registry is custom_registry
    assert simulator.system_prompt_template == template
    assert simulator.model is not None


def test_function_tool_decorator_registration():
    """Test function tool decorator registration."""
    @ToolSimulator.function_tool()
    def test_function(x: int, y: str) -> dict:
        """A sample function for testing."""
        return {"x": x, "y": y}

    assert "test_function" in ToolSimulator._registered_tools
    registered_tool = ToolSimulator._registered_tools["test_function"]
    assert registered_tool.name == "test_function"
    assert registered_tool.tool_type == ToolType.FUNCTION
    assert registered_tool.function == test_function


def test_mcp_tool_decorator_registration():
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


def test_api_tool_decorator_registration():
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


def test_function_tool_simulation(mock_model):
    """Test function tool simulation."""
    # Register and create simulator with mock model
    @ToolSimulator.function_tool("test_function")
    def test_func(message: str) -> dict:
        """Test function that should be simulated."""
        pass
    
    # Mock the structured_output method to return expected JSON
    def mock_structured_output(output_type, messages, system_prompt=None):
        yield {"output": '{"result": "simulated response"}'}
    
    mock_model.structured_output = mock_structured_output
    simulator = ToolSimulator(model=mock_model)
    
    # Execute simulated function
    result = simulator.test_function("Hello, world!")
    
    assert result == {"result": "simulated response"}


def test_mcp_tool_simulation(mock_model):
    """Test MCP tool simulation."""
    # Register and create simulator with mock model
    schema = {"type": "object", "properties": {"param": {"type": "string"}}}
    @ToolSimulator.mcp_tool("test_mcp", schema=schema)
    def test_mcp(**params):
        """Test MCP tool that should be simulated."""
        pass
    
    # Mock the structured_output method to return expected JSON
    def mock_structured_output(output_type, messages, system_prompt=None):
        yield {"output": '{"content": [{"type": "text", "text": "MCP response"}]}'}
    
    mock_model.structured_output = mock_structured_output
    simulator = ToolSimulator(model=mock_model)
    
    # Execute simulated MCP tool
    result = simulator.test_mcp(param="test_value")
    
    assert result == {"content": [{"type": "text", "text": "MCP response"}]}


def test_api_tool_simulation(mock_model):
    """Test API tool simulation."""
    # Register and create simulator with mock model
    @ToolSimulator.api_tool("test_api", path="/test", method="POST")
    def test_api(**kwargs):
        """Test API tool that should be simulated."""
        pass
    
    # Mock the structured_output method to return expected JSON
    def mock_structured_output(output_type, messages, system_prompt=None):
        yield {"output": '{"status": 200, "data": {"key": "value"}}'}
    
    mock_model.structured_output = mock_structured_output
    simulator = ToolSimulator(model=mock_model)
    
    # Execute simulated API tool
    result = simulator.test_api(key="value")
    
    assert result == {"status": 200, "data": {"key": "value"}}


def test_failure_conditions_trigger_error():
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


def test_list_tools():
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


@patch("strands_evals.simulation.tool_simulator.Agent")
def test_from_case_for_tool_simulator(mock_agent_class, sample_case):
    """Test factory method creates simulator from case."""
    # Register a test tool first
    @ToolSimulator.function_tool("account_balance_check")
    def check_balance(account_id: str) -> dict:
        """Check account balance."""
        pass
    
    # Mock agent response for override generation
    mock_agent = MagicMock()
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
    mock_agent.return_value = json.dumps(mock_override_response)
    mock_agent_class.return_value = mock_agent
    
    simulator = ToolSimulator.from_case_for_tool_simulator(
        case=sample_case,
        system_prompt_template="Test template",
        model="test-model"
    )
    
    assert simulator is not None
    assert simulator.system_prompt_template == "Test template"


@patch("strands_evals.simulation.tool_simulator.Agent")
def test_generate_override_from_case(mock_agent_class, sample_case):
    """Test override generation from case."""
    # Register test tools
    @ToolSimulator.function_tool("test_function")
    def test_func(param: str) -> dict:
        """Test function."""
        pass
    
    # Mock agent response
    mock_agent = MagicMock()
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
    mock_agent.return_value = json.dumps(mock_response)
    mock_agent_class.return_value = mock_agent
    
    overrides = ToolSimulator._generate_override_from_case(sample_case)
    
    assert "test_function" in overrides
    override = overrides["test_function"]
    assert override.failure_conditions.enabled is True
    assert override.failure_conditions.error_rate == 0.1
    assert override.failure_conditions.error_type == "network_error"


def test_shared_state_registry(mock_model):
    """Test that function, MCP, and API tools can share the same state registry."""
    shared_state_id = "shared_banking_state"
    initial_state = "Initial banking system state with account balances"
    
    # Register three different tools that share the same state
    @ToolSimulator.function_tool(
        "check_balance", 
        initial_state_description=initial_state,
        share_state_id=shared_state_id
    )
    def check_balance(account_id: str):
        """Check account balance."""
        pass
    
    @ToolSimulator.mcp_tool(
        "transfer_funds", 
        schema={"type": "object", "properties": {"from_account": {"type": "string"}, "to_account": {"type": "string"}}},
        initial_state_description=initial_state,
        share_state_id=shared_state_id
    )
    def transfer_funds(**params):
        """Transfer funds between accounts."""
        pass
    
    @ToolSimulator.api_tool(
        "get_transactions",
        path="/transactions", 
        method="GET",
        initial_state_description=initial_state,
        share_state_id=shared_state_id
    )
    def get_transactions(**kwargs):
        """Get transaction history."""
        pass
    
    # Mock responses for each tool type based on call count
    call_count = 0
    def mock_structured_output(output_type, messages, system_prompt=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # First call (check_balance)
            yield {"output": '{"balance": 1000, "currency": "USD"}'}
        elif call_count == 2:  # Second call (transfer_funds)
            yield {"output": '{"content": [{"type": "text", "text": "Transfer completed"}]}'}
        elif call_count == 3:  # Third call (get_transactions)
            yield {"output": '{"status": 200, "data": {"transactions": []}}'}
    
    mock_model.structured_output = mock_structured_output
    simulator = ToolSimulator(model=mock_model)
    
    # Execute each tool in order
    balance_result = simulator.check_balance("12345")
    transfer_result = simulator.transfer_funds(from_account="12345", to_account="67890")
    transactions_result = simulator.get_transactions(account_id="12345")
    
    # Verify results
    assert balance_result == {"balance": 1000, "currency": "USD"}
    assert transfer_result == {"content": [{"type": "text", "text": "Transfer completed"}]}
    assert transactions_result == {"status": 200, "data": {"transactions": []}}
    
    # Verify all tools accessed the same shared state
    shared_state = simulator._state_registry.get_state(shared_state_id)
    assert "initial_state" in shared_state
    assert shared_state["initial_state"] == initial_state
    assert "previous_calls" in shared_state
    assert len(shared_state["previous_calls"]) == 3
    
    # Check that all three tool calls are recorded in the shared state
    tool_names = [call["tool_name"] for call in shared_state["previous_calls"]]
    assert "check_balance" in tool_names
    assert "transfer_funds" in tool_names  
    assert "get_transactions" in tool_names
    
    # Verify each tool type recorded its specific data correctly
    function_call = next(call for call in shared_state["previous_calls"] if call["tool_name"] == "check_balance")
    assert "parameters" in function_call
    
    mcp_call = next(call for call in shared_state["previous_calls"] if call["tool_name"] == "transfer_funds")
    assert "input_mcp_payload" in mcp_call
    
    api_call = next(call for call in shared_state["previous_calls"] if call["tool_name"] == "get_transactions")
    assert "path" in api_call
    assert "method" in api_call


def test_record_function_call():
    """Test recording function call in state registry."""
    registry = StateRegistry()
    
    registry.record_function_call(
        tool_name="test_tool",
        state_key="test_state",
        parameters={"param": "value"},
        response_data={"result": "success"}
    )
    
    state = registry.get_state("test_state")
    assert "previous_calls" in state
    assert len(state["previous_calls"]) == 1
    call = state["previous_calls"][0]
    assert call["tool_name"] == "test_tool"
    assert call["parameters"] == {"param": "value"}
    assert call["response"] == {"result": "success"}


def test_record_mcp_tool_call():
    """Test recording MCP tool call in state registry."""
    registry = StateRegistry()
    
    registry.record_mcp_tool_call(
        tool_name="mcp_tool",
        state_key="mcp_state",
        input_mcp_payload={"input": "data"},
        response_data={"content": [{"type": "text", "text": "result"}]}
    )
    
    state = registry.get_state("mcp_state")
    assert "previous_calls" in state
    assert len(state["previous_calls"]) == 1
    call = state["previous_calls"][0]
    assert call["tool_name"] == "mcp_tool"
    assert call["input_mcp_payload"] == {"input": "data"}


def test_record_api_call():
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
    assert "previous_calls" in state
    assert len(state["previous_calls"]) == 1
    call = state["previous_calls"][0]
    assert call["tool_name"] == "api_tool"
    assert call["path"] == "/test"
    assert call["method"] == "POST"
    assert call["input"] == {"data": "test"}


def test_parse_llm_response_valid_json():
    """Test parsing valid JSON response."""
    simulator = ToolSimulator()
    
    response = simulator._parse_llm_response('{"key": "value"}')
    
    assert response == {"key": "value"}


def test_parse_llm_response_json_in_code_block():
    """Test parsing JSON from code blocks."""
    simulator = ToolSimulator()
    
    llm_text = '```json\n{"key": "value"}\n```'
    response = simulator._parse_llm_response(llm_text)
    
    assert response == {"key": "value"}


def test_parse_llm_response_invalid_json_fallback():
    """Test fallback for invalid JSON."""
    simulator = ToolSimulator()
    
    response = simulator._parse_llm_response("This is not JSON")
    
    assert response == {"result": "This is not JSON"}


def test_create_error_response():
    """Test error response creation."""
    simulator = ToolSimulator()
    
    error = simulator._create_error_response("test_error", "Test message", 400)
    
    assert error["status"] == 400
    assert error["error"]["type"] == "test_error"
    assert error["error"]["detail"] == "Test message"
    assert error["error"]["title"] == "Bad Request"


def test_get_error_title():
    """Test error title mapping."""
    simulator = ToolSimulator()
    
    assert simulator._get_error_title(400) == "Bad Request"
    assert simulator._get_error_title(404) == "Not Found"
    assert simulator._get_error_title(500) == "Internal Server Error"
    assert simulator._get_error_title(999) == "Error"  # Unknown status code


def test_clear_registry():
    """Test clearing tool registry."""
    @ToolSimulator.function_tool("test_function")
    def test_func():
        pass
    
    assert len(ToolSimulator._registered_tools) == 1
    
    ToolSimulator.clear_registry()
    
    assert len(ToolSimulator._registered_tools) == 0
    assert ToolSimulator._state_registry is None


def test_function_has_implementation_detection():
    """Test detection of function implementation."""
    simulator = ToolSimulator()
    
    # Empty function should be detected as not implemented
    def empty_func():
        pass
    
    def implemented_func():
        return {"result": "value"}
    
    assert not simulator._function_has_implementation(empty_func)
    assert simulator._function_has_implementation(implemented_func)


def test_function_tool_decorator_stacking_with_strands_tool():
    """Test function tool decorator stacking with Strands @tool decorator."""
    # Mock function that handles parameters with **kwargs
    def mock_function(**kwargs):
        input_value = kwargs.get("input_value", "")
        return {"result": f"processed {input_value}"}
    
    # Define tool with stacked decorators
    @tool
    @ToolSimulator.function_tool("stacked_function_tool", mode="mock", 
                                 mock_function=mock_function)
    def stacked_function_tool(input_value: str) -> Dict[str, Any]:
        """Test function tool with stacked decorators.
        
        Args:
            input_value: Input parameter for processing
        """
        pass
    
    # Create simulator
    simulator = ToolSimulator()
    
    # Test that the tool is callable and returns expected result
    result = simulator.stacked_function_tool(input_value="test_input")
    assert result == {"result": "processed test_input"}
    
    # Verify the tool is registered in ToolSimulator
    assert "stacked_function_tool" in ToolSimulator._registered_tools
    registered_tool = ToolSimulator._registered_tools["stacked_function_tool"]
    assert registered_tool.tool_type == ToolType.FUNCTION
    assert registered_tool.mode == "mock"
    assert registered_tool.mock_function == mock_function
    
    # Validate Strands tool creation
    assert stacked_function_tool.tool_spec is not None
    spec = stacked_function_tool.tool_spec
    
    # Check basic spec properties
    assert spec["name"] == "stacked_function_tool"
    assert spec["description"] == "Test function tool with stacked decorators."
    
    # Check input schema
    schema = spec["inputSchema"]["json"]
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"input_value"}
    
    # Check parameter properties
    assert schema["properties"]["input_value"]["type"] == "string"
    assert schema["properties"]["input_value"]["description"] == "Input parameter for processing"
    
    # Make sure these are set properly
    assert stacked_function_tool.__wrapped__ is not None
    assert stacked_function_tool.__doc__ == stacked_function_tool._tool_func.__doc__


def test_mcp_tool_decorator_stacking_with_strands_tool():
    """Test MCP tool decorator stacking with Strands @tool decorator."""
    # Mock function for MCP tool
    def mock_mcp_processor(param1, param2=42):
        return {
            "content": [
                {"type": "text", "text": f"MCP processed: {param1} with value {param2}"}
            ],
            "isError": False
        }
    
    schema = {
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
            "param2": {"type": "integer", "default": 42}
        },
        "required": ["param1"]
    }
    
    # Define tool with stacked decorators
    @tool
    @ToolSimulator.mcp_tool("stacked_mcp_tool", schema=schema, mode="mock",
                           mock_function=mock_mcp_processor)
    def stacked_mcp_tool(param1: str, param2: int = 42) -> Dict[str, Any]:
        """Test MCP tool with stacked decorators.
        
        Args:
            param1: First parameter for MCP processing
            param2: Second parameter with default value
        """
        pass
    
    # Create simulator
    simulator = ToolSimulator()
    
    # Test that the tool is callable and returns expected result
    result = simulator.stacked_mcp_tool(param1="test", param2=100)
    expected = {
        "content": [{"type": "text", "text": "MCP processed: test with value 100"}],
        "isError": False
    }
    assert result == expected
    
    # Verify the tool is registered in ToolSimulator
    assert "stacked_mcp_tool" in ToolSimulator._registered_tools
    registered_tool = ToolSimulator._registered_tools["stacked_mcp_tool"]
    assert registered_tool.tool_type == ToolType.MCP
    assert registered_tool.mode == "mock"
    assert registered_tool.mock_function == mock_mcp_processor
    
    # Validate Strands tool creation
    assert stacked_mcp_tool.tool_spec is not None
    spec = stacked_mcp_tool.tool_spec
    
    # Check basic spec properties
    assert spec["name"] == "stacked_mcp_tool"
    assert spec["description"] == "Test MCP tool with stacked decorators."
    
    # Check input schema
    schema = spec["inputSchema"]["json"]
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"param1"}
    
    # Check parameter properties
    assert schema["properties"]["param1"]["type"] == "string"
    assert schema["properties"]["param2"]["type"] == "integer"
    assert schema["properties"]["param1"]["description"] == "First parameter for MCP processing"
    assert schema["properties"]["param2"]["description"] == "Second parameter with default value"
    
    # Make sure these are set properly
    assert stacked_mcp_tool.__wrapped__ is not None
    assert stacked_mcp_tool.__doc__ == stacked_mcp_tool._tool_func.__doc__


def test_api_tool_decorator_stacking_with_strands_tool():
    """Test API tool decorator stacking with Strands @tool decorator."""
    # Static response for API tool
    static_response = {
        "status": 200,
        "data": {
            "message": "API tool working",
            "timestamp": "2024-01-01T12:00:00Z",
            "endpoint": "/test/api"
        }
    }
    
    # Define tool with stacked decorators
    @tool
    @ToolSimulator.api_tool("stacked_api_tool", path="/test/api", method="GET",
                           mode="static", static_response=static_response)
    def stacked_api_tool(query: str = "") -> Dict[str, Any]:
        """Test API tool with stacked decorators.
        
        Args:
            query: Query parameter for API call
        """
        pass
    
    # Create simulator
    simulator = ToolSimulator()
    
    # Test that the tool is callable and returns expected result
    result = simulator.stacked_api_tool(query="test_query")
    assert result == static_response
    
    # Verify the tool is registered in ToolSimulator
    assert "stacked_api_tool" in ToolSimulator._registered_tools
    registered_tool = ToolSimulator._registered_tools["stacked_api_tool"]
    assert registered_tool.tool_type == ToolType.API
    assert registered_tool.mode == "static"
    assert registered_tool.api_path == "/test/api"
    assert registered_tool.api_method == "GET"
    assert registered_tool.static_response == static_response
    
    # Validate Strands tool creation
    assert stacked_api_tool.tool_spec is not None
    spec = stacked_api_tool.tool_spec
    
    # Check basic spec properties
    assert spec["name"] == "stacked_api_tool"
    assert spec["description"] == "Test API tool with stacked decorators."
    
    # Check input schema
    schema = spec["inputSchema"]["json"]
    assert schema["type"] == "object"
    # query parameter is optional, so required list may be empty or missing
    required_fields = set(schema.get("required", []))
    assert required_fields == set()
    
    # Check parameter properties
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["properties"]["query"]["description"] == "Query parameter for API call"
    
    # Make sure these are set properly
    assert stacked_api_tool.__wrapped__ is not None
    assert stacked_api_tool.__doc__ == stacked_api_tool._tool_func.__doc__
