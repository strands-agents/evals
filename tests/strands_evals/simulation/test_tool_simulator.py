"""Tests for ToolSimulator class."""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from strands_evals.case import Case
from strands_evals.simulation.tool_simulator import StateRegistry, ToolSimulator
from strands_evals.types.simulation.tool import ToolType


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
        yield {"contentBlockDelta": {"text": '{"result": "mocked response"}'}}

    mock.structured_output = mock_structured_output
    return mock


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear tool registry before each test."""
    ToolSimulator.clear_registry()
    yield
    ToolSimulator.clear_registry()


def test_tool_simulator_init():
    """Test ToolSimulator initialization with all parameters."""
    custom_registry = StateRegistry()

    simulator = ToolSimulator(
        state_registry=custom_registry,
        model=None,
    )

    assert simulator._state_registry is custom_registry
    assert simulator.model is None  # model is now used for LLM inference
    assert simulator.function_tool_prompt is not None  # Check that prompt templates are loaded
    assert simulator.mcp_tool_prompt is not None
    assert simulator.api_tool_prompt is not None


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
    schema = {"type": "object", "properties": {"param": {"type": "string"}}, "required": ["param"]}

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

    simulator = ToolSimulator(model=mock_model)

    # Mock the Agent constructor and its result to avoid real LLM calls
    mock_agent_instance = MagicMock()
    mock_result = MagicMock()
    # Mock __str__ method to return expected JSON string
    mock_result.__str__ = MagicMock(return_value='{"result": "simulated response"}')
    mock_agent_instance.return_value = mock_result

    with pytest.MonkeyPatch().context() as m:
        # Mock the Agent class constructor
        m.setattr("strands_evals.simulation.tool_simulator.Agent", lambda **kwargs: mock_agent_instance)

        # Execute simulated function
        result = simulator.test_function("Hello, world!")

        assert result == {"result": "simulated response"}
        assert mock_agent_instance.called


def test_mcp_tool_simulation(mock_model):
    """Test MCP tool simulation."""
    # Register and create simulator with mock model
    schema = {"type": "object", "properties": {"param": {"type": "string"}}}

    @ToolSimulator.mcp_tool("test_mcp", schema=schema)
    def test_mcp(**params):
        """Test MCP tool that should be simulated."""
        pass

    simulator = ToolSimulator(model=mock_model)

    # Mock the Agent constructor and its result to avoid real LLM calls
    mock_agent_instance = MagicMock()
    mock_result = MagicMock()
    # Mock __str__ method to return expected JSON string
    mock_result.__str__ = MagicMock(return_value='{"content": [{"type": "text", "text": "MCP response"}]}')
    mock_agent_instance.return_value = mock_result

    with pytest.MonkeyPatch().context() as m:
        # Mock the Agent class constructor
        m.setattr("strands_evals.simulation.tool_simulator.Agent", lambda **kwargs: mock_agent_instance)

        # Execute simulated MCP tool
        result = simulator.test_mcp(param="test_value")

        assert result == {"content": [{"type": "text", "text": "MCP response"}]}
        assert mock_agent_instance.called


def test_api_tool_simulation(mock_model):
    """Test API tool simulation."""

    # Register and create simulator with mock model
    @ToolSimulator.api_tool("test_api", path="/test", method="POST")
    def test_api(**kwargs):
        """Test API tool that should be simulated."""
        pass

    simulator = ToolSimulator(model=mock_model)

    # Mock the Agent constructor and its result to avoid real LLM calls
    mock_agent_instance = MagicMock()
    mock_result = MagicMock()
    # Mock __str__ method to return expected JSON string
    mock_result.__str__ = MagicMock(return_value='{"status": 200, "data": {"key": "value"}}')
    mock_agent_instance.return_value = mock_result

    with pytest.MonkeyPatch().context() as m:
        # Mock the Agent class constructor
        m.setattr("strands_evals.simulation.tool_simulator.Agent", lambda **kwargs: mock_agent_instance)

        # Execute simulated API tool
        result = simulator.test_api(key="value")

        assert result == {"status": 200, "data": {"key": "value"}}
        assert mock_agent_instance.called


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


def test_shared_state_registry(mock_model):
    """Test that function, MCP, and API tools can share the same state registry."""
    shared_state_id = "shared_banking_state"
    initial_state = "Initial banking system state with account balances"

    # Register three different tools that share the same state
    @ToolSimulator.function_tool(
        "check_balance", initial_state_description=initial_state, share_state_id=shared_state_id
    )
    def check_balance(account_id: str):
        """Check account balance."""
        pass

    @ToolSimulator.mcp_tool(
        "transfer_funds",
        schema={"type": "object", "properties": {"from_account": {"type": "string"}, "to_account": {"type": "string"}}},
        initial_state_description=initial_state,
        share_state_id=shared_state_id,
    )
    def transfer_funds(**params):
        """Transfer funds between accounts."""
        pass

    @ToolSimulator.api_tool(
        "get_transactions",
        path="/transactions",
        method="GET",
        initial_state_description=initial_state,
        share_state_id=shared_state_id,
    )
    def get_transactions(**kwargs):
        """Get transaction history."""
        pass

    simulator = ToolSimulator(model=mock_model)

    # Mock the Agent constructor to avoid real LLM calls
    mock_agent_instances = []
    expected_responses = [
        {"balance": 1000, "currency": "USD"},  # Function response
        {"content": [{"type": "text", "text": "Transfer completed"}]},  # MCP response
        {"status": 200, "data": {"transactions": []}},  # API response
    ]

    def create_mock_agent(**kwargs):
        mock_agent = MagicMock()
        mock_result = MagicMock()

        if len(mock_agent_instances) < len(expected_responses):
            response = expected_responses[len(mock_agent_instances)]
            import json

            # Simplified approach: Mock __str__ method to return JSON string for all tool types
            mock_result.__str__ = MagicMock(return_value=json.dumps(response))

        mock_agent.return_value = mock_result
        mock_agent_instances.append(mock_agent)
        return mock_agent

    with pytest.MonkeyPatch().context() as m:
        # Mock the Agent class constructor
        m.setattr("strands_evals.simulation.tool_simulator.Agent", create_mock_agent)

        # Execute each tool in order
        balance_result = simulator.check_balance("12345")
        transfer_result = simulator.transfer_funds(from_account="12345", to_account="67890")
        transactions_result = simulator.get_transactions(account_id="12345")

        # Verify results
        assert balance_result == {"balance": 1000, "currency": "USD"}
        assert transfer_result == {"content": [{"type": "text", "text": "Transfer completed"}]}
        assert transactions_result == {"status": 200, "data": {"transactions": []}}

        # Verify all agents were called
        assert len(mock_agent_instances) == 3
        for agent in mock_agent_instances:
            assert agent.called

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


def test_cache_tool_call_function():
    """Test recording function call in state registry using unified method."""
    registry = StateRegistry()

    registry.cache_tool_call(
        tool_name="test_tool",
        state_key="test_state",
        tool_type=ToolType.FUNCTION,
        response_data={"result": "success"},
        parameters={"param": "value"},
    )

    state = registry.get_state("test_state")
    assert "previous_calls" in state
    assert len(state["previous_calls"]) == 1
    call = state["previous_calls"][0]
    assert call["tool_name"] == "test_tool"
    assert call["tool_type"] == "function"
    assert call["parameters"] == {"param": "value"}
    assert call["response"] == {"result": "success"}


def test_cache_tool_call_mcp():
    """Test recording MCP tool call in state registry using unified method."""
    registry = StateRegistry()

    registry.cache_tool_call(
        tool_name="mcp_tool",
        state_key="mcp_state",
        tool_type=ToolType.MCP,
        response_data={"content": [{"type": "text", "text": "result"}]},
        input_mcp_payload={"input": "data"},
    )

    state = registry.get_state("mcp_state")
    assert "previous_calls" in state
    assert len(state["previous_calls"]) == 1
    call = state["previous_calls"][0]
    assert call["tool_name"] == "mcp_tool"
    assert call["tool_type"] == "mcp"
    assert call["input_mcp_payload"] == {"input": "data"}
    assert call["response"] == {"content": [{"type": "text", "text": "result"}]}


def test_cache_tool_call_api():
    """Test recording API call in state registry using unified method."""
    registry = StateRegistry()

    registry.cache_tool_call(
        tool_name="api_tool",
        state_key="api_state",
        tool_type=ToolType.API,
        response_data={"status": 200},
        path="/test",
        method="POST",
        input_data={"data": "test"},
    )

    state = registry.get_state("api_state")
    assert "previous_calls" in state
    assert len(state["previous_calls"]) == 1
    call = state["previous_calls"][0]
    assert call["tool_name"] == "api_tool"
    assert call["tool_type"] == "api"
    assert call["path"] == "/test"
    assert call["method"] == "POST"
    assert call["input"] == {"data": "test"}
    assert call["response"] == {"status": 200}


def test_tool_not_found_raises_error():
    """Test that accessing non-existent tools raises ValueError."""
    simulator = ToolSimulator()

    # Test that accessing a non-existent tool via __getattr__ raises AttributeError
    with pytest.raises(AttributeError) as excinfo:
        _ = simulator.nonexistent_tool

    assert "not found in registered tools" in str(excinfo.value)


def test_mock_mode_missing_function_raises_error():
    """Test that mock mode raises ValueError when mock_function is missing."""

    # Register a tool without mock_function but with mock mode
    @ToolSimulator.function_tool("test_mock_tool", mode="mock")
    def test_mock_tool():
        pass

    simulator = ToolSimulator()
    registered_tool = ToolSimulator._registered_tools["test_mock_tool"]

    with pytest.raises(ValueError) as excinfo:
        simulator._handle_mock_mode(
            registered_tool=registered_tool,
            input_data={"tool_name": "test_mock_tool", "parameters": {}},
            state_key="test",
        )

    assert "mock_function is required for tool simulator mock mode" in str(excinfo.value)


def test_clear_registry():
    """Test clearing tool registry."""

    @ToolSimulator.function_tool("test_function")
    def test_func():
        pass

    assert len(ToolSimulator._registered_tools) == 1

    ToolSimulator.clear_registry()

    assert len(ToolSimulator._registered_tools) == 0
    assert ToolSimulator._state_registry is None


def test_attaching_function_tool_simulator_to_strands_agent():
    """Test attaching function tool simulator to Strands agent."""

    # Mock function that handles parameters
    def mock_function(input_value):
        return {"result": f"processed {input_value}"}

    # Register a function tool simulator
    @ToolSimulator.function_tool("test_function_tool", mode="mock", mock_function=mock_function)
    def test_function_tool(input_value: str) -> Dict[str, Any]:
        """Test function tool for agent attachment.

        Args:
            input_value: Input parameter for processing
        """
        pass

    # Create simulator and get the tool
    simulator = ToolSimulator()
    tool_wrapper = simulator.get_tool("test_function_tool")

    # Create a Strands Agent with the tool simulator
    from strands import Agent

    agent = Agent(tools=[tool_wrapper])

    # Verify the agent has access to the tool
    assert "test_function_tool" in agent.tool_names
    assert hasattr(agent.tool, "test_function_tool")


def test_attaching_mcp_tool_simulator_to_strands_agent():
    """Test attaching MCP tool simulator to Strands agent."""

    # Mock function for MCP tool
    def mock_mcp_processor(param1, param2=42):
        return {"content": [{"type": "text", "text": f"MCP processed: {param1} with value {param2}"}], "isError": False}

    schema = {
        "type": "object",
        "properties": {"param1": {"type": "string"}, "param2": {"type": "integer", "default": 42}},
        "required": ["param1"],
    }

    # Register an MCP tool simulator
    @ToolSimulator.mcp_tool("test_mcp_tool", schema=schema, mode="mock", mock_function=mock_mcp_processor)
    def test_mcp_tool(param1: str, param2: int = 42) -> Dict[str, Any]:
        """Test MCP tool for agent attachment.

        Args:
            param1: First parameter for MCP processing
            param2: Second parameter with default value
        """
        pass

    # Create simulator and get the tool
    simulator = ToolSimulator()
    tool_wrapper = simulator.get_tool("test_mcp_tool")

    # Create a Strands Agent with the tool simulator
    from strands import Agent

    agent = Agent(tools=[tool_wrapper])

    # Verify the agent has access to the tool
    assert "test_mcp_tool" in agent.tool_names
    assert hasattr(agent.tool, "test_mcp_tool")


def test_attaching_api_tool_simulator_to_strands_agent():
    """Test attaching API tool simulator to Strands agent."""

    # Static response for API tool
    static_response = {
        "status": 200,
        "data": {"message": "API tool working", "timestamp": "2024-01-01T12:00:00Z", "endpoint": "/test/api"},
    }

    # Register an API tool simulator
    @ToolSimulator.api_tool(
        "test_api_tool", path="/test/api", method="GET", mode="static", static_response=static_response
    )
    def test_api_tool(query: str = "") -> Dict[str, Any]:
        """Test API tool for agent attachment.

        Args:
            query: Query parameter for API call
        """
        pass

    # Create simulator and get the tool
    simulator = ToolSimulator()
    tool_wrapper = simulator.get_tool("test_api_tool")

    # Create a Strands Agent with the tool simulator
    from strands import Agent

    agent = Agent(tools=[tool_wrapper])

    # Verify the agent has access to the tool
    assert "test_api_tool" in agent.tool_names
    assert hasattr(agent.tool, "test_api_tool")
