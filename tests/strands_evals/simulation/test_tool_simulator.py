"""Tests for ToolSimulator class."""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from strands_evals.case import Case
from strands_evals.simulation.tool_simulator import StateRegistry, ToolSimulator


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
    assert simulator.function_tool_prompt is not None  # Check that prompt template is loaded


def test_tool_decorator_registration():
    """Test tool decorator registration."""
    simulator = ToolSimulator()

    @simulator()
    def test_function(x: int, y: str) -> dict:
        """A sample function for testing."""
        return {"x": x, "y": y}

    assert "test_function" in ToolSimulator._registered_tools
    registered_tool = ToolSimulator._registered_tools["test_function"]
    assert registered_tool.name == "test_function"
    assert registered_tool.function == test_function


def test_tool_decorator_with_name():
    """Test tool decorator with custom name."""
    simulator = ToolSimulator()

    @simulator(name="custom_name")
    def test_function(x: int) -> dict:
        """A sample function for testing."""
        return {"x": x}

    assert "custom_name" in ToolSimulator._registered_tools
    registered_tool = ToolSimulator._registered_tools["custom_name"]
    assert registered_tool.name == "custom_name"
    assert registered_tool.function == test_function


def test_tool_simulation(mock_model):
    """Test tool simulation."""
    simulator = ToolSimulator(model=mock_model)

    # Register tool
    @simulator()
    def test_func(message: str) -> dict:
        """Test function that should be simulated."""
        pass

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
        result = simulator.test_func("Hello, world!")

        assert result == {"result": "simulated response"}
        assert mock_agent_instance.called


def test_list_tools():
    """Test listing registered tools."""
    simulator = ToolSimulator()

    @simulator()
    def func1():
        pass

    @simulator()
    def func2():
        pass

    tools = simulator.list_tools()

    assert set(tools) == {"func1", "func2"}


def test_shared_state_registry(mock_model):
    """Test that tools can share the same state registry."""
    shared_state_id = "shared_banking_state"
    initial_state = "Initial banking system state with account balances"

    simulator = ToolSimulator(model=mock_model)

    # Register tools that share the same state
    @simulator(initial_state_description=initial_state, share_state_id=shared_state_id)
    def check_balance(account_id: str):
        """Check account balance."""
        pass

    @simulator(initial_state_description=initial_state, share_state_id=shared_state_id)
    def transfer_funds(from_account: str, to_account: str):
        """Transfer funds between accounts."""
        pass

    @simulator(initial_state_description=initial_state, share_state_id=shared_state_id)
    def get_transactions(account_id: str):
        """Get transaction history."""
        pass

    # Mock the Agent constructor to avoid real LLM calls
    mock_agent_instances = []
    expected_responses = [
        {"balance": 1000, "currency": "USD"},  # Function response
        {"status": "success", "message": "Transfer completed"},  # Function response
        {"transactions": []},  # Function response
    ]

    def create_mock_agent(**kwargs):
        mock_agent = MagicMock()
        mock_result = MagicMock()

        if len(mock_agent_instances) < len(expected_responses):
            response = expected_responses[len(mock_agent_instances)]
            import json

            # Mock __str__ method to return JSON string
            mock_result.__str__ = MagicMock(return_value=json.dumps(response))

        mock_agent.return_value = mock_result
        mock_agent_instances.append(mock_agent)
        return mock_agent

    with pytest.MonkeyPatch().context() as m:
        # Mock the Agent class constructor
        m.setattr("strands_evals.simulation.tool_simulator.Agent", create_mock_agent)

        # Execute each tool in order
        balance_result = simulator.check_balance("12345")
        transfer_result = simulator.transfer_funds("12345", "67890")
        transactions_result = simulator.get_transactions("12345")

        # Verify results
        assert balance_result == {"balance": 1000, "currency": "USD"}
        assert transfer_result == {"status": "success", "message": "Transfer completed"}
        assert transactions_result == {"transactions": []}

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

    # Verify each tool call recorded its parameters correctly
    for call in shared_state["previous_calls"]:
        assert "parameters" in call


def test_cache_tool_call_function():
    """Test recording function call in state registry."""
    registry = StateRegistry()

    registry.cache_tool_call(
        tool_name="test_tool",
        state_key="test_state",
        response_data={"result": "success"},
        parameters={"param": "value"},
    )

    state = registry.get_state("test_state")
    assert "previous_calls" in state
    assert len(state["previous_calls"]) == 1
    call = state["previous_calls"][0]
    assert call["tool_name"] == "test_tool"
    assert call["parameters"] == {"param": "value"}
    assert call["response"] == {"result": "success"}


def test_tool_not_found_raises_error():
    """Test that accessing non-existent tools raises ValueError."""
    simulator = ToolSimulator()

    # Test that accessing a non-existent tool via __getattr__ raises AttributeError
    with pytest.raises(AttributeError) as excinfo:
        _ = simulator.nonexistent_tool

    assert "not found in registered tools" in str(excinfo.value)


def test_clear_registry():
    """Test clearing tool registry."""
    simulator = ToolSimulator()

    @simulator()
    def test_func():
        pass

    assert len(ToolSimulator._registered_tools) == 1

    ToolSimulator.clear_registry()

    assert len(ToolSimulator._registered_tools) == 0


def test_attaching_tool_simulator_to_strands_agent():
    """Test attaching tool simulator to Strands agent."""
    simulator = ToolSimulator()

    # Register a tool simulator
    @simulator()
    def test_tool(input_value: str) -> Dict[str, Any]:
        """Test tool for agent attachment.

        Args:
            input_value: Input parameter for processing
        """
        pass

    # Get the tool wrapper
    tool_wrapper = simulator.get_tool("test_tool")

    # Create a Strands Agent with the tool simulator
    from strands import Agent

    agent = Agent(tools=[tool_wrapper])

    # Verify the agent has access to the tool
    assert "test_tool" in agent.tool_names
    assert hasattr(agent.tool, "test_tool")


def test_get_state_method():
    """Test the get_state method for direct state access."""
    simulator = ToolSimulator()

    @simulator(initial_state_description="Test initial state", share_state_id="test_state")
    def test_tool():
        pass

    # Test get_state method
    state = simulator.get_state("test_state")
    assert "initial_state" in state
    assert state["initial_state"] == "Test initial state"
    assert "previous_calls" in state


def test_output_schema_parameter():
    """Test that output_schema parameter is accepted and stored."""
    from pydantic import BaseModel

    class CustomOutput(BaseModel):
        result: str
        count: int

    simulator = ToolSimulator()

    @simulator(output_schema=CustomOutput)
    def test_tool_with_schema():
        pass

    registered_tool = ToolSimulator._registered_tools["test_tool_with_schema"]
    assert registered_tool.output_schema == CustomOutput
