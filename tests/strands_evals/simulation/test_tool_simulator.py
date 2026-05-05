"""Tests for ToolSimulator class."""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from strands_evals.case import Case
from strands_evals.simulation.tool_simulator import StateRegistry, ToolSimulator


# Common output schemas for testing
class GenericOutput(BaseModel):
    """Generic output schema for test tools."""

    result: str


class DictOutput(BaseModel):
    """Dict output schema for test tools."""

    x: int
    y: str


class BalanceOutput(BaseModel):
    """Balance output schema."""

    balance: int
    currency: str


class TransferOutput(BaseModel):
    """Transfer output schema."""

    status: str
    message: str


class TransactionsOutput(BaseModel):
    """Transactions output schema."""

    transactions: list


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


def test_tool_simulator_init():
    """Test ToolSimulator initialization with all parameters."""
    custom_registry = StateRegistry()

    simulator = ToolSimulator(
        state_registry=custom_registry,
        model=None,
    )

    assert simulator.state_registry is custom_registry
    assert simulator.model is None  # model is now used for LLM inference


def test_tool_decorator_registration():
    """Test tool decorator registration."""
    simulator = ToolSimulator()

    @simulator.tool(output_schema=DictOutput)
    def test_function(x: int, y: str) -> dict:
        """A sample function for testing."""
        return {"x": x, "y": y}

    assert "test_function" in simulator._registered_tools
    registered_tool = simulator._registered_tools["test_function"]
    assert registered_tool.name == "test_function"
    assert registered_tool.function == test_function


def test_tool_decorator_with_name():
    """Test tool decorator with custom name."""
    simulator = ToolSimulator()

    class SingleIntOutput(BaseModel):
        x: int

    @simulator.tool(output_schema=SingleIntOutput, name="custom_name")
    def test_function(x: int) -> dict:
        """A sample function for testing."""
        return {"x": x}

    assert "custom_name" in simulator._registered_tools
    registered_tool = simulator._registered_tools["custom_name"]
    assert registered_tool.name == "custom_name"
    assert registered_tool.function == test_function


def test_tool_simulation(mock_model):
    """Test tool simulation."""
    simulator = ToolSimulator(model=mock_model)

    # Register tool
    @simulator.tool(output_schema=GenericOutput)
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

    @simulator.tool(output_schema=GenericOutput)
    def func1():
        pass

    @simulator.tool(output_schema=GenericOutput)
    def func2():
        pass

    tools = simulator.list_tools()

    assert set(tools) == {"func1", "func2"}


def test_sharedstate_registry(mock_model):
    """Test that tools can share the same state registry."""
    shared_state_id = "shared_banking_state"
    initial_state = "Initial banking system state with account balances"

    simulator = ToolSimulator(model=mock_model)

    # Register tools that share the same state
    @simulator.tool(
        output_schema=BalanceOutput,
        share_state_id=shared_state_id,
        initial_state_description=initial_state,
    )
    def check_balance(account_id: str):
        """Check account balance."""
        pass

    @simulator.tool(
        output_schema=TransferOutput,
        share_state_id=shared_state_id,
        initial_state_description=initial_state,
    )
    def transfer_funds(from_account: str, to_account: str):
        """Transfer funds between accounts."""
        pass

    @simulator.tool(
        output_schema=TransactionsOutput,
        share_state_id=shared_state_id,
        initial_state_description=initial_state,
    )
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
    shared_state = simulator.state_registry.get_state(shared_state_id)
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


def test_clear_tools():
    """Test clearing tool registry for a specific simulator instance."""
    simulator = ToolSimulator()

    @simulator.tool(output_schema=GenericOutput)
    def test_func():
        pass

    assert len(simulator._registered_tools) == 1

    simulator.clear_tools()

    assert len(simulator._registered_tools) == 0


def test_attaching_tool_simulator_to_strands_agent():
    """Test attaching tool simulator to Strands agent."""
    simulator = ToolSimulator()

    # Register a tool simulator
    @simulator.tool(output_schema=GenericOutput)
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

    @simulator.tool(
        output_schema=GenericOutput,
        share_state_id="test_state",
        initial_state_description="Test initial state",
    )
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

    @simulator.tool(output_schema=CustomOutput)
    def test_tool_with_schema():
        pass

    registered_tool = simulator._registered_tools["test_tool_with_schema"]
    assert registered_tool.output_schema == CustomOutput


def test_automatic_input_model_as_output_schema():
    """Test that explicit output_schema is required and properly stored."""
    from pydantic import BaseModel

    class ParametersOutput(BaseModel):
        """Output schema matching tool parameters."""

        name: str
        age: int
        active: bool = True

    simulator = ToolSimulator()

    # Create a tool with typed parameters and explicit output_schema
    @simulator.tool(output_schema=ParametersOutput)
    def test_tool_with_parameters(name: str, age: int, active: bool = True) -> dict:
        """A test tool with typed parameters.

        Args:
            name: The person's name
            age: The person's age
            active: Whether the person is active
        """
        pass

    # Check that the tool was registered
    assert "test_tool_with_parameters" in simulator._registered_tools
    registered_tool = simulator._registered_tools["test_tool_with_parameters"]

    # Verify that output_schema was set
    assert registered_tool.output_schema is not None, "output_schema should be set"

    # Verify it's a Pydantic BaseModel class
    assert issubclass(registered_tool.output_schema, BaseModel), "output_schema should be a Pydantic BaseModel"

    # Check the schema has the expected fields
    schema = registered_tool.output_schema.model_json_schema()
    properties = schema.get("properties", {})
    assert "name" in properties, "Schema should have 'name' field"
    assert "age" in properties, "Schema should have 'age' field"
    assert "active" in properties, "Schema should have 'active' field"


def test_explicit_output_schema_override():
    """Test that explicit output_schema takes precedence over automatic input_model."""
    from pydantic import BaseModel

    simulator = ToolSimulator()

    class CustomOutput(BaseModel):
        result: str
        count: int

    # Create a tool with explicit output_schema
    @simulator.tool(output_schema=CustomOutput)
    def test_tool_with_explicit_schema(name: str, age: int) -> dict:
        """A test tool with explicit output_schema.

        Args:
            name: The person's name
            age: The person's age
        """
        pass

    # Check that the tool was registered
    assert "test_tool_with_explicit_schema" in simulator._registered_tools
    registered_tool = simulator._registered_tools["test_tool_with_explicit_schema"]

    # Verify that the explicit output_schema is used, not the input_model
    assert registered_tool.output_schema is CustomOutput, "Explicit output_schema should be used"


def test_no_parameters_tool_input_model():
    """Test tool with no parameters and simple output schema."""
    from pydantic import BaseModel

    class EmptyOutput(BaseModel):
        """Empty output schema for tool with no parameters."""

        pass

    simulator = ToolSimulator()

    # Create a tool with no parameters
    @simulator.tool(output_schema=EmptyOutput)
    def test_tool_no_params() -> dict:
        """A test tool with no parameters."""
        pass

    # Check that the tool was registered
    assert "test_tool_no_params" in simulator._registered_tools
    registered_tool = simulator._registered_tools["test_tool_no_params"]

    # The output_schema should be set
    assert registered_tool.output_schema is not None, "output_schema should be set"
    assert issubclass(registered_tool.output_schema, BaseModel), "output_schema should be a Pydantic BaseModel"

    # Check that it has no required properties (since no parameters)
    schema = registered_tool.output_schema.model_json_schema()
    properties = schema.get("properties", {})
    # Empty model should mean no properties
    assert len(properties) == 0, "Tool with empty schema should have no properties"


def test_pre_call_hook_short_circuits_llm(mock_model):
    """Test that pre_call_hook can short-circuit the LLM call by returning a dict."""
    hook_calls = []

    def fault_injector(event):
        hook_calls.append({"tool_name": event.tool_name, "parameters": event.parameters})
        return {"error": {"code": "QuotaExceeded", "retryAfterSeconds": 2}}

    simulator = ToolSimulator(model=mock_model, pre_call_hook=fault_injector)

    @simulator.tool(output_schema=GenericOutput)
    def my_tool(message: str) -> dict:
        """A tool that should be intercepted."""
        pass

    # The LLM should never be called because the hook short-circuits
    mock_agent_instance = MagicMock()
    mock_agent_instance.return_value = MagicMock()

    with pytest.MonkeyPatch().context() as m:
        m.setattr("strands_evals.simulation.tool_simulator.Agent", lambda **kwargs: mock_agent_instance)

        result = simulator.my_tool(message="hello")

    assert result == {"error": {"code": "QuotaExceeded", "retryAfterSeconds": 2}}
    assert len(hook_calls) == 1
    assert hook_calls[0]["tool_name"] == "my_tool"
    assert hook_calls[0]["parameters"] == {"message": "hello"}
    # LLM agent should NOT have been called
    assert not mock_agent_instance.called


def test_pre_call_hook_returns_none_proceeds_normally(mock_model):
    """Test that pre_call_hook returning None lets normal LLM simulation proceed."""
    hook_calls = []

    def passthrough_hook(event):
        hook_calls.append(event.tool_name)
        return None  # Don't short-circuit

    simulator = ToolSimulator(model=mock_model, pre_call_hook=passthrough_hook)

    @simulator.tool(output_schema=GenericOutput)
    def my_tool(message: str) -> dict:
        """A tool."""
        pass

    mock_agent_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.__str__ = MagicMock(return_value='{"result": "llm response"}')
    mock_agent_instance.return_value = mock_result

    with pytest.MonkeyPatch().context() as m:
        m.setattr("strands_evals.simulation.tool_simulator.Agent", lambda **kwargs: mock_agent_instance)

        result = simulator.my_tool(message="hello")

    assert result == {"result": "llm response"}
    assert len(hook_calls) == 1
    # LLM agent SHOULD have been called
    assert mock_agent_instance.called


def test_post_call_hook_modifies_response(mock_model):
    """Test that post_call_hook can modify the LLM response before caching."""

    def response_modifier(event):
        event.response["_simulated_latency_ms"] = 42
        return event.response

    simulator = ToolSimulator(model=mock_model, post_call_hook=response_modifier)

    @simulator.tool(output_schema=GenericOutput)
    def my_tool(message: str) -> dict:
        """A tool."""
        pass

    mock_agent_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.__str__ = MagicMock(return_value='{"result": "llm response"}')
    mock_agent_instance.return_value = mock_result

    with pytest.MonkeyPatch().context() as m:
        m.setattr("strands_evals.simulation.tool_simulator.Agent", lambda **kwargs: mock_agent_instance)

        result = simulator.my_tool(message="hello")

    assert result["result"] == "llm response"
    assert result["_simulated_latency_ms"] == 42

    # Verify the cached response also has the modification
    state = simulator.get_state("my_tool")
    cached = state["previous_calls"][0]
    assert cached["response"]["_simulated_latency_ms"] == 42


def test_pre_call_hook_response_is_cached():
    """Test that a short-circuited response from pre_call_hook is cached in state registry."""

    def always_fail(event):
        return {"error": "service_unavailable"}

    simulator = ToolSimulator(pre_call_hook=always_fail)

    @simulator.tool(output_schema=GenericOutput)
    def my_tool(x: int) -> dict:
        """A tool."""
        pass

    # No need to mock Agent since LLM is never called
    result = simulator.my_tool(x=1)

    assert result == {"error": "service_unavailable"}

    state = simulator.get_state("my_tool")
    assert len(state["previous_calls"]) == 1
    assert state["previous_calls"][0]["response"] == {"error": "service_unavailable"}
    assert state["previous_calls"][0]["tool_name"] == "my_tool"


def test_pre_call_hook_receives_previous_calls():
    """Test that pre_call_hook receives the accumulated call history."""
    received_histories = []

    call_count = 0

    def counting_hook(event):
        nonlocal call_count
        received_histories.append(list(event.previous_calls))
        call_count += 1
        return {"call_number": call_count}

    simulator = ToolSimulator(pre_call_hook=counting_hook)

    @simulator.tool(output_schema=GenericOutput)
    def my_tool(x: int) -> dict:
        """A tool."""
        pass

    simulator.my_tool(x=1)
    simulator.my_tool(x=2)
    simulator.my_tool(x=3)

    # First call should see empty history
    assert len(received_histories[0]) == 0
    # Second call should see 1 previous call
    assert len(received_histories[1]) == 1
    # Third call should see 2 previous calls
    assert len(received_histories[2]) == 2


def test_both_hooks_together(mock_model):
    """Test that pre_call_hook and post_call_hook work together when pre_call_hook returns None."""
    pre_calls = []
    post_calls = []

    def pre_hook(event):
        pre_calls.append(event.tool_name)
        return None  # Let LLM proceed

    def post_hook(event):
        post_calls.append(event.tool_name)
        event.response["modified"] = True
        return event.response

    simulator = ToolSimulator(model=mock_model, pre_call_hook=pre_hook, post_call_hook=post_hook)

    @simulator.tool(output_schema=GenericOutput)
    def my_tool(message: str) -> dict:
        """A tool."""
        pass

    mock_agent_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.__str__ = MagicMock(return_value='{"result": "ok"}')
    mock_agent_instance.return_value = mock_result

    with pytest.MonkeyPatch().context() as m:
        m.setattr("strands_evals.simulation.tool_simulator.Agent", lambda **kwargs: mock_agent_instance)

        result = simulator.my_tool(message="test")

    assert result == {"result": "ok", "modified": True}
    assert len(pre_calls) == 1
    assert len(post_calls) == 1


def test_post_call_hook_skipped_when_pre_hook_short_circuits():
    """Test that post_call_hook is NOT called when pre_call_hook short-circuits."""
    post_calls = []

    def pre_hook(event):
        return {"error": "injected fault"}

    def post_hook(event):
        post_calls.append(True)
        return event.response

    simulator = ToolSimulator(pre_call_hook=pre_hook, post_call_hook=post_hook)

    @simulator.tool(output_schema=GenericOutput)
    def my_tool(x: int) -> dict:
        """A tool."""
        pass

    result = simulator.my_tool(x=1)

    assert result == {"error": "injected fault"}
    assert len(post_calls) == 0  # post_call_hook should not have been called


def test_pre_call_hook_with_shared_state():
    """Test that pre_call_hook works correctly with shared state between tools."""
    call_counts: Dict[str, int] = {}

    def rate_limiter(event):
        call_counts[event.tool_name] = call_counts.get(event.tool_name, 0) + 1
        if call_counts[event.tool_name] > 2:
            return {"error": {"code": "RateLimited", "tool": event.tool_name}}
        return {"result": f"{event.tool_name} ok"}

    simulator = ToolSimulator(pre_call_hook=rate_limiter)

    @simulator.tool(output_schema=GenericOutput, share_state_id="shared")
    def tool_a(x: int) -> dict:
        """Tool A."""
        pass

    @simulator.tool(output_schema=GenericOutput, share_state_id="shared")
    def tool_b(x: int) -> dict:
        """Tool B."""
        pass

    # Both tools share state, but rate limiter tracks per-tool
    assert simulator.tool_a(x=1) == {"result": "tool_a ok"}
    assert simulator.tool_a(x=2) == {"result": "tool_a ok"}
    assert simulator.tool_a(x=3) == {"error": {"code": "RateLimited", "tool": "tool_a"}}

    # tool_b has its own counter
    assert simulator.tool_b(x=1) == {"result": "tool_b ok"}

    # Shared state should have all calls
    state = simulator.get_state("shared")
    assert len(state["previous_calls"]) == 4


def test_init_without_hooks():
    """Test that ToolSimulator works normally without hooks (backward compatibility)."""
    simulator = ToolSimulator()

    assert simulator._pre_call_hook is None
    assert simulator._post_call_hook is None
