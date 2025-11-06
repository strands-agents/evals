"""Tests for ActorSimulator class."""

from unittest.mock import MagicMock, patch

import pytest
from strands.agent.agent_result import AgentResult

from strands_evals import Case
from strands_evals.simulation import ActorSimulator
from strands_evals.types.simulation import ActorProfile, ActorResponse


@pytest.fixture
def sample_actor_profile():
    """Fixture providing a sample actor profile."""
    return ActorProfile(
        traits={
            "expertise_level": "beginner",
            "communication_style": "casual",
            "patience_level": "high",
        },
        context="A beginner user learning about travel planning.",
        actor_goal="Book a complete trip to Tokyo including flights and hotel.",
    )


@pytest.fixture
def sample_case():
    """Fixture providing a sample test case."""
    return Case(
        input="I want to plan a trip to Tokyo",
        metadata={"task_description": "Complete travel package arranged"},
    )


def test_actor_simulator_init(sample_actor_profile):
    """Test ActorSimulator initialization with profile."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello, I need help",
        system_prompt_template="Test prompt: {actor_profile}",
        tools=None,
        model=None,
    )

    assert simulator.actor_profile == sample_actor_profile
    assert simulator.initial_query == "Hello, I need help"
    assert simulator.agent is not None
    assert len(simulator.conversation_history) == 2  # greeting + initial query


def test_initialize_conversation(sample_actor_profile):
    """Test conversation initialization creates greeting and initial query."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="I need help with travel",
        system_prompt_template="Test: {actor_profile}",
    )

    history = simulator.conversation_history
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert any(greeting in history[0]["content"][0]["text"] for greeting in ActorSimulator.INITIAL_GREETINGS)
    assert history[1]["role"] == "assistant"
    assert history[1]["content"][0]["text"] == "I need help with travel"


@patch("strands_evals.simulation.actor_simulator.Agent")
def test_from_case_for_user_simulator(mock_agent_class, sample_case):
    """Test factory method creates simulator from case."""
    # Mock the profile generation agent
    mock_profile_agent = MagicMock()
    mock_profile = ActorProfile(
        traits={"test": "trait"},
        context="Test context",
        actor_goal="Test goal",
    )
    mock_result = MagicMock()
    mock_result.structured_output = mock_profile
    mock_profile_agent.return_value = mock_result

    # Mock the main simulator agent
    mock_simulator_agent = MagicMock()

    # Configure mock to return different instances
    mock_agent_class.side_effect = [mock_profile_agent, mock_simulator_agent]

    simulator = ActorSimulator.from_case_for_user_simulator(case=sample_case)

    assert simulator.actor_profile == mock_profile
    assert simulator.initial_query == sample_case.input
    assert mock_agent_class.call_count == 2  # Once for profile gen, once for simulator


@patch("strands_evals.simulation.actor_simulator.Agent")
def test_generate_profile_from_case(mock_agent_class, sample_case):
    """Test profile generation from case."""
    mock_agent = MagicMock()
    mock_profile = ActorProfile(
        traits={"generated": "trait"},
        context="Generated context",
        actor_goal="Generated goal",
    )
    mock_result = MagicMock()
    mock_result.structured_output = mock_profile
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent

    profile = ActorSimulator._generate_profile_from_case(sample_case)

    assert profile == mock_profile
    assert mock_agent.called
    # Verify structured_output_model was passed
    call_args = mock_agent.call_args
    assert call_args[1]["structured_output_model"] == ActorProfile


def test_act_generates_response(sample_actor_profile):
    """Test act method generates actor response."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    # Mock the agent's response
    mock_response = MagicMock(spec=AgentResult)
    mock_actor_response = ActorResponse(
        reasoning="Test reasoning",
        message="Test response message",
    )
    mock_response.structured_output = mock_actor_response
    simulator.agent = MagicMock(return_value=mock_response)

    result = simulator.act("What can I help you with?")

    assert result == mock_response
    assert result.structured_output.message == "Test response message"
    simulator.agent.assert_called_once()


def test_act_uses_structured_output(sample_actor_profile):
    """Test act method requests structured output."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_actor_response = ActorResponse(reasoning="Test", message="Test message")
    mock_response.structured_output = mock_actor_response
    simulator.agent = MagicMock(return_value=mock_response)

    simulator.act("Test message")

    # Verify structured_output_model parameter
    call_kwargs = simulator.agent.call_args[1]
    assert call_kwargs["structured_output_model"] == ActorResponse


def test_has_next_returns_true_initially(sample_actor_profile):
    """Test has_next returns True before any turns."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    assert simulator.has_next() is True


def test_has_next_respects_max_turns(sample_actor_profile):
    """Test has_next returns False after max_turns reached."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
        max_turns=3,
    )

    # Mock responses
    mock_response = MagicMock(spec=AgentResult)
    mock_actor_response = ActorResponse(reasoning="Test", message="Continue")
    mock_response.structured_output = mock_actor_response
    simulator.agent = MagicMock(return_value=mock_response)

    # Simulate 3 turns with max_turns=3
    for _ in range(3):
        assert simulator.has_next() is True
        simulator.act("Test message")

    # After 3 turns, should return False
    assert simulator.has_next() is False


def test_has_next_detects_stop_token(sample_actor_profile):
    """Test has_next returns False when stop token is present."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    # Mock response with stop token
    mock_response = MagicMock(spec=AgentResult)
    mock_actor_response = ActorResponse(reasoning="Done", message="Thanks! <stop/>")
    mock_response.structured_output = mock_actor_response
    simulator.agent = MagicMock(return_value=mock_response)

    # After act with stop token, has_next should return False
    simulator.act("Test message")
    assert simulator.has_next() is False
