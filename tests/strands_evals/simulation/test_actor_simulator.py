"""Tests for ActorSimulator class."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
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
    mock_profile_agent = MagicMock()
    mock_profile = ActorProfile(
        traits={"test": "trait"},
        context="Test context",
        actor_goal="Test goal",
    )
    mock_result = MagicMock()
    mock_result.structured_output = mock_profile
    mock_profile_agent.return_value = mock_result

    mock_simulator_agent = MagicMock()

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
    call_args = mock_agent.call_args
    assert call_args[1]["structured_output_model"] == ActorProfile


def test_act_generates_response(sample_actor_profile):
    """Test act method generates actor response."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_actor_response = ActorResponse(
        reasoning="Test reasoning",
        message="Test response message",
        stop=False,
    )
    mock_response.structured_output = mock_actor_response
    simulator.agent = MagicMock(return_value=mock_response)

    result = simulator.act("What can I help you with?")

    assert result == mock_response
    assert result.structured_output.message == "Test response message"
    simulator.agent.assert_called_once()


def test_act_uses_actor_response_by_default(sample_actor_profile):
    """Test act method uses ActorResponse as default structured output model."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_actor_response = ActorResponse(reasoning="Test", message="Test message", stop=False)
    mock_response.structured_output = mock_actor_response
    simulator.agent = MagicMock(return_value=mock_response)

    simulator.act("Test message")

    call_kwargs = simulator.agent.call_args[1]
    assert call_kwargs["structured_output_model"] == ActorResponse


def test_act_with_custom_structured_output_model(sample_actor_profile):
    """Test act passes custom structured_output_model to the agent."""

    class CustomOutput(BaseModel):
        answer: str
        confidence: float
        stop: bool = False
        message: str | None = None

    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = CustomOutput(answer="test", confidence=0.9, stop=False, message="hi")
    simulator.agent = MagicMock(return_value=mock_response)

    result = simulator.act("Test message", structured_output_model=CustomOutput)

    call_kwargs = simulator.agent.call_args[1]
    assert call_kwargs["structured_output_model"] == CustomOutput
    assert result.structured_output.answer == "test"


def test_act_custom_model_without_stop_raises(sample_actor_profile):
    """Test act raises ValueError if custom model has no stop field."""

    class NoStopModel(BaseModel):
        message: str | None = None

    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    with pytest.raises(ValueError, match="must have a 'stop' field"):
        simulator.act("Test message", structured_output_model=NoStopModel)


def test_act_custom_model_without_message_raises(sample_actor_profile):
    """Test act raises ValueError if custom model has no message field."""

    class NoMessageModel(BaseModel):
        answer: str = ""
        stop: bool = False

    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    with pytest.raises(ValueError, match="must have a 'message' field"):
        simulator.act("Test message", structured_output_model=NoMessageModel)


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

    mock_response = MagicMock(spec=AgentResult)
    mock_actor_response = ActorResponse(reasoning="Test", message="Continue", stop=False)
    mock_response.structured_output = mock_actor_response
    simulator.agent = MagicMock(return_value=mock_response)

    for _ in range(3):
        assert simulator.has_next() is True
        simulator.act("Test message")

    assert simulator.has_next() is False


def test_has_next_detects_stop(sample_actor_profile):
    """Test has_next returns False when actor signals stop."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_actor_response = ActorResponse(reasoning="Done", message=None, stop=True)
    mock_response.structured_output = mock_actor_response
    simulator.agent = MagicMock(return_value=mock_response)

    simulator.act("Test message")
    assert simulator.has_next() is False


def test_act_sets_stop_reason_goal_completed(sample_actor_profile):
    """Test act sets stop_reason to 'goal_completed' when actor signals stop."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_actor_response = ActorResponse(reasoning="Done", message=None, stop=True)
    mock_response.structured_output = mock_actor_response
    simulator.agent = MagicMock(return_value=mock_response)

    result = simulator.act("Test message")

    assert result.structured_output.stop_reason == "goal_completed"


def test_act_sets_stop_reason_max_turns(sample_actor_profile):
    """Test act sets stop_reason to 'max_turns' when turn cap is reached."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
        max_turns=1,
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_actor_response = ActorResponse(reasoning="r", message="more please", stop=False)
    mock_response.structured_output = mock_actor_response
    simulator.agent = MagicMock(return_value=mock_response)

    result = simulator.act("agent reply")

    assert result.structured_output.stop is True
    assert result.structured_output.stop_reason == "max_turns"


def test_act_continuing_turn_no_stop_reason(sample_actor_profile):
    """Test act leaves stop_reason as None for normal continuing turns."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_actor_response = ActorResponse(reasoning="thinking", message="keep going", stop=False)
    mock_response.structured_output = mock_actor_response
    simulator.agent = MagicMock(return_value=mock_response)

    result = simulator.act("agent reply")

    assert result.structured_output.stop is False
    assert result.structured_output.stop_reason is None
    assert result.structured_output.message == "keep going"


def test_act_custom_model_manages_stop(sample_actor_profile):
    """When structured_output_model is provided, act() still manages stop via the stop field."""

    class CustomOutput(BaseModel):
        stop: bool = False
        message: str | None = None

    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = CustomOutput(message="done", stop=True)
    simulator.agent = MagicMock(return_value=mock_response)

    simulator.act("agent reply", structured_output_model=CustomOutput)

    assert simulator.stop is True
    assert simulator.has_next() is False


def test_act_custom_model_max_turns(sample_actor_profile):
    """Custom model path still enforces max_turns."""

    class CustomOutput(BaseModel):
        stop: bool = False
        message: str | None = None

    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
        max_turns=1,
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = CustomOutput(message="hi", stop=False)
    simulator.agent = MagicMock(return_value=mock_response)

    simulator.act("agent reply", structured_output_model=CustomOutput)

    assert simulator.stop is True
    assert simulator.has_next() is False


def test_system_prompt_template_none_uses_default(sample_actor_profile):
    """When system_prompt_template is None, the default template is rendered with the profile."""
    from strands_evals.simulation.prompt_templates.actor_system_prompt import (
        DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE,
    )

    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
    )

    expected = DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE.format(actor_profile=sample_actor_profile.model_dump())
    assert simulator.agent.system_prompt == expected


def test_system_prompt_template_prerendered_passes_through(sample_actor_profile):
    """A template with no {actor_profile} placeholder is passed through verbatim."""
    prerendered = "You are simulating Alice, a beginner user. Keep replies short."

    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template=prerendered,
    )

    assert simulator.agent.system_prompt == prerendered


def test_system_prompt_contains_stop_instruction(sample_actor_profile):
    """Default prompt instructs the actor to set stop=true."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
    )

    assert "stop=true" in simulator.agent.system_prompt


def test_explicit_template_overrides_default(sample_actor_profile):
    """Explicit system_prompt_template is used instead of the default."""
    custom = "Custom prompt for {actor_profile}"
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template=custom,
    )

    assert simulator.agent.system_prompt == custom.format(actor_profile=sample_actor_profile.model_dump())


def test_init_structured_output_model_used_by_act(sample_actor_profile):
    """structured_output_model set at init is used as default for act()."""

    class CustomOutput(BaseModel):
        stop: bool = False
        message: str | None = None
        extra: str = "default"

    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
        structured_output_model=CustomOutput,
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = CustomOutput(message="hi", stop=False)
    simulator.agent = MagicMock(return_value=mock_response)

    simulator.act("agent reply")

    call_kwargs = simulator.agent.call_args[1]
    assert call_kwargs["structured_output_model"] == CustomOutput


def test_init_structured_output_model_overridden_per_call(sample_actor_profile):
    """Per-call structured_output_model overrides the init-level default."""

    class InitModel(BaseModel):
        stop: bool = False
        message: str | None = None

    class CallModel(BaseModel):
        stop: bool = False
        message: str | None = None
        priority: int = 0

    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
        structured_output_model=InitModel,
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = CallModel(message="hi", stop=False)
    simulator.agent = MagicMock(return_value=mock_response)

    simulator.act("agent reply", structured_output_model=CallModel)

    call_kwargs = simulator.agent.call_args[1]
    assert call_kwargs["structured_output_model"] == CallModel


def test_init_structured_output_model_validates_stop_field(sample_actor_profile):
    """Init raises ValueError if structured_output_model has no stop field."""

    class NoStopModel(BaseModel):
        message: str | None = None

    with pytest.raises(ValueError, match="must have a 'stop' field"):
        ActorSimulator(
            actor_profile=sample_actor_profile,
            initial_query="Hello",
            system_prompt_template="Test: {actor_profile}",
            structured_output_model=NoStopModel,
        )


def test_init_structured_output_model_validates_message_field(sample_actor_profile):
    """Init raises ValueError if structured_output_model has no message field."""

    class NoMessageModel(BaseModel):
        answer: str = ""
        stop: bool = False

    with pytest.raises(ValueError, match="must have a 'message' field"):
        ActorSimulator(
            actor_profile=sample_actor_profile,
            initial_query="Hello",
            system_prompt_template="Test: {actor_profile}",
            structured_output_model=NoMessageModel,
        )
