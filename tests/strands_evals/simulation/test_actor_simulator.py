"""Tests for ActorSimulator class."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from strands.agent.agent_result import AgentResult

from strands_evals import Case
from strands_evals.simulation import ActorSimulator
from strands_evals.types.simulation import ActorProfile, ActorResponse, SimulatorResult


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


# ---------------------------------------------------------------------------
# input_type + act_structured()
# ---------------------------------------------------------------------------


class _AgentInput(BaseModel):
    """Sample input schema for input_type tests."""

    query: str
    urgency: str = "normal"


def test_init_without_input_type_uses_simulator_result(sample_actor_profile):
    """Without input_type, act_structured() hands SimulatorResult to the underlying agent."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = SimulatorResult(reasoning="r", stop=False, message="hi")
    simulator.agent = MagicMock(return_value=mock_response)

    simulator.act_structured("agent reply")

    assert simulator.agent.call_args[1]["structured_output_model"] is SimulatorResult


def test_init_with_input_type_narrows_message_schema(sample_actor_profile):
    """With input_type, act_structured() hands a SimulatorResult subclass whose message is typed."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
        input_type=_AgentInput,
    )

    typed_message = _AgentInput(query="hi")
    captured_model: list = []

    def _capture(agent_message, *, structured_output_model):
        captured_model.append(structured_output_model)
        mock_response = MagicMock(spec=AgentResult)
        mock_response.structured_output = structured_output_model(reasoning="r", stop=False, message=typed_message)
        return mock_response

    simulator.agent = MagicMock(side_effect=_capture)

    simulator.act_structured("agent reply")

    (model_used,) = captured_model
    # It must be a SimulatorResult subclass (so act_structured can return SimulatorResult).
    assert issubclass(model_used, SimulatorResult)
    # And it must schema-accept input_type instances on `message`.
    msg_field = model_used.model_fields["message"]
    assert msg_field.annotation == _AgentInput | None


def test_act_structured_passes_structured_model_to_agent(sample_actor_profile):
    """act_structured() reuses the same structured model across turns (cached at construction)."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
        input_type=_AgentInput,
    )

    seen_models: list = []

    def _capture(agent_message, *, structured_output_model):
        seen_models.append(structured_output_model)
        mock_response = MagicMock(spec=AgentResult)
        mock_response.structured_output = structured_output_model(
            reasoning="r", stop=False, message=_AgentInput(query="x")
        )
        return mock_response

    simulator.agent = MagicMock(side_effect=_capture)

    simulator.act_structured("turn 1")
    simulator.act_structured("turn 2")

    assert len(seen_models) == 2
    assert seen_models[0] is seen_models[1]  # same cached class across turns


def test_act_structured_continuing_turn(sample_actor_profile):
    """act_structured() returns SimulatorResult with stop=False for normal continuing turns."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = SimulatorResult(reasoning="thinking", stop=False, message="keep going")
    simulator.agent = MagicMock(return_value=mock_response)

    result = simulator.act_structured("agent reply")

    assert isinstance(result, SimulatorResult)
    assert result.message == "keep going"
    assert result.reasoning == "thinking"
    assert result.stop is False
    assert result.stop_reason is None


def test_act_structured_explicit_stop(sample_actor_profile):
    """act_structured() records stop_reason='goal_completed' when the LLM sets stop=True."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = SimulatorResult(reasoning="done", stop=True, message=None)
    simulator.agent = MagicMock(return_value=mock_response)

    result = simulator.act_structured("agent reply")

    assert result.stop is True
    assert result.stop_reason == "goal_completed"


def test_act_structured_hits_max_turns(sample_actor_profile):
    """act_structured() reports max_turns when the cap trips while LLM said stop=False."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
        max_turns=1,
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = SimulatorResult(reasoning="r", stop=False, message="more please")
    simulator.agent = MagicMock(return_value=mock_response)

    result = simulator.act_structured("agent reply")

    assert result.stop is True
    assert result.stop_reason == "max_turns"


def test_act_structured_input_type_returns_typed_message(sample_actor_profile):
    """act_structured() with input_type returns the typed message instance."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
        input_type=_AgentInput,
    )

    typed_message = _AgentInput(query="ship it", urgency="high")
    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = simulator._structured_model(reasoning="r", stop=False, message=typed_message)
    simulator.agent = MagicMock(return_value=mock_response)

    result = simulator.act_structured("agent reply")

    assert result.stop is False
    assert result.stop_reason is None
    assert isinstance(result.message, _AgentInput)
    assert result.message.query == "ship it"


def test_act_structured_input_type_null_message_becomes_implicit_stop(sample_actor_profile):
    """act_structured() with input_type treats null message + stop=False as implicit goal_completed."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
        input_type=_AgentInput,
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = simulator._structured_model(reasoning="r", stop=False, message=None)
    simulator.agent = MagicMock(return_value=mock_response)

    result = simulator.act_structured("agent reply")

    assert result.stop is True
    assert result.stop_reason == "goal_completed"
    assert result.message is None


def test_act_does_not_use_input_type(sample_actor_profile):
    """act() ignores input_type and uses the legacy ActorResponse schema."""
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template="Test: {actor_profile}",
        input_type=_AgentInput,
    )

    mock_response = MagicMock(spec=AgentResult)
    mock_response.structured_output = ActorResponse(reasoning="r", message="m")
    simulator.agent = MagicMock(return_value=mock_response)

    simulator.act("Test message")

    assert simulator.agent.call_args[1]["structured_output_model"] == ActorResponse


def test_rejects_simulator_result_as_input_type(sample_actor_profile):
    """Passing SimulatorResult (or a subclass) as input_type raises at __init__."""
    with pytest.raises(ValueError, match="SimulatorResult"):
        ActorSimulator(
            actor_profile=sample_actor_profile,
            initial_query="Hello",
            system_prompt_template="Test: {actor_profile}",
            input_type=SimulatorResult,
        )


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


def test_system_prompt_template_autopicks_structured_when_input_type_set(sample_actor_profile):
    """When input_type is set and no template is given, the structured-stop template is used."""
    from strands_evals.simulation.prompt_templates.actor_system_prompt import (
        STRUCTURED_USER_SIMULATOR_PROMPT_TEMPLATE,
    )

    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        input_type=_AgentInput,
    )

    expected = STRUCTURED_USER_SIMULATOR_PROMPT_TEMPLATE.format(actor_profile=sample_actor_profile.model_dump())
    assert simulator.agent.system_prompt == expected


def test_explicit_template_overrides_autopick(sample_actor_profile):
    """Explicit system_prompt_template wins even when input_type is set."""
    custom = "Custom prompt for {actor_profile}"
    simulator = ActorSimulator(
        actor_profile=sample_actor_profile,
        initial_query="Hello",
        system_prompt_template=custom,
        input_type=_AgentInput,
    )

    assert simulator.agent.system_prompt == custom.format(actor_profile=sample_actor_profile.model_dump())
