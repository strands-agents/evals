from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators.coherence_evaluator import CoherenceEvaluator, CoherenceRating, CoherenceScore
from strands_evals.types import EvaluationData
from strands_evals.types.trace import (
    AgentInvocationSpan,
    EvaluationLevel,
    Session,
    SpanInfo,
    Trace,
)


@pytest.fixture
def evaluation_data():
    now = datetime.now()
    span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)
    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="Explain why the sky is blue.",
        agent_response="The sky appears blue due to Rayleigh scattering of sunlight by the atmosphere.",
        available_tools=[],
    )
    trace = Trace(spans=[agent_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="Explain why the sky is blue.",
        actual_output="The sky appears blue due to Rayleigh scattering of sunlight by the atmosphere.",
        actual_trajectory=session,
        name="test",
    )


def test_init_with_defaults():
    evaluator = CoherenceEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.include_inputs is True
    assert evaluator.system_prompt is not None
    assert evaluator.evaluation_level == EvaluationLevel.TRACE_LEVEL


def test_init_with_custom_values():
    evaluator = CoherenceEvaluator(version="v1", model="gpt-4", system_prompt="Custom", include_inputs=False)

    assert evaluator.version == "v1"
    assert evaluator.model == "gpt-4"
    assert evaluator.include_inputs is False
    assert evaluator.system_prompt == "Custom"


@patch("strands_evals.evaluators.coherence_evaluator.Agent")
def test_evaluate(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CoherenceRating(
        reasoning="The response is logically coherent with no contradictions", score=CoherenceScore.COMPLETELY_YES
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CoherenceEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The response is logically coherent with no contradictions"
    assert result[0].label == CoherenceScore.COMPLETELY_YES


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (CoherenceScore.NOT_AT_ALL, 0.0, False),
        (CoherenceScore.NOT_GENERALLY, 0.25, False),
        (CoherenceScore.NEUTRAL_MIXED, 0.5, True),
        (CoherenceScore.GENERALLY_YES, 0.75, True),
        (CoherenceScore.COMPLETELY_YES, 1.0, True),
    ],
)
@patch("strands_evals.evaluators.coherence_evaluator.Agent")
def test_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CoherenceRating(reasoning="Test reasoning", score=score)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CoherenceEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


@pytest.mark.asyncio
@patch("strands_evals.evaluators.coherence_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data):
    mock_agent = Mock()

    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        mock_result.structured_output = CoherenceRating(
            reasoning="The response is logically coherent with no contradictions", score=CoherenceScore.COMPLETELY_YES
        )
        return mock_result

    mock_agent.invoke_async = mock_invoke_async
    mock_agent_class.return_value = mock_agent
    evaluator = CoherenceEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The response is logically coherent with no contradictions"
    assert result[0].label == CoherenceScore.COMPLETELY_YES


def test_coherence_score_enum_values():
    """Test that CoherenceScore enum has all expected values."""
    assert CoherenceScore.NOT_AT_ALL.value == "Not At All"
    assert CoherenceScore.NOT_GENERALLY.value == "Not Generally"
    assert CoherenceScore.NEUTRAL_MIXED.value == "Neutral/Mixed"
    assert CoherenceScore.GENERALLY_YES.value == "Generally Yes"
    assert CoherenceScore.COMPLETELY_YES.value == "Completely Yes"


def test_score_mapping_completeness():
    """Test that all CoherenceScore enum values are in the score mapping."""
    evaluator = CoherenceEvaluator()
    for score in CoherenceScore:
        assert score in evaluator._score_mapping


@patch("strands_evals.evaluators.coherence_evaluator.Agent")
def test_evaluate_with_contradictory_response(mock_agent_class, evaluation_data):
    """Test evaluation of a response with contradictions."""
    # Modify evaluation data to have a contradictory response
    evaluation_data.actual_trajectory.traces[0].spans[
        0
    ].agent_response = "Exercise is good for health. However, exercise is bad for health."

    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CoherenceRating(
        reasoning="The response contains contradictions about exercise", score=CoherenceScore.NOT_AT_ALL
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CoherenceEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 0.0
    assert result[0].test_pass is False
    assert result[0].label == CoherenceScore.NOT_AT_ALL


@patch("strands_evals.evaluators.coherence_evaluator.Agent")
def test_evaluate_with_reasoning_gaps(mock_agent_class, evaluation_data):
    """Test evaluation of a response with reasoning gaps."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CoherenceRating(
        reasoning="The response has some logical gaps but the main point is clear", score=CoherenceScore.GENERALLY_YES
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CoherenceEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 0.75
    assert result[0].test_pass is True
    assert result[0].label == CoherenceScore.GENERALLY_YES


def test_coherence_rating_model():
    """Test the CoherenceRating Pydantic model."""
    rating = CoherenceRating(
        reasoning="This is a test reasoning with less than 250 words", score=CoherenceScore.COMPLETELY_YES
    )

    assert rating.reasoning == "This is a test reasoning with less than 250 words"
    assert rating.score == CoherenceScore.COMPLETELY_YES


def test_get_last_turn_with_empty_trajectory():
    """Test that _get_last_turn raises ValueError with empty trajectory."""
    evaluator = CoherenceEvaluator()
    trace = Trace(spans=[], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    evaluation_data = EvaluationData(input="Test", actual_output="Test", actual_trajectory=session, name="test")

    with pytest.raises(ValueError, match="No turn-level inputs could be parsed"):
        evaluator._get_last_turn(evaluation_data)


@patch("strands_evals.evaluators.coherence_evaluator.Agent")
def test_format_prompt_includes_conversation_history(mock_agent_class, evaluation_data):
    """Test that the formatted prompt includes conversation history."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CoherenceRating(reasoning="Test", score=CoherenceScore.COMPLETELY_YES)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent

    evaluator = CoherenceEvaluator()
    parsed_input = evaluator._get_last_turn(evaluation_data)
    formatted_prompt = evaluator._format_prompt(parsed_input)

    # Check that the prompt contains the expected sections
    assert "Target turn to evaluate:" in formatted_prompt
    assert "User:" in formatted_prompt
    assert "Assistant:" in formatted_prompt
    assert evaluation_data.input in formatted_prompt
    assert evaluation_data.actual_output in formatted_prompt


def test_extract_user_prompt_with_session_history():
    """Test _extract_user_prompt extracts the user prompt from session history."""
    evaluator = CoherenceEvaluator()
    now = datetime.now()
    span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)
    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="Test prompt",
        agent_response="Test response",
        available_tools=[],
    )
    trace = Trace(spans=[agent_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    evaluation_data = EvaluationData(
        input="Test prompt", actual_output="Test response", actual_trajectory=session, name="test"
    )

    parsed_input = evaluator._get_last_turn(evaluation_data)
    user_prompt = evaluator._extract_user_prompt(parsed_input)

    # Should extract the user prompt from session history
    assert user_prompt == "Test prompt"


@pytest.mark.parametrize(
    "model_input,expected_model",
    [
        (None, None),
        ("gpt-4", "gpt-4"),
        ("us.anthropic.claude-sonnet-4-20250514-v1:0", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
    ],
)
def test_init_with_different_models(model_input, expected_model):
    """Test initialization with different model types."""
    evaluator = CoherenceEvaluator(model=model_input)
    assert evaluator.model == expected_model


def test_system_prompt_contains_evaluation_criteria():
    """Test that the system prompt contains key evaluation criteria."""
    evaluator = CoherenceEvaluator()

    # Check for key criteria in the system prompt
    assert "Self-contradictions" in evaluator.system_prompt
    assert "Logic gaps or errors" in evaluator.system_prompt
    assert "Soundness of reasoning" in evaluator.system_prompt
    assert "Logical cohesion vs correctness" in evaluator.system_prompt
    assert "Relevance of logical reasoning" in evaluator.system_prompt

    # Check for scoring scale
    assert "Not At All" in evaluator.system_prompt
    assert "Not Generally" in evaluator.system_prompt
    assert "Neutral/Mixed" in evaluator.system_prompt
    assert "Generally Yes" in evaluator.system_prompt
    assert "Completely Yes" in evaluator.system_prompt

    # Check for important note
    assert "tool output ALWAYS takes priority" in evaluator.system_prompt
