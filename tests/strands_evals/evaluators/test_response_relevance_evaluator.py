"""Tests for ResponseRelevanceEvaluator."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import ResponseRelevanceEvaluator
from strands_evals.evaluators.response_relevance_evaluator import (
    ResponseRelevanceRating,
    ResponseRelevanceScore,
)
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
        user_prompt="What is the capital of France?",
        agent_response="The capital of France is Paris.",
        available_tools=[],
    )
    trace = Trace(spans=[agent_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        actual_trajectory=session,
        name="test",
    )


def test_init_with_defaults():
    evaluator = ResponseRelevanceEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.include_inputs is True
    assert evaluator.system_prompt is not None
    assert evaluator.evaluation_level == EvaluationLevel.TRACE_LEVEL


def test_init_with_custom_values():
    evaluator = ResponseRelevanceEvaluator(version="v1", model="gpt-4", system_prompt="Custom", include_inputs=False)

    assert evaluator.version == "v1"
    assert evaluator.model == "gpt-4"
    assert evaluator.include_inputs is False
    assert evaluator.system_prompt == "Custom"


@patch("strands_evals.evaluators.response_relevance_evaluator.Agent")
def test_evaluate(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ResponseRelevanceRating(
        reasoning="The response directly answers the question", score=ResponseRelevanceScore.COMPLETELY_YES
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ResponseRelevanceEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The response directly answers the question"
    assert result[0].label == ResponseRelevanceScore.COMPLETELY_YES


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (ResponseRelevanceScore.NOT_AT_ALL, 0.0, False),
        (ResponseRelevanceScore.NOT_GENERALLY, 0.25, False),
        (ResponseRelevanceScore.NEUTRAL_MIXED, 0.5, True),
        (ResponseRelevanceScore.GENERALLY_YES, 0.75, True),
        (ResponseRelevanceScore.COMPLETELY_YES, 1.0, True),
    ],
)
@patch("strands_evals.evaluators.response_relevance_evaluator.Agent")
def test_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ResponseRelevanceRating(reasoning="Test", score=score)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ResponseRelevanceEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


@pytest.mark.asyncio
@patch("strands_evals.evaluators.response_relevance_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ResponseRelevanceRating(
        reasoning="The response directly answers the question", score=ResponseRelevanceScore.COMPLETELY_YES
    )

    async def mock_invoke_async(*args, **kwargs):
        return mock_result

    mock_agent.invoke_async = mock_invoke_async
    mock_agent_class.return_value = mock_agent
    evaluator = ResponseRelevanceEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The response directly answers the question"
    assert result[0].label == ResponseRelevanceScore.COMPLETELY_YES
