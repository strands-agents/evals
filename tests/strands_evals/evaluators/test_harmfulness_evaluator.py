from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import HarmfulnessEvaluator
from strands_evals.evaluators.harmfulness_evaluator import HarmfulnessRating, HarmfulnessScore
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
        user_prompt="What is the weather like today?",
        agent_response="The weather is sunny and warm.",
        available_tools=[],
    )
    trace = Trace(spans=[agent_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="What is the weather like today?",
        actual_output="The weather is sunny and warm.",
        actual_trajectory=session,
        name="test",
    )


def test_init_with_defaults():
    evaluator = HarmfulnessEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt is not None
    assert evaluator.evaluation_level == EvaluationLevel.TRACE_LEVEL


def test_init_with_custom_values():
    evaluator = HarmfulnessEvaluator(version="v0", model="gpt-4", system_prompt="Custom")

    assert evaluator.version == "v0"
    assert evaluator.model == "gpt-4"
    assert evaluator.system_prompt == "Custom"


@patch("strands_evals.evaluators.harmfulness_evaluator.Agent")
def test_evaluate(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = HarmfulnessRating(
        reasoning="The response is safe and helpful", score=HarmfulnessScore.NOT_HARMFUL
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = HarmfulnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The response is safe and helpful"
    assert result[0].label == HarmfulnessScore.NOT_HARMFUL


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (HarmfulnessScore.NOT_HARMFUL, 1.0, True),
        (HarmfulnessScore.HARMFUL, 0.0, False),
    ],
)
@patch("strands_evals.evaluators.harmfulness_evaluator.Agent")
def test_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = HarmfulnessRating(reasoning="Test", score=score)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = HarmfulnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


@pytest.mark.asyncio
@patch("strands_evals.evaluators.harmfulness_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data):
    mock_agent = Mock()

    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        mock_result.structured_output = HarmfulnessRating(
            reasoning="The response is safe and helpful", score=HarmfulnessScore.NOT_HARMFUL
        )
        return mock_result

    mock_agent.invoke_async = mock_invoke_async
    mock_agent_class.return_value = mock_agent
    evaluator = HarmfulnessEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The response is safe and helpful"
    assert result[0].label == HarmfulnessScore.NOT_HARMFUL
