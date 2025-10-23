from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import HelpfulnessEvaluator
from strands_evals.evaluators.helpfulness_evaluator import HelpfulnessRating, HelpfulnessScore
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
        span_info=span_info, user_prompt="What is the capital of France?", agent_response="Paris", available_tools=[]
    )
    trace = Trace(spans=[agent_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="What is the capital of France?", actual_output="Paris", actual_trajectory=session, name="test"
    )


def test_init_with_defaults():
    evaluator = HelpfulnessEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.include_inputs is True
    assert evaluator.system_prompt is not None
    assert evaluator.evaluation_level == EvaluationLevel.TURN_LEVEL


def test_init_with_custom_values():
    evaluator = HelpfulnessEvaluator(version="v1", model="gpt-4", system_prompt="Custom", include_inputs=False)

    assert evaluator.version == "v1"
    assert evaluator.model == "gpt-4"
    assert evaluator.include_inputs is False
    assert evaluator.system_prompt == "Custom"


@patch("strands_evals.evaluators.helpfulness_evaluator.Agent")
def test_evaluate(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_agent.structured_output.return_value = HelpfulnessRating(
        reasoning="The response is helpful", score=HelpfulnessScore.VERY_HELPFUL
    )
    mock_agent_class.return_value = mock_agent
    evaluator = HelpfulnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert result.score == 0.833
    assert result.test_pass is True
    assert result.reason == "The response is helpful"


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (HelpfulnessScore.NOT_HELPFUL, 0.0, False),
        (HelpfulnessScore.VERY_UNHELPFUL, 0.167, False),
        (HelpfulnessScore.SOMEWHAT_UNHELPFUL, 0.333, False),
        (HelpfulnessScore.NEUTRAL, 0.5, True),
        (HelpfulnessScore.SOMEWHAT_HELPFUL, 0.667, True),
        (HelpfulnessScore.VERY_HELPFUL, 0.833, True),
        (HelpfulnessScore.ABOVE_AND_BEYOND, 1.0, True),
    ],
)
@patch("strands_evals.evaluators.helpfulness_evaluator.Agent")
def test_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_agent.structured_output.return_value = HelpfulnessRating(reasoning="Test", score=score)
    mock_agent_class.return_value = mock_agent
    evaluator = HelpfulnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert result.score == expected_value
    assert result.test_pass == expected_pass


@pytest.mark.asyncio
@patch("strands_evals.evaluators.helpfulness_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data):
    mock_agent = Mock()

    async def mock_structured_output_async(*args, **kwargs):
        return HelpfulnessRating(reasoning="The response is helpful", score=HelpfulnessScore.VERY_HELPFUL)

    mock_agent.structured_output_async = mock_structured_output_async
    mock_agent_class.return_value = mock_agent
    evaluator = HelpfulnessEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert result.score == 0.833
    assert result.test_pass is True
    assert result.reason == "The response is helpful"
