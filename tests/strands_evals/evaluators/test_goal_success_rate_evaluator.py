from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import GoalSuccessRateEvaluator
from strands_evals.evaluators.goal_success_rate_evaluator import GoalSuccessRating, GoalSuccessScore
from strands_evals.types import EvaluationData
from strands_evals.types.trace import (
    AgentInvocationSpan,
    EvaluationLevel,
    Session,
    SpanInfo,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)


@pytest.fixture
def evaluation_data():
    now = datetime.now()
    span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)

    tool_config = ToolConfig(name="calculator", description="Evaluate mathematical expressions")

    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="What is 2 + 2?",
        agent_response="The answer is 4.",
        available_tools=[tool_config],
    )

    tool_span = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}, tool_call_id="1"),
        tool_result=ToolResult(content="4", tool_call_id="1"),
    )

    trace = Trace(spans=[agent_span, tool_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="What is 2 + 2?", actual_output="The answer is 4.", actual_trajectory=session, name="test"
    )


def test_init_with_defaults():
    evaluator = GoalSuccessRateEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt is not None
    assert evaluator.evaluation_level == EvaluationLevel.SESSION_LEVEL


def test_init_with_custom_values():
    evaluator = GoalSuccessRateEvaluator(version="v1", model="gpt-4", system_prompt="Custom")

    assert evaluator.version == "v1"
    assert evaluator.model == "gpt-4"
    assert evaluator.system_prompt == "Custom"


@patch("strands_evals.evaluators.goal_success_rate_evaluator.Agent")
def test_evaluate(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_agent.structured_output.return_value = GoalSuccessRating(
        reasoning="All goals achieved", score=GoalSuccessScore.YES
    )
    mock_agent_class.return_value = mock_agent
    evaluator = GoalSuccessRateEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "All goals achieved"
    assert result[0].label == GoalSuccessScore.YES


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (GoalSuccessScore.YES, 1.0, True),
        (GoalSuccessScore.NO, 0.0, False),
    ],
)
@patch("strands_evals.evaluators.goal_success_rate_evaluator.Agent")
def test_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_agent.structured_output.return_value = GoalSuccessRating(reasoning="Test", score=score)
    mock_agent_class.return_value = mock_agent
    evaluator = GoalSuccessRateEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


@pytest.mark.asyncio
@patch("strands_evals.evaluators.goal_success_rate_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data):
    mock_agent = Mock()

    async def mock_structured_output_async(*args, **kwargs):
        return GoalSuccessRating(reasoning="All goals achieved", score=GoalSuccessScore.YES)

    mock_agent.structured_output_async = mock_structured_output_async
    mock_agent_class.return_value = mock_agent
    evaluator = GoalSuccessRateEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "All goals achieved"
    assert result[0].label == GoalSuccessScore.YES
