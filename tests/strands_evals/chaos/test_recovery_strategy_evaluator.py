"""Unit tests for RecoveryStrategyEvaluator."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators.chaos import RecoveryStrategyEvaluator
from strands_evals.evaluators.chaos.recovery_strategy_evaluator import (
    RecoveryStrategyRating,
    RecoveryStrategyScore,
)
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

    tool_configs = [
        ToolConfig(name="search_tool", description="Search for flights"),
        ToolConfig(name="fallback_search", description="Fallback search via cache"),
    ]

    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="Find flights to Tokyo",
        agent_response="The primary search is down, but I found cached results.",
        available_tools=tool_configs,
    )

    tool_span_fail = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="search_tool", arguments={"destination": "Tokyo"}, tool_call_id="1"),
        tool_result=ToolResult(content="Error: Connection timed out", tool_call_id="1"),
    )

    tool_span_fallback = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="fallback_search", arguments={"destination": "Tokyo"}, tool_call_id="2"),
        tool_result=ToolResult(content='[{"flight": "AA100", "price": 800}]', tool_call_id="2"),
    )

    trace = Trace(
        spans=[agent_span, tool_span_fail, tool_span_fallback],
        trace_id="trace1",
        session_id="test-session",
    )
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="Find flights to Tokyo",
        actual_output="The primary search is down, but I found cached results.",
        actual_trajectory=session,
        name="test-recovery-strategy",
    )


def test_init_with_defaults():
    evaluator = RecoveryStrategyEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt is not None
    assert evaluator.evaluation_level == EvaluationLevel.TRACE_LEVEL


def test_init_with_custom_values():
    evaluator = RecoveryStrategyEvaluator(version="v0", model="gpt-4", system_prompt="Custom")

    assert evaluator.version == "v0"
    assert evaluator.model == "gpt-4"
    assert evaluator.system_prompt == "Custom"


@patch("strands_evals.evaluators.chaos.recovery_strategy_evaluator.Agent")
def test_evaluate(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = RecoveryStrategyRating(
        reasoning="Agent quickly pivoted to fallback search after timeout",
        score=RecoveryStrategyScore.EXCELLENT,
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = RecoveryStrategyEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "Agent quickly pivoted to fallback search after timeout"
    assert result[0].label == RecoveryStrategyScore.EXCELLENT


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (RecoveryStrategyScore.EXCELLENT, 1.0, True),
        (RecoveryStrategyScore.GOOD, 0.75, True),
        (RecoveryStrategyScore.ACCEPTABLE, 0.5, True),
        (RecoveryStrategyScore.POOR, 0.25, False),
        (RecoveryStrategyScore.FAILURE, 0.0, False),
    ],
)
@patch("strands_evals.evaluators.chaos.recovery_strategy_evaluator.Agent")
def test_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = RecoveryStrategyRating(reasoning="Test", score=score)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = RecoveryStrategyEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


@pytest.mark.asyncio
@patch("strands_evals.evaluators.chaos.recovery_strategy_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = RecoveryStrategyRating(
        reasoning="Good recovery strategy", score=RecoveryStrategyScore.GOOD
    )

    async def mock_invoke_async(*args, **kwargs):
        return mock_result

    mock_agent.invoke_async = mock_invoke_async
    mock_agent_class.return_value = mock_agent
    evaluator = RecoveryStrategyEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 0.75
    assert result[0].test_pass is True
