"""Unit tests for PartialCompletionEvaluator."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators.chaos import PartialCompletionEvaluator
from strands_evals.evaluators.chaos.partial_completion_evaluator import (
    PartialCompletionRating,
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
        ToolConfig(name="booking_tool", description="Book a flight"),
    ]

    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="Find and book a flight to Tokyo",
        agent_response="I found flights but couldn't complete the booking due to a service error.",
        available_tools=tool_configs,
    )

    tool_span_success = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="search_tool", arguments={"destination": "Tokyo"}, tool_call_id="1"),
        tool_result=ToolResult(content='[{"flight": "AA100", "price": 800}]', tool_call_id="1"),
    )

    tool_span_failure = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="booking_tool", arguments={"flight": "AA100"}, tool_call_id="2"),
        tool_result=ToolResult(content="Error: Service unavailable", tool_call_id="2"),
    )

    trace = Trace(
        spans=[agent_span, tool_span_success, tool_span_failure],
        trace_id="trace1",
        session_id="test-session",
    )
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="Find and book a flight to Tokyo",
        actual_output="I found flights but couldn't complete the booking due to a service error.",
        actual_trajectory=session,
        name="test-partial-completion",
    )


def test_init_with_defaults():
    evaluator = PartialCompletionEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt is not None
    assert evaluator.evaluation_level == EvaluationLevel.TRACE_LEVEL


def test_init_with_custom_values():
    evaluator = PartialCompletionEvaluator(version="v0", model="gpt-4", system_prompt="Custom")

    assert evaluator.version == "v0"
    assert evaluator.model == "gpt-4"
    assert evaluator.system_prompt == "Custom"


@patch("strands_evals.evaluators.chaos.partial_completion_evaluator.Agent")
def test_evaluate(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = PartialCompletionRating(
        reasoning="Search succeeded but booking failed — user got flight info but no reservation",
        completion_percentage=0.4,
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = PartialCompletionEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 0.4
    assert result[0].test_pass is False
    assert result[0].reason == "Search succeeded but booking failed — user got flight info but no reservation"
    assert result[0].label == "0.40"


@pytest.mark.parametrize(
    "completion,expected_pass",
    [
        (0.0, False),
        (0.49, False),
        (0.5, True),
        (1.0, True),
    ],
)
@patch("strands_evals.evaluators.chaos.partial_completion_evaluator.Agent")
def test_pass_threshold(mock_agent_class, evaluation_data, completion, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = PartialCompletionRating(reasoning="Test", completion_percentage=completion)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = PartialCompletionEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert result[0].score == completion
    assert result[0].test_pass == expected_pass


@pytest.mark.asyncio
@patch("strands_evals.evaluators.chaos.partial_completion_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = PartialCompletionRating(reasoning="Partial completion", completion_percentage=0.6)

    async def mock_invoke_async(*args, **kwargs):
        return mock_result

    mock_agent.invoke_async = mock_invoke_async
    mock_agent_class.return_value = mock_agent
    evaluator = PartialCompletionEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 0.6
    assert result[0].test_pass is True
