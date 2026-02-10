from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from strands_evals.evaluators import FaithfulnessEvaluator
from strands_evals.evaluators.faithfulness_evaluator import FaithfulnessRating, FaithfulnessScore
from strands_evals.types import EvaluationData
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    EvaluationLevel,
    InferenceSpan,
    Role,
    Session,
    SpanInfo,
    SpanType,
    TextContent,
    ToolCall,
    ToolCallContent,
    ToolExecutionSpan,
    ToolResult,
    Trace,
    UserMessage,
)


@pytest.fixture
def evaluation_data():
    now = datetime.now()

    # Add inference span with tool call
    inference_span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)
    user_msg = UserMessage(role=Role.USER, content=[TextContent(text="What is the capital of France?")])
    assistant_msg = AssistantMessage(
        role=Role.ASSISTANT,
        content=[ToolCallContent(name="search_web", arguments={"query": "capital of France"}, tool_call_id="call_1")],
    )
    inference_span = InferenceSpan(span_info=inference_span_info, messages=[user_msg, assistant_msg])

    # Add tool execution span
    tool_span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)
    tool_call = ToolCall(name="search_web", arguments={"query": "capital of France"}, tool_call_id="call_1")
    tool_result = ToolResult(content="Paris is the capital and most populous city of France", tool_call_id="call_1")
    tool_span = ToolExecutionSpan(
        span_info=tool_span_info, span_type=SpanType.TOOL_EXECUTION, tool_call=tool_call, tool_result=tool_result
    )

    # Add agent invocation span (required for TRACE_LEVEL evaluators)
    agent_span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)
    agent_span = AgentInvocationSpan(
        span_info=agent_span_info,
        user_prompt="What is the capital of France?",
        agent_response="Paris",
        available_tools=[],
    )

    trace = Trace(spans=[inference_span, tool_span, agent_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="What is the capital of France?", actual_output="Paris", actual_trajectory=session, name="test"
    )


def test_init_with_defaults():
    evaluator = FaithfulnessEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt is not None
    assert evaluator.evaluation_level == EvaluationLevel.TRACE_LEVEL


def test_init_with_custom_values():
    evaluator = FaithfulnessEvaluator(version="v1", model="gpt-4", system_prompt="Custom")

    assert evaluator.version == "v1"
    assert evaluator.model == "gpt-4"
    assert evaluator.system_prompt == "Custom"


@patch("strands_evals.evaluators.faithfulness_evaluator.Agent")
def test_evaluate(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = FaithfulnessRating(
        reasoning="The response is faithful", score=FaithfulnessScore.COMPLETELY_YES
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = FaithfulnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The response is faithful"
    assert result[0].label == FaithfulnessScore.COMPLETELY_YES

    # Verify that the prompt includes tool execution information
    mock_agent.assert_called_once()
    call_args = mock_agent.call_args[0]
    prompt = call_args[0]
    assert "Action: search_web(" in prompt
    assert "Tool: Paris is the capital and most populous city of France" in prompt


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (FaithfulnessScore.NOT_AT_ALL, 0.0, False),
        (FaithfulnessScore.NOT_GENERALLY, 0.25, False),
        (FaithfulnessScore.NEUTRAL, 0.5, True),
        (FaithfulnessScore.GENERALLY_YES, 0.75, True),
        (FaithfulnessScore.COMPLETELY_YES, 1.0, True),
    ],
)
@patch("strands_evals.evaluators.faithfulness_evaluator.Agent")
def test_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = FaithfulnessRating(reasoning="Test", score=score)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = FaithfulnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


@pytest.mark.asyncio
@patch("strands_evals.evaluators.faithfulness_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = FaithfulnessRating(
        reasoning="The response is faithful", score=FaithfulnessScore.COMPLETELY_YES
    )

    # Create a proper async mock that returns the result
    mock_agent.invoke_async = AsyncMock(return_value=mock_result)
    mock_agent_class.return_value = mock_agent
    evaluator = FaithfulnessEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The response is faithful"
    assert result[0].label == FaithfulnessScore.COMPLETELY_YES

    # Verify that the prompt includes tool execution information
    mock_agent.invoke_async.assert_called_once()
    call_args = mock_agent.invoke_async.call_args[0]
    prompt = call_args[0]
    assert "Action: search_web(" in prompt
    assert "Tool: Paris is the capital and most populous city of France" in prompt
