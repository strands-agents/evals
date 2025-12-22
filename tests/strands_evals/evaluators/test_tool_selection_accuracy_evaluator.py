from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import ToolSelectionAccuracyEvaluator
from strands_evals.evaluators.tool_selection_accuracy_evaluator import ToolSelectionRating, ToolSelectionScore
from strands_evals.types import EvaluationData
from strands_evals.types.trace import (
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

    inference_span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)
    user_msg = UserMessage(role=Role.USER, content=[TextContent(text="What's the weather?")])
    assistant_msg = AssistantMessage(
        role=Role.ASSISTANT,
        content=[ToolCallContent(name="get_weather", arguments={}, tool_call_id="call_1")],
    )
    inference_span = InferenceSpan(span_info=inference_span_info, messages=[user_msg, assistant_msg])

    tool_span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)
    tool_call = ToolCall(name="get_weather", arguments={}, tool_call_id="call_1")
    tool_result = ToolResult(content="Sunny", tool_call_id="call_1")
    tool_span = ToolExecutionSpan(
        span_info=tool_span_info, span_type=SpanType.TOOL_EXECUTION, tool_call=tool_call, tool_result=tool_result
    )

    trace = Trace(spans=[inference_span, tool_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(input="What's the weather?", actual_output="Sunny", actual_trajectory=session, name="test")


def test_init_with_defaults():
    evaluator = ToolSelectionAccuracyEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt is not None
    assert evaluator.evaluation_level == EvaluationLevel.TOOL_LEVEL


def test_init_with_custom_values():
    evaluator = ToolSelectionAccuracyEvaluator(version="v1", model="gpt-4", system_prompt="Custom")

    assert evaluator.version == "v1"
    assert evaluator.model == "gpt-4"
    assert evaluator.system_prompt == "Custom"


@patch("strands_evals.evaluators.tool_selection_accuracy_evaluator.Agent")
def test_evaluate(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolSelectionRating(
        reasoning="Tool call is appropriate", score=ToolSelectionScore.YES
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ToolSelectionAccuracyEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "Tool call is appropriate"
    assert result[0].label == ToolSelectionScore.YES


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (ToolSelectionScore.YES, 1.0, True),
        (ToolSelectionScore.NO, 0.0, False),
    ],
)
@patch("strands_evals.evaluators.tool_selection_accuracy_evaluator.Agent")
def test_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolSelectionRating(reasoning="Test", score=score)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ToolSelectionAccuracyEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


@pytest.mark.asyncio
@patch("strands_evals.evaluators.tool_selection_accuracy_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data):
    mock_agent = Mock()

    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        mock_result.structured_output = ToolSelectionRating(
            reasoning="Tool call is appropriate", score=ToolSelectionScore.YES
        )
        return mock_result

    mock_agent.invoke_async = mock_invoke_async
    mock_agent_class.return_value = mock_agent
    evaluator = ToolSelectionAccuracyEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "Tool call is appropriate"
    assert result[0].label == ToolSelectionScore.YES
