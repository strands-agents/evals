from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import InstructionFollowingEvaluator
from strands_evals.evaluators.instruction_following_evaluator import (
    InstructionFollowingRating,
    InstructionFollowingScore,
)
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

    inference_span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)
    user_msg = UserMessage(
        role=Role.USER, content=[TextContent(text="Summarize this text in one sentence: The quick brown fox.")]
    )
    assistant_msg = AssistantMessage(
        role=Role.ASSISTANT,
        content=[ToolCallContent(name="summarize", arguments={"text": "The quick brown fox."}, tool_call_id="call_1")],
    )
    inference_span = InferenceSpan(span_info=inference_span_info, messages=[user_msg, assistant_msg])

    tool_span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)
    tool_call = ToolCall(name="summarize", arguments={"text": "The quick brown fox."}, tool_call_id="call_1")
    tool_result = ToolResult(content="A fox jumps quickly.", tool_call_id="call_1")
    tool_span = ToolExecutionSpan(
        span_info=tool_span_info, span_type=SpanType.TOOL_EXECUTION, tool_call=tool_call, tool_result=tool_result
    )

    agent_span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)
    agent_span = AgentInvocationSpan(
        span_info=agent_span_info,
        user_prompt="Summarize this text in one sentence: The quick brown fox.",
        agent_response="A fox jumps quickly.",
        available_tools=[],
    )

    trace = Trace(spans=[inference_span, tool_span, agent_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="Summarize this text in one sentence: The quick brown fox.",
        actual_output="A fox jumps quickly.",
        actual_trajectory=session,
        name="test",
    )


def test_init_with_defaults():
    evaluator = InstructionFollowingEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt is not None
    assert evaluator.evaluation_level == EvaluationLevel.TRACE_LEVEL


def test_init_with_custom_values():
    evaluator = InstructionFollowingEvaluator(version="v0", model="gpt-4", system_prompt="Custom")

    assert evaluator.version == "v0"
    assert evaluator.model == "gpt-4"
    assert evaluator.system_prompt == "Custom"


@patch("strands_evals.evaluators.instruction_following_evaluator.Agent")
def test_evaluate_instructions_followed(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = InstructionFollowingRating(
        reasoning="The response summarizes the text in one sentence as requested",
        score=InstructionFollowingScore.YES,
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = InstructionFollowingEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The response summarizes the text in one sentence as requested"
    assert result[0].label == InstructionFollowingScore.YES


@patch("strands_evals.evaluators.instruction_following_evaluator.Agent")
def test_evaluate_instructions_not_followed(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = InstructionFollowingRating(
        reasoning="The response does not follow the one-sentence constraint",
        score=InstructionFollowingScore.NO,
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = InstructionFollowingEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 0.0
    assert result[0].test_pass is False
    assert result[0].label == InstructionFollowingScore.NO


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (InstructionFollowingScore.YES, 1.0, True),
        (InstructionFollowingScore.NO, 0.0, False),
    ],
)
@patch("strands_evals.evaluators.instruction_following_evaluator.Agent")
def test_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = InstructionFollowingRating(reasoning="Test", score=score)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = InstructionFollowingEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


@pytest.mark.asyncio
@patch("strands_evals.evaluators.instruction_following_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = InstructionFollowingRating(
        reasoning="The response summarizes the text in one sentence as requested",
        score=InstructionFollowingScore.YES,
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = InstructionFollowingEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
