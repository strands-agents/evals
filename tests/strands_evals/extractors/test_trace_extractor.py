from datetime import datetime

import pytest

from strands_evals.extractors import TraceExtractor
from strands_evals.types.trace import (
    AgentInvocationSpan,
    ConversationLevelInput,
    EvaluationLevel,
    Session,
    SpanInfo,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolLevelInput,
    ToolResult,
    Trace,
    TurnLevelInput,
)


@pytest.fixture
def session_with_conversation():
    now = datetime.now()
    span1 = AgentInvocationSpan(
        span_info=SpanInfo(session_id="test", span_id="1", start_time=now, end_time=now),
        user_prompt="What is 2+2?",
        agent_response="4",
        available_tools=[],
    )
    span2 = AgentInvocationSpan(
        span_info=SpanInfo(session_id="test", span_id="2", start_time=now, end_time=now),
        user_prompt="What is 3+3?",
        agent_response="6",
        available_tools=[],
    )
    trace = Trace(spans=[span1, span2], trace_id="trace1", session_id="test")
    return Session(traces=[trace], session_id="test")


@pytest.fixture
def session_with_tools():
    now = datetime.now()
    tool_config = ToolConfig(name="calculator", description="Calculate expressions")
    agent_span = AgentInvocationSpan(
        span_info=SpanInfo(session_id="test", span_id="0", start_time=now, end_time=now),
        user_prompt="Calculate 2+2",
        agent_response="The answer is 4",
        available_tools=[tool_config],
    )
    tool_span = ToolExecutionSpan(
        span_info=SpanInfo(session_id="test", span_id="1", start_time=now, end_time=now),
        tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}),
        tool_result=ToolResult(content="4"),
    )
    trace = Trace(spans=[agent_span, tool_span], trace_id="trace1", session_id="test")
    return Session(traces=[trace], session_id="test")


def test_trace_extractor_initialization():
    extractor = TraceExtractor(EvaluationLevel.TURN_LEVEL)
    assert extractor.evaluation_level == EvaluationLevel.TURN_LEVEL


def test_extract_turn_level(session_with_conversation):
    extractor = TraceExtractor(EvaluationLevel.TURN_LEVEL)
    result = extractor.extract(session_with_conversation)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(item, TurnLevelInput) for item in result)
    assert result[0].agent_response.text == "4"
    assert result[1].agent_response.text == "6"


def test_extract_turn_level_with_conversation_history(session_with_conversation):
    """Test that conversation history accumulates correctly across turns."""
    extractor = TraceExtractor(EvaluationLevel.TURN_LEVEL)
    result = extractor.extract(session_with_conversation)

    assert len(result) == 2
    # Second turn should have history of first turn
    assert result[1].agent_response.text == "6"
    assert len(result[1].conversation_history) == 3
    assert result[1].conversation_history[0].role.value == "user"
    assert result[1].conversation_history[0].content[0].text == "What is 2+2?"
    assert result[1].conversation_history[1].role.value == "assistant"
    assert result[1].conversation_history[1].content[0].text == "4"
    assert result[1].conversation_history[2].role.value == "user"
    assert result[1].conversation_history[2].content[0].text == "What is 3+3?"


def test_extract_tool_level(session_with_tools):
    extractor = TraceExtractor(EvaluationLevel.TOOL_LEVEL)
    result = extractor.extract(session_with_tools)

    assert isinstance(result, list)
    assert len(result) == 1
    assert all(isinstance(item, ToolLevelInput) for item in result)
    assert result[0].tool_execution_details.tool_call.name == "calculator"
    assert result[0].tool_execution_details.tool_call.arguments == {"expression": "2+2"}
    assert result[0].tool_execution_details.tool_result.content == "4"


def test_extract_conversation_level(session_with_conversation):
    extractor = TraceExtractor(EvaluationLevel.CONVERSATION_LEVEL)
    result = extractor.extract(session_with_conversation)

    assert isinstance(result, ConversationLevelInput)
    assert len(result.conversation_history) == 2
    assert result.conversation_history[0].user_prompt.text == "What is 2+2?"
    assert result.conversation_history[0].agent_response.text == "4"
    assert result.conversation_history[1].user_prompt.text == "What is 3+3?"
    assert result.conversation_history[1].agent_response.text == "6"


def test_extract_raises_on_invalid_session_type():
    extractor = TraceExtractor(EvaluationLevel.TURN_LEVEL)

    with pytest.raises(TypeError, match="Expected Session object"):
        extractor.extract(["not", "a", "session"])


def test_extract_raises_on_unsupported_level():
    with pytest.raises(ValueError, match="Unsupported evaluation level"):
        extractor = TraceExtractor("INVALID_LEVEL")
        extractor.extract(Session(traces=[], session_id="test"))


def test_composability_multiple_extractors(session_with_conversation):
    """Test that multiple extractors can be composed for different purposes."""
    turn_extractor = TraceExtractor(EvaluationLevel.TURN_LEVEL)
    conversation_extractor = TraceExtractor(EvaluationLevel.CONVERSATION_LEVEL)

    turn_result = turn_extractor.extract(session_with_conversation)
    conversation_result = conversation_extractor.extract(session_with_conversation)

    assert len(turn_result) == 2
    assert len(conversation_result.conversation_history) == 2


def test_extract_empty_session_turn_level():
    extractor = TraceExtractor(EvaluationLevel.TURN_LEVEL)
    session = Session(traces=[], session_id="test")

    result = extractor.extract(session)

    assert isinstance(result, list)
    assert len(result) == 0


def test_extract_empty_session_tool_level():
    extractor = TraceExtractor(EvaluationLevel.TOOL_LEVEL)
    session = Session(traces=[], session_id="test")

    result = extractor.extract(session)

    assert isinstance(result, list)
    assert len(result) == 0
