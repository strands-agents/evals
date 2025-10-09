from datetime import datetime

import pytest
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData
from strands_evals.types.trace import (
    AgentInvocationSpan,
    EvaluationLevel,
    Session,
    SpanInfo,
    ToolCall,
    ToolExecutionSpan,
    ToolLevelInput,
    ToolResult,
    Trace,
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
    span = ToolExecutionSpan(
        span_info=SpanInfo(session_id="test", span_id="1", start_time=now, end_time=now),
        tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}),
        tool_result=ToolResult(content="4"),
    )
    trace = Trace(spans=[span], trace_id="trace1", session_id="test")
    return Session(traces=[trace], session_id="test")


def test_parse_turn_level_with_conversation_history(session_with_conversation):
    evaluator = Evaluator()

    result = evaluator._parse_turn_level(session_with_conversation)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[1].agent_response.text == "6"
    assert len(result[1].conversation_history) == 3
    assert result[1].conversation_history[0].role.value == "user"
    assert result[1].conversation_history[0].content[0].text == "What is 2+2?"
    assert result[1].conversation_history[1].role.value == "assistant"
    assert result[1].conversation_history[1].content[0].text == "4"
    assert result[1].conversation_history[2].role.value == "user"
    assert result[1].conversation_history[2].content[0].text == "What is 3+3?"


def test_parse_turn_level_empty_session():
    evaluator = Evaluator()
    session = Session(traces=[], session_id="test")

    result = evaluator._parse_turn_level(session)

    assert isinstance(result, list)
    assert len(result) == 0


def test_parse_tool_level(session_with_tools):
    evaluator = Evaluator()

    result = evaluator._parse_tool_level(session_with_tools)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ToolLevelInput)
    assert result[0].tool_execution_details.tool_call.name == "calculator"
    assert result[0].tool_execution_details.tool_call.arguments == {"expression": "2+2"}
    assert result[0].tool_execution_details.tool_result.content == "4"


def test_parse_tool_level_empty_session():
    evaluator = Evaluator()
    session = Session(traces=[], session_id="test")

    result = evaluator._parse_tool_level(session)

    assert isinstance(result, list)
    assert len(result) == 0


def test_parse_trajectory_turn_level(session_with_conversation):
    evaluator = Evaluator()
    evaluator.evaluation_level = EvaluationLevel.TURN_LEVEL

    evaluation_data = EvaluationData(input="test", actual_output="output", actual_trajectory=session_with_conversation)

    result = evaluator._parse_trajectory(evaluation_data)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[1].agent_response.text == "6"


def test_parse_trajectory_tool_level(session_with_tools):
    evaluator = Evaluator()
    evaluator.evaluation_level = EvaluationLevel.TOOL_LEVEL

    evaluation_data = EvaluationData(input="Calculate 2+2", actual_output="4", actual_trajectory=session_with_tools)

    result = evaluator._parse_trajectory(evaluation_data)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].tool_execution_details.tool_call.name == "calculator"


def test_parse_trajectory_raises_on_invalid_type():
    evaluator = Evaluator()
    evaluator.evaluation_level = EvaluationLevel.TURN_LEVEL

    evaluation_data = EvaluationData(input="test", actual_output="output", actual_trajectory=["not", "a", "session"])

    with pytest.raises(TypeError, match="Trace parsing requires actual_trajectory to be a Session object"):
        evaluator._parse_trajectory(evaluation_data)


def test_parse_trajectory_raises_on_unsupported_level(session_with_conversation):
    evaluator = Evaluator()
    evaluator.evaluation_level = EvaluationLevel.CONVERSATION_LEVEL

    evaluation_data = EvaluationData(input="test", actual_output="output", actual_trajectory=session_with_conversation)

    with pytest.raises(ValueError, match="Unsupported evaluation level"):
        evaluator._parse_trajectory(evaluation_data)
