from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import GoalSuccessRateEvaluator
from strands_evals.evaluators.goal_success_rate_evaluator import (
    GoalSuccessAssertionRating,
    GoalSuccessAssertionScore,
    GoalSuccessRating,
    GoalSuccessScore,
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
    assert evaluator.assertion_system_prompt is not None
    assert evaluator.assertion_system_prompt != evaluator.system_prompt
    assert evaluator.evaluation_level == EvaluationLevel.SESSION_LEVEL


def test_init_with_custom_values():
    evaluator = GoalSuccessRateEvaluator(
        version="v1", model="gpt-4", system_prompt="Custom", assertion_system_prompt="Custom assertion"
    )

    assert evaluator.version == "v1"
    assert evaluator.model == "gpt-4"
    assert evaluator.system_prompt == "Custom"
    assert evaluator.assertion_system_prompt == "Custom assertion"


@patch("strands_evals.evaluators.goal_success_rate_evaluator.Agent")
def test_evaluate(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = GoalSuccessRating(reasoning="All goals achieved", score=GoalSuccessScore.YES)
    mock_agent.return_value = mock_result
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
    mock_result = Mock()
    mock_result.structured_output = GoalSuccessRating(reasoning="Test", score=score)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = GoalSuccessRateEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


@pytest.fixture
def evaluation_data_with_assertion():
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
        input="What is 2 + 2?",
        actual_output="The answer is 4.",
        actual_trajectory=session,
        name="test-criteria",
        expected_assertion="The agent should use the calculator tool and return the correct answer of 4.",
    )


def test_has_assertion_true(evaluation_data_with_assertion):
    evaluator = GoalSuccessRateEvaluator()
    assert evaluator._has_assertion(evaluation_data_with_assertion) is True


def test_has_assertion_false(evaluation_data):
    evaluator = GoalSuccessRateEvaluator()
    assert evaluator._has_assertion(evaluation_data) is False


def test_has_assertion_none():
    data = EvaluationData(input="test")
    evaluator = GoalSuccessRateEvaluator()
    assert evaluator._has_assertion(data) is False


def test_has_assertion_empty_string():
    data = EvaluationData(input="test", expected_assertion="")
    evaluator = GoalSuccessRateEvaluator()
    assert evaluator._has_assertion(data) is False


@patch("strands_evals.evaluators.goal_success_rate_evaluator.Agent")
def test_evaluate_with_assertion(mock_agent_class, evaluation_data_with_assertion):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = GoalSuccessAssertionRating(
        reasoning="Agent used calculator and returned 4", verdict=GoalSuccessAssertionScore.SUCCESS
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = GoalSuccessRateEvaluator()

    result = evaluator.evaluate(evaluation_data_with_assertion)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "Agent used calculator and returned 4"
    assert result[0].label == GoalSuccessAssertionScore.SUCCESS


@pytest.mark.parametrize(
    "verdict,expected_value,expected_pass",
    [
        (GoalSuccessAssertionScore.SUCCESS, 1.0, True),
        (GoalSuccessAssertionScore.FAILURE, 0.0, False),
    ],
)
@patch("strands_evals.evaluators.goal_success_rate_evaluator.Agent")
def test_assertion_score_mapping(
    mock_agent_class, evaluation_data_with_assertion, verdict, expected_value, expected_pass
):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = GoalSuccessAssertionRating(reasoning="Test", verdict=verdict)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = GoalSuccessRateEvaluator()

    result = evaluator.evaluate(evaluation_data_with_assertion)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == verdict
