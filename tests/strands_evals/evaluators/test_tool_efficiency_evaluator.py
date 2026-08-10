import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import ToolEfficiencyEvaluator
from strands_evals.evaluators.tool_efficiency_evaluator import (
    ToolCallCategory,
    ToolCallClassification,
    ToolEfficiencyRating,
)
from strands_evals.types import EvaluationData
from strands_evals.types.evaluation import EvaluationOutput
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
def span_info():
    now = datetime.now()
    return SpanInfo(session_id="test-session", start_time=now, end_time=now)


@pytest.fixture
def tool_configs():
    return [
        ToolConfig(name="search", description="Search for information"),
        ToolConfig(name="calculator", description="Evaluate mathematical expressions"),
    ]


@pytest.fixture
def simple_session(span_info, tool_configs):
    """A simple session with one tool call."""
    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="What is 2 + 2?",
        agent_response="The answer is 4.",
        available_tools=tool_configs,
    )
    tool_span = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}, tool_call_id="1"),
        tool_result=ToolResult(content="4", tool_call_id="1"),
    )
    trace = Trace(spans=[agent_span, tool_span], trace_id="trace1", session_id="test-session")
    return Session(traces=[trace], session_id="test-session")


@pytest.fixture
def redundant_session(span_info, tool_configs):
    """A session with redundant tool calls (same search called twice)."""
    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="Search for Python tutorials",
        agent_response="Here are some Python tutorials.",
        available_tools=tool_configs,
    )
    tool_span_1 = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="search", arguments={"query": "Python tutorials"}, tool_call_id="1"),
        tool_result=ToolResult(content="Tutorial 1, Tutorial 2", tool_call_id="1"),
    )
    tool_span_2 = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="search", arguments={"query": "Python tutorials"}, tool_call_id="2"),
        tool_result=ToolResult(content="Tutorial 1, Tutorial 2", tool_call_id="2"),
    )
    trace = Trace(spans=[agent_span, tool_span_1, tool_span_2], trace_id="trace1", session_id="test-session")
    return Session(traces=[trace], session_id="test-session")


@pytest.fixture
def errored_session(span_info, tool_configs):
    """A session with an errored tool call followed by a corrected retry."""
    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="Calculate 10 / 2",
        agent_response="The result is 5.",
        available_tools=tool_configs,
    )
    tool_span_error = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="calculator", arguments={"expression": "10 /"}, tool_call_id="1"),
        tool_result=ToolResult(content="", error="SyntaxError: incomplete expression", tool_call_id="1"),
    )
    tool_span_success = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="calculator", arguments={"expression": "10 / 2"}, tool_call_id="2"),
        tool_result=ToolResult(content="5", tool_call_id="2"),
    )
    trace = Trace(spans=[agent_span, tool_span_error, tool_span_success], trace_id="trace1", session_id="test-session")
    return Session(traces=[trace], session_id="test-session")


@pytest.fixture
def evaluation_data(simple_session):
    return EvaluationData(
        input="What is 2 + 2?",
        actual_output="The answer is 4.",
        actual_trajectory=simple_session,
        name="test",
    )


@pytest.fixture
def evaluation_data_redundant(redundant_session):
    return EvaluationData(
        input="Search for Python tutorials",
        actual_output="Here are some Python tutorials.",
        actual_trajectory=redundant_session,
        name="test-redundant",
    )


@pytest.fixture
def evaluation_data_errored(errored_session):
    return EvaluationData(
        input="Calculate 10 / 2",
        actual_output="The result is 5.",
        actual_trajectory=errored_session,
        name="test-errored",
    )


def test_init_with_defaults():
    evaluator = ToolEfficiencyEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt is not None
    assert evaluator.max_tool_result_length == 2000
    assert evaluator.pass_threshold == 0.5
    assert evaluator.evaluation_level == EvaluationLevel.SESSION_LEVEL


def test_init_with_custom_values():
    evaluator = ToolEfficiencyEvaluator(
        version="v0",
        model="us.anthropic.claude-sonnet-4-20250514",
        system_prompt="Custom prompt",
        max_tool_result_length=500,
        pass_threshold=0.8,
    )

    assert evaluator.model == "us.anthropic.claude-sonnet-4-20250514"
    assert evaluator.system_prompt == "Custom prompt"
    assert evaluator.max_tool_result_length == 500
    assert evaluator.pass_threshold == 0.8


def test_init_with_name():
    evaluator = ToolEfficiencyEvaluator(name="my-efficiency-eval")
    assert evaluator.get_name() == "my-efficiency-eval"


@patch("strands_evals.evaluators.tool_efficiency_evaluator.Agent")
def test_evaluate_all_necessary(mock_agent_class, evaluation_data):
    """Test evaluation where all tool calls are necessary."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolEfficiencyRating(
        classifications=[
            ToolCallClassification(
                tool_name="calculator",
                call_index=0,
                category=ToolCallCategory.NECESSARY,
                reasoning="Result used in final response",
            )
        ],
        reasoning="All tool calls contributed to the final response.",
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ToolEfficiencyEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0] == EvaluationOutput(
        score=1.0,
        test_pass=True,
        reason="All tool calls contributed to the final response.",
        label=ToolEfficiencyRating(
            classifications=[
                ToolCallClassification(
                    tool_name="calculator",
                    call_index=0,
                    category=ToolCallCategory.NECESSARY,
                    reasoning="Result used in final response",
                )
            ],
            reasoning="All tool calls contributed to the final response.",
        ).model_dump_json(),
    )


@patch("strands_evals.evaluators.tool_efficiency_evaluator.Agent")
def test_evaluate_with_redundant_calls(mock_agent_class, evaluation_data_redundant):
    """Test evaluation with redundant tool calls returns lower score."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolEfficiencyRating(
        classifications=[
            ToolCallClassification(
                tool_name="search",
                call_index=0,
                category=ToolCallCategory.NECESSARY,
                reasoning="First search provided the tutorials.",
            ),
            ToolCallClassification(
                tool_name="search",
                call_index=1,
                category=ToolCallCategory.REDUNDANT,
                reasoning="Same search with same parameters was already called.",
            ),
        ],
        reasoning="One of two search calls was redundant.",
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ToolEfficiencyEvaluator()

    result = evaluator.evaluate(evaluation_data_redundant)

    assert len(result) == 1
    assert result[0] == EvaluationOutput(
        score=0.5,
        test_pass=True,
        reason="One of two search calls was redundant.",
        label=mock_result.structured_output.model_dump_json(),
    )


@patch("strands_evals.evaluators.tool_efficiency_evaluator.Agent")
def test_evaluate_with_errored_calls(mock_agent_class, evaluation_data_errored):
    """Test evaluation with errored tool calls."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolEfficiencyRating(
        classifications=[
            ToolCallClassification(
                tool_name="calculator",
                call_index=0,
                category=ToolCallCategory.ERRORED,
                reasoning="Malformed expression caused a syntax error.",
            ),
            ToolCallClassification(
                tool_name="calculator",
                call_index=1,
                category=ToolCallCategory.NECESSARY,
                reasoning="Corrected call produced the result used in response.",
            ),
        ],
        reasoning="One errored call followed by a successful retry.",
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ToolEfficiencyEvaluator()

    result = evaluator.evaluate(evaluation_data_errored)

    assert len(result) == 1
    assert result[0] == EvaluationOutput(
        score=0.5,
        test_pass=True,
        reason="One errored call followed by a successful retry.",
        label=mock_result.structured_output.model_dump_json(),
    )


@patch("strands_evals.evaluators.tool_efficiency_evaluator.Agent")
def test_evaluate_low_efficiency(mock_agent_class, evaluation_data_redundant):
    """Test that score below pass_threshold results in test_pass=False."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolEfficiencyRating(
        classifications=[
            ToolCallClassification(
                tool_name="search",
                call_index=0,
                category=ToolCallCategory.UNNECESSARY,
                reasoning="Result never used.",
            ),
            ToolCallClassification(
                tool_name="search",
                call_index=1,
                category=ToolCallCategory.UNNECESSARY,
                reasoning="Result never used.",
            ),
            ToolCallClassification(
                tool_name="calculator",
                call_index=2,
                category=ToolCallCategory.NECESSARY,
                reasoning="Used in response.",
            ),
        ],
        reasoning="Most tool calls were unnecessary.",
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ToolEfficiencyEvaluator()

    result = evaluator.evaluate(evaluation_data_redundant)

    # Score is derived from classifications: 1 necessary / 3 total
    assert len(result) == 1
    assert result[0].score == pytest.approx(1.0 / 3.0)
    assert result[0].test_pass is False


@patch("strands_evals.evaluators.tool_efficiency_evaluator.Agent")
def test_evaluate_no_tool_calls(mock_agent_class, span_info, tool_configs):
    """Test that a session with no tool calls gets a perfect score."""
    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="Hello",
        agent_response="Hi there!",
        available_tools=tool_configs,
    )
    trace = Trace(spans=[agent_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")
    eval_data = EvaluationData(
        input="Hello",
        actual_output="Hi there!",
        actual_trajectory=session,
        name="no-tools",
    )

    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolEfficiencyRating(
        classifications=[],
        reasoning="No tool calls were made.",
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ToolEfficiencyEvaluator()

    result = evaluator.evaluate(eval_data)

    assert len(result) == 1
    assert result[0] == EvaluationOutput(
        score=1.0,
        test_pass=True,
        reason="No tool calls were made.",
        label=mock_result.structured_output.model_dump_json(),
    )


@patch("strands_evals.evaluators.tool_efficiency_evaluator.Agent")
def test_evaluate_custom_pass_threshold(mock_agent_class, evaluation_data_redundant):
    """Test that custom pass_threshold is respected."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolEfficiencyRating(
        classifications=[
            ToolCallClassification(
                tool_name="search",
                call_index=0,
                category=ToolCallCategory.NECESSARY,
                reasoning="First search provided the tutorials.",
            ),
            ToolCallClassification(
                tool_name="search",
                call_index=1,
                category=ToolCallCategory.REDUNDANT,
                reasoning="Duplicate call.",
            ),
        ],
        reasoning="One redundant call.",
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent

    # With threshold=0.8, a score of 0.5 should fail
    evaluator = ToolEfficiencyEvaluator(pass_threshold=0.8)
    result = evaluator.evaluate(evaluation_data_redundant)

    assert len(result) == 1
    assert result[0] == EvaluationOutput(
        score=0.5,
        test_pass=False,
        reason="One redundant call.",
        label=mock_result.structured_output.model_dump_json(),
    )


@patch("strands_evals.evaluators.tool_efficiency_evaluator.Agent")
def test_evaluate_score_derived_from_classifications(mock_agent_class, evaluation_data):
    """Test that score is derived from classifications, not LLM-provided counts."""
    mock_agent = Mock()
    mock_result = Mock()
    # Classifications have 2 necessary out of 3, score should be 2/3
    mock_result.structured_output = ToolEfficiencyRating(
        classifications=[
            ToolCallClassification(
                tool_name="search",
                call_index=0,
                category=ToolCallCategory.NECESSARY,
                reasoning="Used.",
            ),
            ToolCallClassification(
                tool_name="search",
                call_index=1,
                category=ToolCallCategory.NECESSARY,
                reasoning="Also used.",
            ),
            ToolCallClassification(
                tool_name="search",
                call_index=2,
                category=ToolCallCategory.REDUNDANT,
                reasoning="Duplicate.",
            ),
        ],
        reasoning="Mostly efficient.",
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ToolEfficiencyEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert result[0].score == pytest.approx(2.0 / 3.0)
    assert result[0].test_pass is True


@patch("strands_evals.evaluators.tool_efficiency_evaluator.Agent")
def test_evaluate_passes_correct_model(mock_agent_class, evaluation_data):
    """Test that the evaluator passes the configured model to the Agent."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolEfficiencyRating(
        classifications=[],
        reasoning="No tool calls.",
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent

    evaluator = ToolEfficiencyEvaluator(model="us.anthropic.claude-sonnet-4-20250514")
    evaluator.evaluate(evaluation_data)

    mock_agent_class.assert_called_once_with(
        model="us.anthropic.claude-sonnet-4-20250514",
        system_prompt=evaluator.system_prompt,
        callback_handler=None,
    )


@patch("strands_evals.evaluators.tool_efficiency_evaluator.Agent")
def test_evaluate_uses_structured_output(mock_agent_class, evaluation_data):
    """Test that the evaluator requests ToolEfficiencyRating as structured output."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolEfficiencyRating(
        classifications=[],
        reasoning="Test.",
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent

    evaluator = ToolEfficiencyEvaluator()
    evaluator.evaluate(evaluation_data)

    mock_agent.assert_called_once()
    call_kwargs = mock_agent.call_args
    assert call_kwargs.kwargs.get("structured_output_model") == ToolEfficiencyRating


def test_format_prompt_includes_tools(span_info, tool_configs):
    """Test that _format_prompt includes available tools."""
    evaluator = ToolEfficiencyEvaluator()

    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="test",
        agent_response="response",
        available_tools=tool_configs,
    )
    tool_span = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="search", arguments={"query": "test"}, tool_call_id="1"),
        tool_result=ToolResult(content="result", tool_call_id="1"),
    )
    trace = Trace(spans=[agent_span, tool_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")
    eval_data = EvaluationData(input="test", actual_output="response", actual_trajectory=session)

    session_input = evaluator._parse_trajectory(eval_data)
    prompt = evaluator._format_prompt(session_input)

    assert "# Available tools" in prompt
    assert "search" in prompt
    assert "calculator" in prompt


def test_format_prompt_truncates_long_results(span_info, tool_configs):
    """Test that long tool results are truncated."""
    evaluator = ToolEfficiencyEvaluator(max_tool_result_length=50)

    long_result = "x" * 200

    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="test",
        agent_response="response",
        available_tools=tool_configs,
    )
    tool_span = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="search", arguments={"query": "test"}, tool_call_id="1"),
        tool_result=ToolResult(content=long_result, tool_call_id="1"),
    )
    trace = Trace(spans=[agent_span, tool_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")
    eval_data = EvaluationData(input="test", actual_output="response", actual_trajectory=session)

    session_input = evaluator._parse_trajectory(eval_data)
    prompt = evaluator._format_prompt(session_input)

    assert "... [truncated]" in prompt
    assert long_result not in prompt


def test_format_prompt_shows_errors(span_info, tool_configs):
    """Test that tool errors are included in the prompt."""
    evaluator = ToolEfficiencyEvaluator()

    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="test",
        agent_response="response",
        available_tools=tool_configs,
    )
    tool_span = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="calculator", arguments={"expression": "bad"}, tool_call_id="1"),
        tool_result=ToolResult(content="", error="SyntaxError", tool_call_id="1"),
    )
    trace = Trace(spans=[agent_span, tool_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")
    eval_data = EvaluationData(input="test", actual_output="response", actual_trajectory=session)

    session_input = evaluator._parse_trajectory(eval_data)
    prompt = evaluator._format_prompt(session_input)

    assert "Tool Error: SyntaxError" in prompt


def test_invalid_trajectory_raises_error():
    """Test that a non-Session trajectory raises TypeError."""
    evaluator = ToolEfficiencyEvaluator()
    eval_data = EvaluationData(
        input="test",
        actual_output="response",
        actual_trajectory=["not", "a", "session"],
    )

    with pytest.raises(TypeError, match="Session"):
        evaluator.evaluate(eval_data)


def test_no_trajectory_raises_error():
    """Test that missing trajectory raises appropriate error."""
    evaluator = ToolEfficiencyEvaluator()
    eval_data = EvaluationData(input="test", actual_output="response")

    with pytest.raises((TypeError, ValueError)):
        evaluator.evaluate(eval_data)


def test_to_dict():
    """Test serialization with to_dict."""
    evaluator = ToolEfficiencyEvaluator(model="custom-model", max_tool_result_length=500, pass_threshold=0.7)
    result = evaluator.to_dict()

    assert result["evaluator_type"] == "ToolEfficiencyEvaluator"
    assert result["model"] == "custom-model"
    assert result["max_tool_result_length"] == 500
    assert result["pass_threshold"] == 0.7


def test_to_dict_defaults():
    """Test serialization with default values omits defaults."""
    evaluator = ToolEfficiencyEvaluator()
    result = evaluator.to_dict()

    assert result["evaluator_type"] == "ToolEfficiencyEvaluator"
    # Default model is None, should serialize as model_id with default value
    assert "model_id" in result
    # Default pass_threshold=0.5 should not appear (it's a default)
    assert "pass_threshold" not in result


def test_tool_call_category_values():
    """Test that ToolCallCategory has expected string values."""
    assert ToolCallCategory.NECESSARY == "necessary"
    assert ToolCallCategory.REDUNDANT == "redundant"
    assert ToolCallCategory.ERRORED == "errored"
    assert ToolCallCategory.UNNECESSARY == "unnecessary"


def test_tool_efficiency_rating_serialization():
    """Test that ToolEfficiencyRating serializes to JSON correctly."""
    rating = ToolEfficiencyRating(
        classifications=[
            ToolCallClassification(
                tool_name="search",
                call_index=0,
                category=ToolCallCategory.NECESSARY,
                reasoning="Used in response.",
            ),
            ToolCallClassification(
                tool_name="search",
                call_index=1,
                category=ToolCallCategory.REDUNDANT,
                reasoning="Duplicate call.",
            ),
        ],
        reasoning="One redundant call detected.",
    )

    json_str = rating.model_dump_json()
    parsed = json.loads(json_str)

    assert parsed == {
        "classifications": [
            {
                "tool_name": "search",
                "call_index": 0,
                "category": "necessary",
                "reasoning": "Used in response.",
            },
            {
                "tool_name": "search",
                "call_index": 1,
                "category": "redundant",
                "reasoning": "Duplicate call.",
            },
        ],
        "reasoning": "One redundant call detected.",
    }


@patch("strands_evals.evaluators.tool_efficiency_evaluator.Agent")
def test_evaluate_perfect_efficiency(mock_agent_class, evaluation_data):
    """Test a scenario where all calls are necessary results in score 1.0."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolEfficiencyRating(
        classifications=[
            ToolCallClassification(
                tool_name="calculator",
                call_index=0,
                category=ToolCallCategory.NECESSARY,
                reasoning="Used.",
            ),
        ],
        reasoning="Perfectly efficient.",
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ToolEfficiencyEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert result[0] == EvaluationOutput(
        score=1.0,
        test_pass=True,
        reason="Perfectly efficient.",
        label=mock_result.structured_output.model_dump_json(),
    )


@patch("strands_evals.evaluators.tool_efficiency_evaluator.Agent")
def test_evaluate_zero_efficiency(mock_agent_class, evaluation_data):
    """Test a scenario where no calls are necessary results in score 0.0."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = ToolEfficiencyRating(
        classifications=[
            ToolCallClassification(
                tool_name="search",
                call_index=0,
                category=ToolCallCategory.UNNECESSARY,
                reasoning="Not used.",
            ),
            ToolCallClassification(
                tool_name="search",
                call_index=1,
                category=ToolCallCategory.UNNECESSARY,
                reasoning="Not used.",
            ),
        ],
        reasoning="No calls were necessary.",
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = ToolEfficiencyEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert result[0] == EvaluationOutput(
        score=0.0,
        test_pass=False,
        reason="No calls were necessary.",
        label=mock_result.structured_output.model_dump_json(),
    )
