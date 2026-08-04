from datetime import datetime, timezone

import pytest

from strands_evals.extractors import TraceExtractor
from strands_evals.types.trace import (
    AgentInvocationSpan,
    EvaluationLevel,
    Session,
    SessionLevelInput,
    SpanInfo,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolLevelInput,
    ToolResult,
    Trace,
    TraceLevelInput,
)


def _span_info(
    span_id: str | None = None,
    parent_span_id: str | None = None,
    session_id: str = "test",
) -> SpanInfo:
    """Helper to create a SpanInfo with minimal boilerplate."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SpanInfo(
        session_id=session_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        start_time=now,
        end_time=now,
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


@pytest.fixture
def multi_agent_session():
    """Coordinator delegates to a specialist with its own tools."""
    coordinator = AgentInvocationSpan(
        span_info=_span_info(span_id="coordinator", parent_span_id=None),
        user_prompt="sqrt(16) * 2 and weather in Paris",
        agent_response="8, sunny.",
        available_tools=[ToolConfig(name="ask_math"), ToolConfig(name="ask_research")],
    )
    math_agent = AgentInvocationSpan(
        span_info=_span_info(span_id="math-agent", parent_span_id="coordinator"),
        user_prompt="sqrt(16) * 2",
        agent_response="8",
        available_tools=[ToolConfig(name="square_root"), ToolConfig(name="multiply")],
    )
    # Tool directly under coordinator (delegation call)
    tool_delegate = ToolExecutionSpan(
        span_info=_span_info(span_id="tool-delegate", parent_span_id="coordinator"),
        tool_call=ToolCall(name="ask_math", arguments={"query": "sqrt(16)*2"}),
        tool_result=ToolResult(content="8"),
        agent_span_id="coordinator",
    )
    tool_sqrt = ToolExecutionSpan(
        span_info=_span_info(span_id="tool-sqrt", parent_span_id="math-agent"),
        tool_call=ToolCall(name="square_root", arguments={"n": 16}),
        tool_result=ToolResult(content="4.0"),
        agent_span_id="math-agent",
    )
    tool_mult = ToolExecutionSpan(
        span_info=_span_info(span_id="tool-mult", parent_span_id="math-agent"),
        tool_call=ToolCall(name="multiply", arguments={"a": 4, "b": 2}),
        tool_result=ToolResult(content="8"),
        agent_span_id="math-agent",
    )
    trace = Trace(
        spans=[coordinator, math_agent, tool_delegate, tool_sqrt, tool_mult], trace_id="t1", session_id="test"
    )
    return Session(traces=[trace], session_id="test")


def test_trace_extractor_initialization():
    extractor = TraceExtractor(EvaluationLevel.TRACE_LEVEL)
    assert extractor.evaluation_level == EvaluationLevel.TRACE_LEVEL


def test_extract_trace_level(session_with_conversation):
    extractor = TraceExtractor(EvaluationLevel.TRACE_LEVEL)
    result = extractor.extract(session_with_conversation)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(item, TraceLevelInput) for item in result)
    assert result[0].agent_response.text == "4"
    assert result[1].agent_response.text == "6"


def test_extract_trace_level_with_session_history(session_with_conversation):
    """Test that session history accumulates correctly across turns."""
    extractor = TraceExtractor(EvaluationLevel.TRACE_LEVEL)
    result = extractor.extract(session_with_conversation)

    assert len(result) == 2
    # Second turn should have history of first turn
    assert result[1].agent_response.text == "6"
    assert len(result[1].session_history) == 3
    assert result[1].session_history[0].role.value == "user"
    assert result[1].session_history[0].content[0].text == "What is 2+2?"
    assert result[1].session_history[1].role.value == "assistant"
    assert result[1].session_history[1].content[0].text == "4"
    assert result[1].session_history[2].role.value == "user"
    assert result[1].session_history[2].content[0].text == "What is 3+3?"


def test_extract_tool_level(session_with_tools):
    extractor = TraceExtractor(EvaluationLevel.TOOL_LEVEL)
    result = extractor.extract(session_with_tools)

    assert isinstance(result, list)
    assert len(result) == 1
    assert all(isinstance(item, ToolLevelInput) for item in result)
    assert result[0].tool_execution_details.tool_call.name == "calculator"
    assert result[0].tool_execution_details.tool_call.arguments == {"expression": "2+2"}
    assert result[0].tool_execution_details.tool_result.content == "4"


def test_extract_session_level(session_with_conversation):
    extractor = TraceExtractor(EvaluationLevel.SESSION_LEVEL)
    result = extractor.extract(session_with_conversation)

    assert isinstance(result, SessionLevelInput)
    assert len(result.session_history) == 2
    assert result.session_history[0].user_prompt.text == "What is 2+2?"
    assert result.session_history[0].agent_response.text == "4"
    assert result.session_history[1].user_prompt.text == "What is 3+3?"
    assert result.session_history[1].agent_response.text == "6"


def test_extract_raises_on_invalid_session_type():
    extractor = TraceExtractor(EvaluationLevel.TRACE_LEVEL)

    with pytest.raises(TypeError, match="Expected Session object"):
        extractor.extract(["not", "a", "session"])


def test_extract_raises_on_unsupported_level():
    with pytest.raises(ValueError, match="Unsupported evaluation level"):
        extractor = TraceExtractor("INVALID_LEVEL")
        extractor.extract(Session(traces=[], session_id="test"))


def test_composability_multiple_extractors(session_with_conversation):
    """Test that multiple extractors can be composed for different purposes."""
    trace_extractor = TraceExtractor(EvaluationLevel.TRACE_LEVEL)
    session_extractor = TraceExtractor(EvaluationLevel.SESSION_LEVEL)

    trace_result = trace_extractor.extract(session_with_conversation)
    session_result = session_extractor.extract(session_with_conversation)

    assert len(trace_result) == 2
    assert len(session_result.session_history) == 2


def test_extract_empty_session_trace_level():
    extractor = TraceExtractor(EvaluationLevel.TRACE_LEVEL)
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


def test_extract_tool_level_incremental_session_history():
    """Each tool span sees exactly its causally-completed predecessors in session_history."""
    from datetime import timedelta, timezone

    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    agent_span = AgentInvocationSpan(
        span_info=SpanInfo(session_id="test", span_id="a0", start_time=base, end_time=base + timedelta(seconds=10)),
        user_prompt="What is the square root of 1764, multiplied by 3, then add 1?",
        agent_response="127",
        available_tools=[
            ToolConfig(name="square_root"),
            ToolConfig(name="multiply_numbers"),
            ToolConfig(name="add_numbers"),
        ],
    )
    tool_span_1 = ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="test",
            span_id="t1",
            parent_span_id="a0",
            start_time=base,
            end_time=base + timedelta(seconds=1),
        ),
        tool_call=ToolCall(name="square_root", arguments={"n": 1764}),
        tool_result=ToolResult(content="42.0"),
    )
    # Finishes at t=2.5: AFTER tool_3 starts (t=2) but BEFORE tool_3 ends (t=3)
    # This discriminates start_time vs end_time comparison targets
    tool_span_2 = ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="test",
            span_id="t2",
            parent_span_id="a0",
            start_time=base + timedelta(seconds=1),
            end_time=base + timedelta(milliseconds=2500),
        ),
        tool_call=ToolCall(name="multiply_numbers", arguments={"a": 42, "b": 3}),
        tool_result=ToolResult(content="126"),
    )
    tool_span_3 = ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="test",
            span_id="t3",
            parent_span_id="a0",
            start_time=base + timedelta(seconds=2),
            end_time=base + timedelta(seconds=3),
        ),
        tool_call=ToolCall(name="add_numbers", arguments={"a": 126, "b": 1}),
        tool_result=ToolResult(content="127"),
    )

    trace = Trace(spans=[agent_span, tool_span_1, tool_span_2, tool_span_3], trace_id="trace1", session_id="test")
    session = Session(traces=[trace], session_id="test")

    extractor = TraceExtractor(EvaluationLevel.TOOL_LEVEL)
    result = extractor.extract(session)

    assert len(result) == 3, f"expected 3 tool-level inputs, got {len(result)}"

    # First tool: sees only the user prompt, no prior executions
    assert len(result[0].session_history) == 1, (
        f"first tool should only see user prompt, got {len(result[0].session_history)} entries"
    )
    assert result[0].tool_execution_details.tool_call.name == "square_root"

    # Second tool: sees user prompt + first tool's execution (square_root finished at t=1 <= t=1)
    assert len(result[1].session_history) == 2, (
        f"second tool should see user prompt + 1 prior execution, got {len(result[1].session_history)} entries"
    )
    assert isinstance(result[1].session_history[1], list)
    assert result[1].session_history[1][0].tool_call.name == "square_root"
    assert result[1].session_history[1][0].tool_result.content == "42.0"

    # Third tool: only square_root is a valid prior
    assert len(result[2].session_history) == 2, (
        f"expected 2 entries (user + prior tools), got {len(result[2].session_history)}"
    )
    prior_names = [entry.tool_call.name for entry in result[2].session_history[1]]
    assert prior_names == ["square_root"], (
        f"expected only [square_root] (multiply_numbers still running at tool_3.start_time), got {prior_names}"
    )


def test_extract_tool_level_mixed_tz_timestamps():
    """The causality filter handles mixed naive/aware timestamps without raising TypeError."""
    from datetime import timedelta, timezone

    base_naive = datetime(2024, 1, 1, 12, 0, 0)  # no tzinfo
    base_aware = datetime(2024, 1, 1, 12, 0, 2, tzinfo=timezone.utc)  # aware, starts at +2s

    agent_span = AgentInvocationSpan(
        span_info=SpanInfo(
            session_id="test", span_id="a0", start_time=base_naive, end_time=base_aware + timedelta(seconds=10)
        ),
        user_prompt="Mixed tz test",
        agent_response="Done",
        available_tools=[
            ToolConfig(name="tool_x"),
            ToolConfig(name="tool_long"),
            ToolConfig(name="tool_y"),
        ],
    )
    # tool_x: ends at 12:00:01 (before tool_y starts)
    tool_x = ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="test",
            span_id="tx",
            parent_span_id="a0",
            start_time=base_naive,
            end_time=base_naive + timedelta(seconds=1),
        ),
        tool_call=ToolCall(name="tool_x", arguments={}),
        tool_result=ToolResult(content="x_result"),
    )
    # tool_long: ends at 12:00:05 (after tool_y starts at 12:00:02)
    tool_long = ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="test",
            span_id="tl",
            parent_span_id="a0",
            start_time=base_naive,
            end_time=base_naive + timedelta(seconds=5),
        ),
        tool_call=ToolCall(name="tool_long", arguments={}),
        tool_result=ToolResult(content="long_result"),
    )
    # tool_y: starts at 12:00:02 UTC
    tool_y = ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="test",
            span_id="ty",
            parent_span_id="a0",
            start_time=base_aware,
            end_time=base_aware + timedelta(seconds=1),
        ),
        tool_call=ToolCall(name="tool_y", arguments={}),
        tool_result=ToolResult(content="y_result"),
    )

    trace = Trace(spans=[agent_span, tool_x, tool_long, tool_y], trace_id="trace1", session_id="test")
    session = Session(traces=[trace], session_id="test")

    extractor = TraceExtractor(EvaluationLevel.TOOL_LEVEL)
    # Should not raise TypeError from mixed tz comparison
    result = extractor.extract(session)

    assert len(result) == 3
    # tool_y should see ONLY tool_x as a prior
    assert result[2].tool_execution_details.tool_call.name == "tool_y"
    assert len(result[2].session_history) == 2
    prior_names = [e.tool_call.name for e in result[2].session_history[1]]
    assert prior_names == ["tool_x"], (
        f"expected only ['tool_x'] as prior for tool_y (tool_long still running), got {prior_names}"
    )


def test_tool_ownership_resolves_to_nearest_agent(multi_agent_session):
    """Tools resolve to nearest ancestor agent; coordinator's tools are not leaked."""
    extractor = TraceExtractor(EvaluationLevel.TOOL_LEVEL)
    result = extractor.extract(multi_agent_session)

    assert len(result) == 3
    delegate = next(r for r in result if r.tool_execution_details.tool_call.name == "ask_math")
    sqrt_r = next(r for r in result if r.tool_execution_details.tool_call.name == "square_root")
    mult_r = next(r for r in result if r.tool_execution_details.tool_call.name == "multiply")

    # Coordinator-owned tool gets coordinator's tool list
    assert {t.name for t in delegate.available_tools} == {"ask_math", "ask_research"}
    # Specialist-owned tools get specialist's tool list
    assert {t.name for t in sqrt_r.available_tools} == {"square_root", "multiply"}
    assert {t.name for t in mult_r.available_tools} == {"square_root", "multiply"}


def test_trace_level_scopes_tools_per_agent(multi_agent_session):
    """Each agent's turn only includes tool executions owned by that agent."""
    extractor = TraceExtractor(EvaluationLevel.TRACE_LEVEL)
    result = extractor.extract(multi_agent_session)

    assert len(result) == 2
    coord_turn = next(r for r in result if r.span_info.span_id == "coordinator")
    math_turn = next(r for r in result if r.span_info.span_id == "math-agent")

    # coordinator's turn has only its own delegation tool
    coord_tools = [h for h in coord_turn.session_history if isinstance(h, list)]
    assert len(coord_tools) == 1
    assert coord_tools[0][0].tool_call.name == "ask_math"

    # math_agent's turn includes its own tools (plus coordinator's from history)
    math_tools = [h for h in math_turn.session_history if isinstance(h, list)]
    assert len(math_tools) == 2
    assert math_tools[0][0].tool_call.name == "ask_math"  # accumulated from coordinator's turn
    assert {te.tool_call.name for te in math_tools[1]} == {"square_root", "multiply"}


def test_span_id_none_does_not_collide():
    """Two tool spans with span_id=None under different agents resolve independently."""
    coordinator = AgentInvocationSpan(
        span_info=_span_info(span_id="coordinator", parent_span_id=None),
        user_prompt="Do two things",
        agent_response="Done.",
        available_tools=[ToolConfig(name="delegate")],
    )
    spec_a = AgentInvocationSpan(
        span_info=_span_info(span_id="spec-a", parent_span_id="coordinator"),
        user_prompt="A",
        agent_response="A done",
        available_tools=[ToolConfig(name="tool_a")],
    )
    spec_b = AgentInvocationSpan(
        span_info=_span_info(span_id="spec-b", parent_span_id="coordinator"),
        user_prompt="B",
        agent_response="B done",
        available_tools=[ToolConfig(name="tool_b")],
    )
    tool_a = ToolExecutionSpan(
        span_info=_span_info(span_id=None, parent_span_id="spec-a"),
        tool_call=ToolCall(name="tool_a", arguments={}),
        tool_result=ToolResult(content="a"),
        agent_span_id="spec-a",
    )
    tool_b = ToolExecutionSpan(
        span_info=_span_info(span_id=None, parent_span_id="spec-b"),
        tool_call=ToolCall(name="tool_b", arguments={}),
        tool_result=ToolResult(content="b"),
        agent_span_id="spec-b",
    )

    trace = Trace(spans=[coordinator, spec_a, spec_b, tool_a, tool_b], trace_id="t1", session_id="test")
    session = Session(traces=[trace], session_id="test")

    extractor = TraceExtractor(EvaluationLevel.TOOL_LEVEL)
    result = extractor.extract(session)

    result_a = next(r for r in result if r.tool_execution_details.tool_call.name == "tool_a")
    result_b = next(r for r in result if r.tool_execution_details.tool_call.name == "tool_b")
    assert [t.name for t in result_a.available_tools] == ["tool_a"]
    assert [t.name for t in result_b.available_tools] == ["tool_b"]
