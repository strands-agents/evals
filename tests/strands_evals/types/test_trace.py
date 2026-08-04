from datetime import datetime, timezone

from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    ContentType,
    Context,
    InferenceSpan,
    Role,
    Session,
    SpanInfo,
    SpanType,
    TextContent,
    ToolCall,
    ToolCallContent,
    ToolExecutionSpan,
    ToolLevelInput,
    ToolResult,
    Trace,
    TraceLevelInput,
    UserMessage,
)


def test_span_info_creation():
    """Test SpanInfo model creation"""
    now = datetime.now()
    span_info = SpanInfo(
        trace_id="trace123",
        span_id="span456",
        session_id="session789",
        parent_span_id="parent123",
        start_time=now,
        end_time=now,
    )

    assert span_info.trace_id == "trace123"
    assert span_info.span_id == "span456"
    assert span_info.session_id == "session789"
    assert span_info.parent_span_id == "parent123"
    assert span_info.start_time == now
    assert span_info.end_time == now


def test_text_content_creation():
    """Test TextContent model creation"""
    content = TextContent(text="Hello world")

    assert content.text == "Hello world"
    assert content.content_type == ContentType.TEXT


def test_tool_call_content_creation():
    """Test ToolCallContent model creation"""
    tool_call = ToolCallContent(name="calculator", arguments={"expression": "2+2"}, tool_call_id="call123")

    assert tool_call.name == "calculator"
    assert tool_call.arguments == {"expression": "2+2"}
    assert tool_call.content_type == ContentType.TOOL_USE


def test_user_message_creation():
    """Test UserMessage model creation"""
    message = UserMessage(content=[TextContent(text="Hello")])

    assert message.role == Role.USER
    assert len(message.content) == 1
    assert message.content[0].text == "Hello"


def test_assistant_message_creation():
    """Test AssistantMessage model creation"""
    message = AssistantMessage(content=[TextContent(text="Hi there")])

    assert message.role == Role.ASSISTANT
    assert len(message.content) == 1


def test_inference_span_creation():
    """Test InferenceSpan model creation"""
    now = datetime.now()
    span_info = SpanInfo(session_id="test", start_time=now, end_time=now)
    messages = [UserMessage(content=[TextContent(text="Hello")]), AssistantMessage(content=[TextContent(text="Hi")])]

    span = InferenceSpan(span_info=span_info, messages=messages)

    assert span.span_type == SpanType.INFERENCE
    assert len(span.messages) == 2


def test_tool_execution_span_creation():
    """Test ToolExecutionSpan model creation"""
    now = datetime.now()
    span_info = SpanInfo(session_id="test", start_time=now, end_time=now)
    tool_call = ToolCall(name="calculator", arguments={"expr": "2+2"})
    tool_result = ToolResult(content="4")

    span = ToolExecutionSpan(span_info=span_info, tool_call=tool_call, tool_result=tool_result)

    assert span.span_type == SpanType.TOOL_EXECUTION
    assert span.tool_call.name == "calculator"
    assert span.tool_result.content == "4"


def test_agent_invocation_span_creation():
    """Test AgentInvocationSpan model creation"""
    now = datetime.now()
    span_info = SpanInfo(session_id="test", start_time=now, end_time=now)

    span = AgentInvocationSpan(span_info=span_info, user_prompt="What is 2+2?", agent_response="4", available_tools=[])

    assert span.span_type == SpanType.AGENT_INVOCATION
    assert span.user_prompt == "What is 2+2?"
    assert span.agent_response == "4"


def test_trace_creation():
    """Test Trace model creation"""
    now = datetime.now()
    span_info = SpanInfo(session_id="test", start_time=now, end_time=now)
    span = AgentInvocationSpan(span_info=span_info, user_prompt="test", agent_response="response", available_tools=[])

    trace = Trace(spans=[span], trace_id="trace1", session_id="test")

    assert trace.trace_id == "trace1"
    assert trace.session_id == "test"
    assert len(trace.spans) == 1


def test_session_creation():
    """Test Session model creation"""
    now = datetime.now()
    span_info = SpanInfo(session_id="test", start_time=now, end_time=now)
    span = AgentInvocationSpan(span_info=span_info, user_prompt="test", agent_response="response", available_tools=[])
    trace = Trace(spans=[span], trace_id="trace1", session_id="test")
    session = Session(traces=[trace], session_id="test")

    assert session.session_id == "test"
    assert len(session.traces) == 1


def test_trace_level_input_creation():
    """Test TraceLevelInput model creation"""
    now = datetime.now()
    trace_input = TraceLevelInput(
        span_info=SpanInfo(session_id="test", start_time=now, end_time=now),
        agent_response=TextContent(text="4"),
        session_history=[UserMessage(content=[TextContent(text="Hi")])],
    )

    assert trace_input.agent_response.text == "4"
    assert len(trace_input.session_history) == 1


def test_tool_level_input_creation():
    """Test ToolLevelInput model creation"""
    now = datetime.now()
    span_info = SpanInfo(session_id="test", start_time=now, end_time=now)
    tool_exec = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}),
        tool_result=ToolResult(content="4"),
    )
    tool_input = ToolLevelInput(
        span_info=span_info, available_tools=[], tool_execution_details=tool_exec, session_history=[]
    )

    assert tool_input.tool_execution_details.tool_call.name == "calculator"
    assert tool_input.tool_execution_details.tool_call.arguments == {"expression": "2+2"}
    assert tool_input.tool_execution_details.tool_result.content == "4"


def test_context_creation():
    """Test Context model creation"""
    context = Context(
        user_prompt=TextContent(text="What is 2+2?"), agent_response=TextContent(text="4"), tool_execution_history=None
    )

    assert context.user_prompt.text == "What is 2+2?"
    assert context.agent_response.text == "4"
    assert context.tool_execution_history is None


def _info(span_id: str, parent_span_id: str | None = None) -> SpanInfo:
    """Helper to create SpanInfo with minimal boilerplate."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SpanInfo(
        session_id="s1",
        span_id=span_id,
        parent_span_id=parent_span_id,
        start_time=now,
        end_time=now,
    )


class TestTraceToolOwnership:
    """Tests for automatic tool ownership assignment in Trace.model_post_init."""

    def test_single_agent_assigns_ownership(self):
        """With one agent, all tools are owned by it."""
        agent = AgentInvocationSpan(
            span_info=_info("agent-1"), user_prompt="hi", agent_response="hello", available_tools=[]
        )
        tool = ToolExecutionSpan(
            span_info=_info("tool-1", parent_span_id="agent-1"),
            tool_call=ToolCall(name="calc", arguments={}),
            tool_result=ToolResult(content="42"),
        )

        Trace(spans=[agent, tool], trace_id="t1", session_id="s1")

        assert tool.agent_span_id == "agent-1"

    def test_multi_agent_assigns_to_nearest_ancestor(self):
        """Each tool is owned by the nearest agent in its parent chain."""
        root = AgentInvocationSpan(span_info=_info("root"), user_prompt="hi", agent_response="hey", available_tools=[])
        child = AgentInvocationSpan(
            span_info=_info("child", parent_span_id="root"),
            user_prompt="sub",
            agent_response="done",
            available_tools=[],
        )
        tool_of_root = ToolExecutionSpan(
            span_info=_info("t-root", parent_span_id="root"),
            tool_call=ToolCall(name="a", arguments={}),
            tool_result=ToolResult(content="1"),
        )
        tool_of_child = ToolExecutionSpan(
            span_info=_info("t-child", parent_span_id="child"),
            tool_call=ToolCall(name="b", arguments={}),
            tool_result=ToolResult(content="2"),
        )

        Trace(spans=[root, child, tool_of_root, tool_of_child], trace_id="t1", session_id="s1")

        assert tool_of_root.agent_span_id == "root"
        assert tool_of_child.agent_span_id == "child"

    def test_orphan_tool_falls_back_to_root_agent(self):
        """A tool whose parent chain doesn't reach any agent falls back to root."""
        agent = AgentInvocationSpan(
            span_info=_info("agent-1"), user_prompt="hi", agent_response="hello", available_tools=[]
        )
        tool = ToolExecutionSpan(
            span_info=_info("tool-1", parent_span_id="nonexistent"),
            tool_call=ToolCall(name="x", arguments={}),
            tool_result=ToolResult(content="y"),
        )

        Trace(spans=[agent, tool], trace_id="t1", session_id="s1")

        assert tool.agent_span_id == "agent-1"

    def test_no_agents_leaves_ownership_none(self):
        """With no agents in the trace, ownership is not set."""
        tool = ToolExecutionSpan(
            span_info=_info("tool-1"),
            tool_call=ToolCall(name="x", arguments={}),
            tool_result=ToolResult(content="y"),
        )

        Trace(spans=[tool], trace_id="t1", session_id="s1")

        assert tool.agent_span_id is None

    def test_skips_when_already_set(self):
        """If ownership is already populated, model_post_init is a no-op."""
        agent = AgentInvocationSpan(
            span_info=_info("agent-1"), user_prompt="hi", agent_response="hello", available_tools=[]
        )
        tool = ToolExecutionSpan(
            span_info=_info("tool-1", parent_span_id="agent-1"),
            tool_call=ToolCall(name="x", arguments={}),
            tool_result=ToolResult(content="y"),
            agent_span_id="custom-override",
        )

        Trace(spans=[agent, tool], trace_id="t1", session_id="s1")

        assert tool.agent_span_id == "custom-override"

    def test_idempotent_on_deserialization(self):
        """Deserializing a Trace with ownership already set doesn't re-walk."""
        agent = AgentInvocationSpan(
            span_info=_info("agent-1"), user_prompt="hi", agent_response="hello", available_tools=[]
        )
        tool = ToolExecutionSpan(
            span_info=_info("tool-1", parent_span_id="agent-1"),
            tool_call=ToolCall(name="x", arguments={}),
            tool_result=ToolResult(content="y"),
        )
        trace = Trace(spans=[agent, tool], trace_id="t1", session_id="s1")
        assert tool.agent_span_id == "agent-1"

        # Round-trip through JSON
        restored = Trace.model_validate_json(trace.model_dump_json())
        restored_tool = next(s for s in restored.spans if isinstance(s, ToolExecutionSpan))
        assert restored_tool.agent_span_id == "agent-1"


def test_tools_without_span_ids_each_owned_by_their_own_agent():
    """`span_id` is optional, so the subtree walk must not de-duplicate on it."""
    coordinator = AgentInvocationSpan(
        span_info=SpanInfo(
            session_id="s1",
            span_id="coordinator",
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        user_prompt="weather?",
        agent_response="sunny",
        available_tools=[],
    )
    specialist = AgentInvocationSpan(
        span_info=SpanInfo(
            session_id="s1",
            span_id="weather-agent",
            parent_span_id="coordinator",
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        user_prompt="weather",
        agent_response="sunny",
        available_tools=[],
    )
    forecast = ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="s1",
            parent_span_id="weather-agent",
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        tool_call=ToolCall(name="get_forecast", arguments={}),
        tool_result=ToolResult(content="sunny"),
    )
    alerts = ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="s1",
            parent_span_id="weather-agent",
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        tool_call=ToolCall(name="get_alerts", arguments={}),
        tool_result=ToolResult(content="none"),
    )

    Trace(spans=[coordinator, specialist, forecast, alerts], trace_id="t1", session_id="s1")

    assert forecast.agent_span_id == "weather-agent"
    assert alerts.agent_span_id == "weather-agent"
