from datetime import datetime

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
