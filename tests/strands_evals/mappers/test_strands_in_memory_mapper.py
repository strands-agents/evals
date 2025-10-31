import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags

from strands_evals.mappers import GenAIConventionVersion, StrandsInMemorySessionMapper
from strands_evals.types.trace import AgentInvocationSpan, InferenceSpan, ToolExecutionSpan


@pytest.fixture
def provider():
    return TracerProvider()


def make_span(provider, trace_id, span_id, parent_id, operation, attributes, events_fn):
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(operation, kind=SpanKind.CLIENT) as s:
        for k, v in attributes.items():
            s.set_attribute(k, v)
        events_fn(s)

    return ReadableSpan(
        name=operation,
        context=SpanContext(trace_id, span_id, False, TraceFlags(0x01)),
        parent=SpanContext(trace_id, parent_id, False, TraceFlags(0x01)) if parent_id else None,
        resource=provider.resource,
        attributes=attributes,
        events=tuple(s._events),
        start_time=1700000000000000000,
        end_time=1700000001000000000,
    )


def test_inference_span(provider):
    span = make_span(
        provider,
        0xAAA,
        0xBBB,
        0xCCC,
        "chat",
        {"gen_ai.operation.name": "chat"},
        lambda s: (
            s.add_event("gen_ai.user.message", {"content": '[{"text": "hello"}]'}),
            s.add_event("gen_ai.choice", {"message": '[{"text": "hi"}]'}),
        ),
    )

    session = StrandsInMemorySessionMapper().map_to_session([span], "sid")

    inference = session.traces[0].spans[0]
    assert isinstance(inference, InferenceSpan)
    assert inference.messages[0].content[0].text == "hello"
    assert inference.messages[1].content[0].text == "hi"


def test_agent_span(provider):
    span = make_span(
        provider,
        0xAAA,
        0xBBB,
        None,
        "invoke_agent",
        {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.tools": '["calc"]'},
        lambda s: (
            s.add_event("gen_ai.user.message", {"content": '[{"text": "2+2"}]'}),
            s.add_event("gen_ai.choice", {"message": "4"}),
        ),
    )

    session = StrandsInMemorySessionMapper().map_to_session([span], "sid")

    agent = session.traces[0].spans[0]
    assert isinstance(agent, AgentInvocationSpan)
    assert agent.user_prompt == "2+2"
    assert agent.agent_response == "4"
    assert agent.available_tools[0].name == "calc"


def test_tool_span(provider):
    span = make_span(
        provider,
        0xAAA,
        0xBBB,
        0xCCC,
        "execute_tool",
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "calc",
            "gen_ai.tool.call.id": "t1",
            "tool.status": "success",
        },
        lambda s: (
            s.add_event("gen_ai.tool.message", {"content": '{"expr": "2+2"}'}),
            s.add_event("gen_ai.choice", {"message": '[{"text": "4"}]'}),
        ),
    )

    session = StrandsInMemorySessionMapper().map_to_session([span], "sid")

    tool = session.traces[0].spans[0]
    assert isinstance(tool, ToolExecutionSpan)
    assert tool.tool_call.name == "calc"
    assert tool.tool_result.content == "4"


def test_tool_use_in_message(provider):
    span = make_span(
        provider,
        0xAAA,
        0xBBB,
        0xCCC,
        "chat",
        {"gen_ai.operation.name": "chat"},
        lambda s: (
            s.add_event("gen_ai.user.message", {"content": '[{"text": "calc"}]'}),
            s.add_event(
                "gen_ai.choice", {"message": '[{"toolUse": {"toolUseId": "t1", "name": "calc", "input": {"x": 1}}}]'}
            ),
        ),
    )

    session = StrandsInMemorySessionMapper().map_to_session([span], "sid")

    msg = session.traces[0].spans[0].messages[1]
    assert msg.content[0].name == "calc"
    assert msg.content[0].tool_call_id == "t1"


def test_multiple_traces(provider):
    s1 = make_span(
        provider,
        0xAAA,
        0xB1,
        0xC1,
        "chat",
        {"gen_ai.operation.name": "chat"},
        lambda s: s.add_event("gen_ai.choice", {"message": '[{"text": "a"}]'}),
    )
    s2 = make_span(
        provider,
        0xBBB,
        0xB2,
        0xC2,
        "chat",
        {"gen_ai.operation.name": "chat"},
        lambda s: s.add_event("gen_ai.choice", {"message": '[{"text": "b"}]'}),
    )

    session = StrandsInMemorySessionMapper().map_to_session([s1, s2], "sid")

    assert len(session.traces) == 2


# Tests for convention version auto-detection


def test_auto_detect_legacy_convention(provider):
    """Test that legacy convention (gen_ai.system) is auto-detected."""
    span = make_span(
        provider,
        0xAAA,
        0xBBB,
        0xCCC,
        "chat",
        {"gen_ai.operation.name": "chat", "gen_ai.system": "strands-agents"},
        lambda s: (
            s.add_event("gen_ai.user.message", {"content": '[{"text": "hello"}]'}),
            s.add_event("gen_ai.choice", {"message": '[{"text": "hi"}]'}),
        ),
    )

    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session([span], "sid")

    # Verify auto-detection worked
    assert mapper._convention_version == GenAIConventionVersion.LEGACY

    # Verify message extraction worked
    inference = session.traces[0].spans[0]
    assert isinstance(inference, InferenceSpan)
    assert inference.messages[0].content[0].text == "hello"
    assert inference.messages[1].content[0].text == "hi"


def test_auto_detect_latest_convention(provider):
    """Test that latest convention (gen_ai.provider.name) is auto-detected."""
    span = make_span(
        provider,
        0xAAA,
        0xBBB,
        0xCCC,
        "chat",
        {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "strands-agents"},
        lambda s: s.add_event(
            "gen_ai.client.inference.operation.details",
            {
                "gen_ai.input.messages": """[
                    {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
                    ]""",
                "gen_ai.output.messages": """[
                    {"role": "assistant", "parts": [{"type": "text", "content": "hi"}], "finish_reason": "stop"}
                ]""",
            },
        ),
    )

    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session([span], "sid")

    # Verify auto-detection worked
    assert mapper._convention_version == GenAIConventionVersion.LATEST_EXPERIMENTAL

    # Verify message extraction worked
    inference = session.traces[0].spans[0]
    assert isinstance(inference, InferenceSpan)
    assert inference.messages[0].content[0].text == "hello"
    assert inference.messages[1].content[0].text == "hi"


def test_latest_convention_with_tool_call(provider):
    """Test latest convention format with tool calls."""
    span = make_span(
        provider,
        0xAAA,
        0xBBB,
        0xCCC,
        "chat",
        {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "strands-agents"},
        lambda s: s.add_event(
            "gen_ai.client.inference.operation.details",
            {
                "gen_ai.input.messages": """
                [
                    {"role": "user", "parts": [{"type": "text", "content": "calculate 2+2"}]}
                ]""",
                "gen_ai.output.messages": """
                [{"role": "assistant", "parts": [
                    {"type": "text", "content": "Let me calculate"},
                    {
                        "type": "tool_call", 
                        "name": "calc",
                        "id": "t1", 
                        "arguments": {"expr": "2+2"}}
                    ], "finish_reason": "tool_use"}
                ]
                """,
            },
        ),
    )

    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session([span], "sid")

    inference = session.traces[0].spans[0]
    assert isinstance(inference, InferenceSpan)

    # Check user message
    assert inference.messages[0].content[0].text == "calculate 2+2"

    # Check assistant message with tool call
    assert inference.messages[1].content[0].text == "Let me calculate"
    assert inference.messages[1].content[1].name == "calc"
    assert inference.messages[1].content[1].tool_call_id == "t1"
    assert inference.messages[1].content[1].arguments == {"expr": "2+2"}


def test_latest_convention_with_tool_result(provider):
    """Test latest convention format with tool results."""
    span = make_span(
        provider,
        0xAAA,
        0xBBB,
        0xCCC,
        "chat",
        {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "strands-agents"},
        lambda s: s.add_event(
            "gen_ai.client.inference.operation.details",
            {
                "gen_ai.input.messages": """
                    [
                        {"role": "user", "parts": [
                            {"type": "tool_call_response", "id": "t1", "response": [{"text": "4"}]}]}
                    ]
                """,
                "gen_ai.output.messages": """
                    [{"role": "assistant", "parts": [
                        {"type": "text", "content": "The answer is 4"}
                    ], "finish_reason": "stop"}]
                """,
            },
        ),
    )

    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session([span], "sid")

    inference = session.traces[0].spans[0]

    # Check tool result in user message
    assert inference.messages[0].content[0].content == "4"
    assert inference.messages[0].content[0].tool_call_id == "t1"

    # Check assistant response
    assert inference.messages[1].content[0].text == "The answer is 4"


def test_latest_convention_tool_execution_span(provider):
    """Test tool execution span with latest convention."""
    span = make_span(
        provider,
        0xAAA,
        0xBBB,
        0xCCC,
        "execute_tool",
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.provider.name": "strands-agents",
            "gen_ai.tool.name": "calc",
            "gen_ai.tool.call.id": "t1",
            "gen_ai.tool.status": "success",
        },
        lambda s: (
            s.add_event(
                "gen_ai.client.inference.operation.details",
                {
                    "gen_ai.input.messages": """[
                        {
                            "role": "tool",
                            "parts": [
                                {"type": "tool_call", 
                                "name": "calc", 
                                "id": "t1", 
                                "arguments": {"expr": "2+2"}}]
                        }
                    ]"""
                },
            ),
            s.add_event(
                "gen_ai.client.inference.operation.details",
                {
                    "gen_ai.output.messages": """[
                        {"role": "tool", "parts": [
                            {"type": "tool_call_response", "id": "t1", "response": [{"text": "4"}]}],
                        "finish_reason": "stop"}
                    ]"""
                },
            ),
        ),
    )

    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session([span], "sid")

    tool = session.traces[0].spans[0]
    assert isinstance(tool, ToolExecutionSpan)
    assert tool.tool_call.name == "calc"
    assert tool.tool_call.arguments == {"expr": "2+2"}
    assert tool.tool_result.content == "4"
    assert tool.tool_result.tool_call_id == "t1"


def test_latest_convention_agent_invocation_span(provider):
    """Test agent invocation span with latest convention."""
    span = make_span(
        provider,
        0xAAA,
        0xBBB,
        None,
        "invoke_agent",
        {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name": "strands-agents",
            "gen_ai.agent.tools": '["calc", "search"]',
        },
        lambda s: (
            s.add_event(
                "gen_ai.client.inference.operation.details",
                {
                    "gen_ai.input.messages": """[
                    {"role": "user", "parts": [{"type": "text", "content": "What is 2+2?"}]}
                 ]"""
                },
            ),
            s.add_event(
                "gen_ai.client.inference.operation.details",
                {
                    "gen_ai.output.messages": """[
                        {
                            "role": "assistant", 
                            "parts": [{"type": "text", "content": "The answer is 4"}],
                            "finish_reason": "stop"
                        }
                    ]"""
                },
            ),
        ),
    )

    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session([span], "sid")

    agent = session.traces[0].spans[0]
    assert isinstance(agent, AgentInvocationSpan)
    assert agent.user_prompt == "What is 2+2?"
    assert agent.agent_response == "The answer is 4"
    assert len(agent.available_tools) == 2
    assert agent.available_tools[0].name == "calc"
    assert agent.available_tools[1].name == "search"


def test_convention_detection_defaults_to_legacy_when_no_markers(provider):
    """Test that auto-detection defaults to legacy when no convention markers found."""
    span = make_span(
        provider,
        0xAAA,
        0xBBB,
        0xCCC,
        "chat",
        {"gen_ai.operation.name": "chat"},  # No convention markers
        lambda s: s.add_event("gen_ai.choice", {"message": '[{"text": "response"}]'}),
    )

    mapper = StrandsInMemorySessionMapper()
    mapper.map_to_session([span], "sid")

    # Should default to legacy
    assert mapper._convention_version == GenAIConventionVersion.LEGACY
