"""Helpers for building mock genai-sdk SpanRecord objects for OpenSearch tests."""

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


@dataclass
class SpanRecord:
    """Mimics opensearch_genai_observability_sdk_py.retrieval.SpanRecord."""

    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    name: str = ""
    start_time: str = ""
    end_time: str = ""
    operation_name: str = ""
    agent_name: str = ""
    model: str = ""
    input_messages: list = field(default_factory=list)
    output_messages: list = field(default_factory=list)
    tool_name: str = ""
    tool_call_arguments: str = ""
    tool_call_result: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict = field(default_factory=dict)


def make_agent_span(
    trace_id="trace-1",
    span_id="agent-1",
    parent_span_id="",
    user_prompt="Hello",
    agent_response="Hi there",
    start_time="2026-01-01T00:00:00Z",
    end_time="2026-01-01T00:00:01Z",
) -> SpanRecord:
    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name="invoke_agent",
        start_time=start_time,
        end_time=end_time,
        operation_name="invoke_agent",
        input_messages=[Message(role="user", content=user_prompt)],
        output_messages=[Message(role="assistant", content=agent_response)],
    )


def make_chat_span(
    trace_id="trace-1",
    span_id="chat-1",
    parent_span_id="agent-1",
    user_input="Hello",
    assistant_output="Hi there",
    start_time="2026-01-01T00:00:00.1Z",
    end_time="2026-01-01T00:00:00.5Z",
) -> SpanRecord:
    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name="chat",
        start_time=start_time,
        end_time=end_time,
        operation_name="chat",
        input_messages=[Message(role="user", content=user_input)],
        output_messages=[Message(role="assistant", content=assistant_output)],
    )


def make_tool_span(
    trace_id="trace-1",
    span_id="tool-1",
    parent_span_id="agent-1",
    tool_name="get_weather",
    arguments='{"city": "Paris"}',
    result="Sunny, 22C",
    start_time="2026-01-01T00:00:00.5Z",
    end_time="2026-01-01T00:00:00.8Z",
) -> SpanRecord:
    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name="execute_tool",
        start_time=start_time,
        end_time=end_time,
        operation_name="execute_tool",
        tool_name=tool_name,
        tool_call_arguments=arguments,
        tool_call_result=result,
    )
