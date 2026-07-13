"""Tests for GenericGenAISessionMapper - dict-format spans with gen_ai.* attributes."""

from strands_evals.mappers import GenericGenAISessionMapper
from strands_evals.types.trace import AgentInvocationSpan, InferenceSpan, ToolExecutionSpan


def make_span(
    trace_id="trace-1",
    span_id="span-1",
    parent_span_id=None,
    name="test-span",
    operation_name="",
    attributes=None,
    span_events=None,
    scope_name="custom-tracer",
):
    """Build a dict span with gen_ai.* attributes for GenericGenAISessionMapper."""
    attrs = attributes or {}
    if operation_name:
        attrs.setdefault("gen_ai.operation.name", operation_name)
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "start_time": 1700000000000000000,
        "end_time": 1700000001000000000,
        "attributes": attrs,
        "scope": {"name": scope_name, "version": "1.0"},
        "status": {"code": "OK"},
        "span_events": span_events or [],
    }


class TestToolExecutionSpan:
    def setup_method(self):
        self.mapper = GenericGenAISessionMapper()

    def test_tool_span_from_attributes(self):
        """Tool data extracted from gen_ai.tool.* attributes."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.name": "web_search",
                "gen_ai.tool.call.id": "call-123",
                "gen_ai.tool.input": '{"query": "weather london"}',
                "gen_ai.tool.output": "15C and cloudy",
                "gen_ai.tool.status": "success",
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")

        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_call.name == "web_search"
        assert tool.tool_call.arguments == {"query": "weather london"}
        assert tool.tool_call.tool_call_id == "call-123"
        assert tool.tool_result.content == "15C and cloudy"
        assert tool.tool_result.error is None

    def test_tool_span_from_events(self):
        """Tool data extracted from span_events (Strands telemetry format)."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.name": "calc",
                "gen_ai.tool.call.id": "t1",
                "gen_ai.tool.status": "success",
            },
            span_events=[
                {"event_name": "gen_ai.tool.message", "timestamp": 0, "attributes": {"content": '{"expr": "2+2"}'}},
                {"event_name": "gen_ai.choice", "timestamp": 0, "attributes": {"message": '[{"text": "4"}]'}},
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_call.name == "calc"
        assert tool.tool_call.arguments == {"expr": "2+2"}
        assert tool.tool_result.content == "4"

    def test_tool_span_missing_name_skipped(self):
        """Tool span without gen_ai.tool.name is skipped."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.input": '{"x": 1}',
                "gen_ai.tool.output": "result",
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        assert session.traces == []


class TestAgentInvocationSpan:
    def setup_method(self):
        self.mapper = GenericGenAISessionMapper()

    def test_agent_span_with_events(self):
        """Agent span extracts prompt/response from events and tools from attributes."""
        span = make_span(
            operation_name="invoke_agent",
            attributes={
                "gen_ai.agent.tools": '["web_search", "calculator"]',
                "session.id": "sess-1",
            },
            span_events=[
                {
                    "event_name": "gen_ai.user.message",
                    "timestamp": 0,
                    "attributes": {"content": '[{"text": "What is 2+2?"}]'},
                },
                {
                    "event_name": "gen_ai.choice",
                    "timestamp": 0,
                    "attributes": {"message": "The answer is 4."},
                },
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.user_prompt == "What is 2+2?"
        assert agent.agent_response == "The answer is 4."
        assert len(agent.available_tools) == 2
        assert agent.available_tools[0].name == "web_search"
        assert agent.available_tools[1].name == "calculator"

    def test_agent_span_no_events(self):
        """Agent span with no events still produces a span (empty prompt/response)."""
        span = make_span(operation_name="invoke_agent")
        session = self.mapper.map_to_session([span], "sess-1")

        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.user_prompt == ""
        assert agent.agent_response == ""


class TestInferenceSpan:
    def setup_method(self):
        self.mapper = GenericGenAISessionMapper()

    def test_inference_span_with_events(self):
        """Inference span extracts messages from events."""
        span = make_span(
            operation_name="chat",
            span_events=[
                {"event_name": "gen_ai.user.message", "timestamp": 0, "attributes": {"content": '[{"text": "Hello"}]'}},
                {"event_name": "gen_ai.choice", "timestamp": 0, "attributes": {"message": '[{"text": "Hi there!"}]'}},
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        assert len(inference.messages) == 2

    def test_inference_span_no_events_filtered(self):
        """Inference span with no events (no messages) is filtered out."""
        span = make_span(operation_name="chat")
        session = self.mapper.map_to_session([span], "sess-1")
        assert session.traces == []


class TestSessionBuilding:
    def setup_method(self):
        self.mapper = GenericGenAISessionMapper()

    def test_empty_input(self):
        """Empty list produces empty session."""
        session = self.mapper.map_to_session([], "sess-1")
        assert len(session.traces) == 0

    def test_multi_span_trace(self):
        """Multiple spans in same trace grouped correctly."""
        spans = [
            make_span(trace_id="t1", span_id="s1", operation_name="invoke_agent"),
            make_span(
                trace_id="t1",
                span_id="s2",
                parent_span_id="s1",
                operation_name="execute_tool",
                attributes={"gen_ai.tool.name": "calc", "gen_ai.tool.input": "{}", "gen_ai.tool.output": "4"},
            ),
        ]
        session = self.mapper.map_to_session(spans, "sess-1")

        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 2

    def test_session_id_filtering(self):
        """Spans filtered by session.id when present."""
        spans = [
            make_span(trace_id="t1", span_id="s1", operation_name="invoke_agent", attributes={"session.id": "sess-A"}),
            make_span(trace_id="t2", span_id="s2", operation_name="invoke_agent", attributes={"session.id": "sess-B"}),
        ]
        session = self.mapper.map_to_session(spans, "sess-A")

        assert len(session.traces) == 1
        assert session.traces[0].trace_id == "t1"

    def test_no_session_id_includes_all(self):
        """When no spans have session.id, all are included."""
        spans = [
            make_span(trace_id="t1", span_id="s1", operation_name="invoke_agent"),
            make_span(trace_id="t2", span_id="s2", operation_name="invoke_agent"),
        ]
        session = self.mapper.map_to_session(spans, "any-id")
        assert len(session.traces) == 2
