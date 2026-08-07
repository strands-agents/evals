"""Tests for GenericGenAISessionMapper - dict-format spans with gen_ai.* attributes."""

import json
from pathlib import Path

import pytest

from strands_evals.mappers import GenericGenAISessionMapper, detect_otel_mapper
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

    def test_tool_span_from_operation_details_event(self):
        """Tool data extracted from gen_ai.client.inference.operation.details unified event."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.name": "get_weather",
                "gen_ai.tool.call.id": "call-unified-1",
            },
            span_events=[
                {
                    "event_name": "gen_ai.client.inference.operation.details",
                    "timestamp": 0,
                    "attributes": {
                        "gen_ai.input.messages": json.dumps(
                            [{"role": "user", "parts": [{"type": "text", "content": '{"location": "Seattle"}'}]}]
                        ),
                        "gen_ai.output.messages": json.dumps(
                            [{"role": "assistant", "parts": [{"type": "text", "content": "62F and cloudy"}]}]
                        ),
                    },
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_call.name == "get_weather"
        assert tool.tool_call.arguments == {"location": "Seattle"}
        assert tool.tool_result.content == "62F and cloudy"
        assert tool.tool_result.error is None

    def test_tool_span_operation_details_with_tool_result_content(self):
        """Tool result from operation.details with ToolResultContent (role=tool)."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.name": "search",
                "gen_ai.tool.call.id": "call-unified-2",
            },
            span_events=[
                {
                    "event_name": "gen_ai.client.inference.operation.details",
                    "timestamp": 0,
                    "attributes": {
                        "gen_ai.input.messages": json.dumps(
                            [{"role": "user", "parts": [{"type": "text", "content": '{"query": "cats"}'}]}]
                        ),
                        "gen_ai.output.messages": json.dumps(
                            [{"role": "tool", "id": "call-unified-2", "response": "Found 10 cats"}]
                        ),
                    },
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_call.arguments == {"query": "cats"}
        assert tool.tool_result.content == "Found 10 cats"


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

    def test_agent_span_from_operation_details_event(self):
        """Agent prompt/response extracted from gen_ai.client.inference.operation.details."""
        span = make_span(
            operation_name="invoke_agent",
            attributes={
                "gen_ai.agent.tools": '["get_weather"]',
            },
            span_events=[
                {
                    "event_name": "gen_ai.client.inference.operation.details",
                    "timestamp": 0,
                    "attributes": {
                        "gen_ai.input.messages": json.dumps(
                            [{"role": "user", "parts": [{"type": "text", "content": "What is the weather in Paris?"}]}]
                        ),
                        "gen_ai.output.messages": json.dumps(
                            [{"role": "assistant", "parts": [{"type": "text", "content": "57F and rainy"}]}]
                        ),
                    },
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.user_prompt == "What is the weather in Paris?"
        assert agent.agent_response == "57F and rainy"

    def test_agent_span_operation_details_flat_content(self):
        """Agent response from operation.details with flat content string (no parts)."""
        span = make_span(
            operation_name="invoke_agent",
            span_events=[
                {
                    "event_name": "gen_ai.client.inference.operation.details",
                    "timestamp": 0,
                    "attributes": {
                        "gen_ai.input.messages": json.dumps(
                            [{"role": "user", "parts": [{"type": "text", "content": "Hello"}]}]
                        ),
                        "gen_ai.output.messages": json.dumps([{"role": "assistant", "content": "Hi there!"}]),
                    },
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.user_prompt == "Hello"
        assert agent.agent_response == "Hi there!"


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

    def test_string_nanosecond_timestamp_parsed_correctly(self):
        """String-encoded ns epoch (OTLP/JSON int64 encoding) parses to correct datetime."""
        from datetime import datetime, timezone

        result = self.mapper.parse_timestamp("1700000000000000000")
        expected = datetime.fromtimestamp(1700000000, tz=timezone.utc)
        assert result == expected

    def test_string_timestamp_in_span_produces_correct_time(self):
        """Span with string timestamps (CloudWatch/ADOT path) maps to correct year."""
        span = make_span(
            operation_name="invoke_agent",
            attributes={"gen_ai.tool.definitions": "[]"},
        )
        span["start_time"] = "1700000000000000000"
        span["end_time"] = "1700000001000000000"

        session = self.mapper.map_to_session([span], "sess-1")
        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.span_info.start_time.year == 2023


# =============================================================================
# Fixture-based integration tests — real captured traces
# =============================================================================

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_PYDANTIC_AI_SPANS_FILE = _FIXTURES_DIR / "pydantic_ai_live_spans.json"
_AUTOGEN_SPANS_FILE = _FIXTURES_DIR / "autogen_live_spans.json"


@pytest.fixture(scope="module")
def pydantic_ai_spans():
    """Load real PydanticAI spans captured from Agent.instrument_all()."""
    with open(_PYDANTIC_AI_SPANS_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pydantic_ai_session(pydantic_ai_spans):
    """Map PydanticAI spans to a Session using GenericGenAISessionMapper."""
    mapper = GenericGenAISessionMapper()
    # Auto-detect session_id from gen_ai.conversation.id
    session_id = "test"
    for s in pydantic_ai_spans:
        sid = s.get("attributes", {}).get("gen_ai.conversation.id")
        if sid:
            session_id = str(sid)
            break
    return mapper.map_to_session(pydantic_ai_spans, session_id=session_id)


@pytest.fixture(scope="module")
def autogen_spans():
    """Load real AutoGen spans captured from autogen_core native telemetry."""
    with open(_AUTOGEN_SPANS_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def autogen_session(autogen_spans):
    """Map AutoGen spans to a Session using GenericGenAISessionMapper."""
    mapper = GenericGenAISessionMapper()
    return mapper.map_to_session(autogen_spans, session_id="test")


class TestPydanticAIFixtureRouting:
    """Verify detect_otel_mapper routes PydanticAI spans to GenericGenAISessionMapper."""

    def test_routes_to_generic_mapper(self, pydantic_ai_spans):
        """PydanticAI scope 'pydantic-ai' routes to GenericGenAISessionMapper."""
        mapper = detect_otel_mapper(pydantic_ai_spans)
        assert isinstance(mapper, GenericGenAISessionMapper)

    def test_scope_is_pydantic_ai(self, pydantic_ai_spans):
        """All spans have scope.name = 'pydantic-ai'."""
        scopes = {s["scope"]["name"] for s in pydantic_ai_spans}
        assert scopes == {"pydantic-ai"}

    def test_has_gen_ai_operation_names(self, pydantic_ai_spans):
        """Spans use gen_ai.operation.name with expected values."""
        operations = {s["attributes"].get("gen_ai.operation.name") for s in pydantic_ai_spans}
        assert "chat" in operations
        assert "execute_tool" in operations
        assert "invoke_agent" in operations


class TestPydanticAIFixtureIntegration:
    """Integration tests using real PydanticAI trace (Agent + get_weather tool)."""

    def test_session_has_traces(self, pydantic_ai_session):
        """PydanticAI fixture produces at least one trace."""
        assert len(pydantic_ai_session.traces) >= 1

    def test_produces_expected_span_types(self, pydantic_ai_session):
        """Real trace produces InferenceSpan, ToolExecutionSpan, and AgentInvocationSpan."""
        all_spans = [s for t in pydantic_ai_session.traces for s in t.spans]
        span_types = {type(s) for s in all_spans}
        assert InferenceSpan in span_types
        assert ToolExecutionSpan in span_types
        assert AgentInvocationSpan in span_types

    def test_tool_span_extracts_name_and_args(self, pydantic_ai_session):
        """get_weather tool: extracts name, arguments, and result from attributes."""
        all_spans = [s for t in pydantic_ai_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1

        tool = tool_spans[0]
        assert tool.tool_call.name == "get_weather"
        assert tool.tool_call.arguments == {"city": "Seattle"}
        assert "62°F" in tool.tool_result.content

    def test_inference_span_has_messages(self, pydantic_ai_session):
        """Chat spans produce InferenceSpans with user and assistant messages."""
        all_spans = [s for t in pydantic_ai_session.traces for s in t.spans]
        inference_spans = [s for s in all_spans if isinstance(s, InferenceSpan)]
        assert len(inference_spans) >= 1

        # At least one inference span should have an assistant text response
        has_assistant_text = False
        for span in inference_spans:
            for msg in span.messages:
                if msg.role.value == "assistant":
                    for c in msg.content:
                        if hasattr(c, "text") and c.text and "Seattle" in c.text:
                            has_assistant_text = True
        assert has_assistant_text

    def test_agent_span_extracts_user_prompt(self, pydantic_ai_session):
        """Agent invocation span extracts user prompt from conversation messages."""
        all_spans = [s for t in pydantic_ai_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1

        agent = agent_spans[0]
        assert "weather" in agent.user_prompt.lower()
        assert "Seattle" in agent.user_prompt

    def test_agent_span_extracts_final_result(self, pydantic_ai_session):
        """Agent invocation span extracts final_result as agent_response."""
        all_spans = [s for t in pydantic_ai_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        agent = agent_spans[0]
        assert "62°F" in agent.agent_response

    def test_session_id_filtering_with_conversation_id(self, pydantic_ai_spans):
        """Filtering by gen_ai.conversation.id correctly includes/excludes spans."""
        mapper = GenericGenAISessionMapper()

        # Get the real conversation_id
        real_id = pydantic_ai_spans[0]["attributes"]["gen_ai.conversation.id"]

        # Correct session_id → spans included
        session = mapper.map_to_session(pydantic_ai_spans, session_id=real_id)
        assert len(session.traces) >= 1

        # Wrong session_id → spans excluded
        session_wrong = mapper.map_to_session(pydantic_ai_spans, session_id="nonexistent-id")
        assert len(session_wrong.traces) == 0


class TestAutoGenFixtureRouting:
    """Verify detect_otel_mapper routes AutoGen spans to GenericGenAISessionMapper."""

    def test_routes_to_generic_mapper(self, autogen_spans):
        """AutoGen scope 'autogen-core' routes to GenericGenAISessionMapper."""
        mapper = detect_otel_mapper(autogen_spans)
        assert isinstance(mapper, GenericGenAISessionMapper)

    def test_scope_is_autogen_core(self, autogen_spans):
        """All spans have scope.name = 'autogen-core'."""
        scopes = {s["scope"]["name"] for s in autogen_spans}
        assert scopes == {"autogen-core"}

    def test_has_gen_ai_operation_names(self, autogen_spans):
        """Spans use gen_ai.operation.name with expected values."""
        operations = {s["attributes"].get("gen_ai.operation.name") for s in autogen_spans}
        assert "execute_tool" in operations
        assert "invoke_agent" in operations
        assert "create_agent" in operations


class TestAutoGenFixtureIntegration:
    """Integration tests using real AutoGen trace (AssistantAgent + calculate tool)."""

    def test_session_has_traces(self, autogen_session):
        """AutoGen fixture produces at least one trace."""
        assert len(autogen_session.traces) >= 1

    def test_tool_span_extracted(self, autogen_session):
        """execute_tool span for 'calculate' is correctly parsed."""
        all_spans = [s for t in autogen_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) >= 1

        calc = next(s for s in tool_spans if s.tool_call.name == "calculate")
        assert calc.tool_call.tool_call_id == "toolu_bdrk_011nDreLrC9nv8NZs6iGrx6d"

    def test_agent_span_extracted(self, autogen_session):
        """invoke_agent span for 'math_assistant' is correctly parsed."""
        all_spans = [s for t in autogen_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) >= 1

    def test_create_agent_span_skipped(self, autogen_spans):
        """create_agent operation is not a recognized span type (skipped gracefully)."""
        mapper = GenericGenAISessionMapper()
        session = mapper.map_to_session(autogen_spans, session_id="test")
        all_spans = [s for t in session.traces for s in t.spans]
        # create_agent should not crash; it's simply not mapped to any known type
        # The 3 raw spans are: create_agent, execute_tool, invoke_agent
        # Only execute_tool and invoke_agent should produce mapped spans
        span_types = {type(s).__name__ for s in all_spans}
        assert span_types == {"ToolExecutionSpan", "AgentInvocationSpan"}


# =============================================================================
# Regression tests for jjbuck review feedback
# =============================================================================


class TestToolErrorHandling:
    """Regression: error.type + span status ERROR must surface as tool_result.error."""

    def setup_method(self):
        self.mapper = GenericGenAISessionMapper()

    def test_error_type_attribute_sets_tool_error(self):
        """Tool span with error.type="TimeoutError" and status ERROR reports error."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.name": "slow_api",
                "gen_ai.tool.call.id": "call-err-1",
                "gen_ai.tool.input": '{"timeout": 30}',
                "error.type": "TimeoutError",
            },
        )
        # Override status to ERROR
        span["status"] = {"code": "ERROR"}

        session = self.mapper.map_to_session([span], "sess-1")
        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_result.error == "TimeoutError"

    def test_span_status_error_without_error_type(self):
        """Tool span with status ERROR but no error.type still reports an error."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.name": "flaky_tool",
                "gen_ai.tool.call.id": "call-err-2",
            },
        )
        span["status"] = {"code": "ERROR"}

        session = self.mapper.map_to_session([span], "sess-1")
        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_result.error == "ERROR"

    def test_legacy_tool_status_failure_still_works(self):
        """Legacy gen_ai.tool.status != 'success' still reports as error."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.name": "legacy_tool",
                "gen_ai.tool.call.id": "call-err-3",
                "gen_ai.tool.status": "failed",
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_result.error == "failed"

    def test_error_type_takes_precedence_over_tool_status(self):
        """error.type takes precedence over gen_ai.tool.status."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.name": "dual_error",
                "gen_ai.tool.call.id": "call-err-4",
                "gen_ai.tool.status": "failed",
                "error.type": "ConnectionError",
            },
        )
        span["status"] = {"code": "ERROR"}

        session = self.mapper.map_to_session([span], "sess-1")
        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_result.error == "ConnectionError"

    def test_successful_tool_no_error(self):
        """Successful tool span (no error.type, status OK) has no error."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.name": "good_tool",
                "gen_ai.tool.call.id": "call-ok",
                "gen_ai.tool.status": "success",
                "gen_ai.tool.output": "result",
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_result.error is None

    def test_status_none_does_not_crash(self):
        """Span with status: None must not raise AttributeError."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.name": "safe_tool",
                "gen_ai.tool.call.id": "call-null",
                "gen_ai.tool.output": "ok",
            },
        )
        span["status"] = None

        session = self.mapper.map_to_session([span], "sess-1")
        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_result.error is None
        assert tool.tool_result.content == "ok"

    def test_status_non_dict_does_not_crash(self):
        """Span with non-dict status (e.g. string) must not crash."""
        span = make_span(
            operation_name="execute_tool",
            attributes={
                "gen_ai.tool.name": "robust_tool",
                "gen_ai.tool.call.id": "call-str",
                "gen_ai.tool.output": "done",
            },
        )
        span["status"] = "OK"

        session = self.mapper.map_to_session([span], "sess-1")
        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_result.error is None


class TestOperationDetailsEvent:
    """Regression: gen_ai.client.inference.operation.details event must produce messages."""

    def setup_method(self):
        self.mapper = GenericGenAISessionMapper()

    def test_operation_details_event_structured_messages(self):
        """Chat span with operation.details event produces user and assistant messages."""
        span = make_span(
            operation_name="chat",
            span_events=[
                {
                    "event_name": "gen_ai.client.inference.operation.details",
                    "timestamp": 0,
                    "attributes": {
                        "gen_ai.input.messages": json.dumps(
                            [{"role": "user", "parts": [{"type": "text", "content": "Hello"}]}]
                        ),
                        "gen_ai.output.messages": json.dumps(
                            [{"role": "assistant", "parts": [{"type": "text", "content": "Hi there!"}]}]
                        ),
                    },
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        assert len(inference.messages) == 2
        assert inference.messages[0].role.value == "user"
        assert inference.messages[1].role.value == "assistant"

    def test_operation_details_event_list_format(self):
        """operation.details with already-parsed list (not JSON string) still works."""
        span = make_span(
            operation_name="chat",
            span_events=[
                {
                    "event_name": "gen_ai.client.inference.operation.details",
                    "timestamp": 0,
                    "attributes": {
                        "gen_ai.input.messages": [
                            {"role": "user", "parts": [{"type": "text", "content": "What's 2+2?"}]}
                        ],
                        "gen_ai.output.messages": [{"role": "assistant", "parts": [{"type": "text", "content": "4"}]}],
                    },
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        assert len(inference.messages) == 2

    def test_operation_details_event_with_tool_role(self):
        """operation.details with role 'tool' messages normalizes to ToolResultContent."""
        span = make_span(
            operation_name="chat",
            span_events=[
                {
                    "event_name": "gen_ai.client.inference.operation.details",
                    "timestamp": 0,
                    "attributes": {
                        "gen_ai.input.messages": json.dumps(
                            [
                                {"role": "user", "parts": [{"type": "text", "content": "Check weather"}]},
                                {"role": "tool", "id": "call-1", "response": "72F sunny"},
                            ]
                        ),
                        "gen_ai.output.messages": json.dumps(
                            [{"role": "assistant", "parts": [{"type": "text", "content": "It's 72F and sunny"}]}]
                        ),
                    },
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        # user + tool (as UserMessage) + assistant = 3 messages
        assert len(inference.messages) == 3
        # The tool message should be a UserMessage with ToolResultContent
        tool_msg = inference.messages[1]
        assert tool_msg.role.value == "user"
        assert hasattr(tool_msg.content[0], "content")
        assert tool_msg.content[0].content == "72F sunny"


class TestSessionFilteringTraceLevel:
    """Regression: untagged spans must not leak into sessions with tagged data."""

    def setup_method(self):
        self.mapper = GenericGenAISessionMapper()

    def test_untagged_trace_excluded_when_sessions_present(self):
        """Untagged trace excluded when session-tagged data exists."""
        spans = [
            # Trace 1: tagged with session A
            make_span(
                trace_id="t1",
                span_id="s1",
                operation_name="invoke_agent",
                attributes={"session.id": "sess-A"},
            ),
            # Trace 2: completely untagged (unrelated)
            make_span(
                trace_id="t2",
                span_id="s2",
                operation_name="invoke_agent",
            ),
            # Trace 3: tagged with session B
            make_span(
                trace_id="t3",
                span_id="s3",
                operation_name="invoke_agent",
                attributes={"session.id": "sess-B"},
            ),
        ]
        session = self.mapper.map_to_session(spans, "sess-A")

        # Only trace t1 should be included; t2 (untagged) should NOT leak in
        assert len(session.traces) == 1
        assert session.traces[0].trace_id == "t1"

    def test_untagged_span_in_tagged_trace_included(self):
        """Untagged span within a trace that has a tagged span IS included."""
        spans = [
            # Same trace: one span tagged, one not
            make_span(
                trace_id="t1",
                span_id="s1",
                operation_name="invoke_agent",
                attributes={"session.id": "sess-A"},
            ),
            make_span(
                trace_id="t1",
                span_id="s2",
                parent_span_id="s1",
                operation_name="execute_tool",
                attributes={"gen_ai.tool.name": "calc", "gen_ai.tool.output": "4"},
            ),
        ]
        session = self.mapper.map_to_session(spans, "sess-A")

        # Both spans in the same trace should be included
        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 2

    def test_mixed_tagged_untagged_traces(self):
        """Only traces with at least one matching span are included."""
        spans = [
            # t1: session A
            make_span(trace_id="t1", span_id="s1", operation_name="invoke_agent", attributes={"session.id": "sess-A"}),
            make_span(
                trace_id="t1",
                span_id="s2",
                parent_span_id="s1",
                operation_name="execute_tool",
                attributes={"gen_ai.tool.name": "x", "gen_ai.tool.output": "y"},
            ),
            # t2: untagged
            make_span(trace_id="t2", span_id="s3", operation_name="invoke_agent"),
            # t3: session B
            make_span(
                trace_id="t3",
                span_id="s4",
                operation_name="invoke_agent",
                attributes={"gen_ai.conversation.id": "sess-B"},
            ),
        ]
        session_a = self.mapper.map_to_session(spans, "sess-A")
        session_b = self.mapper.map_to_session(spans, "sess-B")

        assert len(session_a.traces) == 1
        assert session_a.traces[0].trace_id == "t1"
        assert len(session_b.traces) == 1
        assert session_b.traces[0].trace_id == "t3"

    def test_different_session_span_in_same_trace_excluded(self):
        """Two spans sharing a trace with different session tags — only matching one returned."""
        spans = [
            make_span(
                trace_id="tSHARED",
                span_id="s1",
                operation_name="invoke_agent",
                attributes={"session.id": "session-123"},
            ),
            make_span(
                trace_id="tSHARED",
                span_id="s2",
                operation_name="invoke_agent",
                attributes={"session.id": "session-456"},
            ),
        ]
        session = self.mapper.map_to_session(spans, "session-123")

        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1
        assert session.traces[0].spans[0].span_info.span_id == "s1"

    def test_untagged_span_inherits_while_foreign_excluded(self):
        """Untagged span inherits trace match; explicitly foreign-tagged span excluded."""
        spans = [
            make_span(
                trace_id="tSHARED",
                span_id="s1",
                operation_name="invoke_agent",
                attributes={"session.id": "session-123"},
            ),
            make_span(
                trace_id="tSHARED",
                span_id="s2",
                parent_span_id="s1",
                operation_name="execute_tool",
                attributes={"gen_ai.tool.name": "calc", "gen_ai.tool.output": "4"},
            ),
            make_span(
                trace_id="tSHARED",
                span_id="s3",
                operation_name="invoke_agent",
                attributes={"session.id": "session-456"},
            ),
        ]
        session = self.mapper.map_to_session(spans, "session-123")

        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 2
        span_ids = {s.span_info.span_id for s in session.traces[0].spans}
        assert span_ids == {"s1", "s2"}


class TestAgentResponseFromOutputMessages:
    """Regression: invoke_agent must extract agent_response from gen_ai.output.messages."""

    def setup_method(self):
        self.mapper = GenericGenAISessionMapper()

    def test_agent_response_from_output_messages(self):
        """Agent response extracted from gen_ai.output.messages attribute."""
        span = make_span(
            operation_name="invoke_agent",
            attributes={
                "gen_ai.input.messages": json.dumps(
                    [{"role": "user", "parts": [{"type": "text", "content": "What's the weather?"}]}]
                ),
                "gen_ai.output.messages": json.dumps(
                    [{"role": "assistant", "parts": [{"type": "text", "content": "It's sunny and 72F"}]}]
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.user_prompt == "What's the weather?"
        assert agent.agent_response == "It's sunny and 72F"

    def test_agent_response_flat_content_string(self):
        """Agent response from gen_ai.output.messages with flat content (no parts)."""
        span = make_span(
            operation_name="invoke_agent",
            attributes={
                "gen_ai.output.messages": json.dumps([{"role": "assistant", "content": "The answer is 42"}]),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.agent_response == "The answer is 42"

    def test_final_result_used_when_no_output_messages(self):
        """Falls back to final_result when gen_ai.output.messages absent."""
        span = make_span(
            operation_name="invoke_agent",
            attributes={
                "final_result": "Legacy result",
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.agent_response == "Legacy result"


class TestToolRoleMessages:
    """Regression: role 'tool' in gen_ai.input.messages must be parsed as ToolResultContent."""

    def setup_method(self):
        self.mapper = GenericGenAISessionMapper()

    def test_tool_role_with_response_field(self):
        """OTel convention: role 'tool' with 'response' field parsed correctly."""
        span = make_span(
            operation_name="chat",
            attributes={
                "gen_ai.input.messages": json.dumps(
                    [
                        {"role": "user", "parts": [{"type": "text", "content": "Search for cats"}]},
                        {"role": "tool", "id": "call-t1", "response": "Found 10 cats"},
                    ]
                ),
                "gen_ai.output.messages": json.dumps(
                    [{"role": "assistant", "parts": [{"type": "text", "content": "I found 10 cats for you"}]}]
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        assert len(inference.messages) == 3  # user + tool + assistant

        # Check tool message
        tool_msg = inference.messages[1]
        assert tool_msg.role.value == "user"
        assert tool_msg.content[0].content == "Found 10 cats"
        assert tool_msg.content[0].tool_call_id == "call-t1"

    def test_tool_role_with_dict_response(self):
        """Tool role with dict response is JSON-serialized."""
        span = make_span(
            operation_name="chat",
            attributes={
                "gen_ai.input.messages": json.dumps(
                    [
                        {"role": "tool", "id": "call-t2", "response": {"status": "ok", "count": 5}},
                    ]
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        tool_msg = inference.messages[0]
        assert '"count": 5' in tool_msg.content[0].content
        assert '"status": "ok"' in tool_msg.content[0].content

    def test_pydantic_ai_tool_call_response_still_works(self):
        """PydanticAI format: tool_call_response under user role still parsed correctly."""
        span = make_span(
            operation_name="chat",
            attributes={
                "gen_ai.input.messages": json.dumps(
                    [
                        {
                            "role": "user",
                            "parts": [{"type": "tool_call_response", "id": "call-p1", "result": "42"}],
                        }
                    ]
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        tool_msg = inference.messages[0]
        assert tool_msg.content[0].content == "42"
        assert tool_msg.content[0].tool_call_id == "call-p1"

    def test_tool_result_json_block_preserved(self):
        """toolResult.content with a json block serializes it rather than dropping."""
        span = make_span(
            operation_name="chat",
            span_events=[
                {
                    "event_name": "gen_ai.tool.message",
                    "timestamp": 0,
                    "attributes": {
                        "content": json.dumps(
                            [
                                {
                                    "toolResult": {
                                        "toolUseId": "tool-j1",
                                        "content": [{"json": {"result": 42, "status": "ok"}}],
                                    }
                                }
                            ]
                        )
                    },
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        tool_msg = inference.messages[0]
        # Assert exact serialized JSON string, not just substrings
        assert tool_msg.content[0].content == '{"result": 42, "status": "ok"}'

    def test_tool_result_image_block_placeholder(self):
        """toolResult.content with an image block produces [image] placeholder."""
        span = make_span(
            operation_name="chat",
            span_events=[
                {
                    "event_name": "gen_ai.tool.message",
                    "timestamp": 0,
                    "attributes": {
                        "content": json.dumps(
                            [
                                {
                                    "toolResult": {
                                        "toolUseId": "tool-img1",
                                        "content": [{"image": {"format": "png", "data": "base64..."}}],
                                    }
                                }
                            ]
                        )
                    },
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        tool_msg = inference.messages[0]
        assert "[image]" in tool_msg.content[0].content

    def test_tool_result_mixed_text_and_json_blocks(self):
        """toolResult.content with text + json blocks joins both with newline."""
        span = make_span(
            operation_name="chat",
            span_events=[
                {
                    "event_name": "gen_ai.tool.message",
                    "timestamp": 0,
                    "attributes": {
                        "content": json.dumps(
                            [
                                {
                                    "toolResult": {
                                        "toolUseId": "tool-mix1",
                                        "content": [
                                            {"text": "Summary:"},
                                            {"json": {"temp": 72, "unit": "F"}},
                                        ],
                                    }
                                }
                            ]
                        )
                    },
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        tool_msg = inference.messages[0]
        content = tool_msg.content[0].content
        assert "Summary:" in content
        assert "72" in content
        assert "\n" in content

    def test_user_role_tool_call_response_with_list_response(self):
        """role:'user' tool_call_response with list-valued 'response' joins text blocks."""
        span = make_span(
            operation_name="chat",
            attributes={
                "gen_ai.input.messages": json.dumps(
                    [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "type": "tool_call_response",
                                    "id": "call-list-u1",
                                    "response": [{"text": "Seattle: 62F, cloudy with light rain"}],
                                }
                            ],
                        }
                    ]
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        tool_msg = inference.messages[0]
        assert tool_msg.content[0].content == "Seattle: 62F, cloudy with light rain"
        assert tool_msg.content[0].tool_call_id == "call-list-u1"

    def test_tool_role_parts_branch_with_list_response(self):
        """role:'tool' parts branch with list-valued 'response' joins text blocks."""
        span = make_span(
            operation_name="chat",
            attributes={
                "gen_ai.input.messages": json.dumps(
                    [
                        {
                            "role": "tool",
                            "parts": [
                                {
                                    "type": "tool_call_response",
                                    "id": "call-list-t1",
                                    "response": [
                                        {"text": "Paris: 57F, rainy"},
                                        {"text": "Wind: 12mph NW"},
                                    ],
                                }
                            ],
                        }
                    ]
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        tool_msg = inference.messages[0]
        assert "Paris: 57F, rainy" in tool_msg.content[0].content
        assert "Wind: 12mph NW" in tool_msg.content[0].content
        assert tool_msg.content[0].tool_call_id == "call-list-t1"

    def test_tool_role_flat_branch_with_list_response(self):
        """role:'tool' flat (no parts) with list-valued 'response' joins text blocks."""
        span = make_span(
            operation_name="chat",
            attributes={
                "gen_ai.input.messages": json.dumps(
                    [
                        {
                            "role": "tool",
                            "id": "call-list-flat1",
                            "response": [{"text": "London: 55F, overcast"}],
                        }
                    ]
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        tool_msg = inference.messages[0]
        assert tool_msg.content[0].content == "London: 55F, overcast"
        assert tool_msg.content[0].tool_call_id == "call-list-flat1"

    def test_tool_role_flat_branch_with_json_block_in_list(self):
        """role:'tool' flat with json block in list serializes it, not repr."""
        span = make_span(
            operation_name="chat",
            attributes={
                "gen_ai.input.messages": json.dumps(
                    [
                        {
                            "role": "tool",
                            "id": "call-flat-json1",
                            "response": [
                                {"text": "Weather:"},
                                {"json": {"temp": 62, "unit": "F", "condition": "cloudy"}},
                            ],
                        }
                    ]
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        tool_msg = inference.messages[0]
        content = tool_msg.content[0].content
        # Must contain the text block and serialized json, not Python repr
        assert "Weather:" in content
        assert '"temp": 62' in content or '"temp":62' in content
        # Must NOT contain Python repr artifacts
        assert "[{" not in content or '{"' in content  # no list-repr wrapper
        assert "'json'" not in content  # no dict key as Python repr

    def test_non_dict_message_skipped_gracefully(self):
        """A non-dict entry in the message list is skipped without crashing."""
        span = make_span(
            operation_name="chat",
            attributes={
                "gen_ai.input.messages": json.dumps(
                    [
                        "unexpected string entry",
                        None,
                        {"role": "user", "parts": [{"type": "text", "content": "hello"}]},
                    ]
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        # Only the valid dict message should be parsed
        assert len(inference.messages) == 1
        assert inference.messages[0].content[0].text == "hello"

    def test_user_role_tool_call_response_falls_back_to_result_key(self):
        """role:'user' tool_call_response reads 'result' when 'response' is absent."""
        span = make_span(
            operation_name="chat",
            attributes={
                "gen_ai.input.messages": json.dumps(
                    [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "type": "tool_call_response",
                                    "id": "call-fallback1",
                                    "result": [{"text": "fallback content"}],
                                }
                            ],
                        }
                    ]
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        tool_msg = inference.messages[0]
        assert tool_msg.content[0].content == "fallback content"

    def test_tool_role_parts_branch_falls_back_to_result_key(self):
        """role:'tool' parts branch reads 'result' when 'response' is absent."""
        span = make_span(
            operation_name="chat",
            attributes={
                "gen_ai.input.messages": json.dumps(
                    [
                        {
                            "role": "tool",
                            "parts": [
                                {
                                    "type": "tool_call_response",
                                    "id": "call-fallback2",
                                    "result": [{"text": "tool fallback"}],
                                }
                            ],
                        }
                    ]
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        tool_msg = inference.messages[0]
        assert tool_msg.content[0].content == "tool fallback"


# =============================================================================
# OTel GenAI Semantic Convention fixture tests (current spec)
# Covers: gen_ai.client.inference.operation.details event, error.type on tools,
# gen_ai.output.messages on invoke_agent, role "tool" with tool_call_response,
# gen_ai.tool.call.arguments / gen_ai.tool.call.result, trace-level session filtering.
# =============================================================================

_OTEL_CONVENTION_SPANS_FILE = _FIXTURES_DIR / "otel_genai_convention_spans.json"


@pytest.fixture(scope="module")
def otel_convention_spans():
    """Load hand-crafted spans following current OTel GenAI semantic conventions."""
    with open(_OTEL_CONVENTION_SPANS_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def otel_convention_session(otel_convention_spans):
    """Map OTel convention spans to a Session."""
    mapper = GenericGenAISessionMapper()
    return mapper.map_to_session(otel_convention_spans, session_id="conv_otel_test_001")


class TestOtelConventionRouting:
    """Verify detect_otel_mapper routes unrecognized scope to GenericGenAISessionMapper."""

    def test_routes_to_generic_mapper(self, otel_convention_spans):
        """Custom scope 'my-weather-app' routes to GenericGenAISessionMapper."""
        mapper = detect_otel_mapper(otel_convention_spans)
        assert isinstance(mapper, GenericGenAISessionMapper)

    def test_scope_is_custom(self, otel_convention_spans):
        """All spans have unrecognized scope.name."""
        scopes = {s["scope"]["name"] for s in otel_convention_spans}
        assert scopes == {"my-weather-app"}


class TestOtelConventionIntegration:
    """Full integration tests using current OTel GenAI semantic convention format."""

    def test_session_has_single_trace(self, otel_convention_session):
        """All spans share one trace_id → single trace."""
        assert len(otel_convention_session.traces) == 1

    def test_produces_all_span_types(self, otel_convention_session):
        """Trace contains InferenceSpan, ToolExecutionSpan, and AgentInvocationSpan."""
        all_spans = [s for t in otel_convention_session.traces for s in t.spans]
        span_types = {type(s) for s in all_spans}
        assert InferenceSpan in span_types
        assert ToolExecutionSpan in span_types
        assert AgentInvocationSpan in span_types

    def test_agent_span_extracts_from_output_messages(self, otel_convention_session):
        """invoke_agent extracts user_prompt and agent_response from gen_ai.input/output.messages."""
        all_spans = [s for t in otel_convention_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1

        agent = agent_spans[0]
        assert "Paris" in agent.user_prompt and "London" in agent.user_prompt
        assert "57" in agent.agent_response and "timeout" in agent.agent_response.lower()

    def test_agent_span_has_tool_definitions(self, otel_convention_session):
        """invoke_agent extracts available_tools from gen_ai.tool.definitions."""
        all_spans = [s for t in otel_convention_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        agent = agent_spans[0]
        assert len(agent.available_tools) == 1
        assert agent.available_tools[0].name == "get_weather"

    def test_successful_tool_uses_call_arguments_and_result(self, otel_convention_session):
        """Successful tool uses gen_ai.tool.call.arguments and gen_ai.tool.call.result."""
        all_spans = [s for t in otel_convention_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]

        paris_tool = next(s for s in tool_spans if s.tool_call.tool_call_id == "call_paris_001")
        assert paris_tool.tool_call.name == "get_weather"
        assert paris_tool.tool_call.arguments == {"location": "Paris"}
        assert "rainy" in paris_tool.tool_result.content and "57" in paris_tool.tool_result.content
        assert paris_tool.tool_result.error is None

    def test_failed_tool_reports_error_type(self, otel_convention_session):
        """Failed tool with error.type and status ERROR surfaces tool_result.error."""
        all_spans = [s for t in otel_convention_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]

        london_tool = next(s for s in tool_spans if s.tool_call.tool_call_id == "call_london_001")
        assert london_tool.tool_call.name == "get_weather"
        assert london_tool.tool_result.error == "TimeoutError"

    def test_chat_span_with_operation_details_event(self, otel_convention_session):
        """Chat span with gen_ai.client.inference.operation.details produces messages."""
        all_spans = [s for t in otel_convention_session.traces for s in t.spans]
        inference_spans = [s for s in all_spans if isinstance(s, InferenceSpan)]
        assert len(inference_spans) == 2

    def test_operation_details_parses_tool_calls(self, otel_convention_session):
        """operation.details event extracts assistant tool_call parts."""
        all_spans = [s for t in otel_convention_session.traces for s in t.spans]
        inference_spans = [s for s in all_spans if isinstance(s, InferenceSpan)]

        # First chat span has the initial tool_call response
        first_chat = inference_spans[0]
        # Should have user message + assistant message with tool calls
        assert len(first_chat.messages) >= 2

    def test_operation_details_parses_tool_role(self, otel_convention_session):
        """operation.details event with role 'tool' produces ToolResultContent."""
        all_spans = [s for t in otel_convention_session.traces for s in t.spans]
        inference_spans = [s for s in all_spans if isinstance(s, InferenceSpan)]

        # Second chat span has role "tool" messages in input
        second_chat = inference_spans[1]
        # Should contain: user + assistant + 2x tool + assistant output = multiple messages
        assert len(second_chat.messages) >= 4

        # Find the tool role messages (mapped as UserMessage with ToolResultContent)
        tool_messages = []
        for msg in second_chat.messages:
            if msg.role.value == "user":
                for c in msg.content:
                    if hasattr(c, "tool_call_id") and c.tool_call_id:
                        tool_messages.append(c)
        assert len(tool_messages) >= 1
        # The Paris tool result should be present
        paris_result = next((m for m in tool_messages if m.tool_call_id == "call_paris_001"), None)
        assert paris_result is not None
        assert "rainy" in paris_result.content and "57" in paris_result.content

    def test_session_id_filtering(self, otel_convention_spans):
        """Correct session_id includes all spans; wrong one excludes them."""
        mapper = GenericGenAISessionMapper()

        # Correct session_id
        session = mapper.map_to_session(otel_convention_spans, session_id="conv_otel_test_001")
        assert len(session.traces) == 1
        all_spans = [s for t in session.traces for s in t.spans]
        assert len(all_spans) == 5  # 1 agent + 2 chat + 2 tool

        # Wrong session_id → no traces
        session_wrong = mapper.map_to_session(otel_convention_spans, session_id="wrong-id")
        assert len(session_wrong.traces) == 0
