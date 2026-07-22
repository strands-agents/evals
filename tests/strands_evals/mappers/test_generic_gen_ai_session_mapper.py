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
        assert "ToolExecutionSpan" in span_types or "AgentInvocationSpan" in span_types
