"""Tests for OpenSearchSessionMapper."""

from strands_evals.mappers.opensearch_session_mapper import OpenSearchSessionMapper
from strands_evals.types.trace import AgentInvocationSpan, InferenceSpan, ToolExecutionSpan
from tests.strands_evals.opensearch_helpers import (
    Message,
    SpanRecord,
    make_agent_span,
    make_chat_span,
    make_tool_span,
)


class TestSpanConversion:
    def setup_method(self):
        self.mapper = OpenSearchSessionMapper()

    def test_invoke_agent_span_creates_agent_invocation(self):
        """invoke_agent operation maps to AgentInvocationSpan with prompt and response."""
        records = [make_agent_span(user_prompt="What's the weather?", agent_response="It's sunny.")]
        session = self.mapper.map_to_session(records, "sess-1")

        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1
        span = session.traces[0].spans[0]
        assert isinstance(span, AgentInvocationSpan)
        assert span.user_prompt == "What's the weather?"
        assert span.agent_response == "It's sunny."

    def test_chat_span_creates_inference_span(self):
        """chat operation maps to InferenceSpan with user and assistant messages."""
        records = [make_chat_span(user_input="Hello", assistant_output="Hi")]
        session = self.mapper.map_to_session(records, "sess-1")

        span = session.traces[0].spans[0]
        assert isinstance(span, InferenceSpan)
        assert len(span.messages) == 2  # user + assistant

    def test_execute_tool_span_creates_tool_execution(self):
        """execute_tool operation maps to ToolExecutionSpan with call and result."""
        records = [make_tool_span(tool_name="get_weather", arguments='{"city":"Paris"}', result="Sunny")]
        session = self.mapper.map_to_session(records, "sess-1")

        span = session.traces[0].spans[0]
        assert isinstance(span, ToolExecutionSpan)
        assert span.tool_call.name == "get_weather"
        assert span.tool_call.arguments == {"city": "Paris"}
        assert span.tool_result.content == "Sunny"

    def test_tool_call_arguments_invalid_json_kept_as_empty_dict(self):
        """Non-JSON tool arguments fall back to empty dict (ToolCall.arguments requires dict)."""
        records = [make_tool_span(arguments="not json")]
        session = self.mapper.map_to_session(records, "sess-1")

        span = session.traces[0].spans[0]
        assert span.tool_call.arguments == {}

    def test_chat_span_without_messages_is_skipped(self):
        """chat spans with no input/output messages produce no InferenceSpan."""
        records = [SpanRecord(
            trace_id="t1", span_id="s1", operation_name="chat",
            start_time="2026-01-01T00:00:00Z", end_time="2026-01-01T00:00:01Z",
        )]
        session = self.mapper.map_to_session(records, "sess-1")

        # Trace with no typed spans is filtered out entirely
        assert len(session.traces) == 0

    def test_invoke_agent_without_messages_is_skipped(self):
        """invoke_agent spans with no prompt or response produce no span."""
        records = [SpanRecord(
            trace_id="t1", span_id="s1", operation_name="invoke_agent",
            start_time="2026-01-01T00:00:00Z", end_time="2026-01-01T00:00:01Z",
        )]
        session = self.mapper.map_to_session(records, "sess-1")

        assert len(session.traces) == 0

    def test_chat_span_with_tool_role_message(self):
        """Tool-role messages in chat spans map to UserMessage with ToolResultContent."""
        records = [SpanRecord(
            trace_id="t1", span_id="s1", operation_name="chat",
            start_time="2026-01-01T00:00:00Z", end_time="2026-01-01T00:00:01Z",
            input_messages=[Message(role="tool", content="tool output here")],
            output_messages=[Message(role="assistant", content="Got it")],
        )]
        session = self.mapper.map_to_session(records, "sess-1")

        span = session.traces[0].spans[0]
        assert isinstance(span, InferenceSpan)
        assert len(span.messages) == 2


class TestSessionBuilding:
    def setup_method(self):
        self.mapper = OpenSearchSessionMapper()

    def test_empty_records_returns_empty_session(self):
        session = self.mapper.map_to_session([], "sess-1")
        assert session.session_id == "sess-1"
        assert len(session.traces) == 0

    def test_records_grouped_by_trace_id(self):
        """Records with different trace_ids produce separate Traces."""
        records = [
            make_agent_span(trace_id="t1", span_id="s1"),
            make_agent_span(trace_id="t2", span_id="s2"),
        ]
        session = self.mapper.map_to_session(records, "sess-1")

        assert len(session.traces) == 2
        trace_ids = {t.trace_id for t in session.traces}
        assert trace_ids == {"t1", "t2"}

    def test_spans_sorted_by_start_time(self):
        """Spans within a trace are ordered by start_time."""
        records = [
            make_tool_span(span_id="late", start_time="2026-01-01T00:00:02Z"),
            make_tool_span(span_id="early", start_time="2026-01-01T00:00:01Z"),
        ]
        session = self.mapper.map_to_session(records, "sess-1")

        spans = session.traces[0].spans
        assert spans[0].span_info.span_id == "early"
        assert spans[1].span_info.span_id == "late"

    def test_full_trace_with_all_span_types(self):
        """A realistic trace with agent, chat, and tool spans produces all three types."""
        records = [
            make_agent_span(span_id="agent-1", start_time="2026-01-01T00:00:00Z"),
            make_chat_span(span_id="chat-1", parent_span_id="agent-1", start_time="2026-01-01T00:00:00.1Z"),
            make_tool_span(span_id="tool-1", parent_span_id="agent-1", start_time="2026-01-01T00:00:00.5Z"),
        ]
        session = self.mapper.map_to_session(records, "sess-1")

        spans = session.traces[0].spans
        assert len(spans) == 3
        types = {type(s).__name__ for s in spans}
        assert types == {"InferenceSpan", "ToolExecutionSpan", "AgentInvocationSpan"}


class TestSingleAgentTrace:
    """Single agent with tools (no sub-agents)."""

    def setup_method(self):
        self.mapper = OpenSearchSessionMapper()

    def test_single_agent_with_tools(self):
        """A single invoke_agent span with direct child tool spans."""
        records = [
            make_agent_span(
                span_id="agent", parent_span_id="",
                user_prompt="What's the weather?", agent_response="It's sunny in Paris.",
            ),
            make_tool_span(span_id="t1", parent_span_id="agent", tool_name="get_weather"),
            make_tool_span(span_id="t2", parent_span_id="agent", tool_name="get_forecast"),
        ]
        session = self.mapper.map_to_session(records, "sess-1")

        assert len(session.traces) == 1
        spans = session.traces[0].spans
        assert len(spans) == 3

        agent_span = [s for s in spans if isinstance(s, AgentInvocationSpan)][0]
        assert agent_span.user_prompt == "What's the weather?"
        assert agent_span.agent_response == "It's sunny in Paris."
        assert [t.name for t in agent_span.available_tools] == ["get_forecast", "get_weather"]

    def test_agent_without_tools(self):
        """Agent span with no tool children gets empty available_tools."""
        records = [make_agent_span(span_id="agent", parent_span_id="")]
        session = self.mapper.map_to_session(records, "sess-1")

        agent_span = session.traces[0].spans[0]
        assert isinstance(agent_span, AgentInvocationSpan)
        assert agent_span.available_tools == []


class TestInferenceSpanMapping:
    """Chat/inference span mapping with realistic message patterns."""

    def setup_method(self):
        self.mapper = OpenSearchSessionMapper()

    def test_chat_with_user_and_assistant(self):
        """Standard chat span with user input and assistant output."""
        records = [make_chat_span(user_input="Hello", assistant_output="Hi there")]
        session = self.mapper.map_to_session(records, "sess-1")

        span = session.traces[0].spans[0]
        assert isinstance(span, InferenceSpan)
        assert len(span.messages) == 2
        assert span.messages[0].content[0].text == "Hello"
        assert span.messages[1].content[0].text == "Hi there"

    def test_chat_with_only_assistant_output(self):
        """Chat span with no user input, only assistant output."""
        records = [SpanRecord(
            trace_id="t1", span_id="s1", operation_name="chat",
            start_time="2026-01-01T00:00:00Z", end_time="2026-01-01T00:00:01Z",
            input_messages=[],
            output_messages=[Message(role="assistant", content="Thinking out loud")],
        )]
        session = self.mapper.map_to_session(records, "sess-1")

        span = session.traces[0].spans[0]
        assert isinstance(span, InferenceSpan)
        assert len(span.messages) == 1

    def test_chat_with_multiple_turns(self):
        """Multiple user/assistant messages in a single chat span."""
        records = [SpanRecord(
            trace_id="t1", span_id="s1", operation_name="chat",
            start_time="2026-01-01T00:00:00Z", end_time="2026-01-01T00:00:01Z",
            input_messages=[
                Message(role="user", content="First question"),
                Message(role="user", content="Follow up"),
            ],
            output_messages=[
                Message(role="assistant", content="First answer"),
                Message(role="assistant", content="Second answer"),
            ],
        )]
        session = self.mapper.map_to_session(records, "sess-1")

        span = session.traces[0].spans[0]
        assert isinstance(span, InferenceSpan)
        assert len(span.messages) == 4


class TestAgentWithoutMessages:
    """Edge cases for invoke_agent spans with partial or missing messages."""

    def setup_method(self):
        self.mapper = OpenSearchSessionMapper()

    def test_agent_with_input_only(self):
        """invoke_agent with input but no output still produces a span."""
        records = [SpanRecord(
            trace_id="t1", span_id="s1", operation_name="invoke_agent",
            start_time="2026-01-01T00:00:00Z", end_time="2026-01-01T00:00:01Z",
            input_messages=[Message(role="user", content="Hello")],
            output_messages=[],
        )]
        session = self.mapper.map_to_session(records, "sess-1")

        span = session.traces[0].spans[0]
        assert isinstance(span, AgentInvocationSpan)
        assert span.user_prompt == "Hello"
        assert span.agent_response == ""

    def test_agent_with_output_only(self):
        """invoke_agent with output but no input still produces a span."""
        records = [SpanRecord(
            trace_id="t1", span_id="s1", operation_name="invoke_agent",
            start_time="2026-01-01T00:00:00Z", end_time="2026-01-01T00:00:01Z",
            input_messages=[],
            output_messages=[Message(role="assistant", content="Done")],
        )]
        session = self.mapper.map_to_session(records, "sess-1")

        span = session.traces[0].spans[0]
        assert isinstance(span, AgentInvocationSpan)
        assert span.user_prompt == ""
        assert span.agent_response == "Done"


class TestLargeTrace:
    """Behavior with many spans."""

    def setup_method(self):
        self.mapper = OpenSearchSessionMapper()

    def test_trace_with_many_tool_spans(self):
        """Mapper handles a trace with many tool execution spans."""
        records = [make_agent_span(span_id="agent", parent_span_id="")]
        for i in range(100):
            records.append(make_tool_span(
                span_id=f"tool-{i}", parent_span_id="agent",
                tool_name=f"tool_{i}", start_time=f"2026-01-01T00:00:{i:02d}Z",
            ))
        session = self.mapper.map_to_session(records, "sess-1")

        spans = session.traces[0].spans
        tool_spans = [s for s in spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 100

        agent_span = [s for s in spans if isinstance(s, AgentInvocationSpan)][0]
        assert len(agent_span.available_tools) == 100


class TestMultiAgentTraces:
    """Tests for multi-agent scenarios where tool attribution matters."""

    def setup_method(self):
        self.mapper = OpenSearchSessionMapper()

    def test_tool_names_scoped_to_parent_agent(self):
        """Each AgentInvocationSpan should only list tools that are its children, not all tools in the trace."""
        records = [
            # Orchestrator agent
            make_agent_span(
                span_id="orchestrator", parent_span_id="",
                user_prompt="Plan a trip", agent_response="Here's your plan",
                start_time="2026-01-01T00:00:00Z",
            ),
            # Weather sub-agent (child of orchestrator)
            make_agent_span(
                span_id="weather-agent", parent_span_id="orchestrator",
                user_prompt="Get weather", agent_response="Sunny",
                start_time="2026-01-01T00:00:01Z",
            ),
            # Weather tool (child of weather-agent)
            make_tool_span(
                span_id="weather-tool", parent_span_id="weather-agent",
                tool_name="get_weather",
                start_time="2026-01-01T00:00:01.5Z",
            ),
            # Events sub-agent (child of orchestrator)
            make_agent_span(
                span_id="events-agent", parent_span_id="orchestrator",
                user_prompt="Get events", agent_response="Concert tonight",
                start_time="2026-01-01T00:00:02Z",
            ),
            # Events tool (child of events-agent)
            make_tool_span(
                span_id="events-tool", parent_span_id="events-agent",
                tool_name="get_events",
                start_time="2026-01-01T00:00:02.5Z",
            ),
        ]
        session = self.mapper.map_to_session(records, "sess-1")

        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 3

        # Find each agent by span_id
        by_id = {s.span_info.span_id: s for s in agent_spans}

        # Weather agent should only know about get_weather
        weather_tools = [t.name for t in by_id["weather-agent"].available_tools]
        assert weather_tools == ["get_weather"]

        # Events agent should only know about get_events
        events_tools = [t.name for t in by_id["events-agent"].available_tools]
        assert events_tools == ["get_events"]

        # Orchestrator has no direct tool children
        orchestrator_tools = [t.name for t in by_id["orchestrator"].available_tools]
        assert orchestrator_tools == []


class TestParseTime:
    """Tests for timestamp parsing edge cases."""

    def test_valid_iso_with_z_suffix(self):
        from strands_evals.mappers.opensearch_session_mapper import _parse_time
        dt = _parse_time("2026-01-01T00:00:00Z")
        assert dt.year == 2026
        assert dt.tzinfo is not None

    def test_valid_iso_with_offset(self):
        from strands_evals.mappers.opensearch_session_mapper import _parse_time
        dt = _parse_time("2026-01-01T00:00:00+00:00")
        assert dt.year == 2026

    def test_empty_string_returns_epoch(self):
        from strands_evals.mappers.opensearch_session_mapper import _parse_time
        dt = _parse_time("")
        assert dt.year == 1970

    def test_none_returns_epoch(self):
        from strands_evals.mappers.opensearch_session_mapper import _parse_time
        dt = _parse_time(None)
        assert dt.year == 1970

    def test_invalid_string_returns_epoch(self):
        from strands_evals.mappers.opensearch_session_mapper import _parse_time
        dt = _parse_time("not-a-date")
        assert dt.year == 1970
