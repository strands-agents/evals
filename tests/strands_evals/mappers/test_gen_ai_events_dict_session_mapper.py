"""Tests for GenAIEventsDictSessionMapper."""

from strands_evals.mappers import (
    CloudWatchSessionMapper,
    GenAIEventsDictSessionMapper,
    StrandsInMemorySessionMapper,
    detect_otel_mapper,
)
from strands_evals.types.trace import AgentInvocationSpan


def _make_gen_ai_span(
    trace_id="t1",
    span_id="s1",
    user_message="What is AI?",
    assistant_message="AI is artificial intelligence.",
    system_message=None,
):
    """Create a dict span with gen_ai events."""
    events = [
        {"name": "gen_ai.user.message", "attributes": {"content": f'[{{"text": "{user_message}"}}]'}},
        {"name": "gen_ai.choice", "attributes": {"message": f'[{{"text": "{assistant_message}"}}]'}},
    ]
    if system_message:
        system_content = f'[{{"text": "{system_message}"}}]'
        events.insert(0, {"name": "gen_ai.system.message", "attributes": {"content": system_content}})

    return {
        "traceId": trace_id,
        "spanId": span_id,
        "scope": {"name": "strands.telemetry.tracer"},
        "name": "invoke_agent",
        "kind": "INTERNAL",
        "startTimeUnixNano": 1000000000,
        "endTimeUnixNano": 2000000000,
        "attributes": {"gen_ai.operation.name": "invoke_agent", "session.id": "test"},
        "events": events,
    }


class TestDetection:
    """Tests for detect_otel_mapper returning GenAIEventsDictSessionMapper."""

    def test_returns_gen_ai_mapper_for_dict_spans_with_gen_ai_events(self):
        spans = [_make_gen_ai_span()]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, GenAIEventsDictSessionMapper)

    def test_returns_cloudwatch_mapper_for_body_format(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "body": {
                    "input": {"messages": [{"role": "user", "content": "hi"}]},
                    "output": {"messages": [{"role": "assistant", "content": "hello"}]},
                },
            }
        ]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, CloudWatchSessionMapper)

    def test_does_not_match_non_strands_scope(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "unknown.framework"},
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
                "events": [{"name": "gen_ai.user.message", "attributes": {"content": "hi"}}],
            }
        ]
        mapper = detect_otel_mapper(spans)
        assert not isinstance(mapper, GenAIEventsDictSessionMapper)

    def test_strands_dict_spans_without_events_fall_to_in_memory(self):
        """Strands dict spans without events key fall to StrandsInMemorySessionMapper."""
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
            }
        ]
        mapper = detect_otel_mapper(spans)
        # No events key → falls through to default StrandsInMemorySessionMapper
        assert isinstance(mapper, StrandsInMemorySessionMapper)

    def test_strands_dict_spans_with_events_use_gen_ai_mapper(self):
        """Strands dict spans with events (even unknown ones) use GenAIEventsDictSessionMapper."""
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
                "events": [{"name": "some.unknown.event", "attributes": {}}],
            }
        ]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, GenAIEventsDictSessionMapper)


class TestMapToSession:
    """Tests for GenAIEventsDictSessionMapper.map_to_session."""

    def test_extracts_user_prompt_and_agent_response(self):
        spans = [_make_gen_ai_span(user_message="Hello", assistant_message="Hi there")]
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session(spans, session_id="test")

        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1
        span = session.traces[0].spans[0]
        assert isinstance(span, AgentInvocationSpan)
        assert span.user_prompt == "Hello"
        assert span.agent_response == "Hi there"

    def test_extracts_system_prompt(self):
        spans = [_make_gen_ai_span(system_message="You are helpful")]
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session(spans, session_id="test")

        span = session.traces[0].spans[0]
        assert span.system_prompt == "You are helpful"

    def test_no_system_prompt_returns_none(self):
        spans = [_make_gen_ai_span()]
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session(spans, session_id="test")

        span = session.traces[0].spans[0]
        assert span.system_prompt is None

    def test_handles_multiple_traces(self):
        spans = [
            _make_gen_ai_span(trace_id="t1", span_id="s1", user_message="Q1", assistant_message="A1"),
            _make_gen_ai_span(trace_id="t2", span_id="s2", user_message="Q2", assistant_message="A2"),
        ]
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session(spans, session_id="test")

        assert len(session.traces) == 2

    def test_handles_multiple_spans_same_trace(self):
        spans = [
            _make_gen_ai_span(trace_id="t1", span_id="s1", user_message="Q1", assistant_message="A1"),
            _make_gen_ai_span(trace_id="t1", span_id="s2", user_message="Q2", assistant_message="A2"),
        ]
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session(spans, session_id="test")

        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 2

    def test_skips_non_agent_spans(self):
        """Only spans with gen_ai.operation.name == 'invoke_agent' are extracted."""
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "startTimeUnixNano": 1000000000,
                "endTimeUnixNano": 2000000000,
                "attributes": {"gen_ai.operation.name": "chat"},
                "events": [
                    {"name": "gen_ai.user.message", "attributes": {"content": '[{"text": "hi"}]'}},
                    {"name": "gen_ai.choice", "attributes": {"message": '[{"text": "hello"}]'}},
                ],
            },
            _make_gen_ai_span(trace_id="t1", span_id="s2"),
        ]
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session(spans, session_id="test")

        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1

    def test_skips_spans_without_events(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
            },
            _make_gen_ai_span(trace_id="t1", span_id="s2"),
        ]
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session(spans, session_id="test")

        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1

    def test_skips_spans_missing_user_input(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "startTimeUnixNano": 1000000000,
                "endTimeUnixNano": 2000000000,
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
                "events": [
                    {"name": "gen_ai.choice", "attributes": {"message": '[{"text": "response"}]'}},
                ],
            }
        ]
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session(spans, session_id="test")

        assert len(session.traces) == 0

    def test_handles_plain_string_content(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "startTimeUnixNano": 1000000000,
                "endTimeUnixNano": 2000000000,
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
                "events": [
                    {"name": "gen_ai.user.message", "attributes": {"content": "plain text"}},
                    {"name": "gen_ai.choice", "attributes": {"message": "plain response"}},
                ],
            }
        ]
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session(spans, session_id="test")

        span = session.traces[0].spans[0]
        assert span.user_prompt == "plain text"
        assert span.agent_response == "plain response"

    def test_handles_already_parsed_list_content(self):
        """Content may arrive as already-parsed list (not JSON string)."""
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "startTimeUnixNano": 1000000000,
                "endTimeUnixNano": 2000000000,
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
                "events": [
                    {"name": "gen_ai.user.message", "attributes": {"content": [{"text": "parsed input"}]}},
                    {"name": "gen_ai.choice", "attributes": {"message": [{"text": "parsed output"}]}},
                ],
            }
        ]
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session(spans, session_id="test")

        span = session.traces[0].spans[0]
        assert span.user_prompt == "parsed input"
        assert span.agent_response == "parsed output"

    def test_handles_string_encoded_timestamps(self):
        """OTLP JSON may encode timestamps as strings."""
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "startTimeUnixNano": "1000000000",
                "endTimeUnixNano": "2000000000",
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
                "events": [
                    {"name": "gen_ai.user.message", "attributes": {"content": "hi"}},
                    {"name": "gen_ai.choice", "attributes": {"message": "hello"}},
                ],
            }
        ]
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session(spans, session_id="test")

        assert len(session.traces) == 1

    def test_empty_spans_returns_empty_session(self):
        mapper = GenAIEventsDictSessionMapper()
        session = mapper.map_to_session([], session_id="test")

        assert len(session.traces) == 0
        assert session.session_id == "test"
