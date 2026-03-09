"""Tests for mapper utility functions."""

from strands_evals.mappers import (
    CloudWatchSessionMapper,
    LangChainOtelSessionMapper,
    OpenInferenceSessionMapper,
    StrandsInMemorySessionMapper,
    detect_otel_mapper,
    get_scope_name,
    readable_spans_to_dicts,
)


def make_span_dict(scope_name="test.scope", attributes=None, span_events=None):
    """Build a minimal span dict for testing."""
    return {
        "trace_id": "trace-1",
        "span_id": "span-1",
        "scope": {"name": scope_name, "version": "0.1.0"},
        "attributes": attributes or {},
        "span_events": span_events or [],
    }


class TestGetScopeName:
    def test_extracts_from_scope_dict(self):
        """Extracts scope name from scope.name field."""
        span = make_span_dict(scope_name="test.scope")
        assert get_scope_name(span) == "test.scope"

    def test_extracts_from_attributes_event_name(self):
        """Falls back to attributes.event.name if scope.name is empty."""
        span = {
            "scope": {"name": ""},
            "attributes": {"event.name": "fallback.scope"},
        }
        assert get_scope_name(span) == "fallback.scope"

    def test_returns_empty_for_missing_scope(self):
        """Returns empty string if no scope found."""
        span = {"attributes": {}}
        assert get_scope_name(span) == ""

    def test_handles_non_dict_input(self):
        """Returns empty string for non-dict input without scope."""

        class FakeSpan:
            pass

        span = FakeSpan()
        assert get_scope_name(span) == ""


class TestDetectOtelMapper:
    def test_empty_spans_returns_strands_mapper(self):
        """Empty spans list returns StrandsInMemorySessionMapper."""
        mapper = detect_otel_mapper([])
        assert isinstance(mapper, StrandsInMemorySessionMapper)

    def test_detects_langchain_traceloop_scope(self):
        """Detects LangChainOtelSessionMapper for traceloop scope."""
        spans = [make_span_dict(scope_name="opentelemetry.instrumentation.langchain")]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, LangChainOtelSessionMapper)

    def test_detects_openinference_scope(self):
        """Detects OpenInferenceSessionMapper for openinference scope."""
        spans = [make_span_dict(scope_name="openinference.instrumentation.langchain")]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, OpenInferenceSessionMapper)

    def test_detects_strands_cloudwatch_format(self):
        """Detects CloudWatchSessionMapper for strands scope with body format."""
        spans = [
            make_span_dict(
                scope_name="strands.telemetry.tracer",
                span_events=[
                    {
                        "body": {
                            "input": {"messages": []},
                            "output": {"messages": []},
                        }
                    }
                ],
            )
        ]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, CloudWatchSessionMapper)

    def test_detects_strands_in_memory_format(self):
        """Detects StrandsInMemorySessionMapper for strands scope without body format."""
        spans = [
            make_span_dict(
                scope_name="strands.telemetry.tracer",
                attributes={"gen_ai.operation.name": "chat"},
            )
        ]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, StrandsInMemorySessionMapper)

    def test_defaults_to_strands_mapper_for_unknown_scope(self):
        """Defaults to StrandsInMemorySessionMapper for unknown scope."""
        spans = [make_span_dict(scope_name="unknown.scope")]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, StrandsInMemorySessionMapper)

    def test_skips_spans_without_scope(self):
        """Skips spans without scope and continues detection."""
        spans = [
            {"trace_id": "t1", "span_id": "s1"},  # No scope
            make_span_dict(scope_name="openinference.instrumentation.langchain"),
        ]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, OpenInferenceSessionMapper)


class TestReadableSpansToDicts:
    def test_converts_readable_spans(self):
        """Converts ReadableSpan objects to dict format."""
        from unittest.mock import MagicMock

        # Create mock ReadableSpan
        mock_span = MagicMock()
        mock_span.context.trace_id = 0xABCD1234
        mock_span.context.span_id = 0x5678
        mock_span.parent = None
        mock_span.name = "test-span"
        mock_span.start_time = 1700000000000000000
        mock_span.end_time = 1700000001000000000
        mock_span.attributes = {"key": "value"}
        mock_span.instrumentation_scope.name = "test.scope"
        mock_span.instrumentation_scope.version = "1.0"
        mock_span.status.status_code.name = "OK"
        mock_span.events = []

        result = readable_spans_to_dicts([mock_span])

        assert len(result) == 1
        assert result[0]["trace_id"] == "000000000000000000000000abcd1234"
        assert result[0]["span_id"] == "0000000000005678"
        assert result[0]["parent_span_id"] is None
        assert result[0]["name"] == "test-span"
        assert result[0]["attributes"] == {"key": "value"}
        assert result[0]["scope"]["name"] == "test.scope"
        assert result[0]["span_events"] == []

    def test_converts_span_with_parent(self):
        """Converts span with parent reference."""
        from unittest.mock import MagicMock

        mock_span = MagicMock()
        mock_span.context.trace_id = 0xABCD
        mock_span.context.span_id = 0x1234
        mock_span.parent.span_id = 0x5678
        mock_span.name = "child-span"
        mock_span.start_time = 0
        mock_span.end_time = 0
        mock_span.attributes = {}
        mock_span.instrumentation_scope.name = "test"
        mock_span.instrumentation_scope.version = "1.0"
        mock_span.status.status_code.name = "OK"
        mock_span.events = []

        result = readable_spans_to_dicts([mock_span])

        assert result[0]["parent_span_id"] == "0000000000005678"

    def test_converts_span_events(self):
        """Converts span events to dict format."""
        from unittest.mock import MagicMock

        mock_event = MagicMock()
        mock_event.name = "test.event"
        mock_event.timestamp = 1700000000500000000
        mock_event.attributes = {"event_key": "event_value"}

        mock_span = MagicMock()
        mock_span.context.trace_id = 0xABCD
        mock_span.context.span_id = 0x1234
        mock_span.parent = None
        mock_span.name = "span"
        mock_span.start_time = 0
        mock_span.end_time = 0
        mock_span.attributes = {}
        mock_span.instrumentation_scope.name = "test"
        mock_span.instrumentation_scope.version = "1.0"
        mock_span.status.status_code.name = "OK"
        mock_span.events = [mock_event]

        result = readable_spans_to_dicts([mock_span])

        assert len(result[0]["span_events"]) == 1
        assert result[0]["span_events"][0]["event_name"] == "test.event"
        assert result[0]["span_events"][0]["attributes"] == {"event_key": "event_value"}

    def test_handles_empty_list(self):
        """Handles empty span list."""
        result = readable_spans_to_dicts([])
        assert result == []
