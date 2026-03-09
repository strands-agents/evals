"""Tests for CloudWatchLogsParser - raw CloudWatch logs → normalized span format."""

from strands_evals.mappers import CloudWatchLogsParser, parse_cloudwatch_logs


def make_span_record(
    trace_id="trace-1",
    span_id="span-1",
    parent_span_id=None,
    name="test-span",
    start_time=1700000000000000000,
    end_time=1700000001000000000,
    attributes=None,
    scope_name="strands.telemetry.tracer",
):
    """Build a raw CloudWatch SPAN record (camelCase format)."""
    record = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": start_time,
        "endTimeUnixNano": end_time,
        "attributes": attributes or {},
        "scope": {"name": scope_name, "version": "0.1.0"},
        "status": {"code": "OK"},
    }
    if parent_span_id:
        record["parentSpanId"] = parent_span_id
    return record


def make_event_record(
    span_id="span-1",
    event_name="strands.telemetry.tracer",
    time_nano=1700000000500000000,
    attributes=None,
    body=None,
):
    """Build a raw CloudWatch EVENT record."""
    return {
        "spanId": span_id,
        "EventName": event_name,
        "timeUnixNano": time_nano,
        "attributes": attributes or {},
        "body": body or {},
    }


class TestRecordTypeDetection:
    def setup_method(self):
        self.parser = CloudWatchLogsParser([])

    def test_span_record_detected(self):
        """Record with start/end time is detected as span."""
        record = make_span_record()
        assert self.parser._is_span(record) is True
        assert self.parser._is_event(record) is False

    def test_event_record_detected_by_event_name(self):
        """Record with EventName is detected as event."""
        record = make_event_record()
        assert self.parser._is_event(record) is True
        assert self.parser._is_span(record) is False

    def test_event_record_detected_by_attributes_event_name(self):
        """Record with attributes.event.name is detected as event."""
        record = {
            "spanId": "span-1",
            "attributes": {"event.name": "test.event"},
        }
        assert self.parser._is_event(record) is True

    def test_invalid_record_not_detected(self):
        """Invalid records are not detected as span or event."""
        record = {"foo": "bar"}
        assert self.parser._is_span(record) is False
        assert self.parser._is_event(record) is False


class TestSpanNormalization:
    def setup_method(self):
        self.parser = CloudWatchLogsParser([])

    def test_camel_case_to_snake_case(self):
        """camelCase field names are normalized to snake_case."""
        record = make_span_record(
            trace_id="t1",
            span_id="s1",
            parent_span_id="p1",
        )
        normalized = self.parser._normalize_span(record)

        assert normalized["trace_id"] == "t1"
        assert normalized["span_id"] == "s1"
        assert normalized["parent_span_id"] == "p1"

    def test_timestamps_preserved(self):
        """Timestamps are preserved in normalized output."""
        record = make_span_record(
            start_time=1700000000000000000,
            end_time=1700000001000000000,
        )
        normalized = self.parser._normalize_span(record)

        assert normalized["start_time"] == 1700000000000000000
        assert normalized["end_time"] == 1700000001000000000

    def test_scope_normalized(self):
        """Scope is normalized to dict format."""
        record = make_span_record(scope_name="test.scope")
        normalized = self.parser._normalize_span(record)

        assert normalized["scope"]["name"] == "test.scope"

    def test_span_events_initialized_empty(self):
        """Normalized span has empty span_events list."""
        record = make_span_record()
        normalized = self.parser._normalize_span(record)

        assert normalized["span_events"] == []


class TestEventNormalization:
    def setup_method(self):
        self.parser = CloudWatchLogsParser([])

    def test_event_name_extracted(self):
        """Event name is extracted from EventName field."""
        record = make_event_record(event_name="test.event")
        normalized = self.parser._normalize_event(record)

        assert normalized["event_name"] == "test.event"

    def test_span_id_extracted(self):
        """Span ID is extracted from spanId field."""
        record = make_event_record(span_id="s1")
        normalized = self.parser._normalize_event(record)

        assert normalized["span_id"] == "s1"

    def test_body_preserved(self):
        """Body is preserved in normalized event."""
        body = {"input": {"messages": []}, "output": {"messages": []}}
        record = make_event_record(body=body)
        normalized = self.parser._normalize_event(record)

        assert normalized["body"] == body


class TestParserIntegration:
    def test_spans_with_events_associated(self):
        """Events are associated with their parent spans."""
        span = make_span_record(trace_id="t1", span_id="s1")
        event = make_event_record(span_id="s1", body={"data": "test"})

        parser = CloudWatchLogsParser([span, event])
        result = parser.parse()

        assert len(result) == 1
        assert result[0]["span_id"] == "s1"
        assert len(result[0]["span_events"]) == 1
        assert result[0]["span_events"][0]["body"] == {"data": "test"}

    def test_multiple_events_per_span(self):
        """Multiple events are associated with same span."""
        span = make_span_record(trace_id="t1", span_id="s1")
        event1 = make_event_record(span_id="s1", time_nano=1000)
        event2 = make_event_record(span_id="s1", time_nano=2000)

        parser = CloudWatchLogsParser([span, event1, event2])
        result = parser.parse()

        assert len(result) == 1
        assert len(result[0]["span_events"]) == 2

    def test_synthetic_span_from_orphan_events(self):
        """Synthetic span is created when only events exist (no span records)."""
        event = make_event_record(
            span_id="s1",
            event_name="test.event",
            body={"input": {"messages": []}, "output": {"messages": []}},
        )

        parser = CloudWatchLogsParser([event])
        result = parser.parse()

        assert len(result) == 1
        assert result[0]["span_id"] == "s1"
        assert len(result[0]["span_events"]) == 1

    def test_empty_logs(self):
        """Empty logs return empty result."""
        parser = CloudWatchLogsParser([])
        result = parser.parse()

        assert result == []

    def test_multiple_spans_multiple_traces(self):
        """Multiple spans from different traces are all included."""
        span1 = make_span_record(trace_id="t1", span_id="s1")
        span2 = make_span_record(trace_id="t2", span_id="s2")

        parser = CloudWatchLogsParser([span1, span2])
        result = parser.parse()

        assert len(result) == 2
        span_ids = {s["span_id"] for s in result}
        assert span_ids == {"s1", "s2"}


class TestConvenienceFunction:
    def test_parse_cloudwatch_logs_function(self):
        """Convenience function works same as class."""
        span = make_span_record(trace_id="t1", span_id="s1")
        event = make_event_record(span_id="s1")

        result = parse_cloudwatch_logs([span, event])

        assert len(result) == 1
        assert result[0]["span_id"] == "s1"
        assert len(result[0]["span_events"]) == 1
