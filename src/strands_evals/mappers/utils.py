"""
Utility functions for mapper selection and detection.
"""

from typing import Any

from .constants import SCOPE_LANGCHAIN_OTEL, SCOPE_OPENINFERENCE, SCOPE_STRANDS
from .session_mapper import SessionMapper


def detect_otel_mapper(spans: list[Any]) -> SessionMapper:
    """Detect the appropriate mapper based on span scope and data format.

    Examines spans to determine:
    1. Which framework produced the traces (via scope.name)
    2. What data format is used (CloudWatch body vs gen_ai.* attributes)

    For Strands telemetry, auto-detects format:
    - CloudWatch format: span_events[].body.input/output OR body directly
    - InMemory format: gen_ai.* attributes

    Args:
        spans: List of span dictionaries (normalized or raw)

    Returns:
        Appropriate SessionMapper instance

    Example:
        >>> # After CloudWatchLogsParser normalization
        >>> normalized = CloudWatchLogsParser(raw_logs).parse()
        >>> mapper = detect_otel_mapper(normalized)
        >>> session = mapper.map_to_session(normalized, "session-123")

        >>> # After readable_spans_to_dicts
        >>> spans = readable_spans_to_dicts(exporter.get_finished_spans())
        >>> mapper = detect_otel_mapper(spans)
        >>> session = mapper.map_to_session(spans, "session-123")
    """
    # Import here to avoid circular imports
    from .cloudwatch_session_mapper import CloudWatchSessionMapper
    from .langchain_otel_session_mapper import LangChainOtelSessionMapper
    from .openinference_session_mapper import OpenInferenceSessionMapper
    from .strands_in_memory_session_mapper import StrandsInMemorySessionMapper

    if not spans:
        return StrandsInMemorySessionMapper()

    # Detect scope and format from first relevant span
    for span in spans:
        scope_name = get_scope_name(span)

        if scope_name == SCOPE_LANGCHAIN_OTEL:
            return LangChainOtelSessionMapper()

        if scope_name == SCOPE_OPENINFERENCE:
            return OpenInferenceSessionMapper()

        if scope_name == SCOPE_STRANDS:
            # Auto-detect format for Strands
            if get_body(span) is not None:
                return CloudWatchSessionMapper()
            else:
                return StrandsInMemorySessionMapper()

    # Fallback: check if spans use the CloudWatch body format (no scope.name
    # but have body.input/output structure). This handles raw CloudWatch
    # log records that haven't been normalized by CloudWatchLogsParser.
    for span in spans:
        if get_body(span) is not None:
            return CloudWatchSessionMapper()

    # Default to StrandsInMemorySessionMapper
    return StrandsInMemorySessionMapper()


def get_body(span: dict) -> dict | None:
    """Extract body from span, handling both normalized and raw formats.

    Normalized format: span_events[].body
    Raw format: span.body directly
    """
    # Try normalized format first (span_events[].body)
    span_events = span.get("span_events", [])
    for event in span_events:
        body = event.get("body")
        if isinstance(body, dict) and ("input" in body or "output" in body):
            return body

    # Fall back to raw format (body directly on span/record)
    body = span.get("body")
    if isinstance(body, dict):
        return body

    return None


def get_scope_name(span: Any) -> str:
    """Extract the instrumentation scope name from a span or event.

    Handles:
    - Dict-based spans with scope.name
    - Dict-based events with attributes.event.name (CloudWatch format)
    - ReadableSpan objects with instrumentation_scope

    Args:
        span: A span/event dictionary or ReadableSpan object

    Returns:
        The scope name, or empty string if not found
    """
    if isinstance(span, dict):
        # Try scope.name first (standard OTEL span format)
        scope = span.get("scope", {})
        if isinstance(scope, dict):
            scope_name = scope.get("name", "")
            if scope_name:
                return scope_name

        # Try attributes.event.name (CloudWatch event format)
        attrs = span.get("attributes", {})
        if isinstance(attrs, dict):
            event_name = attrs.get("event.name", "")
            if event_name:
                return event_name
    else:
        # Handle ReadableSpan objects
        if hasattr(span, "instrumentation_scope"):
            return getattr(span.instrumentation_scope, "name", "")

    return ""


def readable_spans_to_dicts(spans: Any) -> list[dict]:
    """Convert OpenTelemetry ReadableSpan objects to dict format.

    This utility converts spans from the OpenTelemetry SDK's InMemorySpanExporter
    (which returns ReadableSpan objects) to the dict format expected by mappers.

    Args:
        spans: Iterable of ReadableSpan objects from InMemorySpanExporter

    Returns:
        List of span dictionaries ready for use with mappers

    Example:
        >>> from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        >>> exporter = InMemorySpanExporter()
        >>> # ... run instrumented code ...
        >>> spans = readable_spans_to_dicts(exporter.get_finished_spans())
        >>> mapper = detect_otel_mapper(spans)
        >>> session = mapper.map_to_session(spans, "session-123")
    """
    result = []
    for span in spans:
        span_dict = {
            "trace_id": format(span.context.trace_id, "032x"),
            "span_id": format(span.context.span_id, "016x"),
            "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
            "name": span.name,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "attributes": dict(span.attributes) if span.attributes else {},
            "scope": {
                "name": span.instrumentation_scope.name if span.instrumentation_scope else "",
                "version": span.instrumentation_scope.version if span.instrumentation_scope else "",
            },
            "status": {"code": span.status.status_code.name if span.status else "UNSET"},
            "span_events": [],
        }

        # Convert events
        if span.events:
            for event in span.events:
                event_dict = {
                    "event_name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": dict(event.attributes) if event.attributes else {},
                }
                span_dict["span_events"].append(event_dict)

        result.append(span_dict)
    return result
