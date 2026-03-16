"""Parser for normalizing CloudWatch/ADOT logs to standard span format.

This parser converts raw CloudWatch logs (which may contain spans, events, or both)
into the normalized format expected by mappers - matching the output of
`readable_spans_to_dicts()`.

CloudWatch logs can have:
1. SPAN records (with startTimeUnixNano, endTimeUnixNano) + EVENT records
2. Only EVENT records (common for Strands telemetry)

The parser handles both cases, creating synthetic spans from events when needed.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "default_session"


class CloudWatchLogsParser:
    """Normalizes CloudWatch/ADOT logs to standard span dict format.

    Converts raw CloudWatch logs to match the output format of `readable_spans_to_dicts()`:
    - snake_case field names (trace_id, span_id, parent_span_id)
    - Events attached to spans as span_events[]
    - Consistent structure regardless of input format

    Example:
        >>> raw_logs = cloudwatch_provider.fetch_logs()
        >>> parser = CloudWatchLogsParser(raw_logs)
        >>> normalized = parser.parse()
        >>> mapper = detect_otel_mapper(normalized)
        >>> session = mapper.map_to_session(normalized, session_id)
    """

    def __init__(self, raw_logs: list[dict]):
        """Initialize parser with raw CloudWatch logs.

        Args:
            raw_logs: List of raw JSON records from CloudWatch Logs Insights
        """
        self.raw_logs = raw_logs

    def parse(self) -> list[dict]:
        """Parse CloudWatch logs to normalized span format.

        Returns:
            List of normalized span dicts matching readable_spans_to_dicts() format
        """
        if not self.raw_logs:
            return []

        # Separate spans from events
        spans_by_id: dict[str, dict] = {}
        events_by_span_id: dict[str, list[dict]] = defaultdict(list)

        for record in self.raw_logs:
            if self._is_event(record):
                event = self._normalize_event(record)
                if event:
                    span_id = event.get("span_id", "")
                    events_by_span_id[span_id].append(event)
            elif self._is_span(record):
                span = self._normalize_span(record)
                if span:
                    spans_by_id[span["span_id"]] = span
            # Skip other log records (application logs, etc.)

        # If we have spans, associate events with them
        if spans_by_id:
            for span_id, span in spans_by_id.items():
                span["span_events"] = events_by_span_id.get(span_id, [])
            return list(spans_by_id.values())

        # No spans - create synthetic spans from events
        # Group events by span_id to create one span per event group
        result = []
        for span_id, events in events_by_span_id.items():
            if not events:
                continue
            # Use first event to create synthetic span
            first_event = events[0]
            synthetic_span = self._create_synthetic_span(span_id, events, first_event)
            result.append(synthetic_span)

        return result

    def _is_event(self, record: dict) -> bool:
        """Check if record is an OTEL event (has event.name attribute)."""
        if not isinstance(record, dict):
            return False

        # Check various event name locations
        if "EventName" in record or "eventName" in record:
            return True

        attrs = record.get("attributes", {})
        if isinstance(attrs, dict) and "event.name" in attrs:
            return True

        return False

    def _is_span(self, record: dict) -> bool:
        """Check if record is an OTEL span (has start/end time)."""
        if not isinstance(record, dict):
            return False

        # Spans have startTimeUnixNano or start_time
        has_start = "startTimeUnixNano" in record or "start_time" in record
        has_end = "endTimeUnixNano" in record or "end_time" in record

        return has_start and has_end

    def _normalize_event(self, record: dict) -> dict | None:
        """Normalize a CloudWatch event record."""
        try:
            # Extract event name from various locations
            event_name = (
                record.get("EventName") or record.get("eventName") or record.get("attributes", {}).get("event.name", "")
            )

            # Extract span_id (camelCase in CloudWatch)
            span_id = record.get("spanId") or record.get("span_id", "")

            # Extract timestamp
            timestamp = (
                record.get("timeUnixNano") or record.get("time_unix_nano") or record.get("observedTimeUnixNano") or 0
            )

            return {
                "event_name": event_name,
                "span_id": span_id,
                "timestamp": timestamp,
                "attributes": record.get("attributes", {}),
                "body": record.get("body", {}),
            }
        except Exception as e:
            logger.warning(f"Failed to normalize event: {e}")
            return None

    def _normalize_span(self, record: dict) -> dict | None:
        """Normalize a CloudWatch span record to standard format."""
        try:
            # Normalize field names (camelCase → snake_case)
            trace_id = record.get("traceId") or record.get("trace_id", "")
            span_id = record.get("spanId") or record.get("span_id", "")
            parent_span_id = record.get("parentSpanId") or record.get("parent_span_id")

            # Normalize timestamps
            start_time = record.get("startTimeUnixNano") or record.get("start_time", 0)
            end_time = record.get("endTimeUnixNano") or record.get("end_time", 0)

            # Extract scope
            scope = record.get("scope", {})
            if not isinstance(scope, dict):
                scope = {"name": "", "version": ""}

            # Extract status
            status = record.get("status", {})
            if not isinstance(status, dict):
                status = {"code": "UNSET"}

            return {
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "name": record.get("name", ""),
                "start_time": start_time,
                "end_time": end_time,
                "attributes": record.get("attributes", {}),
                "scope": {
                    "name": scope.get("name", ""),
                    "version": scope.get("version", ""),
                },
                "status": {"code": status.get("code", "UNSET")},
                "span_events": [],  # Will be populated later
            }
        except Exception as e:
            logger.warning(f"Failed to normalize span: {e}")
            return None

    def _create_synthetic_span(self, span_id: str, events: list[dict], first_event: dict) -> dict:
        """Create a synthetic span from events when no span records exist."""
        # Get trace_id and scope from the raw logs
        trace_id = ""
        scope_name = ""
        for record in self.raw_logs:
            if record.get("spanId") == span_id or record.get("span_id") == span_id:
                trace_id = trace_id or record.get("traceId") or record.get("trace_id", "")
                # Prefer scope.name from raw record if available
                raw_scope = record.get("scope", {})
                if isinstance(raw_scope, dict) and raw_scope.get("name"):
                    scope_name = raw_scope["name"]

        # Fall back to event name as scope hint
        if not scope_name:
            event_name = first_event.get("event_name", "")
            scope_name = first_event.get("attributes", {}).get("event.name", event_name)

        # Use event timestamp for span times
        event_name = first_event.get("event_name", "")
        timestamp = first_event.get("timestamp", 0)

        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": None,
            "name": event_name,
            "start_time": timestamp,
            "end_time": timestamp,
            "attributes": first_event.get("attributes", {}),
            "scope": {
                "name": scope_name,
                "version": "",
            },
            "status": {"code": "OK"},
            "span_events": events,
        }


def parse_cloudwatch_logs(raw_logs: list[dict]) -> list[dict]:
    """Convenience function to parse CloudWatch logs.

    Args:
        raw_logs: Raw CloudWatch log records

    Returns:
        Normalized span dicts matching readable_spans_to_dicts() format
    """
    return CloudWatchLogsParser(raw_logs).parse()
