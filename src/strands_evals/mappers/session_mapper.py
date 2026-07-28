"""
SessionMapper - Base class for mapping telemetry data to Session format
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from typing_extensions import Any

from ..types.trace import Session


class SessionMapper(ABC):
    """Base class for mapping telemetry data to Session format for evaluation."""

    @abstractmethod
    def map_to_session(self, data: Any, session_id: str) -> Session:
        """
        Map trace data to Session format.

        Args:
            data: Trace data in various formats:
                - Flat list of spans: [{"trace_id": "x", "span_id": "y", ...}, ...]
                - Grouped by trace_id: {"trace_1": [spans], "trace_2": [spans]}
                - List of trace objects: [{"trace_id": "x", "spans": [...]}, ...]
            session_id: Session identifier

        Returns:
            Session object ready for evaluation
        """
        pass

    def _normalize_to_flat_spans(self, data: Any) -> list[dict]:
        """Normalize various input formats to a flat list of spans.

        Accepts:
        - Flat list of spans: [{"trace_id": "x", "span_id": "y", ...}, ...]
        - Grouped by trace_id: {"trace_1": [spans], "trace_2": [spans]}
        - List of trace objects: [{"trace_id": "x", "spans": [...]}, ...]

        Returns:
            Flat list of span dictionaries
        """
        if not data:
            return []

        # Case 1: Grouped by trace_id: {"trace_1": [spans], "trace_2": [spans]}
        if isinstance(data, dict):
            result: list[dict] = []
            for value in data.values():
                if isinstance(value, list):
                    result.extend(span for span in value if isinstance(span, dict))
            return result

        # Case 2: List input
        if isinstance(data, list):
            # Check if it's a list of trace objects with "spans" key
            # by looking for any element with "spans"
            has_spans_key = any(isinstance(item, dict) and "spans" in item for item in data)

            if has_spans_key:
                # List of trace objects: [{"trace_id": "x", "spans": [...]}, ...]
                result = []
                for trace in data:
                    if isinstance(trace, dict):
                        spans = trace.get("spans", [])
                        if isinstance(spans, list):
                            result.extend(span for span in spans if isinstance(span, dict))
                return result
            else:
                # Flat list of spans: [{"trace_id": "x", "span_id": "y", ...}, ...]
                return [item for item in data if isinstance(item, dict)]

        # Fallback for unexpected types
        return []

    def parse_timestamp(self, value: Any) -> datetime:
        """Parse timestamp from various formats.

        Handles:
        - None → current UTC time
        - datetime → passthrough
        - ISO 8601 string (with optional trailing Z) → parsed datetime
        - Numeric (int/float) nanosecond epoch → datetime
        - String-encoded nanosecond epoch → datetime

        Args:
            value: Raw timestamp value from a span dict.

        Returns:
            Timezone-aware datetime in UTC.
        """
        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            if value.isdigit():
                return datetime.fromtimestamp(int(value) / 1e9, tz=timezone.utc)
            try:
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime.now(timezone.utc)
        if isinstance(value, (int, float)):
            # Handle nanoseconds
            if value > 1e12:
                value = value / 1e9
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return datetime.now(timezone.utc)
