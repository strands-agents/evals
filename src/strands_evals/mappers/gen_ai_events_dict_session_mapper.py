"""Mapper for dict spans with gen_ai semantic convention events.

Handles spans sent as plain dicts (not ReadableSpan objects) that contain
gen_ai.user.message, gen_ai.choice, and gen_ai.system.message events.

This format is used by the AgentCore evaluation service when it normalizes
Strands telemetry spans before passing them to evaluator Lambdas.

Known limitation:
    This mapper only extracts AgentInvocationSpan (user input, agent response,
    system prompt). ToolExecutionSpan extraction is not yet supported for this
    format — metrics requiring retrieval_context or tools_called will not have
    that data available when spans arrive in this format.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from ..types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    SpanUnion,
    Trace,
)
from .session_mapper import SessionMapper


class GenAIEventsDictSessionMapper(SessionMapper):
    """Maps dict spans with gen_ai events to Session format.

    Handles spans that have:
    - Standard dict format (not ReadableSpan objects)
    - Strands telemetry scope
    - Events list with gen_ai.user.message, gen_ai.choice, gen_ai.system.message
    - gen_ai.operation.name == "invoke_agent" (agent invocation spans)

    This sits between:
    - CloudWatchSessionMapper (handles body.input/output dicts)
    - StrandsInMemorySessionMapper (handles ReadableSpan objects with gen_ai events)

    Known limitation:
        Only extracts AgentInvocationSpan. ToolExecutionSpan is not supported
        for this format yet.
    """

    def map_to_session(self, data: list[dict[str, Any]], session_id: str) -> Session:
        """Map dict spans with gen_ai events to Session format.

        Args:
            data: List of span dicts with gen_ai events.
            session_id: Session identifier.

        Returns:
            Session object with extracted AgentInvocationSpans.
        """
        spans = self._normalize_to_flat_spans(data) if not isinstance(data, list) else data

        traces_by_id: dict[str, list[dict]] = defaultdict(list)
        for span in spans:
            trace_id = span.get("traceId", "unknown")
            traces_by_id[trace_id].append(span)

        traces = []
        for trace_id, trace_spans in traces_by_id.items():
            agent_spans = self._extract_agent_spans(trace_spans, session_id, trace_id)
            if agent_spans:
                traces.append(Trace(spans=agent_spans, trace_id=trace_id, session_id=session_id))

        return Session(traces=traces, session_id=session_id)

    def _extract_agent_spans(
        self, spans: list[dict[str, Any]], session_id: str, trace_id: str
    ) -> list[SpanUnion]:
        """Extract AgentInvocationSpans from dict spans with gen_ai events."""
        agent_spans: list[SpanUnion] = []

        for span in spans:
            # Only process agent invocation spans
            attrs = span.get("attributes", {})
            if not isinstance(attrs, dict):
                continue
            if attrs.get("gen_ai.operation.name") != "invoke_agent":
                continue

            events = span.get("events", [])
            if not events:
                continue

            user_input = None
            assistant_output = None
            system_prompt = None

            for event in events:
                if not isinstance(event, dict):
                    continue

                event_name = event.get("name", "")
                event_attrs = event.get("attributes", {})
                if not isinstance(event_attrs, dict):
                    continue

                if event_name == "gen_ai.user.message":
                    user_input = self._extract_text_content(event_attrs.get("content", ""))
                elif event_name == "gen_ai.choice":
                    assistant_output = self._extract_text_content(event_attrs.get("message", ""))
                elif event_name == "gen_ai.system.message":
                    system_prompt = self._extract_text_content(event_attrs.get("content", ""))

            if user_input and assistant_output:
                start_ns = self._safe_int(span.get("startTimeUnixNano", 0))
                end_ns = self._safe_int(span.get("endTimeUnixNano", start_ns))

                span_info = SpanInfo(
                    trace_id=trace_id,
                    span_id=span.get("spanId"),
                    session_id=session_id,
                    parent_span_id=span.get("parentSpanId"),
                    start_time=datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc),
                    end_time=datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc),
                )

                agent_spans.append(
                    AgentInvocationSpan(
                        span_info=span_info,
                        user_prompt=user_input,
                        agent_response=assistant_output,
                        available_tools=[],
                        system_prompt=system_prompt,
                    )
                )

        return agent_spans

    @staticmethod
    def _extract_text_content(raw: Any) -> str | None:
        """Extract text from gen_ai event content.

        Content can be:
        - JSON string of text parts: '[{"text": "hello"}, {"text": "world"}]'
        - Already-parsed list of dicts: [{"text": "hello"}]
        - Plain string: "hello world"
        """
        if not raw:
            return None
        # Already parsed (list of dicts)
        if isinstance(raw, list):
            return " ".join(p.get("text", "") for p in raw if isinstance(p, dict)).strip() or None
        if not isinstance(raw, str):
            return str(raw)
        # Try JSON parsing
        try:
            parts = json.loads(raw)
            if isinstance(parts, list):
                return " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip() or None
        except (ValueError, TypeError):
            pass
        return raw

    @staticmethod
    def _safe_int(value: Any) -> int:
        """Safely convert a value to int (handles string-encoded timestamps)."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
