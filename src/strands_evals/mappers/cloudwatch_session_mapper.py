"""CloudWatch session mapper - converts normalized CloudWatch spans to Session format.

This mapper handles Strands telemetry data that has been normalized by CloudWatchLogsParser.
It expects spans in the normalized format with span_events[].body containing input/output messages.

.. deprecated::
    CloudWatchSessionMapper is deprecated. Use ``CloudWatchProvider`` with auto-detection
    (via ``detect_otel_mapper()``) instead, which automatically selects the correct mapper
    based on the framework that produced the traces.
"""

import json
import logging
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from ..mappers.session_mapper import SessionMapper
from ..mappers.utils import get_body
from ..types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolCallContent,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    ToolResultContent,
    Trace,
    UserMessage,
)

logger = logging.getLogger(__name__)


class CloudWatchSessionMapper(SessionMapper):
    """Maps normalized CloudWatch spans to Session format.

    .. deprecated::
        Use ``CloudWatchProvider`` with auto-detection instead. ``CloudWatchProvider``
        now uses ``detect_otel_mapper()`` to automatically select the correct mapper
        based on the span ``scope.name``, supporting Strands, LangChain, and other
        frameworks stored in CloudWatch.

    This mapper handles Strands telemetry data. It expects spans in the normalized
    format produced by CloudWatchLogsParser:
    - snake_case field names (trace_id, span_id)
    - Messages in span_events[].body.input/output.messages

    For raw CloudWatch logs, use CloudWatchLogsParser first to normalize.
    """

    def __init__(self):
        super().__init__()
        warnings.warn(
            "CloudWatchSessionMapper is deprecated. Use CloudWatchProvider with "
            "auto-detection (detect_otel_mapper) instead, which automatically selects "
            "the correct mapper based on the framework that produced the traces.",
            DeprecationWarning,
            stacklevel=2,
        )

    def map_to_session(self, spans: list[dict[str, Any]], session_id: str) -> Session:
        """Map normalized spans to Session format.

        Args:
            spans: Normalized span dicts from CloudWatchLogsParser
            session_id: Session identifier

        Returns:
            Session object ready for evaluation
        """
        # Group spans by trace_id
        traces_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for span in spans:
            # Support both normalized (trace_id) and raw (traceId) formats
            trace_id = span.get("trace_id") or span.get("traceId", "")
            if trace_id:
                traces_by_id[trace_id].append(span)

        traces: list[Trace] = []
        for trace_id, trace_spans in traces_by_id.items():
            trace = self._convert_trace(trace_id, trace_spans, session_id)
            if trace.spans:
                traces.append(trace)

        return Session(session_id=session_id, traces=traces)

    def _convert_trace(self, trace_id: str, spans: list[dict[str, Any]], session_id: str) -> Trace:
        """Convert spans with the same trace_id into a Trace with typed spans."""
        # Sort by start_time or timestamp
        sorted_spans = sorted(spans, key=lambda s: s.get("start_time") or s.get("timeUnixNano", 0))

        result_spans: list[InferenceSpan | ToolExecutionSpan | AgentInvocationSpan] = []

        # Collect all tool calls and results
        all_tool_calls: dict[str, ToolCall] = {}
        all_tool_results: dict[str, ToolResult] = {}

        for span in sorted_spans:
            body = get_body(span)
            if not body:
                continue

            for tc in self._extract_tool_calls(body):
                if tc.tool_call_id:
                    all_tool_calls[tc.tool_call_id] = tc

            for tr in self._extract_tool_results(body):
                if tr.tool_call_id:
                    all_tool_results[tr.tool_call_id] = tr

        # Create InferenceSpans
        for span in sorted_spans:
            body = get_body(span)
            if not body:
                continue

            try:
                messages = self._body_to_messages(body)
                if messages:
                    span_info = self._create_span_info(span, session_id)
                    result_spans.append(InferenceSpan(span_info=span_info, messages=messages, metadata={}))
            except Exception as e:
                span_id = span.get("span_id") or span.get("spanId", "unknown")
                logger.warning("Failed to create inference span from %s: %s", span_id, e)

        # Create ToolExecutionSpans
        seen_tool_ids: set[str] = set()
        for span in sorted_spans:
            body = get_body(span)
            if not body:
                continue

            for tc in self._extract_tool_calls(body):
                if tc.tool_call_id and tc.tool_call_id not in seen_tool_ids:
                    seen_tool_ids.add(tc.tool_call_id)
                    tr = all_tool_results.get(tc.tool_call_id, ToolResult(content="", tool_call_id=tc.tool_call_id))
                    span_info = self._create_span_info(span, session_id)
                    result_spans.append(
                        ToolExecutionSpan(span_info=span_info, tool_call=tc, tool_result=tr, metadata={})
                    )

        # Create AgentInvocationSpan
        agent_span = self._create_agent_invocation_span(sorted_spans, all_tool_calls, session_id)
        if agent_span:
            result_spans.append(agent_span)

        return Trace(spans=result_spans, trace_id=trace_id, session_id=session_id)

    def _create_span_info(self, span: dict, session_id: str) -> SpanInfo:
        """Create SpanInfo from span dict."""
        # Handle both normalized and raw field names
        trace_id = span.get("trace_id") or span.get("traceId", "")
        span_id = span.get("span_id") or span.get("spanId", "")
        parent_span_id = span.get("parent_span_id") or span.get("parentSpanId")

        # Handle timestamps
        time_value = span.get("start_time") or span.get("timeUnixNano", 0)
        if isinstance(time_value, (int, float)) and time_value > 0:
            ts = datetime.fromtimestamp(time_value / 1e9, tz=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        return SpanInfo(
            trace_id=trace_id,
            span_id=span_id,
            session_id=session_id,
            parent_span_id=parent_span_id,
            start_time=ts,
            end_time=ts,
        )

    def _create_agent_invocation_span(
        self, spans: list[dict], tool_calls: dict[str, ToolCall], session_id: str
    ) -> AgentInvocationSpan | None:
        """Create AgentInvocationSpan from first user prompt and last agent response."""
        user_prompt = None
        for span in spans:
            body = get_body(span)
            if body:
                prompt = self._extract_user_prompt(body)
                if prompt:
                    user_prompt = prompt
                    break

        if not user_prompt:
            return None

        agent_response = None
        best_span = None
        for span in reversed(spans):
            body = get_body(span)
            if body:
                response = self._extract_agent_response(body)
                if response:
                    agent_response = response
                    best_span = span
                    break

        if not agent_response or not best_span:
            return None

        available_tools = [ToolConfig(name=name) for name in sorted({tc.name for tc in tool_calls.values()})]
        span_info = self._create_span_info(best_span, session_id)

        return AgentInvocationSpan(
            span_info=span_info,
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=available_tools,
            metadata={},
        )

    # --- Body-based content extraction ---

    def _parse_message_content(self, raw: str) -> list[dict[str, Any]] | None:
        """Parse double-encoded message content into a list of content blocks."""
        if not isinstance(raw, str):
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else None
        except (json.JSONDecodeError, TypeError):
            return None

    def _extract_content_field(self, content: dict[str, Any]) -> str | None:
        """Extract the raw content field from a message."""
        if not isinstance(content, dict):
            return None
        return content.get("content") or content.get("message")

    def _extract_text_from_content(self, content: Any) -> str | None:
        """Extract text from a content field, handling double-encoded JSON strings."""
        raw = self._extract_content_field(content)
        if not raw:
            return None

        parsed = self._parse_message_content(raw)
        if parsed:
            texts = [item["text"] for item in parsed if isinstance(item, dict) and "text" in item]
            return " ".join(texts) if texts else None

        return raw if isinstance(raw, str) else None

    def _extract_message_text(self, body: dict, message_type: str, role: str) -> str | None:
        """Extract text from a specific message type and role."""
        messages = body.get(message_type, {}).get("messages", [])
        for msg in messages:
            if msg.get("role") == role:
                text = self._extract_text_from_content(msg.get("content", {}))
                if text:
                    return text
        return None

    def _extract_user_prompt(self, body: dict) -> str | None:
        """Extract user prompt text from body.input.messages."""
        return self._extract_message_text(body, "input", "user")

    def _extract_agent_response(self, body: dict) -> str | None:
        """Extract assistant text response from body.output.messages."""
        return self._extract_message_text(body, "output", "assistant")

    def _extract_tool_calls(self, body: dict) -> list[ToolCall]:
        """Extract tool calls from body.output.messages."""
        tool_calls: list[ToolCall] = []

        for msg in body.get("output", {}).get("messages", []):
            if msg.get("role") != "assistant":
                continue

            raw = self._extract_content_field(msg.get("content", {}))
            parsed = self._parse_message_content(raw) if raw else None
            if not parsed:
                continue

            for item in parsed:
                if isinstance(item, dict) and "toolUse" in item:
                    tu = item["toolUse"]
                    tool_calls.append(
                        ToolCall(
                            name=tu.get("name", ""),
                            arguments=tu.get("input", {}),
                            tool_call_id=tu.get("toolUseId"),
                        )
                    )

        return tool_calls

    def _extract_tool_results(self, body: dict) -> list[ToolResult]:
        """Extract tool results from body.input.messages."""
        tool_results: list[ToolResult] = []

        for msg in body.get("input", {}).get("messages", []):
            raw = self._extract_content_field(msg.get("content", {}))
            parsed = self._parse_message_content(raw) if raw else None
            if not parsed:
                continue

            for item in parsed:
                if isinstance(item, dict) and "toolResult" in item:
                    tr_data = item["toolResult"]
                    result_text = self._extract_tool_result_text(tr_data.get("content"))
                    tool_results.append(
                        ToolResult(
                            content=result_text,
                            error=tr_data.get("error"),
                            tool_call_id=tr_data.get("toolUseId"),
                        )
                    )

        return tool_results

    def _extract_tool_result_text(self, content: Any) -> str:
        """Extract text from tool result content."""
        if not content:
            return ""
        if isinstance(content, list) and content:
            return content[0].get("text", "")
        return str(content)

    # --- Body-to-messages conversion ---

    def _body_to_messages(self, body: dict) -> list[UserMessage | AssistantMessage]:
        """Convert body into a list of typed messages for InferenceSpan."""
        messages: list[UserMessage | AssistantMessage] = []

        # Process input messages
        for msg in body.get("input", {}).get("messages", []):
            role = msg.get("role", "")
            raw = self._extract_content_field(msg.get("content", {}))
            parsed = self._parse_message_content(raw) if raw else None
            if not parsed:
                continue

            if role == "user":
                user_content = self._process_user_message(parsed)
                if user_content:
                    messages.append(UserMessage(content=user_content))
            elif role == "tool":
                tool_content = self._process_tool_results(parsed)
                if tool_content:
                    messages.append(UserMessage(content=tool_content))

        # Process output messages
        for msg in body.get("output", {}).get("messages", []):
            if msg.get("role") != "assistant":
                continue

            raw = self._extract_content_field(msg.get("content", {}))
            parsed = self._parse_message_content(raw) if raw else None
            if not parsed:
                continue

            assistant_content = self._process_assistant_content(parsed)
            if assistant_content:
                messages.append(AssistantMessage(content=assistant_content))

        return messages

    # --- Content parsing helpers (Bedrock Converse format) ---

    @staticmethod
    def _process_user_message(content_list: list[dict[str, Any]]) -> list[TextContent | ToolResultContent]:
        return [TextContent(text=item["text"]) for item in content_list if "text" in item]

    @staticmethod
    def _process_assistant_content(content_list: list[dict[str, Any]]) -> list[TextContent | ToolCallContent]:
        result: list[TextContent | ToolCallContent] = []
        for item in content_list:
            if "text" in item:
                result.append(TextContent(text=item["text"]))
            elif "toolUse" in item:
                tool_use = item["toolUse"]
                result.append(
                    ToolCallContent(
                        name=tool_use["name"],
                        arguments=tool_use.get("input", {}),
                        tool_call_id=tool_use.get("toolUseId"),
                    )
                )
        return result

    def _process_tool_results(self, content_list: list[dict[str, Any]]) -> list[TextContent | ToolResultContent]:
        result: list[TextContent | ToolResultContent] = []
        for item in content_list:
            if "toolResult" not in item:
                continue
            tool_result = item["toolResult"]
            result_text = self._extract_tool_result_text(tool_result.get("content"))
            result.append(
                ToolResultContent(
                    content=result_text,
                    error=tool_result.get("error"),
                    tool_call_id=tool_result.get("toolUseId"),
                )
            )
        return result
