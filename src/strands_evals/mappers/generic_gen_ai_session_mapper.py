"""
GenericGenAISessionMapper - Maps dict-format spans with gen_ai.* attributes to Session format.

Handles spans from any framework that uses OpenTelemetry GenAI semantic conventions
(gen_ai.operation.name, gen_ai.tool.name, etc.) but has an unrecognized scope name.

This is the fallback mapper for spans that don't match any known instrumentor scope
(Strands, OpenInference, LangChain) but still carry structured gen_ai.* attributes.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

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
from .session_mapper import SessionMapper

logger = logging.getLogger(__name__)


class GenericGenAISessionMapper(SessionMapper):
    """Maps dict-format spans with gen_ai.* attributes to Session format.

    Classifies spans by gen_ai.operation.name:
    - "chat" → InferenceSpan
    - "execute_tool" → ToolExecutionSpan
    - "invoke_agent" → AgentInvocationSpan

    Extracts data from both span attributes (gen_ai.tool.input, gen_ai.tool.output)
    and span_events (gen_ai.user.message, gen_ai.choice, etc.).
    """

    def map_to_session(self, data: Any, session_id: str) -> Session:
        """Map dict-format spans to Session.

        Args:
            data: List of span dicts (from readable_spans_to_dicts or similar)
            session_id: Session identifier for filtering

        Returns:
            Session object ready for evaluation
        """
        if not data:
            return Session(traces=[], session_id=session_id)

        spans = self._normalize_to_flat_spans(data)

        # Group by trace_id, optionally filtering by session.id
        traces_by_id: dict[str, list[dict]] = defaultdict(list)

        any_span_has_session_id = any(
            "session.id" in span.get("attributes", {}) or "gen_ai.conversation.id" in span.get("attributes", {})
            for span in spans
        )

        for span in spans:
            attrs = span.get("attributes", {})

            if not any_span_has_session_id:
                should_include = True
            else:
                span_session_id = attrs.get("gen_ai.conversation.id") or attrs.get("session.id")
                should_include = str(span_session_id) == session_id if span_session_id else True

            if should_include:
                trace_id = span.get("trace_id", "unknown")
                traces_by_id[trace_id].append(span)

        traces: list[Trace] = []
        for trace_id, trace_spans in traces_by_id.items():
            trace = self._convert_trace(trace_id, trace_spans, session_id)
            if trace.spans:
                traces.append(trace)

        return Session(traces=traces, session_id=session_id)

    def _convert_trace(self, trace_id: str, spans: list[dict], session_id: str) -> Trace:
        """Convert a list of dict spans (same trace_id) to a Trace."""
        converted_spans: list[InferenceSpan | ToolExecutionSpan | AgentInvocationSpan] = []

        for span in spans:
            try:
                attrs = span.get("attributes", {})
                operation_name = attrs.get("gen_ai.operation.name", "")

                span_info = SpanInfo(
                    trace_id=span.get("trace_id"),
                    span_id=span.get("span_id"),
                    session_id=session_id,
                    parent_span_id=span.get("parent_span_id"),
                    start_time=self._parse_timestamp(span.get("start_time")),
                    end_time=self._parse_timestamp(span.get("end_time")),
                )

                if operation_name == "chat":
                    inference_span = self._convert_inference_span(span, span_info)
                    if inference_span and inference_span.messages:
                        converted_spans.append(inference_span)
                elif operation_name == "execute_tool":
                    tool_span = self._convert_tool_execution_span(span, span_info)
                    if tool_span:
                        converted_spans.append(tool_span)
                elif operation_name == "invoke_agent":
                    agent_span = self._convert_agent_invocation_span(span, span_info)
                    if agent_span:
                        converted_spans.append(agent_span)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    "trace_id=%s span_id=%s | Failed to convert span: %s",
                    span.get("trace_id", "?"), span.get("span_id", "?"), e,
                )

        return Trace(spans=converted_spans, trace_id=trace_id, session_id=session_id)

    def _parse_timestamp(self, value: Any) -> datetime:
        """Parse timestamp from various formats.

        readable_spans_to_dicts() preserves OTel's nanosecond epoch timestamps.
        Values > 1e12 are assumed nanoseconds and divided by 1e9.
        """
        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            # OTel timestamps from ReadableSpan are nanoseconds (> 1e18).
            # Seconds-based timestamps are < 1e10. Threshold at 1e12 to distinguish.
            if value > 1e12:
                value = value / 1e9
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return datetime.now(timezone.utc)

    def _parse_json_attr(self, attributes: Any, key: str, default: str = "[]") -> Any:
        """Parse a JSON-encoded attribute value."""
        try:
            value = attributes.get(key, default)
            return json.loads(str(value))
        except (AttributeError, TypeError, json.JSONDecodeError):
            return json.loads(default)

    # =========================================================================
    # Span Conversion
    # =========================================================================

    def _convert_inference_span(self, span: dict, span_info: SpanInfo) -> InferenceSpan | None:
        """Convert a span with gen_ai.operation.name='chat' to InferenceSpan."""
        messages: list[UserMessage | AssistantMessage] = []

        for event in span.get("span_events", []):
            try:
                event_name = event.get("event_name", "")
                event_attrs = event.get("attributes", {})

                if event_name == "gen_ai.user.message":
                    content_list = self._parse_json_attr(event_attrs, "content")
                    user_content: list[TextContent | ToolResultContent] = [
                        TextContent(text=item["text"]) for item in content_list if "text" in item
                    ]
                    if user_content:
                        messages.append(UserMessage(content=user_content))

                elif event_name == "gen_ai.assistant.message":
                    content_list = self._parse_json_attr(event_attrs, "content")
                    assistant_content = self._process_assistant_content(content_list)
                    if assistant_content:
                        messages.append(AssistantMessage(content=assistant_content))

                elif event_name == "gen_ai.tool.message":
                    content_list = self._parse_json_attr(event_attrs, "content")
                    tool_results = self._process_tool_results(content_list)
                    if tool_results:
                        messages.append(UserMessage(content=tool_results))

                elif event_name == "gen_ai.choice":
                    message_list = self._parse_json_attr(event_attrs, "message")
                    assistant_content = self._process_assistant_content(message_list)
                    if assistant_content:
                        messages.append(AssistantMessage(content=assistant_content))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    "span_id=%s event=%s | Failed to process event: %s",
                    span.get("span_id", "?"), event.get("event_name", "?"), e,
                )

        return InferenceSpan(span_info=span_info, messages=messages, metadata={})

    def _convert_tool_execution_span(self, span: dict, span_info: SpanInfo) -> ToolExecutionSpan | None:
        """Convert a span with gen_ai.operation.name='execute_tool' to ToolExecutionSpan."""
        attrs = span.get("attributes", {})

        tool_name = str(attrs.get("gen_ai.tool.name", ""))
        tool_call_id = str(attrs.get("gen_ai.tool.call.id", ""))
        tool_status = attrs.get("gen_ai.tool.status", attrs.get("tool.status", ""))
        tool_error = None if tool_status == "success" else (str(tool_status) if tool_status else None)

        tool_arguments: dict = {}
        tool_result_content: str = ""

        # Try span_events first (Strands telemetry format)
        for event in span.get("span_events", []):
            try:
                event_name = event.get("event_name", "")
                event_attrs = event.get("attributes", {})

                if event_name == "gen_ai.tool.message":
                    tool_arguments = self._parse_json_attr(event_attrs, "content", "{}")
                elif event_name == "gen_ai.choice":
                    message_list = self._parse_json_attr(event_attrs, "message")
                    tool_result_content = message_list[0].get("text", "") if message_list else ""
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    "span_id=%s | Failed to process tool event: %s",
                    span.get("span_id", "?"), e,
                )

        # Fallback: gen_ai.tool.input/output attributes (manual instrumentation)
        if not tool_arguments:
            tool_input = attrs.get("gen_ai.tool.input", "")
            if tool_input:
                try:
                    parsed = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
                    tool_arguments = parsed if isinstance(parsed, dict) else {}
                except (json.JSONDecodeError, TypeError):
                    tool_arguments = {}

        if not tool_result_content:
            tool_output = attrs.get("gen_ai.tool.output", "")
            if tool_output:
                tool_result_content = str(tool_output)

        if not tool_name:
            return None

        tool_call = ToolCall(name=tool_name, arguments=tool_arguments, tool_call_id=tool_call_id)
        tool_result = ToolResult(content=tool_result_content, error=tool_error, tool_call_id=tool_call_id)

        return ToolExecutionSpan(span_info=span_info, tool_call=tool_call, tool_result=tool_result, metadata={})

    def _convert_agent_invocation_span(self, span: dict, span_info: SpanInfo) -> AgentInvocationSpan | None:
        """Convert a span with gen_ai.operation.name='invoke_agent' to AgentInvocationSpan."""
        attrs = span.get("attributes", {})

        user_prompt = ""
        agent_response = ""
        available_tools: list[ToolConfig] = []

        # Parse available tools
        try:
            tools_json = attrs.get("gen_ai.agent.tools", "[]")
            tool_names = json.loads(str(tools_json)) if isinstance(tools_json, str) else tools_json
            if isinstance(tool_names, list):
                available_tools = [ToolConfig(name=name) for name in tool_names]
        except (json.JSONDecodeError, TypeError):
            pass

        # Extract from span_events
        for event in span.get("span_events", []):
            try:
                event_name = event.get("event_name", "")
                event_attrs = event.get("attributes", {})

                if event_name == "gen_ai.user.message":
                    content_list = self._parse_json_attr(event_attrs, "content")
                    user_prompt = content_list[0].get("text", "") if content_list else ""
                elif event_name == "gen_ai.choice":
                    msg = event_attrs.get("message", "")
                    agent_response = str(msg)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    "span_id=%s | Failed to process agent event: %s",
                    span.get("span_id", "?"), e,
                )

        return AgentInvocationSpan(
            span_info=span_info,
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=available_tools,
            metadata={},
        )

    # =========================================================================
    # Content Processing Helpers
    # =========================================================================

    @staticmethod
    def _process_assistant_content(content_list: list[dict]) -> list[TextContent | ToolCallContent]:
        """Process assistant message content blocks."""
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

    @staticmethod
    def _process_tool_results(content_list: list[dict]) -> list[TextContent | ToolResultContent]:
        """Process tool result content blocks."""
        result: list[TextContent | ToolResultContent] = []
        for item in content_list:
            if "toolResult" in item:
                tool_result = item["toolResult"]
                result_text = tool_result.get("content", "")
                if isinstance(result_text, list):
                    result_text = "\n".join(
                        block.get("text", "") for block in result_text if isinstance(block, dict) and "text" in block
                    )
                result.append(
                    ToolResultContent(
                        content=str(result_text),
                        error=tool_result.get("error"),
                        tool_call_id=tool_result.get("toolUseId"),
                    )
                )
        return result
