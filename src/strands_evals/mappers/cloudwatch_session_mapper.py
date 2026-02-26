"""CloudWatch session mapper — converts CW Logs Insights records to Session format."""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from ..mappers.session_mapper import SessionMapper
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
    """Maps CloudWatch Logs Insights records to Session format.

    Parses body.input/output messages from OTEL log records emitted by
    Strands agent runtimes and builds typed Session objects for evaluation.
    """

    def map_to_session(self, records: list[dict[str, Any]], session_id: str) -> Session:
        """Group log records by traceId, convert each group to a Trace, return a Session."""
        traces_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            trace_id = record.get("traceId", "")
            if not trace_id:
                continue
            traces_by_id[trace_id].append(record)

        traces: list[Trace] = []
        for trace_id, trace_records in traces_by_id.items():
            trace = self._convert_trace(trace_id, trace_records, session_id)
            if trace.spans:
                traces.append(trace)

        return Session(session_id=session_id, traces=traces)

    def _convert_trace(self, trace_id: str, records: list[dict[str, Any]], session_id: str) -> Trace:
        """Convert a group of log records (same traceId) into a Trace with typed spans."""
        sorted_records = sorted(records, key=lambda r: r.get("timeUnixNano", 0))

        spans: list[InferenceSpan | ToolExecutionSpan | AgentInvocationSpan] = []

        # Collect all tool calls and results across records
        all_tool_calls: dict[str, ToolCall] = {}
        all_tool_results: dict[str, ToolResult] = {}

        for record in sorted_records:
            if not isinstance(record.get("body"), dict):
                continue

            for tc in self._extract_tool_calls(record):
                if tc.tool_call_id:
                    all_tool_calls[tc.tool_call_id] = tc

            for tr in self._extract_tool_results(record):
                if tr.tool_call_id:
                    all_tool_results[tr.tool_call_id] = tr

        # Create InferenceSpans (one per record with parseable body)
        for record in sorted_records:
            if not isinstance(record.get("body"), dict):
                continue

            try:
                messages = self._record_to_messages(record)
                if messages:
                    span_info = self._create_span_info(record, session_id)
                    spans.append(InferenceSpan(span_info=span_info, messages=messages, metadata={}))
            except Exception as e:
                logger.warning("Failed to create inference span from record %s: %s", record.get("spanId"), e)

        # Create ToolExecutionSpans by matching calls to results
        seen_tool_ids: set[str] = set()
        for record in sorted_records:
            for tc in self._extract_tool_calls(record):
                if tc.tool_call_id and tc.tool_call_id not in seen_tool_ids:
                    seen_tool_ids.add(tc.tool_call_id)
                    tr = all_tool_results.get(tc.tool_call_id, ToolResult(content="", tool_call_id=tc.tool_call_id))
                    span_info = self._create_span_info(record, session_id)
                    spans.append(ToolExecutionSpan(span_info=span_info, tool_call=tc, tool_result=tr, metadata={}))

        # Create AgentInvocationSpan from first user prompt + last agent response
        agent_span = self._create_agent_invocation_span(sorted_records, all_tool_calls, session_id)
        if agent_span:
            spans.append(agent_span)

        return Trace(spans=spans, trace_id=trace_id, session_id=session_id)

    def _create_agent_invocation_span(
        self, records: list[dict[str, Any]], tool_calls: dict[str, ToolCall], session_id: str
    ) -> AgentInvocationSpan | None:
        """Create an AgentInvocationSpan from the first user prompt and last agent response."""
        user_prompt = None
        for record in records:
            prompt = self._extract_user_prompt(record)
            if prompt:
                user_prompt = prompt
                break

        if not user_prompt:
            return None

        agent_response = None
        best_record = None
        for record in reversed(records):
            response = self._extract_agent_response(record)
            if response:
                agent_response = response
                best_record = record
                break

        if not agent_response or not best_record:
            return None

        available_tools = [ToolConfig(name=name) for name in sorted({tc.name for tc in tool_calls.values()})]
        span_info = self._create_span_info(best_record, session_id)

        return AgentInvocationSpan(
            span_info=span_info,
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=available_tools,
            metadata={},
        )

    # --- Span info ---

    def _create_span_info(self, record: dict[str, Any], session_id: str) -> SpanInfo:
        time_nano = record.get("timeUnixNano", 0)
        ts = datetime.fromtimestamp(time_nano / 1e9, tz=timezone.utc)

        return SpanInfo(
            trace_id=record.get("traceId", ""),
            span_id=record.get("spanId", ""),
            session_id=session_id,
            parent_span_id=record.get("parentSpanId") or None,
            start_time=ts,
            end_time=ts,
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

    def _extract_message_text(self, record: dict[str, Any], message_type: str, role: str) -> str | None:
        """Extract text from a specific message type and role in a log record."""
        body = record.get("body", {})
        if not isinstance(body, dict):
            return None

        messages = body.get(message_type, {}).get("messages", [])
        for msg in messages:
            if msg.get("role") == role:
                text = self._extract_text_from_content(msg.get("content", {}))
                if text:
                    return text
        return None

    def _extract_user_prompt(self, record: dict[str, Any]) -> str | None:
        """Extract user prompt text from a log record's body.input.messages."""
        return self._extract_message_text(record, "input", "user")

    def _extract_agent_response(self, record: dict[str, Any]) -> str | None:
        """Extract assistant text response from a log record's body.output.messages."""
        return self._extract_message_text(record, "output", "assistant")

    def _extract_tool_calls(self, record: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from a log record's body.output.messages."""
        tool_calls: list[ToolCall] = []
        body = record.get("body", {})
        if not isinstance(body, dict):
            return tool_calls

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

    def _extract_tool_results(self, record: dict[str, Any]) -> list[ToolResult]:
        """Extract tool results from a log record's body.input.messages."""
        tool_results: list[ToolResult] = []
        body = record.get("body", {})
        if not isinstance(body, dict):
            return tool_results

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

    # --- Record-to-messages conversion ---

    def _record_to_messages(self, record: dict[str, Any]) -> list[UserMessage | AssistantMessage]:
        """Convert a log record's body into a list of typed messages for InferenceSpan."""
        messages: list[UserMessage | AssistantMessage] = []
        body = record.get("body", {})
        if not isinstance(body, dict):
            return messages

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
