"""OpenSearch session mapper — converts genai-sdk SpanRecords to Session format."""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from strands_evals.mappers.session_mapper import SessionMapper
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    ToolResultContent,
    Trace,
    UserMessage,
)

logger = logging.getLogger(__name__)


class OpenSearchSessionMapper(SessionMapper):
    """Maps genai-sdk SpanRecords (from OpenSearch) to strands-evals Session format."""

    def map_to_session(self, span_records: list[Any], session_id: str) -> Session:
        """Group SpanRecords by trace_id, convert each group to a Trace."""
        traces_by_id: dict[str, list[Any]] = defaultdict(list)
        for record in span_records:
            if record.trace_id:
                traces_by_id[record.trace_id].append(record)

        traces = []
        for trace_id, records in traces_by_id.items():
            trace = self._convert_trace(trace_id, records, session_id)
            if trace.spans:
                traces.append(trace)

        return Session(session_id=session_id, traces=traces)

    def _convert_trace(self, trace_id: str, records: list[Any], session_id: str) -> Trace:
        """Convert SpanRecords sharing a trace_id into a Trace with typed spans."""
        sorted_records = sorted(records, key=lambda r: r.start_time or "")
        spans = []

        for record in sorted_records:
            span_info = self._make_span_info(record, session_id)

            if record.operation_name == "chat":
                messages = self._build_messages(record)
                if messages:
                    spans.append(InferenceSpan(span_info=span_info, messages=messages, metadata={}))

            elif record.operation_name == "execute_tool":
                arguments = self._parse_json(record.tool_call_arguments)
                tool_call = ToolCall(
                    name=record.tool_name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    tool_call_id=record.span_id,
                )
                tool_result = ToolResult(
                    content=record.tool_call_result or "",
                    tool_call_id=record.span_id,
                )
                spans.append(ToolExecutionSpan(
                    span_info=span_info, tool_call=tool_call, tool_result=tool_result, metadata={}
                ))

            elif record.operation_name == "invoke_agent":
                user_prompt = self._first_message_by_role(record.input_messages, "user")
                agent_response = self._last_message_by_role(record.output_messages, "assistant")
                if user_prompt or agent_response:
                    # Collect tool names only from direct child spans
                    tool_names = sorted({
                        r.tool_name for r in sorted_records
                        if r.operation_name == "execute_tool" and r.tool_name
                        and r.parent_span_id == record.span_id
                    })
                    spans.append(AgentInvocationSpan(
                        span_info=span_info,
                        user_prompt=user_prompt or "",
                        agent_response=agent_response or "",
                        available_tools=[ToolConfig(name=n) for n in tool_names],
                        metadata={},
                    ))

        return Trace(spans=spans, trace_id=trace_id, session_id=session_id)

    @staticmethod
    def _make_span_info(record: Any, session_id: str) -> SpanInfo:
        return SpanInfo(
            trace_id=record.trace_id,
            span_id=record.span_id,
            session_id=session_id,
            parent_span_id=record.parent_span_id or None,
            start_time=_parse_time(record.start_time),
            end_time=_parse_time(record.end_time),
        )

    @staticmethod
    def _build_messages(record: Any) -> list[UserMessage | AssistantMessage]:
        messages = []
        for msg in record.input_messages:
            if msg.role == "user":
                messages.append(UserMessage(content=[TextContent(text=msg.content)]))
            elif msg.role == "tool":
                messages.append(UserMessage(content=[ToolResultContent(content=msg.content, tool_call_id=None)]))
        for msg in record.output_messages:
            if msg.role == "assistant":
                messages.append(AssistantMessage(content=[TextContent(text=msg.content)]))
        return messages

    @staticmethod
    def _first_message_by_role(messages: list, role: str) -> str | None:
        for msg in messages:
            if msg.role == role and msg.content:
                return msg.content
        return None

    @staticmethod
    def _last_message_by_role(messages: list, role: str) -> str | None:
        for msg in reversed(messages):
            if msg.role == role and msg.content:
                return msg.content
        return None

    @staticmethod
    def _parse_json(value: str) -> dict | str:
        if not value:
            return {}
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value


def _parse_time(value: str) -> datetime:
    """Parse ISO timestamp string to datetime, defaulting to epoch on failure."""
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return datetime.fromtimestamp(0, tz=timezone.utc)
