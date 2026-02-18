"""CloudWatch trace provider for retrieving agent traces from AWS CloudWatch Logs."""

import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3

from ..providers.exceptions import ProviderError, SessionNotFoundError, TraceNotFoundError
from ..providers.trace_provider import SessionFilter, TraceProvider
from ..types.evaluation import TaskOutput
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


class CloudWatchProvider(TraceProvider):
    """Retrieves agent trace data from AWS CloudWatch Logs for evaluation.

    Queries CloudWatch Logs Insights to fetch OTEL log records from an
    agent-specific runtime log group, parses body.input/output messages,
    and returns Session objects ready for the evaluation pipeline.
    """

    def __init__(
        self,
        region: str | None = None,
        log_group: str | None = None,
        agent_name: str | None = None,
        lookback_days: int = 30,
        query_timeout_seconds: float = 60.0,
    ):
        resolved_region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        try:
            self._client = boto3.client("logs", region_name=resolved_region)
        except Exception as e:
            raise ProviderError(f"CloudWatch: failed to create boto3 logs client: {e}") from e

        if log_group:
            self._log_group = log_group
        elif agent_name:
            self._log_group = self._discover_log_group(agent_name)
        else:
            raise ProviderError("CloudWatch: either log_group or agent_name must be provided")

        self._lookback_days = lookback_days
        self._query_timeout_seconds = query_timeout_seconds

    def _discover_log_group(self, agent_name: str) -> str:
        """Discover the runtime log group for an agent via describe_log_groups."""
        prefix = f"/aws/bedrock-agentcore/runtimes/{agent_name}"
        response = self._client.describe_log_groups(logGroupNamePrefix=prefix)
        log_groups = response.get("logGroups", [])
        if not log_groups:
            raise ProviderError(f"CloudWatch: no log group found for agent_name='{agent_name}' (prefix={prefix})")
        return log_groups[0]["logGroupName"]

    def get_evaluation_data(self, session_id: str) -> TaskOutput:
        """Fetch all traces for a session and return evaluation data."""
        query = f"fields @message | filter attributes.session.id = '{session_id}' | sort @timestamp asc | limit 10000"

        try:
            span_dicts = self._run_logs_insights_query(query)
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"CloudWatch: failed to query spans for session '{session_id}': {e}") from e

        if not span_dicts:
            raise SessionNotFoundError(f"CloudWatch: no spans found for session_id='{session_id}'")

        session = self._build_session(session_id, span_dicts)

        if not session.traces:
            raise SessionNotFoundError(
                f"CloudWatch: spans found for session_id='{session_id}' but none contained convertible spans"
            )

        output = self._extract_output(session)
        return TaskOutput(output=output, trajectory=session)

    def get_evaluation_data_by_trace_id(self, trace_id: str) -> TaskOutput:
        """Fetch a single trace by ID and return evaluation data."""
        query = f'fields @message | filter traceId = "{trace_id}" | sort @timestamp asc | limit 10000'

        try:
            span_dicts = self._run_logs_insights_query(query)
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"CloudWatch: failed to query trace '{trace_id}': {e}") from e

        if not span_dicts:
            raise TraceNotFoundError(f"CloudWatch: no spans found for trace_id='{trace_id}'")

        # Extract session_id from the first record's attributes
        first_attrs = span_dicts[0].get("attributes", {})
        session_id = first_attrs.get("session.id") or trace_id

        session = self._build_session(session_id, span_dicts)
        output = self._extract_output(session)
        return TaskOutput(output=output, trajectory=session)

    def list_sessions(self, session_filter: SessionFilter | None = None) -> Iterator[str]:
        """Yield distinct session IDs from CloudWatch Logs."""
        limit = session_filter.limit if session_filter and session_filter.limit else 1000
        start_time = session_filter.start_time if session_filter else None
        end_time = session_filter.end_time if session_filter else None

        query = (
            "fields attributes.session.id as sessionId"
            " | filter ispresent(attributes.session.id)"
            " | stats count(*) as span_count by sessionId"
            " | sort sessionId asc"
            f" | limit {limit}"
        )

        try:
            results = self._run_raw_logs_insights_query(query, start_time=start_time, end_time=end_time)
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"CloudWatch: failed to list sessions: {e}") from e

        for row in results:
            for field in row:
                if field.get("field") == "sessionId":
                    yield field["value"]

    # --- Internal: CW Logs Insights query execution ---

    def _run_logs_insights_query(
        self, query: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Execute a CW Logs Insights query and return parsed span dicts from @message fields."""
        raw_results = self._run_raw_logs_insights_query(query, start_time=start_time, end_time=end_time)
        return self._parse_query_results(raw_results)

    def _run_raw_logs_insights_query(
        self,
        query: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[list[dict[str, str]]]:
        """Execute a CW Logs Insights query and return raw result rows."""
        now = datetime.now(tz=timezone.utc)
        if end_time is None:
            end_time = now
        if start_time is None:
            start_time = now - timedelta(days=self._lookback_days)

        try:
            response = self._client.start_query(
                logGroupName=self._log_group,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query,
            )
        except Exception as e:
            raise ProviderError(f"CloudWatch: failed to start query: {e}") from e

        query_id = response["queryId"]
        return self._poll_query_results(query_id)

    def _poll_query_results(self, query_id: str) -> list[list[dict[str, str]]]:
        """Poll for query completion with exponential backoff. Returns raw result rows."""
        delay = 0.5
        max_delay = 8.0
        deadline = time.monotonic() + self._query_timeout_seconds

        while True:
            response = self._client.get_query_results(queryId=query_id)
            status = response.get("status", "")

            if status == "Complete":
                return response.get("results", [])
            elif status in ("Failed", "Cancelled", "Timeout"):
                raise ProviderError(f"CloudWatch: query {status}")

            if time.monotonic() >= deadline:
                raise ProviderError(f"CloudWatch: query timed out after {self._query_timeout_seconds}s")

            time.sleep(delay)
            delay = min(delay * 2, max_delay)

    @staticmethod
    def _parse_query_results(results: list[list[dict[str, str]]]) -> list[dict[str, Any]]:
        """Parse @message fields from CW Logs Insights results into span dicts."""
        span_dicts = []
        for row in results:
            for field in row:
                if field.get("field") == "@message":
                    try:
                        span_dicts.append(json.loads(field["value"]))
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Failed to parse @message: %s", e)
        return span_dicts

    # --- Internal: session building (body-based parsing) ---

    def _build_session(self, session_id: str, records: list[dict[str, Any]]) -> Session:
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

    def _extract_output(self, session: Session) -> str:
        """Extract the final agent response from the session for TaskOutput.output."""
        for trace in reversed(session.traces):
            for span in reversed(trace.spans):
                if isinstance(span, AgentInvocationSpan):
                    return span.agent_response
        return ""

    # --- Internal: span info ---

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

    # --- Internal: body-based content extraction ---

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

    # --- Internal: record-to-messages conversion ---

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

    # --- Internal: content parsing helpers (Bedrock Converse format) ---

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
