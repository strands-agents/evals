"""HoneyHive trace provider for retrieving agent traces from HoneyHive."""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import Retrying, before_sleep_log, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..types.evaluation import TaskOutput
from ..types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanInfo,
    SpanUnion,
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
from .exceptions import ProviderError, SessionNotFoundError, TraceProviderError
from .trace_provider import TraceProvider

logger = logging.getLogger(__name__)

# The events search API caps `limit` at 1000.
_PAGE_SIZE = 1000
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2


class HoneyHiveProvider(TraceProvider):
    """Retrieves Strands agent sessions from HoneyHive for evaluation.

    HoneyHive stores each OpenTelemetry span as an event. This provider reads a session's events
    and converts the Strands Python and TypeScript spans to typed evals spans. Event loop cycle
    spans are dropped.
    """

    def __init__(self, api_key: str | None = None, api_url: str | None = None, timeout: float = 60.0):
        """Initialize the HoneyHive provider.

        Example::

            from strands_evals.providers import HoneyHiveProvider

            # Reads HH_API_KEY / HH_API_URL from env
            provider = HoneyHiveProvider()

        Args:
            api_key: HoneyHive project API key with read access. Ingestion keys cannot read events.
                Falls back to the HH_API_KEY environment variable.
            api_url: HoneyHive data plane URL. Falls back to the HH_API_URL environment variable,
                then to the HoneyHive cloud URL.
            timeout: Request timeout in seconds.

        Raises:
            ProviderError: If no API key can be resolved.
        """
        resolved_key = api_key or os.environ.get("HH_API_KEY")
        if not resolved_key:
            raise ProviderError("HoneyHive API key required. Provide api_key or set HH_API_KEY.")
        resolved_url = api_url or os.environ.get("HH_API_URL") or "https://api.dp1.us.honeyhive.ai"
        self._client = httpx.Client(
            base_url=resolved_url.rstrip("/"),
            headers={"Authorization": f"Bearer {resolved_key}"},
            timeout=timeout,
        )

    def get_evaluation_data(self, session_id: str) -> TaskOutput:
        """Fetch a HoneyHive session and return its final output and trajectory."""
        try:
            events = self._fetch_session_events(session_id)
        except TraceProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"HoneyHive: failed to fetch events for session '{session_id}': {e}") from e

        if not events:
            raise SessionNotFoundError(f"HoneyHive: no events found for session_id='{session_id}'")

        session = self._build_session(session_id, events)
        if not session.traces:
            raise SessionNotFoundError(
                f"HoneyHive: session_id='{session_id}' has events but none are Strands agent spans"
            )
        return TaskOutput(output=self._extract_output(session), trajectory=session)

    # --- Internal: fetching ---

    def _fetch_session_events(self, session_id: str) -> list[dict[str, Any]]:
        """Fetch every event in a session, one page at a time, until the response `count` is reached."""
        retrier = Retrying(
            stop=stop_after_attempt(_MAX_RETRIES),
            wait=wait_exponential(multiplier=_RETRY_BACKOFF_BASE),
            retry=retry_if_exception_type(httpx.ReadTimeout),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        events: list[dict[str, Any]] = []
        page = 1
        while True:
            body = {
                "filters": [{"field": "session_id", "operator": "is", "value": session_id, "type": "string"}],
                "limit": _PAGE_SIZE,
                "page": page,
            }
            response = retrier(self._client.post, "/v1/events/search", json=body)
            if response.status_code in (401, 403):
                raise ProviderError(
                    "HoneyHive rejected the API key. Use a project API key with read access; "
                    "ingestion keys cannot read events."
                )
            response.raise_for_status()
            payload = response.json()
            events.extend(payload["events"])
            if not payload["events"] or len(events) >= payload["count"]:
                return events
            page += 1

    # --- Internal: building Session ---

    def _build_session(self, session_id: str, events: list[dict[str, Any]]) -> Session:
        """Group span events by OpenTelemetry trace ID. The session event is not a span."""
        by_trace: dict[str, list[SpanUnion]] = defaultdict(list)
        for event in events:
            span: SpanUnion
            name = event["event_name"]
            try:
                if name.startswith("invoke_agent"):
                    span = self._convert_agent_invocation(event, session_id)
                elif name.startswith("execute_tool"):
                    span = self._convert_tool_execution(event, session_id)
                elif event["event_type"] == "model":
                    span = self._convert_inference(event, session_id)
                else:
                    continue
            except Exception as e:
                logger.warning("event_id=<%s>, error=<%s> | failed to convert event", event.get("event_id"), e)
                continue
            by_trace[event["metadata"]["trace_id"]].append(span)

        traces = []
        for trace_id, spans in by_trace.items():
            spans.sort(key=lambda s: s.span_info.start_time)
            traces.append(Trace(trace_id=trace_id, session_id=session_id, spans=spans))
        traces.sort(key=lambda t: t.spans[0].span_info.start_time)
        return Session(session_id=session_id, traces=traces)

    def _span_info(self, event: dict[str, Any], session_id: str) -> SpanInfo:
        metadata = event["metadata"]
        return SpanInfo(
            trace_id=metadata["trace_id"],
            span_id=metadata["span_id"],
            session_id=session_id,
            # Strands TypeScript exports the root span with an empty parent ID. Trace finds root agents by `None`.
            parent_span_id=metadata.get("parent_span_id") or None,
            # HoneyHive stores times as Unix milliseconds.
            start_time=datetime.fromtimestamp(event["start_time"] / 1000, tz=timezone.utc),
            end_time=datetime.fromtimestamp(event["end_time"] / 1000, tz=timezone.utc),
        )

    # --- Internal: span converters ---

    def _convert_agent_invocation(self, event: dict[str, Any], session_id: str) -> AgentInvocationSpan:
        inputs, outputs = event["inputs"], event["outputs"]
        # Recent Strands Python versions send the prompt as a span event and the response as `outputs.result`.
        prompt = json.loads(event["metadata"].get("_event.gen_ai.user.message.0.content", '""'))
        history = inputs.get("chat_history", [{"role": "user", "content": prompt}])
        return AgentInvocationSpan(
            span_info=self._span_info(event, session_id),
            user_prompt=next((_text_of(m["content"]) for m in reversed(history) if m["role"] == "user"), ""),
            agent_response=outputs.get("content", outputs.get("result", "")).strip(),
            available_tools=[ToolConfig(name=name) for name in event["config"]["tools"]],
            metadata=event["metadata"],
        )

    def _convert_inference(self, event: dict[str, Any], session_id: str) -> InferenceSpan:
        """Convert a model event. The completion becomes the final assistant message."""
        history = [*event["inputs"]["chat_history"], {"role": "assistant", "content": event["outputs"]["content"]}]
        return InferenceSpan(
            span_info=self._span_info(event, session_id),
            messages=[m for m in map(_convert_message, history) if m is not None],
            metadata=event["metadata"],
        )

    def _convert_tool_execution(self, event: dict[str, Any], session_id: str) -> ToolExecutionSpan:
        """Convert an `execute_tool` event.

        Strands sends `inputs.parameters` and `outputs.result`. Spans that follow the latest
        OpenTelemetry GenAI conventions send `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result`.
        """
        inputs, outputs, metadata = event["inputs"], event["outputs"], event["metadata"]
        arguments = inputs.get("parameters", inputs.get("gen_ai.tool.call.arguments"))
        result = outputs.get("result", outputs.get("gen_ai.tool.call.result"))
        tool_call_id = outputs.get("tool_call_id") or metadata.get("tool_call_id")
        return ToolExecutionSpan(
            span_info=self._span_info(event, session_id),
            tool_call=ToolCall(
                name=event["config"]["tool_name"], arguments=_json_object(arguments), tool_call_id=tool_call_id
            ),
            tool_result=ToolResult(
                content=_tool_result_text(result), error=event.get("error") or None, tool_call_id=tool_call_id
            ),
            metadata=metadata,
        )

    def _extract_output(self, session: Session) -> str:
        """Return the last agent response in the session, or an empty string."""
        for trace in reversed(session.traces):
            for span in reversed(trace.spans):
                if isinstance(span, AgentInvocationSpan):
                    return span.agent_response
        return ""


# --- Internal: content helpers ---


def _text_of(content: str | list[dict[str, Any]]) -> str:
    """Flatten message content, a string or a list of content blocks, to text."""
    return content if isinstance(content, str) else "".join(block.get("text", "") for block in content)


def _json_object(value: Any) -> dict[str, Any]:
    """Return tool arguments as a dict. Exporters send them as an object or as a JSON string."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _tool_result_text(result: Any) -> str:
    """Flatten a tool result to text.

    Strands TypeScript serializes the result as a JSON list of content blocks, such as
    `[{"json": {"stock": 0}}]`. Text blocks contribute their text and JSON blocks their payload.
    """
    if result is None:
        return ""
    if not isinstance(result, str):
        return json.dumps(result)
    try:
        blocks = json.loads(result)
    except json.JSONDecodeError:
        return result
    # A tool can also return plain text that happens to parse as JSON, such as "100.0" or "[1, 2]".
    if (
        isinstance(blocks, list)
        and blocks
        and all(isinstance(b, dict) and ("text" in b or "json" in b) for b in blocks)
    ):
        return "".join(b["text"] if "text" in b else json.dumps(b["json"]) for b in blocks)
    return result


def _tool_result_content(result: dict[str, Any]) -> ToolResultContent:
    """Convert a Strands `toolResult` block. A failed tool reports `status` "error"."""
    text = _text_of(result["content"])
    return ToolResultContent(
        content=text, error=text if result["status"] == "error" else None, tool_call_id=result["toolUseId"]
    )


def _convert_message(raw: dict[str, Any]) -> UserMessage | AssistantMessage | None:
    """Convert a Strands chat message. System messages are dropped, since evals messages have no system role."""
    role, content = raw["role"], raw["content"]

    if role == "user":
        return UserMessage(content=[TextContent(text=_text_of(content))])

    if role == "tool":
        # A JSON `{"toolResult": {...}}`, or a JSON list of them for parallel tool calls.
        parsed = json.loads(content)
        return UserMessage(
            content=[_tool_result_content(r["toolResult"]) for r in (parsed if isinstance(parsed, list) else [parsed])]
        )

    if role == "assistant":
        if isinstance(content, str):
            return AssistantMessage(content=[TextContent(text=content)])
        blocks: list[TextContent | ToolCallContent] = []
        for block in content:
            # Strands Python nests a tool call as {"toolUse": {...}}. Strands TypeScript flattens it.
            tool_use = block.get("toolUse", block)
            if "toolUseId" in tool_use:
                blocks.append(
                    ToolCallContent(
                        name=tool_use["name"],
                        arguments=_json_object(tool_use["input"]),
                        tool_call_id=tool_use["toolUseId"],
                    )
                )
            elif block.get("text"):
                blocks.append(TextContent(text=block["text"]))
        return AssistantMessage(content=blocks)

    return None
