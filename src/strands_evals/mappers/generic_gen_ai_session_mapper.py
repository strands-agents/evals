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
from .utils import join_tool_result_content

logger = logging.getLogger(__name__)


class GenericGenAISessionMapper(SessionMapper):
    """Maps dict-format spans with gen_ai.* attributes to Session format.

    Classifies spans by gen_ai.operation.name:
    - "chat" → InferenceSpan
    - "execute_tool" → ToolExecutionSpan
    - "invoke_agent" → AgentInvocationSpan

    Supports both the legacy event convention (gen_ai.user.message, gen_ai.choice,
    gen_ai.tool.message) and the current unified convention
    (gen_ai.client.inference.operation.details with gen_ai.input.messages /
    gen_ai.output.messages), including producers running under
    OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental.

    Session filtering policy:
        Trace-level inclusion with span-level exclusion. A trace is included if at
        least one span matches the requested session_id (via gen_ai.conversation.id or
        session.id). Within an included trace, spans explicitly tagged with a *different*
        session are excluded; untagged spans inherit membership from the trace match.
        When no spans carry any session tag, all spans are included unconditionally.
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

        # Group by trace_id, optionally filtering by session.id.
        # Single pass: bucket spans by trace and track which traces match session_id.
        all_traces: dict[str, list[dict]] = defaultdict(list)
        traces_with_session: set[str] = set()
        any_span_has_session_id = False

        for span in spans:
            trace_id = span.get("trace_id", "unknown")
            all_traces[trace_id].append(span)
            attrs = span.get("attributes", {})
            span_session_id = attrs.get("gen_ai.conversation.id") or attrs.get("session.id")
            if span_session_id:
                any_span_has_session_id = True
                if str(span_session_id) == session_id:
                    traces_with_session.add(trace_id)

        traces_by_id: dict[str, list[dict]] = defaultdict(list)
        if not any_span_has_session_id:
            # No spans carry session tags — include everything.
            traces_by_id = all_traces
        else:
            # Include spans from matching traces; own tag wins over inheritance.
            for trace_id, trace_spans in all_traces.items():
                if trace_id not in traces_with_session:
                    continue
                for span in trace_spans:
                    attrs = span.get("attributes", {})
                    own = attrs.get("gen_ai.conversation.id") or attrs.get("session.id")
                    if own and str(own) != session_id:
                        continue
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
                    start_time=self.parse_timestamp(span.get("start_time")),
                    end_time=self.parse_timestamp(span.get("end_time")),
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
                    span.get("trace_id", "?"),
                    span.get("span_id", "?"),
                    e,
                )

        return Trace(spans=converted_spans, trace_id=trace_id, session_id=session_id)

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

        # Try span_events first (Strands telemetry / manual instrumentation format)
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

                elif event_name == "gen_ai.client.inference.operation.details":
                    # Current OTel GenAI convention: single event with input/output messages
                    # https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md
                    details_messages = self._convert_message_list_from_event(event_attrs)
                    messages.extend(details_messages)

            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    "span_id=%s event=%s | Failed to process event: %s",
                    span.get("span_id", "?"),
                    event.get("event_name", "?"),
                    e,
                )

        # Fallback: gen_ai.input.messages / gen_ai.output.messages as span attributes
        # (PydanticAI, AutoGen, and other native gen_ai semconv frameworks)
        if not messages:
            attrs = span.get("attributes", {})
            messages = self._convert_message_list_from_event(attrs)

        return InferenceSpan(span_info=span_info, messages=messages, metadata={})

    def _convert_tool_execution_span(self, span: dict, span_info: SpanInfo) -> ToolExecutionSpan | None:
        """Convert a span with gen_ai.operation.name='execute_tool' to ToolExecutionSpan."""
        attrs = span.get("attributes", {})

        tool_name = str(attrs.get("gen_ai.tool.name", ""))
        tool_call_id = str(attrs.get("gen_ai.tool.call.id", ""))

        # Prefer error.type (current OTel GenAI convention for failed operations),
        # fall back to gen_ai.tool.status / tool.status for legacy compatibility.
        # https://github.com/open-telemetry/semantic-conventions-genai/blob/main/model/gen-ai/spans.yaml#L522-L562
        error_type = attrs.get("error.type", "")
        status_obj = span.get("status") or {}
        span_status = status_obj.get("code", "") if isinstance(status_obj, dict) else ""
        tool_status = attrs.get("gen_ai.tool.status", attrs.get("tool.status", ""))

        if error_type:
            # OTel convention: error.type indicates the error class (e.g., "TimeoutError")
            tool_error = str(error_type)
        elif span_status == "ERROR":
            # Span status set to ERROR without error.type — still a failure
            tool_error = "ERROR"
        elif tool_status and tool_status != "success":
            # Legacy fallback: gen_ai.tool.status or tool.status
            tool_error = str(tool_status)
        else:
            tool_error = None

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
                elif event_name == "gen_ai.client.inference.operation.details":
                    # Current OTel GenAI convention: unified event with input/output messages.
                    # Extract tool arguments from input and tool result from output.
                    messages = self._convert_message_list_from_event(event_attrs)
                    for msg in messages:
                        if msg.role.value == "user":
                            for c in msg.content:
                                if hasattr(c, "text") and c.text and not tool_arguments:
                                    try:
                                        parsed = json.loads(c.text)
                                        if isinstance(parsed, dict):
                                            tool_arguments = parsed
                                    except (json.JSONDecodeError, TypeError):
                                        pass
                                elif hasattr(c, "content") and c.content and not tool_result_content:
                                    tool_result_content = c.content
                        elif msg.role.value == "assistant":
                            for c in msg.content:
                                if hasattr(c, "text") and c.text and not tool_result_content:
                                    tool_result_content = c.text
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    "span_id=%s | Failed to process tool event: %s",
                    span.get("span_id", "?"),
                    e,
                )

        # Fallback: gen_ai.tool.input/output attributes (manual instrumentation)
        if not tool_arguments:
            tool_input = attrs.get("gen_ai.tool.input", "") or attrs.get("gen_ai.tool.call.arguments", "")
            if tool_input:
                try:
                    parsed = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
                    tool_arguments = parsed if isinstance(parsed, dict) else {}
                except (json.JSONDecodeError, TypeError):
                    tool_arguments = {}

        if not tool_result_content:
            tool_output = attrs.get("gen_ai.tool.output", "") or attrs.get("gen_ai.tool.call.result", "")
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

        # Parse available tools from gen_ai.agent.tools or gen_ai.tool.definitions
        try:
            tools_json = attrs.get("gen_ai.agent.tools", "") or attrs.get("gen_ai.tool.definitions", "[]")
            tool_list = json.loads(str(tools_json)) if isinstance(tools_json, str) else tools_json
            if isinstance(tool_list, list):
                for item in tool_list:
                    if isinstance(item, str):
                        available_tools.append(ToolConfig(name=item))
                    elif isinstance(item, dict):
                        available_tools.append(ToolConfig(name=item.get("name", "")))
        except (json.JSONDecodeError, TypeError):
            pass

        # Extract from span_events (manual instrumentation format)
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
                elif event_name == "gen_ai.client.inference.operation.details":
                    # Current OTel GenAI convention: unified event with input/output messages.
                    # Extract user prompt from input messages and agent response from output.
                    messages = self._convert_message_list_from_event(event_attrs)
                    for msg in messages:
                        if msg.role.value == "user" and not user_prompt:
                            for c in msg.content:
                                if hasattr(c, "text") and c.text:
                                    user_prompt = c.text
                                    break
                        elif msg.role.value == "assistant" and not agent_response:
                            for c in msg.content:
                                if hasattr(c, "text") and c.text:
                                    agent_response = c.text
                                    break
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    "span_id=%s | Failed to process agent event: %s",
                    span.get("span_id", "?"),
                    e,
                )

        # Fallback: extract from attributes (PydanticAI / native gen_ai format)
        if not user_prompt:
            # Try pydantic_ai.all_messages or gen_ai.input.messages
            all_msgs_raw = attrs.get("pydantic_ai.all_messages", "") or attrs.get("gen_ai.input.messages", "")
            if all_msgs_raw:
                try:
                    all_msgs = json.loads(all_msgs_raw) if isinstance(all_msgs_raw, str) else all_msgs_raw
                    if isinstance(all_msgs, list):
                        for msg in all_msgs:
                            if msg.get("role") == "user":
                                parts = msg.get("parts", [])
                                for part in parts:
                                    if part.get("type") == "text" and part.get("content"):
                                        user_prompt = part["content"]
                                        break
                            if user_prompt:
                                break
                except (json.JSONDecodeError, TypeError):
                    pass

        if not agent_response:
            # Try gen_ai.output.messages (current OTel GenAI convention for invoke_agent)
            # https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md
            output_msgs_raw = attrs.get("gen_ai.output.messages", "")
            if output_msgs_raw:
                try:
                    output_msgs = json.loads(output_msgs_raw) if isinstance(output_msgs_raw, str) else output_msgs_raw
                    if isinstance(output_msgs, list):
                        for msg in output_msgs:
                            if msg.get("role") == "assistant":
                                parts = msg.get("parts", [])
                                for part in parts:
                                    if part.get("type") == "text" and part.get("content"):
                                        agent_response = part["content"]
                                        break
                                # Also handle flat content string
                                if not agent_response and msg.get("content"):
                                    agent_response = str(msg["content"])
                            if agent_response:
                                break
                except (json.JSONDecodeError, TypeError):
                    pass

        if not agent_response:
            agent_response = str(attrs.get("final_result", ""))

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

    def _convert_message_list_from_event(self, event_attrs: dict) -> list[UserMessage | AssistantMessage]:
        """Parse operation.details event attrs and return typed messages via _convert_message_list."""
        messages: list[UserMessage | AssistantMessage] = []
        for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
            raw = event_attrs.get(key, "")
            if not raw:
                continue
            try:
                msg_list = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(msg_list, list):
                messages.extend(self._convert_message_list(msg_list))
        return messages

    def _convert_message_list(self, msg_list: list[dict]) -> list[UserMessage | AssistantMessage]:
        """Convert a list of role-tagged message dicts to typed UserMessage/AssistantMessage.

        Shared logic for both span-attribute messages (gen_ai.input.messages /
        gen_ai.output.messages) and operation.details event payloads.

        Supports:
        - role "user": text parts and tool_call_response parts
        - role "tool": OTel GenAI convention tool results (parts-based or flat)
        - role "assistant": text parts, tool_call parts, or flat content string
        """
        messages: list[UserMessage | AssistantMessage] = []

        for msg in msg_list:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            parts = msg.get("parts", [])

            if role == "user":
                user_content: list[TextContent | ToolResultContent] = []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and part.get("content"):
                        user_content.append(TextContent(text=part["content"]))
                    elif part.get("type") == "tool_call_response":
                        _r = part.get("response", part.get("result", ""))
                        if isinstance(_r, list):
                            _r = join_tool_result_content(_r)
                        elif isinstance(_r, dict):
                            _r = json.dumps(_r)
                        user_content.append(
                            ToolResultContent(
                                content=str(_r) if _r else "",
                                error=None,
                                tool_call_id=part.get("id"),
                            )
                        )
                if user_content:
                    messages.append(UserMessage(content=user_content))

            elif role == "tool":
                # OTel GenAI convention: role "tool" carries tool results.
                # Two formats:
                #   1. parts: [{"type": "tool_call_response", "id": "...", "response": "..."}]
                #   2. flat: {"role": "tool", "id": "...", "response": "..."}
                if parts:
                    for part in parts:
                        if isinstance(part, dict) and part.get("type") == "tool_call_response":
                            tc_id = part.get("id") or part.get("tool_call_id")
                            resp = part.get("response", part.get("result", ""))
                            if isinstance(resp, dict):
                                resp = json.dumps(resp)
                            elif isinstance(resp, list):
                                resp = join_tool_result_content(resp)
                            messages.append(
                                UserMessage(
                                    content=[
                                        ToolResultContent(
                                            content=str(resp) if resp else "",
                                            error=None,
                                            tool_call_id=tc_id,
                                        )
                                    ]
                                )
                            )
                else:
                    tool_call_id = msg.get("id") or msg.get("tool_call_id")
                    response = msg.get("response", msg.get("content", ""))
                    if isinstance(response, dict):
                        response = json.dumps(response)
                    elif isinstance(response, list):
                        response = join_tool_result_content(response)
                    messages.append(
                        UserMessage(
                            content=[
                                ToolResultContent(
                                    content=str(response) if response else "",
                                    error=None,
                                    tool_call_id=tool_call_id,
                                )
                            ]
                        )
                    )

            elif role == "assistant":
                assistant_content: list[TextContent | ToolCallContent] = []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and part.get("content"):
                        assistant_content.append(TextContent(text=part["content"]))
                    elif part.get("type") == "tool_call":
                        assistant_content.append(
                            ToolCallContent(
                                name=part.get("name", ""),
                                arguments=part.get("arguments", {}),
                                tool_call_id=part.get("id"),
                            )
                        )
                # Also handle flat content string (no parts)
                if not assistant_content and msg.get("content"):
                    assistant_content.append(TextContent(text=str(msg["content"])))
                if assistant_content:
                    messages.append(AssistantMessage(content=assistant_content))

        return messages

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
                    result_text = join_tool_result_content(result_text)
                result.append(
                    ToolResultContent(
                        content=str(result_text),
                        error=tool_result.get("error"),
                        tool_call_id=tool_result.get("toolUseId"),
                    )
                )
        return result
