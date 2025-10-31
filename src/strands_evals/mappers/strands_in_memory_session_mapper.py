import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan

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


class GenAIConventionVersion(Enum):
    """GenAI semantic convention versions following OTEL_SEMCONV_STABILITY_OPT_IN.

    This enum aligns with OpenTelemetry's semantic convention stability options
    as defined in OTEL_SEMCONV_STABILITY_OPT_IN environment variable.

    Attributes:
        LEGACY: Use legacy conventions (v1.36.0 or prior) with gen_ai.system attribute
            and separate message events (gen_ai.user.message, gen_ai.choice, etc.)
        LATEST_EXPERIMENTAL: Use latest experimental conventions (v1.37+) with
            gen_ai.provider.name attribute and unified gen_ai.client.inference.operation.details events.
            Corresponds to OTEL's "gen_ai_latest_experimental" stability option.
    """

    LEGACY = "legacy"
    LATEST_EXPERIMENTAL = "gen_ai_latest_experimental"


class StrandsInMemorySessionMapper(SessionMapper):
    """Maps OpenTelemetry in-memory spans to Session format for evaluation.

    Supports both legacy and latest GenAI semantic conventions:
    - Latest (v1.37+): gen_ai.provider.name with unified gen_ai.client.inference.operation.details events
    - Legacy: gen_ai.system with separate message events (gen_ai.user.message, gen_ai.choice, etc.)

    The mapper automatically detects the convention version. Default to Legacy.
    """

    def __init__(self):
        super().__init__()
        self._convention_version = GenAIConventionVersion.LEGACY

    def map_to_session(self, otel_spans: list[ReadableSpan], session_id: str) -> Session:
        if otel_spans:
            self._convention_version = self._detect_convention_version(otel_spans[0])

        traces_by_id = defaultdict(list)
        for span in otel_spans:
            trace_id = format(span.context.trace_id, "032x")
            traces_by_id[trace_id].append(span)

        traces: list[Trace] = []
        for trace_id, spans in traces_by_id.items():
            trace = self._convert_trace(trace_id, spans, session_id)
            if trace.spans:
                traces.append(trace)

        return Session(traces=traces, session_id=session_id)

    def _detect_convention_version(self, span: ReadableSpan) -> GenAIConventionVersion:
        """Detect which GenAI semantic convention version is being used.

        Returns:
            GenAIConventionVersion.LATEST_EXPERIMENTAL if using latest conventions,
            GenAIConventionVersion.LEGACY otherwise
        """
        if span.attributes and "gen_ai.provider.name" in span.attributes:
            return GenAIConventionVersion.LATEST_EXPERIMENTAL

        return GenAIConventionVersion.LEGACY

    def _use_latest_conventions(self) -> bool:
        """Helper method to determine if latest conventions should be used.

        Returns:
            True if LATEST_EXPERIMENTAL, False if LEGACY
        """
        return self._convention_version == GenAIConventionVersion.LATEST_EXPERIMENTAL

    def _convert_trace(self, trace_id: str, otel_spans: list[ReadableSpan], session_id: str) -> Trace:
        converted_spans: list[InferenceSpan | ToolExecutionSpan | AgentInvocationSpan] = []

        for span in otel_spans:
            try:
                operation_name = span.attributes.get("gen_ai.operation.name", "") if span.attributes else ""

                if operation_name == "chat":
                    inference_span = self._convert_inference_span(span, session_id)
                    if inference_span.messages:
                        converted_spans.append(inference_span)
                elif operation_name == "execute_tool":
                    converted_spans.append(self._convert_tool_execution_span(span, session_id))
                elif operation_name == "invoke_agent":
                    converted_spans.append(self._convert_agent_invocation_span(span, session_id))
            except Exception as e:
                logger.warning(f"Failed to convert span: {e}")

        return Trace(spans=converted_spans, trace_id=trace_id, session_id=session_id)

    def _create_span_info(self, span: ReadableSpan, session_id: str) -> SpanInfo:
        start_time = span.start_time or 0
        end_time = span.end_time or 0

        return SpanInfo(
            trace_id=format(span.context.trace_id, "032x"),
            span_id=format(span.context.span_id, "016x"),
            session_id=session_id,
            parent_span_id=format(span.parent.span_id, "016x") if span.parent else None,
            start_time=datetime.fromtimestamp(start_time / 1e9, tz=timezone.utc),
            end_time=datetime.fromtimestamp(end_time / 1e9, tz=timezone.utc),
        )

    def _parse_json_attr(self, attributes: Any, key: str, default: str = "[]") -> Any:
        try:
            value = attributes.get(key, default)
            return json.loads(str(value))
        except (AttributeError, TypeError, json.JSONDecodeError):
            return json.loads(default)

    def _process_user_message(self, content_list: list[dict[str, Any]]) -> list[TextContent | ToolResultContent]:
        return [TextContent(text=item["text"]) for item in content_list if "text" in item]

    def _process_assistant_content(self, content_list: list[dict[str, Any]]) -> list[TextContent | ToolCallContent]:
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
            result_text = ""
            if "content" in tool_result and tool_result["content"]:
                content = tool_result["content"]
                result_text = content[0].get("text", "") if isinstance(content, list) else str(content)

            result.append(
                ToolResultContent(
                    content=result_text,
                    error=tool_result.get("error"),
                    tool_call_id=tool_result.get("toolUseId"),
                )
            )
        return result

    def _convert_inference_span(self, span: ReadableSpan, session_id: str) -> InferenceSpan:
        span_info = self._create_span_info(span, session_id)

        if self._use_latest_conventions():
            messages = self._extract_messages_from_inference_details(span)
        else:
            messages = self._extract_messages_from_events(span)

        return InferenceSpan(span_info=span_info, messages=messages, metadata={})

    def _extract_messages_from_events(self, span: ReadableSpan) -> list[UserMessage | AssistantMessage]:
        """Extract messages from legacy event format (gen_ai.user.message, etc.)."""
        messages: list[UserMessage | AssistantMessage] = []

        for event in span.events:
            try:
                if event.name == "gen_ai.user.message":
                    content_list = self._parse_json_attr(event.attributes, "content")
                    user_content = self._process_user_message(content_list)
                    if user_content:
                        messages.append(UserMessage(content=user_content))

                elif event.name == "gen_ai.assistant.message":
                    content_list = self._parse_json_attr(event.attributes, "content")
                    assistant_content = self._process_assistant_content(content_list)
                    if assistant_content:
                        messages.append(AssistantMessage(content=assistant_content))

                elif event.name == "gen_ai.tool.message":
                    content_list = self._parse_json_attr(event.attributes, "content")
                    tool_result_content = self._process_tool_results(content_list)
                    if tool_result_content:
                        messages.append(UserMessage(content=tool_result_content))

                elif event.name == "gen_ai.choice":
                    message_list = self._parse_json_attr(event.attributes, "message")
                    assistant_content = self._process_assistant_content(message_list)
                    if assistant_content:
                        messages.append(AssistantMessage(content=assistant_content))
            except Exception as e:
                logger.warning(f"Failed to process event {event.name}: {e}")

        return messages

    def _extract_messages_from_inference_details(self, span: ReadableSpan) -> list[UserMessage | AssistantMessage]:
        """Extract messages from latest event format (gen_ai.client.inference.operation.details)."""
        messages: list[UserMessage | AssistantMessage] = []

        for event in span.events:
            try:
                if event.name == "gen_ai.client.inference.operation.details":
                    event_attributes = event.attributes
                    if not event_attributes:
                        continue
                    # Check for input messages
                    if "gen_ai.input.messages" in event_attributes:
                        input_messages = self._parse_json_attr(event_attributes, "gen_ai.input.messages")
                        for msg in input_messages:
                            input_content = self._convert_inference_messages(msg)
                            if input_content:
                                messages.append(input_content)

                    # Check for output messages
                    if "gen_ai.output.messages" in event_attributes:
                        output_messages = self._parse_json_attr(event_attributes, "gen_ai.output.messages")
                        for msg in output_messages:
                            output_content = self._convert_inference_messages(msg)
                            if output_content:
                                messages.append(output_content)
            except Exception as e:
                logger.warning(f"Failed to process inference details event: {e}")

        return messages

    def _convert_inference_messages(self, otel_msg: dict[str, Any]) -> UserMessage | AssistantMessage | None:
        """Convert OTEL message format (with parts) to internal message types.

        Args:
            otel_msg: Message in OTEL format with 'role' and 'parts' fields

        Returns:
            UserMessage or AssistantMessage, or None if conversion fails
        """
        try:
            role = otel_msg.get("role", "")
            parts = otel_msg.get("parts", [])

            if role == "assistant":
                assistant_content: list[TextContent | ToolCallContent] = []

                for part in parts:
                    part_type = part.get("type", "")

                    if part_type == "text":
                        assistant_content.append(TextContent(text=part.get("content", "")))

                    elif part_type == "tool_call":
                        assistant_content.append(
                            ToolCallContent(
                                name=part.get("name", ""),
                                arguments=part.get("arguments", {}),
                                tool_call_id=part.get("id"),
                            )
                        )
                return AssistantMessage(content=assistant_content) if assistant_content else None

            # Tool messages are represented as UserMessage with ToolResultContent
            content: list[TextContent | ToolResultContent] = []

            for part in parts:
                part_type = part.get("type", "")

                if part_type == "text":
                    content.append(TextContent(text=part.get("content", "")))

                if part_type == "tool_call_response":
                    # Extract text from response array if present
                    response = part.get("response", [])
                    response_text = ""

                    ## To-do: Compare the differences for multiple toolResults
                    if isinstance(response, list) and response:
                        response_text = (
                            response[0].get("text", "") if isinstance(response[0], dict) else str(response[0])
                        )
                    elif isinstance(response, str):
                        response_text = response

                    content.append(
                        ToolResultContent(
                            content=response_text,
                            tool_call_id=part.get("id"),
                        )
                    )
            return UserMessage(content=content) if content else None

        except Exception as e:
            logger.warning(f"Failed to convert OTEL message: {e}")
            return None

    def _convert_tool_execution_span(self, span: ReadableSpan, session_id: str) -> ToolExecutionSpan:
        span_info = self._create_span_info(span, session_id)
        attrs = span.attributes or {}

        tool_name = str(attrs.get("gen_ai.tool.name", ""))
        tool_call_id = str(attrs.get("gen_ai.tool.call.id", ""))
        tool_status = attrs.get("gen_ai.tool.status", attrs.get("tool.status", ""))
        tool_error = None if tool_status == "success" else (str(tool_status) if tool_status else None)

        tool_arguments = {}
        tool_result_content = ""

        if self._use_latest_conventions():
            # Extract from gen_ai.client.inference.operation.details events
            for event in span.events:
                try:
                    if event.name == "gen_ai.client.inference.operation.details":
                        event_attributes = event.attributes
                        if not event_attributes:
                            continue
                        if "gen_ai.input.messages" in event_attributes:
                            input_messages = self._parse_json_attr(event_attributes, "gen_ai.input.messages")
                            if input_messages and input_messages[0].get("parts"):
                                part = input_messages[0]["parts"][0]
                                if part.get("type") == "tool_call":
                                    tool_arguments = part.get("arguments", {})

                        if "gen_ai.output.messages" in event_attributes:
                            output_messages = self._parse_json_attr(event_attributes, "gen_ai.output.messages")
                            if output_messages and output_messages[0].get("parts"):
                                part = output_messages[0]["parts"][0]
                                if part.get("type") == "tool_call_response":
                                    response = part.get("response", [])
                                    if isinstance(response, list) and response:
                                        tool_result_content = (
                                            response[0].get("text", "")
                                            if isinstance(response[0], dict)
                                            else str(response[0])
                                        )
                                    elif isinstance(response, str):
                                        tool_result_content = response
                except Exception as e:
                    logger.warning(f"Failed to process tool event {event.name}: {e}")
        else:
            for event in span.events:
                try:
                    event_attributes = event.attributes
                    if not event_attributes:
                        continue
                    if event.name == "gen_ai.tool.message":
                        tool_arguments = self._parse_json_attr(event_attributes, "content", "{}")
                    elif event.name == "gen_ai.choice":
                        message_list = self._parse_json_attr(event_attributes, "message")
                        tool_result_content = message_list[0].get("text", "") if message_list else ""
                except Exception as e:
                    logger.warning(f"Failed to process tool event {event.name}: {e}")

        tool_call = ToolCall(name=tool_name, arguments=tool_arguments, tool_call_id=tool_call_id)
        tool_result = ToolResult(content=tool_result_content, error=tool_error, tool_call_id=tool_call_id)

        return ToolExecutionSpan(span_info=span_info, tool_call=tool_call, tool_result=tool_result, metadata={})

    def _convert_agent_invocation_span(self, span: ReadableSpan, session_id: str) -> AgentInvocationSpan:
        span_info = self._create_span_info(span, session_id)

        user_prompt = ""
        agent_response = ""
        available_tools: list[ToolConfig] = []

        try:
            tool_names = self._parse_json_attr(span.attributes, "gen_ai.agent.tools")
            available_tools = [ToolConfig(name=name) for name in tool_names]
        except Exception as e:
            logger.warning(f"Failed to parse available tools: {e}")

        if self._use_latest_conventions():
            for event in span.events:
                try:
                    if event.name == "gen_ai.client.inference.operation.details":
                        event_attributes = event.attributes
                        if not event_attributes:
                            continue
                        if "gen_ai.input.messages" in event_attributes:
                            input_messages = self._parse_json_attr(event_attributes, "gen_ai.input.messages")
                            if input_messages and input_messages[0].get("parts"):
                                parts = input_messages[0]["parts"]
                                for part in parts:
                                    if part.get("type") == "text":
                                        user_prompt = part.get("content", "")
                                        break

                        if "gen_ai.output.messages" in event_attributes:
                            output_messages = self._parse_json_attr(event_attributes, "gen_ai.output.messages")
                            if output_messages and output_messages[0].get("parts"):
                                parts = output_messages[0]["parts"]
                                for part in parts:
                                    if part.get("type") == "text":
                                        agent_response = part.get("content", "")
                                        break
                except Exception as e:
                    logger.warning(f"Failed to process agent event {event.name}: {e}")
        else:
            for event in span.events:
                try:
                    event_attributes = event.attributes
                    if not event_attributes:
                        continue
                    if event.name == "gen_ai.user.message":
                        content_list = self._parse_json_attr(event_attributes, "content")
                        user_prompt = content_list[0].get("text", "") if content_list else ""
                    elif event.name == "gen_ai.choice":
                        msg = event_attributes.get("message", "") if event_attributes else ""
                        agent_response = str(msg)
                except Exception as e:
                    logger.warning(f"Failed to process agent event {event.name}: {e}")

        return AgentInvocationSpan(
            span_info=span_info,
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=available_tools,
            metadata={},
        )
