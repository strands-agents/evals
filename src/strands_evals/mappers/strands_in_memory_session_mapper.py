import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

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


class StrandsInMemorySessionMapper(SessionMapper):
    """Maps OpenTelemetry in-memory spans to Session format for evaluation."""

    def map_to_session(self, otel_spans: list[ReadableSpan], session_id: str) -> Session:
        """
        Map OTEL spans to Session format.

        Args:
            otel_spans: List of OpenTelemetry ReadableSpan objects
            session_id: Session identifier

        Returns:
            Session object ready for evaluation
        """
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

    def _convert_trace(self, trace_id: str, otel_spans: list[ReadableSpan], session_id: str) -> Trace:
        """
        Convert a list of OTEL spans with the same trace_id into a Trace.

        Args:
            trace_id: Trace identifier (32-char hex string)
            otel_spans: List of OpenTelemetry ReadableSpan objects with this trace_id
            session_id: Session identifier

        Returns:
            Trace object containing converted spans
        """
        converted_spans: list[InferenceSpan | ToolExecutionSpan | AgentInvocationSpan] = []

        for span in otel_spans:
            operation_name = span.attributes.get("gen_ai.operation.name", "")

            if operation_name == "chat":
                inference_span = self._convert_inference_span(span, session_id)
                if inference_span.messages:
                    converted_spans.append(inference_span)
            elif operation_name == "execute_tool":
                converted_spans.append(self._convert_tool_execution_span(span, session_id))
            elif operation_name == "invoke_agent":
                converted_spans.append(self._convert_agent_invocation_span(span, session_id))

        return Trace(spans=converted_spans, trace_id=trace_id, session_id=session_id)

    def _convert_inference_span(self, span: ReadableSpan, session_id: str) -> InferenceSpan:
        """
        Convert an OTEL chat span to an InferenceSpan.

        Args:
            span: OpenTelemetry ReadableSpan with operation.name = "chat"
            session_id: Session identifier

        Returns:
            InferenceSpan with structured message history including tool calls and results
        """
        span_info = SpanInfo(
            trace_id=format(span.context.trace_id, "032x"),
            span_id=format(span.context.span_id, "016x"),
            session_id=session_id,
            parent_span_id=format(span.parent.span_id, "016x") if span.parent else None,
            start_time=datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc),
            end_time=datetime.fromtimestamp(span.end_time / 1e9, tz=timezone.utc),
        )

        messages: list[UserMessage | AssistantMessage] = []

        for event in span.events:
            if event.name == "gen_ai.user.message":
                content_str = event.attributes.get("content", "[]")
                content_list = json.loads(content_str)
                user_content = [TextContent(text=item["text"]) for item in content_list if "text" in item]
                if user_content:
                    messages.append(UserMessage(content=user_content))

            elif event.name == "gen_ai.assistant.message":
                content_str = event.attributes.get("content", "[]")
                content_list = json.loads(content_str)
                assistant_content: list[TextContent | ToolCallContent] = []

                for item in content_list:
                    if "text" in item:
                        assistant_content.append(TextContent(text=item["text"]))
                    elif "toolUse" in item:
                        tool_use = item["toolUse"]
                        assistant_content.append(
                            ToolCallContent(
                                name=tool_use["name"],
                                arguments=tool_use.get("input", {}),
                                tool_call_id=tool_use.get("toolUseId"),
                            )
                        )

                if assistant_content:
                    messages.append(AssistantMessage(content=assistant_content))

            elif event.name == "gen_ai.tool.message":
                content_str = event.attributes.get("content", "[]")
                content_list = json.loads(content_str)
                tool_result_content: list[ToolResultContent] = []

                for item in content_list:
                    if "toolResult" in item:
                        tool_result = item["toolResult"]
                        result_text = ""
                        if "content" in tool_result and tool_result["content"]:
                            result_text = (
                                tool_result["content"][0].get("text", "")
                                if isinstance(tool_result["content"], list)
                                else str(tool_result["content"])
                            )

                        tool_result_content.append(
                            ToolResultContent(
                                content=result_text,
                                error=tool_result.get("error"),
                                tool_call_id=tool_result.get("toolUseId"),
                            )
                        )

                if tool_result_content:
                    messages.append(UserMessage(content=tool_result_content))

            elif event.name == "gen_ai.choice":
                message_str = event.attributes.get("message", "[]")
                message_list = json.loads(message_str)
                assistant_content: list[TextContent | ToolCallContent] = []

                for item in message_list:
                    if "text" in item:
                        assistant_content.append(TextContent(text=item["text"]))
                    elif "toolUse" in item:
                        tool_use = item["toolUse"]
                        assistant_content.append(
                            ToolCallContent(
                                name=tool_use["name"],
                                arguments=tool_use.get("input", {}),
                                tool_call_id=tool_use.get("toolUseId"),
                            )
                        )

                if assistant_content:
                    messages.append(AssistantMessage(content=assistant_content))

        return InferenceSpan(span_info=span_info, messages=messages, metadata={})

    def _convert_tool_execution_span(self, span: ReadableSpan, session_id: str) -> ToolExecutionSpan:
        """
        Convert an OTEL execute_tool span to a ToolExecutionSpan.

        Args:
            span: OpenTelemetry ReadableSpan with operation.name = "execute_tool"
            session_id: Session identifier

        Returns:
            ToolExecutionSpan with tool call and result information
        """
        span_info = SpanInfo(
            trace_id=format(span.context.trace_id, "032x"),
            span_id=format(span.context.span_id, "016x"),
            session_id=session_id,
            parent_span_id=format(span.parent.span_id, "016x") if span.parent else None,
            start_time=datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc),
            end_time=datetime.fromtimestamp(span.end_time / 1e9, tz=timezone.utc),
        )

        tool_name = span.attributes.get("gen_ai.tool.name", "")
        tool_call_id = span.attributes.get("gen_ai.tool.call.id", "")
        tool_status = span.attributes.get("tool.status", "")

        tool_arguments = {}
        tool_result_content = ""
        tool_error = None if tool_status == "success" else tool_status

        for event in span.events:
            if event.name == "gen_ai.tool.message":
                content_str = event.attributes.get("content", "{}")
                tool_arguments = json.loads(content_str)

            elif event.name == "gen_ai.choice":
                message_str = event.attributes.get("message", "[]")
                message_list = json.loads(message_str)
                tool_result_content = message_list[0].get("text", "") if message_list else ""

        tool_call = ToolCall(name=tool_name, arguments=tool_arguments, tool_call_id=tool_call_id)

        tool_result = ToolResult(content=tool_result_content, error=tool_error, tool_call_id=tool_call_id)

        return ToolExecutionSpan(span_info=span_info, tool_call=tool_call, tool_result=tool_result, metadata={})

    def _convert_agent_invocation_span(self, span: ReadableSpan, session_id: str) -> AgentInvocationSpan:
        """
        Convert an OTEL invoke_agent span to an AgentInvocationSpan.

        Args:
            span: OpenTelemetry ReadableSpan with operation.name = "invoke_agent"
            session_id: Session identifier

        Returns:
            AgentInvocationSpan with user prompt, agent response, and available tools
        """
        span_info = SpanInfo(
            trace_id=format(span.context.trace_id, "032x"),
            span_id=format(span.context.span_id, "016x"),
            session_id=session_id,
            parent_span_id=format(span.parent.span_id, "016x") if span.parent else None,
            start_time=datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc),
            end_time=datetime.fromtimestamp(span.end_time / 1e9, tz=timezone.utc),
        )

        user_prompt = ""
        agent_response = ""
        available_tools: list[ToolConfig] = []

        # Extract available tools from span attributes
        tools_str = span.attributes.get("gen_ai.agent.tools", "[]")
        tool_names = json.loads(tools_str)
        available_tools = [ToolConfig(name=name) for name in tool_names]

        for event in span.events:
            if event.name == "gen_ai.user.message":
                content_str = event.attributes.get("content", "[]")
                content_list = json.loads(content_str)
                user_prompt = content_list[0].get("text", "") if content_list else ""

            elif event.name == "gen_ai.choice":
                agent_response = event.attributes.get("message", "")

        return AgentInvocationSpan(
            span_info=span_info,
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=available_tools,
            metadata={},
        )
