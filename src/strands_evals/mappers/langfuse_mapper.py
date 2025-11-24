"""Convert Langfuse TraceWithFullDetails to Strands Session object."""

import json
import logging
from typing import Any

from langfuse.api.resources.commons.types.trace_with_full_details import (  # type: ignore[import-not-found]
    TraceWithFullDetails,
)

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


class LangfuseSessionMapper(SessionMapper):
    """Maps Langfuse TraceWithFullDetails to Session format for evaluation."""

    def map_to_session(self, spans: list[TraceWithFullDetails], session_id: str) -> Session:
        """Map Langfuse traces to Session format."""
        traces = []
        for langfuse_trace in spans:
            try:
                trace = self._convert_trace(langfuse_trace)
                if trace.spans:
                    traces.append(trace)
            except Exception as e:
                logger.warning(f"Failed to convert trace {langfuse_trace.id}: {e}")

        return Session(traces=traces, session_id=session_id)

    def get_agent_final_response(self, session: Session) -> str:
        """Get the final response from the agent."""
        final_response = ""
        for trace in session.traces:
            for span in trace.spans:
                if isinstance(span, AgentInvocationSpan):
                    final_response = span.agent_response
        return final_response

    def _convert_trace(self, langfuse_trace: TraceWithFullDetails) -> Trace:
        """Convert Langfuse TraceWithFullDetails to Strands Trace."""
        spans: list[InferenceSpan | ToolExecutionSpan | AgentInvocationSpan] = []
        sorted_observations = sorted(
            langfuse_trace.observations,
            key=lambda obs: obs.start_time,
        )

        for obs in sorted_observations:
            try:
                span_info = self._create_span_info(obs, langfuse_trace)
                operation_name = self._get_operation_name(obs)

                if operation_name == "chat":
                    span = self._convert_inference_span(obs, span_info)
                    if span.messages:
                        spans.append(span)
                elif operation_name == "execute_tool":
                    spans.append(self._convert_tool_execution_span(obs, span_info))
                elif operation_name == "invoke_agent":
                    spans.append(self._convert_agent_invocation_span(obs, span_info))
            except Exception as e:
                logger.warning(f"Failed to convert span {obs.id}: {e}")

        return Trace(
            spans=spans,
            trace_id=langfuse_trace.id,
            session_id=langfuse_trace.session_id or langfuse_trace.id,
        )

    def _create_span_info(self, obs: Any, langfuse_trace: TraceWithFullDetails) -> SpanInfo:
        """Create SpanInfo from observation."""
        return SpanInfo(
            trace_id=langfuse_trace.id,
            span_id=obs.id,
            session_id=langfuse_trace.session_id or langfuse_trace.id,
            parent_span_id=obs.parent_observation_id,
            start_time=obs.start_time,
            end_time=obs.end_time or obs.start_time,
        )

    def _get_operation_name(self, obs: Any) -> str:
        """Extract operation name from observation metadata."""
        try:
            return obs.metadata["attributes"].get("gen_ai.operation.name", "") if obs.metadata else ""
        except (AttributeError, TypeError, KeyError):
            return ""

    def _parse_json_attr(self, value: str, default: str = "[]") -> Any:
        """Parse JSON attribute with error handling."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return json.loads(default)

    def _extract_text_from_content(self, content_list: list[dict[str, Any]]) -> str:
        """Extract text from content list."""
        for content_item in content_list:
            if "text" in content_item:
                return content_item["text"]
        return ""

    def _parse_langfuse_messages(self, input_data: list[dict[str, Any]]) -> list[UserMessage | AssistantMessage]:
        """Parse Langfuse message format to Strands message format."""
        messages: list[UserMessage | AssistantMessage] = []

        for item in input_data:
            try:
                role = item.get("role", "")
                content_str = item.get("content", "")

                content_list = self._parse_json_attr(content_str, '[{"text": ""}]')

                if role == "user":
                    user_content = self._process_user_content(content_list)
                    if user_content:
                        messages.append(UserMessage(content=user_content))

                elif role == "assistant":
                    assistant_content = self._process_assistant_content(content_list)
                    if assistant_content:
                        messages.append(AssistantMessage(content=assistant_content))

                elif role == "tool":
                    tool_content = self._process_tool_content(content_list)
                    if tool_content:
                        messages.append(UserMessage(content=list(tool_content)))
            except Exception as e:
                logger.warning(f"Failed to parse message with role {item.get('role', 'unknown')}: {e}")

        return messages

    def _process_user_content(self, content_list: list[dict[str, Any]]) -> list[TextContent | ToolResultContent]:
        """Process user message content."""
        user_content: list[TextContent | ToolResultContent] = []
        for content_item in content_list:
            if "text" in content_item:
                user_content.append(TextContent(text=content_item["text"]))
            elif "toolResult" in content_item:
                tool_result = content_item["toolResult"]
                result_text = self._extract_tool_result_text(tool_result)
                user_content.append(
                    ToolResultContent(
                        content=result_text,
                        tool_call_id=tool_result.get("toolUseId"),
                    )
                )
        return user_content

    def _process_assistant_content(self, content_list: list[dict[str, Any]]) -> list[TextContent | ToolCallContent]:
        """Process assistant message content."""
        assistant_content: list[TextContent | ToolCallContent] = []
        for content_item in content_list:
            if "text" in content_item:
                assistant_content.append(TextContent(text=content_item["text"]))
            elif "toolUse" in content_item:
                tool_use = content_item["toolUse"]
                assistant_content.append(
                    ToolCallContent(
                        name=tool_use["name"],
                        arguments=tool_use.get("input", {}),
                        tool_call_id=tool_use.get("toolUseId"),
                    )
                )
        return assistant_content

    def _process_tool_content(self, content_list: list[dict[str, Any]]) -> list[ToolResultContent]:
        """Process tool message content."""
        tool_content = []
        for content_item in content_list:
            if "toolResult" in content_item:
                tool_result = content_item["toolResult"]
                result_text = self._extract_tool_result_text(tool_result)
                tool_content.append(
                    ToolResultContent(
                        content=result_text,
                        tool_call_id=tool_result.get("toolUseId"),
                    )
                )
        return tool_content

    def _extract_tool_result_text(self, tool_result: dict[str, Any]) -> str:
        """Extract text from tool result."""
        if "content" in tool_result and tool_result["content"]:
            content = tool_result["content"]
            if isinstance(content, list) and content:
                return content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
            return str(content)
        return ""

    def _parse_output_messages(self, output: dict[str, Any]) -> list[AssistantMessage | UserMessage]:
        """Parse output messages from inference span."""
        messages: list[AssistantMessage | UserMessage] = []

        if "message" in output:
            content_list = self._parse_json_attr(output["message"])
            assistant_content = self._process_assistant_content(content_list)
            if assistant_content:
                messages.append(AssistantMessage(content=assistant_content))

        if "tool.result" in output:
            content_list = self._parse_json_attr(output["tool.result"])
            user_content = self._process_user_content(content_list)
            if user_content:
                messages.append(UserMessage(content=user_content))

        return messages

    def _convert_inference_span(self, obs: Any, span_info: SpanInfo) -> InferenceSpan:
        """Convert observation to InferenceSpan."""
        messages = []
        if obs.input:
            messages.extend(self._parse_langfuse_messages(obs.input))

        if obs.output:
            messages.extend(self._parse_output_messages(obs.output))
        return InferenceSpan(span_info=span_info, messages=messages)

    def _convert_tool_execution_span(self, obs: Any, span_info: SpanInfo) -> ToolExecutionSpan:
        """Convert observation to ToolExecutionSpan."""

        tool_arguments = {}

        try:
            tool_arguments = json.loads(obs.input[0]["content"])
        except Exception as e:
            logger.warning(f"Failed to extract tool arguments: {e}")

        tool_call = ToolCall(
            name=obs.name or "unknown_tool",
            arguments=tool_arguments,
            tool_call_id=obs.id,
        )

        tool_content = self._extract_tool_output(obs.output)
        tool_result = ToolResult(content=tool_content, tool_call_id=obs.id)

        return ToolExecutionSpan(span_info=span_info, tool_call=tool_call, tool_result=tool_result)

    def _extract_tool_output(self, output: Any) -> str:
        """Extract tool output content."""
        if not output:
            return ""

        try:
            if isinstance(output, dict) and "message" in output:
                content_list = self._parse_json_attr(output["message"])
                return self._extract_text_from_content(content_list)
            return str(output)
        except Exception as e:
            logger.warning(f"Failed to extract tool output: {e}")
            return str(output) if output else ""

    def _convert_agent_invocation_span(self, obs: Any, span_info: SpanInfo) -> AgentInvocationSpan:
        """Convert observation to AgentInvocationSpan."""
        user_prompt = self._extract_user_prompt(obs.input)
        agent_response = self._extract_agent_response(obs.output)
        available_tools = self._extract_available_tools(obs.metadata)

        return AgentInvocationSpan(
            span_info=span_info,
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=available_tools,
        )

    def _extract_user_prompt(self, input_data: Any) -> str:
        """Extract user prompt from input data."""
        if not input_data:
            return ""

        try:
            for item in input_data:
                if item.get("role") == "user":
                    content_list = self._parse_json_attr(item.get("content", "[]"))
                    return self._extract_text_from_content(content_list)
        except Exception as e:
            logger.warning(f"Failed to extract user prompt: {e}")
        return ""

    def _extract_agent_response(self, output: Any) -> str:
        """Extract agent response from output."""
        if not output:
            return ""

        try:
            if isinstance(output, str):
                return output
            elif isinstance(output, dict) and "message" in output:
                return output["message"]
            return str(output)
        except Exception as e:
            logger.warning(f"Failed to extract agent response: {e}")
            return str(output) if output else ""

    def _extract_available_tools(self, metadata: Any) -> list[ToolConfig]:
        """Extract available tools from metadata."""
        available_tools = []
        try:
            if metadata and "attributes" in metadata:
                tools = self._parse_json_attr(metadata["attributes"].get("gen_ai.agent.tools", "[]"))
                for tool_data in tools:
                    available_tools.append(ToolConfig(name=tool_data))
        except Exception as e:
            logger.warning(f"Failed to extract available tools: {e}")
        return available_tools


def convert_trace_to_session(langfuse_trace: TraceWithFullDetails) -> Session:
    """Convert Langfuse TraceWithFullDetails to Strands Session."""
    mapper = LangfuseSessionMapper()
    return mapper.map_to_session([langfuse_trace], langfuse_trace.session_id or langfuse_trace.id)
