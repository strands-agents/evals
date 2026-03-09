"""
LangChainOtelSessionMapper - Maps Traceloop/OpenLLMetry LangChain traces to Session format.

Handles traces with scope: opentelemetry.instrumentation.langchain
(produced by opentelemetry-instrumentation-langchain from Traceloop's OpenLLMetry project)

Supports two trace formats:
1. ADOT/CloudWatch: Messages in span_events[].body.input/output.messages
2. Live instrumentation: Messages in gen_ai.*/traceloop.entity.* attributes
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

SCOPE_NAME = "opentelemetry.instrumentation.langchain"


class LangChainOtelSessionMapper(SessionMapper):
    """Maps Traceloop/OpenLLMetry LangChain traces to Session format.

    This mapper handles traces with scope 'opentelemetry.instrumentation.langchain',
    produced by Traceloop's OpenLLMetry project (opentelemetry-instrumentation-langchain).

    Span type detection (using Traceloop attributes):
    - Inference spans: llm.request.type == "chat"
    - Tool execution spans: traceloop.span.kind == "tool"
    - Agent invocation spans: traceloop.span.kind == "workflow"

    Supported trace formats:
    - ADOT/CloudWatch: Messages in span_events[].body.input/output.messages
    - Live instrumentation: Messages in gen_ai.*/traceloop.entity.* attributes
    """

    def __init__(self):
        super().__init__()
        # Track tools per trace (LangGraph stores tools in inference spans, not agent spans)
        self._trace_tools_map: dict[str, dict[str, ToolConfig]] = defaultdict(dict)
        # Track system prompts per trace
        self._trace_system_prompt_map: dict[str, str] = defaultdict(str)

    def map_to_session(self, data: Any, session_id: str) -> Session:
        """Map LangChain OTEL spans to Session format.

        Args:
            data: Trace data in various formats:
                - Flat list of spans: [{"trace_id": "x", "span_id": "y", ...}, ...]
                - Grouped by trace_id: {"trace_1": [spans], "trace_2": [spans]}
                - List of trace objects: [{"trace_id": "x", "spans": [...]}, ...]
            session_id: Session identifier

        Returns:
            Session object ready for evaluation
        """
        # Reset state for new mapping
        self._trace_tools_map = defaultdict(dict)
        self._trace_system_prompt_map = defaultdict(str)

        # Normalize input to flat spans
        spans = self._normalize_to_flat_spans(data)

        # Filter to only spans from this scope
        langchain_spans = [s for s in spans if self._get_scope_name(s) == SCOPE_NAME]

        # Group spans by trace_id
        grouped = defaultdict(list)
        for span in langchain_spans:
            trace_id = span.get("trace_id", "")
            grouped[trace_id].append(span)

        # Build traces
        result_traces: list[Trace] = []
        for trace_id, trace_spans in grouped.items():
            trace = self._build_trace(trace_id, trace_spans, session_id)
            if trace.spans:
                result_traces.append(trace)

        return Session(traces=result_traces, session_id=session_id)

    def _build_trace(self, trace_id: str, spans: list[dict], session_id: str) -> Trace:
        """Build a Trace from spans with the same trace_id."""
        converted_spans: list[InferenceSpan | ToolExecutionSpan | AgentInvocationSpan] = []

        for span in spans:
            try:
                if self._is_inference_span(span):
                    inference_span = self._convert_inference_span(span, session_id)
                    if inference_span and inference_span.messages:
                        converted_spans.append(inference_span)
                elif self._is_tool_execution_span(span):
                    tool_span = self._convert_tool_execution_span(span, session_id)
                    if tool_span:
                        converted_spans.append(tool_span)
                elif self._is_agent_invocation_span(span):
                    agent_span = self._convert_agent_invocation_span(span, session_id)
                    if agent_span:
                        converted_spans.append(agent_span)
            except Exception as e:
                logger.warning(f"Failed to convert span {span.get('span_id', 'unknown')}: {e}")

        return Trace(spans=converted_spans, trace_id=trace_id, session_id=session_id)

    # =========================================================================
    # Span Type Detection
    # =========================================================================

    def _is_inference_span(self, span: dict) -> bool:
        """Check if span is an LLM inference span."""
        attrs = span.get("attributes", {})
        return attrs.get("llm.request.type") == "chat"

    def _is_tool_execution_span(self, span: dict) -> bool:
        """Check if span is a tool execution span."""
        attrs = span.get("attributes", {})
        return attrs.get("traceloop.span.kind") == "tool"

    def _is_agent_invocation_span(self, span: dict) -> bool:
        """Check if span is an agent invocation span."""
        attrs = span.get("attributes", {})
        return attrs.get("traceloop.span.kind") == "workflow"

    # =========================================================================
    # Span Conversion
    # =========================================================================

    def _convert_inference_span(self, span: dict, session_id: str) -> InferenceSpan | None:
        """Convert OTEL span to InferenceSpan."""
        span_info = self._create_span_info(span, session_id)
        attrs = span.get("attributes", {})
        trace_id = span.get("trace_id", "")

        # Extract available tools from attributes
        tools = self._extract_tools_from_attributes(attrs)
        self._trace_tools_map[trace_id].update(tools)

        # Extract messages from span events
        input_messages, output_messages = self._get_messages_from_span_events(span)

        messages: list[UserMessage | AssistantMessage] = []

        # Process user message
        if input_messages:
            user_msg = self._extract_user_message(input_messages[-1], len(input_messages) - 1, attrs)
            if user_msg:
                messages.append(user_msg)

        # Process assistant message
        if output_messages:
            assistant_msg = self._extract_assistant_message(output_messages[-1], len(output_messages) - 1, attrs)
            if assistant_msg:
                messages.append(assistant_msg)

        # Extract system prompt
        for idx, msg in enumerate(input_messages):
            if attrs.get(f"gen_ai.prompt.{idx}.role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    self._trace_system_prompt_map[trace_id] = content
                break

        if not messages:
            return None

        return InferenceSpan(span_info=span_info, messages=messages, metadata={})

    def _convert_tool_execution_span(self, span: dict, session_id: str) -> ToolExecutionSpan | None:
        """Convert OTEL span to ToolExecutionSpan.

        Handles two formats:
        1. ADOT/CloudWatch: Data in span_events[].body.input/output.messages
        2. Live instrumentation: Data in traceloop.entity.input/output attributes
        """
        span_info = self._create_span_info(span, session_id)
        attrs = span.get("attributes", {})

        tool_name = attrs.get("traceloop.entity.name")
        tool_parameters: dict | None = None
        tool_output_content: str | None = None
        tool_call_id: str | None = None
        tool_status: str | None = ""

        # Try ADOT/CloudWatch format first (span_events)
        input_messages, output_messages = self._get_messages_from_span_events(span)

        if input_messages and output_messages:
            # ADOT format - extract from messages
            input_content = input_messages[-1].get("content", "")
            if isinstance(input_content, str):
                parsed = self._safe_json_parse(input_content)
                if isinstance(parsed, dict):
                    if "inputs" in parsed and isinstance(parsed.get("inputs"), dict):
                        tool_parameters = parsed.get("inputs")
                    elif "input_str" in parsed:
                        params_parsed = self._safe_json_parse(parsed.get("input_str", ""))
                        if isinstance(params_parsed, dict):
                            tool_parameters = params_parsed

            output_content = output_messages[-1].get("content", "")
            if isinstance(output_content, str):
                parsed = self._safe_json_parse(output_content)
                if isinstance(parsed, dict) and "output" in parsed:
                    output_obj = parsed["output"]
                    if isinstance(output_obj, dict) and "kwargs" in output_obj:
                        kwargs = output_obj["kwargs"]
                        if isinstance(kwargs, dict):
                            tool_output_content = kwargs.get("content")
                            tool_call_id = kwargs.get("tool_call_id")
                            tool_status = kwargs.get("status")
        else:
            # Live format - extract from traceloop.entity.* attributes
            entity_input = attrs.get("traceloop.entity.input", "")
            entity_output = attrs.get("traceloop.entity.output", "")

            if entity_input:
                parsed = self._safe_json_parse(entity_input)
                if isinstance(parsed, dict):
                    if "inputs" in parsed and isinstance(parsed.get("inputs"), dict):
                        tool_parameters = parsed.get("inputs")
                    elif "input_str" in parsed:
                        params_parsed = self._safe_json_parse(parsed.get("input_str", ""))
                        if isinstance(params_parsed, dict):
                            tool_parameters = params_parsed

            if entity_output:
                parsed = self._safe_json_parse(entity_output)
                if isinstance(parsed, dict) and "output" in parsed:
                    output_obj = parsed["output"]
                    if isinstance(output_obj, dict) and "kwargs" in output_obj:
                        kwargs = output_obj["kwargs"]
                        if isinstance(kwargs, dict):
                            tool_output_content = kwargs.get("content")
                            tool_call_id = kwargs.get("tool_call_id")
                            tool_status = kwargs.get("status")

        # Validate required fields
        if not tool_name or tool_parameters is None or tool_output_content is None:
            logger.warning(f"Missing required fields for tool span {span.get('span_id')}")
            return None

        tool_call = ToolCall(name=tool_name, arguments=tool_parameters or {}, tool_call_id=tool_call_id)
        tool_result = ToolResult(
            content=tool_output_content or "",
            error=None if tool_status == "success" else tool_status,
            tool_call_id=tool_call_id,
        )

        return ToolExecutionSpan(span_info=span_info, tool_call=tool_call, tool_result=tool_result, metadata={})

    def _convert_agent_invocation_span(self, span: dict, session_id: str) -> AgentInvocationSpan | None:
        """Convert OTEL span to AgentInvocationSpan.

        Handles two formats:
        1. ADOT/CloudWatch: Data in span_events[].body.input/output.messages
        2. Live instrumentation: Data in traceloop.entity.input/output attributes
        """
        span_info = self._create_span_info(span, session_id)
        trace_id = span.get("trace_id", "")
        attrs = span.get("attributes", {})

        user_query: str | None = None
        agent_response: str | None = None

        # Try ADOT/CloudWatch format first (span_events)
        input_messages, output_messages = self._get_messages_from_span_events(span)

        if input_messages and output_messages:
            # ADOT format
            user_query = self._extract_user_prompt_from_input(input_messages)
            agent_response = self._extract_agent_response_from_output(output_messages)
        else:
            # Live format - extract from traceloop.entity.* attributes
            entity_input = attrs.get("traceloop.entity.input", "")
            entity_output = attrs.get("traceloop.entity.output", "")

            if entity_input:
                parsed = self._safe_json_parse(entity_input)
                if isinstance(parsed, dict) and "inputs" in parsed:
                    inputs = parsed["inputs"]
                    if isinstance(inputs, dict) and "messages" in inputs:
                        user_query = self._get_last_message_text(inputs["messages"])

            if entity_output:
                parsed = self._safe_json_parse(entity_output)
                if isinstance(parsed, dict) and "outputs" in parsed:
                    outputs = parsed["outputs"]
                    if isinstance(outputs, dict) and "messages" in outputs:
                        agent_response = self._get_last_message_text(outputs["messages"])

        if not user_query or not agent_response:
            logger.warning(f"Missing user_query or agent_response for span {span.get('span_id')}")
            return None

        available_tools = sorted(
            self._trace_tools_map.get(trace_id, {}).values(),
            key=lambda t: t.name,
        )

        return AgentInvocationSpan(
            span_info=span_info,
            user_prompt=user_query,
            agent_response=agent_response,
            available_tools=available_tools,
            metadata={},
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_scope_name(self, span: dict) -> str:
        """Extract scope name from span."""
        scope = span.get("scope", {})
        return scope.get("name", "") if isinstance(scope, dict) else ""

    def _create_span_info(self, span: dict, session_id: str) -> SpanInfo:
        """Create SpanInfo from span dict."""
        start_time = self._parse_timestamp(span.get("start_time"))
        end_time = self._parse_timestamp(span.get("end_time"))

        return SpanInfo(
            trace_id=span.get("trace_id"),
            span_id=span.get("span_id"),
            session_id=session_id,
            parent_span_id=span.get("parent_span_id"),
            start_time=start_time,
            end_time=end_time,
        )

    def _parse_timestamp(self, value: Any) -> datetime:
        """Parse timestamp from various formats."""
        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime.now(timezone.utc)
        if isinstance(value, (int, float)):
            # Handle nanoseconds
            if value > 1e12:
                value = value / 1e9
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return datetime.now(timezone.utc)

    def _safe_json_parse(self, content: Any) -> Any:
        """Safely parse JSON content."""
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return content

    def _get_messages_from_span_events(self, span: dict) -> tuple[list[dict], list[dict]]:
        """Extract input and output messages from span events or attributes.

        Handles two formats:
        1. ADOT/CloudWatch: Messages in span_events[].body.input/output.messages
        2. Live instrumentation: Messages in gen_ai.prompt.*/gen_ai.completion.* attributes
        """
        # Try ADOT/CloudWatch format first (span_events)
        span_events = span.get("span_events", [])
        for event in span_events:
            event_name = event.get("event_name", "")
            if event_name == SCOPE_NAME:
                body = event.get("body", {})
                input_group = body.get("input", {})
                output_group = body.get("output", {})
                input_msgs = input_group.get("messages", [])
                output_msgs = output_group.get("messages", [])
                if input_msgs or output_msgs:
                    return input_msgs, output_msgs

        # Fallback to live instrumentation format (gen_ai.* attributes)
        attrs = span.get("attributes", {})
        input_messages = self._extract_messages_from_gen_ai_attrs(attrs, "prompt")
        output_messages = self._extract_messages_from_gen_ai_attrs(attrs, "completion")

        return input_messages, output_messages

    def _extract_messages_from_gen_ai_attrs(self, attrs: dict, msg_type: str) -> list[dict]:
        """Extract messages from gen_ai.prompt.* or gen_ai.completion.* attributes.

        Args:
            attrs: Span attributes dict
            msg_type: Either "prompt" (for input) or "completion" (for output)

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages: list[dict] = []
        idx = 0

        while True:
            role_key = f"gen_ai.{msg_type}.{idx}.role"
            content_key = f"gen_ai.{msg_type}.{idx}.content"

            role = attrs.get(role_key)
            content = attrs.get(content_key)

            if content is None:
                break

            messages.append(
                {
                    "role": role or ("user" if msg_type == "prompt" else "assistant"),
                    "content": content,
                }
            )
            idx += 1

        return messages

    def _extract_tools_from_attributes(self, attrs: dict) -> dict[str, ToolConfig]:
        """Extract tool definitions from llm.request.functions.* attributes."""
        tools: dict[str, ToolConfig] = {}

        # Find all function indices
        indices = set()
        for key in attrs.keys():
            if key.startswith("llm.request.functions.") and key.endswith(".name"):
                try:
                    idx = key.split(".")[3]
                    if idx.isdigit():
                        indices.add(idx)
                except (IndexError, ValueError):
                    continue

        for idx in indices:
            name = attrs.get(f"llm.request.functions.{idx}.name")
            if name:
                description = attrs.get(f"llm.request.functions.{idx}.description")
                params_str = attrs.get(f"llm.request.functions.{idx}.parameters")
                parameters = None
                if params_str:
                    try:
                        parameters = json.loads(params_str)
                    except json.JSONDecodeError:
                        pass
                tools[name] = ToolConfig(name=name, description=description, parameters=parameters)

        return tools

    def _extract_user_message(self, message: dict, index: int, attrs: dict) -> UserMessage | None:
        """Extract user message from span event message."""
        content: list[TextContent | ToolResultContent] = []

        # Check if this is a tool result message
        is_tool_msg, tool_call_id = self._check_tool_result_message(index, attrs)

        msg_content = message.get("content", "")
        if isinstance(msg_content, str):
            if is_tool_msg:
                content.append(ToolResultContent(content=msg_content, error=None, tool_call_id=tool_call_id))
            else:
                content.append(TextContent(text=msg_content))
            return UserMessage(content=content)

        return None

    def _extract_assistant_message(self, message: dict, index: int, attrs: dict) -> AssistantMessage | None:
        """Extract assistant message from span event message."""
        content: list[TextContent | ToolCallContent] = []

        msg_content = message.get("content", "")
        if isinstance(msg_content, str):
            content.append(TextContent(text=msg_content))

            # Check for tool calls
            tool_calls = self._get_assistant_tool_calls(index, attrs)
            content.extend(tool_calls)

            return AssistantMessage(content=content)

        return None

    def _check_tool_result_message(self, index: int, attrs: dict) -> tuple[bool, str]:
        """Check if message at index is a tool result."""
        role_key = f"gen_ai.prompt.{index}.role"
        if attrs.get(role_key) == "tool":
            tool_call_id_key = f"gen_ai.prompt.{index}.tool_call_id"
            return True, attrs.get(tool_call_id_key, "")
        return False, ""

    def _get_assistant_tool_calls(self, index: int, attrs: dict) -> list[ToolCallContent]:
        """Extract tool calls from assistant message attributes."""
        tool_calls: list[ToolCallContent] = []
        tool_idx = 0

        while True:
            name_key = f"gen_ai.completion.{index}.tool_calls.{tool_idx}.name"
            args_key = f"gen_ai.completion.{index}.tool_calls.{tool_idx}.arguments"
            id_key = f"gen_ai.completion.{index}.tool_calls.{tool_idx}.id"

            tool_name = attrs.get(name_key)
            if not tool_name:
                break

            args_str = attrs.get(args_key, "{}")
            tool_call_id = attrs.get(id_key, "")

            try:
                arguments = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                arguments = {}

            tool_calls.append(ToolCallContent(name=tool_name, arguments=arguments, tool_call_id=tool_call_id))
            tool_idx += 1

        return tool_calls

    def _extract_user_prompt_from_input(self, input_messages: list[dict]) -> str | None:
        """Extract user prompt from agent invocation input messages."""
        if not input_messages:
            return None

        msg = input_messages[-1]
        content = msg.get("content", "")
        if isinstance(content, str):
            parsed = self._safe_json_parse(content)
            if isinstance(parsed, dict) and "inputs" in parsed:
                inputs = parsed["inputs"]
                if isinstance(inputs, dict) and "messages" in inputs:
                    messages = inputs["messages"]
                    return self._get_last_message_text(messages)
        return None

    def _extract_agent_response_from_output(self, output_messages: list[dict]) -> str | None:
        """Extract agent response from agent invocation output messages."""
        if not output_messages:
            return None

        msg = output_messages[-1]
        content = msg.get("content", "")
        if isinstance(content, str):
            parsed = self._safe_json_parse(content)
            if isinstance(parsed, dict) and "outputs" in parsed:
                outputs = parsed["outputs"]
                if isinstance(outputs, dict) and "messages" in outputs:
                    messages = outputs["messages"]
                    return self._get_last_message_text(messages)
        return None

    def _get_last_message_text(self, messages: list) -> str | None:
        """Extract text from the last message in a LangGraph state messages list."""
        if not messages:
            return None

        msg = messages[-1]

        # LangGraph-native format: {"kwargs": {"content": "..."}}
        if isinstance(msg, dict):
            if "kwargs" in msg:
                kwargs = msg["kwargs"]
                if isinstance(kwargs, dict) and "content" in kwargs:
                    content = kwargs["content"]
                    # Handle list content (multi-part messages)
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                return item["text"]
                    return content
            # OpenAI format: {"content": "..."}
            if "content" in msg:
                return msg["content"]

        # Tuple format: ["role", "content"]
        elif isinstance(msg, list):
            return msg[-1] if msg else None

        elif isinstance(msg, str):
            return msg

        return None
