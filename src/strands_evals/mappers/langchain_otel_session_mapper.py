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
from .constants import (
    ADOT_INPUT_STR_KEY,
    ADOT_LANGGRAPH_NAME,
    ADOT_ROLE_UNKNOWN,
    ADOT_TOOL_CALL_WITH_CONTEXT,
    ATTR_LLM_REQUEST_TYPE,
    ATTR_TRACELOOP_ENTITY_INPUT,
    ATTR_TRACELOOP_ENTITY_NAME,
    ATTR_TRACELOOP_ENTITY_OUTPUT,
    ATTR_TRACELOOP_SPAN_KIND,
    KIND_TOOL,
    KIND_WORKFLOW,
    LLM_TYPE_CHAT,
    SCOPE_LANGCHAIN_OTEL,
)
from .session_mapper import SessionMapper

logger = logging.getLogger(__name__)


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
        # Cache: span_id → (input_messages, output_messages) from _get_messages_from_span_events
        # Avoids re-parsing span_events body during detection and again during conversion.
        # Reset at the start of each map_to_session call.
        self._span_messages_cache: dict[str, tuple[list[dict], list[dict]]] = {}
        # Cache: span_id → parsed input body from _parse_adot_body
        self._adot_body_cache: dict[str, Any] = {}

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
        self._span_messages_cache = {}
        self._adot_body_cache = {}

        # Normalize input to flat spans
        spans = self._normalize_to_flat_spans(data)

        # Filter to only spans from this scope
        langchain_spans = [s for s in spans if self._get_scope_name(s) == SCOPE_LANGCHAIN_OTEL]

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
        """Build a Trace from spans with the same trace_id.

        Tools are collected from inference spans and passed to agent invocation spans,
        since LangGraph stores tool definitions in inference spans, not agent spans.
        """
        converted_spans: list[InferenceSpan | ToolExecutionSpan | AgentInvocationSpan] = []
        # Local tools accumulator — populated by inference spans, consumed by agent spans
        trace_tools: dict[str, ToolConfig] = {}

        for span in spans:
            try:
                if self._is_inference_span(span):
                    inference_span = self._convert_inference_span(span, session_id, trace_tools)
                    if inference_span and inference_span.messages:
                        converted_spans.append(inference_span)
                elif self._is_tool_execution_span(span):
                    tool_span = self._convert_tool_execution_span(span, session_id)
                    if tool_span:
                        converted_spans.append(tool_span)
                elif self._is_agent_invocation_span(span):
                    agent_span = self._convert_agent_invocation_span(span, session_id, trace_tools)
                    if agent_span:
                        converted_spans.append(agent_span)
            except Exception as e:
                logger.warning(f"Failed to convert span {span.get('span_id', 'unknown')}: {e}")

        # In multi-agent LangGraph systems, each nested sub-graph (from
        # create_react_agent) produces its own kwargs.name="LangGraph" record
        # in ADOT, creating multiple AgentInvocationSpans with intermediate data.
        # The root graph is always the LAST one (outermost scope finishes last).
        # Keep only the root to match the live path behavior where Traceloop
        # only marks the outermost graph as traceloop.span.kind="workflow".
        agent_spans = [s for s in converted_spans if isinstance(s, AgentInvocationSpan)]
        if len(agent_spans) > 1:
            root = agent_spans[-1]
            converted_spans = [s for s in converted_spans if not isinstance(s, AgentInvocationSpan) or s is root]

        # If no tools found from attributes (common in ADOT), build from converted tool spans
        if not trace_tools:
            for converted in converted_spans:
                if isinstance(converted, ToolExecutionSpan):
                    name = converted.tool_call.name
                    if name and name not in trace_tools:
                        trace_tools[name] = ToolConfig(name=name, description=None, parameters=None)
            # Back-fill available_tools into any AgentInvocationSpan already created
            if trace_tools:
                tools_list = sorted(trace_tools.values(), key=lambda t: t.name)
                for converted in converted_spans:
                    if isinstance(converted, AgentInvocationSpan) and not converted.available_tools:
                        converted.available_tools = tools_list

        return Trace(spans=converted_spans, trace_id=trace_id, session_id=session_id)

    # =========================================================================
    # Span Type Detection
    # =========================================================================

    def _is_inference_span(self, span: dict) -> bool:
        """Check if span is an LLM inference span.

        Detection:
        1. Live instrumentation: llm.request.type == "chat"
        2. ADOT body: role == "unknown" with raw string content (direct LLM I/O)
        """
        attrs = span.get("attributes", {})
        if attrs.get(ATTR_LLM_REQUEST_TYPE) == LLM_TYPE_CHAT:
            return True

        # ADOT fallback: records with role="unknown" and raw string content are LLM calls
        input_messages, _ = self._get_messages_from_span_events(span)
        if input_messages and input_messages[0].get("role") == ADOT_ROLE_UNKNOWN:
            return True

        return False

    def _is_tool_execution_span(self, span: dict) -> bool:
        """Check if span is a tool execution span.

        Detection:
        1. Live instrumentation: traceloop.span.kind == "tool"
        2. ADOT body: input has "input_str" key (direct tool call format)
        """
        attrs = span.get("attributes", {})
        if attrs.get(ATTR_TRACELOOP_SPAN_KIND) == KIND_TOOL:
            return True

        # ADOT fallback: "input_str" is the direct tool invocation format.
        # Note: "tool_call_with_context" records are graph-level wrappers that
        # duplicate the actual tool call — skip them to avoid duplicate spans.
        in_parsed = self._parse_adot_body(span)
        if isinstance(in_parsed, dict) and ADOT_INPUT_STR_KEY in in_parsed:
            return True

        return False

    def _is_agent_invocation_span(self, span: dict) -> bool:
        """Check if span is an agent invocation span.

        Detection:
        1. Live instrumentation: traceloop.span.kind == "workflow"
        2. ADOT body: kwargs.name == "LangGraph" (root graph node only)

        Only the root LangGraph node should become an AgentInvocationSpan.
        Intermediate nodes (Prompt, call_model, should_continue, agent,
        RunnableSequence, tools) are internal graph steps and must be skipped.
        """
        attrs = span.get("attributes", {})
        if attrs.get(ATTR_TRACELOOP_SPAN_KIND) == KIND_WORKFLOW:
            return True

        # ADOT fallback: only the root LangGraph node is the agent invocation
        in_parsed = self._parse_adot_body(span)
        if isinstance(in_parsed, dict):
            kwargs = in_parsed.get("kwargs")
            if isinstance(kwargs, dict) and kwargs.get("name") == ADOT_LANGGRAPH_NAME:
                return True

        return False

    def _parse_adot_body(self, span: dict) -> Any:
        """Parse the input content from the first ADOT body message.

        Returns the parsed JSON object or raw string, or None if no body found.
        Result is cached per span_id to avoid re-parsing across detection methods.
        """
        span_id = span.get("span_id", "")
        if span_id in self._adot_body_cache:
            return self._adot_body_cache[span_id]

        input_messages, _ = self._get_messages_from_span_events(span)
        if not input_messages:
            result = None
        else:
            in_content = input_messages[0].get("content", "")
            result = self._safe_json_parse(in_content) if isinstance(in_content, str) else in_content

        if span_id:
            self._adot_body_cache[span_id] = result
        return result

    # =========================================================================
    # Span Conversion
    # =========================================================================

    def _convert_inference_span(
        self, span: dict, session_id: str, trace_tools: dict[str, ToolConfig]
    ) -> InferenceSpan | None:
        """Convert OTEL span to InferenceSpan."""
        span_info = self._create_span_info(span, session_id)
        attrs = span.get("attributes", {})

        # Extract available tools from attributes and accumulate for agent span
        tools = self._extract_tools_from_attributes(attrs)
        trace_tools.update(tools)

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

        tool_name = attrs.get(ATTR_TRACELOOP_ENTITY_NAME)
        tool_parameters: dict | None = None
        tool_output_content: str | None = None
        tool_call_id: str | None = None
        tool_status: str | None = None

        # Try ADOT/CloudWatch format first (span_events)
        input_messages, output_messages = self._get_messages_from_span_events(span)

        if input_messages and output_messages:
            in_parsed, out_parsed = self._parse_adot_tool_content(input_messages, output_messages)

            # Extract tool parameters from input
            if isinstance(in_parsed, dict):
                inputs = in_parsed.get("inputs")
                if isinstance(inputs, dict):
                    # tool_call_with_context: {"inputs": {"__type": "tool_call_with_context", "tool_call": {...}}}
                    if inputs.get("__type") == ADOT_TOOL_CALL_WITH_CONTEXT:
                        tc = inputs.get("tool_call", {})
                        tool_parameters = tc.get("args", {})
                        tool_name = tool_name or tc.get("name")
                        tool_call_id = tool_call_id or tc.get("id")
                    else:
                        # Direct inputs dict: {"inputs": {"a": 1, "b": 2}}
                        tool_parameters = inputs
                if tool_parameters is None and ADOT_INPUT_STR_KEY in in_parsed:
                    params_parsed = self._safe_json_parse(in_parsed.get(ADOT_INPUT_STR_KEY, ""))
                    if isinstance(params_parsed, dict):
                        tool_parameters = params_parsed

            # Extract tool output, name, call_id from output
            if isinstance(out_parsed, dict):
                lc_kwargs = self._extract_lc_kwargs(out_parsed, "output")
                if lc_kwargs is None:
                    lc_kwargs = self._extract_lc_kwargs(out_parsed, "outputs")
                if lc_kwargs:
                    tool_output_content = str(lc_kwargs.get("content", ""))
                    tool_call_id = tool_call_id or lc_kwargs.get("tool_call_id")
                    tool_status = lc_kwargs.get("status", tool_status)
                    tool_name = tool_name or lc_kwargs.get("name")
                # Also check top-level kwargs for name
                top_kwargs = out_parsed.get("kwargs", {})
                if isinstance(top_kwargs, dict):
                    tool_name = tool_name or top_kwargs.get("name")
        else:
            # Live format - extract from traceloop.entity.* attributes
            entity_input = attrs.get(ATTR_TRACELOOP_ENTITY_INPUT, "")
            entity_output = attrs.get(ATTR_TRACELOOP_ENTITY_OUTPUT, "")

            if entity_input:
                parsed = self._safe_json_parse(entity_input)
                if isinstance(parsed, dict):
                    if "inputs" in parsed and isinstance(parsed.get("inputs"), dict):
                        tool_parameters = parsed.get("inputs")
                    elif ADOT_INPUT_STR_KEY in parsed:
                        params_parsed = self._safe_json_parse(parsed.get(ADOT_INPUT_STR_KEY, ""))
                        if isinstance(params_parsed, dict):
                            tool_parameters = params_parsed

            if entity_output:
                parsed = self._safe_json_parse(entity_output)
                lc_kwargs = self._extract_lc_kwargs(parsed, "output") if isinstance(parsed, dict) else None
                if lc_kwargs:
                    tool_output_content = str(lc_kwargs.get("content", ""))
                    tool_call_id = lc_kwargs.get("tool_call_id")
                    tool_status = lc_kwargs.get("status", tool_status)

        # Validate required fields
        if not tool_name or tool_parameters is None or tool_output_content is None:
            logger.warning(f"Missing required fields for tool span {span.get('span_id')}")
            return None

        tool_call = ToolCall(name=tool_name, arguments=tool_parameters or {}, tool_call_id=tool_call_id)
        tool_result = ToolResult(
            content=tool_output_content or "",
            error=None if tool_status in ("success", None) else tool_status,
            tool_call_id=tool_call_id,
        )

        return ToolExecutionSpan(span_info=span_info, tool_call=tool_call, tool_result=tool_result, metadata={})

    def _convert_agent_invocation_span(
        self, span: dict, session_id: str, trace_tools: dict[str, ToolConfig]
    ) -> AgentInvocationSpan | None:
        """Convert OTEL span to AgentInvocationSpan.

        Handles two formats:
        1. ADOT/CloudWatch: Data in span_events[].body.input/output.messages
        2. Live instrumentation: Data in traceloop.entity.input/output attributes
        """
        span_info = self._create_span_info(span, session_id)
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
            entity_input = attrs.get(ATTR_TRACELOOP_ENTITY_INPUT, "")
            entity_output = attrs.get(ATTR_TRACELOOP_ENTITY_OUTPUT, "")

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

        if user_query is None or agent_response is None:
            logger.warning(f"Missing user_query or agent_response for span {span.get('span_id')}")
            return None

        available_tools = sorted(
            trace_tools.values(),
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

    def _parse_adot_tool_content(self, input_messages: list[dict], output_messages: list[dict]) -> tuple[Any, Any]:
        """Parse and return (input_parsed, output_parsed) from ADOT body messages."""
        in_content = input_messages[-1].get("content", "")
        in_parsed = self._safe_json_parse(in_content) if isinstance(in_content, str) else in_content

        out_content = output_messages[-1].get("content", "")
        out_parsed = self._safe_json_parse(out_content) if isinstance(out_content, str) else out_content

        return in_parsed, out_parsed

    def _extract_lc_kwargs(self, parsed: dict, key: str) -> dict | None:
        """Extract kwargs from a LangChain serialized object at parsed[key].

        Handles: {"output": {"lc": 1, "kwargs": {...}}}
        Also handles list of LC objects: {"outputs": [{"lc": 1, "kwargs": {...}}]}
        Also handles nested messages: {"outputs": {"messages": [{"kwargs": {...}}]}}
        """
        obj = parsed.get(key)
        if isinstance(obj, dict):
            if "kwargs" in obj:
                return obj["kwargs"]
            # Nested messages list
            if "messages" in obj:
                msgs = obj["messages"]
                if isinstance(msgs, list) and msgs:
                    last = msgs[-1]
                    if isinstance(last, dict) and "kwargs" in last:
                        return last["kwargs"]
        if isinstance(obj, list) and obj:
            last = obj[-1]
            if isinstance(last, dict) and "kwargs" in last:
                return last["kwargs"]
        return None

    def _get_messages_from_span_events(self, span: dict) -> tuple[list[dict], list[dict]]:
        """Extract input and output messages from span events or attributes.

        Handles two formats:
        1. ADOT/CloudWatch: Messages in span_events[].body.input/output.messages
        2. Live instrumentation: Messages in gen_ai.prompt.*/gen_ai.completion.* attributes
        """
        span_id = span.get("span_id", "")
        if span_id in self._span_messages_cache:
            return self._span_messages_cache[span_id]

        result = self._extract_messages_from_span(span)
        if span_id:
            self._span_messages_cache[span_id] = result
        return result

    def _extract_messages_from_span(self, span: dict) -> tuple[list[dict], list[dict]]:
        """Extract messages without caching — called by _get_messages_from_span_events."""
        # Try ADOT/CloudWatch format first (span_events)
        span_events = span.get("span_events", [])
        for event in span_events:
            event_name = event.get("event_name", "")
            if event_name == SCOPE_LANGCHAIN_OTEL:
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
            # ADOT double-encodes strings — decode outer JSON quotes if present
            text = self._safe_json_parse(msg_content) if msg_content.startswith('"') else msg_content
            if not isinstance(text, str):
                text = msg_content
            if is_tool_msg:
                content.append(ToolResultContent(content=text, error=None, tool_call_id=tool_call_id))
            elif text:
                content.append(TextContent(text=text))
            if content:
                return UserMessage(content=content)

        return None

    def _extract_assistant_message(self, message: dict, index: int, attrs: dict) -> AssistantMessage | None:
        """Extract assistant message from span event message."""
        content: list[TextContent | ToolCallContent] = []

        msg_content = message.get("content", "")
        if isinstance(msg_content, str):
            # ADOT double-encodes empty strings as '""' — decode and skip if empty
            text = self._safe_json_parse(msg_content) if msg_content.startswith('"') else msg_content
            if not isinstance(text, str):
                text = msg_content
            if text:
                content.append(TextContent(text=text))

            # Check for tool calls
            tool_calls = self._get_assistant_tool_calls(index, attrs)
            content.extend(tool_calls)

            if content:
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
        """Extract agent response from agent invocation output messages.

        Handles multiple LangChain output formats:
        1. {"outputs": {"messages": [...]}} — LangGraph state with messages list
        2. {"outputs": {"lc": 1, "kwargs": {"content": "..."}}} — single LC object
        3. {"outputs": [{"lc": 1, "kwargs": {"content": "..."}}]} — list of LC objects
        4. {"output": "..."} / {"output": {"kwargs": {"content": "..."}}} — singular key
        """
        if not output_messages:
            return None

        msg = output_messages[-1]
        content = msg.get("content", "")
        if not isinstance(content, str):
            return None

        parsed = self._safe_json_parse(content)
        if not isinstance(parsed, dict):
            return None

        # Try "outputs" key first, then "output"
        for key in ("outputs", "output"):
            if key not in parsed:
                continue
            value = parsed[key]

            # Direct string
            if isinstance(value, str) and value:
                return value

            # Dict with messages list: {"messages": [...]}
            if isinstance(value, dict) and "messages" in value:
                text = self._get_last_message_text(value["messages"])
                if text:
                    return text

            # Single LC object: {"lc": 1, "kwargs": {"content": "..."}}
            if isinstance(value, dict) and "kwargs" in value:
                text = self._get_lc_text_content(value)
                if text:
                    return text

            # List of LC objects: [{"lc": 1, "kwargs": {"content": "..."}}]
            if isinstance(value, list) and value:
                text = self._get_last_message_text(value)
                if text:
                    return text

        return None

    def _get_lc_text_content(self, lc_obj: dict) -> str | None:
        """Extract text content from a LangChain serialized object.

        Format: {"lc": 1, "type": "constructor", "id": [...], "kwargs": {"content": "..."}}
        Content can be a string or a list of content blocks.
        """
        kwargs = lc_obj.get("kwargs", {})
        if not isinstance(kwargs, dict):
            return None
        content = kwargs.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
        return None

    def _get_last_message_text(self, messages: list) -> str | None:
        """Extract text from the last message in a LangGraph state messages list.

        Searches backward through messages to find the last one with non-empty text content.
        Skips tool messages (type="tool") since they aren't agent responses.
        """
        if not messages:
            return None

        # Search backward for last message with non-empty content
        for msg in reversed(messages):
            text = self._extract_message_text(msg)
            if text:
                return text

        return None

    def _extract_message_text(self, msg) -> str | None:
        """Extract text content from a single message."""
        if isinstance(msg, dict):
            # Skip tool messages — they aren't agent responses
            if isinstance(msg.get("kwargs"), dict) and msg["kwargs"].get("type") == "tool":
                return None

            if "kwargs" in msg:
                kwargs = msg["kwargs"]
                if isinstance(kwargs, dict) and "content" in kwargs:
                    content = kwargs["content"]
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                return item["text"]
                    return content
            if "content" in msg:
                return msg["content"]

        elif isinstance(msg, list):
            return msg[-1] if msg else None

        elif isinstance(msg, str):
            return msg

        return None
