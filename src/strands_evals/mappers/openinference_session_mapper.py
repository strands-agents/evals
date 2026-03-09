"""
OpenInferenceSessionMapper - Maps OpenInference LangChain traces to Session format.

Handles traces with scope: openinference.instrumentation.langchain
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

SCOPE_NAME = "openinference.instrumentation.langchain"


class OpenInferenceSessionMapper(SessionMapper):
    """Maps OpenInference LangChain traces to Session format.

    This mapper handles traces produced by the openinference-instrumentation-langchain library.
    It identifies span types using:
    - Inference spans: openinference.span.kind == "LLM"
    - Tool execution spans: openinference.span.kind == "TOOL"
    - Agent invocation spans: openinference.span.kind == "CHAIN" (name=LangGraph) or "AGENT"
    """

    def __init__(self):
        super().__init__()
        # Track tools per trace (tools appear in LLM spans, not agent spans)
        self._trace_tools_map: dict[str, dict[str, ToolConfig]] = defaultdict(dict)
        # Track if agent invocation span already found for trace
        self._trace_has_agent_invocation: dict[str, bool] = defaultdict(bool)
        # Track system prompts per trace
        self._trace_system_prompt_map: dict[str, str] = defaultdict(str)

    def map_to_session(self, data: Any, session_id: str) -> Session:
        """Map OpenInference LangChain spans to Session format.

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
        self._trace_has_agent_invocation = defaultdict(bool)
        self._trace_system_prompt_map = defaultdict(str)

        # Normalize input to flat spans
        spans = self._normalize_to_flat_spans(data)

        # Filter to only spans from this scope
        openinference_spans = [s for s in spans if self._get_scope_name(s) == SCOPE_NAME]

        # Group spans by trace_id
        grouped = defaultdict(list)
        for span in openinference_spans:
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
        return attrs.get("openinference.span.kind") == "LLM"

    def _is_tool_execution_span(self, span: dict) -> bool:
        """Check if span is a tool execution span."""
        attrs = span.get("attributes", {})
        return attrs.get("openinference.span.kind") == "TOOL"

    def _is_agent_invocation_span(self, span: dict) -> bool:
        """Check if span is an agent invocation span.

        Returns True for:
        1. Default agent: CHAIN + name=LangGraph (takes priority)
        2. Custom-named agent: AGENT + langgraph_step >= 2
        3. Model chain: CHAIN + name=model + langgraph_step >= 2

        Enforces: Only ONE agent invocation span per trace.
        """
        attrs = span.get("attributes", {})
        span_kind = attrs.get("openinference.span.kind", "")
        span_name = span.get("name", "")
        trace_id = span.get("trace_id", "")

        # If we've already found an agent invocation for this trace, skip
        if self._trace_has_agent_invocation.get(trace_id, False):
            return False

        # Case 1: Default unnamed agent (standard LangGraph pattern)
        if span_kind == "CHAIN" and span_name == "LangGraph":
            self._trace_has_agent_invocation[trace_id] = True
            return True

        # Case 2: Custom-named agent
        if span_kind == "AGENT":
            metadata_str = attrs.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                step = metadata.get("langgraph_step", 0)
                if step >= 2:
                    self._trace_has_agent_invocation[trace_id] = True
                    return True
            except (json.JSONDecodeError, AttributeError):
                pass

        # Case 3: Model chain
        if span_kind == "CHAIN" and span_name == "model":
            metadata_str = attrs.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                step = metadata.get("langgraph_step", 0)
                if step >= 2:
                    self._trace_has_agent_invocation[trace_id] = True
                    return True
            except (json.JSONDecodeError, AttributeError):
                pass

        return False

    # =========================================================================
    # Span Conversion
    # =========================================================================

    def _convert_inference_span(self, span: dict, session_id: str) -> InferenceSpan | None:
        """Convert OTEL span to InferenceSpan."""
        span_info = self._create_span_info(span, session_id)
        attrs = span.get("attributes", {})
        trace_id = span.get("trace_id", "")

        input_messages, output_messages = self._get_messages_from_span_events(span)

        # Extract user message
        user_contents, system_prompt = self._extract_user_contents(input_messages, span)
        if not user_contents:
            logger.warning(f"No user contents for span {span.get('span_id')}")
            return None

        user_message = UserMessage(content=user_contents)

        # Track system prompt
        if system_prompt:
            self._trace_system_prompt_map[trace_id] = system_prompt

        # Extract assistant message
        assistant_contents = self._extract_assistant_contents(output_messages, span)
        if not assistant_contents:
            logger.warning(f"No assistant contents for span {span.get('span_id')}")
            return None

        assistant_message = AssistantMessage(content=assistant_contents)

        # Extract tool schemas
        tools = self._extract_tools_from_attributes(attrs)
        self._trace_tools_map[trace_id].update({t.name: t for t in tools})

        return InferenceSpan(span_info=span_info, messages=[user_message, assistant_message], metadata={})

    def _convert_tool_execution_span(self, span: dict, session_id: str) -> ToolExecutionSpan | None:
        """Convert OTEL span to ToolExecutionSpan."""
        span_info = self._create_span_info(span, session_id)
        attrs = span.get("attributes", {})

        tool_name: str | None = None
        tool_parameters: dict | None = None
        tool_output_content: str | None = None
        tool_call_id: str | None = None
        tool_status: str | None = None

        # Try live format first (attributes)
        tool_name = attrs.get("tool.name") or span.get("name")

        # Get input from attributes
        input_value = attrs.get("input.value")
        if input_value:
            try:
                if isinstance(input_value, str):
                    tool_parameters = json.loads(input_value.replace("'", '"'))
                elif isinstance(input_value, dict):
                    tool_parameters = input_value
            except json.JSONDecodeError:
                tool_parameters = {}

        # Get output from attributes
        output_value = attrs.get("output.value")
        if output_value:
            try:
                if isinstance(output_value, str):
                    parsed = json.loads(output_value)
                    tool_output_content = parsed.get("content", str(parsed))
                    tool_call_id = parsed.get("tool_call_id")
                    tool_status = parsed.get("status", "success")
                elif isinstance(output_value, dict):
                    tool_output_content = output_value.get("content", str(output_value))
            except json.JSONDecodeError:
                tool_output_content = str(output_value)

        # Fallback to span_events format (CloudWatch/ADOT)
        if not tool_name or tool_parameters is None or tool_output_content is None:
            input_messages, output_messages = self._get_messages_from_span_events(span)

            # Extract tool parameters from input
            if input_messages and tool_parameters is None:
                input_msg = input_messages[-1]
                if input_msg.get("role") == "user":
                    content_str = input_msg.get("content", "")
                    if isinstance(content_str, str):
                        try:
                            parsed = json.loads(content_str.replace("'", '"'))
                            if isinstance(parsed, dict):
                                tool_parameters = parsed
                        except json.JSONDecodeError:
                            pass

            # Extract tool result from output
            if output_messages and tool_output_content is None:
                output_msg = output_messages[-1]
                if output_msg.get("role") == "assistant":
                    content_str = output_msg.get("content", "")
                    if isinstance(content_str, str):
                        try:
                            tool_data = json.loads(content_str)
                            tool_output_content = tool_data.get("content")
                            tool_call_id = tool_data.get("tool_call_id")
                            tool_status = tool_data.get("status")
                            if not tool_name:
                                tool_name = tool_data.get("name")
                        except json.JSONDecodeError:
                            pass

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
        """Convert OTEL span to AgentInvocationSpan."""
        span_info = self._create_span_info(span, session_id)
        trace_id = span.get("trace_id", "")

        input_messages, output_messages = self._get_messages_from_span_events(span)

        user_prompt = self._extract_user_prompt(input_messages, span)
        agent_response = self._extract_agent_response(output_messages, span)

        if not user_prompt:
            logger.warning(f"No user_prompt for agent span {span.get('span_id')}")
            return None

        if not agent_response:
            logger.warning(f"No agent_response for agent span {span.get('span_id')}")
            return None

        available_tools = sorted(
            self._trace_tools_map.get(trace_id, {}).values(),
            key=lambda t: t.name,
        )

        return AgentInvocationSpan(
            span_info=span_info,
            user_prompt=user_prompt,
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
            if value > 1e12:
                value = value / 1e9
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return datetime.now(timezone.utc)

    def _get_messages_from_span_events(self, span: dict) -> tuple[list[dict], list[dict]]:
        """Extract input and output messages from span events or attributes.

        Handles two formats:
        1. CloudWatch/ADOT: span_events[].body.input/output.messages
        2. Live instrumentation: attributes.input.value / output.value
        """
        # Try span_events first (CloudWatch/ADOT format)
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

        # Fallback: try attributes (live instrumentation format)
        attrs = span.get("attributes", {})
        input_messages = []
        output_messages = []

        # Parse input.value
        input_value = attrs.get("input.value")
        if input_value:
            try:
                parsed = json.loads(input_value) if isinstance(input_value, str) else input_value
                if isinstance(parsed, dict):
                    # Format: {"messages": [...]}
                    if "messages" in parsed:
                        input_messages = [{"content": json.dumps(parsed), "role": "user"}]
                    else:
                        input_messages = [{"content": json.dumps(parsed), "role": "user"}]
            except (json.JSONDecodeError, TypeError):
                pass

        # Parse output.value
        output_value = attrs.get("output.value")
        if output_value:
            try:
                parsed = json.loads(output_value) if isinstance(output_value, str) else output_value
                output_messages = [
                    {
                        "content": json.dumps(parsed) if isinstance(parsed, (dict, list)) else str(parsed),
                        "role": "assistant",
                    }
                ]
            except (json.JSONDecodeError, TypeError):
                pass

        return input_messages, output_messages

    def _extract_tools_from_attributes(self, attrs: dict) -> list[ToolConfig]:
        """Extract tool definitions from llm.tools.*.tool.json_schema attributes."""
        tools: list[ToolConfig] = []

        for key, value in attrs.items():
            if key.startswith("llm.tools.") and key.endswith(".tool.json_schema"):
                try:
                    tool_info = json.loads(value) if isinstance(value, str) else value
                    tools.append(
                        ToolConfig(
                            name=tool_info.get("name"),
                            description=tool_info.get("description"),
                            parameters=tool_info.get("input_schema"),
                        )
                    )
                except (json.JSONDecodeError, AttributeError):
                    pass

        return sorted(tools, key=lambda t: t.name or "")

    def _extract_user_contents(
        self, input_messages: list[dict], span: dict
    ) -> tuple[list[TextContent | ToolResultContent], str | None]:
        """Extract user contents from input messages or attributes."""
        results: list[TextContent | ToolResultContent] = []
        system_prompt: str | None = None
        attrs = span.get("attributes", {})

        # First, try to get from llm.input_messages attributes (live format)
        user_content = attrs.get("llm.input_messages.0.message.content")
        if user_content:
            results.append(TextContent(text=user_content))
            return results, system_prompt

        # Try to find structured message list from input_messages
        structured_messages = None
        for msg in input_messages:
            if msg.get("role") != "user":
                continue
            try:
                content_json = json.loads(msg.get("content", ""))
                if "messages" in content_json:
                    structured_messages = content_json["messages"]
                    break
            except (json.JSONDecodeError, TypeError):
                pass

        if structured_messages:
            # Handle nested list
            if (
                isinstance(structured_messages, list)
                and structured_messages
                and isinstance(structured_messages[0], list)
            ):
                structured_messages = structured_messages[0]

            for item in structured_messages:
                if not isinstance(item, dict):
                    continue
                kwargs = item.get("kwargs", {})
                data_type = kwargs.get("type")

                if data_type == "human":
                    results.append(TextContent(text=kwargs.get("content", "")))
                elif data_type == "tool":
                    status = kwargs.get("status")
                    results.append(
                        ToolResultContent(
                            content=kwargs.get("content", ""),
                            tool_call_id=kwargs.get("tool_call_id"),
                            error=None if status == "success" else status,
                        )
                    )
                elif data_type == "system":
                    system_prompt = kwargs.get("content")
        else:
            # Fallback: use raw content
            for msg in input_messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    # Try to parse if it's a JSON string
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "messages" in parsed:
                            # Extract first human message
                            for m in parsed.get("messages", []):
                                if isinstance(m, dict):
                                    c = m.get("content", "")
                                    if c:
                                        results.append(TextContent(text=c))
                                        break
                        else:
                            results.append(TextContent(text=content))
                    except (json.JSONDecodeError, TypeError):
                        results.append(TextContent(text=content))
                    break

        return results, system_prompt

    def _extract_assistant_contents(
        self, output_messages: list[dict], span: dict
    ) -> list[TextContent | ToolCallContent]:
        """Extract assistant contents from output messages or attributes."""
        results: list[TextContent | ToolCallContent] = []
        attrs = span.get("attributes", {})

        # First, try to get from llm.output_messages attributes (live format)
        assistant_content = attrs.get("llm.output_messages.0.message.content")
        if assistant_content:
            results.append(TextContent(text=assistant_content))

            # Check for tool calls in attributes
            tool_idx = 0
            while True:
                tool_name = attrs.get(f"llm.output_messages.0.message.tool_calls.{tool_idx}.tool_call.function.name")
                if not tool_name:
                    break
                tool_args_str = attrs.get(
                    f"llm.output_messages.0.message.tool_calls.{tool_idx}.tool_call.function.arguments", "{}"
                )
                tool_id = attrs.get(f"llm.output_messages.0.message.tool_calls.{tool_idx}.tool_call.id")
                try:
                    tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                except json.JSONDecodeError:
                    tool_args = {}
                results.append(ToolCallContent(name=tool_name, arguments=tool_args, tool_call_id=tool_id))
                tool_idx += 1

            if results:
                return results

        # Try parsing from output_messages
        for msg in output_messages:
            if msg.get("role") != "assistant":
                continue

            content_str = msg.get("content", "")
            if not isinstance(content_str, str):
                content_str = json.dumps(content_str)

            try:
                gen = json.loads(content_str)
                if "generations" in gen:
                    gen_item = gen["generations"][0][0]
                    results.append(TextContent(text=gen_item.get("text", "")))

                    # Extract tool calls
                    kwargs = gen_item.get("message", {}).get("kwargs", {})
                    for call in kwargs.get("tool_calls", []):
                        results.append(
                            ToolCallContent(
                                name=call.get("name", ""),
                                arguments=call.get("args", {}),
                                tool_call_id=call.get("id"),
                            )
                        )
                    return results
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                pass

        # Fallback: use raw content
        for msg in output_messages:
            if msg.get("role") == "assistant":
                content_str = msg.get("content", "")
                if isinstance(content_str, str):
                    results.append(TextContent(text=content_str))

        return results

    def _extract_user_prompt(self, input_messages: list[dict], span: dict) -> str | None:
        """Extract user prompt from agent invocation input."""
        for msg in input_messages:
            if msg.get("role") != "user":
                continue

            content_str = msg.get("content", "")
            if not isinstance(content_str, str):
                continue

            try:
                j = json.loads(content_str)
                if "messages" in j:
                    messages = j["messages"]
                    for item in messages:
                        if isinstance(item, dict):
                            kwargs = item.get("kwargs", {})
                            if kwargs.get("type") == "human":
                                return kwargs.get("content", "")
                            if item.get("type") == "human":
                                return item.get("content", "")
                            if item.get("role") == "user":
                                return item.get("content", "")
            except json.JSONDecodeError:
                return content_str

        return None

    def _extract_agent_response(self, output_messages: list[dict], span: dict) -> str | None:
        """Extract agent response from agent invocation output."""
        for msg in output_messages:
            if msg.get("role") != "assistant":
                continue

            content_str = msg.get("content", "")
            if not isinstance(content_str, str):
                content_str = json.dumps(content_str)

            try:
                j = json.loads(content_str)

                # Check for messages structure
                if "messages" in j:
                    messages = j["messages"]
                    if isinstance(messages, list):
                        # Iterate backwards to find last AI message
                        for item in reversed(messages):
                            if isinstance(item, dict):
                                kwargs = item.get("kwargs", {}) if "kwargs" in item else item
                                if item.get("type") == "ai" or kwargs.get("type") == "ai":
                                    return kwargs.get("content", "")

                # Check for generations structure
                if "generations" in j:
                    return j["generations"][0][0].get("text", "")

            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                pass

        return None
