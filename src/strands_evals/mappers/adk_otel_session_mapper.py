"""Google ADK session mapper - converts ADK OTel spans to Session format.

Google ADK (Agent Development Kit) produces OpenTelemetry spans with `gen_ai.*` and
`gcp.vertex.agent.*` attributes under the instrumentation scope `gcp.vertex.agent`.
Detection uses scope name as the primary signal.

Span hierarchy (single tool-use turn):
    invocation (root)
      invoke_agent <agent_name>
        call_llm
          generate_content <model>
            execute_tool <tool_name>
        call_llm
          generate_content <model>

The `call_llm` spans carry the full serialized LLM request/response as JSON strings in
`gcp.vertex.agent.llm_request` and `gcp.vertex.agent.llm_response`. These are the primary
data source for reconstructing messages, system prompts, and tool definitions.

Limitations:
    - tool_call_id on InferenceSpan messages requires Gemini 3+; Gemini 2.x
      yields None. ToolExecutionSpan always has the ID from gen_ai.tool.call.id.
    - The skip_summarization agent_response fallback exposes raw tool_response
      JSON (e.g. '{"result": "555"}'), not display output. ADK telemetry has no
      separate display-output field.
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
from .constants import SCOPE_ADK
from .session_mapper import SessionMapper
from .utils import bridge_parent_gaps, get_scope_name, safe_json_parse

logger = logging.getLogger(__name__)


class ADKOtelSessionMapper(SessionMapper):
    """Maps Google ADK OTel spans to Session format.

    This mapper handles traces produced by the Google ADK framework. ADK spans use:
    - `gen_ai.operation.name` for span type detection (`invoke_agent`, `call_llm`,
      `generate_content`, `execute_tool`)
    - `gcp.vertex.agent.llm_request` / `gcp.vertex.agent.llm_response` for full
      request/response payloads (JSON strings on `call_llm` spans)
    - `gcp.vertex.agent.tool_call_args` / `gcp.vertex.agent.tool_response` for tool I/O
    """

    def map_to_session(self, data: Any, session_id: str) -> Session:
        """Map ADK spans to Session format.

        Args:
            data: Trace data in various formats:
                - Flat list of spans: [{"trace_id": "x", "span_id": "y", ...}, ...]
                - Grouped by trace_id: {"trace_1": [spans], "trace_2": [spans]}
                - List of trace objects: [{"trace_id": "x", "spans": [...]}, ...]
            session_id: Session identifier.

        Returns:
            Session object ready for evaluation.
        """
        spans = self._normalize_to_flat_spans(data)

        # Filter to only spans from the ADK instrumentation scope.
        # Include spans with no scope (e.g. to_json format) since this mapper
        # was explicitly selected for ADK traces.
        spans = [s for s in spans if get_scope_name(s) in (SCOPE_ADK, "")]

        # Group spans by trace_id
        grouped: dict[str, list[dict]] = defaultdict(list)
        for span in spans:
            trace_id = self._extract_trace_id(span)
            if trace_id:
                grouped[trace_id].append(span)

        result_traces: list[Trace] = []
        for trace_id, trace_spans in grouped.items():
            trace = self._build_trace(trace_id, trace_spans, session_id)
            if trace.spans:
                result_traces.append(trace)

        # Sort traces chronologically by earliest span start_time
        result_traces.sort(key=lambda t: min(s.span_info.start_time for s in t.spans))

        return Session(session_id=session_id, traces=result_traces)

    def _build_trace(self, trace_id: str, spans: list[dict], session_id: str) -> Trace:
        """Build a Trace from spans sharing the same trace_id."""
        spans_by_id: dict[str, dict] = {}
        children_by_parent: dict[str, list[dict]] = defaultdict(list)

        for span in spans:
            span_id = self._extract_span_id(span)
            if span_id:
                spans_by_id[span_id] = span
            parent_span_id = self._extract_parent_span_id(span)
            if parent_span_id:
                children_by_parent[parent_span_id].append(span)

        converted_spans: list[InferenceSpan | ToolExecutionSpan | AgentInvocationSpan] = []

        for span in spans:
            operation_name = self._get_operation_name(span)

            try:
                if operation_name == "invoke_agent":
                    agent_span = self._convert_agent_invocation_span(span, session_id, children_by_parent)
                    if agent_span:
                        converted_spans.append(agent_span)
                elif operation_name == "generate_content":
                    inference_span = self._convert_inference_span(span, session_id, spans_by_id)
                    if inference_span:
                        converted_spans.append(inference_span)
                elif operation_name == "execute_tool":
                    converted_tool = self._convert_tool_execution_span(span, session_id)
                    if converted_tool:
                        converted_spans.append(converted_tool)
                # Skip `call_llm` and `invocation` — data sources, not evals span types
            except Exception as e:
                span_id = self._extract_span_id(span) or "unknown"
                logger.warning("span_id=<%s>, error=<%s> | failed to convert ADK span", span_id, e)

        # Fix parent_span_id on converted spans that point to skipped intermediaries
        raw_parent_map: dict[str, str | None] = {
            self._extract_span_id(s): self._extract_parent_span_id(s) for s in spans if self._extract_span_id(s)
        }

        converted_spans.sort(key=lambda s: s.span_info.start_time)
        bridge_parent_gaps(converted_spans, raw_parent_map)

        return Trace(spans=converted_spans, trace_id=trace_id, session_id=session_id)

    # =========================================================================
    # Span Type Detection
    # =========================================================================

    def _get_operation_name(self, span: dict) -> str:
        """Get gen_ai.operation.name from span attributes."""
        attrs = span.get("attributes") or {}
        return attrs.get("gen_ai.operation.name", "")

    def _is_call_llm_span(self, span: dict) -> bool:
        """Check if a span is a `call_llm` span.

        ADK `call_llm` spans don't set `gen_ai.operation.name`; they are identified
        by their name field or having a non-empty `gcp.vertex.agent.llm_request`.
        """
        name = span.get("name", "")
        if name == "call_llm" or name.startswith("call_llm "):
            return True
        attrs = span.get("attributes", {})
        llm_request = attrs.get("gcp.vertex.agent.llm_request", "")
        return bool(llm_request) and llm_request != "{}"

    # =========================================================================
    # Span Conversion
    # =========================================================================

    def _convert_agent_invocation_span(
        self,
        span: dict,
        session_id: str,
        children_by_parent: dict[str, list[dict]],
    ) -> AgentInvocationSpan | None:
        """Convert an ADK `invoke_agent` span to AgentInvocationSpan.

        User prompt, agent response, system prompt, and available tools are
        extracted from child `call_llm` spans.
        """
        span_info = self._create_span_info(span, session_id)
        attrs = span.get("attributes", {})

        span_id = self._extract_span_id(span)
        call_llm_spans = sorted(
            [child for child in children_by_parent.get(span_id, []) if self._is_call_llm_span(child)],
            key=lambda s: self.parse_timestamp(s.get("start_time")),
        )

        user_prompt = ""
        system_prompt: str | None = None
        available_tools: list[ToolConfig] = []
        agent_response = ""

        # System prompt and tools from first call_llm; user_prompt from last (full history).
        if call_llm_spans:
            first_request = self._parse_llm_request(call_llm_spans[0])
            if first_request:
                system_prompt = self._extract_system_prompt_from_request(first_request)
                available_tools = self._extract_tools_from_request(first_request)

            last_request = self._parse_llm_request(call_llm_spans[-1])
            if last_request:
                user_prompt = self._extract_user_prompt_from_request(last_request)

        # Skip preamble text when the response contains a function_call.
        if call_llm_spans:
            llm_response = self._parse_llm_response(call_llm_spans[-1])
            if llm_response:
                response_parts = llm_response.get("content", {}).get("parts", [])
                has_function_call = any("function_call" in part for part in response_parts)
                if not has_function_call:
                    agent_response = self._extract_text_from_response(llm_response)

        # Fallback: use the last tool result from the invocation subtree.
        if not agent_response:
            tool_descendants = sorted(
                [
                    desc
                    for desc in self._get_descendants(span_id, children_by_parent, stop_at_agents=True)
                    if self._get_operation_name(desc) == "execute_tool"
                    and desc.get("attributes", {}).get("gen_ai.tool.name") != "(merged tools)"
                ],
                key=lambda s: self.parse_timestamp(s.get("start_time")),
            )
            if tool_descendants:
                last_tool_attrs = tool_descendants[-1].get("attributes", {})
                agent_response = last_tool_attrs.get("gcp.vertex.agent.tool_response", "")

        if not user_prompt and not agent_response:
            return None

        metadata: dict[str, Any] = {}
        if attrs.get("gen_ai.agent.name"):
            metadata["agent_name"] = attrs["gen_ai.agent.name"]
        if attrs.get("gen_ai.agent.description"):
            metadata["agent_description"] = attrs["gen_ai.agent.description"]

        return AgentInvocationSpan(
            span_info=span_info,
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=available_tools,
            system_prompt=system_prompt,
            metadata=metadata,
        )

    def _convert_inference_span(
        self,
        span: dict,
        session_id: str,
        spans_by_id: dict[str, dict],
    ) -> InferenceSpan | None:
        """Convert an ADK `generate_content` span to InferenceSpan.

        Messages are reconstructed from the parent `call_llm` span's
        llm_request/llm_response attributes.
        """
        span_info = self._create_span_info(span, session_id)
        attrs = span.get("attributes", {})

        # Find parent call_llm span
        parent_span_id = self._extract_parent_span_id(span)
        parent_span = spans_by_id.get(parent_span_id, {}) if parent_span_id else {}

        messages: list[UserMessage | AssistantMessage] = []

        if parent_span:
            llm_request = self._parse_llm_request(parent_span)
            llm_response = self._parse_llm_response(parent_span)

            if llm_request:
                messages.extend(self._extract_messages_from_request(llm_request))

            if llm_response:
                assistant_msg = self._extract_assistant_message_from_response(llm_response)
                if assistant_msg:
                    messages.append(assistant_msg)

        if not messages:
            return None

        metadata: dict[str, Any] = {}
        if attrs.get("gen_ai.system"):
            metadata["gen_ai.system"] = attrs["gen_ai.system"]
        if attrs.get("gen_ai.request.model"):
            metadata["model"] = attrs["gen_ai.request.model"]
        if attrs.get("gen_ai.agent.name"):
            metadata["agent_name"] = attrs["gen_ai.agent.name"]
        if attrs.get("gen_ai.usage.input_tokens") is not None:
            metadata["input_tokens"] = attrs["gen_ai.usage.input_tokens"]
        if attrs.get("gen_ai.usage.output_tokens") is not None:
            metadata["output_tokens"] = attrs["gen_ai.usage.output_tokens"]
        if attrs.get("gen_ai.usage.reasoning.output_tokens") is not None:
            metadata["reasoning_tokens"] = attrs["gen_ai.usage.reasoning.output_tokens"]
        if attrs.get("gen_ai.response.finish_reasons"):
            metadata["finish_reasons"] = list(attrs["gen_ai.response.finish_reasons"])
        if attrs.get("gcp.vertex.agent.invocation_id"):
            metadata["invocation_id"] = attrs["gcp.vertex.agent.invocation_id"]
        if attrs.get("gcp.vertex.agent.event_id"):
            metadata["event_id"] = attrs["gcp.vertex.agent.event_id"]

        return InferenceSpan(span_info=span_info, messages=messages, metadata=metadata)

    def _convert_tool_execution_span(self, span: dict, session_id: str) -> ToolExecutionSpan | None:
        """Convert an ADK `execute_tool` span to ToolExecutionSpan."""
        span_info = self._create_span_info(span, session_id)
        attrs = span.get("attributes", {})

        tool_name = attrs.get("gen_ai.tool.name", "")
        tool_call_id = attrs.get("gen_ai.tool.call.id")

        tool_parameters = safe_json_parse(attrs.get("gcp.vertex.agent.tool_call_args", "{}"))
        if not isinstance(tool_parameters, dict):
            tool_parameters = {}

        tool_response_raw = attrs.get("gcp.vertex.agent.tool_response", "")
        tool_output_content = tool_response_raw if isinstance(tool_response_raw, str) else str(tool_response_raw)

        if not tool_name:
            return None

        if tool_name == "(merged tools)":
            return None

        tool_error: str | None = attrs.get("error.type") or None
        if not tool_error:
            span_status = span.get("status", {})
            if isinstance(span_status, dict) and span_status.get("code") == "ERROR":
                tool_error = span_status.get("description") or "error"

        tool_call = ToolCall(name=tool_name, arguments=tool_parameters, tool_call_id=tool_call_id)
        tool_result = ToolResult(content=tool_output_content, error=tool_error, tool_call_id=tool_call_id)

        metadata: dict[str, Any] = {}
        if attrs.get("gen_ai.tool.description"):
            metadata["description"] = attrs["gen_ai.tool.description"]
        if attrs.get("gen_ai.tool.type"):
            metadata["tool_type"] = attrs["gen_ai.tool.type"]
        if attrs.get("gcp.vertex.agent.event_id"):
            metadata["event_id"] = attrs["gcp.vertex.agent.event_id"]

        return ToolExecutionSpan(span_info=span_info, tool_call=tool_call, tool_result=tool_result, metadata=metadata)

    # =========================================================================
    # Data Extraction Helpers
    # =========================================================================

    def _parse_llm_request(self, call_llm_span: dict) -> dict | None:
        """Parse the gcp.vertex.agent.llm_request JSON attribute from a call_llm span."""
        attrs = call_llm_span.get("attributes", {})
        raw = attrs.get("gcp.vertex.agent.llm_request", "")
        if not raw or raw == "{}":
            return None
        return safe_json_parse(raw) if isinstance(raw, str) else None

    def _parse_llm_response(self, call_llm_span: dict) -> dict | None:
        """Parse the gcp.vertex.agent.llm_response JSON attribute from a call_llm span."""
        attrs = call_llm_span.get("attributes", {})
        raw = attrs.get("gcp.vertex.agent.llm_response", "")
        if not raw or raw == "{}":
            return None
        return safe_json_parse(raw) if isinstance(raw, str) else None

    def _extract_user_prompt_from_request(self, llm_request: dict) -> str:
        """Extract the latest user text from llm_request.contents.

        ADK requests carry accumulated conversation history; the last user
        message is the prompt that triggered this invocation.
        """
        for content_item in reversed(llm_request.get("contents", [])):
            if content_item.get("role") == "user":
                texts = [
                    part["text"] for part in content_item.get("parts", []) if "text" in part and not part.get("thought")
                ]
                if texts:
                    return "".join(texts)
        return ""

    def _extract_system_prompt_from_request(self, llm_request: dict) -> str | None:
        """Extract system_instruction from llm_request.config."""
        config = llm_request.get("config", {})
        return config.get("system_instruction")

    def _extract_tools_from_request(self, llm_request: dict) -> list[ToolConfig]:
        """Extract tool definitions from llm_request.config.tools."""
        available_tools: list[ToolConfig] = []
        config = llm_request.get("config", {})
        for tool_group in config.get("tools", []):
            for func_decl in tool_group.get("function_declarations", []):
                available_tools.append(
                    ToolConfig(
                        name=func_decl.get("name", ""),
                        description=func_decl.get("description"),
                        parameters=func_decl.get("parameters_json_schema") or func_decl.get("parameters"),
                    )
                )
        return available_tools

    def _extract_text_from_response(self, llm_response: dict) -> str:
        """Extract visible text from llm_response, filtering out thought parts."""
        content = llm_response.get("content", {})
        texts = [part["text"] for part in content.get("parts", []) if "text" in part and not part.get("thought")]
        return "".join(texts)

    def _extract_messages_from_request(
        self,
        llm_request: dict,
    ) -> list[UserMessage | AssistantMessage]:
        """Extract typed messages from llm_request.contents (Gemini format).

        Reads function_call.id / function_response.id when present (Gemini 3+);
        for Gemini 2.x models these fields are absent and tool_call_id is None.
        """
        messages: list[UserMessage | AssistantMessage] = []

        for content_item in llm_request.get("contents", []):
            role = content_item.get("role", "")
            parts = content_item.get("parts", [])

            if role == "user":
                user_content: list[TextContent | ToolResultContent] = []
                for part in parts:
                    if "text" in part and not part.get("thought"):
                        user_content.append(TextContent(text=part["text"]))
                    elif "function_response" in part:
                        func_resp = part["function_response"]
                        user_content.append(
                            ToolResultContent(
                                content=json.dumps(func_resp.get("response", {})),
                                tool_call_id=func_resp.get("id"),
                            )
                        )
                if user_content:
                    messages.append(UserMessage(content=user_content))

            elif role == "model":
                assistant_content: list[TextContent | ToolCallContent] = []
                for part in parts:
                    if "text" in part and not part.get("thought"):
                        assistant_content.append(TextContent(text=part["text"]))
                    elif "function_call" in part:
                        func_call = part["function_call"]
                        assistant_content.append(
                            ToolCallContent(
                                name=func_call.get("name", ""),
                                arguments=func_call.get("args", {}),
                                tool_call_id=func_call.get("id"),
                            )
                        )
                if assistant_content:
                    messages.append(AssistantMessage(content=assistant_content))

        return messages

    def _extract_assistant_message_from_response(
        self,
        llm_response: dict,
    ) -> AssistantMessage | None:
        """Extract assistant message from llm_response.content.parts."""
        content = llm_response.get("content", {})
        parts = content.get("parts", [])

        assistant_content: list[TextContent | ToolCallContent] = []
        for part in parts:
            if "text" in part and not part.get("thought"):
                assistant_content.append(TextContent(text=part["text"]))
            elif "function_call" in part:
                func_call = part["function_call"]
                assistant_content.append(
                    ToolCallContent(
                        name=func_call.get("name", ""),
                        arguments=func_call.get("args", {}),
                        tool_call_id=func_call.get("id"),
                    )
                )

        return AssistantMessage(content=assistant_content) if assistant_content else None

    # =========================================================================
    # Common Helpers
    # =========================================================================

    def _create_span_info(self, span: dict, session_id: str) -> SpanInfo:
        """Create SpanInfo from an ADK span dict."""
        return SpanInfo(
            trace_id=self._extract_trace_id(span),
            span_id=self._extract_span_id(span),
            session_id=session_id,
            parent_span_id=self._extract_parent_span_id(span),
            start_time=self.parse_timestamp(span.get("start_time")),
            end_time=self.parse_timestamp(span.get("end_time")),
        )

    def _extract_trace_id(self, span: dict) -> str:
        """Extract trace_id from span dict.

        Falls back to span["context"]["trace_id"] for the to_json export format.
        """
        trace_id = span.get("trace_id", "")
        if not trace_id:
            context = span.get("context", {})
            if isinstance(context, dict):
                trace_id = context.get("trace_id", "")
        return self._strip_hex_prefix(trace_id)

    def _get_descendants(
        self,
        span_id: str,
        children_by_parent: dict[str, list[dict]],
        stop_at_agents: bool = False,
    ) -> list[dict]:
        """Get all descendants of a span by traversing the parent-child hierarchy.

        Args:
            span_id: Root span to start traversal from.
            children_by_parent: Full parent-child index.
            stop_at_agents: If True, stop traversal at nested invoke_agent spans
                (don't include them or their descendants). Used in multi-agent
                scenarios to scope results to a single agent's subtree.
        """
        descendants: list[dict] = []
        queue = list(children_by_parent.get(span_id, []))
        while queue:
            child = queue.pop(0)
            if stop_at_agents and self._get_operation_name(child) == "invoke_agent":
                continue
            descendants.append(child)
            child_id = self._extract_span_id(child)
            if child_id:
                queue.extend(children_by_parent.get(child_id, []))
        return descendants

    def _extract_span_id(self, span: dict) -> str:
        """Extract span_id from span dict.

        Falls back to span["context"]["span_id"] for the to_json export format.
        """
        span_id = span.get("span_id", "")
        if not span_id:
            context = span.get("context", {})
            if isinstance(context, dict):
                span_id = context.get("span_id", "")
        return self._strip_hex_prefix(span_id)

    def _extract_parent_span_id(self, span: dict) -> str | None:
        """Extract parent_span_id, falling back to span["parent_id"]."""
        parent = span.get("parent_span_id") or span.get("parent_id")
        if parent is None:
            return None
        return self._strip_hex_prefix(str(parent))

    @staticmethod
    def _strip_hex_prefix(value: Any) -> str:
        """Strip `0x` prefix from hex IDs if present."""
        value = str(value)
        if value.startswith("0x"):
            return value[2:]
        return value
