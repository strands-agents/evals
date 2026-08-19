"""OpenAI Agents GenAI session mapper - converts OpenAI Agents SDK spans to Session format.

Handles traces produced by the OpenAI Agents SDK and instrumented with
Traceloop/OpenLLMetry (scope: opentelemetry.instrumentation.openai_agents), which emit
spans that mostly follow the OTel GenAI semantic conventions.

This mapper is private to avoid committing to a public-facing inheritance contract. This
lets future mappers use other extension mechanisms as requirements change.
"""

import logging
from collections import defaultdict

from ..types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    TextContent,
    ToolConfig,
    ToolExecutionSpan,
    Trace,
    _to_aware_utc,
)
from .constants import SCOPE_OPENAI_AGENTS
from .generic_gen_ai_session_mapper import GenericGenAISessionMapper
from .utils import get_scope_name

logger = logging.getLogger(__name__)


class _OpenAIAgentsGenAISessionMapper(GenericGenAISessionMapper):
    """Maps OpenAI Agents SDK spans (via Traceloop/OpenLLMetry) to Session format.

    This mapper handles traces with scope `opentelemetry.instrumentation.openai_agents`. It uses:
    - `gen_ai.operation.name` for span classification
    - `gen_ai.agent.name` to distinguish real agent spans from empty wrappers
    - `gen_ai.agent.tools` / `gen_ai.tool.definitions` for tool discovery
    - Child inference spans for prompt/response backfilling on agent spans
    """

    def _prepare_trace_spans(self, spans: list[dict]) -> list[dict]:
        """Suppress the Traceloop workflow wrapper span."""
        filtered: list[dict] = []
        for s in spans:
            attrs = s.get("attributes", {})
            is_empty_wrapper = (
                get_scope_name(s) == SCOPE_OPENAI_AGENTS
                and attrs.get("gen_ai.operation.name") == "invoke_agent"
                and not attrs.get("gen_ai.agent.name")
                and not attrs.get("gen_ai.input.messages")
            )
            if not is_empty_wrapper:
                filtered.append(s)
        return filtered

    def _enrich_trace(self, trace: Trace, spans: list[dict]) -> Trace:
        """Backfill OpenAI agent prompts, responses, and tools from child spans."""
        # Identify which spans come from OpenAI Agents
        openai_span_ids: set[str] = set()
        for raw in spans:
            sid = raw.get("span_id")
            if isinstance(sid, str) and get_scope_name(raw) == SCOPE_OPENAI_AGENTS:
                openai_span_ids.add(sid)

        agent_spans: dict[str, AgentInvocationSpan] = {
            s.span_info.span_id: s
            for s in trace.spans
            if isinstance(s, AgentInvocationSpan) and s.span_info.span_id in openai_span_ids
        }

        self._backfill_agent_tools(trace, agent_spans, spans, openai_span_ids)
        self._backfill_agent_prompts(trace, agent_spans, openai_span_ids)

        return trace

    def _backfill_agent_tools(
        self,
        trace: Trace,
        agent_spans: dict[str, AgentInvocationSpan],
        spans: list[dict],
        scoped_span_ids: set[str],
    ) -> None:
        """Populate available_tools on agent spans using their child spans."""
        # Fill agent spans' available tools from child chat spans' tool definitions.
        raw_attrs_by_id = {s.get("span_id"): s.get("attributes", {}) for s in spans}
        for span in trace.spans:
            if not isinstance(span, InferenceSpan) or span.span_info.span_id not in scoped_span_ids:
                continue
            agent = agent_spans.get(span.span_info.parent_span_id or "")
            if agent is None or agent.available_tools:
                continue
            configs = self._parse_tool_definitions(raw_attrs_by_id.get(span.span_info.span_id, {}))
            if configs:
                agent.available_tools = list(configs)

        # Fallback: get the names of tools the agent called.
        for span in trace.spans:
            if (
                isinstance(span, ToolExecutionSpan)
                and span.span_info.span_id in scoped_span_ids
                and span.tool_call.name
            ):
                agent = agent_spans.get(span.agent_span_id or "")
                if agent and not any(t.name == span.tool_call.name for t in agent.available_tools):
                    agent.available_tools.append(ToolConfig(name=span.tool_call.name))

    def _backfill_agent_prompts(
        self,
        trace: Trace,
        agent_spans: dict[str, AgentInvocationSpan],
        scoped_span_ids: set[str],
    ) -> None:
        """Backfill user_prompt and agent_response on OpenAI Agents invoke_agent spans."""
        if not agent_spans:
            return

        # Collect inference spans under each agent span
        inference_by_agent: dict[str, list[InferenceSpan]] = defaultdict(list)
        for span in trace.spans:
            if isinstance(span, InferenceSpan) and span.span_info.span_id in scoped_span_ids:
                parent_id = span.span_info.parent_span_id
                if parent_id and parent_id in agent_spans:
                    inference_by_agent[parent_id].append(span)

        # Modify the agent span using the first user text and last assistant text
        for agent_id, inference_spans in inference_by_agent.items():
            agent = agent_spans[agent_id]
            inference_spans.sort(key=lambda s: _to_aware_utc(s.span_info.start_time))

            if not agent.user_prompt:
                user_prompt = self._get_first_user_text(inference_spans)
                if user_prompt:
                    agent.user_prompt = user_prompt

            if not agent.agent_response:
                agent_response = self._get_last_assistant_text(inference_spans)
                if agent_response:
                    agent.agent_response = agent_response

    @staticmethod
    def _get_first_user_text(inference_spans: list[InferenceSpan]) -> str:
        """Return the first user text content across ordered inference spans."""
        for span in inference_spans:
            for msg in span.messages:
                if msg.role.value != "user":
                    continue
                for content in msg.content:
                    if isinstance(content, TextContent) and content.text:
                        return content.text
        return ""

    @staticmethod
    def _get_last_assistant_text(inference_spans: list[InferenceSpan]) -> str:
        """Return the last assistant text content across ordered inference spans."""
        for span in reversed(inference_spans):
            for msg in reversed(span.messages):
                if msg.role.value != "assistant":
                    continue
                for content in reversed(msg.content):
                    if isinstance(content, TextContent) and content.text:
                        return content.text
        return ""
