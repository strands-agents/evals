"""OpenAI Agents session mapper - Maps Traceloop/OpenLLMetry OpenAI Agents spans to Session format.

Handles traces with scope: opentelemetry.instrumentation.openai_agents
(produced by opentelemetry-instrumentation-openai-agents from Traceloop's OpenLLMetry project)

Supports two trace formats:
1. Live instrumentation: Messages in gen_ai.* attributes
2. ADOT/CloudWatch: the same as above plus `aws.*` attributes
"""

import logging
from typing import Any

from ..types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    ToolConfig,
    ToolExecutionSpan,
    Trace,
)
from .constants import SCOPE_OPENAI_AGENTS
from .generic_gen_ai_session_mapper import GenericGenAISessionMapper
from .utils import get_scope_name

logger = logging.getLogger(__name__)


class OpenAIAgentsOtelSessionMapper(GenericGenAISessionMapper):
    """Maps OpenAI Agents SDK OTel spans to Session format.

    Inherits from GenericGenAISessionMapper to reuse GenAI semantic conventions parsing.
    """

    def map_to_session(self, data: Any, session_id: str) -> Session:
        """Map OpenAI Agents SDK spans to Session format.

        Args:
            data: Trace data as a flat list of span dicts, grouped dict, or trace objects.
            session_id: Session identifier.

        Returns:
            Session object ready for evaluation.
        """
        spans = self._normalize_to_flat_spans(data)

        openai_agents_spans = [s for s in spans if get_scope_name(s) in (SCOPE_OPENAI_AGENTS, "")]

        return super().map_to_session(openai_agents_spans, session_id)

    def _convert_trace(self, trace_id: str, spans: list[dict], session_id: str) -> Trace:
        """Convert a list of dict spans to a Trace."""
        trace = super()._convert_trace(trace_id, spans, session_id)

        agent_spans = {
            s.span_info.span_id: s for s in trace.spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id
        }

        self._apply_handoff_reparenting(agent_spans, spans)

        # Back-fill agent's available_tools, which is not otherwise provided in the trace.
        for span in trace.spans:
            if isinstance(span, ToolExecutionSpan) and span.tool_call.name:
                owner_id = span.agent_span_id
                if owner_id and owner_id in agent_spans:
                    agent = agent_spans[owner_id]
                    if not any(t.name == span.tool_call.name for t in agent.available_tools):
                        agent.available_tools.append(ToolConfig(name=span.tool_call.name))

        return trace

    def _apply_handoff_reparenting(
        self,
        agent_spans: dict[str, AgentInvocationSpan],
        raw_spans: list[dict],
    ) -> None:
        """Re-parent agent invocation spans using OpenAI Agent's agent_handoff span.

        Uses agent name as span identity (last-write-wins for duplicate names or
        competing handoffs).
        """
        agent_spans_by_name: dict[str, AgentInvocationSpan] = {}
        handoffs: list[tuple[str, str]] = []

        # Collect converted agent spans and handoff edges
        for raw_span in raw_spans:
            attrs = raw_span.get("attributes", {})
            op = attrs.get("gen_ai.operation.name", "")
            if op == "invoke_agent":
                name = attrs.get("gen_ai.agent.name", "")
                span_id = raw_span.get("span_id", "")
                if name and span_id and span_id in agent_spans:
                    agent_spans_by_name[name] = agent_spans[span_id]
            elif op == "agent_handoff":
                from_name = attrs.get("gen_ai.handoff.from_agent", "")
                to_name = attrs.get("gen_ai.handoff.to_agent", "")
                if from_name and to_name:
                    handoffs.append((from_name, to_name))

        # Re-parent sub-agents to parent agents
        for from_name, to_name in handoffs:
            from_span = agent_spans_by_name.get(from_name)
            to_span = agent_spans_by_name.get(to_name)
            if from_span and to_span and from_span.span_info.span_id:
                to_span.span_info.parent_span_id = from_span.span_info.span_id

    def _convert_agent_invocation_span(self, span: dict, span_info: SpanInfo) -> AgentInvocationSpan | None:
        """Convert an 'invoke_agent' span to AgentInvocationSpan.

        Skips the root "Agent workflow" wrapper emitted by Traceloop's instrumentation.
        """
        attrs = span.get("attributes", {})

        # Agent workflow span has no agent_name or input messages
        if not attrs.get("gen_ai.agent.name") and not attrs.get("gen_ai.input.messages"):
            return None

        return super()._convert_agent_invocation_span(span, span_info)
