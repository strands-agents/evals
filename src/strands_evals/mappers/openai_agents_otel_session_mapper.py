"""OpenAI Agents session mapper - Maps Traceloop/OpenLLMetry OpenAI Agents spans to Session format.

Handles traces with scope: opentelemetry.instrumentation.openai_agents
(produced by opentelemetry-instrumentation-openai-agents from Traceloop's OpenLLMetry project)

Supports two trace formats:
1. ADOT/CloudWatch: Messages in gen_ai.* attributes
2. Live instrumentation: Messages in gen_ai.* attributes
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
    Overrides workflow-span skipping and tool back-filling which assume a single-agent-per-
    workflow trace shape and would break multi-agent traces or frameworks that use empty
    invoke_agent spans as session anchors if applied in the base class.
    """

    def map_to_session(self, data: Any, session_id: str) -> Session:
        """Map OpenAI Agents SDK spans to Session format.

        Args:
            data: Trace data as a flat list of span dicts, grouped dict, or trace objects.
            session_id: Session identifier.

        Returns:
            Session object ready for evaluation.
        """
        # Normalize input to flat spans
        spans = self._normalize_to_flat_spans(data)

        # Filter to only spans from this scope
        openai_agents_spans = [s for s in spans if get_scope_name(s) in (SCOPE_OPENAI_AGENTS, "")]

        return super().map_to_session(openai_agents_spans, session_id)

    def _convert_trace(self, trace_id: str, spans: list[dict], session_id: str) -> Trace:
        """Convert a list of dict spans to a Trace, with per-agent tool back-filling.

        After base conversion (which sets agent_span_id via Trace.model_post_init),
        assigns each ToolExecutionSpan's tool only to its owning AgentInvocationSpan.
        """
        trace = super()._convert_trace(trace_id, spans, session_id)

        # Back-fill available_tools scoped per agent using agent_span_id (set by model_post_init).
        agent_spans = {
            s.span_info.span_id: s for s in trace.spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id
        }

        for span in trace.spans:
            if isinstance(span, ToolExecutionSpan) and span.tool_call.name:
                owner_id = span.agent_span_id
                if owner_id and owner_id in agent_spans:
                    agent = agent_spans[owner_id]
                    if not any(t.name == span.tool_call.name for t in agent.available_tools):
                        agent.available_tools.append(ToolConfig(name=span.tool_call.name))

        return trace

    def _convert_agent_invocation_span(self, span: dict, span_info: SpanInfo) -> AgentInvocationSpan | None:
        """Convert an 'invoke_agent' span to AgentInvocationSpan.

        Skips the root "Agent workflow" span emitted by Traceloop's instrumentation.
        This span has invoke_agent operation but no agent_name or input messages —
        it's a structural wrapper, not a real agent invocation.
        """
        attrs = span.get("attributes", {})

        if not attrs.get("gen_ai.agent.name") and not attrs.get("gen_ai.input.messages"):
            return None

        return super()._convert_agent_invocation_span(span, span_info)
