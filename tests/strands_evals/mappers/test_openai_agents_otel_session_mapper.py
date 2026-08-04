"""Tests for OpenAIAgentsOtelSessionMapper - OpenAI Agents SDK OTel spans → Session conversion."""

import json
from pathlib import Path

import pytest

from strands_evals.mappers import OpenAIAgentsOtelSessionMapper, detect_otel_mapper
from strands_evals.types.trace import AgentInvocationSpan, InferenceSpan, ToolExecutionSpan

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_LIVE_SPANS_FILE = _FIXTURES_DIR / "openai_agents_otel_live_spans.json"
_ADOT_SPANS_FILE = _FIXTURES_DIR / "openai_agents_otel_adot_spans.json"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def openai_agents_live_spans():
    """Load real OpenAI Agents SDK spans captured via Traceloop instrumentation."""
    with open(_LIVE_SPANS_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def openai_agents_adot_spans():
    """Load OpenAI Agents SDK spans as seen through ADOT/CloudWatch export."""
    with open(_ADOT_SPANS_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def live_session(openai_agents_live_spans):
    """Map live OpenAI Agents spans to a Session."""
    mapper = OpenAIAgentsOtelSessionMapper()
    return mapper.map_to_session(openai_agents_live_spans, session_id="test")


@pytest.fixture(scope="module")
def adot_session(openai_agents_adot_spans):
    """Map ADOT OpenAI Agents spans to a Session."""
    mapper = OpenAIAgentsOtelSessionMapper()
    data = openai_agents_adot_spans
    session_id = data.get("session_id", "test") if isinstance(data, dict) else "test"
    return mapper.map_to_session(data, session_id=session_id)


# =============================================================================
# Routing Tests
# =============================================================================


class TestRouting:
    """Verify detect_otel_mapper routes OpenAI Agents spans to OpenAIAgentsOtelSessionMapper."""

    def test_live_spans_route_to_openai_mapper(self, openai_agents_live_spans):
        """Live spans with openai_agents scope route to OpenAIAgentsOtelSessionMapper."""
        mapper = detect_otel_mapper(openai_agents_live_spans)
        assert isinstance(mapper, OpenAIAgentsOtelSessionMapper)

    def test_adot_spans_route_to_openai_mapper(self, openai_agents_adot_spans):
        """ADOT spans with openai_agents scope route to OpenAIAgentsOtelSessionMapper."""
        data = openai_agents_adot_spans
        spans = data.get("spans", data) if isinstance(data, dict) else data
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, OpenAIAgentsOtelSessionMapper)


# =============================================================================
# Live Span Tests
# =============================================================================


class TestLiveSession:
    """Tests against live OpenAI Agents SDK spans (Traceloop instrumentation)."""

    def test_session_has_traces(self, live_session):
        """Session should contain at least one trace."""
        assert len(live_session.traces) > 0

    def test_session_has_inference_spans(self, live_session):
        """Session should contain InferenceSpans from chat operations."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        inference_spans = [s for s in all_spans if isinstance(s, InferenceSpan)]
        assert len(inference_spans) > 0

    def test_session_has_tool_execution_spans(self, live_session):
        """Session should contain ToolExecutionSpans from tool calls."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) > 0

    def test_session_has_agent_invocation_spans(self, live_session):
        """Session should contain AgentInvocationSpans from named agents."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) > 0

    def test_no_unknown_operation_spans(self, live_session):
        """Unknown/wrapper spans should be filtered out (no empty agent spans)."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        # Every AgentInvocationSpan should have content
        for span in all_spans:
            if isinstance(span, AgentInvocationSpan):
                assert span.user_prompt or span.agent_response

    def test_inference_spans_have_messages(self, live_session):
        """InferenceSpans should have at least one message."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        for span in all_spans:
            if isinstance(span, InferenceSpan):
                assert len(span.messages) > 0

    def test_tool_spans_have_name(self, live_session):
        """ToolExecutionSpans should have a tool name."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        for span in all_spans:
            if isinstance(span, ToolExecutionSpan):
                assert span.tool_call.name

    def test_agent_spans_have_tools_backfilled(self, live_session):
        """AgentInvocationSpans should have available_tools back-filled from ToolExecutionSpans."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        # At least one agent span should have tools populated
        has_tools = any(s.available_tools for s in agent_spans)
        assert has_tools


# =============================================================================
# ADOT Span Tests
# =============================================================================


class TestAdotSession:
    """Tests against ADOT/CloudWatch-exported OpenAI Agents SDK spans."""

    def test_session_has_traces(self, adot_session):
        """Session should contain at least one trace."""
        assert len(adot_session.traces) > 0

    def test_session_has_inference_spans(self, adot_session):
        """Session should contain InferenceSpans."""
        all_spans = [s for t in adot_session.traces for s in t.spans]
        inference_spans = [s for s in all_spans if isinstance(s, InferenceSpan)]
        assert len(inference_spans) > 0

    def test_session_has_tool_execution_spans(self, adot_session):
        """Session should contain ToolExecutionSpans."""
        all_spans = [s for t in adot_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) > 0

    def test_session_has_agent_invocation_spans(self, adot_session):
        """Session should contain AgentInvocationSpans."""
        all_spans = [s for t in adot_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) > 0


# =============================================================================
# Per-Agent Tool Attribution Tests
# =============================================================================


class TestPerAgentToolAttribution:
    """Verify tool back-filling is scoped per agent, not trace-wide."""

    def test_live_session_has_exactly_two_agent_spans(self, live_session):
        """Live session should have exactly 2 agent spans (coordinator and math_specialist)."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 2

    def test_coordinator_tools(self, live_session):
        """Coordinator agent (span 91578cd4f1232ef0) should only have ask_math_specialist."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        coordinator = next(
            s for s in all_spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id == "91578cd4f1232ef0"
        )
        tool_names = sorted(t.name for t in coordinator.available_tools)
        assert tool_names == ["ask_math_specialist"]

    def test_math_specialist_tools(self, live_session):
        """Math specialist (span 8c1028e91e04dfa2) should only have multiply_numbers and divide_numbers."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        math_specialist = next(
            s for s in all_spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id == "8c1028e91e04dfa2"
        )
        tool_names = sorted(t.name for t in math_specialist.available_tools)
        assert tool_names == ["divide_numbers", "multiply_numbers"]

    def test_tools_not_shared_across_agents(self, live_session):
        """No tool should appear in both agents' available_tools lists."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        tool_sets = [frozenset(t.name for t in s.available_tools) for s in agent_spans]
        assert tool_sets[0] & tool_sets[1] == frozenset()
