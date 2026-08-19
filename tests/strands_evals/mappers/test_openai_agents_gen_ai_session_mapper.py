"""Tests for _OpenAIAgentsGenAISessionMapper - OpenAI Agents SDK spans."""

import json
from pathlib import Path

import pytest

from strands_evals.mappers import detect_otel_mapper
from strands_evals.mappers.constants import SCOPE_OPENAI_AGENTS
from strands_evals.mappers.openai_agents_gen_ai_session_mapper import _OpenAIAgentsGenAISessionMapper
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    ToolExecutionSpan,
    _find_root_agent_span,
)


def make_span(
    trace_id="trace-1",
    span_id="span-1",
    parent_span_id=None,
    name="test-span",
    operation_name="",
    attributes=None,
    span_events=None,
    scope_name="custom-tracer",
):
    """Build a dict span with gen_ai.* attributes."""
    attrs = attributes or {}
    if operation_name:
        attrs.setdefault("gen_ai.operation.name", operation_name)
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "start_time": 1700000000000000000,
        "end_time": 1700000001000000000,
        "attributes": attrs,
        "scope": {"name": scope_name, "version": "1.0"},
        "status": {"code": "OK"},
        "span_events": span_events or [],
    }


def _user_msg(text):
    """Build a gen_ai.input.messages JSON string with a single user text part."""
    return json.dumps([{"role": "user", "parts": [{"type": "text", "content": text}]}])


def _assistant_msg(text):
    """Build a gen_ai.output.messages JSON string with a single assistant text part."""
    return json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": text}]}])


# =============================================================================
# Fixture data
# =============================================================================

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_OPENAI_AGENTS_LIVE_SPANS_FILE = _FIXTURES_DIR / "openai_agents_genai_live_spans.json"
_OPENAI_AGENTS_ADOT_SPANS_FILE = _FIXTURES_DIR / "openai_agents_genai_adot_spans.json"


@pytest.fixture(scope="module")
def openai_agents_live_spans():
    """Load real OpenAI Agents SDK spans captured via Traceloop instrumentation."""
    with open(_OPENAI_AGENTS_LIVE_SPANS_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def openai_agents_adot_spans():
    """Load OpenAI Agents SDK spans as seen through ADOT/CloudWatch export."""
    with open(_OPENAI_AGENTS_ADOT_SPANS_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def openai_agents_live_session(openai_agents_live_spans):
    """Map live OpenAI Agents spans to a Session."""
    mapper = _OpenAIAgentsGenAISessionMapper()
    return mapper.map_to_session(openai_agents_live_spans, session_id="test")


@pytest.fixture(scope="module")
def openai_agents_adot_session(openai_agents_adot_spans):
    """Map ADOT OpenAI Agents spans to a Session."""
    mapper = _OpenAIAgentsGenAISessionMapper()
    data = openai_agents_adot_spans
    session_id = data.get("session_id", "test") if isinstance(data, dict) else "test"
    return mapper.map_to_session(data, session_id=session_id)


# =============================================================================
# Routing
# =============================================================================


class TestOpenAIAgentsFixtureRouting:
    """Verify detect_otel_mapper routes OpenAI Agents spans to _OpenAIAgentsGenAISessionMapper."""

    def test_live_spans_route_to_openai_agents_mapper(self, openai_agents_live_spans):
        """Live spans with openai_agents scope route to _OpenAIAgentsGenAISessionMapper."""
        mapper = detect_otel_mapper(openai_agents_live_spans)
        assert isinstance(mapper, _OpenAIAgentsGenAISessionMapper)

    def test_adot_spans_route_to_openai_agents_mapper(self, openai_agents_adot_spans):
        """ADOT spans with openai_agents scope route to _OpenAIAgentsGenAISessionMapper."""
        data = openai_agents_adot_spans
        spans = data.get("spans", data) if isinstance(data, dict) else data
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, _OpenAIAgentsGenAISessionMapper)


# =============================================================================
# Live session tests
# =============================================================================


class TestOpenAIAgentsLiveSession:
    """Tests against live OpenAI Agents SDK spans (Traceloop instrumentation)."""

    def test_produces_all_span_types(self, openai_agents_live_session):
        """Live trace yields inference, tool, and agent spans."""
        all_spans = [s for t in openai_agents_live_session.traces for s in t.spans]
        span_types = {type(s) for s in all_spans}
        assert {InferenceSpan, ToolExecutionSpan, AgentInvocationSpan} <= span_types

    def test_inference_spans_have_messages(self, openai_agents_live_session):
        """InferenceSpans should have at least one message."""
        all_spans = [s for t in openai_agents_live_session.traces for s in t.spans]
        for span in all_spans:
            if isinstance(span, InferenceSpan):
                assert len(span.messages) > 0

    def test_tool_spans_have_name(self, openai_agents_live_session):
        """ToolExecutionSpans should have a tool name."""
        all_spans = [s for t in openai_agents_live_session.traces for s in t.spans]
        for span in all_spans:
            if isinstance(span, ToolExecutionSpan):
                assert span.tool_call.name


# =============================================================================
# ADOT session tests
# =============================================================================


class TestOpenAIAgentsAdotSession:
    """Tests against ADOT/CloudWatch-exported OpenAI Agents SDK spans."""

    def test_produces_all_span_types(self, openai_agents_adot_session):
        """ADOT trace yields inference, tool, and agent spans."""
        all_spans = [s for t in openai_agents_adot_session.traces for s in t.spans]
        span_types = {type(s) for s in all_spans}
        assert {InferenceSpan, ToolExecutionSpan, AgentInvocationSpan} <= span_types

    def test_handoff_agents_are_flat_with_coordinator_as_root(self, openai_agents_adot_session):
        """Handoff sub-agents are left independent (parentless); root is the earliest-start agent."""
        all_spans = [s for t in openai_agents_adot_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]

        coordinator = next(s for s in agent_spans if s.span_info.span_id == "1afcb5ca9fe89dda")
        math_specialist = next(s for s in agent_spans if s.span_info.span_id == "c2828d53371818f0")

        assert coordinator.span_info.parent_span_id is None
        assert math_specialist.span_info.parent_span_id is None
        assert _find_root_agent_span(agent_spans).span_info.span_id == "1afcb5ca9fe89dda"

    def test_adot_coordinator_tools_empty(self, openai_agents_adot_session):
        """ADOT coordinator has no structured tool calls (handoffs emitted as repr text)."""
        all_spans = [s for t in openai_agents_adot_session.traces for s in t.spans]
        coordinator = next(
            s for s in all_spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id == "1afcb5ca9fe89dda"
        )
        assert coordinator.available_tools == []

    def test_adot_math_specialist_tools(self, openai_agents_adot_session):
        """ADOT math_specialist should have multiply_numbers back-filled."""
        all_spans = [s for t in openai_agents_adot_session.traces for s in t.spans]
        math_specialist = next(
            s for s in all_spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id == "c2828d53371818f0"
        )
        tool_names = sorted(t.name for t in math_specialist.available_tools)
        assert tool_names == ["multiply_numbers"]


# =============================================================================
# Per-agent tool attribution
# =============================================================================


class TestOpenAIAgentsPerAgentToolAttribution:
    """Verify tool back-filling is scoped per agent, not trace-wide."""

    def test_live_session_has_three_agent_spans(self, openai_agents_live_session):
        """Live session should have 3 agent spans (coordinator, research, math)."""
        all_spans = [s for t in openai_agents_live_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 3

    def test_coordinator_tools(self, openai_agents_live_session):
        """Coordinator's available_tools come from declared definitions (both delegates)."""
        all_spans = [s for t in openai_agents_live_session.traces for s in t.spans]
        coordinator = next(
            s for s in all_spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id == "d72488b1a2d28f70"
        )
        tool_names = sorted(t.name for t in coordinator.available_tools)
        assert tool_names == ["ask_math_specialist", "ask_research_specialist"]
        assert all(t.description and t.parameters for t in coordinator.available_tools)

    def test_research_specialist_tools(self, openai_agents_live_session):
        """Research specialist lists get_weather (never called) alongside lookup_stock_price."""
        all_spans = [s for t in openai_agents_live_session.traces for s in t.spans]
        research_specialist = next(
            s for s in all_spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id == "54044faa1adce44c"
        )
        tool_names = sorted(t.name for t in research_specialist.available_tools)
        assert tool_names == ["get_weather", "lookup_stock_price"]

    def test_math_specialist_tools(self, openai_agents_live_session):
        """Math specialist declares all three arithmetic tools, not just the one it used."""
        all_spans = [s for t in openai_agents_live_session.traces for s in t.spans]
        math_specialist = next(
            s for s in all_spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id == "186e8db9421d49d0"
        )
        tool_names = sorted(t.name for t in math_specialist.available_tools)
        assert tool_names == ["add_numbers", "divide_numbers", "multiply_numbers"]

    def test_tools_not_shared_across_agents(self, openai_agents_live_session):
        """No tool should appear in multiple agents' available_tools lists."""
        all_spans = [s for t in openai_agents_live_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        tool_sets = [set(t.name for t in s.available_tools) for s in agent_spans]
        assert not set.intersection(*tool_sets)

    def test_foreign_scope_tool_not_attributed_to_openai_agent(self):
        """A tool span from a different scope should not appear in an OpenAI agent's tools."""
        mapper = _OpenAIAgentsGenAISessionMapper()
        spans = [
            make_span(
                trace_id="t1",
                span_id="agent-1",
                operation_name="invoke_agent",
                attributes={
                    "gen_ai.agent.name": "coordinator",
                    "gen_ai.input.messages": json.dumps(
                        [{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]
                    ),
                },
                scope_name=SCOPE_OPENAI_AGENTS,
            ),
            make_span(
                trace_id="t1",
                span_id="tool-openai",
                parent_span_id="agent-1",
                operation_name="execute_tool",
                attributes={
                    "gen_ai.tool.name": "search",
                    "gen_ai.tool.call.id": "call-1",
                },
                scope_name=SCOPE_OPENAI_AGENTS,
            ),
            make_span(
                trace_id="t1",
                span_id="tool-foreign",
                parent_span_id="agent-1",
                operation_name="execute_tool",
                attributes={
                    "gen_ai.tool.name": "foreign_retriever",
                    "gen_ai.tool.call.id": "call-2",
                },
                scope_name="custom-instrumentor",
            ),
        ]
        session = mapper.map_to_session(spans, "test")
        agent_spans = [s for t in session.traces for s in t.spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        tool_names = [t.name for t in agent_spans[0].available_tools]
        assert "search" in tool_names
        assert "foreign_retriever" not in tool_names


# =============================================================================
# Prompt enrichment
# =============================================================================


class TestOpenAIAgentsPromptEnrichment:
    """Verify prompt/response back-filling from child inference spans.

    Traceloop's openai_agents instrumentation emits *.agent spans with no
    message payload; the mapper derives each agent's prompt/response from its child
    openai.response (chat) spans.
    """

    def test_all_agent_spans_have_prompt_and_response(self, openai_agents_live_session):
        """Every agent span should be enriched with a prompt and a response."""
        all_spans = [s for t in openai_agents_live_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert agent_spans
        for span in agent_spans:
            assert span.user_prompt
            assert span.agent_response

    def test_coordinator_prompt_is_top_level_user_request(self, openai_agents_live_session):
        """Coordinator prompt/response come from its own child chat spans, not a sub-agent's."""
        all_spans = [s for t in openai_agents_live_session.traces for s in t.spans]
        coordinator = next(
            s for s in all_spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id == "d72488b1a2d28f70"
        )
        assert coordinator.user_prompt == (
            "Look up the stock price for AMZN, then calculate what 1000 shares "
            "would be worth by multiplying the price by 1000."
        )
        assert coordinator.agent_response.startswith("The current stock price")

    def test_math_specialist_prompt_is_delegated_query(self, openai_agents_live_session):
        """Math specialist prompt/response reflect the delegated sub-task."""
        all_spans = [s for t in openai_agents_live_session.traces for s in t.spans]
        math_specialist = next(
            s for s in all_spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id == "186e8db9421d49d0"
        )
        assert math_specialist.user_prompt == "240.98 multiplied by 1000"
        assert math_specialist.agent_response

    def test_backfill_from_child_inference_when_agent_span_has_no_payload(self):
        """Traceloop emits payload-less invoke_agent spans; prompt/response
        must be derived from child chat spans.

        Both shipped fixtures carry agent-span payloads, so this shape is only
        exercisable by building it directly.
        """
        spans = [
            make_span(
                trace_id="t1",
                span_id="ag",
                parent_span_id=None,
                operation_name="invoke_agent",
                attributes={"gen_ai.agent.name": "solo"},
                scope_name=SCOPE_OPENAI_AGENTS,
            ),
            make_span(
                trace_id="t1",
                span_id="chat",
                parent_span_id="ag",
                operation_name="chat",
                attributes={
                    "gen_ai.input.messages": _user_msg("real user question"),
                    "gen_ai.output.messages": _assistant_msg("real assistant answer"),
                },
                scope_name=SCOPE_OPENAI_AGENTS,
            ),
        ]
        session = _OpenAIAgentsGenAISessionMapper().map_to_session(spans, "test")
        agent = next(s for t in session.traces for s in t.spans if isinstance(s, AgentInvocationSpan))
        assert agent.user_prompt == "real user question"
        assert agent.agent_response == "real assistant answer"

    def test_backfill_does_not_override_existing_agent_payload(self):
        """When the agent span already carries messages, backfill must
        not overwrite them with child-inference text."""
        spans = [
            make_span(
                trace_id="t1",
                span_id="ag",
                parent_span_id=None,
                operation_name="invoke_agent",
                attributes={
                    "gen_ai.agent.name": "solo",
                    "gen_ai.input.messages": _user_msg("agent-level question"),
                    "gen_ai.output.messages": _assistant_msg("agent-level answer"),
                },
                scope_name=SCOPE_OPENAI_AGENTS,
            ),
            make_span(
                trace_id="t1",
                span_id="chat",
                parent_span_id="ag",
                operation_name="chat",
                attributes={
                    "gen_ai.input.messages": _user_msg("child question"),
                    "gen_ai.output.messages": _assistant_msg("child answer"),
                },
                scope_name=SCOPE_OPENAI_AGENTS,
            ),
        ]
        session = _OpenAIAgentsGenAISessionMapper().map_to_session(spans, "test")
        agent = next(s for t in session.traces for s in t.spans if isinstance(s, AgentInvocationSpan))
        assert agent.user_prompt == "agent-level question"
        assert agent.agent_response == "agent-level answer"


# =============================================================================
# Wrapper dropping
# =============================================================================


class TestOpenAIAgentsWrapperDropping:
    """Verify that empty workflow wrapper spans are suppressed."""

    def test_wrapper_session_id_used_for_filtering_then_wrapper_dropped(self):
        """Wrapper's session.id is used for filtering; the wrapper itself is excluded from output."""
        mapper = _OpenAIAgentsGenAISessionMapper()
        spans = [
            # Trace A: wrapper carries session A, child agent is untagged.
            make_span(
                trace_id="tA",
                span_id="wrapper-a",
                operation_name="invoke_agent",
                attributes={"session.id": "sess-A"},
                scope_name=SCOPE_OPENAI_AGENTS,
            ),
            make_span(
                trace_id="tA",
                span_id="agent-a",
                parent_span_id="wrapper-a",
                operation_name="invoke_agent",
                attributes={"gen_ai.agent.name": "coordinator", "gen_ai.input.messages": "[]"},
                scope_name=SCOPE_OPENAI_AGENTS,
            ),
            # Trace B: wrapper carries session B.
            make_span(
                trace_id="tB",
                span_id="wrapper-b",
                operation_name="invoke_agent",
                attributes={"session.id": "sess-B"},
                scope_name=SCOPE_OPENAI_AGENTS,
            ),
            make_span(
                trace_id="tB",
                span_id="agent-b",
                parent_span_id="wrapper-b",
                operation_name="invoke_agent",
                attributes={"gen_ai.agent.name": "helper", "gen_ai.input.messages": "[]"},
                scope_name=SCOPE_OPENAI_AGENTS,
            ),
        ]
        # Session A includes only trace A; wrapper itself is suppressed.
        session_a = mapper.map_to_session(spans, "sess-A")
        assert len(session_a.traces) == 1
        assert session_a.traces[0].trace_id == "tA"
        span_ids = {s.span_info.span_id for s in session_a.traces[0].spans}
        assert "wrapper-a" not in span_ids
        assert "agent-a" in span_ids

        # Wrong session_id returns nothing.
        session_wrong = mapper.map_to_session(spans, "wrong-id")
        assert len(session_wrong.traces) == 0
