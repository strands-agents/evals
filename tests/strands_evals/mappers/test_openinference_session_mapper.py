"""Tests for OpenInferenceSessionMapper - OpenInference LangChain trace → Session conversion."""

import json
from pathlib import Path

import pytest

from strands_evals.mappers import OpenInferenceSessionMapper
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    ToolExecutionSpan,
    ToolResultContent,
)

# Path to fixture files
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_LIVE_SPANS_FILE = _FIXTURES_DIR / "openinference_live_spans.json"
_ADOT_SPANS_FILE = _FIXTURES_DIR / "openinference_adot_spans.json"
_SMOLAGENTS_SPANS_FILE = _FIXTURES_DIR / "smolagents_live_spans.json"
_CLAUDE_SPANS_FILE = _FIXTURES_DIR / "claude_live_spans.json"
_CLAUDE_ADOT_FILE = _FIXTURES_DIR / "claude_adot_spans.json"

SCOPE_NAME = "openinference.instrumentation.langchain"
SMOLAGENTS_SCOPE_NAME = "openinference.instrumentation.smolagents"
CLAUDE_SDK_SCOPE_NAME = "openinference.instrumentation.claude_agent_sdk"


def make_span(
    trace_id="trace-1",
    span_id="span-1",
    parent_span_id=None,
    name="test-span",
    attributes=None,
    span_events=None,
    start_time=1700000000000000000,
    end_time=1700000001000000000,
    scope_name=None,
):
    """Build a normalized span dict for OpenInference LangChain traces."""
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "start_time": start_time,
        "end_time": end_time,
        "attributes": attributes or {},
        "scope": {"name": scope_name or SCOPE_NAME, "version": "0.1.0"},
        "status": {"code": "OK"},
        "span_events": span_events or [],
    }


def make_llm_span(
    trace_id="trace-1",
    span_id="span-1",
    user_content="Hello",
    assistant_content="Hi there",
    tool_calls=None,
    scope_name=None,
):
    """Build an LLM inference span with openinference attributes."""
    attrs = {
        "openinference.span.kind": "LLM",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": user_content,
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": assistant_content,
    }

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            attrs[f"llm.output_messages.0.message.tool_calls.{i}.tool_call.function.name"] = tc["name"]
            attrs[f"llm.output_messages.0.message.tool_calls.{i}.tool_call.function.arguments"] = json.dumps(
                tc.get("arguments", {})
            )
            attrs[f"llm.output_messages.0.message.tool_calls.{i}.tool_call.id"] = tc.get("id", f"tool-{i}")

    return make_span(trace_id=trace_id, span_id=span_id, attributes=attrs, scope_name=scope_name)


def make_tool_span(
    trace_id="trace-1",
    span_id="span-1",
    tool_name="calculator",
    tool_input=None,
    tool_output=None,
    tool_call_id="tool-1",
    scope_name=None,
):
    """Build a tool execution span with openinference attributes."""
    tool_input = tool_input or {"expr": "2+2"}
    tool_output = tool_output or "4"

    output_value = json.dumps(
        {
            "content": tool_output,
            "tool_call_id": tool_call_id,
            "status": "success",
        }
    )

    attrs = {
        "openinference.span.kind": "TOOL",
        "tool.name": tool_name,
        "input.value": json.dumps(tool_input),
        "output.value": output_value,
    }

    return make_span(trace_id=trace_id, span_id=span_id, name=tool_name, attributes=attrs, scope_name=scope_name)


def make_chain_span(
    trace_id="trace-1",
    span_id="span-1",
    name="LangGraph",
    user_query="What is 2+2?",
    agent_response="The answer is 4",
):
    """Build a CHAIN span (agent invocation) with openinference attributes."""
    input_value = json.dumps({"messages": [{"kwargs": {"content": user_query, "type": "human"}}]})
    output_value = json.dumps({"messages": [{"kwargs": {"content": agent_response, "type": "ai"}}]})

    attrs = {
        "openinference.span.kind": "CHAIN",
        "input.value": input_value,
        "output.value": output_value,
    }

    return make_span(trace_id=trace_id, span_id=span_id, name=name, attributes=attrs)


def make_adot_span(
    trace_id="trace-1",
    span_id="span-1",
    input_messages=None,
    output_messages=None,
    start_time=1700000000000000000,
    end_time=1700000001000000000,
):
    """Build an ADOT/CloudWatch span with data in span_events body.

    All ADOT spans have the same generic name (scope name), no openinference.span.kind,
    and carry data inside span_events[].body.input/output.messages.
    """
    body = {}
    if input_messages is not None:
        body["input"] = {"messages": input_messages}
    if output_messages is not None:
        body["output"] = {"messages": output_messages}

    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": None,
        "name": SCOPE_NAME,
        "start_time": start_time,
        "end_time": end_time,
        "attributes": {"event.name": SCOPE_NAME},
        "scope": {"name": SCOPE_NAME, "version": ""},
        "status": {"code": "OK"},
        "span_events": [
            {
                "event_name": SCOPE_NAME,
                "span_id": span_id,
                "timestamp": start_time,
                "attributes": {"event.name": SCOPE_NAME},
                "body": body,
            }
        ],
    }


def _load_live_spans():
    """Load real live (in-memory) spans from fixture file."""
    with open(_LIVE_SPANS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _load_adot_spans():
    """Load ADOT/CloudWatch spans from fixture file."""
    with open(_ADOT_SPANS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _load_smolagents_spans():
    """Load real smolagents (openinference-instrumentation-smolagents) spans from fixture file."""
    with open(_SMOLAGENTS_SPANS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _load_claude_spans():
    """Load real Claude Agent SDK (openinference-instrumentation-claude-agent-sdk) spans from fixture file."""
    with open(_CLAUDE_SPANS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _load_claude_adot_spans():
    """Load Claude Agent SDK spans captured from AgentCore (session.id on all spans)."""
    with open(_CLAUDE_ADOT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data["session_id"], data["spans"]


class TestSpanTypeDetection:
    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_llm_span_detected(self):
        """Span with openinference.span.kind=LLM is detected as inference span."""
        span = make_span(attributes={"openinference.span.kind": "LLM"})
        assert self.mapper._is_inference_span(span) is True
        assert self.mapper._is_tool_execution_span(span) is False

    def test_tool_span_detected(self):
        """Span with openinference.span.kind=TOOL is detected as tool span."""
        span = make_span(attributes={"openinference.span.kind": "TOOL"})
        assert self.mapper._is_inference_span(span) is False
        assert self.mapper._is_tool_execution_span(span) is True

    def test_chain_langgraph_span_detected_as_agent(self):
        """CHAIN span with name=LangGraph is detected as agent invocation."""
        span = make_span(name="LangGraph", attributes={"openinference.span.kind": "CHAIN"})
        assert self.mapper._is_agent_invocation_span(span) is True


class TestInferenceSpanConversion:
    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_basic_llm_span(self):
        """Basic LLM span with user and assistant messages."""
        span = make_llm_span(
            user_content="What is the weather?",
            assistant_content="It's sunny today.",
        )
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1

        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        assert inference.messages[0].content[0].text == "What is the weather?"
        assert inference.messages[1].content[0].text == "It's sunny today."

    def test_llm_span_with_tool_calls(self):
        """LLM span with tool calls in assistant response."""
        span = make_llm_span(
            user_content="Calculate 2+2",
            assistant_content="Let me calculate that.",
            tool_calls=[{"name": "calculator", "arguments": {"expr": "2+2"}, "id": "tc-1"}],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        assert len(inference.messages[1].content) == 2
        assert inference.messages[1].content[0].text == "Let me calculate that."
        assert inference.messages[1].content[1].name == "calculator"
        assert inference.messages[1].content[1].tool_call_id == "tc-1"


class TestToolExecutionSpanConversion:
    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_basic_tool_span(self):
        """Tool execution span with input and output."""
        span = make_tool_span(
            tool_name="calculator",
            tool_input={"expr": "6*7"},
            tool_output="42",
            tool_call_id="tc-1",
        )
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces[0].spans) == 1
        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_call.name == "calculator"
        assert tool.tool_result.content == "42"
        assert tool.tool_result.tool_call_id == "tc-1"


class TestAgentInvocationSpanConversion:
    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_langgraph_chain_as_agent_span(self):
        """CHAIN span named LangGraph becomes agent invocation span."""
        span = make_chain_span(
            name="LangGraph",
            user_query="What is the capital of France?",
            agent_response="The capital of France is Paris.",
        )
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces[0].spans) == 1
        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.user_prompt == "What is the capital of France?"
        assert agent.agent_response == "The capital of France is Paris."

    def test_only_one_agent_span_per_trace(self):
        """Only one agent invocation span is created per trace."""
        span1 = make_chain_span(trace_id="t1", span_id="s1", name="LangGraph")
        span2 = make_chain_span(trace_id="t1", span_id="s2", name="LangGraph")

        session = self.mapper.map_to_session([span1, span2], "sess-1")

        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1

    def test_agent_span_collects_tools_from_llm_spans(self):
        """Agent invocation span collects tools from LLM spans in same trace."""
        llm_span = make_llm_span(
            trace_id="t1",
            span_id="s1",
            user_content="Calculate",
            assistant_content="OK",
        )
        llm_span["attributes"]["llm.tools.0.tool.json_schema"] = json.dumps(
            {
                "name": "calculator",
                "description": "A calculator tool",
                "input_schema": {"type": "object"},
            }
        )

        agent_span = make_chain_span(
            trace_id="t1",
            span_id="s2",
            name="LangGraph",
            user_query="Calculate 2+2",
            agent_response="4",
        )

        session = self.mapper.map_to_session([llm_span, agent_span], "sess-1")

        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        assert len(agent_spans[0].available_tools) == 1
        assert agent_spans[0].available_tools[0].name == "calculator"


class TestSessionBuilding:
    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_empty_spans_list(self):
        """Empty spans list produces empty session."""
        session = self.mapper.map_to_session([], "sess-1")
        assert session.session_id == "sess-1"
        assert session.traces == []

    def test_multiple_traces(self):
        """Spans with different trace_ids grouped into separate traces."""
        span1 = make_llm_span(trace_id="t1", span_id="s1", user_content="Q1", assistant_content="A1")
        span2 = make_llm_span(trace_id="t2", span_id="s2", user_content="Q2", assistant_content="A2")

        session = self.mapper.map_to_session([span1, span2], "sess-1")

        assert len(session.traces) == 2
        trace_ids = {t.trace_id for t in session.traces}
        assert trace_ids == {"t1", "t2"}

    def test_filters_by_scope(self):
        """Only spans with matching scope are processed."""
        openinference_span = make_llm_span(user_content="Q1", assistant_content="A1")
        other_span = make_span(
            span_id="other",
            attributes={"openinference.span.kind": "LLM"},
        )
        other_span["scope"]["name"] = "other.scope"

        session = self.mapper.map_to_session([openinference_span, other_span], "sess-1")

        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1

    def test_full_agent_loop(self):
        """Full agent loop produces all span types."""
        spans = [
            make_llm_span(
                trace_id="t1",
                span_id="s1",
                user_content="Calculate 6*7",
                assistant_content="Let me calculate",
                tool_calls=[{"name": "calculator", "arguments": {"expr": "6*7"}, "id": "tc-1"}],
            ),
            make_tool_span(
                trace_id="t1",
                span_id="s2",
                tool_name="calculator",
                tool_input={"expr": "6*7"},
                tool_output="42",
                tool_call_id="tc-1",
            ),
            make_llm_span(
                trace_id="t1",
                span_id="s3",
                user_content="Tool result: 42",
                assistant_content="The answer is 42.",
            ),
            make_chain_span(
                trace_id="t1",
                span_id="s4",
                name="LangGraph",
                user_query="Calculate 6*7",
                agent_response="The answer is 42.",
            ),
        ]

        session = self.mapper.map_to_session(spans, "sess-1")

        assert len(session.traces) == 1
        span_types = [type(s).__name__ for s in session.traces[0].spans]
        assert "InferenceSpan" in span_types
        assert "ToolExecutionSpan" in span_types
        assert "AgentInvocationSpan" in span_types


class TestInputNormalization:
    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_grouped_by_trace_id_input(self):
        """Handles input grouped by trace_id dict format."""
        span1 = make_llm_span(trace_id="t1", span_id="s1", user_content="Q1", assistant_content="A1")
        span2 = make_llm_span(trace_id="t2", span_id="s2", user_content="Q2", assistant_content="A2")

        grouped_data = {
            "t1": [span1],
            "t2": [span2],
        }

        session = self.mapper.map_to_session(grouped_data, "sess-1")

        assert len(session.traces) == 2

    def test_trace_objects_input(self):
        """Handles input as list of trace objects with spans key."""
        span1 = make_llm_span(trace_id="t1", span_id="s1", user_content="Q1", assistant_content="A1")

        trace_objects = [
            {"trace_id": "t1", "spans": [span1]},
        ]

        session = self.mapper.map_to_session(trace_objects, "sess-1")

        assert len(session.traces) == 1


# =========================================================================
# Empty Span Filter Tests
# =========================================================================


class TestEmptySpanFilter:
    """Tests for the empty-response InferenceSpan filter (line 249 of mapper)."""

    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_empty_text_only_assistant_filtered(self):
        """InferenceSpan with assistant=[TextContent(text="")] returns None."""
        span = make_llm_span(
            user_content="Hello",
            assistant_content="",
        )
        session = self.mapper.map_to_session([span], "sess-1")
        # Empty assistant text, no tool calls → filtered out
        assert session.traces == []

    def test_empty_text_with_tool_call_preserved(self):
        """InferenceSpan with empty text + ToolCallContent is preserved (ADOT path)."""
        # In the ADOT/generations path, the LLM can return empty text with tool calls.
        # The live attrs path requires non-empty content to extract tool calls,
        # so we test via ADOT generations format.
        span = make_adot_span(
            span_id="filter-tc",
            input_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [
                                [
                                    {"lc": 1, "kwargs": {"content": "System.", "type": "system"}},
                                    {"lc": 1, "kwargs": {"content": "Calculate 2+2", "type": "human"}},
                                ]
                            ]
                        }
                    ),
                    "role": "user",
                },
                {"content": "Calculate 2+2", "role": "user"},
            ],
            output_messages=[
                {
                    "content": json.dumps(
                        {
                            "generations": [
                                [
                                    {
                                        "text": "",
                                        "type": "ChatGeneration",
                                        "message": {
                                            "lc": 1,
                                            "kwargs": {
                                                "content": "",
                                                "type": "ai",
                                                "tool_calls": [
                                                    {
                                                        "name": "calculator",
                                                        "args": {"expr": "2+2"},
                                                        "id": "tc-1",
                                                        "type": "tool_call",
                                                    },
                                                ],
                                            },
                                        },
                                    }
                                ]
                            ],
                            "type": "LLMResult",
                        }
                    ),
                    "role": "assistant",
                },
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")
        # Has a tool call → NOT filtered by empty span filter
        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        # Should have empty text + tool call in assistant content
        assert len(inference.messages[1].content) == 2

    def test_nonempty_text_assistant_preserved(self):
        """InferenceSpan with non-empty text is preserved."""
        span = make_llm_span(
            user_content="Hello",
            assistant_content="Hi there!",
        )
        session = self.mapper.map_to_session([span], "sess-1")
        assert len(session.traces[0].spans) == 1
        assert isinstance(session.traces[0].spans[0], InferenceSpan)


# =========================================================================
# System Prompt Extraction Tests
# =========================================================================


class TestSystemPromptExtraction:
    """Tests for system prompt at index 0, user at index 1."""

    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_system_prompt_at_index_0_user_at_index_1(self):
        """System prompt at index 0, user message at index 1 — user content extracted correctly."""
        attrs = {
            "openinference.span.kind": "LLM",
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "You are a helpful assistant.",
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": "What's the weather?",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "It's sunny.",
        }
        span = make_span(attributes=attrs)
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces[0].spans) == 1
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        # User content should be the actual user message, not the system prompt
        assert inference.messages[0].content[0].text == "What's the weather?"

    def test_system_prompt_only_returns_no_span(self):
        """If only system message exists (no user), span is skipped."""
        attrs = {
            "openinference.span.kind": "LLM",
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "You are a helpful assistant.",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "Hello!",
        }
        span = make_span(attributes=attrs)
        session = self.mapper.map_to_session([span], "sess-1")
        # No user content → no InferenceSpan produced
        assert session.traces == []


# =========================================================================
# Python Repr Parsing Tests
# =========================================================================


class TestPythonReprParsing:
    """Tests for ast.literal_eval fallback when input.value is Python repr."""

    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_python_repr_tool_input(self):
        """Tool span with Python repr input.value parsed via ast.literal_eval."""
        # Real pattern from OpenInference: input.value = "{'city': 'Alaska'}"
        output_value = json.dumps(
            {
                "content": "Weather in Alaska: Partly cloudy, 65°F",
                "tool_call_id": "toolu_bdrk_123",
                "status": "success",
            }
        )
        attrs = {
            "openinference.span.kind": "TOOL",
            "tool.name": "get_weather",
            "input.value": "{'city': 'Alaska'}",  # Python repr, not JSON
            "output.value": output_value,
        }
        span = make_span(span_id="repr-tool", name="get_weather", attributes=attrs)
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces[0].spans) == 1
        tool_span = session.traces[0].spans[0]
        assert isinstance(tool_span, ToolExecutionSpan)
        assert tool_span.tool_call.name == "get_weather"
        assert tool_span.tool_call.arguments == {"city": "Alaska"}
        assert "Partly cloudy" in tool_span.tool_result.content


# =========================================================================
# Empty Trailing AI Message Tests
# =========================================================================


class TestEmptyTrailingAIMessage:
    """Tests for skipping empty trailing AI messages in agent response extraction."""

    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_empty_trailing_ai_message_skipped(self):
        """LangGraph output with empty trailing AI message → non-empty AI content returned."""
        # Real pattern: output.value has messages list where last AI message is empty
        output_value = json.dumps(
            {
                "messages": [
                    {"kwargs": {"content": "The answer is 42.", "type": "ai"}},
                    {"kwargs": {"content": "", "type": "ai"}},  # Empty trailing AI message
                ]
            }
        )
        input_value = json.dumps({"messages": [{"kwargs": {"content": "Calculate 6*7", "type": "human"}}]})
        attrs = {
            "openinference.span.kind": "CHAIN",
            "input.value": input_value,
            "output.value": output_value,
        }
        span = make_span(name="LangGraph", span_id="trailing-ai", attributes=attrs)
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces[0].spans) == 1
        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.agent_response == "The answer is 42."


# =========================================================================
# Multi-Agent Dedup Tests
# =========================================================================


class TestMultiAgentDedup:
    """Tests for multi-agent LangGraph dedup — only last agent span kept."""

    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_two_langgraph_chains_dedup_to_one(self):
        """2 LangGraph CHAIN spans in same trace → 1 AgentInvocationSpan (last kept)."""
        span1 = make_chain_span(
            trace_id="t1",
            span_id="s1",
            name="LangGraph",
            user_query="Q1",
            agent_response="A1",
        )
        span2 = make_chain_span(
            trace_id="t1",
            span_id="s2",
            name="LangGraph",
            user_query="Q2",
            agent_response="A2",
        )
        session = self.mapper.map_to_session([span1, span2], "sess-1")

        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        # Last one (s2) should be kept
        assert agent_spans[0].user_prompt == "Q2"
        assert agent_spans[0].agent_response == "A2"

    def test_single_agent_span_not_deduped(self):
        """Single agent span is preserved as-is."""
        span = make_chain_span(
            trace_id="t1",
            span_id="s1",
            name="LangGraph",
            user_query="Hello",
            agent_response="Hi",
        )
        session = self.mapper.map_to_session([span], "sess-1")
        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1


# =========================================================================
# AGENT-Kind Span Rejection Tests
# =========================================================================


class TestAgentKindSpanRejection:
    """Tests for AGENT-kind spans NOT matched as agent invocation."""

    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_agent_kind_not_detected_as_agent_invocation(self):
        """Span with kind=AGENT is NOT detected as agent invocation."""
        # OpenInference mislabels route_to_agent as AGENT instead of TOOL
        span = make_span(
            name="route_to_agent",
            attributes={
                "openinference.span.kind": "AGENT",
                "input.value": "{'agent_name': 'research_agent'}",
            },
        )
        assert self.mapper._is_agent_invocation_span(span) is False

    def test_agent_kind_not_detected_as_tool(self):
        """Span with kind=AGENT is NOT detected as tool."""
        span = make_span(
            name="route_to_agent",
            attributes={"openinference.span.kind": "AGENT"},
        )
        assert self.mapper._is_tool_execution_span(span) is False

    def test_agent_kind_not_detected_as_inference(self):
        """Span with kind=AGENT is NOT detected as inference."""
        span = make_span(
            name="route_to_agent",
            attributes={"openinference.span.kind": "AGENT"},
        )
        assert self.mapper._is_inference_span(span) is False


# =========================================================================
# ADOT Path Detection Tests
# =========================================================================


class TestAdotSpanTypeDetection:
    """Tests for ADOT/CloudWatch span type detection via body content."""

    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_adot_inference_detected_by_generations(self):
        """ADOT span with 'generations' in output detected as inference."""
        span = make_adot_span(
            span_id="adot-llm",
            input_messages=[
                {"content": "What's the weather?", "role": "user"},
            ],
            output_messages=[
                {
                    "content": json.dumps(
                        {
                            "generations": [
                                [
                                    {
                                        "text": "It's sunny.",
                                        "type": "ChatGeneration",
                                        "message": {"lc": 1, "kwargs": {"content": "It's sunny.", "type": "ai"}},
                                    }
                                ]
                            ],
                            "type": "LLMResult",
                        }
                    ),
                    "role": "assistant",
                }
            ],
        )
        assert self.mapper._is_inference_span(span) is True
        assert self.mapper._is_tool_execution_span(span) is False
        assert self.mapper._is_agent_invocation_span(span) is False

    def test_adot_tool_detected_by_type_tool(self):
        """ADOT span with 'type': 'tool' in output detected as tool execution."""
        span = make_adot_span(
            span_id="adot-tool",
            input_messages=[
                {"content": "{'city': 'Alaska'}", "role": "user"},
            ],
            output_messages=[
                {
                    "content": json.dumps(
                        {
                            "content": "Weather in Alaska: 65°F",
                            "type": "tool",
                            "name": "get_weather",
                            "tool_call_id": "toolu_123",
                            "status": "success",
                        }
                    ),
                    "role": "assistant",
                }
            ],
        )
        assert self.mapper._is_tool_execution_span(span) is True
        assert self.mapper._is_inference_span(span) is False
        assert self.mapper._is_agent_invocation_span(span) is False

    def test_adot_tool_call_with_context_skipped(self):
        """ADOT tool_call_with_context wrapper is NOT detected as tool."""
        span = make_adot_span(
            span_id="adot-wrapper",
            input_messages=[
                {
                    "content": json.dumps(
                        {
                            "__type": "tool_call_with_context",
                            "tool_call": {"name": "get_weather", "args": {"city": "Alaska"}},
                            "state": {"messages": [], "remaining_steps": 9999},
                        }
                    ),
                    "role": "user",
                }
            ],
            output_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [
                                {
                                    "content": "Weather: 65°F",
                                    "type": "tool",
                                    "name": "get_weather",
                                    "tool_call_id": "toolu_123",
                                    "status": "success",
                                }
                            ]
                        }
                    ),
                    "role": "assistant",
                }
            ],
        )
        assert self.mapper._is_tool_execution_span(span) is False

    def test_adot_agent_detected_by_messages_without_remaining_steps(self):
        """ADOT root graph span detected as agent invocation."""
        span = make_adot_span(
            span_id="adot-agent",
            input_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [{"content": "What's the weather?", "type": "human"}],
                        }
                    ),
                    "role": "user",
                }
            ],
            output_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [{"content": "It's sunny.", "type": "ai"}],
                        }
                    ),
                    "role": "assistant",
                }
            ],
        )
        assert self.mapper._is_agent_invocation_span(span) is True

    def test_adot_intermediate_node_with_remaining_steps_skipped(self):
        """ADOT intermediate node with remaining_steps is NOT agent invocation."""
        span = make_adot_span(
            span_id="adot-intermediate",
            input_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [{"content": "Hello", "type": "human"}],
                            "remaining_steps": 9999,
                        }
                    ),
                    "role": "user",
                }
            ],
            output_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [{"content": "Hi", "type": "ai"}],
                        }
                    ),
                    "role": "assistant",
                }
            ],
        )
        assert self.mapper._is_agent_invocation_span(span) is False

    def test_adot_end_node_not_matched(self):
        """ADOT __end__ output node is not matched as any span type."""
        span = make_adot_span(
            span_id="adot-end",
            input_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [{"content": "Hello", "type": "human"}],
                            "remaining_steps": 9997,
                        }
                    ),
                    "role": "user",
                }
            ],
            output_messages=[
                {"content": "__end__", "role": "assistant"},
            ],
        )
        assert self.mapper._is_inference_span(span) is False
        assert self.mapper._is_tool_execution_span(span) is False
        assert self.mapper._is_agent_invocation_span(span) is False


# =========================================================================
# ADOT Path Conversion Tests
# =========================================================================


class TestAdotSpanConversion:
    """Tests for ADOT span conversion to Session types."""

    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_adot_inference_span_conversion(self):
        """ADOT inference span with generations → InferenceSpan with correct content."""
        span = make_adot_span(
            span_id="adot-llm-conv",
            input_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [
                                [
                                    {"lc": 1, "kwargs": {"content": "You are helpful.", "type": "system"}},
                                    {
                                        "lc": 1,
                                        "kwargs": {"content": "What's the weather?", "type": "human", "id": "msg-1"},
                                    },
                                ]
                            ]
                        }
                    ),
                    "role": "user",
                },
                {"content": "What's the weather?", "role": "user"},
            ],
            output_messages=[
                {
                    "content": json.dumps(
                        {
                            "generations": [
                                [
                                    {
                                        "text": "It's sunny today.",
                                        "type": "ChatGeneration",
                                        "message": {
                                            "lc": 1,
                                            "kwargs": {
                                                "content": "It's sunny today.",
                                                "type": "ai",
                                                "tool_calls": [],
                                            },
                                        },
                                    }
                                ]
                            ],
                            "type": "LLMResult",
                        }
                    ),
                    "role": "assistant",
                },
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces[0].spans) == 1
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        assert inference.messages[1].content[0].text == "It's sunny today."

    def test_adot_tool_span_conversion(self):
        """ADOT tool span → ToolExecutionSpan with correct name, params, result."""
        span = make_adot_span(
            span_id="adot-tool-conv",
            input_messages=[
                {"content": "{'city': 'Alaska'}", "role": "user"},
            ],
            output_messages=[
                {
                    "content": json.dumps(
                        {
                            "content": "Weather in Alaska: Partly cloudy, 65°F",
                            "type": "tool",
                            "name": "get_weather",
                            "tool_call_id": "toolu_bdrk_123",
                            "status": "success",
                        }
                    ),
                    "role": "assistant",
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces[0].spans) == 1
        tool_span = session.traces[0].spans[0]
        assert isinstance(tool_span, ToolExecutionSpan)
        assert tool_span.tool_call.name == "get_weather"
        assert tool_span.tool_call.arguments == {"city": "Alaska"}
        assert "Partly cloudy" in tool_span.tool_result.content
        assert tool_span.tool_result.tool_call_id == "toolu_bdrk_123"

    def test_adot_agent_span_conversion(self):
        """ADOT root graph span → AgentInvocationSpan with user_prompt and agent_response."""
        span = make_adot_span(
            span_id="adot-agent-conv",
            input_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [
                                {"content": "What's the weather?", "type": "human", "id": "msg-1"},
                            ]
                        }
                    ),
                    "role": "user",
                },
                {"content": "What's the weather?", "role": "user"},
            ],
            output_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [
                                {"content": "What's the weather?", "type": "human"},
                                {"content": "It's sunny in both cities.", "type": "ai"},
                            ]
                        }
                    ),
                    "role": "assistant",
                }
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces[0].spans) == 1
        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.user_prompt == "What's the weather?"
        assert agent.agent_response == "It's sunny in both cities."

    def test_adot_inference_with_tool_calls(self):
        """ADOT inference span with tool calls in generations → preserved."""
        span = make_adot_span(
            span_id="adot-llm-tc",
            input_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [
                                [
                                    {"lc": 1, "kwargs": {"content": "System prompt.", "type": "system"}},
                                    {"lc": 1, "kwargs": {"content": "Check weather in Alaska", "type": "human"}},
                                ]
                            ]
                        }
                    ),
                    "role": "user",
                },
                {"content": "Check weather in Alaska", "role": "user"},
            ],
            output_messages=[
                {
                    "content": json.dumps(
                        {
                            "generations": [
                                [
                                    {
                                        "text": "",
                                        "type": "ChatGeneration",
                                        "message": {
                                            "lc": 1,
                                            "kwargs": {
                                                "content": "",
                                                "type": "ai",
                                                "tool_calls": [
                                                    {
                                                        "name": "get_weather",
                                                        "args": {"city": "Alaska"},
                                                        "id": "toolu_1",
                                                        "type": "tool_call",
                                                    },
                                                ],
                                            },
                                        },
                                    }
                                ]
                            ],
                            "type": "LLMResult",
                        }
                    ),
                    "role": "assistant",
                },
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces[0].spans) == 1
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        # Should have empty text + tool call (not filtered because tool call present)
        assert len(inference.messages[1].content) == 2
        assert inference.messages[1].content[1].name == "get_weather"

    def test_adot_structured_user_messages_with_tool_results(self):
        """ADOT structured messages extract human + tool result contents."""
        span = make_adot_span(
            span_id="adot-structured",
            input_messages=[
                {
                    "content": json.dumps(
                        {
                            "messages": [
                                [
                                    {"lc": 1, "kwargs": {"content": "You are helpful.", "type": "system"}},
                                    {"lc": 1, "kwargs": {"content": "What's the weather?", "type": "human"}},
                                    {
                                        "lc": 1,
                                        "kwargs": {
                                            "content": "Weather in Alaska: 65°F",
                                            "type": "tool",
                                            "name": "get_weather",
                                            "tool_call_id": "toolu_1",
                                            "status": "success",
                                        },
                                    },
                                ]
                            ]
                        }
                    ),
                    "role": "user",
                },
            ],
            output_messages=[
                {
                    "content": json.dumps(
                        {
                            "generations": [
                                [
                                    {
                                        "text": "The weather is 65°F.",
                                        "type": "ChatGeneration",
                                        "message": {
                                            "lc": 1,
                                            "kwargs": {"content": "The weather is 65°F.", "type": "ai"},
                                        },
                                    }
                                ]
                            ],
                            "type": "LLMResult",
                        }
                    ),
                    "role": "assistant",
                },
            ],
        )
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces[0].spans) == 1
        inference = session.traces[0].spans[0]
        assert isinstance(inference, InferenceSpan)
        user_msg = inference.messages[0]
        # Should have human text + tool result
        assert len(user_msg.content) == 2
        assert user_msg.content[0].text == "What's the weather?"
        assert isinstance(user_msg.content[1], ToolResultContent)
        assert "65°F" in user_msg.content[1].content


# =========================================================================
# Integration Tests: Real Fixture Files
# =========================================================================


@pytest.fixture(scope="module")
def live_session():
    """Map real live (in-memory) spans to a Session."""
    spans = _load_live_spans()
    mapper = OpenInferenceSessionMapper()
    return mapper.map_to_session(spans, "live-sess")


@pytest.fixture(scope="module")
def adot_session():
    """Map real ADOT/CloudWatch spans to a Session."""
    spans = _load_adot_spans()
    mapper = OpenInferenceSessionMapper()
    return mapper.map_to_session(spans, "adot-sess")


class TestLiveFixtureIntegration:
    """Integration tests using real live (in-memory) OpenInference trace data."""

    def test_session_has_traces(self, live_session):
        """Live fixture produces at least one trace."""
        assert len(live_session.traces) >= 1

    def test_produces_all_span_types(self, live_session):
        """Live fixture produces InferenceSpan, ToolExecutionSpan, and AgentInvocationSpan."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        span_types = {type(s).__name__ for s in all_spans}
        assert "InferenceSpan" in span_types
        assert "ToolExecutionSpan" in span_types
        assert "AgentInvocationSpan" in span_types

    def test_inference_spans_have_user_and_assistant(self, live_session):
        """Every InferenceSpan has user and assistant messages."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        inference_spans = [s for s in all_spans if isinstance(s, InferenceSpan)]
        assert len(inference_spans) > 0
        for span in inference_spans:
            assert len(span.messages) == 2
            assert span.messages[0].role.value == "user"
            assert span.messages[1].role.value == "assistant"
            assert len(span.messages[0].content) > 0
            assert len(span.messages[1].content) > 0

    def test_tool_spans_have_name_and_result(self, live_session):
        """Every ToolExecutionSpan has a tool name and result."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        # Only finish tool is correctly labeled as TOOL (route_to_agent is mislabeled as AGENT)
        assert len(tool_spans) >= 1
        for span in tool_spans:
            assert span.tool_call.name
            assert span.tool_result.content is not None

    def test_agent_spans_have_prompt_and_response(self, live_session):
        """Every AgentInvocationSpan has user_prompt and agent_response."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) >= 1
        for span in agent_spans:
            assert span.user_prompt
            assert span.agent_response

    def test_agent_kind_spans_not_included(self, live_session):
        """AGENT-kind spans (route_to_agent) are not mapped as AgentInvocationSpan."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        # Agent spans should only come from CHAIN+LangGraph, not AGENT-kind
        for span in agent_spans:
            assert span.user_prompt != "{'agent_name': 'research_agent'}"

    def test_one_agent_span_per_trace(self, live_session):
        """Each trace has at most 1 AgentInvocationSpan (multi-agent dedup)."""
        for trace in live_session.traces:
            agent_spans = [s for s in trace.spans if isinstance(s, AgentInvocationSpan)]
            assert len(agent_spans) <= 1

    def test_no_empty_response_inference_spans(self, live_session):
        """No InferenceSpan has empty-text-only assistant content."""
        all_spans = [s for t in live_session.traces for s in t.spans]
        for span in all_spans:
            if isinstance(span, InferenceSpan):
                assistant_content = span.messages[1].content
                # If all content is text, at least one must be non-empty
                text_only = all(hasattr(c, "text") for c in assistant_content)
                if text_only:
                    assert any(c.text for c in assistant_content)


class TestAdotFixtureIntegration:
    """Integration tests using real ADOT/CloudWatch OpenInference trace data."""

    def test_session_has_traces(self, adot_session):
        """ADOT fixture produces traces (3 trace_ids in the file)."""
        assert len(adot_session.traces) >= 1

    def test_produces_expected_span_types(self, adot_session):
        """ADOT fixture produces InferenceSpan and AgentInvocationSpan."""
        all_spans = [s for t in adot_session.traces for s in t.spans]
        span_types = {type(s).__name__ for s in all_spans}
        assert "InferenceSpan" in span_types
        # Tool spans should be present (get_weather)
        assert "ToolExecutionSpan" in span_types

    def test_tool_spans_have_correct_names(self, adot_session):
        """ADOT tool spans extract correct tool names from output body."""
        all_spans = [s for t in adot_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        tool_names = {s.tool_call.name for s in tool_spans}
        assert "get_weather" in tool_names

    def test_tool_spans_have_python_repr_params(self, adot_session):
        """ADOT tool spans correctly parse Python repr input (e.g. {'city': 'Alaska'})."""
        all_spans = [s for t in adot_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        # Should have parsed the Python repr dict
        for span in tool_spans:
            assert isinstance(span.tool_call.arguments, dict)
            assert len(span.tool_call.arguments) > 0

    def test_inference_spans_have_content(self, adot_session):
        """ADOT inference spans have non-empty user and assistant content."""
        all_spans = [s for t in adot_session.traces for s in t.spans]
        inference_spans = [s for s in all_spans if isinstance(s, InferenceSpan)]
        assert len(inference_spans) > 0
        for span in inference_spans:
            assert len(span.messages[0].content) > 0
            assert len(span.messages[1].content) > 0

    def test_agent_spans_have_content(self, adot_session):
        """ADOT agent spans have user_prompt and agent_response."""
        all_spans = [s for t in adot_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) >= 1
        for span in agent_spans:
            assert span.user_prompt
            assert span.agent_response

    def test_one_agent_span_per_trace(self, adot_session):
        """Each trace has at most 1 AgentInvocationSpan."""
        for trace in adot_session.traces:
            agent_spans = [s for s in trace.spans if isinstance(s, AgentInvocationSpan)]
            assert len(agent_spans) <= 1

    def test_intermediate_nodes_filtered(self, adot_session):
        """ADOT intermediate nodes (remaining_steps, __end__) are filtered out."""
        # Total spans should be much less than 41 raw input spans
        # Each trace should have: ~2 inference + 2 tool + 1 agent = ~5 max
        for trace in adot_session.traces:
            assert len(trace.spans) <= 10, f"Trace {trace.trace_id} has too many spans: {len(trace.spans)}"

    def test_no_empty_response_inference_spans(self, adot_session):
        """No InferenceSpan has empty-text-only assistant content (filter applied)."""
        all_spans = [s for t in adot_session.traces for s in t.spans]
        for span in all_spans:
            if isinstance(span, InferenceSpan):
                assistant_content = span.messages[1].content
                text_only = all(hasattr(c, "text") for c in assistant_content)
                if text_only:
                    assert any(c.text for c in assistant_content)


# =============================================================================
# Smolagents Scope Support (regression: openinference.instrumentation.smolagents)
#
# smolagents (and the OpenInference semantic conventions it follows) sets
# output.value as a bare string rather than a JSON object with a "content" key,
# and uses a distinct scope name ("openinference.instrumentation.smolagents")
# that differs from the LangChain variant. These tests verify the mapper
# accepts this scope and correctly parses the bare-string output format.
# =============================================================================


class TestSmolagentsScopeSupport:
    """Smolagents-scoped spans must be accepted and correctly normalized."""

    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_smolagents_tool_span(self):
        """TOOL span with smolagents scope produces ToolExecutionSpan with correct fields."""
        span = make_tool_span(
            tool_name="web_search",
            tool_input={"query": "weather london"},
            tool_output="Temperature: 15C, cloudy",
            scope_name=SMOLAGENTS_SCOPE_NAME,
        )
        session = self.mapper.map_to_session([span], "sess-1")

        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_call.name == "web_search"
        assert tool.tool_call.arguments == {"query": "weather london"}
        assert tool.tool_result.content == "Temperature: 15C, cloudy"

    def test_smolagents_llm_span(self):
        """LLM span with smolagents scope produces InferenceSpan."""
        span = make_llm_span(
            user_content="What is 2+2?",
            assistant_content="4",
            scope_name=SMOLAGENTS_SCOPE_NAME,
        )
        session = self.mapper.map_to_session([span], "sess-1")
        assert isinstance(session.traces[0].spans[0], InferenceSpan)

    @pytest.mark.parametrize(
        "output_value,expected_content",
        [("42", "42"), ("[1, 2, 3]", "[1, 2, 3]"), ('"quoted"', "quoted")],
        ids=["number", "list", "quoted-string"],
    )
    def test_non_dict_json_output_not_crash(self, output_value, expected_content):
        """TOOL span with non-dict JSON output.value must not crash."""
        span = make_span(
            name="compute_tool",
            scope_name=SMOLAGENTS_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.name": "compute_tool",
                "input.value": json.dumps({"x": 1}),
                "output.value": output_value,
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_result.content == expected_content

    def test_mixed_positional_and_kwargs_merged(self):
        """Positional args mapped via tool.parameters and merged with kwargs.

        {"args": ["tokyo", 5], "kwargs": {"offset": 10}} + params [query, limit, offset]
        → {"query": "tokyo", "limit": 5, "offset": 10}
        """
        span = make_span(
            name="search",
            scope_name=SMOLAGENTS_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.name": "search",
                "tool.parameters": json.dumps(
                    {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    }
                ),
                "input.value": json.dumps(
                    {
                        "args": ["tokyo", 5],
                        "kwargs": {"offset": 10},
                        "sanitize_inputs_outputs": False,
                    }
                ),
                "output.value": "results",
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        tool = session.traces[0].spans[0]
        assert tool.tool_call.arguments == {"query": "tokyo", "limit": 5, "offset": 10}

    def test_langchain_agent_span_rejected(self):
        """LangChain AGENT spans (e.g. route_to_agent) with input+output explicitly rejected."""
        span = make_span(
            name="route_to_agent",
            scope_name="openinference.instrumentation.langchain",
            attributes={
                "openinference.span.kind": "AGENT",
                "input.value": '{"agent_name": "research_agent"}',
                "output.value": '{"result": "done"}',
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        agent_spans = [s for t in session.traces for s in t.spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 0

    def test_agent_span_extracts_task_from_json_wrapper(self):
        """AGENT span input.value JSON wrapper is parsed to extract just the 'task' field.

        Smolagents CodeAgent.run emits input.value as:
          {"task": "What is 2+2?", "stream": false, "reset": true, ...}
        The mapper must return only the task string, not the full JSON wrapper.
        """
        span = make_span(
            name="CodeAgent.run",
            scope_name=SMOLAGENTS_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "AGENT",
                "input.value": json.dumps(
                    {
                        "task": "What is 2+2?",
                        "stream": False,
                        "reset": True,
                        "images": None,
                        "additional_args": None,
                    }
                ),
                "output.value": "The answer is 4.",
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")
        all_spans = [s for t in session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        assert agent_spans[0].user_prompt == "What is 2+2?"


# =============================================================================
# Integration Tests: Real smolagents fixture
# (captured from openinference-instrumentation-smolagents v0.1.31,
#  smolagents CodeAgent + DuckDuckGoSearchTool, scope
#  "openinference.instrumentation.smolagents")
#
# These tests use ACTUAL spans produced by the instrumentor, validating that the
# mapper handles real-world encoding:
# - LLM spans with plural `contents` path (message.contents.0.message_content.text)
# - TOOL spans with wrapper input.value ({"args":[], "kwargs":{...}, ...})
# - TOOL spans with bare-string output.value
# - Root AGENT span (CodeAgent.run) with input.value (task) and output.value (answer)
# =============================================================================


@pytest.fixture(scope="module")
def smolagents_session():
    """Map real smolagents spans to a Session."""
    spans = _load_smolagents_spans()
    mapper = OpenInferenceSessionMapper()
    return mapper.map_to_session(spans, "smolagents-sess")


class TestSmolagentsFixtureIntegration:
    """Integration tests using real smolagents trace (CodeAgent + DuckDuckGoSearchTool)."""

    def test_session_has_traces(self, smolagents_session):
        """Smolagents fixture produces at least one trace."""
        assert len(smolagents_session.traces) >= 1

    def test_produces_expected_span_types(self, smolagents_session):
        """Real trace produces InferenceSpan, ToolExecutionSpan, and AgentInvocationSpan."""
        all_spans = [s for t in smolagents_session.traces for s in t.spans]
        span_types = {type(s) for s in all_spans}
        assert InferenceSpan in span_types
        assert ToolExecutionSpan in span_types
        assert AgentInvocationSpan in span_types

    def test_plural_contents_path_normalized(self, smolagents_session):
        """LLM spans with plural contents path produce InferenceSpans with user+assistant."""
        all_spans = [s for t in smolagents_session.traces for s in t.spans]
        inference_spans = [s for s in all_spans if isinstance(s, InferenceSpan)]
        assert len(inference_spans) >= 1
        for span in inference_spans:
            assert span.messages[0].role.value == "user"
            assert span.messages[1].role.value == "assistant"

    def test_tool_spans_kwargs_normalized(self, smolagents_session):
        """web_search: kwargs wrapper normalized to logical {"query": "..."}."""
        all_spans = [s for t in smolagents_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        ws = next(s for s in tool_spans if s.tool_call.name == "web_search")
        assert ws.tool_call.arguments == {"query": "population of Tokyo"}

    def test_tool_spans_positional_args_mapped(self, smolagents_session):
        """FinalAnswerTool: positional arg mapped to "answer" via tool.parameters.

        {"args": ["...answer text..."], "kwargs": {}} → {"answer": "...answer text..."}
        """
        all_spans = [s for t in smolagents_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        fa = next(s for s in tool_spans if s.tool_call.name == "final_answer")

        expected_answer = (
            "The population of Tokyo is approximately 14 million people as of 2023. "
            "(The city proper has over 14 million residents, while the greater Tokyo "
            "metropolitan area has approximately 37 million people.)"
        )
        assert fa.tool_call.arguments == {"answer": expected_answer}
        assert fa.tool_call.arguments["answer"] == fa.tool_result.content

    def test_agent_span_extracts_task_field(self, smolagents_session):
        """AGENT span: parses {"task": "...", "stream": ...} → user_prompt = task value."""
        all_spans = [s for t in smolagents_session.traces for s in t.spans]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1

        agent = agent_spans[0]
        assert agent.user_prompt == "What is the population of Tokyo?"
        assert "population" in agent.agent_response.lower()

    def test_one_agent_span_per_trace(self, smolagents_session):
        """Each trace has at most 1 AgentInvocationSpan."""
        for trace in smolagents_session.traces:
            agent_spans = [s for s in trace.spans if isinstance(s, AgentInvocationSpan)]
            assert len(agent_spans) <= 1

    def test_no_empty_response_inference_spans(self, smolagents_session):
        """No InferenceSpan has empty-text-only assistant content."""
        all_spans = [s for t in smolagents_session.traces for s in t.spans]
        for span in all_spans:
            if isinstance(span, InferenceSpan):
                assistant_content = span.messages[1].content
                text_only = all(hasattr(c, "text") for c in assistant_content)
                if text_only:
                    assert any(c.text for c in assistant_content)


class TestClaudeAgentSdkScopeSupport:
    """Claude Agent SDK-scoped spans: acceptance and conversion."""

    def setup_method(self):
        self.mapper = OpenInferenceSessionMapper()

    def test_claude_agent_span_with_plain_text_input_detected(self):
        """Root AGENT span with plain-text input/output → AgentInvocationSpan."""
        spans = _load_claude_spans()
        # Root span has both input.value and output.value populated
        root_span = next(
            s
            for s in spans
            if s["attributes"].get("openinference.span.kind") == "AGENT"
            and s["attributes"].get("input.value")
            and s["attributes"].get("output.value")
        )
        session = self.mapper.map_to_session([root_span], "sess-1")

        agent_spans = [s for t in session.traces for s in t.spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        assert agent_spans[0].user_prompt == (
            "Look up the weather in New York and Seattle, then calculate the temperature difference."
        )
        assert agent_spans[0].agent_response == (
            "New York is 89°F and Seattle is 72°F. The temperature difference is 17°F — New York is warmer."
        )

    def test_nested_claude_agent_span_without_input_output_rejected(self):
        """Nested ClaudeAgentSDK.Agent span (no input/output) is not an agent invocation."""
        spans = _load_claude_spans()
        # Nested AGENT spans have kind=AGENT but no input.value/output.value
        nested_span = next(
            s
            for s in spans
            if s["attributes"].get("openinference.span.kind") == "AGENT" and not s["attributes"].get("input.value")
        )
        session = self.mapper.map_to_session([nested_span], "sess-1")

        all_spans = [s for t in session.traces for s in t.spans]
        assert not any(isinstance(s, AgentInvocationSpan) for s in all_spans)

    def test_claude_tool_span_json_input_parsed(self):
        """TOOL span with JSON input.value extracts tool_call.arguments correctly."""
        spans = _load_claude_spans()
        # Pick an Agent tool span (subagent delegation)
        agent_tool_span = next(
            s
            for s in spans
            if s["attributes"].get("openinference.span.kind") == "TOOL" and s["attributes"].get("tool.name") == "Agent"
        )
        session = self.mapper.map_to_session([agent_tool_span], "sess-1")

        tool_spans = [s for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        tool = tool_spans[0]
        assert tool.tool_call.name == "Agent"
        assert "subagent_type" in tool.tool_call.arguments
        assert "prompt" in tool.tool_call.arguments
        assert tool.tool_call.tool_call_id is not None

    def test_claude_tool_span_content_blocks_output(self):
        """TOOL output with content as list of blocks joins to newline-separated text."""
        span = make_span(
            name="Agent",
            scope_name=CLAUDE_SDK_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.id": "toolu_bdrk_xyz789",
                "tool.name": "Agent",
                "input.value": json.dumps(
                    {
                        "description": "Research task",
                        "subagent_type": "research-specialist",
                        "prompt": "Look up weather in NYC.",
                    }
                ),
                "output.value": json.dumps(
                    {
                        "status": "completed",
                        "content": [
                            {"type": "text", "text": "Temperature: 89°F"},
                            {"type": "text", "text": "Conditions: Partly cloudy"},
                        ],
                    }
                ),
            },
        )
        session = self.mapper.map_to_session([span], "sess-1")

        tool_spans = [s for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        assert tool_spans[0].tool_result.content == "Temperature: 89°F\nConditions: Partly cloudy"
        assert tool_spans[0].tool_result.error is None
        # Non-ASCII must be preserved literally, not escaped
        assert "\\u" not in tool_spans[0].tool_result.content

    @pytest.mark.parametrize(
        "status,span_events,expected_error",
        [
            pytest.param(
                {"code": "ERROR", "description": "EACCES: permission denied"},
                [],
                "EACCES: permission denied",
                id="description",
            ),
            pytest.param(
                {"code": "ERROR"},
                [
                    {
                        "event_name": "exception",
                        "timestamp": 1700000000500000000,
                        "attributes": {"exception.message": "Permission denied: /etc/shadow"},
                    }
                ],
                "Permission denied: /etc/shadow",
                id="exception_message",
            ),
            pytest.param({"code": "ERROR"}, [], "error", id="bare_fallback"),
            pytest.param(
                {"code": "ERROR"},
                [
                    {
                        "event_name": "exception",
                        "timestamp": 1700000000500000000,
                        "attributes": {"exception.message": '[{"type":"text","text":""}]'},
                    }
                ],
                '[{"type":"text","text":""}]',
                id="empty_text_block_falls_back_to_raw",
            ),
        ],
    )
    def test_claude_failed_tool_span_no_output_preserved(self, status, span_events, expected_error):
        """status=ERROR with no output.value is preserved via description/exception/fallback."""
        span = make_span(
            name="Bash",
            scope_name=CLAUDE_SDK_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.id": "toolu_failed",
                "tool.name": "Bash",
                "input.value": json.dumps({"command": "false"}),
            },
            span_events=span_events,
        )
        span["status"] = status

        session = self.mapper.map_to_session([span], "sess-1")

        tool_spans = [s for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        assert tool_spans[0].tool_result.error == "error"
        assert tool_spans[0].tool_result.content == expected_error

    def test_empty_text_block_among_non_text_blocks_preserves_siblings(self):
        """An empty text block among non-text blocks should preserve siblings as JSON, not discard."""
        non_text_block = {"type": "resource", "resource": {"text": "IMPORTANT DATA"}}
        span = make_span(
            name="ReadFile",
            scope_name=CLAUDE_SDK_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.id": "toolu_mixed",
                "tool.name": "ReadFile",
                "input.value": json.dumps({"path": "/tmp/data"}),
                "output.value": json.dumps(
                    {"status": "completed", "content": [{"type": "text", "text": ""}, non_text_block]}
                ),
            },
        )

        session = self.mapper.map_to_session([span], "sess-1")

        tool_spans = [s for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        # Non-text sibling must be preserved (as JSON), not silently discarded
        assert "IMPORTANT DATA" in tool_spans[0].tool_result.content

    def test_exception_message_returns_first_not_last(self):
        """_exception_message returns the first exception event, not the last."""
        span = make_span(
            name="Bash",
            scope_name=CLAUDE_SDK_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.id": "toolu_multi_exc",
                "tool.name": "Bash",
                "input.value": json.dumps({"command": "fail"}),
            },
            span_events=[
                {
                    "event_name": "exception",
                    "timestamp": 1700000000100000000,
                    "attributes": {"exception.message": "first error"},
                },
                {
                    "event_name": "exception",
                    "timestamp": 1700000000200000000,
                    "attributes": {"exception.message": "second error"},
                },
            ],
        )
        span["status"] = {"code": "ERROR"}

        session = self.mapper.map_to_session([span], "sess-1")

        tool_spans = [s for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        assert tool_spans[0].tool_result.content == "first error"

    @pytest.mark.parametrize(
        "output_value,status,span_events,expected_content",
        [
            pytest.param(
                json.dumps([{"type": "text", "text": "EACCES: permission denied"}]),
                {"code": "OK"},
                [],
                "EACCES: permission denied",
                id="bare_list_success",
            ),
            pytest.param(
                None,
                {"code": "ERROR"},
                [
                    {
                        "event_name": "exception",
                        "timestamp": 1700000000500000000,
                        "attributes": {
                            "exception.message": json.dumps(
                                [{"type": "text", "text": "EACCES: permission denied, open '/etc/shadow'"}]
                            )
                        },
                    }
                ],
                "EACCES: permission denied, open '/etc/shadow'",
                id="error_path_block_list",
            ),
        ],
    )
    def test_content_block_flattening_beyond_dict_envelope(self, output_value, status, span_events, expected_content):
        """Content blocks are flattened on both the bare-list success path and the ERROR path."""
        attrs = {
            "openinference.span.kind": "TOOL",
            "tool.id": "toolu_flatten",
            "tool.name": "Bash",
            "input.value": json.dumps({"command": "ls"}),
        }
        if output_value is not None:
            attrs["output.value"] = output_value
        span = make_span(
            name="Bash",
            scope_name=CLAUDE_SDK_SCOPE_NAME,
            attributes=attrs,
            span_events=span_events,
        )
        span["status"] = status

        session = self.mapper.map_to_session([span], "sess-1")

        tool_spans = [s for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        assert tool_spans[0].tool_result.content == expected_content

    def test_ensure_ascii_false_on_json_dumps_fallback(self):
        """Non-text blocks with unicode hit json.dumps and must not produce escape sequences."""
        span = make_span(
            name="ImageGen",
            scope_name=CLAUDE_SDK_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.id": "toolu_img",
                "tool.name": "ImageGen",
                "input.value": json.dumps({"prompt": "weather"}),
                "output.value": json.dumps({"status": "completed", "content": [{"type": "image", "alt": "25°C 東京"}]}),
            },
        )

        session = self.mapper.map_to_session([span], "sess-1")

        tool_spans = [s for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        assert "25°C 東京" in tool_spans[0].tool_result.content
        assert "\\u" not in tool_spans[0].tool_result.content

    def test_non_str_text_value_not_blanked(self):
        """A block with text: 500 (non-str) must preserve the block as JSON, not return ''."""
        span = make_span(
            name="Bash",
            scope_name=CLAUDE_SDK_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.id": "toolu_nonstr",
                "tool.name": "Bash",
                "input.value": json.dumps({"command": "echo 500"}),
                "output.value": json.dumps({"status": "completed", "content": [{"type": "text", "text": 500}]}),
            },
        )

        session = self.mapper.map_to_session([span], "sess-1")

        tool_spans = [s for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        assert tool_spans[0].tool_result.content != ""

    def test_errored_agent_span_detected(self):
        """AGENT span with input.value + ERROR status (no output.value) is detected."""
        span = make_span(
            name="query",
            scope_name=CLAUDE_SDK_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "AGENT",
                "input.value": "What is the weather?",
            },
        )
        span["status"] = {"code": "ERROR"}

        assert self.mapper._is_agent_invocation_span(span) is True

    def test_errored_agent_span_without_input_not_detected(self):
        """AGENT span with ERROR status but no input.value is not detected."""
        span = make_span(
            name="query",
            scope_name=CLAUDE_SDK_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "AGENT",
            },
        )
        span["status"] = {"code": "ERROR"}

        assert self.mapper._is_agent_invocation_span(span) is False

    @pytest.mark.parametrize(
        "status,span_events,expected_response",
        [
            pytest.param(
                {"code": "ERROR", "description": "context deadline exceeded"},
                [{"event_name": "exception", "timestamp": 0, "attributes": {"exception.message": "ignored"}}],
                "context deadline exceeded",
                id="prefers_status_description",
            ),
            pytest.param(
                {"code": "ERROR"},
                [{"event_name": "exception", "timestamp": 0, "attributes": {"exception.message": "rate limit"}}],
                "rate limit",
                id="falls_back_to_exception_message",
            ),
            pytest.param(
                {"code": "ERROR"},
                [],
                "error",
                id="falls_back_to_generic_error",
            ),
        ],
    )
    def test_errored_agent_span_response_fallback(self, status, span_events, expected_response):
        """Errored agent span surfaces error via: status.description > exception > 'error'."""
        span = make_span(
            name="query",
            scope_name=CLAUDE_SDK_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "AGENT",
                "input.value": "What is the weather?",
            },
            span_events=span_events,
        )
        span["status"] = status

        session = self.mapper.map_to_session([span], "sess-1")

        agent_spans = [s for t in session.traces for s in t.spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        assert agent_spans[0].user_prompt == "What is the weather?"
        assert agent_spans[0].agent_response == expected_response

    def test_claude_agent_span_preserves_json_prompt_with_task_key(self):
        """Claude prompt that is a JSON object with 'task' key is NOT unwrapped."""
        json_prompt = json.dumps({"task": "summarize", "context": "some data"})
        span = make_span(
            name="query",
            scope_name=CLAUDE_SDK_SCOPE_NAME,
            attributes={
                "openinference.span.kind": "AGENT",
                "input.value": json_prompt,
                "output.value": "Here is the summary.",
            },
        )

        session = self.mapper.map_to_session([span], "sess-1")

        agent_spans = [s for t in session.traces for s in t.spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        assert agent_spans[0].user_prompt == json_prompt


@pytest.fixture()
def claude_session():
    """Map real Claude Agent SDK spans to a Session."""
    spans = _load_claude_spans()
    mapper = OpenInferenceSessionMapper()
    return mapper.map_to_session(spans, "claude-sess")


class TestClaudeFixtureIntegration:
    """Integration tests using real Claude Agent SDK trace (all 10 spans fed together)."""

    def test_span_counts_and_types(self, claude_session):
        """All 10 fixture spans map to 6 ToolExecutionSpans + 1 AgentInvocationSpan.

        The 3 nested AGENT spans (no input/output) are rejected.
        """
        assert len(claude_session.traces) >= 1
        all_spans = [s for t in claude_session.traces for s in t.spans]
        assert len(all_spans) == 7

        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(tool_spans) == 6
        assert len(agent_spans) == 1

    def test_agent_span_has_prompt_and_response(self, claude_session):
        """Root agent span has user_prompt and agent_response populated."""
        all_spans = [s for t in claude_session.traces for s in t.spans]
        agent = next(s for s in all_spans if isinstance(s, AgentInvocationSpan))
        assert agent.user_prompt
        assert agent.agent_response

    def test_agent_span_metadata_has_token_counts(self, claude_session):
        """Root agent span metadata contains model name and token counts."""
        all_spans = [s for t in claude_session.traces for s in t.spans]
        agent = next(s for s in all_spans if isinstance(s, AgentInvocationSpan))
        assert agent.metadata["llm.model_name"] == "us.anthropic.claude-sonnet-4-6"
        assert agent.metadata["llm.token_count.total"] > 0

    def test_tool_spans_have_call_ids_and_content(self, claude_session):
        """All tool spans have tool_call_id populated and non-empty content."""
        all_spans = [s for t in claude_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        for tool_span in tool_spans:
            assert tool_span.tool_call.tool_call_id is not None
            assert tool_span.tool_result.content


@pytest.fixture()
def claude_adot_session():
    """Map real Claude AgentCore spans (session.id on all spans) to a Session."""
    session_id, spans = _load_claude_adot_spans()
    mapper = OpenInferenceSessionMapper()
    return mapper.map_to_session(spans, session_id)


class TestClaudeAgentCoreFixtureIntegration:
    """Integration tests using Claude Agent SDK spans from AgentCore deployment.

    AgentCore's span processor stamps session.id on all spans, so the full set
    survives a CloudWatch Logs Insights query filtered by session.id.
    """

    def test_span_counts(self, claude_adot_session):
        """AgentCore trace produces 4 tool spans + 1 agent span."""
        all_spans = [s for t in claude_adot_session.traces for s in t.spans]
        tool_spans = [s for s in all_spans if isinstance(s, ToolExecutionSpan)]
        agent_spans = [s for s in all_spans if isinstance(s, AgentInvocationSpan)]
        assert len(tool_spans) == 4
        assert len(agent_spans) == 1

    def test_all_spans_have_session_id_attribute(self, claude_adot_session):
        """Verify the fixture has session.id on all spans (AgentCore behavior)."""
        session_id, spans = _load_claude_adot_spans()
        for span in spans:
            attrs = span.get("attributes", {})
            assert attrs.get("session.id") == session_id
