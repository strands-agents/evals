"""Tests for OpenInferenceSessionMapper - OpenInference LangChain trace → Session conversion."""

import json

from strands_evals.mappers import OpenInferenceSessionMapper
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    ToolExecutionSpan,
)

SCOPE_NAME = "openinference.instrumentation.langchain"


def make_span(
    trace_id="trace-1",
    span_id="span-1",
    parent_span_id=None,
    name="test-span",
    attributes=None,
    span_events=None,
    start_time=1700000000000000000,
    end_time=1700000001000000000,
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
        "scope": {"name": SCOPE_NAME, "version": "0.1.0"},
        "status": {"code": "OK"},
        "span_events": span_events or [],
    }


def make_llm_span(
    trace_id="trace-1",
    span_id="span-1",
    user_content="Hello",
    assistant_content="Hi there",
    tool_calls=None,
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

    return make_span(trace_id=trace_id, span_id=span_id, attributes=attrs)


def make_tool_span(
    trace_id="trace-1",
    span_id="span-1",
    tool_name="calculator",
    tool_input=None,
    tool_output=None,
    tool_call_id="tool-1",
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

    return make_span(trace_id=trace_id, span_id=span_id, name=tool_name, attributes=attrs)


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
