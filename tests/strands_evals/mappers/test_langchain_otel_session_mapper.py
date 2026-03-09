"""Tests for LangChainOtelSessionMapper - Traceloop/OpenLLMetry trace → Session conversion."""

import json

from strands_evals.mappers import LangChainOtelSessionMapper
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    ToolExecutionSpan,
)

SCOPE_NAME = "opentelemetry.instrumentation.langchain"


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
    """Build a normalized span dict for LangChain OTEL traces."""
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


def make_inference_span(
    trace_id="trace-1",
    span_id="span-1",
    user_content="Hello",
    assistant_content="Hi there",
    tool_calls=None,
):
    """Build an LLM inference span with gen_ai.* attributes."""
    attrs = {
        "llm.request.type": "chat",
        "gen_ai.prompt.0.role": "user",
        "gen_ai.prompt.0.content": user_content,
        "gen_ai.completion.0.role": "assistant",
        "gen_ai.completion.0.content": assistant_content,
    }

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            attrs[f"gen_ai.completion.0.tool_calls.{i}.name"] = tc["name"]
            attrs[f"gen_ai.completion.0.tool_calls.{i}.arguments"] = json.dumps(tc.get("arguments", {}))
            attrs[f"gen_ai.completion.0.tool_calls.{i}.id"] = tc.get("id", f"tool-{i}")

    return make_span(trace_id=trace_id, span_id=span_id, attributes=attrs)


def make_tool_span(
    trace_id="trace-1",
    span_id="span-1",
    tool_name="calculator",
    tool_input=None,
    tool_output=None,
    tool_call_id="tool-1",
):
    """Build a tool execution span with traceloop.* attributes."""
    tool_input = tool_input or {"expr": "2+2"}
    tool_output = tool_output or "4"

    entity_input = json.dumps({"inputs": tool_input})
    entity_output = json.dumps(
        {
            "output": {
                "kwargs": {
                    "content": tool_output,
                    "tool_call_id": tool_call_id,
                    "status": "success",
                }
            }
        }
    )

    attrs = {
        "traceloop.span.kind": "tool",
        "traceloop.entity.name": tool_name,
        "traceloop.entity.input": entity_input,
        "traceloop.entity.output": entity_output,
    }

    return make_span(trace_id=trace_id, span_id=span_id, attributes=attrs)


def make_workflow_span(
    trace_id="trace-1",
    span_id="span-1",
    user_query="What is 2+2?",
    agent_response="The answer is 4",
):
    """Build a workflow/agent invocation span."""
    entity_input = json.dumps({"inputs": {"messages": [{"kwargs": {"content": user_query, "type": "human"}}]}})
    entity_output = json.dumps({"outputs": {"messages": [{"kwargs": {"content": agent_response, "type": "ai"}}]}})

    attrs = {
        "traceloop.span.kind": "workflow",
        "traceloop.entity.input": entity_input,
        "traceloop.entity.output": entity_output,
    }

    return make_span(trace_id=trace_id, span_id=span_id, attributes=attrs)


class TestSpanTypeDetection:
    def setup_method(self):
        self.mapper = LangChainOtelSessionMapper()

    def test_inference_span_detected(self):
        """Span with llm.request.type=chat is detected as inference span."""
        span = make_span(attributes={"llm.request.type": "chat"})
        assert self.mapper._is_inference_span(span) is True
        assert self.mapper._is_tool_execution_span(span) is False
        assert self.mapper._is_agent_invocation_span(span) is False

    def test_tool_span_detected(self):
        """Span with traceloop.span.kind=tool is detected as tool span."""
        span = make_span(attributes={"traceloop.span.kind": "tool"})
        assert self.mapper._is_inference_span(span) is False
        assert self.mapper._is_tool_execution_span(span) is True
        assert self.mapper._is_agent_invocation_span(span) is False

    def test_workflow_span_detected(self):
        """Span with traceloop.span.kind=workflow is detected as agent span."""
        span = make_span(attributes={"traceloop.span.kind": "workflow"})
        assert self.mapper._is_inference_span(span) is False
        assert self.mapper._is_tool_execution_span(span) is False
        assert self.mapper._is_agent_invocation_span(span) is True


class TestInferenceSpanConversion:
    def setup_method(self):
        self.mapper = LangChainOtelSessionMapper()

    def test_basic_inference_span(self):
        """Basic inference span with user and assistant messages."""
        span = make_inference_span(
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

    def test_inference_span_with_tool_calls(self):
        """Inference span with tool calls in assistant response."""
        span = make_inference_span(
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
        self.mapper = LangChainOtelSessionMapper()

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
        assert tool.tool_call.arguments == {"expr": "6*7"}
        assert tool.tool_result.content == "42"
        assert tool.tool_result.tool_call_id == "tc-1"


class TestAgentInvocationSpanConversion:
    def setup_method(self):
        self.mapper = LangChainOtelSessionMapper()

    def test_basic_agent_span(self):
        """Agent invocation span with user query and response."""
        span = make_workflow_span(
            user_query="What is the capital of France?",
            agent_response="The capital of France is Paris.",
        )
        session = self.mapper.map_to_session([span], "sess-1")

        assert len(session.traces[0].spans) == 1
        agent = session.traces[0].spans[0]
        assert isinstance(agent, AgentInvocationSpan)
        assert agent.user_prompt == "What is the capital of France?"
        assert agent.agent_response == "The capital of France is Paris."

    def test_agent_span_collects_tools(self):
        """Agent invocation span collects tools from inference spans."""
        inference = make_inference_span(
            trace_id="t1",
            span_id="s1",
            user_content="Calculate",
            assistant_content="OK",
        )
        inference["attributes"]["llm.request.functions.0.name"] = "calculator"
        inference["attributes"]["llm.request.functions.0.description"] = "A calculator"

        workflow = make_workflow_span(
            trace_id="t1",
            span_id="s2",
            user_query="Calculate 2+2",
            agent_response="4",
        )

        session = self.mapper.map_to_session([inference, workflow], "sess-1")

        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        assert len(agent_spans[0].available_tools) == 1
        assert agent_spans[0].available_tools[0].name == "calculator"


class TestSessionBuilding:
    def setup_method(self):
        self.mapper = LangChainOtelSessionMapper()

    def test_empty_spans_list(self):
        """Empty spans list produces empty session."""
        session = self.mapper.map_to_session([], "sess-1")
        assert session.session_id == "sess-1"
        assert session.traces == []

    def test_multiple_traces(self):
        """Spans with different trace_ids grouped into separate traces."""
        span1 = make_inference_span(trace_id="t1", span_id="s1", user_content="Q1", assistant_content="A1")
        span2 = make_inference_span(trace_id="t2", span_id="s2", user_content="Q2", assistant_content="A2")

        session = self.mapper.map_to_session([span1, span2], "sess-1")

        assert len(session.traces) == 2
        trace_ids = {t.trace_id for t in session.traces}
        assert trace_ids == {"t1", "t2"}

    def test_filters_by_scope(self):
        """Only spans with matching scope are processed."""
        langchain_span = make_inference_span(user_content="Q1", assistant_content="A1")
        other_span = make_span(
            span_id="other",
            attributes={"llm.request.type": "chat"},
        )
        other_span["scope"]["name"] = "other.scope"

        session = self.mapper.map_to_session([langchain_span, other_span], "sess-1")

        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1

    def test_full_agent_loop(self):
        """Full agent loop: inference → tool → inference → agent produces all span types."""
        spans = [
            make_inference_span(
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
            make_inference_span(
                trace_id="t1",
                span_id="s3",
                user_content="Tool result: 42",
                assistant_content="The answer is 42.",
            ),
            make_workflow_span(
                trace_id="t1",
                span_id="s4",
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
