"""Tests for ADKOtelSessionMapper - ADK OTel spans → Session conversion."""

import json
from datetime import datetime, timezone
from pathlib import Path

from strands_evals.mappers import ADKOtelSessionMapper
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    ToolCallContent,
    ToolExecutionSpan,
)

SESSION_ID = "test-session-1"
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_LIVE_SPANS_FILE = _FIXTURES_DIR / "adk_live_spans.json"


# ============================================================================
# Fixture Helpers
# ============================================================================


def make_span(
    trace_id="trace-1",
    span_id="span-1",
    parent_span_id=None,
    name="test-span",
    attributes=None,
    start_time=1700000000000000000,
    end_time=1700000001000000000,
):
    """Build a normalized ADK span dict."""
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "start_time": start_time,
        "end_time": end_time,
        "attributes": attributes or {},
        "scope": {"name": "gcp.vertex.agent", "version": "2.5.0"},
        "status": {"code": "UNSET"},
        "span_events": [],
    }


def make_llm_request(
    user_text="What is 15 multiplied by 37?",
    system_instruction="You are a math assistant.",
    tools=None,
    history=None,
):
    """Build a gcp.vertex.agent.llm_request JSON string."""
    contents = list(history) if history else [{"parts": [{"text": user_text}], "role": "user"}]
    config = {"system_instruction": system_instruction}
    if tools:
        config["tools"] = [{"function_declarations": tools}]
    return json.dumps({"model": "gemini-2.5-flash", "config": config, "contents": contents})


def make_llm_response_text(text="15 multiplied by 37 is 555."):
    """Build a text-only llm_response JSON string."""
    return json.dumps({"content": {"parts": [{"text": text}], "role": "model"}, "finish_reason": "STOP"})


def make_llm_response_tool_call(name="calculator", args=None, call_id="fc-001"):
    """Build a tool-call llm_response (Gemini 3+ style with id)."""
    args = args or {"expression": "15 * 37"}
    func_call = {"name": name, "args": args}
    if call_id is not None:
        func_call["id"] = call_id
    return json.dumps({"content": {"parts": [{"function_call": func_call}], "role": "model"}, "finish_reason": "STOP"})


def load_full_trace():
    """Load the full ADK trace fixture from JSON."""
    return json.loads(_LIVE_SPANS_FILE.read_text())


# ============================================================================
# Tests: Full Trace (fixture-based)
# ============================================================================


class TestFullTrace:
    """End-to-end session mapping with a complete ADK trace loaded from fixture."""

    def setup_method(self):
        self.mapper = ADKOtelSessionMapper()
        self.spans = load_full_trace()

    def test_session_structure(self):
        """Full trace produces correct session/trace/span counts and types."""
        session = self.mapper.map_to_session(self.spans, SESSION_ID)
        assert session.session_id == SESSION_ID
        assert len(session.traces) == 1
        trace = session.traces[0]
        assert len(trace.spans) == 4
        types = {type(s).__name__ for s in trace.spans}
        assert types == {"AgentInvocationSpan", "InferenceSpan", "ToolExecutionSpan"}

    def test_agent_invocation_span(self):
        """Agent span extracts prompt, response, system prompt, tools, and metadata."""
        session = self.mapper.map_to_session(self.spans, SESSION_ID)
        agent = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)][0]
        assert agent.user_prompt == "What is 15 multiplied by 37?"
        assert agent.agent_response == "15 multiplied by 37 is 555."
        assert "math assistant" in agent.system_prompt
        assert len(agent.available_tools) == 1
        assert agent.available_tools[0].name == "calculator"
        assert agent.metadata["agent_name"] == "math_agent"

    def test_tool_execution_span(self):
        """Tool span extracts name, arguments, call_id, and response."""
        session = self.mapper.map_to_session(self.spans, SESSION_ID)
        tool = [s for s in session.traces[0].spans if isinstance(s, ToolExecutionSpan)][0]
        assert tool.tool_call.name == "calculator"
        assert tool.tool_call.arguments == {"expression": "15 * 37"}
        assert tool.tool_call.tool_call_id == "uwpsprd2"
        assert "555" in tool.tool_result.content

    def test_inference_spans(self):
        """Inference spans extract messages and metadata from parent call_llm."""
        session = self.mapper.map_to_session(self.spans, SESSION_ID)
        inf_spans = [s for s in session.traces[0].spans if isinstance(s, InferenceSpan)]
        assert len(inf_spans) == 2
        first = inf_spans[0]
        assert len(first.messages) >= 2
        assert first.metadata["model"] == "gemini-3.5-flash"
        assert first.metadata["input_tokens"] == 157
        assert first.metadata["reasoning_tokens"] == 32
        assert first.metadata["invocation_id"] == "e-5742fbee-9bb9-4445-b224-7ae0041d6631"

    def test_tool_call_id_in_messages(self):
        """tool_call_id is read from function_call.id in Gemini 3+ payloads."""
        session = self.mapper.map_to_session(self.spans, SESSION_ID)
        inf_spans = [s for s in session.traces[0].spans if isinstance(s, InferenceSpan)]
        first = inf_spans[0]
        assistant_msgs = [m for m in first.messages if isinstance(m, AssistantMessage)]
        tool_calls = [c for m in assistant_msgs for c in m.content if isinstance(c, ToolCallContent)]
        assert tool_calls[0].tool_call_id == "uwpsprd2"


# ============================================================================
# Tests: Tool Execution Span Conversion
# ============================================================================


class TestToolExecutionSpan:
    """Tests for tool execution span conversion."""

    def setup_method(self):
        self.mapper = ADKOtelSessionMapper()

    def test_basic_tool_span(self):
        """Tool span with standard attributes is converted correctly."""
        spans = [
            make_span(
                span_id="tool-1",
                name="execute_tool calculator",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "calculator",
                    "gen_ai.tool.call.id": "call-123",
                    "gcp.vertex.agent.tool_call_args": '{"expression": "2+2"}',
                    "gcp.vertex.agent.tool_response": '{"result": "4"}',
                },
            )
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        tool = session.traces[0].spans[0]
        assert isinstance(tool, ToolExecutionSpan)
        assert tool.tool_call.name == "calculator"
        assert tool.tool_call.arguments == {"expression": "2+2"}
        assert tool.tool_result.content == '{"result": "4"}'
        assert tool.tool_result.error is None

    def test_missing_tool_name_skipped(self):
        """Tool span without gen_ai.tool.name is skipped."""
        spans = [
            make_span(
                name="execute_tool",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gcp.vertex.agent.tool_call_args": "{}",
                    "gcp.vertex.agent.tool_response": "ok",
                },
            )
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        assert session.traces == []

    def test_metadata_populated(self):
        """Tool span metadata includes description, type, and event_id."""
        spans = [
            make_span(
                name="execute_tool calculator",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "calculator",
                    "gen_ai.tool.description": "Does math",
                    "gen_ai.tool.type": "FunctionTool",
                    "gen_ai.tool.call.id": "call-1",
                    "gcp.vertex.agent.tool_call_args": "{}",
                    "gcp.vertex.agent.tool_response": "result",
                    "gcp.vertex.agent.event_id": "evt-123",
                },
            )
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        tool = session.traces[0].spans[0]
        assert tool.metadata["description"] == "Does math"
        assert tool.metadata["tool_type"] == "FunctionTool"
        assert tool.metadata["event_id"] == "evt-123"

    def test_tool_error_preserved(self):
        """Tool span with ERROR status populates ToolResult.error."""
        spans = [
            {
                "trace_id": "trace-1",
                "span_id": "tool-err",
                "parent_span_id": None,
                "name": "execute_tool failing_tool",
                "start_time": 1700000000000000000,
                "end_time": 1700000001000000000,
                "attributes": {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "failing_tool",
                    "gen_ai.tool.call.id": "call-err",
                    "gcp.vertex.agent.tool_call_args": "{}",
                    "gcp.vertex.agent.tool_response": "",
                    "error.type": "MCP_TOOL_ERROR",
                },
                "scope": {"name": "gcp.vertex.agent", "version": "2.5.0"},
                "status": {"code": "ERROR", "description": "Tool execution failed"},
                "span_events": [],
            }
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        tool = session.traces[0].spans[0]
        assert tool.tool_result.error == "MCP_TOOL_ERROR"


# ============================================================================
# Tests: Inference Span Conversion
# ============================================================================


class TestInferenceSpan:
    """Tests for inference span conversion."""

    def setup_method(self):
        self.mapper = ADKOtelSessionMapper()

    def test_simple_text_response(self):
        """generate_content with parent call_llm produces inference span."""
        spans = [
            make_span(
                span_id="callllm-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(),
                    "gcp.vertex.agent.llm_response": make_llm_response_text("Hello!"),
                },
            ),
            make_span(
                span_id="gen-1",
                parent_span_id="callllm-1",
                name="generate_content gemini-2.5-flash",
                attributes={"gen_ai.operation.name": "generate_content", "gen_ai.request.model": "gemini-2.5-flash"},
            ),
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        inf = [s for s in session.traces[0].spans if isinstance(s, InferenceSpan)]
        assert len(inf) == 1
        assert len(inf[0].messages) == 2

    def test_orphan_generate_content_skipped(self):
        """generate_content without parent call_llm is skipped."""
        spans = [
            make_span(
                name="generate_content gemini-2.5-flash",
                attributes={"gen_ai.operation.name": "generate_content"},
            )
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        assert session.traces == []

    def test_empty_llm_request_skipped(self):
        """call_llm with empty llm_request ('{}') produces no inference span."""
        spans = [
            make_span(
                span_id="callllm-1",
                name="call_llm",
                attributes={"gcp.vertex.agent.llm_request": "{}", "gcp.vertex.agent.llm_response": "{}"},
            ),
            make_span(
                span_id="gen-1",
                parent_span_id="callllm-1",
                name="generate_content gemini-2.5-flash",
                attributes={"gen_ai.operation.name": "generate_content"},
            ),
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        assert session.traces == []

    def test_thought_parts_filtered(self):
        """Thought parts (thought=True) are excluded from the assistant message."""
        llm_response = json.dumps(
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "Let me think..."},
                        {"text": "The answer is 42."},
                    ],
                    "role": "model",
                },
                "finish_reason": "STOP",
            }
        )
        spans = [
            make_span(
                span_id="callllm-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(user_text="Question?"),
                    "gcp.vertex.agent.llm_response": llm_response,
                },
            ),
            make_span(
                span_id="gen-1",
                parent_span_id="callllm-1",
                name="generate_content gemini-3.5-flash",
                attributes={"gen_ai.operation.name": "generate_content"},
            ),
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        inf = [s for s in session.traces[0].spans if isinstance(s, InferenceSpan)]
        assistant_msgs = [m for m in inf[0].messages if isinstance(m, AssistantMessage)]
        assert len(assistant_msgs[0].content) == 1
        assert assistant_msgs[0].content[0].text == "The answer is 42."

    def test_tool_call_id_none_for_gemini2(self):
        """tool_call_id is None when function_call lacks id field (Gemini 2.x)."""
        history = [
            {"parts": [{"text": "What is 2+2?"}], "role": "user"},
            {"parts": [{"function_call": {"name": "calc", "args": {"x": "2+2"}}}], "role": "model"},
            {"parts": [{"function_response": {"name": "calc", "response": {"r": "4"}}}], "role": "user"},
        ]
        spans = [
            make_span(
                span_id="callllm-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(history=history),
                    "gcp.vertex.agent.llm_response": make_llm_response_tool_call(call_id=None),
                },
            ),
            make_span(
                span_id="gen-1",
                parent_span_id="callllm-1",
                name="generate_content gemini-2.5-flash",
                attributes={"gen_ai.operation.name": "generate_content"},
            ),
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        inf = [s for s in session.traces[0].spans if isinstance(s, InferenceSpan)]
        assistant_msgs = [m for m in inf[0].messages if isinstance(m, AssistantMessage)]
        tool_calls = [c for m in assistant_msgs for c in m.content if isinstance(c, ToolCallContent)]
        assert all(tc.tool_call_id is None for tc in tool_calls)

    def test_multiple_tool_calls_in_single_response(self):
        """LLM response with two function_calls produces two ToolCallContent items."""
        llm_response = json.dumps(
            {
                "content": {
                    "parts": [
                        {"function_call": {"id": "c-1", "name": "get_weather", "args": {"city": "Seattle"}}},
                        {"function_call": {"id": "c-2", "name": "get_weather", "args": {"city": "Tokyo"}}},
                    ],
                    "role": "model",
                },
                "finish_reason": "STOP",
            }
        )
        spans = [
            make_span(
                span_id="callllm-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(user_text="Weather?"),
                    "gcp.vertex.agent.llm_response": llm_response,
                },
            ),
            make_span(
                span_id="gen-1",
                parent_span_id="callllm-1",
                name="generate_content gemini-3.5-flash",
                attributes={"gen_ai.operation.name": "generate_content"},
            ),
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        inf = [s for s in session.traces[0].spans if isinstance(s, InferenceSpan)]
        assistant_msgs = [m for m in inf[0].messages if isinstance(m, AssistantMessage)]
        tool_calls = [c for m in assistant_msgs for c in m.content if isinstance(c, ToolCallContent)]
        assert len(tool_calls) == 2
        assert tool_calls[0].tool_call_id == "c-1"
        assert tool_calls[1].tool_call_id == "c-2"


# ============================================================================
# Tests: Agent Invocation Span Conversion
# ============================================================================


class TestAgentInvocationSpan:
    """Tests for agent invocation span conversion."""

    def setup_method(self):
        self.mapper = ADKOtelSessionMapper()

    def test_no_child_call_llm_skipped(self):
        """invoke_agent without child call_llm spans is skipped."""
        spans = [
            make_span(
                name="invoke_agent math_agent",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "math_agent"},
            )
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        assert session.traces == []

    def test_with_single_call_llm(self):
        """invoke_agent with one call_llm child extracts prompt, response, system prompt."""
        spans = [
            make_span(
                span_id="agent-1",
                name="invoke_agent x",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "x"},
            ),
            make_span(
                span_id="callllm-1",
                parent_span_id="agent-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(
                        user_text="Hello", system_instruction="Be helpful."
                    ),
                    "gcp.vertex.agent.llm_response": make_llm_response_text("Hi there!"),
                },
            ),
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        agent = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)][0]
        assert agent.user_prompt == "Hello"
        assert agent.agent_response == "Hi there!"
        assert agent.system_prompt == "Be helpful."

    def test_skip_summarization_fallback(self):
        """When last call_llm has no text response, agent_response uses the last tool result."""
        llm_response_tool_only = json.dumps(
            {
                "content": {
                    "parts": [{"function_call": {"id": "fc-1", "name": "lookup", "args": {"q": "x"}}}],
                    "role": "model",
                },
                "finish_reason": "STOP",
            }
        )
        spans = [
            make_span(
                span_id="agent-1",
                name="invoke_agent helper",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "helper"},
            ),
            make_span(
                span_id="callllm-1",
                parent_span_id="agent-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(user_text="Find it"),
                    "gcp.vertex.agent.llm_response": llm_response_tool_only,
                },
                start_time=1700000001000000000,
                end_time=1700000002000000000,
            ),
            make_span(
                span_id="tool-1",
                parent_span_id="callllm-1",
                name="execute_tool lookup",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "lookup",
                    "gen_ai.tool.call.id": "fc-1",
                    "gcp.vertex.agent.tool_call_args": '{"q": "x"}',
                    "gcp.vertex.agent.tool_response": "THE ANSWER",
                },
                start_time=1700000002000000000,
                end_time=1700000003000000000,
            ),
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        agent = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)][0]
        assert agent.agent_response == "THE ANSWER"
        assert agent.user_prompt == "Find it"

    def test_preamble_text_ignored_when_function_call_present(self):
        """When response has both text and function_call, agent_response comes from tool result."""
        llm_response = json.dumps(
            {
                "content": {
                    "parts": [
                        {"text": "I will check."},
                        {"function_call": {"id": "fc-1", "name": "lookup", "args": {"q": "x"}}},
                    ],
                    "role": "model",
                },
                "finish_reason": "STOP",
            }
        )
        spans = [
            make_span(
                span_id="agent-1",
                name="invoke_agent helper",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "helper"},
            ),
            make_span(
                span_id="callllm-1",
                parent_span_id="agent-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(user_text="Answer?"),
                    "gcp.vertex.agent.llm_response": llm_response,
                },
                start_time=1700000001000000000,
                end_time=1700000002000000000,
            ),
            make_span(
                span_id="tool-1",
                parent_span_id="callllm-1",
                name="execute_tool lookup",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "lookup",
                    "gen_ai.tool.call.id": "fc-1",
                    "gcp.vertex.agent.tool_call_args": '{"q": "x"}',
                    "gcp.vertex.agent.tool_response": '{"result": "A"}',
                },
                start_time=1700000002000000000,
                end_time=1700000003000000000,
            ),
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        agent = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)][0]
        assert agent.agent_response == '{"result": "A"}'

    def test_user_prompt_extracts_latest_message(self):
        """In multi-turn conversations, user_prompt is the latest user message."""
        history = [
            {"parts": [{"text": "First question"}], "role": "user"},
            {"parts": [{"function_call": {"name": "calc", "args": {}}}], "role": "model"},
            {"parts": [{"function_response": {"name": "calc", "response": {}}}], "role": "user"},
            {"parts": [{"text": "Second question"}], "role": "user"},
        ]
        spans = [
            make_span(
                span_id="agent-1",
                name="invoke_agent x",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "x"},
            ),
            make_span(
                span_id="callllm-1",
                parent_span_id="agent-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(),
                    "gcp.vertex.agent.llm_response": make_llm_response_tool_call(),
                },
                start_time=1700000001000000000,
                end_time=1700000002000000000,
            ),
            make_span(
                span_id="callllm-2",
                parent_span_id="agent-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(history=history),
                    "gcp.vertex.agent.llm_response": make_llm_response_text("30"),
                },
                start_time=1700000003000000000,
                end_time=1700000004000000000,
            ),
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        agent = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)][0]
        assert agent.user_prompt == "Second question"
        assert agent.agent_response == "30"


# ============================================================================
# Tests: Session-Level Behavior
# ============================================================================


class TestSessionBehavior:
    """Tests for session-level grouping and edge cases."""

    def setup_method(self):
        self.mapper = ADKOtelSessionMapper()

    def test_empty_spans_returns_empty_session(self):
        """Empty spans list produces empty session."""
        session = self.mapper.map_to_session([], SESSION_ID)
        assert session.session_id == SESSION_ID
        assert session.traces == []

    def test_multiple_traces_grouped(self):
        """Spans with different trace_ids grouped into separate traces."""
        spans = [
            make_span(
                trace_id="trace-1",
                span_id="agent-1",
                name="invoke_agent a",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "a"},
            ),
            make_span(
                trace_id="trace-1",
                span_id="callllm-1",
                parent_span_id="agent-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(user_text="Hi"),
                    "gcp.vertex.agent.llm_response": make_llm_response_text("Hello"),
                },
            ),
            make_span(
                trace_id="trace-2",
                span_id="agent-2",
                name="invoke_agent b",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "b"},
            ),
            make_span(
                trace_id="trace-2",
                span_id="callllm-2",
                parent_span_id="agent-2",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(user_text="Bye"),
                    "gcp.vertex.agent.llm_response": make_llm_response_text("Goodbye"),
                },
            ),
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        assert {t.trace_id for t in session.traces} == {"trace-1", "trace-2"}


# ============================================================================
# Tests: Data Format Compatibility
# ============================================================================


class TestDataFormatCompatibility:
    """Tests for handling different span dict formats."""

    def setup_method(self):
        self.mapper = ADKOtelSessionMapper()

    def test_to_json_format_with_hex_prefix(self):
        """Spans in to_json format (context.trace_id, parent_id with 0x prefix) are parsed."""
        spans = [
            {
                "name": "invoke_agent my_agent",
                "context": {"trace_id": "0xabc123", "span_id": "0x789abc"},
                "start_time": "2026-07-22T16:34:19.900000Z",
                "end_time": "2026-07-22T16:34:19.920000Z",
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": "my_agent",
                    "gcp.vertex.agent.llm_request": "",
                },
                "events": [
                    {
                        "name": "gen_ai.user.message",
                        "timestamp": "2026-07-22T16:34:19.900000Z",
                        "attributes": {"content": "hello"},
                    }
                ],
            },
            {
                "name": "execute_tool calculator",
                "context": {"trace_id": "0xabc123", "span_id": "0xdef456"},
                "parent_id": "0x789abc",
                "start_time": "2026-07-22T16:34:19.917561Z",
                "end_time": "2026-07-22T16:34:19.917765Z",
                "attributes": {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "calculator",
                    "gen_ai.tool.call.id": "call-1",
                    "gcp.vertex.agent.tool_call_args": '{"x": 1}',
                    "gcp.vertex.agent.tool_response": "result",
                },
            },
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        tool_spans = [s for s in session.traces[0].spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        tool = tool_spans[0]
        assert tool.span_info.trace_id == "abc123"
        assert tool.span_info.span_id == "def456"
        assert tool.span_info.parent_span_id == "789abc"


# ============================================================================
# Tests: Timestamp Parsing
# ============================================================================


class TestTimestampParsing:
    """Tests for parse_timestamp handling various formats."""

    def setup_method(self):
        self.mapper = ADKOtelSessionMapper()

    def test_iso_string_with_z(self):
        ts = self.mapper.parse_timestamp("2026-07-22T16:34:19.917561Z")
        assert ts.year == 2026 and ts.month == 7

    def test_nanosecond_epoch(self):
        """Nanosecond epoch integer is converted to datetime."""
        ts_int = self.mapper.parse_timestamp(1700000000000000000)
        assert ts_int.year == 2023

    def test_string_nanosecond_epoch(self):
        """String-encoded nanosecond epoch (OTLP JSON uint64) is correctly parsed."""
        ts_str = self.mapper.parse_timestamp("1700000000000000000")
        assert ts_str.year == 2023 and ts_str.month == 11

    def test_none_returns_now(self):
        assert self.mapper.parse_timestamp(None) is not None

    def test_datetime_passthrough(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert self.mapper.parse_timestamp(dt) == dt


# ============================================================================
# Tests: Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for graceful error handling."""

    def setup_method(self):
        self.mapper = ADKOtelSessionMapper()

    def test_malformed_llm_request_json(self):
        """Malformed JSON in llm_request does not crash the mapper."""
        spans = [
            make_span(
                span_id="agent-1",
                name="invoke_agent x",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "x"},
            ),
            make_span(
                span_id="callllm-1",
                parent_span_id="agent-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": "not-valid-json{{{",
                    "gcp.vertex.agent.llm_response": "also-invalid",
                },
            ),
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        assert session.traces == []

    def test_missing_attributes_key(self):
        """Span without attributes key does not crash."""
        spans = [{"trace_id": "t1", "span_id": "s1", "name": "test"}]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        assert session.traces == [] or session.traces[0].spans == []

    def test_tool_call_args_as_dict(self):
        """tool_call_args provided as dict (not JSON string) is handled."""
        spans = [
            make_span(
                name="execute_tool calc",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "calc",
                    "gen_ai.tool.call.id": "c1",
                    "gcp.vertex.agent.tool_call_args": {"x": 42},
                    "gcp.vertex.agent.tool_response": "result",
                },
            )
        ]
        session = self.mapper.map_to_session(spans, SESSION_ID)
        assert session.traces[0].spans[0].tool_call.arguments == {"x": 42}


# ============================================================================
# Tests: Multi-Agent Trace Splitting
# ============================================================================


class TestMultiAgentSplitting:
    """Tests that multi-agent traces are kept as a single Trace (scoping handled by TraceExtractor)."""

    def setup_method(self):
        self.mapper = ADKOtelSessionMapper()

    def test_coordinator_specialist_produces_single_trace(self):
        """Two invoke_agent spans in one OTel trace produce a single Trace with both agents."""
        spans = [
            # Coordinator agent
            make_span(
                span_id="coordinator",
                name="invoke_agent coordinator",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "coordinator"},
                start_time=1700000001000000000,
                end_time=1700000009000000000,
            ),
            make_span(
                span_id="coord-callllm",
                parent_span_id="coordinator",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(
                        user_text="Book a flight",
                        system_instruction="You coordinate tasks.",
                        tools=[{"name": "delegate", "description": "Delegate to specialist"}],
                    ),
                    "gcp.vertex.agent.llm_response": make_llm_response_tool_call(
                        name="delegate", args={"task": "book"}, call_id="fc-coord"
                    ),
                },
                start_time=1700000002000000000,
                end_time=1700000003000000000,
            ),
            # Specialist agent (nested under coordinator's call_llm)
            make_span(
                span_id="specialist",
                parent_span_id="coord-callllm",
                name="invoke_agent specialist",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "specialist"},
                start_time=1700000004000000000,
                end_time=1700000008000000000,
            ),
            make_span(
                span_id="spec-callllm",
                parent_span_id="specialist",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(
                        user_text="Book a flight",
                        system_instruction="You book flights.",
                        tools=[{"name": "book_flight", "description": "Book a flight"}],
                    ),
                    "gcp.vertex.agent.llm_response": make_llm_response_text("Booked seat 4A."),
                },
                start_time=1700000005000000000,
                end_time=1700000006000000000,
            ),
            make_span(
                span_id="spec-tool",
                parent_span_id="spec-callllm",
                name="execute_tool book_flight",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "book_flight",
                    "gen_ai.tool.call.id": "fc-spec",
                    "gcp.vertex.agent.tool_call_args": '{"destination": "NYC"}',
                    "gcp.vertex.agent.tool_response": '{"result": "SPECIALIST_TOOL_OUTPUT"}',
                },
                start_time=1700000006000000000,
                end_time=1700000007000000000,
            ),
        ]

        session = self.mapper.map_to_session(spans, SESSION_ID)

        # Single trace containing both agents
        assert len(session.traces) == 1
        trace = session.traces[0]

        # Both agent spans present
        agent_spans = [s for s in trace.spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 2

        # Each agent retains its own available_tools
        coord_agent = next(s for s in agent_spans if s.metadata.get("agent_name") == "coordinator")
        spec_agent = next(s for s in agent_spans if s.metadata.get("agent_name") == "specialist")
        assert coord_agent.available_tools[0].name == "delegate"
        assert spec_agent.available_tools[0].name == "book_flight"

        # Tool execution span present in the trace
        tool_spans = [s for s in trace.spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        assert tool_spans[0].tool_call.name == "book_flight"

    def test_unclaimed_spans_in_single_trace(self):
        """Spans not nested under any invoke_agent remain in the single trace."""
        spans = [
            # Orphan tool span — not parented to either agent
            make_span(
                span_id="orphan-tool",
                name="execute_tool audit",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "audit",
                    "gen_ai.tool.call.id": "fc-audit",
                    "gcp.vertex.agent.tool_call_args": "{}",
                    "gcp.vertex.agent.tool_response": "audited",
                },
                start_time=1700000001000000000,
                end_time=1700000002000000000,
            ),
            # Two minimal agent spans (each needs a call_llm child to be non-empty)
            make_span(
                span_id="agent-1",
                name="invoke_agent alpha",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "alpha"},
                start_time=1700000003000000000,
                end_time=1700000005000000000,
            ),
            make_span(
                span_id="callllm-1",
                parent_span_id="agent-1",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(user_text="Hello"),
                    "gcp.vertex.agent.llm_response": make_llm_response_text("Hi"),
                },
                start_time=1700000004000000000,
                end_time=1700000005000000000,
            ),
            make_span(
                span_id="agent-2",
                name="invoke_agent beta",
                attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "beta"},
                start_time=1700000006000000000,
                end_time=1700000008000000000,
            ),
            make_span(
                span_id="callllm-2",
                parent_span_id="agent-2",
                name="call_llm",
                attributes={
                    "gcp.vertex.agent.llm_request": make_llm_request(user_text="Bye"),
                    "gcp.vertex.agent.llm_response": make_llm_response_text("Goodbye"),
                },
                start_time=1700000007000000000,
                end_time=1700000008000000000,
            ),
        ]

        session = self.mapper.map_to_session(spans, SESSION_ID)

        # Single trace with everything
        assert len(session.traces) == 1
        trace = session.traces[0]

        # Both agents and the orphan tool are in the same trace
        agent_spans = [s for s in trace.spans if isinstance(s, AgentInvocationSpan)]
        tool_spans = [s for s in trace.spans if isinstance(s, ToolExecutionSpan)]
        assert len(agent_spans) == 2
        assert len(tool_spans) == 1
        assert tool_spans[0].tool_call.name == "audit"


# ============================================================================
# Tests: Scope Filtering
# ============================================================================


class TestScopeFiltering:
    """Tests for instrumentation scope filtering."""

    def setup_method(self):
        self.mapper = ADKOtelSessionMapper()

    def test_foreign_scope_spans_are_dropped(self):
        """Spans from non-ADK instrumentation scopes are excluded from the session.

        Uses a foreign-scope execute_tool span that would convert unconditionally
        if not filtered, ensuring the scope check is the only barrier.
        """
        spans = [
            # ADK span — should be kept
            make_span(
                span_id="tool-1",
                name="execute_tool calculator",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "calculator",
                    "gen_ai.tool.call.id": "c1",
                    "gcp.vertex.agent.tool_call_args": '{"x": 1}',
                    "gcp.vertex.agent.tool_response": "2",
                },
            ),
            # Foreign scope span with full tool attributes — would convert if not filtered
            {
                "trace_id": "trace-1",
                "span_id": "foreign-1",
                "parent_span_id": None,
                "name": "execute_tool foreign_tool",
                "start_time": 1700000000000000000,
                "end_time": 1700000001000000000,
                "attributes": {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "foreign_tool",
                    "gen_ai.tool.call.id": "c2",
                    "gcp.vertex.agent.tool_call_args": '{"q": "test"}',
                    "gcp.vertex.agent.tool_response": "foreign result",
                },
                "scope": {"name": "opentelemetry.instrumentation.vertexai", "version": "1.0.0"},
                "status": {"code": "UNSET"},
                "span_events": [],
            },
        ]

        session = self.mapper.map_to_session(spans, SESSION_ID)

        # Only the ADK tool span should survive
        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1
        assert isinstance(session.traces[0].spans[0], ToolExecutionSpan)
        assert session.traces[0].spans[0].tool_call.name == "calculator"
