"""Tests for CloudWatchSessionMapper — body-format CW log record → Session conversion."""

import json

from strands_evals.mappers.cloudwatch_session_mapper import CloudWatchSessionMapper
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    ToolExecutionSpan,
)
from tests.strands_evals.cloudwatch_helpers import make_assistant_text_message, make_log_record, make_user_message

# --- Helpers for body-format log records ---


def _make_assistant_tool_use_message(tool_name, tool_input, tool_use_id):
    """Build an assistant output message with a toolUse block."""
    return {
        "role": "assistant",
        "content": {
            "message": json.dumps([{"toolUse": {"name": tool_name, "input": tool_input, "toolUseId": tool_use_id}}]),
            "finish_reason": "tool_use",
        },
    }


def _make_tool_result_message(tool_use_id, result_text):
    """Build a tool result input message with double-encoded content."""
    return {
        "role": "tool",
        "content": {
            "content": json.dumps([{"toolResult": {"content": [{"text": result_text}], "toolUseId": tool_use_id}}])
        },
    }


# --- Span conversion (body-based parsing) ---


class TestSpanConversion:
    def setup_method(self):
        self.mapper = CloudWatchSessionMapper()

    def test_single_record_produces_inference_span(self):
        """One log record produces an InferenceSpan with input/output messages."""
        record = make_log_record(
            input_messages=[make_user_message("Hi")],
            output_messages=[make_assistant_text_message("Hello!")],
        )
        session = self.mapper.map_to_session([record], "sess-1")
        spans = session.traces[0].spans
        inference_spans = [s for s in spans if isinstance(s, InferenceSpan)]
        assert len(inference_spans) == 1
        assert inference_spans[0].messages[0].content[0].text == "Hi"
        assert inference_spans[0].messages[1].content[0].text == "Hello!"

    def test_record_with_tool_use_and_result(self):
        """toolUse in output + toolResult in next record's input → ToolExecutionSpan."""
        record1 = make_log_record(
            trace_id="t1",
            span_id="s1",
            input_messages=[make_user_message("Calculate 6*7")],
            output_messages=[_make_assistant_tool_use_message("calculator", {"expr": "6*7"}, "tu-1")],
            time_nano=1000,
        )
        record2 = make_log_record(
            trace_id="t1",
            span_id="s2",
            input_messages=[
                make_user_message("Calculate 6*7"),
                _make_tool_result_message("tu-1", "42"),
            ],
            output_messages=[make_assistant_text_message("The answer is 42.")],
            time_nano=2000,
        )
        session = self.mapper.map_to_session([record1, record2], "sess-1")
        tool_spans = [s for s in session.traces[0].spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        assert tool_spans[0].tool_call.name == "calculator"
        assert tool_spans[0].tool_call.arguments == {"expr": "6*7"}
        assert tool_spans[0].tool_result.content == "42"

    def test_agent_invocation_from_trace(self):
        """User prompt from first record, response from last → AgentInvocationSpan."""
        record1 = make_log_record(
            trace_id="t1",
            span_id="s1",
            input_messages=[make_user_message("Tell me a joke")],
            output_messages=[make_assistant_text_message("Why did the chicken cross the road?")],
            time_nano=1000,
        )
        session = self.mapper.map_to_session([record1], "sess-1")
        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        assert agent_spans[0].user_prompt == "Tell me a joke"
        assert agent_spans[0].agent_response == "Why did the chicken cross the road?"

    def test_agent_invocation_extracts_tools(self):
        """available_tools populated from tool call names in the trace."""
        record1 = make_log_record(
            trace_id="t1",
            span_id="s1",
            input_messages=[make_user_message("Search for X")],
            output_messages=[_make_assistant_tool_use_message("web_search", {"q": "X"}, "tu-1")],
            time_nano=1000,
        )
        record2 = make_log_record(
            trace_id="t1",
            span_id="s2",
            input_messages=[make_user_message("Search for X"), _make_tool_result_message("tu-1", "found X")],
            output_messages=[make_assistant_text_message("Here's what I found about X.")],
            time_nano=2000,
        )
        session = self.mapper.map_to_session([record1, record2], "sess-1")
        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        tool_names = [t.name for t in agent_spans[0].available_tools]
        assert "web_search" in tool_names

    def test_double_encoded_content_parsed(self):
        """Content field is a JSON string that must be parsed to get content blocks."""
        record = make_log_record(
            input_messages=[make_user_message("test double encoding")],
            output_messages=[make_assistant_text_message("parsed correctly")],
        )
        session = self.mapper.map_to_session([record], "sess-1")
        inference_spans = [s for s in session.traces[0].spans if isinstance(s, InferenceSpan)]
        assert inference_spans[0].messages[0].content[0].text == "test double encoding"
        assert inference_spans[0].messages[1].content[0].text == "parsed correctly"

    def test_tool_call_matched_to_result_by_id(self):
        """toolUseId matching works across records in the same trace."""
        record1 = make_log_record(
            trace_id="t1",
            span_id="s1",
            input_messages=[make_user_message("Do two things")],
            output_messages=[
                _make_assistant_tool_use_message("tool_a", {"x": 1}, "tu-a"),
            ],
            time_nano=1000,
        )
        record2 = make_log_record(
            trace_id="t1",
            span_id="s2",
            input_messages=[
                make_user_message("Do two things"),
                _make_tool_result_message("tu-a", "result-a"),
            ],
            output_messages=[
                _make_assistant_tool_use_message("tool_b", {"y": 2}, "tu-b"),
            ],
            time_nano=2000,
        )
        record3 = make_log_record(
            trace_id="t1",
            span_id="s3",
            input_messages=[
                make_user_message("Do two things"),
                _make_tool_result_message("tu-a", "result-a"),
                _make_tool_result_message("tu-b", "result-b"),
            ],
            output_messages=[make_assistant_text_message("Both done.")],
            time_nano=3000,
        )
        session = self.mapper.map_to_session([record1, record2, record3], "sess-1")
        tool_spans = [s for s in session.traces[0].spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 2
        tool_span_by_name = {ts.tool_call.name: ts for ts in tool_spans}
        assert tool_span_by_name["tool_a"].tool_result.content == "result-a"
        assert tool_span_by_name["tool_b"].tool_result.content == "result-b"


# --- Session building ---


class TestSessionBuilding:
    def setup_method(self):
        self.mapper = CloudWatchSessionMapper()

    def test_multiple_records_grouped_by_trace_id(self):
        """Records with different traceIds become separate Trace objects."""
        records = [
            make_log_record(
                trace_id="t1",
                input_messages=[make_user_message("q1")],
                output_messages=[make_assistant_text_message("a1")],
            ),
            make_log_record(
                trace_id="t2",
                input_messages=[make_user_message("q2")],
                output_messages=[make_assistant_text_message("a2")],
            ),
        ]
        session = self.mapper.map_to_session(records, "sess-1")
        assert session.session_id == "sess-1"
        assert len(session.traces) == 2
        trace_ids = {t.trace_id for t in session.traces}
        assert trace_ids == {"t1", "t2"}

    def test_multi_step_agent_loop(self):
        """user→LLM→tool→LLM→response produces InferenceSpan + ToolExecutionSpan + AgentInvocationSpan."""
        record1 = make_log_record(
            trace_id="t1",
            span_id="s1",
            input_messages=[make_user_message("What is 6*7?")],
            output_messages=[_make_assistant_tool_use_message("calculator", {"expr": "6*7"}, "tu-1")],
            time_nano=1000,
        )
        record2 = make_log_record(
            trace_id="t1",
            span_id="s2",
            input_messages=[
                make_user_message("What is 6*7?"),
                _make_tool_result_message("tu-1", "42"),
            ],
            output_messages=[make_assistant_text_message("The answer is 42.")],
            time_nano=2000,
        )
        session = self.mapper.map_to_session([record1, record2], "sess-1")
        assert len(session.traces) == 1
        spans = session.traces[0].spans
        span_types = [type(s).__name__ for s in spans]
        assert "InferenceSpan" in span_types
        assert "ToolExecutionSpan" in span_types
        assert "AgentInvocationSpan" in span_types

    def test_empty_records_list(self):
        session = self.mapper.map_to_session([], "sess-1")
        assert session.session_id == "sess-1"
        assert session.traces == []

    def test_record_with_no_body_skipped(self):
        """Malformed records without body don't crash."""
        records = [
            {"traceId": "t1", "spanId": "s1", "timeUnixNano": 1000},
            make_log_record(
                trace_id="t1",
                input_messages=[make_user_message("Hi")],
                output_messages=[make_assistant_text_message("Hello!")],
                time_nano=2000,
            ),
        ]
        session = self.mapper.map_to_session(records, "sess-1")
        assert len(session.traces) == 1
        assert len(session.traces[0].spans) > 0
