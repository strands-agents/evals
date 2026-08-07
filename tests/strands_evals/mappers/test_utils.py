"""Tests for mapper utility functions."""

from datetime import datetime, timezone

from strands_evals.mappers import (
    CloudWatchSessionMapper,
    GenericGenAISessionMapper,
    LangChainOtelSessionMapper,
    OpenInferenceSessionMapper,
    StrandsInMemorySessionMapper,
    detect_otel_mapper,
    get_scope_name,
    readable_spans_to_dicts,
)
from strands_evals.mappers.utils import bridge_parent_gaps, join_tool_result_content
from strands_evals.types.trace import AgentInvocationSpan, SpanInfo, ToolCall, ToolExecutionSpan, ToolResult


class TestJoinToolResultContent:
    def test_empty_list(self):
        assert join_tool_result_content([]) == ""

    def test_none_input(self):
        assert join_tool_result_content(None) == ""

    def test_non_list_input(self):
        assert join_tool_result_content("raw string") == "raw string"

    def test_single_text_block(self):
        assert join_tool_result_content([{"text": "hello"}]) == "hello"

    def test_multi_text_blocks(self):
        assert join_tool_result_content([{"text": "a"}, {"text": "b"}]) == "a\nb"

    def test_json_block(self):
        assert join_tool_result_content([{"json": {"k": 1}}]) == '{"k": 1}'

    def test_json_block_sort_keys(self):
        assert join_tool_result_content([{"json": {"b": 2, "a": 1}}]) == '{"a": 1, "b": 2}'

    def test_image_placeholder(self):
        assert join_tool_result_content([{"image": {}}]) == "[image]"

    def test_document_placeholder(self):
        assert join_tool_result_content([{"document": {}}]) == "[document]"

    def test_video_placeholder(self):
        assert join_tool_result_content([{"video": {}}]) == "[video]"

    def test_unknown_key_silently_dropped(self):
        assert join_tool_result_content([{"unknown_type": "value"}]) == ""

    def test_text_none_value_no_crash(self):
        assert join_tool_result_content([{"text": None}]) == ""

    def test_text_none_value_with_sibling(self):
        assert join_tool_result_content([{"text": None}, {"text": "ok"}]) == "ok"

    def test_text_non_str_coerced(self):
        assert join_tool_result_content([{"text": 123}]) == "123"

    def test_non_dict_block(self):
        assert join_tool_result_content([42]) == "42"

    def test_mixed_blocks(self):
        result = join_tool_result_content([{"text": "label:"}, {"json": {"v": 1}}, {"image": {}}])
        assert result == 'label:\n{"v": 1}\n[image]'

    def test_empty_string_input(self):
        assert join_tool_result_content("") == ""

    def test_integer_zero_input(self):
        # int 0 is not None and not an empty list, so it coerces to "0"
        assert join_tool_result_content(0) == "0"

    def test_false_input(self):
        # False is not None and not an empty list, so it coerces to "False"
        assert join_tool_result_content(False) == "False"

    def test_block_with_multiple_keys_prefers_text(self):
        # When a block has multiple keys, 'text' wins (first match in if/elif chain)
        result = join_tool_result_content([{"text": "hello", "json": {"x": 1}}])
        assert result == "hello"

    def test_json_block_none_value(self):
        # json.dumps(None) == 'null', included in output
        assert join_tool_result_content([{"json": None}]) == "null"

    def test_json_block_inf_serialized(self):
        import math

        # Python's json.dumps may serialize inf as "Infinity" or raise; either way no crash
        result = join_tool_result_content([{"json": math.inf}, {"text": "after"}])
        assert "after" in result or result in ("Infinity\nafter", "after")

    def test_text_empty_string_block_filtered(self):
        # An empty-string text block is silently filtered by the join guard
        assert join_tool_result_content([{"text": ""}]) == ""

    def test_text_empty_string_block_with_sibling(self):
        assert join_tool_result_content([{"text": ""}, {"text": "ok"}]) == "ok"

    def test_bare_dict_not_in_list(self):
        # A bare dict (not wrapped in a list) is coerced to str
        result = join_tool_result_content({"text": "hello"})
        assert result == str({"text": "hello"})


def make_span_dict(scope_name="test.scope", attributes=None, span_events=None):
    """Build a minimal span dict for testing."""
    return {
        "trace_id": "trace-1",
        "span_id": "span-1",
        "scope": {"name": scope_name, "version": "0.1.0"},
        "attributes": attributes or {},
        "span_events": span_events or [],
    }


class TestGetScopeName:
    def test_extracts_from_scope_dict(self):
        """Extracts scope name from scope.name field."""
        span = make_span_dict(scope_name="test.scope")
        assert get_scope_name(span) == "test.scope"

    def test_extracts_from_attributes_event_name(self):
        """Falls back to attributes.event.name if scope.name is empty."""
        span = {
            "scope": {"name": ""},
            "attributes": {"event.name": "fallback.scope"},
        }
        assert get_scope_name(span) == "fallback.scope"

    def test_returns_empty_for_missing_scope(self):
        """Returns empty string if no scope found."""
        span = {"attributes": {}}
        assert get_scope_name(span) == ""

    def test_handles_non_dict_input(self):
        """Returns empty string for non-dict input without scope."""

        class FakeSpan:
            pass

        span = FakeSpan()
        assert get_scope_name(span) == ""


class TestDetectOtelMapper:
    def test_empty_spans_returns_strands_mapper(self):
        """Empty spans list returns StrandsInMemorySessionMapper."""
        mapper = detect_otel_mapper([])
        assert isinstance(mapper, StrandsInMemorySessionMapper)

    def test_detects_langchain_traceloop_scope(self):
        """Detects LangChainOtelSessionMapper for traceloop scope."""
        spans = [make_span_dict(scope_name="opentelemetry.instrumentation.langchain")]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, LangChainOtelSessionMapper)

    def test_detects_openinference_scope(self):
        """Detects OpenInferenceSessionMapper for openinference scope."""
        spans = [make_span_dict(scope_name="openinference.instrumentation.langchain")]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, OpenInferenceSessionMapper)

    def test_detects_strands_cloudwatch_format(self):
        """Detects CloudWatchSessionMapper for strands scope with body format."""
        spans = [
            make_span_dict(
                scope_name="strands.telemetry.tracer",
                span_events=[
                    {
                        "body": {
                            "input": {"messages": []},
                            "output": {"messages": []},
                        }
                    }
                ],
            )
        ]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, CloudWatchSessionMapper)

    def test_detects_strands_cloudwatch_split_format(self):
        """CloudWatch split format: Strands scope entry has no body, a later
        entry carries the body — must still route to CloudWatchSessionMapper."""
        spans = [
            # Entry 1: metadata — Strands scope, no body
            make_span_dict(
                scope_name="strands.telemetry.tracer",
                attributes={"gen_ai.operation.name": "invoke_agent"},
            ),
            # Entry 2: log event — body with input/output, no scope
            {
                "trace_id": "trace-1",
                "span_id": "span-1",
                "body": {"input": {"messages": []}, "output": {"messages": []}},
            },
        ]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, CloudWatchSessionMapper)

    def test_detects_strands_in_memory_format(self):
        """Detects StrandsInMemorySessionMapper for strands scope without body format."""
        spans = [
            make_span_dict(
                scope_name="strands.telemetry.tracer",
                attributes={"gen_ai.operation.name": "chat"},
            )
        ]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, StrandsInMemorySessionMapper)

    def test_defaults_to_strands_mapper_for_unknown_scope(self):
        """Defaults to StrandsInMemorySessionMapper for unknown scope without gen_ai attrs."""
        spans = [make_span_dict(scope_name="unknown.scope")]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, StrandsInMemorySessionMapper)

    def test_skips_spans_without_scope(self):
        """Skips spans without scope and continues detection."""
        spans = [
            {"trace_id": "t1", "span_id": "s1"},  # No scope
            make_span_dict(scope_name="openinference.instrumentation.langchain"),
        ]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, OpenInferenceSessionMapper)

    def test_detects_smolagents_openinference_scope(self):
        """Detects OpenInferenceSessionMapper for smolagents openinference scope."""
        spans = [make_span_dict(scope_name="openinference.instrumentation.smolagents")]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, OpenInferenceSessionMapper)

    def test_non_dict_spans_fall_through_to_strands_mapper(self):
        """Non-dict spans (no scope/body/gen_ai attrs to match) fall through to default StrandsInMemorySessionMapper."""
        from unittest.mock import MagicMock

        # Simulates the case where raw ReadableSpan objects are passed without
        # calling readable_spans_to_dicts() first. detect_otel_mapper doesn't
        # crash but falls through to the default mapper since get_scope_name()
        # handles non-dict inputs via hasattr checks.
        mock_span = MagicMock()
        mock_span.instrumentation_scope.name = ""
        mapper = detect_otel_mapper([mock_span])
        assert isinstance(mapper, StrandsInMemorySessionMapper)

    def test_unrecognized_scope_with_gen_ai_attrs_routes_to_generic_mapper(self):
        """Dict spans with unrecognized scope but gen_ai.operation.name route to GenericGenAISessionMapper."""
        spans = [
            make_span_dict(
                scope_name="my-custom-tracer",
                attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "calc"},
            )
        ]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, GenericGenAISessionMapper)

    def test_unrecognized_scope_without_gen_ai_attrs_defaults_to_strands(self):
        """Dict spans with unrecognized scope and no gen_ai attrs default to StrandsInMemorySessionMapper."""
        spans = [make_span_dict(scope_name="totally.unknown", attributes={"custom.key": "val"})]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, StrandsInMemorySessionMapper)

    def test_detects_claude_agent_sdk_openinference_scope(self):
        """Detects OpenInferenceSessionMapper for Claude Agent SDK openinference scope."""
        spans = [make_span_dict(scope_name="openinference.instrumentation.claude_agent_sdk")]
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, OpenInferenceSessionMapper)


class TestReadableSpansToDicts:
    def test_converts_readable_spans(self):
        """Converts ReadableSpan objects to dict format."""
        from unittest.mock import MagicMock

        # Create mock ReadableSpan
        mock_span = MagicMock()
        mock_span.context.trace_id = 0xABCD1234
        mock_span.context.span_id = 0x5678
        mock_span.parent = None
        mock_span.name = "test-span"
        mock_span.start_time = 1700000000000000000
        mock_span.end_time = 1700000001000000000
        mock_span.attributes = {"key": "value"}
        mock_span.instrumentation_scope.name = "test.scope"
        mock_span.instrumentation_scope.version = "1.0"
        mock_span.status.status_code.name = "OK"
        mock_span.events = []

        result = readable_spans_to_dicts([mock_span])

        assert len(result) == 1
        assert result[0]["trace_id"] == "000000000000000000000000abcd1234"
        assert result[0]["span_id"] == "0000000000005678"
        assert result[0]["parent_span_id"] is None
        assert result[0]["name"] == "test-span"
        assert result[0]["attributes"] == {"key": "value"}
        assert result[0]["scope"]["name"] == "test.scope"
        assert result[0]["span_events"] == []

    def test_converts_span_with_parent(self):
        """Converts span with parent reference."""
        from unittest.mock import MagicMock

        mock_span = MagicMock()
        mock_span.context.trace_id = 0xABCD
        mock_span.context.span_id = 0x1234
        mock_span.parent.span_id = 0x5678
        mock_span.name = "child-span"
        mock_span.start_time = 0
        mock_span.end_time = 0
        mock_span.attributes = {}
        mock_span.instrumentation_scope.name = "test"
        mock_span.instrumentation_scope.version = "1.0"
        mock_span.status.status_code.name = "OK"
        mock_span.events = []

        result = readable_spans_to_dicts([mock_span])

        assert result[0]["parent_span_id"] == "0000000000005678"

    def test_converts_span_events(self):
        """Converts span events to dict format."""
        from unittest.mock import MagicMock

        mock_event = MagicMock()
        mock_event.name = "test.event"
        mock_event.timestamp = 1700000000500000000
        mock_event.attributes = {"event_key": "event_value"}

        mock_span = MagicMock()
        mock_span.context.trace_id = 0xABCD
        mock_span.context.span_id = 0x1234
        mock_span.parent = None
        mock_span.name = "span"
        mock_span.start_time = 0
        mock_span.end_time = 0
        mock_span.attributes = {}
        mock_span.instrumentation_scope.name = "test"
        mock_span.instrumentation_scope.version = "1.0"
        mock_span.status.status_code.name = "OK"
        mock_span.events = [mock_event]

        result = readable_spans_to_dicts([mock_span])

        assert len(result[0]["span_events"]) == 1
        assert result[0]["span_events"][0]["event_name"] == "test.event"
        assert result[0]["span_events"][0]["attributes"] == {"event_key": "event_value"}

    def test_handles_empty_list(self):
        """Handles empty span list."""
        result = readable_spans_to_dicts([])
        assert result == []


def _span_info(span_id: str, parent_span_id: str | None = None) -> SpanInfo:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SpanInfo(session_id="s1", span_id=span_id, parent_span_id=parent_span_id, start_time=now, end_time=now)


class TestBridgeParentGaps:
    def test_bridges_to_converted_ancestor(self):
        """Tool pointing to unconverted span gets reparented to the nearest converted ancestor."""
        agent = AgentInvocationSpan(
            span_info=_span_info("agent", None), user_prompt="hi", agent_response="hey", available_tools=[]
        )
        tool = ToolExecutionSpan(
            span_info=_span_info("tool", parent_span_id="call_llm"),
            tool_call=ToolCall(name="x", arguments={}),
            tool_result=ToolResult(content="y"),
        )
        # call_llm is unconverted but its parent is the converted agent
        raw_parent_map = {"agent": None, "call_llm": "agent", "tool": "call_llm"}

        bridge_parent_gaps([agent, tool], raw_parent_map)

        assert tool.span_info.parent_span_id == "agent"

    def test_sets_none_when_no_converted_ancestor(self):
        """Tool whose chain never reaches a converted span gets parent_span_id=None."""
        tool = ToolExecutionSpan(
            span_info=_span_info("tool", parent_span_id="orphan"),
            tool_call=ToolCall(name="x", arguments={}),
            tool_result=ToolResult(content="y"),
        )
        raw_parent_map = {"orphan": "also_gone", "also_gone": None, "tool": "orphan"}

        bridge_parent_gaps([tool], raw_parent_map)

        assert tool.span_info.parent_span_id is None
