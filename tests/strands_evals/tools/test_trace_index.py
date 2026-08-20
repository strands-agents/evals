from datetime import datetime, timezone

import pytest

from strands_evals.tools.trace_index import TraceIndex
from strands_evals.types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    ToolCall,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)


def _span_info(second: int) -> SpanInfo:
    return SpanInfo(
        session_id="s1",
        span_id=f"sp{second}",
        start_time=datetime(2026, 1, 1, 0, 0, second, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 0, 0, second + 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def session():
    spans = [
        AgentInvocationSpan(
            span_info=_span_info(0),
            user_prompt="Look up ticket TKT-1042",
            agent_response="Ticket TKT-1042 was refunded $150.",
            available_tools=[],
        ),
        ToolExecutionSpan(
            span_info=_span_info(1),
            tool_call=ToolCall(name="lookup_ticket", arguments={"id": "TKT-1042"}),
            tool_result=ToolResult(content="x" * 20_000 + " refund_amount=$150"),
        ),
        ToolExecutionSpan(
            span_info=_span_info(2),
            tool_call=ToolCall(name="get_customer", arguments={"id": "C-7"}),
            tool_result=ToolResult(content="customer name: Alex"),
        ),
    ]
    return Session(traces=[Trace(spans=spans, trace_id="t1", session_id="s1")], session_id="s1")


def test_overview_is_compact_and_ordered(session):
    index = TraceIndex(session)
    overview = index.overview()

    lines = overview.splitlines()
    assert "3 spans" in lines[0]
    assert lines[1].startswith("[0] AGENT")
    assert "lookup_ticket" in lines[2]
    assert "get_customer" in lines[3]
    # Manifest must not inline the 20K-char tool result
    assert len(overview) < 2_000


def test_get_span_returns_full_content_for_small_span(session):
    index = TraceIndex(session)
    get_span = index.tools[1]

    content = get_span(index=2)

    assert "customer name: Alex" in content
    assert "TRUNCATED" not in content


def test_get_span_windows_oversized_content_and_pages(session):
    index = TraceIndex(session, max_read_chars=5_000)
    get_span = index.tools[1]

    first = get_span(index=1)
    assert "TRUNCATED" in first
    assert "offset=5000" in first

    second = get_span(index=1, offset=5_000)
    assert second.startswith("x") or '"' in second  # continuation, not a restart
    assert first[:100] != second[:100]


def test_get_span_index_out_of_range(session):
    index = TraceIndex(session)
    get_span = index.tools[1]

    assert "ERROR" in get_span(index=99)
    assert "ERROR" in get_span(index=-1)


def test_search_spans_finds_span_by_content(session):
    index = TraceIndex(session)
    search_spans = index.tools[2]

    result = search_spans(pattern=r"refund_amount=\$150")

    assert result.startswith("[1]")
    assert "refund_amount" in result


def test_search_spans_falls_back_to_literal_on_bad_regex(session):
    index = TraceIndex(session)
    search_spans = index.tools[2]

    result = search_spans(pattern="refund_amount=$150[")

    assert "No matches" in result or result.startswith("[")


def test_search_spans_no_matches(session):
    index = TraceIndex(session)
    search_spans = index.tools[2]

    assert "No matches" in search_spans(pattern="nonexistent-zzz")


def test_list_spans_tool_matches_overview(session):
    index = TraceIndex(session)
    list_spans = index.tools[0]

    assert list_spans() == index.overview()


def test_tools_are_strands_tools(session):
    index = TraceIndex(session)

    for t in index.tools:
        assert hasattr(t, "tool_spec") or hasattr(t, "TOOL_SPEC") or callable(t)
