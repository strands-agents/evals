from datetime import datetime, timezone

import pytest

from strands_evals.tools.trace_index import TraceIndex
from strands_evals.types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)


def _span_info(second: int, tz: timezone | None = timezone.utc) -> SpanInfo:
    return SpanInfo(
        session_id="s1",
        span_id=f"sp{second}",
        start_time=datetime(2026, 1, 1, 0, 0, second, tzinfo=tz),
        end_time=datetime(2026, 1, 1, 0, 0, second + 1, tzinfo=tz),
    )


@pytest.fixture
def session():
    spans = [
        AgentInvocationSpan(
            span_info=_span_info(0),
            user_prompt="Look up ticket TKT-1042",
            agent_response="Ticket TKT-1042 was refunded $150.",
            system_prompt="You are a support agent. Never issue a refund over $500.",
            available_tools=[ToolConfig(name="lookup_ticket", description="Look up a ticket by id")],
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
    # Header + "Showing spans" banner precede the span lines.
    span_lines = [ln for ln in lines if ln.startswith("[")]
    assert span_lines[0].startswith("[0] AGENT")
    assert "lookup_ticket" in span_lines[1]
    assert "get_customer" in span_lines[2]
    # Overview must not inline the 20K-char tool result.
    assert len(overview) < 2_000
    # Result size of the big span is surfaced exactly.
    assert "20019 chars" in span_lines[1]


def test_overview_header_tells_the_model_previews_are_truncated(session):
    index = TraceIndex(session)
    header = index.overview().splitlines()[0]
    assert "truncated" in header.lower()
    assert "verify" in header.lower()
    assert "8000" in header  # max_read_chars surfaced so N-chars counts are actionable


def test_overview_pages_when_it_exceeds_max_read_chars(session):
    # Tiny budget forces paging across the three spans.
    index = TraceIndex(session, max_read_chars=200)
    first = index.overview()
    assert "Showing spans 0-" in first
    assert "MORE:" in first

    # Follow the offset the tool reported, not a hand-computed one.
    next_offset = int(first.split("offset=")[1].split("]")[0])
    assert next_offset > 0
    second = index.overview(offset=next_offset)
    assert f"Showing spans {next_offset}-" in second

    # Every span shows up exactly once across the pages.
    seen = set()
    offset, guard = 0, 0
    while True:
        page = index.overview(offset=offset)
        for ln in page.splitlines():
            idx = ln[1 : ln.index("]")] if ln.startswith("[") and "]" in ln else ""
            if idx.isdigit():
                seen.add(int(idx))
        if "MORE:" not in page:
            break
        offset = int(page.split("offset=")[1].split("]")[0])
        guard += 1
        assert guard < 10, "paging did not terminate"
    assert seen == {0, 1, 2}


def test_overview_rejects_out_of_range_offset(session):
    index = TraceIndex(session)
    assert "ERROR" in index.overview(offset=99)
    assert "ERROR" in index.overview(offset=-1)


def test_list_spans_tool_is_paged(session):
    index = TraceIndex(session, max_read_chars=200)
    list_spans = index.tools[0]
    assert list_spans(offset=0) == index.overview(0)
    assert "MORE:" in list_spans(offset=0)


def test_max_read_chars_must_be_positive(session):
    with pytest.raises(ValueError, match="max_read_chars"):
        TraceIndex(session, max_read_chars=0)
    with pytest.raises(ValueError, match="max_read_chars"):
        TraceIndex(session, max_read_chars=-5)


def test_get_span_returns_full_content_for_small_span(session):
    index = TraceIndex(session)
    get_span = index.tools[1]

    content = get_span(index=2)

    assert "customer name: Alex" in content
    assert "TRUNCATED" not in content


def test_get_span_exposes_system_prompt_and_tools(session):
    index = TraceIndex(session)
    get_span = index.tools[1]

    content = get_span(index=0)

    assert "Never issue a refund over $500" in content
    assert "lookup_ticket" in content


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


def test_get_span_rejects_negative_offset(session):
    index = TraceIndex(session)
    get_span = index.tools[1]
    out = get_span(index=2, offset=-10)
    assert "ERROR" in out
    assert "-10" not in out.split("offset=")[-1] if "offset=" in out else True


def test_search_finds_literal_the_judge_would_quote(session):
    """The flagship case: a literal '$150' the judge copies from the overview must
    match without the judge knowing to escape regex metacharacters."""
    index = TraceIndex(session)
    search_spans = index.tools[2]

    result = search_spans(pattern="$150")

    assert result.startswith("[0]") or "\n[0]" in result  # agent_response has "$150"
    assert "[1]" in result  # tool result has "refund_amount=$150"
    assert "No matches" not in result


def test_search_literal_does_not_treat_pattern_as_regex(session):
    index = TraceIndex(session)
    search_spans = index.tools[2]
    # refund_amount=$150 as a literal matches the tool result exactly.
    result = search_spans(pattern="refund_amount=$150")
    assert result.startswith("[1]")


def test_search_regex_opt_in(session):
    index = TraceIndex(session)
    search_spans = index.tools[2]
    result = search_spans(pattern=r"refund_amount=\$\d+", is_regex=True)
    assert "[1]" in result


def test_search_invalid_regex_reports_error_not_silent_miss(session):
    index = TraceIndex(session)
    search_spans = index.tools[2]
    result = search_spans(pattern="refund[", is_regex=True)
    assert "ERROR" in result
    assert "is_regex=False" in result  # actionable hint


def test_search_covers_system_prompt(session):
    index = TraceIndex(session)
    search_spans = index.tools[2]
    result = search_spans(pattern="Never issue a refund")
    assert result.startswith("[0]")


def test_search_does_not_false_positive_on_serialization_artifacts(session):
    """'error' must not match the JSON key that every successful tool span carries."""
    index = TraceIndex(session)
    search_spans = index.tools[2]
    assert "No matches" in search_spans(pattern="error")


def test_search_reports_per_span_match_count(session):
    spans = [
        ToolExecutionSpan(
            span_info=_span_info(0),
            tool_call=ToolCall(name="t", arguments={}),
            tool_result=ToolResult(content="foo foo foo bar"),
        ),
    ]
    sess = Session(traces=[Trace(spans=spans, trace_id="t1", session_id="s1")], session_id="s1")
    search_spans = TraceIndex(sess).tools[2]
    result = search_spans(pattern="foo")
    assert "3 matches" in result


def test_search_signals_truncation_at_max_matches(session):
    # Build many matching spans, cap below the total.
    spans = [
        ToolExecutionSpan(
            span_info=_span_info(i),
            tool_call=ToolCall(name="t", arguments={}),
            tool_result=ToolResult(content="needle here"),
        )
        for i in range(5)
    ]
    sess = Session(traces=[Trace(spans=spans, trace_id="t1", session_id="s1")], session_id="s1")
    search_spans = TraceIndex(sess).tools[2]
    result = search_spans(pattern="needle", max_matches=2)
    assert "stopped at 2" in result
    assert result.count("[") >= 3  # 2 hits + the truncation marker line


def test_search_no_matches(session):
    index = TraceIndex(session)
    search_spans = index.tools[2]

    assert "No matches" in search_spans(pattern="nonexistent-zzz")


def test_overview_flags_tool_errors(session):
    spans = [
        ToolExecutionSpan(
            span_info=_span_info(0),
            tool_call=ToolCall(name="fetch", arguments={}),
            tool_result=ToolResult(content="", error="ConnectionError: timed out"),
        ),
    ]
    sess = Session(traces=[Trace(spans=spans, trace_id="t1", session_id="s1")], session_id="s1")
    overview = TraceIndex(sess).overview()
    assert "[ERROR]" in overview
    assert "ConnectionError" in overview


def test_flatten_tolerates_mixed_naive_and_aware_timestamps():
    """Different mappers can emit naive and aware start_time; construction must not crash."""
    spans = [
        ToolExecutionSpan(
            span_info=_span_info(1, tz=None),  # naive
            tool_call=ToolCall(name="a", arguments={}),
            tool_result=ToolResult(content="first"),
        ),
        ToolExecutionSpan(
            span_info=_span_info(0, tz=timezone.utc),  # aware, earlier
            tool_call=ToolCall(name="b", arguments={}),
            tool_result=ToolResult(content="second"),
        ),
    ]
    sess = Session(traces=[Trace(spans=spans, trace_id="t1", session_id="s1")], session_id="s1")
    index = TraceIndex(sess)  # must not raise TypeError
    overview = index.overview()
    # Sorted by normalized time: the aware second==0 span comes first.
    span_lines = [ln for ln in overview.splitlines() if ln.startswith("[")]
    assert "TOOL b" in span_lines[0]
    assert "TOOL a" in span_lines[1]


def test_list_spans_tool_matches_overview(session):
    index = TraceIndex(session)
    list_spans = index.tools[0]

    assert list_spans() == index.overview()


def test_for_judge_returns_overview_block_and_tools(session):
    index = TraceIndex(session)
    prompt_section, tools = index.for_judge()

    assert prompt_section.startswith("<TraceOverview>")
    assert prompt_section.endswith("</TraceOverview>")
    assert index.overview() in prompt_section
    assert tools is index.tools


def test_tools_are_strands_tools(session):
    index = TraceIndex(session)

    for t in index.tools:
        assert hasattr(t, "tool_spec") or hasattr(t, "TOOL_SPEC") or callable(t)
