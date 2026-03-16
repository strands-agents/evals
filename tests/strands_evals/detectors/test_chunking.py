"""Tests for chunking utilities."""

from datetime import datetime

from strands_evals.detectors.chunking import (
    CHARS_PER_TOKEN,
    DEFAULT_MAX_INPUT_TOKENS,
    estimate_tokens,
    merge_chunk_failures,
    split_spans_by_tokens,
    would_exceed_context,
)
from strands_evals.types.detector import FailureItem
from strands_evals.types.trace import (
    InferenceSpan,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolExecutionSpan,
    ToolResult,
    UserMessage,
)


def _make_span_info(span_id: str) -> SpanInfo:
    now = datetime.now()
    return SpanInfo(
        session_id="sess_1",
        span_id=span_id,
        trace_id="trace_1",
        start_time=now,
        end_time=now,
    )


def _make_tool_span(span_id: str, content_size: int = 100) -> ToolExecutionSpan:
    """Create a ToolExecutionSpan with controllable content size."""
    return ToolExecutionSpan(
        span_info=_make_span_info(span_id),
        tool_call=ToolCall(name="test_tool", arguments={"data": "x" * content_size}),
        tool_result=ToolResult(content="y" * content_size),
    )


def _make_inference_span(span_id: str) -> InferenceSpan:
    return InferenceSpan(
        span_info=_make_span_info(span_id),
        messages=[UserMessage(content=[TextContent(text="Hello")])],
    )


# --- estimate_tokens ---


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_short():
    text = "Hello world"
    assert estimate_tokens(text) == len(text) // CHARS_PER_TOKEN


def test_estimate_tokens_long():
    text = "a" * 1000
    assert estimate_tokens(text) == 250


# --- would_exceed_context ---


def test_would_exceed_context_small():
    assert not would_exceed_context("Hello world")


def test_would_exceed_context_large():
    # Create text that exceeds 128K * 0.75 = 96K tokens = 384K chars
    big_text = "x" * 400_000
    assert would_exceed_context(big_text)


def test_would_exceed_context_custom_limit():
    # 100 chars = 25 tokens; limit 30 * 0.75 = 22.5 -> exceeds
    assert would_exceed_context("x" * 100, max_input_tokens=30)


def test_would_exceed_context_just_under():
    # effective limit = 100 * 0.75 = 75 tokens = 300 chars
    assert not would_exceed_context("x" * 299, max_input_tokens=100)


# --- split_spans_by_tokens ---


def test_split_spans_single_chunk():
    """Small spans should stay in one chunk."""
    spans = [_make_tool_span(f"span_{i}", content_size=10) for i in range(3)]
    chunks = split_spans_by_tokens(spans, max_tokens=DEFAULT_MAX_INPUT_TOKENS)
    assert len(chunks) == 1
    assert len(chunks[0]) == 3


def test_split_spans_multiple_chunks():
    """Large spans should be split into multiple chunks."""
    # Each span ~10K chars = ~2500 tokens. With limit of 5000 tokens (effective 3750),
    # we need multiple chunks. But MIN_CHUNK_SIZE=5, so we need enough spans.
    spans = [_make_tool_span(f"span_{i}", content_size=2000) for i in range(10)]
    chunks = split_spans_by_tokens(spans, max_tokens=5000, overlap_spans=1)
    assert len(chunks) > 1
    # All spans should be present across chunks
    all_span_ids = set()
    for chunk in chunks:
        for span in chunk:
            all_span_ids.add(span.span_info.span_id)
    assert len(all_span_ids) == 10


def test_split_spans_overlap():
    """Adjacent chunks should share overlap spans."""
    spans = [_make_tool_span(f"span_{i}", content_size=2000) for i in range(10)]
    chunks = split_spans_by_tokens(spans, max_tokens=5000, overlap_spans=2)
    if len(chunks) > 1:
        # Last 2 spans of chunk 0 should be first 2 of chunk 1
        last_of_first = [s.span_info.span_id for s in chunks[0][-2:]]
        first_of_second = [s.span_info.span_id for s in chunks[1][:2]]
        assert last_of_first == first_of_second


def test_split_spans_no_overlap():
    """With overlap=0, chunks should not share spans."""
    spans = [_make_tool_span(f"span_{i}", content_size=2000) for i in range(10)]
    chunks = split_spans_by_tokens(spans, max_tokens=5000, overlap_spans=0)
    if len(chunks) > 1:
        first_ids = {s.span_info.span_id for s in chunks[0]}
        second_ids = {s.span_info.span_id for s in chunks[1]}
        assert first_ids.isdisjoint(second_ids)


def test_split_spans_empty():
    chunks = split_spans_by_tokens([], max_tokens=1000)
    assert chunks == []


# --- merge_chunk_failures ---


def test_merge_no_overlap():
    """Failures from different spans should be preserved."""
    chunk1 = [
        FailureItem(span_id="span_1", category=["error_a"], confidence=["high"], evidence=["ev_a"]),
    ]
    chunk2 = [
        FailureItem(span_id="span_2", category=["error_b"], confidence=["high"], evidence=["ev_b"]),
    ]
    merged = merge_chunk_failures([chunk1, chunk2])
    assert len(merged) == 2


def test_merge_same_span_different_category():
    """Same span, different categories should be combined."""
    chunk1 = [
        FailureItem(span_id="span_1", category=["error_a"], confidence=["high"], evidence=["ev_a"]),
    ]
    chunk2 = [
        FailureItem(span_id="span_1", category=["error_b"], confidence=["high"], evidence=["ev_b"]),
    ]
    merged = merge_chunk_failures([chunk1, chunk2])
    assert len(merged) == 1
    assert len(merged[0].category) == 2
    assert "error_a" in merged[0].category
    assert "error_b" in merged[0].category


def test_merge_same_span_same_category_keeps_highest():
    """Same span+category: keep the higher confidence."""
    chunk1 = [
        FailureItem(span_id="span_1", category=["error_a"], confidence=["low"], evidence=["weak"]),
    ]
    chunk2 = [
        FailureItem(span_id="span_1", category=["error_a"], confidence=["high"], evidence=["strong"]),
    ]
    merged = merge_chunk_failures([chunk1, chunk2])
    assert len(merged) == 1
    assert merged[0].confidence[0] == "high"
    assert merged[0].evidence[0] == "strong"


def test_merge_same_span_same_category_no_downgrade():
    """Same span+category: lower confidence should not replace higher."""
    chunk1 = [
        FailureItem(span_id="span_1", category=["error_a"], confidence=["high"], evidence=["strong"]),
    ]
    chunk2 = [
        FailureItem(span_id="span_1", category=["error_a"], confidence=["low"], evidence=["weak"]),
    ]
    merged = merge_chunk_failures([chunk1, chunk2])
    assert merged[0].confidence[0] == "high"
    assert merged[0].evidence[0] == "strong"


def test_merge_empty():
    assert merge_chunk_failures([]) == []
    assert merge_chunk_failures([[]]) == []


def test_merge_does_not_mutate_originals():
    """Merging should not modify the input FailureItems."""
    item = FailureItem(span_id="span_1", category=["error_a"], confidence=["high"], evidence=["ev_a"])
    chunk1 = [item]
    chunk2 = [
        FailureItem(span_id="span_1", category=["error_b"], confidence=["high"], evidence=["ev_b"]),
    ]
    merge_chunk_failures([chunk1, chunk2])
    # Original item should be unchanged
    assert len(item.category) == 1
    assert item.category[0] == "error_a"
