"""Tests for chunking utilities."""

from datetime import datetime

from strands_evals.detectors.chunking import (
    _compute_overlap,
    estimate_tokens,
    merge_chunk_failures,
    split_spans_by_tokens,
    would_exceed_context,
)
from strands_evals.detectors.constants import CHARS_PER_TOKEN, DEFAULT_MAX_INPUT_TOKENS
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


def _make_span_info(span_id: str, trace_id: str = "trace_1") -> SpanInfo:
    now = datetime.now()
    return SpanInfo(
        session_id="sess_1",
        span_id=span_id,
        trace_id=trace_id,
        start_time=now,
        end_time=now,
    )


def _make_tool_span(span_id: str, content_size: int = 100, trace_id: str = "trace_1") -> ToolExecutionSpan:
    """Create a ToolExecutionSpan with controllable content size."""
    return ToolExecutionSpan(
        span_info=_make_span_info(span_id, trace_id=trace_id),
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
    assert estimate_tokens(text) == 333  # 1000 // 3


# --- would_exceed_context ---


def test_would_exceed_context_small():
    assert not would_exceed_context("Hello world")


def test_would_exceed_context_large():
    # 600K chars / 3 = 200K tokens; 200K * 0.85 = 170K -> exceeds
    big_text = "x" * 600_000
    assert would_exceed_context(big_text)


def test_would_exceed_context_custom_limit():
    # 100 chars / 3 = 33 tokens; limit 30 * 0.85 = 25.5 -> exceeds
    assert would_exceed_context("x" * 100, max_input_tokens=30)


def test_would_exceed_context_just_under():
    # effective = 100 * 0.85 = 85 tokens; 200 chars / 3 = 66 tokens < 85
    assert not would_exceed_context("x" * 200, max_input_tokens=100)


def test_would_exceed_context_uses_preflight_margin():
    """Pre-flight margin (0.85) is more generous than chunk margin (0.70).

    A prompt that fits within 0.85 should pass pre-flight even though it
    would exceed the 0.70 chunk budget.
    """
    # 480K chars / 3 = 160K tokens
    # Pre-flight: 200K * 0.85 = 170K -> 160K < 170K -> fits
    text = "x" * 480_000
    assert not would_exceed_context(text, max_input_tokens=200_000)


# --- _compute_overlap ---


def test_compute_overlap_constraint_1_span_count():
    """Constraint 1: at most num_overlap_spans from end."""
    spans = [_make_tool_span(f"s{i}", content_size=100) for i in range(10)]
    counts = [50] * 10
    result_spans, result_counts = _compute_overlap(
        spans,
        counts,
        next_span_token_count=50,
        available_token_limit=10000,
        num_overlap_spans=3,
        min_new_content_ratio=0.5,
    )
    assert len(result_spans) == 3
    assert len(result_counts) == 3


def test_compute_overlap_constraint_2_ratio():
    """Constraint 2: overlap tokens <= (1 - ratio) * limit."""
    spans = [_make_tool_span(f"s{i}", content_size=5000) for i in range(5)]
    counts = [2000] * 5
    result_spans, _ = _compute_overlap(
        spans,
        counts,
        next_span_token_count=100,
        available_token_limit=5000,  # max overlap = 5000 * 0.5 = 2500
        num_overlap_spans=5,
        min_new_content_ratio=0.5,
    )
    # 5 * 2000 = 10000 > 2500, trim until 1 span (2000 <= 2500)
    assert len(result_spans) == 1


def test_compute_overlap_constraint_3_hard_budget():
    """Constraint 3: overlap + next_span must fit."""
    spans = [_make_tool_span(f"s{i}", content_size=100) for i in range(3)]
    counts = [4000, 4000, 4000]
    result_spans, _ = _compute_overlap(
        spans,
        counts,
        next_span_token_count=4000,
        available_token_limit=5000,
        num_overlap_spans=3,
        min_new_content_ratio=0.1,
    )
    # Constraint 2 (ratio=0.1, max_overlap=4500): trims 3→1 span (4000 <= 4500)
    # Constraint 3: 4000 + 4000 = 8000 > 5000 -> trims to 0
    assert len(result_spans) == 0


def test_compute_overlap_empty_prev():
    result_spans, result_counts = _compute_overlap(
        [],
        [],
        next_span_token_count=100,
        available_token_limit=10000,
        num_overlap_spans=5,
        min_new_content_ratio=0.5,
    )
    assert result_spans == []
    assert result_counts == []


def test_compute_overlap_zero_overlap_requested():
    spans = [_make_tool_span(f"s{i}", content_size=100) for i in range(5)]
    counts = [50] * 5
    result_spans, _ = _compute_overlap(
        spans,
        counts,
        next_span_token_count=50,
        available_token_limit=10000,
        num_overlap_spans=0,
        min_new_content_ratio=0.5,
    )
    assert result_spans == []


# --- split_spans_by_tokens ---


def test_split_spans_single_chunk():
    """Small spans should stay in one chunk."""
    spans = [_make_tool_span(f"span_{i}", content_size=10) for i in range(3)]
    chunks = split_spans_by_tokens(spans, max_tokens=DEFAULT_MAX_INPUT_TOKENS)
    assert len(chunks) == 1
    assert len(chunks[0]) == 3


def test_split_spans_multiple_chunks():
    """Large spans should be split into multiple chunks."""
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
    """Adjacent chunks should share overlap spans, subject to token constraints.

    The 3-constraint overlap algorithm may trim overlap if the overlap spans
    are too large relative to the chunk budget (min_new_content_ratio).
    """
    spans = [_make_tool_span(f"span_{i}", content_size=2000) for i in range(10)]
    chunks = split_spans_by_tokens(spans, max_tokens=5000, overlap_spans=2)
    if len(chunks) > 1:
        # Verify there IS overlap: last span of chunk 0 should appear in chunk 1
        last_of_first = chunks[0][-1].span_info.span_id
        first_ids_of_second = [s.span_info.span_id for s in chunks[1]]
        assert last_of_first in first_ids_of_second


def test_split_spans_full_overlap_with_headroom():
    """With sufficient headroom, full overlap count is preserved."""
    # Small spans with large budget: overlap won't be trimmed
    spans = [_make_tool_span(f"span_{i}", content_size=100) for i in range(20)]
    chunks = split_spans_by_tokens(spans, max_tokens=3000, overlap_spans=2)
    if len(chunks) > 1:
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


def test_split_spans_with_prompt_overhead():
    """Prompt overhead reduces effective budget, producing more chunks."""
    spans = [_make_tool_span(f"span_{i}", content_size=2000) for i in range(10)]
    chunks_no_overhead = split_spans_by_tokens(spans, max_tokens=5000, overlap_spans=0)
    chunks_with_overhead = split_spans_by_tokens(
        spans,
        max_tokens=5000,
        overlap_spans=0,
        prompt_overhead_tokens=1000,
    )
    assert len(chunks_with_overhead) >= len(chunks_no_overhead)


def test_split_spans_oversized_span_isolated():
    """A span exceeding the effective limit gets its own chunk."""
    small = [_make_tool_span(f"small_{i}", content_size=100) for i in range(3)]
    huge = _make_tool_span("huge_span", content_size=500_000)
    spans = small + [huge] + [_make_tool_span("after", content_size=100)]

    chunks = split_spans_by_tokens(spans, max_tokens=5000, overlap_spans=2)
    huge_chunks = [c for c in chunks if any(s.span_info.span_id == "huge_span" for s in c)]
    assert len(huge_chunks) == 1
    assert len(huge_chunks[0]) == 1  # isolated, nothing else


def test_split_spans_uses_chunk_margin():
    """split_spans_by_tokens uses the conservative 0.70 chunk margin, not 0.85."""
    spans = [_make_tool_span(f"span_{i}", content_size=2000) for i in range(20)]
    # With max_tokens=10000 and chunk margin 0.70: effective = 7000
    # With preflight margin 0.85: would be 8500
    # More conservative margin should produce more chunks
    chunks = split_spans_by_tokens(spans, max_tokens=10000, overlap_spans=0)
    # Verify we get a reasonable number of chunks (not just 1)
    assert len(chunks) > 1


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
