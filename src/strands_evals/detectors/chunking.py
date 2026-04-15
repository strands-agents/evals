"""Shared utilities for context window management and span chunking.

Used by detect_failures, summarize_execution, and analyze_root_cause
to handle sessions that exceed LLM context limits.

Ported from AgentCoreLens failure_detector._split_spans_by_tokens() and
related helpers. Uses tiktoken (via strands SDK) for accurate token
estimation, with a character-based fallback when tiktoken is unavailable.
"""

import json
import logging
from typing import Any

from ..types.detector import ConfidenceLevel, FailureItem
from ..types.trace import SpanUnion
from .constants import (
    CHARS_PER_TOKEN,
    CHUNK_MIN_NEW_CONTENT_RATIO,
    CHUNK_OVERLAP_SPANS,
    CHUNK_SAFETY_MARGIN,
    DEFAULT_MAX_INPUT_TOKENS,
    PREFLIGHT_SAFETY_MARGIN,
    TIKTOKEN_FALLBACK_MULTIPLIER,
)

logger = logging.getLogger(__name__)
_CONFIDENCE_RANK: dict[ConfidenceLevel, int] = {"low": 0, "medium": 1, "high": 2}

# Cached tiktoken encoding — loaded once on first use.
_tiktoken_encoding: Any = None
_tiktoken_available: bool | None = None


def _get_tiktoken_encoding() -> Any:
    """Return the cached tiktoken encoding, or None if unavailable."""
    global _tiktoken_encoding, _tiktoken_available
    if _tiktoken_available is None:
        try:
            from strands.models.model import _get_encoding

            _tiktoken_encoding = _get_encoding()
            _tiktoken_available = True
            logger.debug("tiktoken encoding loaded via strands SDK")
        except (ImportError, Exception) as exc:
            _tiktoken_available = False
            logger.info("tiktoken unavailable, using char-based fallback: %s", exc)
    return _tiktoken_encoding


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses tiktoken (cl100k_base via strands SDK) when available, then applies
    a correction multiplier to compensate for the systematic underestimation
    vs Anthropic Claude's actual tokenizer. Falls back to
    ``len(text) // CHARS_PER_TOKEN`` otherwise.
    """
    enc = _get_tiktoken_encoding()
    if enc is not None:
        return int(len(enc.encode(text)) * TIKTOKEN_FALLBACK_MULTIPLIER)
    return len(text) // CHARS_PER_TOKEN


def would_exceed_context(prompt_text: str, max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS) -> bool:
    """Pre-flight check: would this prompt likely exceed context?

    Uses a generous safety margin (0.85) so that more sessions attempt
    direct analysis, which is always higher quality than chunked.
    If the estimate is wrong (false negative), the model returns a
    context-exceeded error, which triggers the chunking fallback.
    """
    return estimate_tokens(prompt_text) > int(max_input_tokens * PREFLIGHT_SAFETY_MARGIN)


def _compute_overlap(
    prev_chunk_spans: list[SpanUnion],
    prev_chunk_token_counts: list[int],
    next_span_token_count: int,
    available_token_limit: int,
    num_overlap_spans: int,
    min_new_content_ratio: float,
) -> tuple[list[SpanUnion], list[int]]:
    """Compute overlap spans for the next chunk, applying all constraints.

    Constraints applied in order:
    1. num_overlap_spans: maximum number of spans to consider
    2. min_new_content_ratio: limits overlap to (1 - ratio) of available tokens
    3. Token limit: overlap + next_span must fit within available_token_limit

    Ported from Lens failure_detector._compute_overlap().
    """
    if not prev_chunk_spans or num_overlap_spans <= 0:
        return [], []

    # Constraint 1: take at most num_overlap_spans from end of previous chunk
    overlap_spans = list(prev_chunk_spans[-num_overlap_spans:])
    overlap_token_counts = list(prev_chunk_token_counts[-num_overlap_spans:])

    # Constraint 2: limit overlap tokens based on min_new_content_ratio
    max_overlap_tokens = int(available_token_limit * (1 - min_new_content_ratio))
    while overlap_spans and sum(overlap_token_counts) > max_overlap_tokens:
        overlap_spans = overlap_spans[1:]
        overlap_token_counts = overlap_token_counts[1:]

    # Constraint 3: ensure overlap + next span fits within available_token_limit
    while overlap_spans and sum(overlap_token_counts) + next_span_token_count > available_token_limit:
        overlap_spans = overlap_spans[1:]
        overlap_token_counts = overlap_token_counts[1:]

    return overlap_spans, overlap_token_counts


def split_spans_by_tokens(
    spans: list[SpanUnion],
    max_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    overlap_spans: int = CHUNK_OVERLAP_SPANS,
    min_new_content_ratio: float = CHUNK_MIN_NEW_CONTENT_RATIO,
    prompt_overhead_tokens: int = 0,
) -> list[list[SpanUnion]]:
    """Split spans into chunks fitting within token limits.

    Adjacent chunks share up to ``overlap_spans`` spans for context continuity,
    subject to token-aware constraints that guarantee each chunk has at least
    ``min_new_content_ratio`` fraction of new content.

    Uses a conservative safety margin (0.70) for chunk sizing so that chunks
    have headroom and don't overflow at runtime.

    Ported from Lens ``failure_detector._split_spans_by_tokens()``.

    Args:
        spans: Flat list of span objects to chunk.
        max_tokens: Maximum model input tokens.
        overlap_spans: Maximum number of spans shared between adjacent chunks.
        min_new_content_ratio: Minimum fraction of each chunk that must be new
            content (not overlap).
        prompt_overhead_tokens: Tokens consumed by the prompt template itself.
            Subtracted from the budget so spans fill only the remaining space.

    Returns:
        List of span chunks. Each chunk fits within the effective token limit.
    """
    if not spans:
        return []

    effective_limit = int(max_tokens * CHUNK_SAFETY_MARGIN) - prompt_overhead_tokens

    # Pre-compute token counts for all spans (O(n), done once).
    # indent=2 matches the actual serialization format sent to the LLM
    # (see _serialize_spans / _serialize_session). Without it, compact JSON
    # underestimates by ~30-50%, causing chunks to overflow at runtime.
    span_token_counts = [estimate_tokens(json.dumps(span.model_dump(), indent=2, default=str)) for span in spans]

    chunks: list[list[SpanUnion]] = []
    prev_chunk_spans: list[SpanUnion] = []
    prev_chunk_token_counts: list[int] = []
    span_idx = 0

    while span_idx < len(spans):
        span_token_count = span_token_counts[span_idx]

        # Step 1: compute overlap from previous chunk
        overlap_result_spans, overlap_result_counts = _compute_overlap(
            prev_chunk_spans,
            prev_chunk_token_counts,
            next_span_token_count=span_token_count,
            available_token_limit=effective_limit,
            num_overlap_spans=overlap_spans,
            min_new_content_ratio=min_new_content_ratio,
        )

        if not overlap_result_spans and prev_chunk_spans and overlap_spans > 0:
            logger.warning(
                "Chunk %d: overlap skipped (overlap + next span > available limit)",
                len(chunks) + 1,
            )

        # Initialize chunk with overlap
        current_chunk = list(overlap_result_spans)
        current_chunk_token_counts = list(overlap_result_counts)
        current_token_count = sum(overlap_result_counts) if overlap_result_counts else 0

        # Step 2: add new spans until limit or oversized span
        while span_idx < len(spans):
            span_token_count = span_token_counts[span_idx]

            # Handle oversized span: isolate into its own chunk so it doesn't
            # poison neighbors. The LLM call for this chunk may still fail,
            # but at least surrounding spans get their own properly-sized chunks.
            if span_token_count >= effective_limit:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []

                chunks.append([spans[span_idx]])
                logger.warning(
                    "Span has ~%d tokens, exceeds limit %d -- isolated in own chunk",
                    span_token_count,
                    effective_limit,
                )
                prev_chunk_spans = [spans[span_idx]]
                prev_chunk_token_counts = [span_token_count]
                span_idx += 1
                break

            # Check if span fits in current chunk
            if current_token_count + span_token_count > effective_limit:
                break  # chunk full -- finalize in Step 3

            current_chunk.append(spans[span_idx])
            current_chunk_token_counts.append(span_token_count)
            current_token_count += span_token_count
            span_idx += 1

        # Step 3: finalize current chunk (skip if oversized span was just handled)
        if current_chunk:
            chunks.append(current_chunk)
            prev_chunk_spans = current_chunk
            prev_chunk_token_counts = current_chunk_token_counts

    logger.info(
        "Split %d spans into %d chunks (effective_limit=%d, overlap=%d, min_new_ratio=%.2f)",
        len(spans),
        len(chunks),
        effective_limit,
        overlap_spans,
        min_new_content_ratio,
    )
    return chunks


def merge_chunk_failures(chunk_results: list[list[FailureItem]]) -> list[FailureItem]:
    """Merge failures from overlapping chunks, deduplicating by span_id.

    Keeps highest confidence per category when the same span_id appears
    in multiple chunks. Ported from Lens ``_merge_chunk_failures()``.

    Args:
        chunk_results: List of failure lists, one per chunk.

    Returns:
        Deduplicated list of FailureItems.
    """
    seen: dict[str, FailureItem] = {}
    for chunk in chunk_results:
        for failure in chunk:
            if failure.span_id not in seen:
                # Deep copy to avoid mutating originals
                seen[failure.span_id] = FailureItem(
                    span_id=failure.span_id,
                    category=list(failure.category),
                    confidence=list(failure.confidence),
                    evidence=list(failure.evidence),
                )
            else:
                existing = seen[failure.span_id]
                for i, cat in enumerate(failure.category):
                    if cat in existing.category:
                        idx = existing.category.index(cat)
                        if _CONFIDENCE_RANK.get(failure.confidence[i], 0) > _CONFIDENCE_RANK.get(
                            existing.confidence[idx], 0
                        ):
                            existing.confidence[idx] = failure.confidence[i]
                            existing.evidence[idx] = failure.evidence[i]
                    else:
                        existing.category.append(cat)
                        existing.confidence.append(failure.confidence[i])
                        existing.evidence.append(failure.evidence[i])
    return list(seen.values())
