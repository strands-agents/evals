"""Shared utilities for context window management and span chunking.

Used by detect_failures, summarize_execution, and analyze_root_cause
to handle sessions that exceed LLM context limits.

Ported from AgentCoreLens failure_detector._split_spans_by_tokens() and
related helpers, replacing litellm.token_counter() with a conservative
character-based estimate.
"""

import json
import logging

from ..types.detector import ConfidenceLevel, FailureItem
from ..types.trace import SpanUnion

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
CONTEXT_SAFETY_MARGIN = 0.75
DEFAULT_MAX_INPUT_TOKENS = 128_000
CHUNK_OVERLAP_SPANS = 2
MIN_CHUNK_SIZE = 5
_CONFIDENCE_RANK: dict[ConfidenceLevel, int] = {"low": 0, "medium": 1, "high": 2}


def estimate_tokens(text: str) -> int:
    """Conservative token estimate from text length."""
    return len(text) // CHARS_PER_TOKEN


def would_exceed_context(prompt_text: str, max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS) -> bool:
    """Pre-flight check: would this prompt likely exceed context?

    If wrong (false negative), the model returns a context-exceeded error,
    which triggers the chunking fallback in each detector.
    """
    return estimate_tokens(prompt_text) > int(max_input_tokens * CONTEXT_SAFETY_MARGIN)


def split_spans_by_tokens(
    spans: list[SpanUnion],
    max_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    overlap_spans: int = CHUNK_OVERLAP_SPANS,
) -> list[list[SpanUnion]]:
    """Split spans into chunks fitting within token limits.

    Adjacent chunks share ``overlap_spans`` spans for context continuity.
    Ported from Lens ``failure_detector._split_spans_by_tokens()``.

    Args:
        spans: Flat list of span objects to chunk.
        max_tokens: Maximum model input tokens.
        overlap_spans: Number of spans shared between adjacent chunks.

    Returns:
        List of span chunks. Each chunk fits within the effective token limit.
    """
    effective_limit = int(max_tokens * CONTEXT_SAFETY_MARGIN)
    chunks: list[list[SpanUnion]] = []
    current_chunk: list[SpanUnion] = []
    current_tokens = 0

    for span in spans:
        span_tokens = estimate_tokens(json.dumps(span.model_dump(), default=str))
        if current_tokens + span_tokens > effective_limit and len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append(current_chunk)
            overlap = current_chunk[-overlap_spans:] if overlap_spans > 0 else []
            current_chunk = list(overlap)
            current_tokens = sum(estimate_tokens(json.dumps(s.model_dump(), default=str)) for s in overlap)
        current_chunk.append(span)
        current_tokens += span_tokens

    if current_chunk:
        chunks.append(current_chunk)

    logger.info(
        "Split %d spans into %d chunks (max_tokens=%d, overlap=%d)",
        len(spans),
        len(chunks),
        max_tokens,
        overlap_spans,
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
