"""Failure detection for agent execution sessions.

Identifies semantic failures (hallucinations, task errors, tool misuse, etc.)
in Session traces using LLM-based analysis with automatic chunking fallback
for sessions exceeding context limits.
"""

import functools
import json
import logging
import re

from pydantic import ValidationError
from strands.models.model import Model
from strands.types.content import ContentBlock, Message, Messages

from .._async import run_async
from ..types.detector import ConfidenceLevel, FailureDetectionStructuredOutput, FailureItem, FailureOutput
from ..types.trace import Session
from .chunking import merge_chunk_failures, split_spans_by_tokens, would_exceed_context
from .constants import CONFIDENCE_MAP, MIN_CHUNK_SIZE
from .prompt_templates.failure_detection import get_template
from .utils import (
    _flatten_traces_to_spans,
    _is_context_exceeded,
    _resolve_model,
    _serialize_session,
    _serialize_spans,
)

logger = logging.getLogger(__name__)


def _resolve_confidence_threshold(threshold: ConfidenceLevel) -> float:
    """Map the categorical ``"low"|"medium"|"high"`` threshold to the numeric
    value used for comparison (see ``CONFIDENCE_MAP``).
    """
    return CONFIDENCE_MAP[threshold]


@functools.lru_cache(maxsize=1)
def _get_prompt_overhead_tokens() -> int:
    """Estimate token count of the prompt template with no session data.

    Computed lazily on first call and cached: render the template with an
    empty session and measure tokens. This is the fixed overhead per chunk
    that must be subtracted from the token budget so span data fills only
    the remaining space.
    """
    from .chunking import estimate_tokens

    template = get_template("v0")
    empty_prompt = template.build_prompt(session_json="[]")
    return estimate_tokens(template.SYSTEM_PROMPT + empty_prompt)


def detect_failures(
    session: Session,
    *,
    confidence_threshold: ConfidenceLevel = "low",
    model: Model | str | None = None,
) -> FailureOutput:
    """Detect semantic failures in an agent execution session.

    Args:
        session: The Session object to analyze.
        confidence_threshold: Minimum categorical confidence to include a
            failure (``"low"`` | ``"medium"`` | ``"high"``). Internally mapped
            to ``0.5 | 0.75 | 0.9`` via ``CONFIDENCE_MAP`` before filtering.
            Defaults to ``"low"`` (include everything the LLM flagged).
        model: A Model instance, model ID string (wrapped in BedrockModel),
            or None (uses default Haiku).

    Returns:
        FailureOutput with list of FailureItems, each with span_id, category,
        confidence (float in [0.0, 1.0]), and evidence.
    """
    threshold = _resolve_confidence_threshold(confidence_threshold)
    effective_model = _resolve_model(model)
    template = get_template("v0")
    session_json = _serialize_session(session)
    user_prompt = template.build_prompt(session_json=session_json)

    if would_exceed_context(user_prompt):
        raw = _detect_chunked(session, effective_model, template)
    else:
        try:
            raw = _detect_direct(user_prompt, effective_model, template)
        except Exception as e:
            if _is_context_exceeded(e):
                logger.warning("Context exceeded despite pre-flight check, falling back to chunking")
                raw = _detect_chunked(session, effective_model, template)
            else:
                raise

    filtered = []
    for f in raw:
        valid_indices = [i for i, conf in enumerate(f.confidence) if conf >= threshold]
        if valid_indices:
            filtered.append(
                FailureItem(
                    span_id=f.span_id,
                    category=[f.category[i] for i in valid_indices],
                    confidence=[f.confidence[i] for i in valid_indices],
                    evidence=[f.evidence[i] for i in valid_indices],
                )
            )
    return FailureOutput(session_id=session.session_id, failures=filtered)


def _detect_direct(user_prompt: str, model: Model, template: object) -> list[FailureItem]:
    """Attempt direct LLM detection on the full session."""
    text = _call_model(model, system_prompt=template.SYSTEM_PROMPT, user_prompt=user_prompt)
    return _parse_text_result(text)


def _detect_chunked(
    session: Session,
    model: Model,
    template: object,
) -> list[FailureItem]:
    """Chunk session and detect failures per chunk, then merge."""
    spans = _flatten_traces_to_spans(session.traces)

    if len(spans) <= MIN_CHUNK_SIZE:
        logger.warning("Cannot split further: %d spans <= min_chunk_size", len(spans))
        return []

    chunks = split_spans_by_tokens(spans, prompt_overhead_tokens=_get_prompt_overhead_tokens())

    logger.info("Chunked detection: %d spans -> %d chunks", len(spans), len(chunks))

    chunk_results: list[list[FailureItem]] = []
    for i, chunk_spans in enumerate(chunks):
        try:
            chunk_json = _serialize_spans(chunk_spans)
            user_prompt = template.build_prompt(session_json=chunk_json)
            text = _call_model(model, system_prompt=template.SYSTEM_PROMPT, user_prompt=user_prompt)
            chunk_results.append(_parse_text_result(text))
            logger.info("Chunk %d/%d: processed %d spans", i + 1, len(chunks), len(chunk_spans))
        except Exception as e:
            if _is_context_exceeded(e):
                logger.warning("Chunk %d/%d still exceeds context, skipping", i + 1, len(chunks))
            else:
                raise

    return merge_chunk_failures(chunk_results)


def _call_model(model: Model, *, system_prompt: str, user_prompt: str) -> str:
    """Call the model directly and return the full text response.

    Prompt delivery: everything goes in a single user message with no
    separate system prompt. When the failure taxonomy and guidelines
    are in the system role, the model treats them with higher authority and
    over-applies categories (more false positives). Putting everything in
    the user message gives the model a balanced view between "what to look
    for" and "what actually happened".

    Uses Model.stream() in text mode to avoid tool-use structured output
    overhead, which can degrade detection quality.
    """
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    messages: Messages = [Message(role="user", content=[ContentBlock(text=full_prompt)])]

    async def _stream() -> str:
        chunks: list[str] = []
        async for event in model.stream(messages):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    chunks.append(delta["text"])
        return "".join(chunks)

    return run_async(_stream)


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_text_result(text: str) -> list[FailureItem]:
    """Parse raw LLM text response into list[FailureItem].

    Returns an empty list on malformed JSON or validation errors rather than
    crashing, to keep failure detection resilient to occasional bad model output.
    """
    try:
        json_str = _extract_json(text)
        output = FailureDetectionStructuredOutput.model_validate_json(json_str)
    except (json.JSONDecodeError, KeyError, AttributeError, ValidationError) as e:
        logger.warning("Failed to parse LLM response for failure detection: %s", e)
        return []

    items = []
    for err in output.errors:
        # Validate element-wise correspondence
        if not (len(err.category) == len(err.evidence) == len(err.confidence)):
            logger.warning(
                "Mismatched array lengths for location %s: category=%d, evidence=%d, confidence=%d. Skipping.",
                err.location,
                len(err.category),
                len(err.evidence),
                len(err.confidence),
            )
            continue

        # Map the LLM's "low" | "medium" | "high" strings to numeric confidence
        # via CONFIDENCE_MAP. Unknown values map to 0.0 so they get filtered
        # out by any non-zero confidence_threshold.
        confidence_values = [CONFIDENCE_MAP.get(c.lower() if isinstance(c, str) else "", 0.0) for c in err.confidence]

        items.append(
            FailureItem(
                span_id=err.location,
                category=list(err.category),
                confidence=confidence_values,
                evidence=list(err.evidence),
            )
        )
    return items
