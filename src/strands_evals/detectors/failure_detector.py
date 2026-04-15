"""Failure detection for agent execution sessions.

Identifies semantic failures (hallucinations, task errors, tool misuse, etc.)
in Session traces using LLM-based analysis with automatic chunking fallback
for sessions exceeding context limits.

Ported from AgentCoreLens tools/failure_detector.py.
"""

import functools
import json
import logging
import re
from collections import OrderedDict

from pydantic import ValidationError
from strands.models.bedrock import BedrockModel
from strands.models.model import Model
from strands.types.content import ContentBlock, Message, Messages
from strands.types.exceptions import ContextWindowOverflowException
from typing_extensions import Union

from .._async import run_async
from ..types.detector import ConfidenceLevel, FailureDetectionStructuredOutput, FailureItem, FailureOutput
from ..types.trace import Session, SpanUnion
from .chunking import merge_chunk_failures, split_spans_by_tokens, would_exceed_context
from .constants import DEFAULT_DETECTOR_MODEL, MIN_CHUNK_SIZE
from .prompt_templates.failure_detection import get_template

logger = logging.getLogger(__name__)
CONFIDENCE_ORDER: dict[ConfidenceLevel, int] = {"low": 0, "medium": 1, "high": 2}


@functools.lru_cache(maxsize=1)
def _get_prompt_overhead_tokens() -> int:
    """Estimate token count of the prompt template with no session data.

    Computed lazily on first call and cached. Matches Lens pattern: render
    template with empty session, measure tokens. This is the fixed overhead
    per chunk that must be subtracted from the token budget so span data
    fills only the remaining space.
    """
    from .chunking import estimate_tokens

    template = get_template("v0")
    empty_prompt = template.build_prompt(session_json="[]")
    return estimate_tokens(template.SYSTEM_PROMPT + empty_prompt)


def detect_failures(
    session: Session,
    *,
    confidence_threshold: ConfidenceLevel = "low",
    model: Union[Model, str, None] = None,
) -> FailureOutput:
    """Detect semantic failures in an agent execution session.

    Args:
        session: The Session object to analyze.
        confidence_threshold: Minimum confidence level ("low", "medium", "high")
            to include a failure. Defaults to "low" (include all).
        model: A Model instance, model ID string (wrapped in BedrockModel),
            or None (uses default Haiku).

    Returns:
        FailureOutput with list of FailureItems, each with span_id, category,
        confidence, and evidence.
    """
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

    threshold_rank = CONFIDENCE_ORDER[confidence_threshold]
    filtered = []
    for f in raw:
        valid_indices = [i for i, conf in enumerate(f.confidence) if CONFIDENCE_ORDER.get(conf, 0) >= threshold_rank]
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
            chunk_json = _serialize_spans(chunk_spans, session.session_id)
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


def _resolve_model(model: Union[Model, str, None]) -> Model:
    """Resolve a model parameter to a Model instance.

    Args:
        model: A Model instance (returned as-is), a model ID string
            (wrapped in BedrockModel), or None (uses default).
    """
    if model is None:
        return BedrockModel(model_id=DEFAULT_DETECTOR_MODEL)
    if isinstance(model, str):
        return BedrockModel(model_id=model)
    return model


def _call_model(model: Model, *, system_prompt: str, user_prompt: str) -> str:
    """Call the model directly and return the full text response.

    Matches Lens prompt delivery: everything goes in a single user message
    with no separate system prompt. When the failure taxonomy and guidelines
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
    crashing, matching Lens's graceful degradation behavior.
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

        items.append(
            FailureItem(
                span_id=err.location,
                category=list(err.category),
                confidence=list(err.confidence),
                evidence=list(err.evidence),
            )
        )
    return items


def _is_context_exceeded(exception: Exception) -> bool:
    """Check if the exception indicates a context window overflow.

    Strands Agent raises ContextWindowOverflowException for context overflows.
    Also checks error message strings as a fallback for providers that raise
    generic errors (e.g., ValidationException from Bedrock).
    """
    if isinstance(exception, ContextWindowOverflowException):
        return True
    msg = str(exception).lower()
    return any(
        p in msg
        for p in [
            "context window",
            "context length",
            "context_length",
            "too long",
            "max_tokens",
            "input_limit",
            "input too large",
        ]
    )


def _flatten_traces_to_spans(traces: list) -> list[SpanUnion]:
    """Flatten all traces into a single span list."""
    return [span for trace in traces for span in trace.spans]


def _serialize_session(session: Session) -> str:
    """Serialize a full Session to JSON for the prompt.

    Matches Lens format: bare traces array, not wrapped in a session object.
    Lens: json.dumps([t.model_dump() for t in traces], indent=2)
    """
    return json.dumps([t.model_dump() for t in session.traces], indent=2, default=str)


def _group_spans_into_traces(spans: list[SpanUnion]) -> list[dict]:
    """Group spans back into trace-like structures for prompt formatting.

    When spans from different traces end up in the same chunk (because
    flattening loses trace boundaries), this reconstructs the per-trace
    grouping so the LLM can distinguish which spans belong to which turn.

    Ported from Lens failure_detector._group_spans_into_traces().
    """
    traces_map: OrderedDict[str, list[SpanUnion]] = OrderedDict()
    for span in spans:
        trace_id = span.span_info.trace_id or "unknown"
        if trace_id not in traces_map:
            traces_map[trace_id] = []
        traces_map[trace_id].append(span)

    traces = []
    for trace_id, trace_spans in traces_map.items():
        trace_dict = {
            "trace_id": trace_id,
            "session_id": trace_spans[0].span_info.session_id,
            "spans": [s.model_dump() for s in trace_spans],
        }
        traces.append(trace_dict)
    return traces


def _serialize_spans(spans: list[SpanUnion], session_id: str) -> str:
    """Serialize a list of spans as a bare traces array for chunk prompts.

    Matches Lens format: bare traces array, not wrapped in a session object.
    Lens: json.dumps(chunk_traces, indent=2)
    """
    traces = _group_spans_into_traces(spans)
    return json.dumps(traces, indent=2, default=str)
