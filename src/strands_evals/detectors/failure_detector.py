"""Failure detection for agent execution sessions.

Identifies semantic failures (hallucinations, task errors, tool misuse, etc.)
in Session traces using LLM-based analysis with automatic chunking fallback
for sessions exceeding context limits.

Ported from AgentCoreLens tools/failure_detector.py.
"""

import json
import logging
import re

from strands.models.bedrock import BedrockModel
from strands.models.model import Model
from strands.types.content import ContentBlock, Message, Messages
from strands.types.exceptions import ContextWindowOverflowException
from typing_extensions import Union

from .._async import run_async
from ..types.detector import ConfidenceLevel, FailureDetectionStructuredOutput, FailureItem, FailureOutput
from ..types.trace import Session, SpanUnion
from .chunking import merge_chunk_failures, split_spans_by_tokens, would_exceed_context
from .constants import DEFAULT_DETECTOR_MODEL
from .prompt_templates.failure_detection import get_template

logger = logging.getLogger(__name__)
CONFIDENCE_ORDER: dict[ConfidenceLevel, int] = {"low": 0, "medium": 1, "high": 2}


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
    filtered = [f for f in raw if _max_confidence_rank(f) >= threshold_rank]
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
    chunks = split_spans_by_tokens(spans)

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

    Uses Model.stream() in text mode to avoid tool-use structured output
    overhead, which can degrade detection quality.
    """
    messages: Messages = [Message(role="user", content=[ContentBlock(text=user_prompt)])]

    async def _stream() -> str:
        chunks: list[str] = []
        async for event in model.stream(messages, system_prompt=system_prompt):
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
    """Parse raw LLM text response into list[FailureItem]."""
    json_str = _extract_json(text)
    output = FailureDetectionStructuredOutput.model_validate_json(json_str)

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


def _max_confidence_rank(item: FailureItem) -> int:
    """Return the maximum confidence rank across all failure modes."""
    if not item.confidence:
        return -1
    return max(CONFIDENCE_ORDER.get(c, 0) for c in item.confidence)


def _is_context_exceeded(exception: Exception) -> bool:
    """Check if the exception indicates a context window overflow.

    Strands Agent raises ContextWindowOverflowException for context overflows.
    Also checks error message strings as a fallback for providers that raise
    generic errors (e.g., ValidationException from Bedrock).
    """
    if isinstance(exception, ContextWindowOverflowException):
        return True
    msg = str(exception).lower()
    return any(p in msg for p in ["context", "too long", "max_tokens", "input_limit"])


def _flatten_traces_to_spans(traces: list) -> list[SpanUnion]:
    """Flatten all traces into a single span list."""
    return [span for trace in traces for span in trace.spans]


def _serialize_session(session: Session) -> str:
    """Serialize a full Session to JSON for the prompt."""
    return json.dumps(session.model_dump(), indent=2, default=str)


def _serialize_spans(spans: list[SpanUnion], session_id: str) -> str:
    """Serialize a list of spans as a minimal session JSON for chunk prompts."""
    return json.dumps(
        {
            "session_id": session_id,
            "traces": [{"spans": [s.model_dump() for s in spans]}],
        },
        indent=2,
        default=str,
    )
