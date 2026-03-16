"""Failure detection for agent execution sessions.

Identifies semantic failures (hallucinations, task errors, tool misuse, etc.)
in Session traces using LLM-based analysis with automatic chunking fallback
for sessions exceeding context limits.

Ported from AgentCoreLens tools/failure_detector.py.
"""

import json
import logging

from strands import Agent
from strands.models.model import Model
from strands.types.exceptions import ContextWindowOverflowException
from typing_extensions import Union, cast

from ..types.detector import ConfidenceLevel, FailureDetectionStructuredOutput, FailureItem, FailureOutput
from ..types.trace import Session, SpanUnion
from .chunking import merge_chunk_failures, split_spans_by_tokens, would_exceed_context
from .prompt_templates.failure_detection import get_template

logger = logging.getLogger(__name__)

DEFAULT_DETECTOR_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
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
        model: Any Strands model provider. None uses default Haiku.

    Returns:
        FailureOutput with list of FailureItems, each with span_id, category,
        confidence, and evidence.
    """
    effective_model = model if model is not None else DEFAULT_DETECTOR_MODEL
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


def _detect_direct(user_prompt: str, model: Union[Model, str], template: object) -> list[FailureItem]:
    """Attempt direct LLM detection on the full session."""
    agent = Agent(model=model, system_prompt=template.SYSTEM_PROMPT, callback_handler=None)
    result = agent(user_prompt, structured_output_model=FailureDetectionStructuredOutput)
    return _parse_structured_result(cast(FailureDetectionStructuredOutput, result.structured_output))


def _detect_chunked(
    session: Session,
    model: Union[Model, str],
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
            agent = Agent(model=model, system_prompt=template.SYSTEM_PROMPT, callback_handler=None)
            result = agent(user_prompt, structured_output_model=FailureDetectionStructuredOutput)
            chunk_results.append(
                _parse_structured_result(cast(FailureDetectionStructuredOutput, result.structured_output))
            )
            logger.info("Chunk %d/%d: processed %d spans", i + 1, len(chunks), len(chunk_spans))
        except Exception as e:
            if _is_context_exceeded(e):
                logger.warning("Chunk %d/%d still exceeds context, skipping", i + 1, len(chunks))
            else:
                raise

    return merge_chunk_failures(chunk_results)


def _parse_structured_result(output: FailureDetectionStructuredOutput) -> list[FailureItem]:
    """Convert LLM structured output to list[FailureItem]."""
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
