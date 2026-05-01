"""Shared helpers used by multiple detectors.

Keeps internal plumbing (model resolution, context-overflow detection,
session/span serialization) in one place so failure detection, root
cause analysis, and future detectors can reuse it.
"""

import json
from collections import OrderedDict

from strands.models.bedrock import BedrockModel
from strands.models.model import Model
from strands.types.exceptions import ContextWindowOverflowException

from ..types.trace import Session, SpanUnion
from .constants import DEFAULT_DETECTOR_MODEL


def _resolve_model(model: Model | str | None) -> Model:
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

    Uses a bare traces array rather than wrapping in a session object:
    ``json.dumps([t.model_dump() for t in traces], indent=2)``.
    """
    return json.dumps([t.model_dump() for t in session.traces], indent=2, default=str)


def _group_spans_into_traces(spans: list[SpanUnion]) -> list[dict]:
    """Group spans back into trace-like structures for prompt formatting.

    When spans from different traces end up in the same chunk (because
    flattening loses trace boundaries), this reconstructs the per-trace
    grouping so the LLM can distinguish which spans belong to which turn.
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


def _serialize_spans(spans: list[SpanUnion]) -> str:
    """Serialize a list of spans as a bare traces array for chunk prompts.

    Uses the same bare traces array format as _serialize_session, not
    wrapped in a session object:
    ``json.dumps(chunk_traces, indent=2)``.
    """
    traces = _group_spans_into_traces(spans)
    return json.dumps(traces, indent=2, default=str)
