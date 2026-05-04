"""Root cause analysis for agent execution sessions.

Performs deep causal analysis of detected failures using a 3-tier hierarchical
strategy to handle sessions that exceed LLM context limits:

1. Direct analysis (full session)
2. Failure path pruning (keep only ancestor + descendant spans of failures)
3. Chunked analysis with merge (split pruned session, analyze each, merge)
"""

import json
import logging
from collections import deque
from copy import deepcopy

from strands import Agent
from strands.models.model import Model
from strands.types.exceptions import ContextWindowOverflowException
from typing_extensions import cast

from ..types.detector import FailureItem, RCAItem, RCAOutput, RCAStructuredOutput
from ..types.trace import Session, SpanUnion
from .chunking import would_exceed_context
from .constants import (
    RCA_MAX_DESCENDANTS,
    RCA_MIN_WINDOW_SIZE,
    RCA_WINDOW_SPLIT_FACTOR,
)
from .failure_detector import detect_failures
from .prompt_templates.root_cause import get_merge_template, get_template
from .utils import (
    _is_context_exceeded,
    _resolve_model,
    _serialize_session,
    _serialize_spans,
)

logger = logging.getLogger(__name__)

_TEMPLATE_VERSION = "v0"
_TEMPLATE = get_template(_TEMPLATE_VERSION)
_MERGE_TEMPLATE = get_merge_template(_TEMPLATE_VERSION)


def analyze_root_cause(
    session: Session,
    failures: list[FailureItem] | None = None,
    *,
    model: Model | str | None = None,
) -> RCAOutput:
    """Perform root cause analysis on detected failures in a session.

    Uses a 3-tier fallback strategy for handling large sessions:
    1. Direct: analyze full session in one LLM call
    2. Pruned: keep only spans on failure paths, retry
    3. Chunked: split pruned session into windows, analyze each, merge

    Args:
        session: The Session object to analyze.
        failures: List of FailureItems from detect_failures(). If None,
            detect_failures() is called automatically.
        model: A Model instance, model ID string (wrapped in BedrockModel),
            or None (uses default). Passed to Strands Agent.

    Returns:
        RCAOutput with list of RCAItems, each with failure_span_id, location,
        causality, propagation_impact, root_cause_explanation, fix_type, and
        fix_recommendation.
    """
    if failures is None:
        failures = detect_failures(session, model=model).failures
    if not failures:
        return RCAOutput()

    effective_model = _resolve_model(model)

    session_json = _serialize_session(session)
    failures_json = _serialize_failures(failures)

    # Tier 1: Direct analysis
    system_prompt = _TEMPLATE.build_prompt(
        execution_json=session_json,
        execution_failures_json=failures_json,
    )

    if not would_exceed_context(system_prompt):
        try:
            raw = _rca_direct(system_prompt, effective_model)
            return RCAOutput(root_causes=raw)
        except Exception as e:
            if not _is_context_exceeded(e):
                raise
            logger.warning("Context exceeded despite pre-flight check, falling back to pruning")

    # Tier 2: Pruned analysis
    failure_span_ids = [f.span_id for f in failures]
    pruned_session = _prune_session_to_failure_paths(session, failure_span_ids)
    pruned_json = _serialize_session(pruned_session)

    pruned_prompt = _TEMPLATE.build_prompt(
        execution_json=pruned_json,
        execution_failures_json=failures_json,
    )

    if not would_exceed_context(pruned_prompt):
        try:
            raw = _rca_direct(pruned_prompt, effective_model)
            return RCAOutput(root_causes=raw)
        except Exception as e:
            if not _is_context_exceeded(e):
                raise
            logger.warning("Pruned session still exceeds context, falling back to chunking")

    # Tier 3: Chunked analysis with merge
    raw = _rca_chunked(pruned_session, failures, effective_model)
    return RCAOutput(root_causes=raw)


def _rca_direct(system_prompt: str, model: Model) -> list[RCAItem]:
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        callback_handler=None,
    )
    result = agent(_TEMPLATE.USER_PROMPT, structured_output_model=RCAStructuredOutput)
    return _parse_structured_result(cast(RCAStructuredOutput, result.structured_output))


def _rca_chunked(
    pruned_session: Session,
    failures: list[FailureItem],
    model: Model,
) -> list[RCAItem]:
    """Split pruned session into per-trace windows, analyze each, merge results."""
    total_spans = sum(len(t.spans) for t in pruned_session.traces)

    if total_spans <= RCA_MIN_WINDOW_SIZE:
        logger.warning("Cannot split session further (%d spans), returning empty", total_spans)
        return []

    window_size = max(RCA_MIN_WINDOW_SIZE, total_spans // RCA_WINDOW_SPLIT_FACTOR)
    failures_json = _serialize_failures(failures)

    logger.info(
        "Chunking RCA: %d spans, window size %d",
        total_spans,
        window_size,
    )

    chunk_results: list[str] = []
    window_num = 0

    for trace in pruned_session.traces:
        if not trace.spans:
            continue

        for i in range(0, len(trace.spans), window_size):
            window_spans = trace.spans[i : i + window_size]
            window_num += 1

            window_span_ids = {_get_span_id(s) for s in window_spans}
            window_failures = [f for f in failures if f.span_id in window_span_ids]
            window_failures_json = _serialize_failures(window_failures) if window_failures else failures_json

            window_json = _serialize_spans(window_spans)
            system_prompt = _TEMPLATE.build_prompt(
                execution_json=window_json,
                execution_failures_json=window_failures_json,
            )

            try:
                agent = Agent(model=model, system_prompt=system_prompt, callback_handler=None)
                result = agent(_TEMPLATE.USER_PROMPT, structured_output_model=RCAStructuredOutput)
                structured = cast(RCAStructuredOutput, result.structured_output)
                chunk_results.append(structured.model_dump_json(by_alias=True, indent=2))
                logger.info("RCA chunk %d: processed %d spans", window_num, len(window_spans))
            except ContextWindowOverflowException:
                logger.warning("RCA chunk %d still too large, skipping", window_num)
            except Exception as e:
                if _is_context_exceeded(e):
                    logger.warning("RCA chunk %d context exceeded (string match), skipping", window_num)
                else:
                    raise

    if not chunk_results:
        return []

    if len(chunk_results) == 1:
        output = RCAStructuredOutput.model_validate_json(chunk_results[0])
        return _parse_structured_result(output)

    # Merge chunk results
    logger.info("Merging %d chunk RCA results", len(chunk_results))
    merge_prompt = _MERGE_TEMPLATE.build_prompt(
        chunk_results=chunk_results,
        execution_failures_json=failures_json,
    )

    agent = Agent(model=model, system_prompt=merge_prompt, callback_handler=None)
    result = agent(_MERGE_TEMPLATE.USER_PROMPT, structured_output_model=RCAStructuredOutput)
    return _parse_structured_result(cast(RCAStructuredOutput, result.structured_output))


def _parse_structured_result(output: RCAStructuredOutput) -> list[RCAItem]:
    items = []
    for rc in output.root_causes:
        items.append(
            RCAItem(
                failure_span_id=rc.failure_span_id,
                location=rc.location,
                causality=rc.failure_causality,
                propagation_impact=list(rc.failure_propagation_impact),
                failure_detection_timing=rc.failure_detection_timing,
                completion_status=rc.completion_status,
                root_cause_explanation=rc.root_cause_explanation,
                fix_type=rc.fix_recommendation.fix_type,
                fix_recommendation=rc.fix_recommendation.recommendation,
            )
        )
    return items


def _prune_session_to_failure_paths(
    session: Session,
    failure_span_ids: list[str],
    max_descendants: int = RCA_MAX_DESCENDANTS,
) -> Session:
    """Prune session to keep only paths from root spans to failure spans.

    Keeps:
    1. All spans on paths from root to failure spans (ancestors)
    2. Up to max_descendants descendants for each failure span (context)
    """
    if not failure_span_ids:
        return session

    pruned_session = deepcopy(session)
    failure_set = set(failure_span_ids)

    for trace in pruned_session.traces:
        span_map: dict[str, SpanUnion] = {}
        parent_map: dict[str, str | None] = {}
        children_map: dict[str, list[str]] = {}

        for span in trace.spans:
            sid = _get_span_id(span)
            pid = _get_parent_span_id(span)
            span_map[sid] = span
            parent_map[sid] = pid
            if pid:
                children_map.setdefault(pid, []).append(sid)

        spans_to_keep: set[str] = set()
        for fid in failure_set:
            if fid in span_map:
                _trace_to_roots(fid, parent_map, spans_to_keep)
                _collect_descendants(fid, children_map, spans_to_keep, max_descendants)

        original_count = len(trace.spans)
        trace.spans = [s for s in trace.spans if _get_span_id(s) in spans_to_keep]

        if len(trace.spans) < original_count:
            logger.info(
                "Pruned %d spans from trace (kept %d/%d)",
                original_count - len(trace.spans),
                len(trace.spans),
                original_count,
            )

    return pruned_session


def _trace_to_roots(span_id: str, parent_map: dict[str, str | None], visited: set[str]) -> None:
    if span_id in visited or span_id not in parent_map:
        return
    visited.add(span_id)
    parent_id = parent_map[span_id]
    if parent_id:
        _trace_to_roots(parent_id, parent_map, visited)


def _collect_descendants(
    span_id: str,
    children_map: dict[str, list[str]],
    visited: set[str],
    max_count: int,
) -> None:
    if max_count <= 0:
        return
    queue: deque[str] = deque([span_id])
    collected = 0
    while queue and collected < max_count:
        current = queue.popleft()
        for child_id in children_map.get(current, []):
            if collected >= max_count:
                break
            if child_id not in visited:
                visited.add(child_id)
                queue.append(child_id)
                collected += 1


def _get_span_id(span: SpanUnion) -> str:
    return span.span_info.span_id or ""


def _get_parent_span_id(span: SpanUnion) -> str | None:
    return span.span_info.parent_span_id


def _serialize_failures(failures: list[FailureItem]) -> str:
    return json.dumps([f.model_dump() for f in failures], indent=2, default=str)
