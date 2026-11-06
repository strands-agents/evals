"""Per-trial trace capture: the trial's `Session`, usage, and evaluator outcomes."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import opentelemetry.trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.trace import StatusCode

from ...mappers.strands_in_memory_session_mapper import StrandsInMemorySessionMapper
from ...telemetry.config import StrandsEvalsTelemetry
from ...types.optimization import Usage
from ...types.trace import Session

# Set on the spans `rollout.py` opens, so they are found by kind, never by a name an agent could share.
SPAN_KIND = "strands_evals.optimization.span"
ATTEMPT_SPAN = "attempt"
ROLLOUT_SPAN = "rollout"


@dataclass(frozen=True)
class EvaluatorView:
    """What the `evaluator {name}` spans of one trial say.

    `labels` come from each evaluator span's `gen_ai.evaluation.score.label`, which `Experiment`
    computes but doesn't return. `errors` holds exception type names for the record only; whether a
    judge failure was infrastructure comes from `TrialRunner`'s evaluator proxy, never from these names.
    """

    judge_usage: Usage
    errors: dict[str, str]
    labels: dict[str, str]


class _TrialSpanFilter(SpanProcessor):
    """Keeps the finished spans of registered traces, keyed by hex trace id; every other span is dropped."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: dict[str, list[ReadableSpan]] = {}

    def register(self, trace_id: str) -> None:
        with self._lock:
            self._spans.setdefault(trace_id, [])

    def spans(self, trace_id: str) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans.get(trace_id, ()))

    def release(self, trace_ids: Iterable[str]) -> None:
        with self._lock:
            for trace_id in trace_ids:
                self._spans.pop(trace_id, None)

    def on_end(self, span: ReadableSpan) -> None:
        if span.context is None:
            return
        trace_id = format(span.context.trace_id, "032x")
        with self._lock:
            kept = self._spans.get(trace_id)
            if kept is not None:
                kept.append(span)


# One filter per tracer provider, shared by every `TraceCapture` on it: a provider can't drop a processor,
# so a filter per run would pile up for the life of the process.
_FILTERS: weakref.WeakKeyDictionary[SDKTracerProvider, _TrialSpanFilter] = weakref.WeakKeyDictionary()
_FILTERS_LOCK = threading.Lock()


class TraceCapture:
    """One per run; keeps only spans of registered trial traces.

    Trace and span ids are hex strings, as in a `Session` (`format(trace_id, "032x")`,
    `format(span_id, "016x")`).
    """

    def __init__(self) -> None:
        """Install or attach the in-memory exporter; must run before any `Experiment` is built.

        With no OpenTelemetry SDK tracer provider installed, installs one (`StrandsEvalsTelemetry`) as the
        global provider; then attaches a span processor keeping only registered traces, so nothing else
        piles up in memory.

        Raises:
            RuntimeError: A global tracer provider that isn't the SDK's is already installed, so spans
                can't be read.
        """
        provider = trace_api.get_tracer_provider()
        if not isinstance(provider, SDKTracerProvider):
            StrandsEvalsTelemetry()
            provider = trace_api.get_tracer_provider()
        if not isinstance(provider, SDKTracerProvider):
            raise RuntimeError(
                f"the global tracer provider is a {type(provider).__name__}, not opentelemetry-sdk's "
                "TracerProvider, so the optimizer can't read trial traces; install an SDK TracerProvider instead"
            )
        with _FILTERS_LOCK:
            span_filter = _FILTERS.get(provider)
            if span_filter is None:
                span_filter = _TrialSpanFilter()
                provider.add_span_processor(span_filter)
                _FILTERS[provider] = span_filter
        self._filter = span_filter

    def register(self, trace_id: str) -> None:
        """Keep this trace's spans; spans of unregistered traces are dropped at export."""
        self._filter.register(trace_id)

    def session(self, trace_id: str, attempt_span_id: str) -> Session | None:
        """The attempt's agents as a `Session`, judge spans excluded; None without agent spans.

        Args:
            trace_id: The trial's trace.
            attempt_span_id: The attempt's span; only its subtree is read.
        """
        return _session(trace_id, _subtree(self._filter.spans(trace_id), [attempt_span_id]))

    def usage(self, trace_id: str, attempt_span_id: str) -> Usage:
        """The attempt's usage, judge spans excluded: the last attempt's, or one in `previous_attempts`.

        Sums the subtree's `chat` spans; `seconds` is its `rollout` span's duration, or the attempt span's
        when there is none (a `build_task` task).

        Args:
            trace_id: The trial's trace.
            attempt_span_id: The attempt's span; only its subtree is read.
        """
        spans = self._filter.spans(trace_id)
        subtree = _subtree(spans, [attempt_span_id])
        return _usage(subtree, _attempt_seconds(spans, subtree, attempt_span_id))

    def evaluator_view(self, trace_id: str) -> EvaluatorView:
        """One walk over the trial's `evaluator {name}` spans: judge usage, errors, and labels.

        Judge usage sums the `chat` spans under every evaluator span; its `seconds` is the evaluator
        spans' total duration.
        """
        spans = self._filter.spans(trace_id)
        evaluator_span_ids: list[str] = []
        seconds = 0.0
        errors: dict[str, str] = {}
        labels: dict[str, str] = {}
        for span in spans:
            attributes = span.attributes or {}
            name = attributes.get("gen_ai.evaluation.name")
            if name is None or not span.name.startswith("evaluator "):
                continue
            evaluator_span_ids.append(_span_id(span))
            seconds += _duration(span)
            if span.status.status_code is StatusCode.ERROR:
                errors[str(name)] = _exception_type(span)
            elif "gen_ai.evaluation.score.label" in attributes:
                labels[str(name)] = str(attributes["gen_ai.evaluation.score.label"])
        judge_spans = _subtree(spans, evaluator_span_ids)
        return EvaluatorView(judge_usage=_usage(judge_spans, seconds), errors=errors, labels=labels)

    def release(self, trace_ids: Iterable[str]) -> None:
        """Drop only these trials' spans."""
        self._filter.release(trace_ids)


def _span_id(span: ReadableSpan) -> str:
    return format(span.context.span_id, "016x") if span.context is not None else ""


def _duration(span: ReadableSpan) -> float:
    if span.start_time is None or span.end_time is None:
        return 0.0
    return (span.end_time - span.start_time) / 1e9


def _subtree(spans: Sequence[ReadableSpan], root_span_ids: Iterable[str]) -> list[ReadableSpan]:
    """The descendants of the roots (not the roots themselves), in the order the spans finished."""
    children: dict[str, list[str]] = {}
    for span in spans:
        if span.parent is not None:
            children.setdefault(format(span.parent.span_id, "016x"), []).append(_span_id(span))
    inside: set[str] = set()
    pending = list(root_span_ids)
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in inside:
                inside.add(child)
                pending.append(child)
    return [span for span in spans if _span_id(span) in inside]


def _attempt_seconds(spans: Sequence[ReadableSpan], subtree: Sequence[ReadableSpan], attempt_span_id: str) -> float:
    """The `rollout` span's duration, else the attempt span's own."""
    for span in subtree:
        if (span.attributes or {}).get(SPAN_KIND) == ROLLOUT_SPAN:
            return _duration(span)
    for span in spans:
        if _span_id(span) == attempt_span_id:
            return _duration(span)
    return 0.0


def _int_attribute(attributes: Any, *names: str) -> int | None:
    """The first of `names` present, as an int; None when none is."""
    for name in names:
        value = attributes.get(name)
        if value is not None:
            return int(value)
    return None


def _usage(spans: Sequence[ReadableSpan], seconds: float) -> Usage:
    """Usage summed over the `chat` spans: never `invoke_agent` spans, which carry running totals."""
    usage = Usage.zero().model_copy(update={"seconds": seconds})
    for span in spans:
        attributes = span.attributes or {}
        if attributes.get("gen_ai.operation.name") != "chat":
            continue
        output_tokens = _int_attribute(attributes, "gen_ai.usage.output_tokens") or 0
        tokens = _int_attribute(attributes, "gen_ai.usage.total_tokens")
        if tokens is None:
            tokens = (_int_attribute(attributes, "gen_ai.usage.input_tokens") or 0) + output_tokens
        model = attributes.get("gen_ai.request.model")
        usage += Usage(
            tokens=tokens,
            # The span's own input_tokens excludes cache tokens on some SDK versions and includes them on others.
            input_tokens=tokens - output_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=_int_attribute(
                attributes, "gen_ai.usage.cache_read.input_tokens", "gen_ai.usage.cache_read_input_tokens"
            ),
            cache_write_tokens=_int_attribute(
                attributes, "gen_ai.usage.cache_creation.input_tokens", "gen_ai.usage.cache_write_input_tokens"
            ),
            model_calls=1,
            seconds=0.0,
            models=[str(model)] if model else [],
        )
    return usage


def _session(trace_id: str, spans: Sequence[ReadableSpan]) -> Session | None:
    """The spans as a one-trace `Session`; None when none of them is an agent span.

    The spans are already exactly one attempt's subtree, so every agent in it is kept, whatever session
    id each one set; `map_to_session` would keep only spans carrying the id it is given. The first id
    found (else the trace id) names the session. The convention is read from the first Strands span (one
    with `gen_ai.operation.name`), since a span the rollout opened itself may finish first.
    """
    if not spans:
        return None
    session_id = trace_id
    for span in spans:
        attributes = span.attributes or {}
        found = attributes.get("gen_ai.conversation.id", attributes.get("session.id"))
        if found is not None:
            session_id = str(found)
            break
    mapper = StrandsInMemorySessionMapper()
    strands_span = next((span for span in spans if "gen_ai.operation.name" in (span.attributes or {})), spans[0])
    mapper._convention_version = mapper._detect_convention_version(strands_span)
    trace = mapper._convert_trace(trace_id, list(spans), session_id)
    return Session(traces=[trace], session_id=session_id) if trace.spans else None


def _exception_type(span: ReadableSpan) -> str:
    """The type name of the exception an ERROR span recorded."""
    for event in span.events:
        if event.name == "exception" and event.attributes and "exception.type" in event.attributes:
            return str(event.attributes["exception.type"])
    return "Exception"
