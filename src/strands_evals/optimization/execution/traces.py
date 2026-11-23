"""Per-trial trace capture: the trial's `Session`, usage, and evaluator outcomes."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import opentelemetry.trace as trace_api
from opentelemetry.sdk.trace import Event, ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags

from ...mappers.strands_in_memory_session_mapper import StrandsInMemorySessionMapper
from ...telemetry.config import StrandsEvalsTelemetry
from ...types.optimization import CANDIDATE_SPAN_KEY, Usage
from ...types.trace import Session

# Set on the spans `rollout.py` opens, so they are found by kind, never by a name an agent could share.
SPAN_KIND = "strands_evals.optimization.span"
ATTEMPT_SPAN = "attempt"
ROLLOUT_SPAN = "rollout"
# True in the `trace_attributes` of each agent the factory builds in a rollout, so its spans are found.
FACTORY_AGENT = "strands_evals.optimization.factory_agent"
# On a `rollout` span: True until the factory builds something it can't tag (then False). While True, the
# attempt's other agents are the benchmark's.
FACTORY_TAGGED = "strands_evals.optimization.factory_tagged"


@dataclass(frozen=True)
class EvaluatorView:
    """What the `evaluator {name}` spans of one trial say, keyed by evaluator name.

    `judge_usage` is each evaluator's own spend (the spans under its evaluator span), so a judge-only
    re-run can keep each score's judge spend with that score. `labels` come from each evaluator span's
    `gen_ai.evaluation.score.label`, which `Experiment` computes but doesn't return. `errors` holds the
    exception type names of evaluators whose span ended in ERROR.
    """

    judge_usage: dict[str, Usage]
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
        """Install an SDK tracer provider if needed and attach the trial span filter; must run before any
        `Experiment` is built.

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

        Every agent in the attempt is kept. When the attempt's agents are told apart (`_candidate_spans`),
        each span's `metadata[CANDIDATE_SPAN_KEY]` says whether it is the candidate's.

        Args:
            trace_id: The trial's trace.
            attempt_span_id: The attempt's span; only its subtree is read.
        """
        subtree = _subtree(self._filter.spans(trace_id), [attempt_span_id])
        return _session(trace_id, subtree, _candidate_spans(subtree))

    def usage(self, trace_id: str, attempt_span_id: str) -> Usage:
        """The candidate's usage in the attempt, judge spans excluded: the last attempt's, or one in
        `previous_attempts`.

        Sums the `chat` spans of the agents the factory built and of everything under them
        (`_candidate_spans`), or of the whole subtree when they can't be told apart; `seconds` is the
        `rollout` span's duration, or the attempt span's when there is none (a `build_task` task).

        Args:
            trace_id: The trial's trace.
            attempt_span_id: The attempt's span; only its subtree is read.
        """
        spans = self._filter.spans(trace_id)
        subtree = _subtree(spans, [attempt_span_id])
        candidate = _candidate_spans(subtree)
        seconds = _attempt_seconds(spans, subtree, attempt_span_id)
        return _usage(subtree if candidate is None else [s for s in subtree if _span_id(s) in candidate], seconds)

    def benchmark_usage(self, trace_id: str, attempt_span_id: str) -> Usage:
        """The usage of the attempt's other agents: the `chat` spans `usage` leaves out (a simulated user).

        `seconds` is always 0. Zero when the attempt's agents can't be told apart (`_candidate_spans`).

        Args:
            trace_id: The trial's trace.
            attempt_span_id: The attempt's span; only its subtree is read.
        """
        subtree = _subtree(self._filter.spans(trace_id), [attempt_span_id])
        candidate = _candidate_spans(subtree)
        return _usage([] if candidate is None else [s for s in subtree if _span_id(s) not in candidate], 0.0)

    def evaluator_view(self, trace_id: str) -> EvaluatorView:
        """One walk over the trial's `evaluator {name}` spans: judge usage, errors, and labels.

        Each evaluator's judge usage sums the `chat` spans under its evaluator span; its `seconds` is
        that span's duration. One pass over the spans for every evaluator: each judge span goes to its
        nearest evaluator ancestor.
        """
        spans = self._filter.spans(trace_id)
        evaluator_of: dict[str, str] = {}  # evaluator span id -> evaluator name
        seconds: dict[str, float] = {}
        errors: dict[str, str] = {}
        labels: dict[str, str] = {}
        for span in spans:
            attributes = span.attributes or {}
            name = attributes.get("gen_ai.evaluation.name")
            if name is None or not span.name.startswith("evaluator "):
                continue
            name = str(name)
            evaluator_of[_span_id(span)] = name
            seconds[name] = seconds.get(name, 0.0) + _duration(span)
            if span.status.status_code is StatusCode.ERROR:
                errors[name] = _exception_type(span)
            elif "gen_ai.evaluation.score.label" in attributes:
                labels[name] = str(attributes["gen_ai.evaluation.score.label"])
        parent_of = {_span_id(span): format(span.parent.span_id, "016x") for span in spans if span.parent is not None}
        judge_spans = _subtree(spans, evaluator_of.keys())  # every span under an evaluator span, in one call
        spans_by_evaluator: dict[str, list[ReadableSpan]] = {name: [] for name in seconds}
        for span in judge_spans:
            ancestor = parent_of.get(_span_id(span))
            while ancestor is not None and ancestor not in evaluator_of:
                ancestor = parent_of.get(ancestor)
            if ancestor is not None:
                spans_by_evaluator[evaluator_of[ancestor]].append(span)
        judge_usage = {name: _usage(spans_by_evaluator[name], seconds[name]) for name in seconds}
        return EvaluatorView(judge_usage=judge_usage, errors=errors, labels=labels)

    def case_error(self, trace_id: str) -> tuple[str | None, str | None]:
        """The exception `Experiment` caught for the trial's case, read from its `execute_case` span.

        That span is the trial's root. `Experiment` records an exception on it whenever it gives up on the
        case and writes a placeholder row for every evaluator instead of its result. That includes a task
        that raised: its own exception is recorded there too, so the result means `Experiment` failed after
        the task returned only for a trial whose task completed.

        The message is the span event's `exception.message`, so an OTel attribute length limit
        (`OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT`) can cut it shorter than the reason `Experiment` reported.

        Args:
            trace_id: The trial's trace.

        Returns:
            The exception's type name (without its module) and message, or (None, None) if none was recorded.
        """
        for span in self._filter.spans(trace_id):
            if span.parent is not None or not span.name.startswith("execute_case "):
                continue
            for event in span.events:
                if event.name == "exception" and event.attributes:
                    error_type = str(event.attributes.get("exception.type", "Exception")).rsplit(".", 1)[-1]
                    return error_type, str(event.attributes.get("exception.message", ""))
        return None, None

    def release(self, trace_ids: Iterable[str]) -> None:
        """Drop only these trials' spans."""
        self._filter.release(trace_ids)


def ingest(spans: Iterable[Mapping[str, Any]]) -> None:
    """Add spans another process recorded (a Node agent's, `node.py`) as if they had finished here.

    Each span is JSON as `node/runner.mjs` writes it: hex ids (its own trace's, so a span of a registered trial
    trace joins it), attributes, events, status, kind, and times in nanoseconds. Rebuilt as a `ReadableSpan` and
    given to the trial span filter of the current tracer provider, which keeps it only when its trace is
    registered. Does nothing outside a run (no filter).
    """
    span_filter = _FILTERS.get(trace_api.get_tracer_provider())  # type: ignore[call-overload]
    if span_filter is None:
        return
    for data in spans:
        trace_id = int(data["trace_id"], 16)
        parent = data.get("parent_span_id")
        status = data.get("status") or {}
        scope = data.get("scope")
        span_filter.on_end(
            ReadableSpan(
                name=str(data["name"]),
                context=SpanContext(trace_id, int(data["span_id"], 16), is_remote=True, trace_flags=TraceFlags(1)),
                parent=SpanContext(trace_id, int(parent, 16), is_remote=True) if parent else None,
                attributes=dict(data.get("attributes") or {}),
                events=[
                    Event(str(event["name"]), dict(event.get("attributes") or {}), int(event["time_unix_nano"]))
                    for event in data.get("events") or ()
                ],
                kind=SpanKind(int(data.get("kind") or 0)),
                status=Status(StatusCode(int(status.get("code") or 0)), status.get("message")),
                start_time=int(data["start_time_unix_nano"]),
                end_time=int(data["end_time_unix_nano"]),
                instrumentation_scope=InstrumentationScope(str(scope)) if scope else None,
            )
        )


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


def _candidate_spans(subtree: Sequence[ReadableSpan]) -> set[str] | None:
    """The span ids of the agents the factory built and of everything under them (sub-agents, conversation
    managers, tools), or None when the attempt's agents can't be told apart.

    They can when the attempt's `rollout` span says the factory tagged everything it built (`FACTORY_TAGGED`),
    even if that was nothing; then every other span in the attempt (a simulated user, an agent the environment
    built, a model call the rollout made itself) is the benchmark's. They can't when there is no `rollout` span
    (the `build_task` path, or a `make_task` attempt whose environment failed before the rollout began), or
    when the factory built something it couldn't tag.
    """
    if not any(
        (attributes := span.attributes or {}).get(SPAN_KIND) == ROLLOUT_SPAN and attributes.get(FACTORY_TAGGED)
        for span in subtree
    ):
        return None
    tagged = [_span_id(span) for span in subtree if (span.attributes or {}).get(FACTORY_AGENT)]
    return set(tagged) | {_span_id(span) for span in _subtree(subtree, tagged)}


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


def _session(trace_id: str, spans: Sequence[ReadableSpan], candidate: set[str] | None) -> Session | None:
    """The spans as a one-trace `Session`; None when none of them is an agent span.

    The spans are already exactly one attempt's subtree, so every agent in it is kept, whatever session
    id each one set; `map_to_session` would keep only spans carrying the id it is given. The first id
    found (else the trace id) names the session. The convention is read from the first Strands span (one
    with `gen_ai.operation.name`), since a span the rollout opened itself may finish first. With
    `candidate` (from `_candidate_spans`), each span's `metadata[CANDIDATE_SPAN_KEY]` says whether it is in it.
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
    if candidate is not None:
        for mapped in trace.spans:
            mapped.metadata = {**(mapped.metadata or {}), CANDIDATE_SPAN_KEY: mapped.span_info.span_id in candidate}
    return Session(traces=[trace], session_id=session_id) if trace.spans else None


def _exception_type(span: ReadableSpan) -> str:
    """The type name of the exception an ERROR span recorded."""
    for event in span.events:
        if event.name == "exception" and event.attributes and "exception.type" in event.attributes:
            return str(event.attributes["exception.type"])
    return "Exception"
