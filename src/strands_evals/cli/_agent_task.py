"""Synthesize a `Callable[[Case], dict]` from a resolved `--agent` target.

The synthesized wrapper covers the single-agent, single-turn evaluation case:
attach a per-case OTel context (carrying `session.id` and any user-supplied
`--trace-attributes` as W3C Baggage), build a fresh agent via the resolved
factory callable, invoke it, capture finished spans, map them to a
:class:`Session`, and return `{"output", "trajectory"}`.

Trace-based evaluators (:class:`Helpfulness`, :class:`Faithfulness`,
:class:`GoalSuccessRate`, etc.) read the returned `trajectory` directly.

A baggage-to-attribute :class:`SpanProcessor` registered by
:meth:`StrandsEvalsTelemetry.setup_in_memory_exporter` copies the baggage
onto every span emitted inside the context, so the mapper's per-session
filter (in :class:`StrandsInMemorySessionMapper`) can split spans cleanly
back to the originating case under any concurrency setting.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import Token
from typing import Any

from opentelemetry import baggage, context
from opentelemetry.context import Context

from ..case import Case
from ..mappers.strands_in_memory_session_mapper import StrandsInMemorySessionMapper
from ..telemetry import StrandsEvalsTelemetry
from ._entrypoint import ResolvedEntryPoint


def _build_agent(entry: ResolvedEntryPoint, case: Case) -> Any:
    """Invoke the resolved factory callable to produce a fresh agent.

    `classify_agent` guarantees `entry.kind` is one of the two callable
    shapes — instance and subclass shapes are rejected at resolution time.
    """
    kind = entry.kind
    obj = entry.obj
    if kind == "callable_no_arg":
        return obj()
    if kind == "callable_one_arg":
        return obj(case)
    raise ValueError(f"unsupported entry-point kind for --agent: {kind}")


def _attach_baggage(case: Case, extra: dict[str, Any]) -> Token[Context]:
    """Attach a context with `session.id`, `gen_ai.conversation.id`, and extras as baggage.

    Returns the token caller must pass to :func:`opentelemetry.context.detach`.
    """
    ctx = baggage.set_baggage("session.id", case.session_id)
    ctx = baggage.set_baggage("gen_ai.conversation.id", case.session_id, ctx)
    for key, value in extra.items():
        ctx = baggage.set_baggage(key, str(value), ctx)
    return context.attach(ctx)


def synthesize_task_function(
    entry: ResolvedEntryPoint,
    extra_trace_attributes: dict[str, Any] | None = None,
) -> Callable[[Case], dict[str, Any]]:
    """Return a `task_function` suitable for `Experiment.run_evaluations_async`.

    Args:
        entry: A resolved `--agent` reference (already classified as one of the
            two callable shapes).
        extra_trace_attributes: User-supplied trace attributes from
            `--trace-attributes KEY=VALUE`. Set as baggage on the per-case
            context alongside the defaults `session.id` and
            `gen_ai.conversation.id`. User keys win on collision.

    Returns:
        A callable matching the `task` parameter of
        `Experiment.run_evaluations_async`: takes a :class:`Case`, returns
        `{"output": str, "trajectory": Session}`.

    Notes:
        Telemetry setup happens once at synthesis time. The in-memory exporter
        is shared across cases and is NOT cleared between calls — clearing
        would race with concurrent workers and silently drop spans. Instead,
        every span emitted inside the per-case context is stamped with
        `session.id` / `gen_ai.conversation.id` by the baggage span processor
        registered in :meth:`StrandsEvalsTelemetry.setup_in_memory_exporter`,
        and `StrandsInMemorySessionMapper.map_to_session` filters by
        `session_id` so each case gets only its own spans regardless of
        `--max-workers`.

        The exporter buffer therefore grows for the lifetime of the run; this
        is acceptable for a single-shot CLI invocation.
    """
    extra = extra_trace_attributes or {}
    telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
    mapper = StrandsInMemorySessionMapper()

    def task_function(case: Case) -> dict[str, Any]:
        token = _attach_baggage(case, extra)
        try:
            agent = _build_agent(entry, case)
            result = agent(case.input)
        finally:
            context.detach(token)

        spans = list(telemetry.in_memory_exporter.get_finished_spans())
        session = mapper.map_to_session(spans, case.session_id)

        return {"output": str(result), "trajectory": session}

    return task_function
