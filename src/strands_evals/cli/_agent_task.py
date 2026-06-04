"""Synthesize a `Callable[[Case], dict]` from a resolved `--agent` target.

The synthesized wrapper covers the single-agent, single-turn evaluation case:
clear the in-memory exporter, build/invoke the agent, capture finished spans,
map them to a :class:`Session`, and return `{"output", "trajectory"}`.

Trace-based evaluators (:class:`Helpfulness`, :class:`Faithfulness`,
:class:`GoalSuccessRate`, etc.) read the returned `trajectory` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..case import Case
from ..mappers.strands_in_memory_session_mapper import StrandsInMemorySessionMapper
from ..telemetry import StrandsEvalsTelemetry
from ._entrypoint import ResolvedEntryPoint


def _default_trace_attributes(case: Case) -> dict[str, Any]:
    return {
        "session.id": case.session_id,
        "gen_ai.conversation.id": case.session_id,
    }


def _build_agent(entry: ResolvedEntryPoint, case: Case, trace_attrs: dict[str, Any]) -> Any:
    """Construct or fetch the agent instance for a given case.

    For `agent_instance`, returns the prebuilt instance verbatim — trace
    attributes cannot be applied post-hoc and are silently dropped (the run
    command warns at the call site).

    For `agent_class`, instantiates with no positional args and injects
    `trace_attributes=...`. For `callable_*`, invokes the factory; the
    factory is responsible for honoring trace attributes if it accepts them.
    """
    kind = entry.kind
    obj = entry.obj
    if kind == "agent_instance":
        return obj
    if kind == "agent_class":
        return obj(trace_attributes=trace_attrs)
    if kind == "callable_no_arg":
        return obj()
    if kind == "callable_one_arg":
        return obj(case)
    raise ValueError(f"unsupported entry-point kind for --agent: {kind}")


def synthesize_task_function(
    entry: ResolvedEntryPoint,
    extra_trace_attributes: dict[str, Any] | None = None,
) -> Callable[[Case], dict[str, Any]]:
    """Return a `task_function` suitable for `Experiment.run_evaluations_async`.

    Args:
        entry: A resolved `--agent` reference (already classified).
        extra_trace_attributes: User-supplied trace attributes from
            `--trace-attributes KEY=VALUE`. Merged on top of the default
            `{"session.id", "gen_ai.conversation.id"}` (user values win).

    Returns:
        A callable matching the `task` parameter of
        `Experiment.run_evaluations_async`: takes a :class:`Case`, returns
        `{"output": str, "trajectory": Session}`.

    Notes:
        Telemetry setup happens once at synthesis time; the in-memory exporter
        is cleared per case. This means the returned wrapper is NOT safe to
        run concurrently across worker threads — the experiment must be run
        with `max_workers=1` for trace fidelity. `run_evaluations_async`
        with higher worker counts will mix spans across cases.
    """
    extra = extra_trace_attributes or {}
    telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
    mapper = StrandsInMemorySessionMapper()

    def task_function(case: Case) -> dict[str, Any]:
        trace_attrs = {**_default_trace_attributes(case), **extra}

        telemetry.in_memory_exporter.clear()
        agent = _build_agent(entry, case, trace_attrs)
        result = agent(case.input)

        spans = list(telemetry.in_memory_exporter.get_finished_spans())
        session = mapper.map_to_session(spans, case.session_id)

        return {"output": str(result), "trajectory": session}

    return task_function
