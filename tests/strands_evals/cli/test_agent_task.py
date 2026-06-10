"""Tests for ``strands_evals.cli._agent_task``."""

from __future__ import annotations

from unittest.mock import MagicMock

from opentelemetry import baggage, context

from strands_evals.case import Case
from strands_evals.cli._agent_task import synthesize_task_function
from strands_evals.cli._entrypoint import ResolvedEntryPoint, classify_agent
from strands_evals.types.trace import Session


def test_synth_callable_no_arg_returns_output_and_session():
    from tests.strands_evals.cli.fixtures.agents import build_agent

    entry = classify_agent(build_agent, "fixtures.agents:build_agent")
    task = synthesize_task_function(entry)

    case = Case[str, str](name="c1", input="hello")
    result = task(case)

    assert result["output"] == "hello from build_agent"
    assert isinstance(result["trajectory"], Session)
    assert result["trajectory"].session_id == case.session_id


def test_synth_callable_one_arg_receives_case():
    from tests.strands_evals.cli.fixtures.agents import build_agent_for_case

    entry = classify_agent(build_agent_for_case, "fixtures.agents:build_agent_for_case")
    task = synthesize_task_function(entry)

    case = Case[str, str](name="c1", input="hello", metadata={"expected_response": "custom"})
    result = task(case)
    assert result["output"] == "custom"


def test_synth_returns_session_keyed_to_case():
    """Each task call returns a Session whose session_id matches the case."""
    from tests.strands_evals.cli.fixtures.agents import build_agent

    entry = classify_agent(build_agent, "fixtures.agents:build_agent")
    task = synthesize_task_function(entry)

    case1 = Case[str, str](name="c1", input="q1")
    case2 = Case[str, str](name="c2", input="q2")

    r1 = task(case1)
    r2 = task(case2)

    assert r1["trajectory"].session_id == case1.session_id
    assert r2["trajectory"].session_id == case2.session_id


def test_synth_does_not_clear_shared_exporter_between_cases(monkeypatch):
    """The synthesizer must not clear the shared in-memory exporter between
    cases — clearing races with concurrent workers under --max-workers > 1
    and silently drops spans. Per-case isolation is the mapper's job
    (it filters spans by session.id / gen_ai.conversation.id, which the
    baggage span processor stamps onto every span emitted in-context).
    """
    from strands_evals.cli import _agent_task as agent_task_module

    fake_exporter = MagicMock()
    fake_exporter.get_finished_spans.return_value = []
    fake_exporter.clear.side_effect = AssertionError("must not clear shared exporter")

    fake_telemetry = MagicMock()
    fake_telemetry.in_memory_exporter = fake_exporter
    fake_telemetry.setup_in_memory_exporter.return_value = fake_telemetry

    monkeypatch.setattr(
        agent_task_module,
        "StrandsEvalsTelemetry",
        MagicMock(return_value=fake_telemetry),
    )

    fake_agent = MagicMock(return_value="ok")
    entry = ResolvedEntryPoint(kind="callable_no_arg", obj=lambda: fake_agent, spec="x:y")
    task = synthesize_task_function(entry)

    case = Case[str, str](name="c1", input="hi")
    task(case)
    task(case)

    fake_exporter.clear.assert_not_called()
    assert fake_exporter.get_finished_spans.call_count == 2


def test_synth_attaches_session_id_baggage_during_invocation():
    """The wrapper must set session.id and gen_ai.conversation.id on the
    active OTel context as W3C Baggage while the agent runs, so the
    BaggageSpanProcessor in telemetry/config.py can stamp them on every
    span the SDK emits.
    """
    captured: dict = {}

    def factory():
        def fake_agent(prompt):
            ctx = context.get_current()
            captured["session.id"] = baggage.get_baggage("session.id", ctx)
            captured["gen_ai.conversation.id"] = baggage.get_baggage("gen_ai.conversation.id", ctx)
            captured["foo"] = baggage.get_baggage("foo", ctx)
            return "ok"

        return fake_agent

    entry = ResolvedEntryPoint(kind="callable_no_arg", obj=factory, spec="x:y")
    task = synthesize_task_function(entry, extra_trace_attributes={"foo": "bar"})

    case = Case[str, str](name="c1", input="hello")
    task(case)

    assert captured["session.id"] == case.session_id
    assert captured["gen_ai.conversation.id"] == case.session_id
    assert captured["foo"] == "bar"


def test_synth_baggage_detached_after_invocation():
    """Baggage must be removed from context once the wrapper returns,
    so it does not leak into other workers / unrelated spans.
    """

    def factory():
        return MagicMock(return_value="ok")

    entry = ResolvedEntryPoint(kind="callable_no_arg", obj=factory, spec="x:y")
    task = synthesize_task_function(entry)

    case = Case[str, str](name="c1", input="hello")
    task(case)

    assert baggage.get_baggage("session.id") is None
    assert baggage.get_baggage("gen_ai.conversation.id") is None


def test_synth_one_arg_factory_called_with_case():
    """callable_one_arg shape must receive the Case so it can read metadata."""
    received: dict = {}

    def factory(case):
        received["case"] = case

        def fake(prompt):
            return f"answered {prompt} for {case.name}"

        return fake

    entry = ResolvedEntryPoint(kind="callable_one_arg", obj=factory, spec="x:y")
    task = synthesize_task_function(entry)

    case = Case[str, str](name="c1", input="hi")
    result = task(case)

    assert received["case"] is case
    assert result["output"] == "answered hi for c1"
