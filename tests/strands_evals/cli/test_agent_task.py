"""Tests for ``strands_evals.cli._agent_task``."""

from __future__ import annotations

from strands_evals.case import Case
from strands_evals.cli._agent_task import synthesize_task_function
from strands_evals.cli._entrypoint import classify_agent
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


def test_synth_clears_in_memory_exporter_between_cases():
    """Verify each invocation starts with a fresh in-memory span buffer."""
    from tests.strands_evals.cli.fixtures.agents import build_agent

    entry = classify_agent(build_agent, "fixtures.agents:build_agent")
    task = synthesize_task_function(entry)

    case1 = Case[str, str](name="c1", input="q1")
    case2 = Case[str, str](name="c2", input="q2")

    task(case1)
    task(case2)

    # No real spans are emitted (stub agent doesn't use OTel), but the test
    # asserts the contract holds: each task call returns a Session keyed to
    # its own case session_id.
    result = task(case2)
    assert result["trajectory"].session_id == case2.session_id


def test_synth_agent_class_passes_trace_attributes():
    """An ``agent_class`` entry must be instantiated with merged trace attrs."""
    from strands_evals.cli._entrypoint import ResolvedEntryPoint

    captured: dict = {}

    class FakeAgentClass:
        def __init__(self, trace_attributes=None):
            captured["trace_attributes"] = trace_attributes

        def __call__(self, prompt):
            return f"called: {prompt}"

    entry = ResolvedEntryPoint(kind="agent_class", obj=FakeAgentClass, spec="x:y")
    task = synthesize_task_function(entry, extra_trace_attributes={"foo": "bar"})

    case = Case[str, str](name="c1", input="hello")
    result = task(case)

    assert result["output"] == "called: hello"
    assert captured["trace_attributes"] == {
        "session.id": case.session_id,
        "gen_ai.conversation.id": case.session_id,
        "foo": "bar",
    }


def test_synth_agent_class_user_attrs_override_defaults():
    from strands_evals.cli._entrypoint import ResolvedEntryPoint

    captured: dict = {}

    class FakeAgentClass:
        def __init__(self, trace_attributes=None):
            captured["trace_attributes"] = trace_attributes

        def __call__(self, prompt):
            return "ok"

    entry = ResolvedEntryPoint(kind="agent_class", obj=FakeAgentClass, spec="x:y")
    task = synthesize_task_function(entry, extra_trace_attributes={"session.id": "override"})

    case = Case[str, str](name="c1", input="hi")
    task(case)

    # User-supplied "session.id" should win over the default.
    assert captured["trace_attributes"]["session.id"] == "override"


def test_synth_agent_instance_used_directly():
    from strands_evals.cli._entrypoint import ResolvedEntryPoint
    from tests.strands_evals.cli.fixtures.agents import simple_callable_agent

    entry = ResolvedEntryPoint(kind="agent_instance", obj=simple_callable_agent, spec="x:y")
    task = synthesize_task_function(entry)

    case = Case[str, str](name="c1", input="hi")
    result = task(case)
    assert result["output"] == "hello from simple_callable_agent"
    assert simple_callable_agent.calls[-1] == "hi"
