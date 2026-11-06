import logging
import threading
from contextlib import asynccontextmanager, contextmanager

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from strands import Agent
from strands.types.exceptions import EventLoopException, ModelThrottledException

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import InfrastructureError, TextValue, tunable
from strands_evals.optimization.declaration.tunable import _FactoryArgumentError
from strands_evals.optimization.execution.rollout import (
    TrialLog,
    default_rollout,
    instrument,
    make_task,
)
from strands_evals.optimization.execution.traces import TraceCapture
from strands_evals.types import EvaluationOutput
from strands_evals.types.optimization import TrialStatus
from strands_evals.types.trace import AgentInvocationSpan, Session
from tests.strands_evals.optimization.scripted_model import ScriptedModel


@pytest.fixture
def traces():
    return TraceCapture()


@pytest.fixture
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("strands_evals.experiment._INITIAL_RETRY_DELAY", 0)


def _case(name="case-1", text="question"):
    return Case(name=name, input=text)


def _agent_factory(reply="answer", **model_kwargs):
    def build():
        return Agent(model=ScriptedModel([reply], **model_kwargs), callback_handler=None)

    return build


async def _run(task, case, traces):
    """Run a task the way `Experiment` does: inside a fresh root span, so each trial is its own trace."""
    with trace_api.get_tracer("optimization-tests").start_as_current_span("execute_case", context=Context()):
        return await task(case)


# default_rollout


async def test_default_rollout_returns_the_agents_reply_as_text():
    out = await default_rollout(_agent_factory("forty-two"), _case())

    assert list(out) == ["output"]
    assert out["output"].strip() == "forty-two"


async def test_default_rollout_builds_a_fresh_agent_each_time():
    built = []

    def build():
        agent = Agent(model=ScriptedModel(["x"]), callback_handler=None)
        built.append(agent)
        return agent

    await default_rollout(build, _case())
    await default_rollout(build, _case())

    assert len(built) == 2
    assert built[0] is not built[1]


# make_task


async def test_make_task_without_environment_passes_the_factory():
    seen = []

    async def rollout(agent_factory, case):
        seen.append(agent_factory)
        return {"output": case.input}

    factory = _agent_factory()
    out = await make_task(factory, rollout)(_case(text="hello"))

    assert out == {"output": "hello"}
    assert seen == [factory]


async def test_make_task_runs_a_sync_rollout():
    def rollout(agent_factory, case):
        return {"output": f"sync {case.input}"}

    assert await make_task(_agent_factory(), rollout)(_case(text="q")) == {"output": "sync q"}


@tunable(system_prompt=TextValue(default="be brief"))
def _build(system_prompt, sandbox=None, context_manager=True):
    return {"system_prompt": system_prompt, "sandbox": sandbox, "context_manager": context_manager}


async def _built(agent_factory, case):
    return {"output": agent_factory()}


async def test_resources_reach_a_factory_that_declares_them():
    task = make_task(_build.bind({}), _built, environment=lambda case: {"sandbox": f"box-{case.name}"})

    out = await task(_case())

    assert out["output"]["sandbox"] == "box-case-1"


async def test_resources_the_factory_doesnt_declare_are_dropped():
    task = make_task(_build.bind({}), _built, environment=lambda case: {"tools": ["unused"]})

    out = await task(_case())

    assert out["output"]["sandbox"] is None


async def test_a_custom_rollout_passes_a_non_tunable_kwarg():
    async def rollout(agent_factory, case):
        return {"output": agent_factory(context_manager=False)}

    out = await make_task(_build.bind({}), rollout)(_case())

    assert out["output"]["context_manager"] is False


async def test_a_resource_named_like_a_marker_is_a_setup_error():
    task = make_task(_build.bind({}), _built, environment=lambda case: {"system_prompt": "sneaky"})

    with pytest.raises(_FactoryArgumentError):
        await task(_case())


@pytest.mark.parametrize("kind", ["plain", "context_manager"])
async def test_a_sync_environment_runs_off_the_event_loop(kind):
    loop_thread = threading.get_ident()
    threads = []

    @contextmanager
    def managed():
        threads.append(threading.get_ident())
        yield {"sandbox": "box"}
        threads.append(threading.get_ident())

    def environment(case):
        threads.append(threading.get_ident())
        return {"sandbox": "box"} if kind == "plain" else managed()

    out = await make_task(_build.bind({}), _built, environment=environment)(_case())

    assert out["output"]["sandbox"] == "box"
    assert len(threads) == (1 if kind == "plain" else 3)
    assert loop_thread not in threads


@pytest.mark.parametrize("kind", ["sync", "async"])
async def test_a_context_manager_is_exited_after_the_rollout(kind):
    events = []

    @contextmanager
    def sync_environment(case):
        events.append("enter")
        yield {"sandbox": "box"}
        events.append("exit")

    @asynccontextmanager
    async def async_environment(case):
        events.append("enter")
        yield {"sandbox": "box"}
        events.append("exit")

    async def rollout(agent_factory, case):
        events.append("rollout")
        return {"output": agent_factory()["sandbox"]}

    environment = sync_environment if kind == "sync" else async_environment
    out = await make_task(_build.bind({}), rollout, environment)(_case())

    assert out == {"output": "box"}
    assert events == ["enter", "rollout", "exit"]


@pytest.mark.parametrize("kind", ["bare yield", "async def"])
async def test_an_environment_that_gives_no_mapping_is_a_setup_error(kind, traces):
    exited = []

    @contextmanager
    def bare_yield(case):
        try:
            yield
        finally:
            exited.append(case.name)

    async def async_def(case):
        return {"sandbox": "box"}

    log = TrialLog()
    environment = bare_yield if kind == "bare yield" else async_def
    task = instrument(make_task(_build.bind({}), _built, environment), log, traces)

    with pytest.raises(_FactoryArgumentError, match="not a mapping of resources"):
        await _run(task, _case(), traces)

    assert isinstance(log.setup_error, _FactoryArgumentError)
    assert log.records() == {}
    assert exited == (["case-1"] if kind == "bare yield" else [])


@pytest.mark.parametrize("where", ["call", "enter"])
async def test_an_environment_setup_error_is_infrastructure(where):
    @contextmanager
    def failing_enter(case):
        raise KeyError("no sandbox")
        yield {}

    def failing_call(case):
        raise KeyError("no sandbox")

    environment = failing_call if where == "call" else failing_enter

    with pytest.raises(InfrastructureError) as raised:
        await make_task(_build.bind({}), _built, environment)(_case())

    assert isinstance(raised.value.__cause__, KeyError)


async def test_an_environment_exit_error_only_warns(caplog):
    @contextmanager
    def environment(case):
        yield {"sandbox": "box"}
        raise RuntimeError("teardown failed")

    with caplog.at_level(logging.WARNING):
        out = await make_task(_build.bind({}), _built, environment)(_case())

    assert out["output"]["sandbox"] == "box"
    assert "environment failed to exit" in caplog.text


async def test_a_rollout_error_still_exits_the_environment_and_is_raised():
    seen = []

    @contextmanager
    def environment(case):
        try:
            yield {}
        except ValueError as error:
            seen.append(error)
            raise

    async def rollout(agent_factory, case):
        raise ValueError("agent broke")

    with pytest.raises(ValueError, match="agent broke"):
        await make_task(_build.bind({}), rollout, environment)(_case())

    assert len(seen) == 1


async def test_a_context_manager_cant_swallow_a_rollout_error():
    @contextmanager
    def environment(case):
        try:
            yield {}
        except ValueError:
            pass

    async def rollout(agent_factory, case):
        raise ValueError("agent broke")

    with pytest.raises(ValueError, match="agent broke"):
        await make_task(_build.bind({}), rollout, environment)(_case())


async def test_an_exit_error_after_a_rollout_error_keeps_the_rollout_error(caplog):
    @contextmanager
    def environment(case):
        try:
            yield {}
        finally:
            raise RuntimeError("teardown failed")

    async def rollout(agent_factory, case):
        raise ValueError("agent broke")

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="agent broke"):
        await make_task(_build.bind({}), rollout, environment)(_case())

    assert "environment failed to exit" in caplog.text


# instrument


async def test_a_completed_trial_is_logged(traces):
    log = TrialLog()

    await _run(instrument(make_task(_agent_factory(), default_rollout), log, traces), _case(), traces)

    entry = log.records()["case-1"]
    assert entry.status is TrialStatus.COMPLETED
    assert (entry.error_type, entry.error) == (None, None)
    assert entry.attempts == 1
    assert entry.previous_attempts == []
    assert len(entry.trace_id) == 32
    assert len(entry.attempt_span_id) == 16
    assert log.trace_ids == {entry.trace_id}


async def test_the_default_rollout_gets_the_session_as_trajectory(traces):
    returned = {}

    async def rollout(agent_factory, case):
        returned.update(await default_rollout(agent_factory, case))
        return returned

    out = await _run(instrument(make_task(_agent_factory(), rollout), TrialLog(), traces), _case(), traces)

    assert isinstance(out["trajectory"], Session)
    assert out["output"] == returned["output"]
    assert "trajectory" not in returned


async def test_a_dict_with_trajectory_none_gets_the_session(traces):
    async def rollout(agent_factory, case):
        await default_rollout(agent_factory, case)
        return {"output": "x", "trajectory": None}

    out = await _run(instrument(make_task(_agent_factory(), rollout), TrialLog(), traces), _case(), traces)

    assert isinstance(out["trajectory"], Session)


async def test_a_rollout_returned_trajectory_wins(traces):
    async def rollout(agent_factory, case):
        await default_rollout(agent_factory, case)
        return {"output": "x", "trajectory": ["read_page", "answer"]}

    out = await _run(instrument(make_task(_agent_factory(), rollout), TrialLog(), traces), _case(), traces)

    assert out["trajectory"] == ["read_page", "answer"]


async def test_a_non_dict_return_is_wrapped_with_the_session(traces):
    async def rollout(agent_factory, case):
        return str(await agent_factory().invoke_async(case.input))

    out = await _run(instrument(make_task(_agent_factory("hi"), rollout), TrialLog(), traces), _case(), traces)

    assert out["output"].strip() == "hi"
    assert isinstance(out["trajectory"], Session)


async def test_no_agent_spans_means_no_trajectory(traces):
    async def plain(case):
        return {"output": "no agent here"}

    async def bare(case):
        return "no agent here"

    assert await _run(instrument(plain, TrialLog(), traces), _case(), traces) == {"output": "no agent here"}
    assert await _run(instrument(bare, TrialLog(), traces), _case(), traces) == {
        "output": "no agent here",
        "trajectory": None,
    }


async def test_an_agent_failure_is_logged_and_raised(traces):
    log = TrialLog()

    async def task(case):
        raise EventLoopException(ValueError("bad tool input"))

    with pytest.raises(EventLoopException):
        await _run(instrument(task, log, traces), _case(), traces)

    entry = log.records()["case-1"]
    assert entry.status is TrialStatus.AGENT_FAILURE
    assert entry.error_type == "EventLoopException"


async def test_an_environment_error_records_its_cause(traces):
    log = TrialLog()

    def environment(case):
        raise ConnectionRefusedError("sandbox down")

    task = make_task(_build.bind({}), _built, environment)
    with pytest.raises(InfrastructureError):
        await _run(instrument(task, log, traces), _case(), traces)

    entry = log.records()["case-1"]
    assert entry.status is TrialStatus.INFRASTRUCTURE_INVALID
    assert (entry.error_type, entry.error) == ("ConnectionRefusedError", "sandbox down")


async def test_a_setup_error_is_stored_not_classified(traces):
    log = TrialLog()
    task = make_task(_build.bind({}), _built, environment=lambda case: {"system_prompt": "x"})

    with pytest.raises(_FactoryArgumentError):
        await _run(instrument(task, log, traces), _case(), traces)

    assert isinstance(log.setup_error, _FactoryArgumentError)
    assert log.records() == {}
    assert len(log.trace_ids) == 1  # registered with no entry, so TrialRunner can still release it


async def test_an_unrecorded_attempt_span_is_a_setup_error(traces, monkeypatch):
    unsampled = SDKTracerProvider(sampler=ALWAYS_OFF).get_tracer("optimization-tests")
    monkeypatch.setattr("strands_evals.optimization.execution.rollout.get_tracer", lambda: unsampled)
    log = TrialLog()

    with pytest.raises(RuntimeError, match="aren't recorded"):
        await _run(instrument(make_task(_agent_factory(), default_rollout), log, traces), _case(), traces)

    assert isinstance(log.setup_error, RuntimeError)
    assert log.records() == {}


async def test_a_retried_attempt_moves_into_previous_attempts(traces):
    log = TrialLog()
    calls = []

    async def task(case):
        calls.append(case.name)
        if len(calls) == 1:
            raise ModelThrottledException("slow down")
        return {"output": "ok"}

    wrapped = instrument(task, log, traces)
    with trace_api.get_tracer("optimization-tests").start_as_current_span("execute_case", context=Context()):
        with pytest.raises(ModelThrottledException):
            await wrapped(_case())
        first = log.records()["case-1"]
        await wrapped(_case())

    entry = log.records()["case-1"]
    assert entry.status is TrialStatus.COMPLETED
    assert entry.attempts == 2
    assert entry.trace_id == first.trace_id
    assert entry.attempt_span_id != first.attempt_span_id
    assert entry.previous_attempts == [
        (first.attempt_span_id, TrialStatus.INFRASTRUCTURE_INVALID, "ModelThrottledException", "slow down")
    ]


async def test_a_sync_task_is_instrumented(traces):
    log = TrialLog()

    def task(case):
        return {"output": case.input.upper()}

    out = await _run(instrument(task, log, traces), _case(text="abc"), traces)

    assert out["output"] == "ABC"
    assert log.records()["case-1"].status is TrialStatus.COMPLETED


# Through Experiment


class _SeesSession(Evaluator):
    """Scores 1 when the evaluator receives a `Session` trajectory, like every trace-level evaluator needs."""

    def evaluate(self, evaluation_case):
        is_session = isinstance(evaluation_case.actual_trajectory, Session)
        return [EvaluationOutput(score=float(is_session), test_pass=is_session, reason="")]


async def test_experiment_evaluators_see_the_session(traces):
    log = TrialLog()
    experiment = Experiment(cases=[_case("a"), _case("b")], evaluators=[_SeesSession()])

    report = await experiment.run_evaluations_async(
        instrument(make_task(_agent_factory(), default_rollout), log, traces)
    )

    assert report.scores == [1.0, 1.0]
    entries = log.records()
    assert {name: entry.status for name, entry in entries.items()} == {
        "a": TrialStatus.COMPLETED,
        "b": TrialStatus.COMPLETED,
    }
    assert entries["a"].trace_id != entries["b"].trace_id
    for entry in entries.values():
        session, usage = (
            traces.session(entry.trace_id, entry.attempt_span_id),
            traces.usage(entry.trace_id, entry.attempt_span_id),
        )
        assert isinstance(session, Session)
        assert usage.model_calls == 1


async def test_experiment_throttling_retry_keeps_the_earlier_attempt(traces, no_retry_delay):
    log = TrialLog()
    attempts = []

    # Thrown by the rollout, not the model: Strands' event loop retries a throttled model call itself.
    async def rollout(agent_factory, case):
        attempts.append(case.name)
        if len(attempts) == 1:
            raise ModelThrottledException("slow down")
        return await default_rollout(agent_factory, case)

    experiment = Experiment(cases=[_case()], evaluators=[_SeesSession()])
    await experiment.run_evaluations_async(instrument(make_task(_agent_factory(), rollout), log, traces))

    entry = log.records()["case-1"]
    assert entry.status is TrialStatus.COMPLETED
    assert entry.attempts == 2
    ((span_id, status, error_type, _),) = entry.previous_attempts
    assert status is TrialStatus.INFRASTRUCTURE_INVALID
    assert error_type == "ModelThrottledException"
    assert traces.usage(entry.trace_id, entry.attempt_span_id).model_calls == 1
    assert traces.usage(entry.trace_id, span_id).tokens == 0


async def test_experiment_agent_failure_is_logged(traces):
    log = TrialLog()

    async def rollout(agent_factory, case):
        raise ValueError("bad output")

    experiment = Experiment(cases=[_case()], evaluators=[_SeesSession()])
    await experiment.run_evaluations_async(instrument(make_task(_agent_factory(), rollout), log, traces))

    assert log.records()["case-1"].status is TrialStatus.AGENT_FAILURE


async def test_usage_covers_every_agent_in_a_multi_agent_rollout(traces):
    async def rollout(agent_factory, case):
        planner = agent_factory()
        plan = await planner.invoke_async(case.input)
        # A second agent built outside the factory still lands in the trial's trace.
        writer = Agent(
            model=ScriptedModel(
                ["draft", "final"],
                model_id="writer-model",
                usage={"inputTokens": 100, "outputTokens": 20, "totalTokens": 120, "cacheReadInputTokens": 40},
            ),
            callback_handler=None,
            trace_attributes={"session.id": "writer-session"},  # a different id must not drop either agent
        )
        await writer.invoke_async(str(plan))
        return {"output": str(await writer.invoke_async("polish"))}

    log = TrialLog()
    task = instrument(make_task(_agent_factory(model_id="planner-model"), rollout), log, traces)
    await _run(task, _case(), traces)

    entry = log.records()["case-1"]
    session, usage = (
        traces.session(entry.trace_id, entry.attempt_span_id),
        traces.usage(entry.trace_id, entry.attempt_span_id),
    )
    assert sum(isinstance(span, AgentInvocationSpan) for trace in session.traces for span in trace.spans) == 3
    assert usage.model_calls == 3
    assert usage.tokens == 15 + 120 + 120
    assert usage.output_tokens == 5 + 20 + 20
    assert usage.input_tokens == usage.tokens - usage.output_tokens
    assert usage.cache_read_tokens == 80
    assert usage.cache_write_tokens is None
    assert usage.models == ["planner-model", "writer-model"]
