import asyncio
import contextvars
import logging
import threading
from contextlib import asynccontextmanager, contextmanager

import opentelemetry.trace as trace_api
import pytest
from botocore.exceptions import ProfileNotFound
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from strands import Agent, tool
from strands.multiagent import Swarm
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
from strands_evals.optimization.execution.traces import FACTORY_AGENT, TraceCapture
from strands_evals.types import EvaluationOutput
from strands_evals.types.optimization import CANDIDATE_SPAN_KEY, TrialStatus
from strands_evals.types.trace import AgentInvocationSpan, Session
from tests.strands_evals.optimization.scripted_model import ScriptedModel, ToolUse


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


async def test_make_task_without_environment_builds_through_the_factory():
    built = []

    def factory(**kwargs):
        agent = Agent(model=ScriptedModel(["x"]), callback_handler=None)
        built.append((kwargs, agent))
        return agent

    async def rollout(agent_factory, case):
        return {"output": agent_factory(flavor=case.input)}

    out = await make_task(factory, rollout)(_case(text="hello"))

    assert built == [({"flavor": "hello"}, out["output"])]


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


async def test_a_sync_environment_resets_in_exit_what_it_set_in_enter():
    sandbox = contextvars.ContextVar("sandbox", default=None)
    seen = []

    @contextmanager
    def environment(case):
        token = sandbox.set("box")
        try:
            yield {"sandbox": "box"}
        finally:
            seen.append(sandbox.get())
            sandbox.reset(token)  # raises ValueError if exit runs in another context
            seen.append("reset")

    await make_task(_build.bind({}), _built, environment=environment)(_case())

    assert seen == ["box", "reset"]


async def test_a_span_around_a_sync_environment_detaches_cleanly(caplog):
    @contextmanager
    def environment(case):
        with trace_api.get_tracer("optimization-tests").start_as_current_span("sandbox"):
            yield {"sandbox": "box"}

    with caplog.at_level(logging.ERROR):
        await make_task(_build.bind({}), _built, environment=environment)(_case())

    assert "Failed to detach context" not in caplog.text


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


async def test_a_trial_cancelled_while_a_sync_environment_enters_still_exits_it():
    entering, release = threading.Event(), threading.Event()
    events = []

    class Sandbox:  # not a generator, which would also exit when garbage-collected
        def __enter__(self):
            entering.set()
            release.wait(5)
            events.append("enter")
            return {"sandbox": "box"}

        def __exit__(self, *exc_info):
            events.append("exit")

    task = asyncio.create_task(make_task(_build.bind({}), _built, lambda case: Sandbox())(_case()))
    await asyncio.to_thread(entering.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)  # without the fix, the worker thread is still finishing `__enter__`

    assert events == ["enter", "exit"]


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
    assert entry.usage.model_calls == 1
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


async def test_a_credential_error_building_the_agent_is_infrastructure(traces):
    log = TrialLog()

    def build():
        raise ProfileNotFound(profile="bedrock")

    with pytest.raises(ProfileNotFound):
        await _run(instrument(make_task(build, default_rollout), log, traces), _case(), traces)

    entry = log.records()["case-1"]
    assert entry.status is TrialStatus.INFRASTRUCTURE_INVALID
    assert entry.error_type == "ProfileNotFound"


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
    assert entry.previous_attempts == [
        (first.usage, first.benchmark_usage, TrialStatus.INFRASTRUCTURE_INVALID, "ModelThrottledException", "slow down")
    ]


async def test_an_attempt_that_raised_still_has_its_usage_and_spans_dropped(traces):
    log = TrialLog()

    async def rollout(agent_factory, case):
        await agent_factory().invoke_async(case.input)
        raise ValueError("bad output")

    with pytest.raises(ValueError):
        await _run(instrument(make_task(_agent_factory(), rollout), log, traces), _case(), traces)

    entry = log.records()["case-1"]
    assert entry.status is TrialStatus.AGENT_FAILURE
    assert entry.usage.model_calls == 1
    assert [span.name for span in traces._filter.spans(entry.trace_id)] == ["execute_case"]


async def test_an_error_reading_a_failed_attempt_is_a_setup_error(traces, monkeypatch):
    def unreadable(trace_id, attempt_span_id):
        raise KeyError("unreadable span")

    monkeypatch.setattr(traces, "usage", unreadable)
    log = TrialLog()

    async def task(case):
        raise ModelThrottledException("slow down")

    with pytest.raises(ModelThrottledException):  # the attempt's own error, so Experiment still retries it
        await _run(instrument(task, log, traces), _case(), traces)

    assert isinstance(log.setup_error, KeyError)
    assert log.records() == {}
    (trace_id,) = log.trace_ids
    assert [span.name for span in traces._filter.spans(trace_id)] == ["execute_case"]


async def test_an_error_reading_a_completed_attempt_is_raised(traces, monkeypatch):
    def unreadable(trace_id, attempt_span_id):
        raise KeyError("unreadable span")

    monkeypatch.setattr(traces, "usage", unreadable)
    log = TrialLog()

    with pytest.raises(KeyError):  # so Experiment runs no judge on it
        await _run(instrument(make_task(_agent_factory(), default_rollout), log, traces), _case(), traces)

    assert isinstance(log.setup_error, KeyError)
    assert log.records() == {}
    (trace_id,) = log.trace_ids
    assert [span.name for span in traces._filter.spans(trace_id)] == ["execute_case"]


async def test_a_session_that_cant_be_built_is_raised_with_the_attempt_logged(traces, monkeypatch):
    def unmappable(trace_id, attempt_span_id):
        raise KeyError("unmappable span")

    monkeypatch.setattr(traces, "session", unmappable)
    log = TrialLog()

    with pytest.raises(KeyError):  # Experiment fails the case, so `TrialRunner` records it unmeasured
        await _run(instrument(make_task(_agent_factory(), default_rollout), log, traces), _case(), traces)

    assert log.setup_error is None
    entry = log.records()["case-1"]
    assert entry.status is TrialStatus.COMPLETED
    assert entry.usage.model_calls == 1
    assert [span.name for span in traces._filter.spans(entry.trace_id)] == ["execute_case"]


async def _leave_the_stream_early(agent_factory, case):
    agent = agent_factory()
    async for event in agent.stream_async(case.input):
        if "result" in event:
            break  # the agent's span ends only when the event loop closes the generator, after this returns
    return {"output": "stopped early"}


async def test_spans_that_end_just_after_the_rollout_returned_are_still_counted(traces, caplog):
    log = TrialLog()

    with caplog.at_level(logging.WARNING, logger="strands_evals.optimization.execution.traces"):
        out = await _run(instrument(make_task(_agent_factory(), _leave_the_stream_early), log, traces), _case(), traces)

    entry = log.records()["case-1"]
    assert entry.usage.model_calls == 1
    assert entry.usage.tokens == 15
    assert entry.benchmark_usage.model_calls == 0
    assert sum(isinstance(span, AgentInvocationSpan) for trace in out["trajectory"].traces for span in trace.spans) == 1
    assert "still open" not in caplog.text


async def test_a_span_that_never_ends_is_waited_for_then_warned_about_once(traces, caplog, monkeypatch):
    monkeypatch.setattr("strands_evals.optimization.execution.traces.SETTLE_SECONDS", 0.05)
    left_open = []

    async def rollout(agent_factory, case):
        left_open.append(trace_api.get_tracer("optimization-tests").start_span("background work"))
        return await default_rollout(agent_factory, case)

    log = TrialLog()
    try:
        with caplog.at_level(logging.DEBUG, logger="strands_evals.optimization.execution.traces"):
            for name in ("case-1", "case-2"):
                await _run(instrument(make_task(_agent_factory(), rollout), log, traces), _case(name), traces)
    finally:
        for span in left_open:
            span.end()

    assert [entry.usage.model_calls for entry in log.records().values()] == [1, 1]  # what had ended is counted
    timeouts = [r for r in caplog.records if "open_spans=<1>" in r.getMessage()]
    assert [r.levelno for r in timeouts] == [logging.WARNING, logging.DEBUG]  # every later timeout is still logged


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
        assert entry.usage.model_calls == 1


class _SpansHeld(Evaluator):
    """Records the names of the spans the capture holds for its trial's trace while it judges."""

    def __init__(self, traces):
        super().__init__()
        self.traces = traces
        self.held = []

    def evaluate(self, evaluation_case):
        trace_id = format(trace_api.get_current_span().get_span_context().trace_id, "032x")
        self.held.append(sorted(span.name for span in self.traces._filter.spans(trace_id)))
        return [EvaluationOutput(score=1.0, test_pass=True, reason="")]


async def test_judges_run_after_the_attempts_spans_are_dropped(traces):
    log = TrialLog()
    judge = _SpansHeld(traces)
    experiment = Experiment(cases=[_case()], evaluators=[judge])

    report = await experiment.run_evaluations_async(
        instrument(make_task(_agent_factory(), default_rollout), log, traces)
    )

    assert report.scores == [1.0]
    assert log.records()["case-1"].usage.model_calls == 1
    assert judge.held == [["task_execution case-1"]]  # Experiment's own span: none of the attempt's


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
    ((usage, _, status, error_type, _),) = entry.previous_attempts
    assert status is TrialStatus.INFRASTRUCTURE_INVALID
    assert error_type == "ModelThrottledException"
    assert entry.usage.model_calls == 1
    assert usage.tokens == 0


async def test_experiment_agent_failure_is_logged(traces):
    log = TrialLog()

    async def rollout(agent_factory, case):
        raise ValueError("bad output")

    experiment = Experiment(cases=[_case()], evaluators=[_SeesSession()])
    await experiment.run_evaluations_async(instrument(make_task(_agent_factory(), rollout), log, traces))

    assert log.records()["case-1"].status is TrialStatus.AGENT_FAILURE


def _user(reply="I need more help.", tokens=1000):
    """A simulated user: an agent the rollout builds itself, outside the factory."""
    usage = {"inputTokens": tokens - 20, "outputTokens": 20, "totalTokens": tokens}
    return Agent(model=ScriptedModel([reply], model_id="user-model", usage=usage), callback_handler=None)


async def _attempt(task, traces):
    """Run `task` on one case; return the attempt's usage, benchmark usage, and `Session`."""
    log = TrialLog()
    out = await _run(instrument(task, log, traces), _case(), traces)
    entry = log.records()["case-1"]
    return entry.usage, entry.benchmark_usage, out["trajectory"]


def _marked(session):
    """Each agent invocation's prompt, with whether its span is marked as the candidate's."""
    return [
        (span.user_prompt, span.metadata.get(CANDIDATE_SPAN_KEY))
        for trace in session.traces
        for span in trace.spans
        if isinstance(span, AgentInvocationSpan)
    ]


async def test_usage_counts_only_the_factorys_agents_and_the_rest_is_benchmark_usage(traces):
    async def rollout(agent_factory, case):
        planner = agent_factory()
        plan = await planner.invoke_async(case.input)
        # A second agent built outside the factory is the benchmark's, not the candidate's.
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

    usage, benchmark, session = await _attempt(make_task(_agent_factory(model_id="planner-model"), rollout), traces)

    # The session keeps every agent, for the evaluators; only the planner's spans are the candidate's.
    assert sum(isinstance(span, AgentInvocationSpan) for trace in session.traces for span in trace.spans) == 3
    assert [marked for _, marked in _marked(session)] == [True, False, False]
    assert (usage.model_calls, usage.tokens, usage.output_tokens, usage.models) == (1, 15, 5, ["planner-model"])
    assert usage.seconds > 0
    assert benchmark.model_calls == 2
    assert benchmark.tokens == 120 + 120
    assert benchmark.output_tokens == 20 + 20
    assert benchmark.input_tokens == benchmark.tokens - benchmark.output_tokens
    assert benchmark.cache_read_tokens == 80
    assert benchmark.cache_write_tokens is None
    assert benchmark.models == ["writer-model"]
    assert benchmark.seconds == 0.0


async def test_a_simulated_users_turns_are_benchmark_usage(traces):
    async def conversation(agent_factory, case):
        agent, user = agent_factory(), _user()
        message = case.input
        for _ in range(3):
            message = str(await user.invoke_async(str(await agent.invoke_async(message))))
        return {"output": message}

    usage, benchmark, session = await _attempt(make_task(_agent_factory(), conversation), traces)

    assert (usage.model_calls, usage.tokens) == (3, 3 * 15)
    assert (benchmark.model_calls, benchmark.tokens, benchmark.models) == (3, 3 * 1000, ["user-model"])
    assert [marked for _, marked in _marked(session)] == [True, False] * 3


async def test_a_sync_rollout_building_an_agent_per_turn_splits_the_same_way(traces):
    def conversation(agent_factory, case):  # runs in a worker thread
        message = case.input
        for _ in range(2):
            message = str(_user()(str(agent_factory()(message))))
        return {"output": message}

    usage, benchmark, _ = await _attempt(make_task(_agent_factory(), conversation), traces)

    assert (usage.model_calls, usage.tokens) == (2, 2 * 15)
    assert (benchmark.model_calls, benchmark.tokens) == (2, 2 * 1000)


@tunable(system_prompt=TextValue(default="be brief"))
def _tunable_agent(system_prompt):
    return Agent(model=ScriptedModel(["answer"]), system_prompt=system_prompt, callback_handler=None)


async def test_an_agent_the_environment_runs_is_the_benchmarks(traces):
    @asynccontextmanager
    async def environment(case):
        await _user("a profile").invoke_async(f"describe a customer for {case.name}")  # before the rollout
        yield {}

    task = make_task(_tunable_agent.bind({}), default_rollout, environment)
    usage, benchmark, session = await _attempt(task, traces)

    assert (usage.model_calls, usage.tokens) == (1, 15)
    assert (benchmark.model_calls, benchmark.tokens) == (1, 1000)
    assert [marked for _, marked in _marked(session)] == [False, True]


async def test_a_sub_agent_called_from_the_factorys_agent_is_the_candidates(traces):
    @tool
    def ask_expert(question: str) -> str:
        """Ask the expert."""
        return str(_user("the expert says yes", tokens=200)(question))

    def factory():
        model = ScriptedModel([ToolUse("ask_expert", {"question": "is it?"}), "done"])
        return Agent(model=model, tools=[ask_expert], callback_handler=None)

    usage, benchmark, session = await _attempt(make_task(factory, default_rollout), traces)

    assert (usage.model_calls, usage.tokens, usage.models) == (3, 15 + 200 + 15, ["scripted-model", "user-model"])
    assert benchmark == benchmark.zero()
    assert [marked for _, marked in _marked(session)] == [True, True]


async def test_an_async_factorys_agent_is_tagged_once_awaited(traces):
    async def factory():
        return Agent(model=ScriptedModel(["hi"]), callback_handler=None)

    async def rollout(agent_factory, case):
        agent = await agent_factory()
        await _user().invoke_async(str(await agent.invoke_async(case.input)))
        return {"output": "x"}

    usage, benchmark, _ = await _attempt(make_task(factory, rollout), traces)

    assert (usage.tokens, benchmark.tokens) == (15, 1000)


async def test_a_factory_returning_a_task_is_tagged_once_awaited(traces):
    async def build_agent():
        return Agent(model=ScriptedModel(["hi"]), callback_handler=None)

    def factory():
        return asyncio.ensure_future(build_agent())  # awaitable, yet not a coroutine

    async def rollout(agent_factory, case):
        agent = await agent_factory()
        await _user().invoke_async(str(await agent.invoke_async(case.input)))
        return {"output": "x"}

    usage, benchmark, _ = await _attempt(make_task(factory, rollout), traces)

    assert (usage.tokens, benchmark.tokens) == (15, 1000)


async def test_a_multi_agent_system_the_factory_builds_is_the_candidates(traces):
    def factory():
        return Swarm([Agent(name="solo", model=ScriptedModel(["done"]), callback_handler=None)])

    async def rollout(agent_factory, case):
        swarm = agent_factory()
        assert swarm.trace_attributes[FACTORY_AGENT] is True
        await swarm.invoke_async(case.input)
        await _user().invoke_async("thanks")
        return {"output": "x"}

    usage, benchmark, _ = await _attempt(make_task(factory, rollout), traces)

    assert (usage.model_calls, usage.tokens) == (1, 15)
    assert benchmark.tokens == 1000


@pytest.mark.parametrize("builds", [True, False], ids=["built", "never-built"])
async def test_a_factory_agent_that_never_runs_spends_nothing(traces, builds):
    async def rollout(agent_factory, case):
        if builds:
            agent_factory()
        await _user().invoke_async(case.input)
        return {"output": "the user gave up"}

    usage, benchmark, session = await _attempt(make_task(_agent_factory(), rollout), traces)

    assert usage.model_calls == 0
    assert benchmark.tokens == 1000
    assert _marked(session) == [("question", False)]


async def test_an_untaggable_build_counts_every_agent_with_one_warning(traces, caplog, monkeypatch):
    monkeypatch.setattr("strands_evals.optimization.execution.rollout._UNTAGGED_WARNED", set())

    def factory():
        return {"system_prompt": "be brief"}  # the rollout builds the agent from what the factory returns

    async def rollout(agent_factory, case):
        config = agent_factory()
        agent = Agent(model=ScriptedModel(["hi"]), callback_handler=None, **config)
        await _user().invoke_async(str(await agent.invoke_async(case.input)))
        return {"output": "x"}

    with caplog.at_level(logging.WARNING):
        usage, benchmark, session = await _attempt(make_task(factory, rollout), traces)
        await _attempt(make_task(factory, rollout), traces)

    assert (usage.tokens, benchmark.tokens) == (15 + 1000, 0)
    assert caplog.text.count("built=<dict>") == 1
    assert [marked for _, marked in _marked(session)] == [None, None]


async def test_one_untaggable_build_among_tagged_ones_counts_every_agent(traces, monkeypatch):
    monkeypatch.setattr("strands_evals.optimization.execution.rollout._UNTAGGED_WARNED", set())

    def factory(part="agent"):
        agent = Agent(model=ScriptedModel(["hi"]), callback_handler=None)
        return agent if part == "agent" else {"agent": agent}

    def rollout(agent_factory, case):  # sync: the rollout span is marked from a worker thread
        reply = agent_factory()(case.input)
        _user()(str(agent_factory(part="wrapped")["agent"](str(reply))))
        return {"output": "x"}

    usage, benchmark, session = await _attempt(make_task(factory, rollout), traces)

    assert (usage.model_calls, usage.tokens, benchmark.tokens) == (3, 15 + 15 + 1000, 0)
    assert all(marked is None for _, marked in _marked(session))


async def test_a_build_task_task_counts_every_agent(traces):
    async def task(case):  # no rollout span: the agent is built outside any factory
        agent = Agent(model=ScriptedModel(["hi"]), callback_handler=None)
        await _user().invoke_async(str(await agent.invoke_async(case.input)))
        return {"output": "x"}

    usage, benchmark, session = await _attempt(task, traces)

    assert (usage.tokens, benchmark.tokens) == (15 + 1000, 0)
    assert all(CANDIDATE_SPAN_KEY not in span.metadata for trace in session.traces for span in trace.spans)
