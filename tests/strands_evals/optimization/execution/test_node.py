"""`NodeAgent` and `traces.ingest`, on a reply the Node runner really wrote for a strands-ts agent (no Node needed).

`strands_ts_reply.json` is `runner.mjs`'s reply to a `run` request for a harness agent on a scripted model
(@strands-agents/harness 0.5.0, @strands-agents/sdk 1.26.0), traced under trace `0123…cdef`, parent span
`0123456789abcdef`; each test puts its spans in its own trial's trace.
"""

import json
from pathlib import Path

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.context import Context

from strands_evals.optimization.execution.node import NodeAgent, NodeError, NodeRunnerGone
from strands_evals.optimization.execution.rollout import _tagging
from strands_evals.optimization.execution.status import InfrastructureError, classify
from strands_evals.optimization.execution.traces import (
    ATTEMPT_SPAN,
    FACTORY_AGENT,
    FACTORY_TAGGED,
    ROLLOUT_SPAN,
    SPAN_KIND,
    TraceCapture,
    ingest,
)
from strands_evals.types.optimization import CANDIDATE_SPAN_KEY, TrialStatus
from strands_evals.types.trace import AgentInvocationSpan, InferenceSpan

REPLY = json.loads((Path(__file__).parent / "strands_ts_reply.json").read_text())
RECORDED_TRACE, RECORDED_PARENT = "0123456789abcdef0123456789abcdef", "0123456789abcdef"


class FakeRunner:
    """Answers `run` with the recorded reply, its spans moved into the request's trace under the request's span."""

    def __init__(self, **reply):
        self.reply = {**REPLY, **reply}
        self.requests, self.late = [], []

    async def call_async(self, message, on_late=None):
        self.requests.append(message)
        self.late.append(on_late)
        text = json.dumps(self.reply)
        if "trace_id" in message:
            text = text.replace(RECORDED_TRACE, message["trace_id"]).replace(RECORDED_PARENT, message["span_id"])
        return json.loads(text)


class GoneRunner:
    async def call_async(self, message, on_late=None):
        raise NodeRunnerGone("the Node agent runner exited with code 1")


@pytest.fixture
def traces():
    return TraceCapture()


@pytest.fixture
def tracer(traces):
    return trace_api.get_tracer("optimization-tests")


async def _invoke(tracer, traces, agent, *, register=True):
    """Invoke `agent` inside a trial's attempt and rollout spans, as a factory-path trial does; returns
    `(trace_id, attempt_span_id, output or the exception)`."""
    with tracer.start_as_current_span("execute_case case", context=Context()) as root:
        trace_id = format(root.get_span_context().trace_id, "032x")
        if register:
            traces.register(trace_id)
        with tracer.start_as_current_span("attempt case", attributes={SPAN_KIND: ATTEMPT_SPAN}) as attempt:
            attempt_id = format(attempt.get_span_context().span_id, "016x")
            rollout = {SPAN_KIND: ROLLOUT_SPAN, FACTORY_TAGGED: True}
            with tracer.start_as_current_span("rollout case", attributes=rollout) as span:
                built = _tagging(lambda: agent, span)()
                try:
                    outcome = await built.invoke_async("question 7")
                except Exception as error:
                    outcome = error
    return trace_id, attempt_id, outcome


async def test_a_node_agents_spans_read_as_a_python_agents(traces, tracer):
    runner = FakeRunner()
    agent = NodeAgent(runner, "agent", 0, [(["instructions"], "Be exact.")])
    trace_id, attempt_id, output = await _invoke(tracer, traces, agent)

    assert output == "answer 7"
    (request,) = runner.requests
    assert request["trace_id"] == trace_id  # the agent runs in the trial's trace
    assert request["trace_attributes"] == {FACTORY_AGENT: True}  # tagged as the factory's agent
    assert request["values"] == [(["instructions"], "Be exact.")] and request["input"] == "question 7"

    usage = traces.usage(trace_id, attempt_id)
    assert (usage.tokens, usage.input_tokens, usage.output_tokens, usage.model_calls) == (700, 680, 20, 1)
    assert usage.models == ["scripted-deterministic"] and usage.cache_read_tokens is None
    assert traces.benchmark_usage(trace_id, attempt_id).tokens == 0

    session = traces.session(trace_id, attempt_id)
    spans = session.traces[0].spans
    assert [type(span) for span in spans] == [InferenceSpan, AgentInvocationSpan]
    invocation = spans[1]
    assert (invocation.user_prompt, invocation.agent_response) == ("question 7", "answer 7")
    assert "Be brief" not in invocation.system_prompt and invocation.system_prompt.startswith("You are an agent")
    assert all(span.metadata[CANDIDATE_SPAN_KEY] for span in spans)


async def test_spans_of_an_unregistered_trace_are_dropped(traces, tracer):
    trace_id, attempt_id, _ = await _invoke(tracer, traces, NodeAgent(FakeRunner(), "agent", 0, []), register=False)
    traces.register(trace_id)
    assert traces.usage(trace_id, attempt_id).tokens == 0


async def test_an_agent_failure_keeps_the_javascript_type_and_its_spend(traces, tracer):
    runner = FakeRunner(status="agent_failure", error_type="ValidationError", error="bad tool input", output=None)
    trace_id, attempt_id, error = await _invoke(tracer, traces, NodeAgent(runner, "agent", 0, []))
    assert isinstance(error, NodeError) and type(error).__name__ == "ValidationError" and str(error) == "bad tool input"
    assert classify(error) is TrialStatus.AGENT_FAILURE
    assert traces.usage(trace_id, attempt_id).tokens == 700  # a crash's tokens are real spend


async def test_an_infrastructure_failure_is_an_infrastructure_error(traces, tracer):
    runner = FakeRunner(status="infrastructure_invalid", error_type="ModelThrottledError", error="slow down")
    _, _, error = await _invoke(tracer, traces, NodeAgent(runner, "agent", 0, []))
    assert isinstance(error, InfrastructureError) and type(error.__cause__).__name__ == "ModelThrottledError"
    assert classify(error) is TrialStatus.INFRASTRUCTURE_INVALID


async def test_a_runner_that_exited_is_infrastructure(traces, tracer):
    _, _, error = await _invoke(tracer, traces, NodeAgent(GoneRunner(), "agent", 0, []))
    assert isinstance(error, InfrastructureError) and classify(error) is TrialStatus.INFRASTRUCTURE_INVALID


class SessionRunner(FakeRunner):
    """Answers `start` / `turn` / `release` as the runner would; turns reply with the recorded run reply."""

    def __init__(self, start_reply=None, **reply):
        super().__init__(**reply)
        self.start_reply = start_reply
        self.opened = 0

    async def call_async(self, message, on_late=None):
        if message.get("type") == "start":
            self.requests.append(message)
            self.opened += 1
            return self.start_reply or {"ok": True, "session": f"s{self.opened}"}
        if message.get("type") == "release":
            self.requests.append(message)
            return {"ok": True}
        return await super().call_async(message, on_late)


async def _converse(tracer, traces, agent, prompts):
    """Run one session over `prompts` inside a trial's spans; returns `(trace_id, attempt_span_id, outcomes)`."""
    outcomes = []
    with tracer.start_as_current_span("execute_case case", context=Context()) as root:
        trace_id = format(root.get_span_context().trace_id, "032x")
        traces.register(trace_id)
        with tracer.start_as_current_span("attempt case", attributes={SPAN_KIND: ATTEMPT_SPAN}) as attempt:
            attempt_id = format(attempt.get_span_context().span_id, "016x")
            rollout = {SPAN_KIND: ROLLOUT_SPAN, FACTORY_TAGGED: True}
            with tracer.start_as_current_span("rollout case", attributes=rollout) as span:
                built = _tagging(lambda: agent, span)()
                async with built.session() as session:
                    for prompt in prompts:
                        try:
                            outcomes.append(await session.turn(prompt))
                        except Exception as error:
                            outcomes.append(error)
    return trace_id, attempt_id, outcomes


async def test_a_session_asks_one_held_agent_turn_after_turn(traces, tracer):
    runner = SessionRunner(result={"stop_reason": "endTurn"})
    agent = NodeAgent(runner, "agent", 0, [(["instructions"], "Be exact.")])
    agent.trace_attributes["tagged"] = True
    trace_id, attempt_id, outcomes = await _converse(tracer, traces, agent, ["question 7", "question 7 again"])

    assert outcomes == ["answer 7", "answer 7"]
    start, first, second, release = runner.requests
    assert start["type"] == "start" and start["values"] == [(["instructions"], "Be exact.")]
    assert start["trace_attributes"]["tagged"] is True  # the factory's tag reaches the one build
    assert [first["type"], second["type"]] == ["turn", "turn"]
    assert first["session"] == second["session"] == "s1"  # the same held agent answers both
    assert "values" not in first and first["trace_id"] == trace_id  # a turn carries the trial's trace, not a build
    assert release == {"type": "release", "session": "s1"}
    assert traces.usage(trace_id, attempt_id).tokens == 1400  # both turns' spans land in the trial


async def test_a_sessions_last_result_is_the_turns_structured_result(traces, tracer):
    runner = SessionRunner(result={"stop_reason": "endTurn", "structured_output": {"n": 7}})
    _, _, outcomes = await _converse(tracer, traces, NodeAgent(runner, "agent", 0, []), ["question 7"])
    assert outcomes == ["answer 7"]
    # The session object was context-managed away; read the result off a fresh session instead.
    async with NodeAgent(runner, "agent", 0, []).session() as session:
        assert await session.turn("question 7") == "answer 7"
        assert session.last_result == {"stop_reason": "endTurn", "structured_output": {"n": 7}}


async def test_a_start_the_values_break_raises_and_opens_nothing():
    broken = {"ok": False, "status": "agent_failure", "error_type": "ValidationError", "error": "bad effort"}
    runner = SessionRunner(start_reply=broken)
    with pytest.raises(NodeError) as caught:
        async with NodeAgent(runner, "agent", 0, []).session():
            pass
    assert type(caught.value).__name__ == "ValidationError" and str(caught.value) == "bad effort"
    assert [request["type"] for request in runner.requests] == ["start"]  # nothing to release

    runner = SessionRunner(start_reply={**broken, "status": "infrastructure_invalid"})
    with pytest.raises(InfrastructureError):
        async with NodeAgent(runner, "agent", 0, []).session():
            pass


async def test_a_failed_turn_raises_and_the_session_is_still_released(traces, tracer):
    runner = SessionRunner(status="agent_failure", error_type="ValidationError", error="bad tool input", output=None)
    _, _, (error,) = await _converse(tracer, traces, NodeAgent(runner, "agent", 0, []), ["question 7"])
    assert isinstance(error, NodeError) and classify(error) is TrialStatus.AGENT_FAILURE
    assert runner.requests[-1]["type"] == "release"  # leaving the `async with` still lets the agent go


async def test_a_worker_that_dies_mid_session_is_infrastructure():
    class DiesOnTurn(SessionRunner):
        async def call_async(self, message, on_late=None):
            if message.get("type") in ("turn", "release"):
                raise NodeRunnerGone("the Node agent runner exited with code 1")
            return await super().call_async(message, on_late)

    agent = NodeAgent(DiesOnTurn(), "agent", 0, [])
    with pytest.raises(InfrastructureError):
        # The dead worker also makes `release` moot: leaving the session must not raise over the turn's error.
        async with agent.session() as session:
            await session.turn("question 7")


async def test_a_start_is_traced_and_its_spans_land_in_the_trial(traces, tracer):
    class SpansOnStart(SessionRunner):
        async def call_async(self, message, on_late=None):
            if message.get("type") != "start":
                return await super().call_async(message, on_late)
            self.requests.append(message)
            spans = json.dumps(REPLY["spans"])
            spans = spans.replace(RECORDED_TRACE, message["trace_id"]).replace(RECORDED_PARENT, message["span_id"])
            return {"ok": True, "session": "s1", "spans": json.loads(spans)}

    runner = SpansOnStart()
    trace_id, attempt_id, outcomes = await _converse(tracer, traces, NodeAgent(runner, "agent", 0, []), [])
    assert outcomes == []
    start = runner.requests[0]
    assert start["type"] == "start" and start["trace_id"] == trace_id  # the build runs in the trial's trace
    assert traces.usage(trace_id, attempt_id).tokens == 700  # and what it spent counts


async def test_invoke_async_keeps_the_answers_structured_result(traces, tracer):
    agent = NodeAgent(FakeRunner(result={"stop_reason": "endTurn"}), "agent", 0, [])
    assert agent.last_result is None
    _, _, output = await _invoke(tracer, traces, agent)
    assert output == "answer 7" and agent.last_result == {"stop_reason": "endTurn"}


async def test_every_traced_call_reads_its_late_reply_for_spans(traces, tracer):
    """A cancelled run, start, or turn still has its spans read off the reply the worker sends late."""
    runner = SessionRunner()
    agent = NodeAgent(runner, "agent", 0, [])
    await _invoke(tracer, traces, agent)
    trace_id, attempt_id, _ = await _converse(tracer, traces, agent, ["question 7"])
    assert [r["type"] for r in runner.requests] == ["run", "start", "turn", "release"]
    run_late, turn_late = runner.late[0], runner.late[1]  # SessionRunner records start / release itself
    assert run_late is turn_late is not None
    before = traces.usage(trace_id, attempt_id).tokens
    spans = json.loads(
        json.dumps(REPLY["spans"])
        .replace(RECORDED_TRACE, trace_id)
        .replace(RECORDED_PARENT, runner.requests[2]["span_id"])
        .replace('"span_id": "', '"span_id": "f')  # another call's spans, not the turn's own
    )
    turn_late({"status": "agent_failure", "spans": spans})
    assert traces.usage(trace_id, attempt_id).tokens == before + 700


async def test_a_worker_gone_at_start_is_infrastructure():
    with pytest.raises(InfrastructureError):
        async with NodeAgent(GoneRunner(), "agent", 0, []).session():
            pass


async def test_a_turn_outside_the_session_says_to_enter_it():
    with pytest.raises(RuntimeError, match="enter it first"):
        await NodeAgent(SessionRunner(), "agent", 0, []).session().turn("question 7")


def test_ingest_rebuilds_every_field_the_readers_use(traces, tracer):
    with tracer.start_as_current_span("execute_case case", context=Context()) as root:
        trace_id = format(root.get_span_context().trace_id, "032x")
        traces.register(trace_id)
        parent = format(root.get_span_context().span_id, "016x")
    ingest(json.loads(json.dumps(REPLY["spans"]).replace(RECORDED_TRACE, trace_id).replace(RECORDED_PARENT, parent)))
    (chat,) = [span for span in traces._filter.spans(trace_id) if span.name == "chat"]
    assert chat.attributes["gen_ai.usage.total_tokens"] == 700
    assert [event.name for event in chat.events] == ["gen_ai.system.message", "gen_ai.user.message", "gen_ai.choice"]
    assert chat.end_time > chat.start_time > 1_700_000_000 * 10**9  # nanoseconds, kept exact as strings
    assert chat.instrumentation_scope is not None
