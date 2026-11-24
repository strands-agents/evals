import json
import logging
from contextlib import contextmanager

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.context import Context
from opentelemetry.sdk.trace import SpanLimits
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from strands import Agent
from strands.telemetry.tracer import get_tracer as strands_tracer
from strands.types.exceptions import ModelThrottledException

from strands_evals.optimization.execution.traces import (
    _FILTERS,
    ATTEMPT_SPAN,
    FACTORY_AGENT,
    FACTORY_TAGGED,
    ROLLOUT_SPAN,
    SPAN_KIND,
    TraceCapture,
)
from strands_evals.types.optimization import CANDIDATE_SPAN_KEY
from strands_evals.types.trace import AgentInvocationSpan, InferenceSpan, Session
from tests.strands_evals.optimization.scripted_model import ScriptedModel


@pytest.fixture
def traces():
    return TraceCapture()


@pytest.fixture
def tracer(traces):
    return trace_api.get_tracer("optimization-tests")


def _ids(span):
    context = span.get_span_context()
    return format(context.trace_id, "032x"), format(context.span_id, "016x")


@contextmanager
def _trial(tracer, traces, *, register=True):
    """An `execute_case`-like root span in an empty context; yields its trace id."""
    with tracer.start_as_current_span("execute_case case", context=Context()) as root:
        trace_id, _ = _ids(root)
        if register:
            traces.register(trace_id)
        yield trace_id


def _chat(tracer, *, total=15, output=5, model="model-a", **extra):
    attributes = {
        "gen_ai.operation.name": "chat",
        "gen_ai.usage.total_tokens": total,
        "gen_ai.usage.input_tokens": total - output,
        "gen_ai.usage.output_tokens": output,
        **extra,
    }
    if model is not None:
        attributes["gen_ai.request.model"] = model
    with tracer.start_as_current_span("chat", attributes=attributes):
        pass


def _attempt(tracer):
    return tracer.start_as_current_span("attempt case", attributes={SPAN_KIND: ATTEMPT_SPAN})


def _rollout(tracer):
    return tracer.start_as_current_span("rollout case", attributes={SPAN_KIND: ROLLOUT_SPAN})


@contextmanager
def _timed(tracer, name, start, end, **attributes):
    """A current span with explicit start and end times, in seconds.

    Durations of real spans can be 0: before Python 3.13, Windows' clock ticks every ~15.6 ms.
    """
    span = tracer.start_span(name, attributes=attributes, start_time=int(start * 1e9))
    try:
        with trace_api.use_span(span, end_on_exit=False):  # records an exception and sets ERROR, as usual
            yield span
    finally:
        span.end(end_time=int(end * 1e9))


def test_spans_of_unregistered_traces_are_dropped(traces, tracer):
    with _trial(tracer, traces, register=False) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(tracer)

    session, usage = traces.session(trace_id, _ids(attempt)[1]), traces.usage(trace_id, _ids(attempt)[1])

    assert session is None
    assert usage.model_calls == 0


def test_usage_reads_only_the_attempts_subtree(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer):
            _chat(tracer, total=100, output=10, model="first-attempt")
        with _attempt(tracer) as last:
            with _rollout(tracer):
                _chat(tracer, total=15, output=5, model="model-a")
                _chat(tracer, total=30, output=8, model="model-b")
                _chat(tracer, total=20, output=2, model="model-a")
        with tracer.start_as_current_span("evaluator Judge", attributes={"gen_ai.evaluation.name": "Judge"}):
            _chat(tracer, total=1000, output=100, model="judge")

    usage = traces.usage(trace_id, _ids(last)[1])

    assert usage.model_calls == 3
    assert usage.tokens == 65
    assert usage.output_tokens == 15
    assert usage.input_tokens == 50
    assert usage.models == ["model-a", "model-b"]


def test_usage_ignores_invoke_agent_running_totals(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            with tracer.start_as_current_span(
                "invoke_agent", attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.usage.total_tokens": 999}
            ):
                _chat(tracer, total=15, output=5)

    usage = traces.usage(trace_id, _ids(attempt)[1])

    assert usage.tokens == 15


def test_input_tokens_include_cache_tokens(traces, tracer):
    # On some SDK versions the span's input_tokens leaves cache tokens out; total_tokens always has them.
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(
                tracer,
                total=150,
                output=10,
                **{"gen_ai.usage.input_tokens": 40, "gen_ai.usage.cache_read.input_tokens": 100},
            )

    usage = traces.usage(trace_id, _ids(attempt)[1])

    assert usage.input_tokens == 140
    assert usage.cache_read_tokens == 100


def test_cache_tokens_unmeasured_when_no_span_reports_them(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(tracer)

    usage = traces.usage(trace_id, _ids(attempt)[1])

    assert usage.cache_read_tokens is None
    assert usage.cache_write_tokens is None


def test_cache_tokens_read_from_either_attribute_name(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(tracer, **{"gen_ai.usage.cache_read_input_tokens": 3, "gen_ai.usage.cache_write_input_tokens": 4})
            _chat(tracer, **{"gen_ai.usage.cache_creation.input_tokens": 0})

    usage = traces.usage(trace_id, _ids(attempt)[1])

    assert usage.cache_read_tokens == 3
    assert usage.cache_write_tokens == 4


def test_spans_without_a_model_id_add_no_model(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(tracer, model=None)

    usage = traces.usage(trace_id, _ids(attempt)[1])

    assert usage.models == []


def test_a_rollout_whose_factory_built_an_agent_splits_its_spans_from_the_benchmarks(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _timed(tracer, "attempt case", 100.0, 110.0, **{SPAN_KIND: ATTEMPT_SPAN}) as attempt:
            _chat(tracer, total=7, output=1, model="environment")  # environment setup, outside the rollout
            with _timed(tracer, "rollout case", 101.0, 109.0, **{SPAN_KIND: ROLLOUT_SPAN, FACTORY_TAGGED: True}):
                agent = {"gen_ai.operation.name": "invoke_agent", FACTORY_AGENT: True}
                with tracer.start_as_current_span("invoke_agent", attributes=agent):
                    _chat(tracer, total=15, output=5, model="agent")
                    with tracer.start_as_current_span("execute_tool"):
                        _chat(tracer, total=30, output=6, model="sub-agent")  # no tag: under the agent
                _chat(tracer, total=1000, output=20, model="user")

    usage, benchmark = traces.usage(trace_id, _ids(attempt)[1]), traces.benchmark_usage(trace_id, _ids(attempt)[1])

    assert (usage.model_calls, usage.tokens, usage.output_tokens) == (2, 45, 11)
    assert usage.models == ["agent", "sub-agent"]
    assert usage.seconds == pytest.approx(8.0)  # the rollout's duration, as before
    assert (benchmark.model_calls, benchmark.tokens, benchmark.output_tokens) == (2, 1007, 21)
    assert benchmark.models == ["environment", "user"]
    assert benchmark.seconds == 0.0


def test_a_rollout_whose_agents_cant_be_told_apart_is_all_the_candidates(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            with tracer.start_as_current_span(
                "rollout case", attributes={SPAN_KIND: ROLLOUT_SPAN, FACTORY_TAGGED: False}
            ):
                _chat(tracer, total=15)
                _chat(tracer, total=1000)

    usage, benchmark = traces.usage(trace_id, _ids(attempt)[1]), traces.benchmark_usage(trace_id, _ids(attempt)[1])

    assert (usage.tokens, benchmark) == (1015, benchmark.zero())


def test_seconds_come_from_the_rollout_span(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _timed(tracer, "attempt case", 100.0, 110.0, **{SPAN_KIND: ATTEMPT_SPAN}) as attempt:
            with _timed(tracer, "rollout case", 102.0, 105.5, **{SPAN_KIND: ROLLOUT_SPAN}):
                _chat(tracer)

    usage = traces.usage(trace_id, _ids(attempt)[1])

    assert usage.seconds == pytest.approx(3.5)


def test_seconds_fall_back_to_the_attempt_span(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _timed(tracer, "attempt case", 100.0, 104.0, **{SPAN_KIND: ATTEMPT_SPAN}) as attempt:
            _chat(tracer)

    usage = traces.usage(trace_id, _ids(attempt)[1])

    assert usage.seconds == pytest.approx(4.0)


async def test_session_maps_agent_spans(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            agent = Agent(model=ScriptedModel(["hello there"]), callback_handler=None)
            await agent.invoke_async("hi")

    session, usage = traces.session(trace_id, _ids(attempt)[1]), traces.usage(trace_id, _ids(attempt)[1])

    assert isinstance(session, Session)
    assert [trace.trace_id for trace in session.traces] == [trace_id]
    assert usage.model_calls == 1
    assert usage.tokens == 15
    assert usage.models == ["scripted-model"]


async def test_session_reads_the_convention_from_a_strands_span(traces, tracer, monkeypatch):
    monkeypatch.setattr(strands_tracer(), "use_latest_genai_conventions", True)
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            with tracer.start_as_current_span("rollout's own setup"):  # finishes first, no gen_ai attributes
                pass
            await Agent(model=ScriptedModel(["hello there"]), callback_handler=None).invoke_async("hi")

    session = traces.session(trace_id, _ids(attempt)[1])

    assert session is not None
    invocations = [span for span in session.traces[0].spans if isinstance(span, AgentInvocationSpan)]
    assert [(span.user_prompt, span.agent_response.strip()) for span in invocations] == [("hi", "hello there")]
    assert any(isinstance(span, InferenceSpan) for span in session.traces[0].spans)


async def test_session_marks_which_agents_are_the_candidates(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            flagged = {SPAN_KIND: ROLLOUT_SPAN, FACTORY_TAGGED: True}
            with tracer.start_as_current_span("rollout case", attributes=flagged):
                tagged = Agent(model=ScriptedModel(["mine"]), callback_handler=None)
                tagged.trace_attributes[FACTORY_AGENT] = True  # as `make_task` tags what the factory builds
                await tagged.invoke_async("to the candidate")
                await Agent(model=ScriptedModel(["theirs"]), callback_handler=None).invoke_async("to the user")

    spans = traces.session(trace_id, _ids(attempt)[1]).traces[0].spans

    agents = [(s.user_prompt, s.metadata[CANDIDATE_SPAN_KEY]) for s in spans if isinstance(s, AgentInvocationSpan)]
    calls = [
        (s.messages[0].content[0].text, s.metadata[CANDIDATE_SPAN_KEY]) for s in spans if isinstance(s, InferenceSpan)
    ]
    assert agents == [("to the candidate", True), ("to the user", False)]
    assert calls == [("to the candidate", True), ("to the user", False)]


def test_no_agent_spans_means_no_session(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            pass

    session = traces.session(trace_id, _ids(attempt)[1])

    assert session is None


def test_evaluator_view_reads_judge_usage_errors_and_labels(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer):
            _chat(tracer, total=500, output=50, model="agent-model")
        with _timed(
            tracer,
            "evaluator Helpfulness",
            200.0,
            201.0,
            **{"gen_ai.evaluation.name": "Helpfulness", "gen_ai.evaluation.score.label": "Very helpful"},
        ):
            _chat(tracer, total=30, output=3, model="judge-model")
        with pytest.raises(ConnectionError):
            with _timed(tracer, "evaluator Broken", 201.0, 202.5, **{"gen_ai.evaluation.name": "Broken"}):
                _chat(tracer, total=12, output=2, model="judge-model")
                raise ConnectionError("judge unreachable")

    view = traces.evaluator_view(trace_id)

    assert view.labels == {"Helpfulness": "Very helpful"}
    assert view.errors == {"Broken": "ConnectionError"}
    assert set(view.judge_usage) == {"Helpfulness", "Broken"}
    assert view.judge_usage["Helpfulness"].tokens == 30
    assert view.judge_usage["Broken"].tokens == 12
    assert view.judge_usage["Helpfulness"].model_calls == 1
    assert view.judge_usage["Helpfulness"].models == ["judge-model"]
    assert view.judge_usage["Helpfulness"].seconds == pytest.approx(1.0)
    assert view.judge_usage["Broken"].seconds == pytest.approx(1.5)


def test_evaluator_view_with_no_evaluators_is_empty(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        pass

    view = traces.evaluator_view(trace_id)

    assert view.errors == {}
    assert view.labels == {}
    assert view.judge_usage == {}


def test_case_error_reads_the_exception_recorded_on_the_case_span(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with pytest.raises(ConnectionError):
            with tracer.start_as_current_span("task_execution case"):  # a nested span's exception isn't read
                raise ConnectionError("unreachable")
        trace_api.get_current_span().record_exception(ModelThrottledException("slow down"))
    with _trial(tracer, traces) as clean_id:
        pass

    assert traces.case_error(trace_id) == ("ModelThrottledException", "slow down")  # no module path
    assert traces.case_error(clean_id) == (None, None)


def test_release_drops_only_those_trials(traces, tracer):
    with _trial(tracer, traces) as first:
        with _attempt(tracer) as first_attempt:
            _chat(tracer)
    with _trial(tracer, traces) as second:
        with _attempt(tracer) as second_attempt:
            _chat(tracer)

    traces.release([first])

    assert traces.usage(first, _ids(first_attempt)[1]).model_calls == 0
    assert traces.usage(second, _ids(second_attempt)[1]).model_calls == 1


def test_a_released_trace_keeps_nothing_that_finishes_later(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        traces.release([trace_id])
        with _attempt(tracer) as attempt:
            _chat(tracer)

    assert traces.usage(trace_id, _ids(attempt)[1]).model_calls == 0


def test_drop_attempt_drops_the_attempt_and_keeps_the_judges(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(tracer, total=500, output=50)
        traces.drop_attempt(trace_id, _ids(attempt)[1])
        with tracer.start_as_current_span(
            "evaluator Helpfulness", attributes={"gen_ai.evaluation.name": "Helpfulness"}
        ):
            _chat(tracer, total=30, output=3)

    assert traces.usage(trace_id, _ids(attempt)[1]).model_calls == 0
    assert traces.session(trace_id, _ids(attempt)[1]) is None
    assert [span.name for span in traces._filter.spans(trace_id)] == [
        "chat",
        "evaluator Helpfulness",
        "execute_case case",
    ]
    assert traces.evaluator_view(trace_id).judge_usage["Helpfulness"].tokens == 30


def test_drop_attempt_after_release_keeps_nothing_that_finishes_later(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            pass
        traces.release([trace_id])
        traces.drop_attempt(trace_id, _ids(attempt)[1])
        with _attempt(tracer) as later:
            _chat(tracer)

    assert traces.usage(trace_id, _ids(later)[1]).model_calls == 0


def test_a_span_of_the_attempt_that_ends_after_the_drop_is_kept_until_release(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            late = tracer.start_span("chat", attributes={"gen_ai.operation.name": "chat"})  # e.g. a task not awaited
        traces.drop_attempt(trace_id, _ids(attempt)[1])
        late.end()

    assert [span.name for span in traces._filter.spans(trace_id)] == ["chat", "execute_case case"]
    traces.release([trace_id])
    assert traces._filter.spans(trace_id) == []


@pytest.fixture
def installed(monkeypatch):
    """The provider `TraceCapture()` installs in a process with none yet; the real global one is untouched."""
    providers = []
    monkeypatch.setattr(
        trace_api, "get_tracer_provider", lambda: providers[-1] if providers else trace_api.ProxyTracerProvider()
    )
    monkeypatch.setattr(trace_api, "set_tracer_provider", providers.append)
    monkeypatch.setattr("strands_evals.telemetry.config.propagate.set_global_textmap", lambda propagator: None)

    def install():
        TraceCapture()
        (provider,) = providers
        return provider

    return install


def _events_kept(provider, count):
    """Add `count` events to one span of `provider`: (the first event kept, how many were dropped)."""
    span = provider.get_tracer("optimization-tests").start_span("chat", context=Context())
    for i in range(count):
        span.add_event(f"message {i}")
    span.end()
    return span.events[0].name, span.dropped_events


def test_the_installed_provider_keeps_every_span_event(installed, monkeypatch):
    monkeypatch.delenv("OTEL_SPAN_EVENT_COUNT_LIMIT", raising=False)

    assert _events_kept(installed(), 300) == ("message 0", 0)


@pytest.mark.parametrize(("limit", "kept"), [("5", ("message 295", 295)), ("", ("message 0", 0))])
def test_the_installed_provider_follows_the_span_event_limit_variable(installed, monkeypatch, limit, kept):
    monkeypatch.setenv("OTEL_SPAN_EVENT_COUNT_LIMIT", limit)

    assert _events_kept(installed(), 300) == kept


def _limited(traces):
    """A tracer on a provider that keeps 2 events per span, feeding the capture as if it were the global one."""
    limited = SDKTracerProvider(span_limits=SpanLimits(max_events=2))
    limited.add_span_processor(traces._filter)
    return limited.get_tracer("optimization-tests")


def test_session_warns_once_when_a_provider_dropped_messages(traces, caplog):
    tracer = _limited(traces)
    attempts = []
    for _ in range(2):
        with _trial(tracer, traces) as trace_id:
            with _attempt(tracer) as attempt:
                with tracer.start_as_current_span("chat", attributes={"gen_ai.operation.name": "chat"}) as chat:
                    for text in ("question", "first reply", "second question"):
                        chat.add_event("gen_ai.user.message", {"content": json.dumps([{"text": text}])})
        attempts.append((trace_id, _ids(attempt)[1]))

    with caplog.at_level(logging.WARNING, logger="strands_evals.optimization.execution.traces"):
        sessions = [traces.session(trace_id, span_id) for trace_id, span_id in attempts]

    assert [message.content[0].text for message in sessions[0].traces[0].spans[0].messages] == [
        "first reply",
        "second question",
    ]
    warnings = [record for record in caplog.records if "span event limit" in record.getMessage()]
    assert len(warnings) == 1
    assert "dropped_events=<1>" in warnings[0].getMessage()


@pytest.mark.parametrize(
    ("installed", "advice"),
    [
        (False, "StrandsEvalsTelemetry(span_limits=SpanLimits(max_events=SpanLimits.UNSET))"),
        (True, "unset OTEL_SPAN_EVENT_COUNT_LIMIT or set it empty for no limit"),
    ],
)
def test_the_dropped_messages_warning_says_how_to_lift_the_limit_that_applied(traces, caplog, installed, advice):
    """On a provider the optimizer installed only the variable set a limit; on the user's own, their provider did."""
    traces._installed = installed
    tracer = _limited(traces)
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            with tracer.start_as_current_span("chat", attributes={"gen_ai.operation.name": "chat"}) as chat:
                for text in ("question", "first reply", "second question"):
                    chat.add_event("gen_ai.user.message", {"content": json.dumps([{"text": text}])})

    with caplog.at_level(logging.WARNING, logger="strands_evals.optimization.execution.traces"):
        traces.session(trace_id, _ids(attempt)[1])

    assert advice in caplog.text


def test_session_ignores_dropped_events_on_spans_that_arent_model_calls(traces, caplog):
    tracer = _limited(traces)
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            with tracer.start_as_current_span("execute_event_loop_cycle") as cycle:
                for i in range(3):
                    cycle.add_event(f"message {i}")

    with caplog.at_level(logging.WARNING, logger="strands_evals.optimization.execution.traces"):
        traces.session(trace_id, _ids(attempt)[1])

    assert "span event limit" not in caplog.text


def test_captures_on_one_provider_share_one_span_processor(traces):
    provider = trace_api.get_tracer_provider()
    processors = len(provider._active_span_processor._span_processors)

    TraceCapture()
    TraceCapture()

    assert len(provider._active_span_processor._span_processors) == processors
    assert TraceCapture()._filter is _FILTERS[provider]
