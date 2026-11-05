from contextlib import contextmanager

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.context import Context
from strands import Agent

from strands_evals.optimization.execution.traces import (
    _FILTERS,
    ATTEMPT_SPAN,
    ROLLOUT_SPAN,
    SPAN_KIND,
    TraceCapture,
)
from strands_evals.types.trace import Session
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


def test_spans_of_unregistered_traces_are_dropped(traces, tracer):
    with _trial(tracer, traces, register=False) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(tracer)

    session, usage = traces.agent_view(trace_id, _ids(attempt)[1])

    assert session is None
    assert usage.model_calls == 0


def test_agent_view_reads_only_the_attempts_subtree(traces, tracer):
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

    _, usage = traces.agent_view(trace_id, _ids(last)[1])

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

    _, usage = traces.agent_view(trace_id, _ids(attempt)[1])

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

    _, usage = traces.agent_view(trace_id, _ids(attempt)[1])

    assert usage.input_tokens == 140
    assert usage.cache_read_tokens == 100


def test_cache_tokens_unmeasured_when_no_span_reports_them(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(tracer)

    _, usage = traces.agent_view(trace_id, _ids(attempt)[1])

    assert usage.cache_read_tokens is None
    assert usage.cache_write_tokens is None


def test_cache_tokens_read_from_either_attribute_name(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(tracer, **{"gen_ai.usage.cache_read_input_tokens": 3, "gen_ai.usage.cache_write_input_tokens": 4})
            _chat(tracer, **{"gen_ai.usage.cache_creation.input_tokens": 0})

    _, usage = traces.agent_view(trace_id, _ids(attempt)[1])

    assert usage.cache_read_tokens == 3
    assert usage.cache_write_tokens == 4


def test_spans_without_a_model_id_add_no_model(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(tracer, model=None)

    _, usage = traces.agent_view(trace_id, _ids(attempt)[1])

    assert usage.models == []


def test_seconds_come_from_the_rollout_span(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            with _rollout(tracer) as rollout:
                _chat(tracer)

    _, usage = traces.agent_view(trace_id, _ids(attempt)[1])

    assert usage.seconds == pytest.approx((rollout.end_time - rollout.start_time) / 1e9)
    assert usage.seconds < (attempt.end_time - attempt.start_time) / 1e9


def test_seconds_fall_back_to_the_attempt_span(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(tracer)

    _, usage = traces.agent_view(trace_id, _ids(attempt)[1])

    assert usage.seconds == pytest.approx((attempt.end_time - attempt.start_time) / 1e9)


def test_attempt_usage_matches_agent_view(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            _chat(tracer, total=40, output=4)

    span_id = _ids(attempt)[1]

    assert traces.attempt_usage(trace_id, span_id) == traces.agent_view(trace_id, span_id)[1]


async def test_agent_view_maps_agent_spans_to_a_session(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            agent = Agent(model=ScriptedModel(["hello there"]), callback_handler=None)
            await agent.invoke_async("hi")

    session, usage = traces.agent_view(trace_id, _ids(attempt)[1])

    assert isinstance(session, Session)
    assert [trace.trace_id for trace in session.traces] == [trace_id]
    assert usage.model_calls == 1
    assert usage.tokens == 15
    assert usage.models == ["scripted-model"]


def test_agent_view_without_agent_spans_has_no_session(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer) as attempt:
            pass

    session, _ = traces.agent_view(trace_id, _ids(attempt)[1])

    assert session is None


def test_evaluator_view_reads_judge_usage_errors_and_labels(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        with _attempt(tracer):
            _chat(tracer, total=500, output=50, model="agent-model")
        with tracer.start_as_current_span(
            "evaluator Helpfulness",
            attributes={"gen_ai.evaluation.name": "Helpfulness", "gen_ai.evaluation.score.label": "Very helpful"},
        ):
            _chat(tracer, total=30, output=3, model="judge-model")
        with pytest.raises(ConnectionError):
            with tracer.start_as_current_span("evaluator Broken", attributes={"gen_ai.evaluation.name": "Broken"}):
                _chat(tracer, total=12, output=2, model="judge-model")
                raise ConnectionError("judge unreachable")

    view = traces.evaluator_view(trace_id)

    assert view.labels == {"Helpfulness": "Very helpful"}
    assert view.errors == {"Broken": "ConnectionError"}
    assert view.judge_usage.tokens == 42
    assert view.judge_usage.model_calls == 2
    assert view.judge_usage.models == ["judge-model"]
    assert view.judge_usage.seconds > 0


def test_evaluator_view_with_no_evaluators_is_empty(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        pass

    view = traces.evaluator_view(trace_id)

    assert view.errors == {}
    assert view.labels == {}
    assert view.judge_usage.model_calls == 0


def test_release_drops_only_those_trials(traces, tracer):
    with _trial(tracer, traces) as first:
        with _attempt(tracer) as first_attempt:
            _chat(tracer)
    with _trial(tracer, traces) as second:
        with _attempt(tracer) as second_attempt:
            _chat(tracer)

    traces.release([first])

    assert traces.agent_view(first, _ids(first_attempt)[1])[1].model_calls == 0
    assert traces.agent_view(second, _ids(second_attempt)[1])[1].model_calls == 1


def test_a_released_trace_keeps_nothing_that_finishes_later(traces, tracer):
    with _trial(tracer, traces) as trace_id:
        traces.release([trace_id])
        with _attempt(tracer) as attempt:
            _chat(tracer)

    assert traces.agent_view(trace_id, _ids(attempt)[1])[1].model_calls == 0


def test_captures_on_one_provider_share_one_span_processor(traces):
    provider = trace_api.get_tracer_provider()
    processors = len(provider._active_span_processor._span_processors)

    TraceCapture()
    TraceCapture()

    assert len(provider._active_span_processor._span_processors) == processors
    assert TraceCapture()._filter is _FILTERS[provider]
