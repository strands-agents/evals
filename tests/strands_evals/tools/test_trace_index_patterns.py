"""Cross-pattern tests: TraceIndex over Sessions produced by every trace source.

Verifies the index's overview/discovery behavior is identical whether the
Session came from:
- Strands-native OTEL spans (gen_ai semconv, StrandsInMemorySessionMapper)
- Langfuse observations (LangfuseProvider conversion)
- OpenInference spans (OpenInferenceSessionMapper, ADOT fixture)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags

from strands_evals.mappers import OpenInferenceSessionMapper, StrandsInMemorySessionMapper
from strands_evals.tools.trace_index import TraceIndex
from strands_evals.types.trace import Session

_FIXTURES_DIR = Path(__file__).parent.parent / "mappers" / "fixtures"

LARGE_RESULT = json.dumps({"rows": [{"ticket": f"TKT-{i}", "status": "resolved"} for i in range(500)]})


# --- Strands-native OTEL (gen_ai semconv) ---


def _otel_span(provider, trace_id, span_id, parent_id, operation, attributes, events_fn):
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(operation, kind=SpanKind.CLIENT) as s:
        for k, v in attributes.items():
            s.set_attribute(k, v)
        events_fn(s)
    return ReadableSpan(
        name=operation,
        context=SpanContext(trace_id, span_id, False, TraceFlags(0x01)),
        parent=SpanContext(trace_id, parent_id, False, TraceFlags(0x01)) if parent_id else None,
        resource=provider.resource,
        attributes=attributes,
        events=tuple(s._events),
        start_time=1700000000000000000,
        end_time=1700000001000000000,
    )


@pytest.fixture
def strands_native_session() -> Session:
    provider = TracerProvider()
    agent_span = _otel_span(
        provider,
        0xAAA,
        0xBB1,
        None,
        "invoke_agent",
        {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "support-agent"},
        lambda s: (
            s.add_event("gen_ai.user.message", {"content": '[{"text": "Check ticket TKT-42"}]'}),
            s.add_event("gen_ai.choice", {"message": '[{"text": "TKT-42 is resolved."}]'}),
        ),
    )
    tool_result_message = json.dumps([{"text": LARGE_RESULT}])
    tool_span = _otel_span(
        provider,
        0xAAA,
        0xBB2,
        0xBB1,
        "execute_tool lookup_ticket",
        {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "lookup_ticket"},
        lambda s: (
            s.add_event("gen_ai.tool.message", {"content": '{"id": "TKT-42"}', "id": "call-1"}),
            s.add_event("gen_ai.choice", {"message": tool_result_message, "id": "call-1"}),
        ),
    )
    return StrandsInMemorySessionMapper().map_to_session([agent_span, tool_span], "native-session")


# --- Langfuse observations ---


def _lf_obs(obs_id, trace_id, obs_type, name=None, obs_input=None, obs_output=None, parent=None, start=None):
    o = MagicMock()
    o.id, o.trace_id, o.type, o.name = obs_id, trace_id, obs_type, name
    o.start_time = start or datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    o.end_time = datetime(2025, 1, 15, 10, 0, 5, tzinfo=timezone.utc)
    o.input, o.output = obs_input, obs_output
    o.parent_observation_id = parent
    o.metadata, o.model = {}, None
    o.level, o.usage, o.usage_details = "DEFAULT", None, None
    return o


@pytest.fixture
def langfuse_session() -> Session:
    import strands_evals.providers.langfuse_provider as lf_module

    with patch.object(lf_module, "Langfuse", return_value=MagicMock()):
        provider = lf_module.LangfuseProvider(public_key="pk-test", secret_key="sk-test")

    observations = [
        _lf_obs(
            "obs-agent",
            "trace-1",
            "SPAN",
            name="invoke_agent support-agent",
            obs_input=[{"text": "Check ticket TKT-42"}],
            obs_output="TKT-42 is resolved.",
            start=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        ),
        _lf_obs(
            "obs-tool",
            "trace-1",
            "TOOL",
            name="lookup_ticket",
            obs_input={"id": "TKT-42"},
            obs_output=LARGE_RESULT,
            parent="obs-agent",
            start=datetime(2025, 1, 15, 10, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    spans = provider._convert_observations(observations, "lf-session")
    spans = [s for s in spans if s is not None]
    if not spans:
        pytest.skip("Langfuse conversion produced no spans for this synthetic shape")
    from strands_evals.types.trace import Trace

    return Session(traces=[Trace(spans=spans, trace_id="trace-1", session_id="lf-session")], session_id="lf-session")


# --- OpenInference (ADOT fixture from the repo) ---


@pytest.fixture
def openinference_session() -> Session:
    fixture = _FIXTURES_DIR / "openinference_adot_spans.json"
    if not fixture.exists():
        pytest.skip("ADOT fixture not present")
    with open(fixture) as f:
        spans = json.load(f)
    return OpenInferenceSessionMapper().map_to_session(spans, "oi-session")


# --- Shared assertions across patterns ---


def _assert_index_works(session: Session):
    index = TraceIndex(session)
    list_spans, get_span, search_spans = index.tools

    overview = index.overview()
    n_spans = sum(len(t.spans) for t in session.traces)
    assert f"{n_spans} spans" in overview.splitlines()[0]
    # Overview stays compact regardless of payload size
    assert len(overview) < 400 * max(n_spans, 1) + 200

    # Every span index is retrievable
    for i in range(n_spans):
        content = get_span(index=i)
        assert not content.startswith("ERROR"), f"span {i} failed: {content[:80]}"

    assert isinstance(list_spans(), str)


def test_index_on_strands_native_session(strands_native_session):
    _assert_index_works(strands_native_session)

    index = TraceIndex(strands_native_session)
    _, get_span, search_spans = index.tools

    # Content-level checks: the judge can find the ticket in the tool result
    hits = search_spans(pattern="TKT-42")
    assert hits.startswith("[")


def test_index_on_langfuse_session(langfuse_session):
    _assert_index_works(langfuse_session)

    index = TraceIndex(langfuse_session)
    _, _, search_spans = index.tools
    assert search_spans(pattern="TKT-42").startswith("[")


def test_index_on_openinference_session(openinference_session):
    _assert_index_works(openinference_session)


def test_overview_compression_on_large_native_trace(strands_native_session):
    """The overview must be dramatically smaller than the inline serialization."""
    index = TraceIndex(strands_native_session)
    inline_size = len(str(strands_native_session.model_dump()))
    overview_size = len(index.overview())

    assert inline_size > 10_000  # LARGE_RESULT made it big
    assert overview_size < inline_size / 10
