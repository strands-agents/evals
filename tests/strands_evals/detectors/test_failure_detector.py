"""Tests for failure detector."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands_evals.detectors.failure_detector import (
    _extract_json,
    _parse_text_result,
    detect_failures,
    detect_failures_async,
)
from strands_evals.detectors.utils import (
    _group_spans_into_traces,
    _is_context_exceeded,
    _resolve_model,
    _serialize_session,
    _serialize_spans,
)
from strands_evals.types.detector import ConfidenceLevel, FailureOutput
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    Trace,
    UserMessage,
)


def _span_info(span_id: str = "span_1") -> SpanInfo:
    now = datetime.now()
    return SpanInfo(session_id="sess_1", span_id=span_id, trace_id="trace_1", start_time=now, end_time=now)


def _make_session(spans=None) -> Session:
    if spans is None:
        spans = [
            AgentInvocationSpan(
                span_info=_span_info("span_1"),
                user_prompt="Hello",
                agent_response="Hi there",
                available_tools=[ToolConfig(name="search")],
            ),
            InferenceSpan(
                span_info=_span_info("span_2"),
                messages=[UserMessage(content=[TextContent(text="Hello")])],
            ),
        ]
    return Session(
        session_id="sess_1",
        traces=[Trace(trace_id="trace_1", session_id="sess_1", spans=spans)],
    )


def _make_json_response(errors: list[dict]) -> str:
    """Build a JSON string matching FailureDetectionStructuredOutput schema."""
    return json.dumps({"errors": errors})


def test_resolve_model_none():
    from strands.models.bedrock import BedrockModel

    model = _resolve_model(None)
    assert isinstance(model, BedrockModel)


def test_resolve_model_string():
    from strands.models.bedrock import BedrockModel

    model = _resolve_model("us.anthropic.claude-sonnet-4-20250514-v1:0")
    assert isinstance(model, BedrockModel)


def test_resolve_model_instance():
    mock_model = MagicMock()
    assert _resolve_model(mock_model) is mock_model


def test_extract_json_raw():
    raw = '{"errors": []}'
    assert _extract_json(raw) == '{"errors": []}'


def test_extract_json_markdown_fenced():
    text = 'Here is the result:\n```json\n{"errors": []}\n```'
    assert _extract_json(text) == '{"errors": []}'


def test_extract_json_markdown_fenced_no_lang():
    text = 'Result:\n```\n{"errors": []}\n```'
    assert _extract_json(text) == '{"errors": []}'


def test_extract_json_with_surrounding_text():
    text = 'I found these failures:\n```json\n{"errors": [{"location": "s1"}]}\n```\nDone.'
    assert '"errors"' in _extract_json(text)


def test_parse_text_result_basic():
    text = _make_json_response(
        [
            {
                "location": "span_1",
                "category": ["hallucination-category-hall-usage"],
                "confidence": ["high"],
                "evidence": ["Agent claimed to use tool without calling it"],
            }
        ]
    )
    result = _parse_text_result(text)
    assert len(result) == 1
    assert result[0].span_id == "span_1"
    assert result[0].category == ["hallucination-category-hall-usage"]
    assert result[0].confidence == [0.9]
    assert result[0].evidence == ["Agent claimed to use tool without calling it"]


def test_parse_text_result_multiple_modes():
    text = _make_json_response(
        [
            {
                "location": "span_1",
                "category": ["error_a", "error_b"],
                "confidence": ["low", "high"],
                "evidence": ["ev_a", "ev_b"],
            }
        ]
    )
    result = _parse_text_result(text)
    assert len(result) == 1
    assert len(result[0].category) == 2
    assert result[0].confidence == [0.5, 0.9]


def test_parse_text_result_mismatched_arrays():
    text = _make_json_response(
        [
            {
                "location": "span_1",
                "category": ["error_a", "error_b"],
                "confidence": ["high"],
                "evidence": ["only one"],
            }
        ]
    )
    result = _parse_text_result(text)
    assert len(result) == 0  # skipped due to mismatch


def test_parse_text_result_empty():
    text = _make_json_response([])
    result = _parse_text_result(text)
    assert result == []


def test_parse_text_result_markdown_fenced():
    errors = [{"location": "s1", "category": ["err"], "confidence": ["high"], "evidence": ["ev"]}]
    text = f"```json\n{json.dumps({'errors': errors})}\n```"
    result = _parse_text_result(text)
    assert len(result) == 1
    assert result[0].span_id == "s1"


def test_parse_text_result_malformed_json():
    """Malformed JSON should return empty list, not crash."""
    result = _parse_text_result("this is not json at all")
    assert result == []


def test_parse_text_result_invalid_schema():
    """Valid JSON but wrong schema should return empty list, not crash."""
    result = _parse_text_result('{"wrong_key": "wrong_value"}')
    assert result == []


def test_parse_text_result_partial_json():
    """Truncated JSON should return empty list, not crash."""
    result = _parse_text_result('{"errors": [{"location": "s1", "category":')
    assert result == []


def test_is_context_exceeded_strands_exception():
    from strands.types.exceptions import ContextWindowOverflowException

    assert _is_context_exceeded(ContextWindowOverflowException("too big"))


def test_is_context_exceeded_string_match():
    assert _is_context_exceeded(Exception("The context window is exceeded"))
    assert _is_context_exceeded(Exception("Input too long for model"))
    assert _is_context_exceeded(Exception("max_tokens limit reached"))
    assert _is_context_exceeded(Exception("context length exceeded"))
    assert _is_context_exceeded(Exception("input too large for this model"))


def test_is_context_exceeded_unrelated():
    assert not _is_context_exceeded(Exception("Something else went wrong"))
    assert not _is_context_exceeded(ValueError("bad value"))
    # "context" alone should NOT match — must be "context window" or "context length"
    assert not _is_context_exceeded(Exception("invalid context parameter"))
    assert not _is_context_exceeded(Exception("missing context"))


def test_serialize_session():
    session = _make_session()
    result = _serialize_session(session)
    assert "sess_1" in result
    assert "span_1" in result


def test_serialize_spans():
    span = ToolExecutionSpan(
        span_info=_span_info("span_10"),
        tool_call=ToolCall(name="test", arguments={}),
        tool_result=ToolResult(content="ok"),
    )
    result = _serialize_spans([span])
    assert "sess_1" in result
    assert "span_10" in result


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
def test_detect_failures_no_failures(mock_call_model):
    mock_call_model.return_value = _make_json_response([])

    session = _make_session()
    output = detect_failures(session)

    assert isinstance(output, FailureOutput)
    assert output.session_id == "sess_1"
    assert output.failures == []


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
def test_detect_failures_with_failures(mock_call_model):
    mock_call_model.return_value = _make_json_response(
        [
            {
                "location": "span_1",
                "category": ["hallucination-category-hall-usage"],
                "confidence": ["high"],
                "evidence": ["Fabricated tool output"],
            },
        ]
    )

    session = _make_session()
    output = detect_failures(session)

    assert len(output.failures) == 1
    assert output.failures[0].span_id == "span_1"
    assert output.failures[0].confidence == [0.9]


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
def test_detect_failures_confidence_threshold(mock_call_model):
    mock_call_model.return_value = _make_json_response(
        [
            {"location": "span_1", "category": ["err_a"], "confidence": ["low"], "evidence": ["weak"]},
            {"location": "span_2", "category": ["err_b"], "confidence": ["high"], "evidence": ["strong"]},
        ]
    )

    session = _make_session()
    output = detect_failures(session, confidence_threshold=ConfidenceLevel.MEDIUM)

    # "low" is below "medium" threshold
    assert len(output.failures) == 1
    assert output.failures[0].span_id == "span_2"


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
def test_detect_failures_per_mode_filtering(mock_call_model):
    """Individual failure modes below threshold are pruned, not just whole spans."""
    mock_call_model.return_value = _make_json_response(
        [
            {
                "location": "span_1",
                "category": ["hallucination", "repetition"],
                "confidence": ["high", "low"],
                "evidence": ["fabricated response", "minor repeat"],
            }
        ]
    )

    session = _make_session()
    output = detect_failures(session, confidence_threshold=ConfidenceLevel.MEDIUM)

    # Span is kept (has a high-confidence mode) but low-confidence mode is pruned
    assert len(output.failures) == 1
    assert output.failures[0].category == ["hallucination"]
    assert output.failures[0].confidence == [0.9]
    assert "repetition" not in output.failures[0].category


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
def test_detect_failures_context_overflow_fallback(mock_call_model):
    """When direct call raises context overflow, should fall back to chunking."""
    from strands.types.exceptions import ContextWindowOverflowException

    # First call raises context overflow, subsequent calls succeed (chunking)
    mock_call_model.side_effect = [
        ContextWindowOverflowException("too big"),
        _make_json_response([]),
        _make_json_response([]),
    ]

    session = _make_session()
    output = detect_failures(session)

    assert isinstance(output, FailureOutput)


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
def test_detect_failures_non_context_error_raises(mock_call_model):
    """Non-context errors should propagate."""
    mock_call_model.side_effect = RuntimeError("Something else broke")

    session = _make_session()
    with pytest.raises(RuntimeError, match="Something else broke"):
        detect_failures(session)


@patch("strands_evals.detectors.failure_detector._resolve_model")
@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
def test_detect_failures_passes_model(mock_call_model, mock_resolve_model):
    mock_model = MagicMock()
    mock_resolve_model.return_value = mock_model
    mock_call_model.return_value = _make_json_response([])

    session = _make_session()
    detect_failures(session, model="us.anthropic.claude-sonnet-4-20250514-v1:0")

    # Verify _resolve_model was called with the custom model string
    mock_resolve_model.assert_called_once_with("us.anthropic.claude-sonnet-4-20250514-v1:0")
    # Verify _call_model was called with the resolved model
    mock_call_model.assert_called_once()
    assert mock_call_model.call_args.args[0] is mock_model


def test_prompt_overhead_is_positive():
    """Sanity check: prompt overhead should be a reasonable token count."""
    from strands_evals.detectors.failure_detector import _get_prompt_overhead_tokens

    overhead = _get_prompt_overhead_tokens()
    assert overhead > 200
    assert overhead < 10_000


def test_serialize_spans_preserves_trace_structure():
    """Spans from different traces should be grouped by trace_id."""
    span_a = ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="sess_1",
            span_id="s1",
            trace_id="trace_A",
            start_time=datetime.now(),
            end_time=datetime.now(),
        ),
        tool_call=ToolCall(name="t", arguments={}),
        tool_result=ToolResult(content="ok"),
    )
    span_b = ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="sess_1",
            span_id="s2",
            trace_id="trace_B",
            start_time=datetime.now(),
            end_time=datetime.now(),
        ),
        tool_call=ToolCall(name="t", arguments={}),
        tool_result=ToolResult(content="ok"),
    )
    span_a2 = ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="sess_1",
            span_id="s3",
            trace_id="trace_A",
            start_time=datetime.now(),
            end_time=datetime.now(),
        ),
        tool_call=ToolCall(name="t", arguments={}),
        tool_result=ToolResult(content="ok"),
    )

    result = json.loads(_serialize_spans([span_a, span_b, span_a2]))

    # Result is now a bare traces array
    assert len(result) == 2
    trace_ids = [t["trace_id"] for t in result]
    assert "trace_A" in trace_ids
    assert "trace_B" in trace_ids

    trace_a = [t for t in result if t["trace_id"] == "trace_A"][0]
    span_ids = [s["span_info"]["span_id"] for s in trace_a["spans"]]
    assert span_ids == ["s1", "s3"]


def test_group_spans_into_traces_preserves_order():
    """Trace groups should appear in insertion order."""
    spans = [
        _make_tool_span_with_trace("s1", "trace_B"),
        _make_tool_span_with_trace("s2", "trace_A"),
        _make_tool_span_with_trace("s3", "trace_B"),
    ]
    traces = _group_spans_into_traces(spans)
    assert len(traces) == 2
    assert traces[0]["trace_id"] == "trace_B"
    assert traces[1]["trace_id"] == "trace_A"
    assert len(traces[0]["spans"]) == 2
    assert len(traces[1]["spans"]) == 1


def _make_tool_span_with_trace(span_id: str, trace_id: str) -> ToolExecutionSpan:
    return ToolExecutionSpan(
        span_info=SpanInfo(
            session_id="sess_1",
            span_id=span_id,
            trace_id=trace_id,
            start_time=datetime.now(),
            end_time=datetime.now(),
        ),
        tool_call=ToolCall(name="t", arguments={}),
        tool_result=ToolResult(content="ok"),
    )


# --- async tests ---


def _make_multi_span_session(n_spans: int = 4) -> Session:
    """A session with enough spans to clear MIN_CHUNK_SIZE guard in _detect_chunked_async."""
    spans = [
        ToolExecutionSpan(
            span_info=_span_info(f"span_{i}"),
            tool_call=ToolCall(name="t", arguments={}),
            tool_result=ToolResult(content="ok"),
        )
        for i in range(n_spans)
    ]
    return Session(
        session_id="sess_1",
        traces=[Trace(trace_id="trace_1", session_id="sess_1", spans=spans)],
    )


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
async def test_detect_failures_async_no_failures(mock_call_model):
    mock_call_model.return_value = _make_json_response([])

    output = await detect_failures_async(_make_session())

    assert isinstance(output, FailureOutput)
    assert output.failures == []


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
async def test_detect_failures_async_with_failures(mock_call_model):
    mock_call_model.return_value = _make_json_response(
        [
            {
                "location": "span_1",
                "category": ["err"],
                "confidence": ["high"],
                "evidence": ["bad"],
            }
        ]
    )

    output = await detect_failures_async(_make_session())

    assert len(output.failures) == 1
    assert output.failures[0].span_id == "span_1"


@patch("strands_evals.detectors.failure_detector._detect_chunked_async", new_callable=AsyncMock)
@patch("strands_evals.detectors.failure_detector.would_exceed_context")
async def test_detect_failures_async_passes_max_workers_to_chunked(mock_would_exceed, mock_chunked):
    """When the session triggers chunked detection, max_workers is forwarded."""
    mock_would_exceed.return_value = True
    mock_chunked.return_value = []

    await detect_failures_async(_make_session(), max_workers=4)

    mock_chunked.assert_awaited_once()
    # signature: _detect_chunked_async(session, model, template, max_workers)
    assert mock_chunked.call_args.args[3] == 4


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
@patch("strands_evals.detectors.failure_detector.split_spans_by_tokens")
@patch("strands_evals.detectors.failure_detector.would_exceed_context")
async def test_detect_failures_chunked_concurrent_calls(mock_would_exceed, mock_split, mock_call_model):
    """Chunked path issues per-chunk LLM calls under the gather; with max_workers > 1
    they should overlap. Asserts that >=2 chunks are in flight at once."""
    import asyncio

    mock_would_exceed.return_value = True
    # 3 fake chunks, each containing one span
    session = _make_multi_span_session(4)
    spans = session.traces[0].spans
    mock_split.return_value = [[spans[0]], [spans[1]], [spans[2]]]

    in_flight = 0
    peak = 0
    enter = asyncio.Event()

    async def fake_call(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        # First chunk waits until a sibling has entered, proving overlap
        if not enter.is_set():
            enter.set()
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _make_json_response([])

    mock_call_model.side_effect = fake_call

    await detect_failures_async(session, max_workers=3)

    assert peak >= 2, f"expected concurrent chunks; observed peak={peak}"


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
@patch("strands_evals.detectors.failure_detector.split_spans_by_tokens")
@patch("strands_evals.detectors.failure_detector.would_exceed_context")
async def test_detect_failures_chunked_one_overflow_does_not_kill_batch(mock_would_exceed, mock_split, mock_call_model):
    """One chunk raising ContextWindowOverflowException is logged and skipped;
    sibling chunks still produce results."""
    from strands.types.exceptions import ContextWindowOverflowException

    mock_would_exceed.return_value = True
    session = _make_multi_span_session(4)
    spans = session.traces[0].spans
    mock_split.return_value = [[spans[0]], [spans[1]], [spans[2]]]

    mock_call_model.side_effect = [
        _make_json_response([{"location": "span_1", "category": ["e"], "confidence": ["high"], "evidence": ["x"]}]),
        ContextWindowOverflowException("nope"),
        _make_json_response([{"location": "span_1", "category": ["e"], "confidence": ["high"], "evidence": ["y"]}]),
    ]

    output = await detect_failures_async(session, max_workers=1)

    # Two surviving chunks, each contributing the same span — merge_chunk_failures dedupes.
    assert len(output.failures) >= 1
    assert output.failures[0].span_id == "span_1"


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
@patch("strands_evals.detectors.failure_detector.split_spans_by_tokens")
@patch("strands_evals.detectors.failure_detector.would_exceed_context")
async def test_detect_failures_chunked_non_context_error_propagates(mock_would_exceed, mock_split, mock_call_model):
    """A non-context exception in any chunk should raise out of the gather."""
    mock_would_exceed.return_value = True
    session = _make_multi_span_session(4)
    spans = session.traces[0].spans
    mock_split.return_value = [[spans[0]], [spans[1]]]

    mock_call_model.side_effect = [
        _make_json_response([]),
        RuntimeError("Something else broke"),
    ]

    with pytest.raises(RuntimeError, match="Something else broke"):
        await detect_failures_async(session, max_workers=1)


# --- sync-wrapper-from-inside-event-loop regression ---


@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
async def test_detect_failures_sync_works_inside_running_loop(mock_call_model):
    """`detect_failures` (the sync wrapper) must be safe to call from inside an
    async function. `pyproject.toml` sets `asyncio_mode = "auto"`, so this test
    body runs inside an event loop -- exactly the regression case.
    """
    mock_call_model.return_value = _make_json_response([])

    output = detect_failures(_make_session())

    assert isinstance(output, FailureOutput)
    assert output.failures == []



@patch("strands_evals.detectors.failure_detector._call_model_async", new_callable=AsyncMock)
@patch("strands_evals.detectors.failure_detector.split_spans_by_tokens")
@patch("strands_evals.detectors.failure_detector.would_exceed_context")
async def test_detect_failures_chunked_fails_fast_cancels_siblings(
    mock_would_exceed, mock_split, mock_call_model
):
    """When one chunk raises a non-context error, sibling chunks still in
    flight must be cancelled before they complete their LLM call -- otherwise
    we burn tokens on work whose result we'll throw away."""
    import asyncio

    mock_would_exceed.return_value = True
    session = _make_multi_span_session(4)
    spans = session.traces[0].spans
    mock_split.return_value = [[spans[0]], [spans[1]], [spans[2]]]

    sibling_finished = False
    sibling_started = asyncio.Event()

    async def fake_call(*args, **kwargs):
        # Chunk 0 starts first and is still mid-flight when chunk 1 raises.
        # Chunk 2 (queued behind the semaphore) is also still pending.
        if not sibling_started.is_set():
            sibling_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise
            nonlocal sibling_finished
            sibling_finished = True
            return _make_json_response([])
        # Subsequent calls: the failing chunk waits until sibling has
        # actually started, then raises.
        await sibling_started.wait()
        raise RuntimeError("hard fail")

    mock_call_model.side_effect = fake_call

    with pytest.raises(RuntimeError, match="hard fail"):
        await detect_failures_async(session, max_workers=3)

    await asyncio.sleep(0)
    assert sibling_finished is False, (
        "fail-fast chunked detection must cancel the slow sibling, "
        "not let it complete its LLM call"
    )
