"""Tests for failure detector."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from strands_evals.detectors.failure_detector import (
    CONFIDENCE_ORDER,
    _extract_json,
    _is_context_exceeded,
    _max_confidence_rank,
    _parse_text_result,
    _resolve_model,
    _serialize_session,
    _serialize_spans,
    detect_failures,
)
from strands_evals.types.detector import FailureItem, FailureOutput
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


# --- _resolve_model ---


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


# --- _extract_json ---


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


# --- _parse_text_result ---


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
    assert result[0].confidence == ["high"]
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
    assert result[0].confidence == ["low", "high"]


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


# --- _max_confidence_rank ---


def test_max_confidence_rank():
    item = FailureItem(span_id="s", category=["a", "b"], confidence=["low", "high"], evidence=["x", "y"])
    assert _max_confidence_rank(item) == CONFIDENCE_ORDER["high"]


def test_max_confidence_rank_empty():
    item = FailureItem(span_id="s", category=[], confidence=[], evidence=[])
    assert _max_confidence_rank(item) == -1


# --- _is_context_exceeded ---


def test_is_context_exceeded_strands_exception():
    from strands.types.exceptions import ContextWindowOverflowException

    assert _is_context_exceeded(ContextWindowOverflowException("too big"))


def test_is_context_exceeded_string_match():
    assert _is_context_exceeded(Exception("The context window is exceeded"))
    assert _is_context_exceeded(Exception("Input too long for model"))
    assert _is_context_exceeded(Exception("max_tokens limit reached"))


def test_is_context_exceeded_unrelated():
    assert not _is_context_exceeded(Exception("Something else went wrong"))
    assert not _is_context_exceeded(ValueError("bad value"))


# --- _serialize_session / _serialize_spans ---


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
    result = _serialize_spans([span], "sess_1")
    assert "sess_1" in result
    assert "span_10" in result


# --- detect_failures (with mocked _call_model) ---


@patch("strands_evals.detectors.failure_detector._call_model")
def test_detect_failures_no_failures(mock_call_model):
    mock_call_model.return_value = _make_json_response([])

    session = _make_session()
    output = detect_failures(session)

    assert isinstance(output, FailureOutput)
    assert output.session_id == "sess_1"
    assert output.failures == []


@patch("strands_evals.detectors.failure_detector._call_model")
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
    assert output.failures[0].confidence == ["high"]


@patch("strands_evals.detectors.failure_detector._call_model")
def test_detect_failures_confidence_threshold(mock_call_model):
    mock_call_model.return_value = _make_json_response(
        [
            {"location": "span_1", "category": ["err_a"], "confidence": ["low"], "evidence": ["weak"]},
            {"location": "span_2", "category": ["err_b"], "confidence": ["high"], "evidence": ["strong"]},
        ]
    )

    session = _make_session()
    output = detect_failures(session, confidence_threshold="medium")

    # "low" is below "medium" threshold
    assert len(output.failures) == 1
    assert output.failures[0].span_id == "span_2"


@patch("strands_evals.detectors.failure_detector._call_model")
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


@patch("strands_evals.detectors.failure_detector._call_model")
def test_detect_failures_non_context_error_raises(mock_call_model):
    """Non-context errors should propagate."""
    mock_call_model.side_effect = RuntimeError("Something else broke")

    session = _make_session()
    with pytest.raises(RuntimeError, match="Something else broke"):
        detect_failures(session)


@patch("strands_evals.detectors.failure_detector._resolve_model")
@patch("strands_evals.detectors.failure_detector._call_model")
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
