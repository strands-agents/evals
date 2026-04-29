"""Tests for detector Pydantic models."""

from strands_evals.types.detector import (
    FailureDetectionStructuredOutput,
    FailureError,
    FailureItem,
    FailureOutput,
)


def test_failure_item_creation():
    item = FailureItem(
        span_id="span_123",
        category=["hallucination", "incomplete_task"],
        confidence=[0.9, 0.75],
        evidence=["Made up product ID", "Did not finish checkout"],
    )
    assert item.span_id == "span_123"
    assert len(item.category) == 2
    assert item.confidence[0] == 0.9
    assert item.evidence[1] == "Did not finish checkout"


def test_failure_output_empty():
    output = FailureOutput(session_id="sess_1")
    assert output.session_id == "sess_1"
    assert output.failures == []


def test_failure_output_with_failures():
    item = FailureItem(
        span_id="span_1",
        category=["error"],
        confidence=[0.9],
        evidence=["timeout"],
    )
    output = FailureOutput(session_id="sess_1", failures=[item])
    assert len(output.failures) == 1
    assert output.failures[0].span_id == "span_1"


def test_failure_error():
    err = FailureError(
        location="span_1",
        category=["hallucination"],
        confidence=["high"],
        evidence=["Made up data"],
    )
    assert err.location == "span_1"
    assert err.confidence == ["high"]


def test_failure_detection_structured_output():
    output = FailureDetectionStructuredOutput(
        errors=[
            FailureError(
                location="span_1",
                category=["error"],
                confidence=["low"],
                evidence=["Timed out"],
            )
        ]
    )
    assert len(output.errors) == 1


def test_failure_output_serialization_roundtrip():
    """Test that models can serialize and deserialize."""
    item = FailureItem(
        span_id="span_1",
        category=["hallucination"],
        confidence=[0.9],
        evidence=["Made up data"],
    )
    output = FailureOutput(session_id="sess_1", failures=[item])
    json_str = output.model_dump_json()
    restored = FailureOutput.model_validate_json(json_str)
    assert restored.session_id == "sess_1"
    assert restored.failures[0].confidence[0] == 0.9
