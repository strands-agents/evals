"""Tests for detector Pydantic models."""

from strands_evals.types.detector import (
    DiagnosisResult,
    FailureDetectionStructuredOutput,
    FailureError,
    FailureItem,
    FailureOutput,
    FixRecommendation,
    RCAItem,
    RCAOutput,
    RCAStructuredOutput,
    RootCauseItem,
    SummaryOutput,
)

# --- Output types ---


def test_failure_item_creation():
    item = FailureItem(
        span_id="span_123",
        category=["hallucination", "incomplete_task"],
        confidence=["high", "medium"],
        evidence=["Made up product ID", "Did not finish checkout"],
    )
    assert item.span_id == "span_123"
    assert len(item.category) == 2
    assert item.confidence[0] == "high"
    assert item.evidence[1] == "Did not finish checkout"


def test_failure_output_empty():
    output = FailureOutput(session_id="sess_1")
    assert output.session_id == "sess_1"
    assert output.failures == []


def test_failure_output_with_failures():
    item = FailureItem(
        span_id="span_1",
        category=["error"],
        confidence=["high"],
        evidence=["timeout"],
    )
    output = FailureOutput(session_id="sess_1", failures=[item])
    assert len(output.failures) == 1
    assert output.failures[0].span_id == "span_1"


def test_rca_item_creation():
    item = RCAItem(
        failure_span_id="span_1",
        location="span_0",
        causality="PRIMARY_FAILURE",
        propagation_impact=["TASK_TERMINATION"],
        root_cause_explanation="The tool returned ambiguous results",
        fix_type="TOOL_DESCRIPTION_FIX",
        fix_recommendation="Add disambiguation instructions",
    )
    assert item.failure_span_id == "span_1"
    assert item.causality == "PRIMARY_FAILURE"


def test_rca_output_empty():
    output = RCAOutput()
    assert output.root_causes == []


def test_summary_output():
    output = SummaryOutput(
        primary_goal={"description": "Book a flight", "evidence": ["span_1"]},
        approach_taken={"description": "Used search tool", "evidence": ["span_2"]},
        final_outcome={"description": "Successfully booked", "success": True, "evidence": ["span_5"]},
    )
    assert output.primary_goal["description"] == "Book a flight"
    assert output.tools_used == []
    assert output.observed_failures == []


def test_diagnosis_result():
    item = FailureItem(
        span_id="span_1",
        category=["error"],
        confidence=["high"],
        evidence=["timeout"],
    )
    rca = RCAItem(
        failure_span_id="span_1",
        location="span_0",
        causality="PRIMARY_FAILURE",
        root_cause_explanation="Timeout due to slow API",
        fix_type="OTHERS",
        fix_recommendation="Add retry logic",
    )
    result = DiagnosisResult(
        session_id="sess_1",
        user_requests=["Book a flight"],
        failures=[item],
        summary="The agent attempted to book a flight but timed out.",
        root_causes=[rca],
    )
    assert result.session_id == "sess_1"
    assert len(result.user_requests) == 1
    assert len(result.failures) == 1
    assert result.summary.startswith("The agent")
    assert len(result.root_causes) == 1


def test_diagnosis_result_empty():
    result = DiagnosisResult(session_id="sess_1")
    assert result.user_requests == []
    assert result.failures == []
    assert result.summary == ""
    assert result.root_causes == []


# --- LLM structured output schemas ---


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


def test_root_cause_item_with_aliases():
    """RootCauseItem should accept both alias names and field names."""
    item = RootCauseItem(
        failure_span_id="span_1",
        location="span_0",
        failure_causality="PRIMARY_FAILURE",
        failure_propagation_impact=["TASK_TERMINATION"],
        failure_detection_timing="IMMEDIATELY_AT_OCCURRENCE",
        completion_status="COMPLETE_FAILURE",
        root_cause_explanation="Tool returned bad data",
        fix_recommendation=FixRecommendation(
            fix_type="TOOL_DESCRIPTION_FIX",
            recommendation="Add validation",
        ),
    )
    assert item.failure_span_id == "span_1"
    assert item.failure_causality == "PRIMARY_FAILURE"


def test_root_cause_item_from_llm_aliases():
    """RootCauseItem should parse from LLM output using JSON aliases."""
    data = {
        "Failure Span ID": "span_1",
        "Location": "span_0",
        "Failure Causality": "SECONDARY_FAILURE",
        "Failure Propagation Impact": ["QUALITY_DEGRADATION"],
        "Failure Detection Timing": "SEVERAL_STEPS_LATER",
        "Completion Status": "PARTIAL_SUCCESS",
        "Root Cause Explanation": "Ambiguous tool output",
        "Fix Recommendation": {
            "Fix Type": "SYSTEM_PROMPT_FIX",
            "Recommendation": "Add disambiguation",
        },
    }
    item = RootCauseItem.model_validate(data)
    assert item.failure_span_id == "span_1"
    assert item.failure_causality == "SECONDARY_FAILURE"
    assert item.fix_recommendation.fix_type == "SYSTEM_PROMPT_FIX"


def test_rca_structured_output():
    output = RCAStructuredOutput(
        root_causes=[
            RootCauseItem(
                failure_span_id="span_1",
                location="span_0",
                failure_causality="PRIMARY_FAILURE",
                failure_propagation_impact=["TASK_TERMINATION"],
                failure_detection_timing="IMMEDIATELY_AT_OCCURRENCE",
                completion_status="COMPLETE_FAILURE",
                root_cause_explanation="Error",
                fix_recommendation=FixRecommendation(
                    fix_type="OTHERS",
                    recommendation="Fix it",
                ),
            )
        ]
    )
    assert len(output.root_causes) == 1


def test_failure_output_serialization_roundtrip():
    """Test that models can serialize and deserialize."""
    item = FailureItem(
        span_id="span_1",
        category=["hallucination"],
        confidence=["high"],
        evidence=["Made up data"],
    )
    output = FailureOutput(session_id="sess_1", failures=[item])
    json_str = output.model_dump_json()
    restored = FailureOutput.model_validate_json(json_str)
    assert restored.session_id == "sess_1"
    assert restored.failures[0].confidence[0] == "high"
