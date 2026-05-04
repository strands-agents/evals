"""Tests for diagnosis module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from strands_evals.detectors.diagnosis import diagnose_session
from strands_evals.types.detector import (
    DiagnosisResult,
    FailureItem,
    FailureOutput,
    RCAItem,
    RCAOutput,
)
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolConfig,
    Trace,
    UserMessage,
)


def _span_info(span_id: str = "span_1") -> SpanInfo:
    now = datetime.now()
    return SpanInfo(session_id="sess_1", span_id=span_id, trace_id="trace_1", start_time=now, end_time=now)


def _make_session() -> Session:
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


class TestDiagnoseSession:
    @patch("strands_evals.detectors.diagnosis.detect_failures")
    def test_no_failures_skips_rca(self, mock_detect):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[])
        session = _make_session()

        result = diagnose_session(session)

        assert isinstance(result, DiagnosisResult)
        assert result.session_id == "sess_1"
        assert result.failures == []
        assert result.root_causes == []

    @patch("strands_evals.detectors.diagnosis.analyze_root_cause")
    @patch("strands_evals.detectors.diagnosis.detect_failures")
    def test_with_failures_runs_rca(self, mock_detect, mock_rca):
        failures = [
            FailureItem(span_id="span_1", category=["hallucination"], confidence=[0.9], evidence=["made up data"])
        ]
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=failures)
        rca_items = [
            RCAItem(
                failure_span_id="span_1",
                location="span_1",
                causality="PRIMARY_FAILURE",
                propagation_impact=["QUALITY_DEGRADATION"],
                failure_detection_timing="IMMEDIATELY_AT_OCCURRENCE",
                completion_status="PARTIAL_SUCCESS",
                root_cause_explanation="Bad tool output",
                fix_type="TOOL_DESCRIPTION_FIX",
                fix_recommendation="Improve tool description",
            )
        ]
        mock_rca.return_value = RCAOutput(root_causes=rca_items)
        session = _make_session()

        result = diagnose_session(session)

        assert len(result.failures) == 1
        assert len(result.root_causes) == 1
        assert result.root_causes[0].fix_recommendation == "Improve tool description"
        mock_rca.assert_called_once()

    @patch("strands_evals.detectors.diagnosis.detect_failures")
    def test_passes_model_and_threshold(self, mock_detect):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[])
        session = _make_session()
        mock_model = MagicMock()

        diagnose_session(session, model=mock_model, confidence_threshold="high")

        mock_detect.assert_called_once_with(
            session,
            confidence_threshold="high",
            model=mock_model,
        )

    @patch("strands_evals.detectors.diagnosis.analyze_root_cause")
    @patch("strands_evals.detectors.diagnosis.detect_failures")
    def test_passes_model_to_rca(self, mock_detect, mock_rca):
        failures = [FailureItem(span_id="s1", category=["error"], confidence=[0.9], evidence=["e"])]
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=failures)
        mock_rca.return_value = RCAOutput(root_causes=[])
        session = _make_session()
        mock_model = MagicMock()

        diagnose_session(session, model=mock_model)

        mock_rca.assert_called_once_with(session, failures=failures, model=mock_model)

    @patch("strands_evals.detectors.diagnosis.detect_failures")
    def test_serialization_round_trip(self, mock_detect):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[])
        session = _make_session()

        result = diagnose_session(session)
        dumped = result.model_dump()

        assert dumped["session_id"] == "sess_1"
        assert dumped["failures"] == []
        assert dumped["root_causes"] == []

        restored = DiagnosisResult.model_validate(dumped)
        assert restored == result
