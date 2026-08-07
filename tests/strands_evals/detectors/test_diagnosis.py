"""Tests for diagnosis module."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from strands_evals.detectors.diagnosis import (
    diagnose_session,
    diagnose_session_async,
    diagnose_sessions_async,
)
from strands_evals.types.detector import (
    ConfidenceLevel,
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


def _span_info(span_id: str = "span_1", session_id: str = "sess_1") -> SpanInfo:
    now = datetime.now()
    return SpanInfo(session_id=session_id, span_id=span_id, trace_id="trace_1", start_time=now, end_time=now)


def _make_session(session_id: str = "sess_1") -> Session:
    spans = [
        AgentInvocationSpan(
            span_info=_span_info("span_1", session_id),
            user_prompt="Hello",
            agent_response="Hi there",
            available_tools=[ToolConfig(name="search")],
        ),
        InferenceSpan(
            span_info=_span_info("span_2", session_id),
            messages=[UserMessage(content=[TextContent(text="Hello")])],
        ),
    ]
    return Session(
        session_id=session_id,
        traces=[Trace(trace_id="trace_1", session_id=session_id, spans=spans)],
    )


def _failure(span_id: str = "span_1") -> FailureItem:
    return FailureItem(span_id=span_id, category=["hallucination"], confidence=[0.9], evidence=["made up data"])


def _rca(span_id: str = "span_1") -> RCAItem:
    return RCAItem(
        failure_span_id=span_id,
        location=span_id,
        causality="PRIMARY_FAILURE",
        propagation_impact=["QUALITY_DEGRADATION"],
        failure_detection_timing="IMMEDIATELY_AT_OCCURRENCE",
        completion_status="PARTIAL_SUCCESS",
        root_cause_explanation="Bad tool output",
        fix_type="TOOL_DESCRIPTION_FIX",
        fix_recommendation="Improve tool description",
    )


class TestDiagnoseSession:
    """Sync `diagnose_session` is a thin asyncio.run shim — patch the async impls."""

    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    def test_no_failures_skips_rca(self, mock_detect):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[])

        result = diagnose_session(_make_session())

        assert isinstance(result, DiagnosisResult)
        assert result.session_id == "sess_1"
        assert result.failures == []
        assert result.root_causes == []

    @patch("strands_evals.detectors.diagnosis.analyze_root_cause_async", new_callable=AsyncMock)
    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    def test_with_failures_runs_rca(self, mock_detect, mock_rca):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[_failure()])
        mock_rca.return_value = RCAOutput(root_causes=[_rca()])

        result = diagnose_session(_make_session())

        assert len(result.failures) == 1
        assert len(result.root_causes) == 1
        assert result.root_causes[0].fix_recommendation == "Improve tool description"
        mock_rca.assert_awaited_once()

    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    def test_passes_model_and_threshold(self, mock_detect):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[])
        mock_model = MagicMock()

        diagnose_session(_make_session(), model=mock_model, confidence_threshold=ConfidenceLevel.HIGH)

        kwargs = mock_detect.call_args.kwargs
        assert kwargs["confidence_threshold"] == ConfidenceLevel.HIGH
        assert kwargs["model"] is mock_model

    @patch("strands_evals.detectors.diagnosis.analyze_root_cause_async", new_callable=AsyncMock)
    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    def test_passes_model_to_rca(self, mock_detect, mock_rca):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[_failure("s1")])
        mock_rca.return_value = RCAOutput(root_causes=[])
        mock_model = MagicMock()

        diagnose_session(_make_session(), model=mock_model)

        mock_rca.assert_awaited_once()
        kwargs = mock_rca.call_args.kwargs
        assert kwargs["model"] is mock_model
        assert kwargs["failures"] == [_failure("s1")]

    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    def test_serialization_round_trip(self, mock_detect):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[])

        result = diagnose_session(_make_session())
        dumped = result.model_dump()

        assert dumped["session_id"] == "sess_1"
        assert dumped["failures"] == []
        assert dumped["root_causes"] == []

        restored = DiagnosisResult.model_validate(dumped)
        assert restored == result


class TestDiagnoseSessionAsync:
    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    async def test_no_failures_skips_rca(self, mock_detect):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[])

        result = await diagnose_session_async(_make_session())

        assert isinstance(result, DiagnosisResult)
        assert result.failures == []
        assert result.root_causes == []

    @patch("strands_evals.detectors.diagnosis.analyze_root_cause_async", new_callable=AsyncMock)
    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    async def test_with_failures_runs_rca(self, mock_detect, mock_rca):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[_failure()])
        mock_rca.return_value = RCAOutput(root_causes=[_rca()])

        result = await diagnose_session_async(_make_session())

        assert len(result.failures) == 1
        assert len(result.root_causes) == 1

    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    async def test_threads_max_workers(self, mock_detect):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[])

        await diagnose_session_async(_make_session(), max_workers=3)

        assert mock_detect.call_args.kwargs["max_workers"] == 3


class TestDiagnoseSessionsAsync:
    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    async def test_batch_runs_in_order(self, mock_detect):
        mock_detect.side_effect = [
            FailureOutput(session_id="a", failures=[]),
            FailureOutput(session_id="b", failures=[]),
            FailureOutput(session_id="c", failures=[]),
        ]
        sessions = [_make_session("a"), _make_session("b"), _make_session("c")]

        results = await diagnose_sessions_async(sessions)

        assert len(results) == 3
        ids = [r.session_id for r in results]
        assert ids == ["a", "b", "c"]

    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    async def test_one_session_failure_returns_none_for_that_slot(self, mock_detect):
        mock_detect.side_effect = [
            FailureOutput(session_id="ok1", failures=[]),
            RuntimeError("LLM down"),
            FailureOutput(session_id="ok2", failures=[]),
        ]
        sessions = [_make_session("ok1"), _make_session("bad"), _make_session("ok2")]

        results = await diagnose_sessions_async(sessions)

        assert len(results) == 3
        assert results[0] is not None and results[0].session_id == "ok1"
        assert results[1] is None
        assert results[2] is not None and results[2].session_id == "ok2"

    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    async def test_threads_per_session_workers(self, mock_detect):
        mock_detect.return_value = FailureOutput(session_id="x", failures=[])

        await diagnose_sessions_async([_make_session("x")], per_session_workers=7)

        assert mock_detect.call_args.kwargs["max_workers"] == 7


class TestSyncWrappersFromInsideEventLoop:
    """The sync wrappers must work when called from inside a running loop
    (Jupyter, FastAPI handlers, pytest-asyncio in `auto` mode). Because
    `pyproject.toml` sets `asyncio_mode = "auto"`, every `async def test_...`
    in this suite runs inside an event loop, which is exactly the regression
    case we want to lock in.
    """

    @patch("strands_evals.detectors.diagnosis.detect_failures_async", new_callable=AsyncMock)
    async def test_diagnose_session_works_inside_running_loop(self, mock_detect):
        mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[])

        # The sync wrapper, called from inside this async test, must NOT raise
        # `RuntimeError: asyncio.run() cannot be called from a running event loop`.
        result = diagnose_session(_make_session())

        assert isinstance(result, DiagnosisResult)
        assert result.failures == []
