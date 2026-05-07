"""Tests for OpenSearchProvider."""

from unittest.mock import MagicMock, patch

import pytest

from strands_evals.providers.exceptions import ProviderError, SessionNotFoundError
from strands_evals.providers.opensearch_provider import OpenSearchProvider
from tests.strands_evals.opensearch_helpers import make_agent_span, make_tool_span


# Mock the genai-sdk types
class MockSessionRecord:
    def __init__(self, session_id="sess-1", traces=None, truncated=False):
        self.session_id = session_id
        self.traces = traces or []
        self.truncated = truncated


class MockTraceRecord:
    def __init__(self, trace_id="trace-1", spans=None):
        self.trace_id = trace_id
        self.spans = spans or []


@pytest.fixture
def mock_retriever():
    return MagicMock()


@pytest.fixture
def provider(mock_retriever):
    with patch(
        "strands_evals.providers.opensearch_provider.OpenSearchProvider.__init__",
        lambda self, **kwargs: None,
    ):
        p = OpenSearchProvider()
        p._retriever = mock_retriever
        from strands_evals.mappers.opensearch_session_mapper import OpenSearchSessionMapper

        p._mapper = OpenSearchSessionMapper()
        return p


class TestGetEvaluationData:
    def test_happy_path(self, provider, mock_retriever):
        """Returns TaskOutput with output and trajectory from a valid trace."""
        spans = [
            make_agent_span(user_prompt="Hello", agent_response="Hi there"),
            make_tool_span(tool_name="greet"),
        ]
        mock_retriever.get_traces.return_value = MockSessionRecord(traces=[MockTraceRecord(spans=spans)])

        result = provider.get_evaluation_data("sess-1")

        assert result["output"] == "Hi there"
        assert result["trajectory"] is not None
        assert len(result["trajectory"].traces) == 1

    def test_no_traces_raises_session_not_found(self, provider, mock_retriever):
        """Raises SessionNotFoundError when no traces exist for the session."""
        mock_retriever.get_traces.return_value = MockSessionRecord(traces=[])

        with pytest.raises(SessionNotFoundError, match="No traces found"):
            provider.get_evaluation_data("nonexistent")

    def test_retriever_error_raises_provider_error(self, provider, mock_retriever):
        """Wraps retriever exceptions in ProviderError."""
        mock_retriever.get_traces.side_effect = ConnectionError("connection refused")

        with pytest.raises(ProviderError, match="Failed to query OpenSearch"):
            provider.get_evaluation_data("sess-1")


class TestMultipleTraces:
    def test_spans_from_multiple_traces_are_regrouped(self, provider, mock_retriever):
        """Spans from multiple TraceRecords are flattened and re-grouped by trace_id."""
        mock_retriever.get_traces.return_value = MockSessionRecord(
            traces=[
                MockTraceRecord(
                    trace_id="t1",
                    spans=[
                        make_agent_span(trace_id="t1", span_id="a1", user_prompt="Q1", agent_response="A1"),
                    ],
                ),
                MockTraceRecord(
                    trace_id="t2",
                    spans=[
                        make_agent_span(trace_id="t2", span_id="a2", user_prompt="Q2", agent_response="A2"),
                    ],
                ),
            ]
        )

        result = provider.get_evaluation_data("sess-1")
        session = result["trajectory"]
        assert len(session.traces) == 2
        trace_ids = {t.trace_id for t in session.traces}
        assert trace_ids == {"t1", "t2"}


class TestExtractOutput:
    def test_extracts_last_agent_response(self, provider, mock_retriever):
        """Output comes from the last AgentInvocationSpan's response."""
        spans = [
            make_agent_span(span_id="a1", agent_response="First", start_time="2026-01-01T00:00:00Z"),
            make_agent_span(span_id="a2", agent_response="Final answer", start_time="2026-01-01T00:00:01Z"),
        ]
        mock_retriever.get_traces.return_value = MockSessionRecord(traces=[MockTraceRecord(spans=spans)])

        result = provider.get_evaluation_data("sess-1")
        assert result["output"] == "Final answer"

    def test_empty_output_when_no_agent_spans(self, provider, mock_retriever):
        """Returns empty string when there are no AgentInvocationSpans."""
        spans = [make_tool_span()]
        mock_retriever.get_traces.return_value = MockSessionRecord(traces=[MockTraceRecord(spans=spans)])

        result = provider.get_evaluation_data("sess-1")
        assert result["output"] == ""


class TestImportError:
    def test_missing_sdk_raises_import_error(self):
        """Raises ImportError with install instructions when SDK is missing."""
        with patch.dict("sys.modules", {"opensearch_genai_observability_sdk_py": None}):
            with pytest.raises(ImportError, match="opensearch-genai-observability-sdk-py is required"):
                OpenSearchProvider(host="https://localhost:9200")
