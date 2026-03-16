"""Tests for CloudWatchProvider — mocked boto3 CloudWatch Logs client."""

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from strands_evals.mappers import LangChainOtelSessionMapper, OpenInferenceSessionMapper
from strands_evals.providers.cloudwatch_provider import CloudWatchProvider
from strands_evals.providers.exceptions import (
    ProviderError,
    SessionNotFoundError,
)
from strands_evals.types.trace import AgentInvocationSpan, Session, SpanInfo, Trace
from tests.strands_evals.cloudwatch_helpers import make_assistant_text_message, make_log_record, make_user_message

# --- Helpers ---


def _session_with_agent_span(session_id: str, trace_id: str, user_prompt: str, agent_response: str) -> Session:
    """Build a minimal valid Session that passes CloudWatchProvider's post-mapping checks."""
    now = datetime.now(timezone.utc)
    span = AgentInvocationSpan(
        span_info=SpanInfo(session_id=session_id, trace_id=trace_id, span_id="s1", start_time=now, end_time=now),
        user_prompt=user_prompt,
        agent_response=agent_response,
        available_tools=[],
    )
    return Session(session_id=session_id, traces=[Trace(trace_id=trace_id, session_id=session_id, spans=[span])])


# --- Fixtures ---


@pytest.fixture
def mock_logs_client():
    return MagicMock()


@pytest.fixture
def provider(mock_logs_client):
    with patch("boto3.client", return_value=mock_logs_client):
        p = CloudWatchProvider(log_group="/test/group")
    # Skip hierarchy lookup for tests that only exercise the events path
    p._fetch_span_hierarchy = lambda session_id: {}
    return p


# --- Constructor ---


class TestConstructor:
    def test_explicit_log_group(self, mock_logs_client):
        with patch("boto3.client", return_value=mock_logs_client):
            p = CloudWatchProvider(log_group="/custom/group")
            assert p._log_group == "/custom/group"
            assert p._lookback_days == 30
            assert p._query_timeout_seconds == 60.0

    def test_custom_params(self, mock_logs_client):
        with patch("boto3.client", return_value=mock_logs_client) as mock_boto:
            p = CloudWatchProvider(
                region="eu-west-1",
                log_group="/custom/log-group",
                lookback_days=7,
                query_timeout_seconds=120.0,
            )
            mock_boto.assert_called_once_with("logs", region_name="eu-west-1")
            assert p._log_group == "/custom/log-group"
            assert p._lookback_days == 7
            assert p._query_timeout_seconds == 120.0

    def test_agent_name_discovery(self, mock_logs_client):
        mock_logs_client.describe_log_groups.return_value = {
            "logGroups": [{"logGroupName": "/aws/bedrock-agentcore/runtimes/my-agent-abc-DEFAULT"}]
        }
        with patch("boto3.client", return_value=mock_logs_client):
            p = CloudWatchProvider(agent_name="my-agent")
            assert p._log_group == "/aws/bedrock-agentcore/runtimes/my-agent-abc-DEFAULT"
            mock_logs_client.describe_log_groups.assert_called_once_with(
                logGroupNamePrefix="/aws/bedrock-agentcore/runtimes/my-agent"
            )

    def test_agent_name_no_match_raises(self, mock_logs_client):
        mock_logs_client.describe_log_groups.return_value = {"logGroups": []}
        with (
            patch("boto3.client", return_value=mock_logs_client),
            pytest.raises(ProviderError, match="no log group found"),
        ):
            CloudWatchProvider(agent_name="nonexistent-agent")

    def test_neither_log_group_nor_agent_name_raises(self, mock_logs_client):
        with (
            patch("boto3.client", return_value=mock_logs_client),
            pytest.raises(ProviderError, match="log_group.*agent_name"),
        ):
            CloudWatchProvider()

    def test_region_from_aws_region_env(self, mock_logs_client):
        env = os.environ.copy()
        env.pop("AWS_DEFAULT_REGION", None)
        env["AWS_REGION"] = "ap-southeast-1"
        with (
            patch.dict(os.environ, env, clear=True),
            patch("boto3.client", return_value=mock_logs_client) as mock_boto,
        ):
            CloudWatchProvider(log_group="/test/group")
            mock_boto.assert_called_once_with("logs", region_name="ap-southeast-1")

    def test_region_from_aws_default_region_env(self, mock_logs_client):
        env = os.environ.copy()
        env.pop("AWS_REGION", None)
        env["AWS_DEFAULT_REGION"] = "us-west-2"
        with (
            patch.dict(os.environ, env, clear=True),
            patch("boto3.client", return_value=mock_logs_client) as mock_boto,
        ):
            CloudWatchProvider(log_group="/test/group")
            mock_boto.assert_called_once_with("logs", region_name="us-west-2")

    def test_aws_region_takes_precedence_over_default_region(self, mock_logs_client):
        env = {"AWS_REGION": "eu-central-1", "AWS_DEFAULT_REGION": "us-west-2"}
        with (
            patch.dict(os.environ, env),
            patch("boto3.client", return_value=mock_logs_client) as mock_boto,
        ):
            CloudWatchProvider(log_group="/test/group")
            mock_boto.assert_called_once_with("logs", region_name="eu-central-1")

    def test_explicit_region_overrides_env(self, mock_logs_client):
        env = {"AWS_REGION": "eu-central-1"}
        with (
            patch.dict(os.environ, env),
            patch("boto3.client", return_value=mock_logs_client) as mock_boto,
        ):
            CloudWatchProvider(region="ca-central-1", log_group="/test/group")
            mock_boto.assert_called_once_with("logs", region_name="ca-central-1")

    def test_boto3_client_creation_failure(self):
        with (
            patch("boto3.client", side_effect=Exception("bad credentials")),
            pytest.raises(ProviderError, match="bad credentials"),
        ):
            CloudWatchProvider(log_group="/test/group")


# --- Helpers for body-format log records ---


def _setup_query_results(mock_logs_client, records):
    """Wire up mock to return records from a CW Logs Insights query."""
    mock_logs_client.start_query.return_value = {"queryId": "q-1"}
    mock_logs_client.get_query_results.return_value = {
        "status": "Complete",
        "results": [[{"field": "@message", "value": json.dumps(r)}] for r in records],
    }


# --- Output extraction ---


class TestExtractOutput:
    def test_extract_output_from_agent_response(self, provider, mock_logs_client):
        """_extract_output returns last agent response text."""
        records = [
            make_log_record(
                trace_id="t1",
                span_id="s1",
                input_messages=[make_user_message("Hi")],
                output_messages=[make_assistant_text_message("First response")],
                time_nano=1000,
            ),
            make_log_record(
                trace_id="t1",
                span_id="s2",
                input_messages=[make_user_message("Hi")],
                output_messages=[make_assistant_text_message("Final response")],
                time_nano=2000,
            ),
        ]
        _setup_query_results(mock_logs_client, records)
        result = provider.get_evaluation_data("sess-1")
        assert result["output"] == "Final response"


# --- CW Logs Insights polling ---


class TestLogsInsightsPolling:
    def _make_record_json(self, trace_id="t1", span_id="s1"):
        """Return a JSON-serialized body-format log record for use in CW Logs @message fields."""
        return json.dumps(
            make_log_record(
                trace_id=trace_id,
                span_id=span_id,
                input_messages=[make_user_message("Hi")],
                output_messages=[make_assistant_text_message("Hello!")],
            )
        )

    def test_happy_path(self, provider, mock_logs_client):
        """Query starts, one poll, completes with results."""
        mock_logs_client.start_query.return_value = {"queryId": "q-1"}
        mock_logs_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [
                [{"field": "@message", "value": self._make_record_json()}],
            ],
        }
        results = provider._run_logs_insights_query("fields @message")
        assert len(results) == 1
        assert results[0]["traceId"] == "t1"

    def test_polls_through_intermediate_statuses(self, provider, mock_logs_client):
        """Query goes through Scheduled → Running → Complete."""
        mock_logs_client.start_query.return_value = {"queryId": "q-1"}
        mock_logs_client.get_query_results.side_effect = [
            {"status": "Scheduled", "results": []},
            {"status": "Running", "results": []},
            {
                "status": "Complete",
                "results": [
                    [{"field": "@message", "value": self._make_record_json()}],
                ],
            },
        ]
        with patch("time.sleep"):
            results = provider._run_logs_insights_query("fields @message")
        assert len(results) == 1

    def test_failed_status_raises(self, provider, mock_logs_client):
        mock_logs_client.start_query.return_value = {"queryId": "q-1"}
        mock_logs_client.get_query_results.return_value = {
            "status": "Failed",
            "results": [],
        }
        with pytest.raises(ProviderError, match="Failed"):
            provider._run_logs_insights_query("fields @message")

    def test_timeout_raises(self, provider, mock_logs_client):
        """If query doesn't complete within timeout, raises ProviderError."""
        mock_logs_client.start_query.return_value = {"queryId": "q-1"}
        mock_logs_client.get_query_results.return_value = {
            "status": "Running",
            "results": [],
        }
        provider._query_timeout_seconds = 0.01
        with (
            patch("time.sleep"),
            patch("time.monotonic", side_effect=[0.0, 0.0, 1.0]),
            pytest.raises(ProviderError, match="timed out"),
        ):
            provider._run_logs_insights_query("fields @message")

    def test_empty_results(self, provider, mock_logs_client):
        mock_logs_client.start_query.return_value = {"queryId": "q-1"}
        mock_logs_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [],
        }
        results = provider._run_logs_insights_query("fields @message")
        assert results == []

    def test_parses_message_field(self, provider, mock_logs_client):
        """Each result row's @message field is parsed as JSON into a record dict."""
        record1 = make_log_record(trace_id="t1", span_id="s1")
        record2 = make_log_record(trace_id="t1", span_id="s2")
        mock_logs_client.start_query.return_value = {"queryId": "q-1"}
        mock_logs_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [
                [{"field": "@message", "value": json.dumps(record1)}],
                [{"field": "@message", "value": json.dumps(record2)}],
            ],
        }
        results = provider._run_logs_insights_query("fields @message")
        assert len(results) == 2
        assert results[0]["spanId"] == "s1"
        assert results[1]["spanId"] == "s2"

    def test_start_query_failure_raises(self, provider, mock_logs_client):
        mock_logs_client.start_query.side_effect = Exception("access denied")
        with pytest.raises(ProviderError, match="access denied"):
            provider._run_logs_insights_query("fields @message")


# --- get_evaluation_data ---


class TestGetEvaluationData:
    def test_happy_path(self, provider, mock_logs_client):
        records = [
            make_log_record(
                trace_id="t1",
                span_id="s1",
                session_id="sess-1",
                input_messages=[make_user_message("What is 6*7?")],
                output_messages=[make_assistant_text_message("The answer is 42.")],
            )
        ]
        _setup_query_results(mock_logs_client, records)

        result = provider.get_evaluation_data("sess-1")
        assert isinstance(result["trajectory"], Session)
        assert result["trajectory"].session_id == "sess-1"
        assert len(result["trajectory"].traces) == 1
        assert result["output"] == "The answer is 42."

    def test_no_results_raises_session_not_found(self, provider, mock_logs_client):
        _setup_query_results(mock_logs_client, [])
        with pytest.raises(SessionNotFoundError, match="sess-missing"):
            provider.get_evaluation_data("sess-missing")

    def test_query_failure_raises_provider_error(self, provider, mock_logs_client):
        mock_logs_client.start_query.side_effect = Exception("throttled")
        with pytest.raises(ProviderError, match="throttled"):
            provider.get_evaluation_data("sess-1")

    def test_query_uses_session_id_filter(self, provider, mock_logs_client):
        """Verify the query string uses attributes.session.id filter."""
        records = [
            make_log_record(
                session_id="sess-1",
                input_messages=[make_user_message("Hi")],
                output_messages=[make_assistant_text_message("Hello")],
            )
        ]
        _setup_query_results(mock_logs_client, records)
        provider.get_evaluation_data("sess-1")
        query_string = mock_logs_client.start_query.call_args[1]["queryString"]
        assert "attributes.session.id" in query_string
        assert "sess-1" in query_string

    def test_multiple_traces(self, provider, mock_logs_client):
        records = [
            make_log_record(
                trace_id="t1",
                span_id="s1",
                input_messages=[make_user_message("q1")],
                output_messages=[make_assistant_text_message("first")],
            ),
            make_log_record(
                trace_id="t2",
                span_id="s2",
                input_messages=[make_user_message("q2")],
                output_messages=[make_assistant_text_message("second")],
            ),
        ]
        _setup_query_results(mock_logs_client, records)
        result = provider.get_evaluation_data("sess-1")
        assert len(result["trajectory"].traces) == 2
        assert result["output"] == "second"

    def test_output_from_last_agent_invocation(self, provider, mock_logs_client):
        records = [
            make_log_record(
                trace_id="t1",
                span_id="s1",
                input_messages=[make_user_message("Hi")],
                output_messages=[make_assistant_text_message("first")],
                time_nano=1000,
            ),
            make_log_record(
                trace_id="t1",
                span_id="s2",
                input_messages=[make_user_message("Hi")],
                output_messages=[make_assistant_text_message("last")],
                time_nano=2000,
            ),
        ]
        _setup_query_results(mock_logs_client, records)
        assert provider.get_evaluation_data("sess-1")["output"] == "last"


# --- Mapper auto-detection ---


class TestMapperAutoDetection:
    def test_default_mapper_is_none(self, mock_logs_client):
        """When no mapper is provided, auto-detection is used."""
        with patch("boto3.client", return_value=mock_logs_client):
            p = CloudWatchProvider(log_group="/test/group")
            assert p._mapper is None

    def test_explicit_mapper_override(self, mock_logs_client):
        """User can provide a specific mapper to skip auto-detection."""
        with patch("boto3.client", return_value=mock_logs_client):
            mapper = LangChainOtelSessionMapper()
            p = CloudWatchProvider(log_group="/test/group", mapper=mapper)
            assert p._mapper is mapper

    def test_auto_detects_strands_mapper_for_strands_spans(self, mock_logs_client):
        """Strands body-format spans auto-detect to CloudWatchSessionMapper."""
        records = [
            make_log_record(
                trace_id="t1",
                input_messages=[make_user_message("Hi")],
                output_messages=[make_assistant_text_message("Hello!")],
            )
        ]
        _setup_query_results(mock_logs_client, records)

        with patch("boto3.client", return_value=mock_logs_client):
            p = CloudWatchProvider(log_group="/test/group")
            p._fetch_span_hierarchy = lambda session_id: {}
            result = p.get_evaluation_data("sess-1")
            assert isinstance(result["trajectory"], Session)
            assert result["output"] == "Hello!"

    def test_auto_detects_langchain_otel_mapper(self, mock_logs_client):
        """Spans with opentelemetry.instrumentation.langchain scope trigger LangChainOtelSessionMapper."""
        span = {
            "trace_id": "t1",
            "span_id": "s1",
            "scope": {"name": "opentelemetry.instrumentation.langchain", "version": "0.1"},
            "attributes": {"traceloop.span.kind": "workflow"},
            "span_events": [],
        }
        _setup_query_results(mock_logs_client, [span])

        with patch("boto3.client", return_value=mock_logs_client):
            p = CloudWatchProvider(log_group="/test/group")
            p._fetch_span_hierarchy = lambda session_id: {}
            with patch.object(LangChainOtelSessionMapper, "map_to_session") as mock_map:
                mock_map.return_value = _session_with_agent_span("sess-1", "t1", "Hello", "Hi!")
                result = p.get_evaluation_data("sess-1")
                mock_map.assert_called_once()
                assert result["output"] == "Hi!"

    def test_auto_detects_openinference_mapper(self, mock_logs_client):
        """Spans with openinference.instrumentation.langchain scope trigger OpenInferenceSessionMapper."""
        span = {
            "trace_id": "t1",
            "span_id": "s1",
            "scope": {"name": "openinference.instrumentation.langchain", "version": "0.1"},
            "attributes": {"openinference.span.kind": "CHAIN"},
            "span_events": [],
        }
        _setup_query_results(mock_logs_client, [span])

        with patch("boto3.client", return_value=mock_logs_client):
            p = CloudWatchProvider(log_group="/test/group")
            p._fetch_span_hierarchy = lambda session_id: {}
            with patch.object(OpenInferenceSessionMapper, "map_to_session") as mock_map:
                mock_map.return_value = _session_with_agent_span("sess-1", "t1", "Hello", "Hi!")
                result = p.get_evaluation_data("sess-1")
                mock_map.assert_called_once()
                assert result["output"] == "Hi!"


# --- Span hierarchy (aws/spans) ---


def _make_hierarchy_row(span_id: str, parent_span_id: str) -> list[dict[str, str]]:
    """Build a Logs Insights result row with spanId and parentSpanId fields."""
    return [
        {"field": "spanId", "value": span_id},
        {"field": "parentSpanId", "value": parent_span_id},
    ]


class TestSpanHierarchy:
    def test_hierarchy_enriches_parent_span_id(self, mock_logs_client):
        """Events get parent_span_id from aws/spans hierarchy lookup."""
        events = [
            make_log_record(
                trace_id="t1",
                span_id="s1",
                input_messages=[make_user_message("Hi")],
                output_messages=[make_assistant_text_message("Hello!")],
            ),
            make_log_record(
                trace_id="t1",
                span_id="s2",
                input_messages=[make_user_message("More")],
                output_messages=[make_assistant_text_message("Done")],
            ),
        ]
        hierarchy_rows = [
            _make_hierarchy_row("s1", "root"),
            _make_hierarchy_row("s2", "s1"),
        ]

        # First query → events, second query → hierarchy
        call_count = 0

        def start_query(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"queryId": f"q-{call_count}"}

        def get_results(**kwargs):
            qid = kwargs["queryId"]
            if qid == "q-1":  # events
                return {
                    "status": "Complete",
                    "results": [[{"field": "@message", "value": json.dumps(e)}] for e in events],
                }
            else:  # hierarchy
                return {"status": "Complete", "results": hierarchy_rows}

        mock_logs_client.start_query.side_effect = start_query
        mock_logs_client.get_query_results.side_effect = get_results

        with patch("boto3.client", return_value=mock_logs_client):
            p = CloudWatchProvider(log_group="/test/group")
            p.get_evaluation_data("sess-1")
            # Verify hierarchy query was made
            assert mock_logs_client.start_query.call_count == 2
            # Second query targets aws/spans
            second_call = mock_logs_client.start_query.call_args_list[1][1]
            assert second_call["logGroupName"] == "aws/spans"
            assert "spanId, parentSpanId" in second_call["queryString"]

    def test_hierarchy_failure_still_works(self, mock_logs_client):
        """If aws/spans query fails, events still produce a valid session."""
        events = [
            make_log_record(
                trace_id="t1",
                span_id="s1",
                input_messages=[make_user_message("Hi")],
                output_messages=[make_assistant_text_message("Works!")],
            ),
        ]
        call_count = 0

        def start_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # hierarchy query
                raise Exception("access denied to aws/spans")
            return {"queryId": "q-1"}

        mock_logs_client.start_query.side_effect = start_query
        mock_logs_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [[{"field": "@message", "value": json.dumps(e)}] for e in events],
        }

        with patch("boto3.client", return_value=mock_logs_client):
            p = CloudWatchProvider(log_group="/test/group")
            result = p.get_evaluation_data("sess-1")
            assert result["output"] == "Works!"

    def test_hierarchy_empty_is_fine(self, mock_logs_client):
        """Empty hierarchy (no spans in aws/spans) doesn't break anything."""
        events = [
            make_log_record(
                trace_id="t1",
                span_id="s1",
                input_messages=[make_user_message("Hi")],
                output_messages=[make_assistant_text_message("OK")],
            ),
        ]
        call_count = 0

        def start_query(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"queryId": f"q-{call_count}"}

        def get_results(**kwargs):
            qid = kwargs["queryId"]
            if qid == "q-1":
                return {
                    "status": "Complete",
                    "results": [[{"field": "@message", "value": json.dumps(e)}] for e in events],
                }
            else:
                return {"status": "Complete", "results": []}

        mock_logs_client.start_query.side_effect = start_query
        mock_logs_client.get_query_results.side_effect = get_results

        with patch("boto3.client", return_value=mock_logs_client):
            p = CloudWatchProvider(log_group="/test/group")
            result = p.get_evaluation_data("sess-1")
            assert result["output"] == "OK"
