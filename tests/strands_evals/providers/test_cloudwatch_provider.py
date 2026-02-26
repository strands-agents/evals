"""Tests for CloudWatchProvider — mocked boto3 CloudWatch Logs client."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from strands_evals.providers.cloudwatch_provider import CloudWatchProvider
from strands_evals.providers.exceptions import (
    ProviderError,
    SessionNotFoundError,
)
from strands_evals.types.trace import Session
from tests.strands_evals.cloudwatch_helpers import make_assistant_text_message, make_log_record, make_user_message

# --- Fixtures ---


@pytest.fixture
def mock_logs_client():
    return MagicMock()


@pytest.fixture
def provider(mock_logs_client):
    with patch("boto3.client", return_value=mock_logs_client):
        return CloudWatchProvider(log_group="/test/group")


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
    def test_extract_output_from_agent_response(self, provider):
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
        session = provider._mapper.map_to_session(records, "sess-1")
        output = provider._extract_output(session)
        assert output == "Final response"


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
                input_messages=[make_user_message("q1")],
                output_messages=[make_assistant_text_message("first")],
            ),
            make_log_record(
                trace_id="t2",
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
