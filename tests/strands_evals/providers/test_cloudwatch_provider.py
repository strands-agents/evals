"""Tests for CloudWatchProvider — mocked boto3 CloudWatch Logs client."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from strands_evals.providers.cloudwatch_provider import CloudWatchProvider
from strands_evals.providers.exceptions import (
    ProviderError,
    SessionNotFoundError,
    TraceNotFoundError,
)
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    Session,
    ToolExecutionSpan,
)

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


def _make_log_record(
    trace_id="abc123",
    span_id="span-1",
    input_messages=None,
    output_messages=None,
    session_id="sess-1",
    time_nano=1000000000000000000,
):
    """Build a body-format OTEL log record dict as found in runtime log groups."""
    record = {
        "traceId": trace_id,
        "spanId": span_id,
        "timeUnixNano": time_nano,
        "body": {
            "input": {"messages": input_messages or []},
            "output": {"messages": output_messages or []},
        },
        "attributes": {"session.id": session_id},
    }
    return record


def _make_user_message(text):
    """Build a user input message with double-encoded content."""
    return {"role": "user", "content": {"content": json.dumps([{"text": text}])}}


def _make_assistant_text_message(text):
    """Build an assistant output message with double-encoded text content."""
    return {
        "role": "assistant",
        "content": {"message": json.dumps([{"text": text}]), "finish_reason": "end_turn"},
    }


def _make_assistant_tool_use_message(tool_name, tool_input, tool_use_id):
    """Build an assistant output message with a toolUse block."""
    return {
        "role": "assistant",
        "content": {
            "message": json.dumps([{"toolUse": {"name": tool_name, "input": tool_input, "toolUseId": tool_use_id}}]),
            "finish_reason": "tool_use",
        },
    }


def _make_tool_result_message(tool_use_id, result_text):
    """Build a tool result input message with double-encoded content."""
    return {
        "role": "tool",
        "content": {
            "content": json.dumps([{"toolResult": {"content": [{"text": result_text}], "toolUseId": tool_use_id}}])
        },
    }


# --- Span conversion (body-based parsing) ---


class TestSpanConversion:
    def test_single_record_produces_inference_span(self, provider):
        """One log record produces an InferenceSpan with input/output messages."""
        record = _make_log_record(
            input_messages=[_make_user_message("Hi")],
            output_messages=[_make_assistant_text_message("Hello!")],
        )
        session = provider._build_session("sess-1", [record])
        spans = session.traces[0].spans
        inference_spans = [s for s in spans if isinstance(s, InferenceSpan)]
        assert len(inference_spans) == 1
        assert inference_spans[0].messages[0].content[0].text == "Hi"
        assert inference_spans[0].messages[1].content[0].text == "Hello!"

    def test_record_with_tool_use_and_result(self, provider):
        """toolUse in output + toolResult in next record's input → ToolExecutionSpan."""
        # Record 1: user asks, assistant calls tool
        record1 = _make_log_record(
            trace_id="t1",
            span_id="s1",
            input_messages=[_make_user_message("Calculate 6*7")],
            output_messages=[_make_assistant_tool_use_message("calculator", {"expr": "6*7"}, "tu-1")],
            time_nano=1000,
        )
        # Record 2: tool result comes back, assistant responds
        record2 = _make_log_record(
            trace_id="t1",
            span_id="s2",
            input_messages=[
                _make_user_message("Calculate 6*7"),
                _make_tool_result_message("tu-1", "42"),
            ],
            output_messages=[_make_assistant_text_message("The answer is 42.")],
            time_nano=2000,
        )
        session = provider._build_session("sess-1", [record1, record2])
        tool_spans = [s for s in session.traces[0].spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 1
        assert tool_spans[0].tool_call.name == "calculator"
        assert tool_spans[0].tool_call.arguments == {"expr": "6*7"}
        assert tool_spans[0].tool_result.content == "42"

    def test_agent_invocation_from_trace(self, provider):
        """User prompt from first record, response from last → AgentInvocationSpan."""
        record1 = _make_log_record(
            trace_id="t1",
            span_id="s1",
            input_messages=[_make_user_message("Tell me a joke")],
            output_messages=[_make_assistant_text_message("Why did the chicken cross the road?")],
            time_nano=1000,
        )
        session = provider._build_session("sess-1", [record1])
        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        assert agent_spans[0].user_prompt == "Tell me a joke"
        assert agent_spans[0].agent_response == "Why did the chicken cross the road?"

    def test_agent_invocation_extracts_tools(self, provider):
        """available_tools populated from tool call names in the trace."""
        record1 = _make_log_record(
            trace_id="t1",
            span_id="s1",
            input_messages=[_make_user_message("Search for X")],
            output_messages=[_make_assistant_tool_use_message("web_search", {"q": "X"}, "tu-1")],
            time_nano=1000,
        )
        record2 = _make_log_record(
            trace_id="t1",
            span_id="s2",
            input_messages=[_make_user_message("Search for X"), _make_tool_result_message("tu-1", "found X")],
            output_messages=[_make_assistant_text_message("Here's what I found about X.")],
            time_nano=2000,
        )
        session = provider._build_session("sess-1", [record1, record2])
        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        tool_names = [t.name for t in agent_spans[0].available_tools]
        assert "web_search" in tool_names

    def test_double_encoded_content_parsed(self, provider):
        """Content field is a JSON string that must be parsed to get content blocks."""
        record = _make_log_record(
            input_messages=[_make_user_message("test double encoding")],
            output_messages=[_make_assistant_text_message("parsed correctly")],
        )
        session = provider._build_session("sess-1", [record])
        inference_spans = [s for s in session.traces[0].spans if isinstance(s, InferenceSpan)]
        assert inference_spans[0].messages[0].content[0].text == "test double encoding"
        assert inference_spans[0].messages[1].content[0].text == "parsed correctly"

    def test_tool_call_matched_to_result_by_id(self, provider):
        """toolUseId matching works across records in the same trace."""
        record1 = _make_log_record(
            trace_id="t1",
            span_id="s1",
            input_messages=[_make_user_message("Do two things")],
            output_messages=[
                _make_assistant_tool_use_message("tool_a", {"x": 1}, "tu-a"),
            ],
            time_nano=1000,
        )
        record2 = _make_log_record(
            trace_id="t1",
            span_id="s2",
            input_messages=[
                _make_user_message("Do two things"),
                _make_tool_result_message("tu-a", "result-a"),
            ],
            output_messages=[
                _make_assistant_tool_use_message("tool_b", {"y": 2}, "tu-b"),
            ],
            time_nano=2000,
        )
        record3 = _make_log_record(
            trace_id="t1",
            span_id="s3",
            input_messages=[
                _make_user_message("Do two things"),
                _make_tool_result_message("tu-a", "result-a"),
                _make_tool_result_message("tu-b", "result-b"),
            ],
            output_messages=[_make_assistant_text_message("Both done.")],
            time_nano=3000,
        )
        session = provider._build_session("sess-1", [record1, record2, record3])
        tool_spans = [s for s in session.traces[0].spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) == 2
        tool_span_by_name = {ts.tool_call.name: ts for ts in tool_spans}
        assert tool_span_by_name["tool_a"].tool_result.content == "result-a"
        assert tool_span_by_name["tool_b"].tool_result.content == "result-b"


# --- Session building ---


class TestSessionBuilding:
    def test_multiple_records_grouped_by_trace_id(self, provider):
        """Records with different traceIds become separate Trace objects."""
        records = [
            _make_log_record(
                trace_id="t1",
                input_messages=[_make_user_message("q1")],
                output_messages=[_make_assistant_text_message("a1")],
            ),
            _make_log_record(
                trace_id="t2",
                input_messages=[_make_user_message("q2")],
                output_messages=[_make_assistant_text_message("a2")],
            ),
        ]
        session = provider._build_session("sess-1", records)
        assert session.session_id == "sess-1"
        assert len(session.traces) == 2
        trace_ids = {t.trace_id for t in session.traces}
        assert trace_ids == {"t1", "t2"}

    def test_multi_step_agent_loop(self, provider):
        """user→LLM→tool→LLM→response produces InferenceSpan + ToolExecutionSpan + AgentInvocationSpan."""
        record1 = _make_log_record(
            trace_id="t1",
            span_id="s1",
            input_messages=[_make_user_message("What is 6*7?")],
            output_messages=[_make_assistant_tool_use_message("calculator", {"expr": "6*7"}, "tu-1")],
            time_nano=1000,
        )
        record2 = _make_log_record(
            trace_id="t1",
            span_id="s2",
            input_messages=[
                _make_user_message("What is 6*7?"),
                _make_tool_result_message("tu-1", "42"),
            ],
            output_messages=[_make_assistant_text_message("The answer is 42.")],
            time_nano=2000,
        )
        session = provider._build_session("sess-1", [record1, record2])
        assert len(session.traces) == 1
        spans = session.traces[0].spans
        span_types = [type(s).__name__ for s in spans]
        assert "InferenceSpan" in span_types
        assert "ToolExecutionSpan" in span_types
        assert "AgentInvocationSpan" in span_types

    def test_empty_records_list(self, provider):
        session = provider._build_session("sess-1", [])
        assert session.session_id == "sess-1"
        assert session.traces == []

    def test_extract_output_from_agent_response(self, provider):
        """_extract_output returns last agent response text."""
        records = [
            _make_log_record(
                trace_id="t1",
                span_id="s1",
                input_messages=[_make_user_message("Hi")],
                output_messages=[_make_assistant_text_message("First response")],
                time_nano=1000,
            ),
            _make_log_record(
                trace_id="t1",
                span_id="s2",
                input_messages=[_make_user_message("Hi")],
                output_messages=[_make_assistant_text_message("Final response")],
                time_nano=2000,
            ),
        ]
        session = provider._build_session("sess-1", records)
        output = provider._extract_output(session)
        assert output == "Final response"

    def test_record_with_no_body_skipped(self, provider):
        """Malformed records without body don't crash."""
        records = [
            {"traceId": "t1", "spanId": "s1", "timeUnixNano": 1000},
            _make_log_record(
                trace_id="t1",
                input_messages=[_make_user_message("Hi")],
                output_messages=[_make_assistant_text_message("Hello!")],
                time_nano=2000,
            ),
        ]
        session = provider._build_session("sess-1", records)
        assert len(session.traces) == 1
        assert len(session.traces[0].spans) > 0


# --- CW Logs Insights polling ---


class TestLogsInsightsPolling:
    def _make_record_json(self, trace_id="t1", span_id="s1"):
        """Return a JSON-serialized body-format log record for use in CW Logs @message fields."""
        return json.dumps(
            _make_log_record(
                trace_id=trace_id,
                span_id=span_id,
                input_messages=[_make_user_message("Hi")],
                output_messages=[_make_assistant_text_message("Hello!")],
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
        record1 = _make_log_record(trace_id="t1", span_id="s1")
        record2 = _make_log_record(trace_id="t1", span_id="s2")
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
            _make_log_record(
                trace_id="t1",
                span_id="s1",
                session_id="sess-1",
                input_messages=[_make_user_message("What is 6*7?")],
                output_messages=[_make_assistant_text_message("The answer is 42.")],
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
            _make_log_record(
                session_id="sess-1",
                input_messages=[_make_user_message("Hi")],
                output_messages=[_make_assistant_text_message("Hello")],
            )
        ]
        _setup_query_results(mock_logs_client, records)
        provider.get_evaluation_data("sess-1")
        query_string = mock_logs_client.start_query.call_args[1]["queryString"]
        assert "attributes.session.id" in query_string
        assert "sess-1" in query_string

    def test_multiple_traces(self, provider, mock_logs_client):
        records = [
            _make_log_record(
                trace_id="t1",
                input_messages=[_make_user_message("q1")],
                output_messages=[_make_assistant_text_message("first")],
            ),
            _make_log_record(
                trace_id="t2",
                input_messages=[_make_user_message("q2")],
                output_messages=[_make_assistant_text_message("second")],
            ),
        ]
        _setup_query_results(mock_logs_client, records)
        result = provider.get_evaluation_data("sess-1")
        assert len(result["trajectory"].traces) == 2
        assert result["output"] == "second"

    def test_output_from_last_agent_invocation(self, provider, mock_logs_client):
        records = [
            _make_log_record(
                trace_id="t1",
                span_id="s1",
                input_messages=[_make_user_message("Hi")],
                output_messages=[_make_assistant_text_message("first")],
                time_nano=1000,
            ),
            _make_log_record(
                trace_id="t1",
                span_id="s2",
                input_messages=[_make_user_message("Hi")],
                output_messages=[_make_assistant_text_message("last")],
                time_nano=2000,
            ),
        ]
        _setup_query_results(mock_logs_client, records)
        assert provider.get_evaluation_data("sess-1")["output"] == "last"


# --- get_evaluation_data_by_trace_id ---


class TestGetEvaluationDataByTraceId:
    def test_happy_path(self, provider, mock_logs_client):
        records = [
            _make_log_record(
                trace_id="t1",
                span_id="s1",
                input_messages=[_make_user_message("What is 6*7?")],
                output_messages=[_make_assistant_text_message("The answer is 42.")],
            )
        ]
        _setup_query_results(mock_logs_client, records)
        result = provider.get_evaluation_data_by_trace_id("t1")
        assert isinstance(result["trajectory"], Session)
        assert result["trajectory"].traces[0].trace_id == "t1"
        assert result["output"] == "The answer is 42."

    def test_not_found_raises(self, provider, mock_logs_client):
        _setup_query_results(mock_logs_client, [])
        with pytest.raises(TraceNotFoundError, match="t-missing"):
            provider.get_evaluation_data_by_trace_id("t-missing")

    def test_query_failure_raises(self, provider, mock_logs_client):
        mock_logs_client.start_query.side_effect = Exception("throttled")
        with pytest.raises(ProviderError, match="throttled"):
            provider.get_evaluation_data_by_trace_id("t1")

    def test_query_uses_trace_id_filter(self, provider, mock_logs_client):
        records = [
            _make_log_record(
                trace_id="t-abc",
                input_messages=[_make_user_message("Hi")],
                output_messages=[_make_assistant_text_message("Hello")],
            )
        ]
        _setup_query_results(mock_logs_client, records)
        provider.get_evaluation_data_by_trace_id("t-abc")
        query_string = mock_logs_client.start_query.call_args[1]["queryString"]
        assert "t-abc" in query_string

    def test_session_id_from_record_attributes(self, provider, mock_logs_client):
        """Session ID is taken from record attributes when available."""
        records = [
            _make_log_record(
                trace_id="t1",
                session_id="sess-from-record",
                input_messages=[_make_user_message("Hi")],
                output_messages=[_make_assistant_text_message("Hello")],
            )
        ]
        _setup_query_results(mock_logs_client, records)
        result = provider.get_evaluation_data_by_trace_id("t1")
        assert result["trajectory"].session_id == "sess-from-record"


# --- list_sessions ---


def _setup_session_query(mock_logs_client, session_ids):
    """Wire up mock to return session IDs from a stats aggregation query."""
    mock_logs_client.start_query.return_value = {"queryId": "q-1"}
    results = []
    for sid in session_ids:
        results.append(
            [
                {"field": "sessionId", "value": sid},
                {"field": "span_count", "value": "5"},
            ]
        )
    mock_logs_client.get_query_results.return_value = {
        "status": "Complete",
        "results": results,
    }


class TestListSessions:
    def test_returns_session_ids(self, provider, mock_logs_client):
        _setup_session_query(mock_logs_client, ["s1", "s2", "s3"])
        assert list(provider.list_sessions()) == ["s1", "s2", "s3"]

    def test_empty_results(self, provider, mock_logs_client):
        _setup_session_query(mock_logs_client, [])
        assert list(provider.list_sessions()) == []

    def test_time_filter_applied(self, provider, mock_logs_client):
        from datetime import datetime, timezone

        from strands_evals.providers.trace_provider import SessionFilter

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 31, tzinfo=timezone.utc)
        _setup_session_query(mock_logs_client, ["s1"])
        list(provider.list_sessions(session_filter=SessionFilter(start_time=start, end_time=end)))
        kw = mock_logs_client.start_query.call_args[1]
        assert kw["startTime"] == int(start.timestamp())
        assert kw["endTime"] == int(end.timestamp())

    def test_limit_applied(self, provider, mock_logs_client):
        from strands_evals.providers.trace_provider import SessionFilter

        _setup_session_query(mock_logs_client, ["s1"])
        list(provider.list_sessions(session_filter=SessionFilter(limit=50)))
        query_string = mock_logs_client.start_query.call_args[1]["queryString"]
        assert "limit 50" in query_string

    def test_default_limit(self, provider, mock_logs_client):
        _setup_session_query(mock_logs_client, ["s1"])
        list(provider.list_sessions())
        query_string = mock_logs_client.start_query.call_args[1]["queryString"]
        assert "limit 1000" in query_string

    def test_query_failure_raises(self, provider, mock_logs_client):
        mock_logs_client.start_query.side_effect = Exception("access denied")
        with pytest.raises(ProviderError, match="access denied"):
            list(provider.list_sessions())
