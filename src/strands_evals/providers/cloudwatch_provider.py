"""CloudWatch trace provider for retrieving agent traces from AWS CloudWatch Logs."""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3

from ..mappers.cloudwatch_parser import CloudWatchLogsParser
from ..mappers.session_mapper import SessionMapper
from ..mappers.utils import detect_otel_mapper
from ..providers.exceptions import ProviderError, SessionNotFoundError
from ..providers.trace_provider import TraceProvider
from ..types.evaluation import TaskOutput
from ..types.trace import (
    AgentInvocationSpan,
    Session,
)

logger = logging.getLogger(__name__)


class CloudWatchProvider(TraceProvider):
    """Retrieves agent trace data from AWS CloudWatch Logs for evaluation.

    Queries CloudWatch Logs Insights to fetch OTEL log records from an
    agent-specific runtime log group, parses body.input/output messages,
    and returns Session objects ready for the evaluation pipeline.
    """

    SPANS_LOG_GROUP = "aws/spans"

    def __init__(
        self,
        region: str | None = None,
        log_group: str | None = None,
        agent_name: str | None = None,
        lookback_days: int = 30,
        query_timeout_seconds: float = 60.0,
        mapper: SessionMapper | None = None,
        end_time: datetime | None = None,
    ):
        """Initialize the CloudWatch provider.

        The log group can be specified directly or discovered automatically
        from an agent name. Region falls back to AWS_REGION, then
        AWS_DEFAULT_REGION, then `us-east-1`.

        Example::

            from strands_evals.providers import CloudWatchProvider

            # Explicit log group — auto-detects framework from span scope
            provider = CloudWatchProvider(
                log_group="/aws/bedrock-agentcore/runtimes/my-agent-abc123-DEFAULT",
            )

            # Discover log group from agent name
            provider = CloudWatchProvider(agent_name="my-agent")

            # Override mapper for a specific framework
            from strands_evals.mappers import LangChainOtelSessionMapper
            provider = CloudWatchProvider(
                log_group="/aws/...",
                mapper=LangChainOtelSessionMapper(),
            )

            # Narrow the query window with an explicit end time
            from datetime import datetime, timezone
            provider = CloudWatchProvider(
                log_group="/aws/...",
                end_time=datetime(2026, 3, 01, 12, 0, 0, tzinfo=timezone.utc),
            )

        Args:
            region: AWS region. Falls back to AWS_REGION / AWS_DEFAULT_REGION env vars.
            log_group: Full CloudWatch log group path.
            agent_name: Agent name used to discover the runtime log group via
                `describe_log_groups`. Exactly one of `log_group` or
                `agent_name` must be provided.
            lookback_days: How many days back to search for traces.
            query_timeout_seconds: Maximum seconds to wait for a Logs Insights query.
            mapper: Optional SessionMapper to use for converting spans to Session.
                If not provided, the mapper is auto-detected from the span data
                using ``detect_otel_mapper()``, which inspects ``scope.name``
                to determine the correct framework mapper.
            end_time: Upper bound for the query window. Defaults to now (UTC).
                Combined with ``lookback_days`` to form the full time range.

        Raises:
            ProviderError: If neither `log_group` nor `agent_name` is provided,
                or if the boto3 client cannot be created.
        """
        resolved_region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        try:
            self._client = boto3.client("logs", region_name=resolved_region)
        except Exception as e:
            raise ProviderError(f"CloudWatch: failed to create boto3 logs client: {e}") from e

        if log_group:
            self._log_group = log_group
        elif agent_name:
            self._log_group = self._discover_log_group(agent_name)
        else:
            raise ProviderError("CloudWatch: either log_group or agent_name must be provided")

        self._lookback_days = lookback_days
        self._query_timeout_seconds = query_timeout_seconds
        self._mapper = mapper
        self._end_time = end_time

    def _discover_log_group(self, agent_name: str) -> str:
        """Discover the runtime log group for an agent via describe_log_groups."""
        prefix = f"/aws/bedrock-agentcore/runtimes/{agent_name}"
        response = self._client.describe_log_groups(logGroupNamePrefix=prefix)
        log_groups = response.get("logGroups", [])
        if not log_groups:
            raise ProviderError(f"CloudWatch: no log group found for agent_name='{agent_name}' (prefix={prefix})")
        return log_groups[0]["logGroupName"]

    def get_evaluation_data(self, session_id: str) -> TaskOutput:
        """Fetch all traces for a session and return evaluation data.

        Queries two CloudWatch log groups:

        1. **Runtimes log group** — event records with ``body.input/output.messages``
        2. **aws/spans** — lightweight hierarchy lookup (``spanId → parentSpanId``)
           to enrich the events with proper parent-child relationships.
        """
        query = (
            f"fields @message | filter attributes.session.id like '{session_id}' | sort @timestamp asc | limit 10000"
        )

        try:
            raw_records = self._run_logs_insights_query(query)
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"CloudWatch: failed to query spans for session '{session_id}': {e}") from e

        if not raw_records:
            raise SessionNotFoundError(f"CloudWatch: no spans found for session_id='{session_id}'")

        # Normalize raw CloudWatch log records into the standard span format.
        span_dicts = CloudWatchLogsParser(raw_records).parse()

        if not span_dicts:
            raise SessionNotFoundError(f"CloudWatch: no normalizable spans found for session_id='{session_id}'")

        # Enrich with hierarchy from aws/spans (best-effort).
        hierarchy = self._fetch_span_hierarchy(session_id)
        if hierarchy:
            for span in span_dicts:
                parent = hierarchy.get(span.get("span_id", ""))
                if parent:
                    span["parent_span_id"] = parent

        mapper = self._mapper or detect_otel_mapper(span_dicts)
        session = mapper.map_to_session(span_dicts, session_id)

        if not session.traces:
            raise SessionNotFoundError(
                f"CloudWatch: spans found for session_id='{session_id}' but none contained convertible spans"
            )

        output = self._extract_output(session)
        return TaskOutput(output=output, trajectory=session)

    def _fetch_span_hierarchy(self, session_id: str) -> dict[str, str]:
        """Fetch spanId→parentSpanId map from aws/spans. Returns empty dict on failure."""
        query = f"fields spanId, parentSpanId | filter attributes.session.id like '{session_id}' | limit 10000"
        end = self._end_time or datetime.now(tz=timezone.utc)
        try:
            response = self._client.start_query(
                logGroupName=self.SPANS_LOG_GROUP,
                startTime=int((end - timedelta(days=self._lookback_days)).timestamp()),
                endTime=int(end.timestamp()),
                queryString=query,
            )
            raw_results = self._poll_query_results(response["queryId"])
        except Exception:
            logger.debug("CloudWatch: aws/spans hierarchy query failed, continuing without hierarchy")
            return {}

        hierarchy: dict[str, str] = {}
        for row in raw_results:
            fields = {f["field"]: f["value"] for f in row}
            span_id = fields.get("spanId", "")
            parent_span_id = fields.get("parentSpanId", "")
            if span_id and parent_span_id:
                hierarchy[span_id] = parent_span_id

        return hierarchy

    # --- Internal: CW Logs Insights query execution ---

    def _run_logs_insights_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a CW Logs Insights query and return parsed span dicts from @message fields."""
        end = self._end_time or datetime.now(tz=timezone.utc)
        start_time = end - timedelta(days=self._lookback_days)

        try:
            response = self._client.start_query(
                logGroupName=self._log_group,
                startTime=int(start_time.timestamp()),
                endTime=int(end.timestamp()),
                queryString=query,
            )
        except Exception as e:
            raise ProviderError(f"CloudWatch: failed to start query: {e}") from e

        query_id = response["queryId"]
        raw_results = self._poll_query_results(query_id)
        return self._parse_query_results(raw_results)

    def _poll_query_results(self, query_id: str) -> list[list[dict[str, str]]]:
        """Poll for query completion with exponential backoff. Returns raw result rows."""
        delay = 0.5
        max_delay = 8.0
        deadline = time.monotonic() + self._query_timeout_seconds

        while True:
            response = self._client.get_query_results(queryId=query_id)
            status = response.get("status", "")

            if status == "Complete":
                return response.get("results", [])
            elif status in ("Failed", "Cancelled", "Timeout"):
                raise ProviderError(f"CloudWatch: query {status}")

            if time.monotonic() >= deadline:
                raise ProviderError(f"CloudWatch: query timed out after {self._query_timeout_seconds}s")

            time.sleep(delay)
            delay = min(delay * 2, max_delay)

    @staticmethod
    def _parse_query_results(results: list[list[dict[str, str]]]) -> list[dict[str, Any]]:
        """Parse @message fields from CW Logs Insights results into span dicts."""
        span_dicts = []
        for row in results:
            for field in row:
                if field.get("field") == "@message":
                    try:
                        span_dicts.append(json.loads(field["value"]))
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Failed to parse @message: %s", e)
        return span_dicts

    # --- Internal: output extraction ---

    def _extract_output(self, session: Session) -> str:
        """Extract the final agent response from the session for TaskOutput.output."""
        for trace in reversed(session.traces):
            for span in reversed(trace.spans):
                if isinstance(span, AgentInvocationSpan):
                    return span.agent_response
        return ""
