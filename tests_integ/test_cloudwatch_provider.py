"""Integration tests for CloudWatchProvider against real CloudWatch Logs data.

Requires AWS credentials with CloudWatch Logs read access. Uses the
'strands_github_bot' AWS profile if available, otherwise falls back to default credentials.
Run with: pytest tests_integ/test_cloudwatch_provider.py -v
"""

import boto3
import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import (
    CoherenceEvaluator,
    HelpfulnessEvaluator,
    OutputEvaluator,
)
from strands_evals.providers.cloudwatch_provider import CloudWatchProvider
from strands_evals.providers.exceptions import (
    ProviderError,
    SessionNotFoundError,
)
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    Session,
    ToolExecutionSpan,
    Trace,
)

LOG_GROUP = "/aws/bedrock-agentcore/runtimes/github_issue_handler-zf6fZR2saQ-DEFAULT"

KNOWN_SESSION_IDS = [
    "github_issue_68_20260218_172558_d27beb07",
    "github-issue-68-1771435544179-d5gc8kk4xan",
]

EXPECTED_ACCOUNT_ID = "249746592913"
AWS_PROFILE = "strands_github_bot"
AWS_REGION = "us-east-1"


def _create_logs_client() -> boto3.client | None:
    """Try the named profile first, then fall back to default credentials."""
    for profile in [AWS_PROFILE, None]:
        try:
            session = boto3.Session(profile_name=profile, region_name=AWS_REGION)
            sts = session.client("sts")
            identity = sts.get_caller_identity()
            if identity["Account"] == EXPECTED_ACCOUNT_ID:
                return session.client("logs")
        except Exception:
            continue
    return None


@pytest.fixture(scope="module")
def provider():
    """Create a CloudWatchProvider targeting the correct AWS account."""
    client = _create_logs_client()
    if client is None:
        pytest.skip(f"No AWS credentials found for account {EXPECTED_ACCOUNT_ID}")

    try:
        cw = CloudWatchProvider(log_group=LOG_GROUP, region=AWS_REGION)
    except ProviderError as e:
        pytest.skip(f"CloudWatch provider creation failed: {e}")

    # Inject the verified client
    cw._client = client
    return cw


@pytest.fixture(scope="module")
def session_id(provider):
    """Try known session IDs; skip if none found."""
    for sid in KNOWN_SESSION_IDS:
        try:
            provider.get_evaluation_data(sid)
            return sid
        except (SessionNotFoundError, ProviderError):
            continue
    pytest.skip("No known sessions found in CloudWatch")


@pytest.fixture(scope="module")
def evaluation_data(provider, session_id):
    """Fetch evaluation data for the discovered session."""
    return provider.get_evaluation_data(session_id)


class TestGetEvaluationData:
    def test_returns_session_with_traces(self, evaluation_data, session_id):
        session = evaluation_data["trajectory"]
        assert isinstance(session, Session)
        assert session.session_id == session_id
        assert len(session.traces) > 0

    def test_traces_have_spans(self, evaluation_data):
        session = evaluation_data["trajectory"]
        for trace in session.traces:
            assert isinstance(trace, Trace)
            assert isinstance(trace.trace_id, str)
            assert len(trace.trace_id) > 0
            assert len(trace.spans) > 0

    def test_spans_are_typed(self, evaluation_data):
        """All spans should be one of the three known types."""
        valid_types = (AgentInvocationSpan, InferenceSpan, ToolExecutionSpan)
        session = evaluation_data["trajectory"]
        for trace in session.traces:
            for span in trace.spans:
                assert isinstance(span, valid_types), f"Unexpected span type: {type(span).__name__}"

    def test_has_agent_invocation_span(self, evaluation_data):
        """At least one trace should have an AgentInvocationSpan."""
        session = evaluation_data["trajectory"]
        agent_spans = [
            span for trace in session.traces for span in trace.spans if isinstance(span, AgentInvocationSpan)
        ]
        assert len(agent_spans) > 0, "Expected at least one AgentInvocationSpan"

    def test_agent_invocation_has_prompt_and_response(self, evaluation_data):
        """AgentInvocationSpan should have user prompt, agent response, and tools list."""
        session = evaluation_data["trajectory"]
        for trace in session.traces:
            for span in trace.spans:
                if isinstance(span, AgentInvocationSpan):
                    assert isinstance(span.user_prompt, str)
                    assert len(span.user_prompt) > 0
                    assert isinstance(span.agent_response, str)
                    assert len(span.agent_response) > 0
                    assert isinstance(span.available_tools, list)
                    return
        pytest.fail("No AgentInvocationSpan found")

    def test_output_is_nonempty_string(self, evaluation_data):
        output = evaluation_data["output"]
        assert isinstance(output, str)
        assert len(output) > 0, "Expected non-empty output from agent response"

    def test_span_info_populated(self, evaluation_data):
        session = evaluation_data["trajectory"]
        for trace in session.traces:
            for span in trace.spans:
                si = span.span_info
                assert si.trace_id is not None
                assert si.span_id is not None
                assert si.session_id == session.session_id
                assert si.start_time is not None
                assert si.end_time is not None
                return
        pytest.fail("No spans found to check")

    def test_nonexistent_session_raises(self, provider):
        with pytest.raises(SessionNotFoundError):
            provider.get_evaluation_data("nonexistent-session-id-that-does-not-exist-12345")


# --- End-to-end: CloudWatch → Evaluator pipeline ---


class TestEndToEnd:
    """Fetch traces from CloudWatch and run real evaluators on them."""

    def test_output_evaluator_on_remote_trace(self, provider, session_id):
        """OutputEvaluator produces a valid score from a CloudWatch session."""

        def task(case: Case) -> dict:
            return provider.get_evaluation_data(case.input)

        cases = [
            Case(
                name="cloudwatch_session",
                input=session_id,
                expected_output="any agent response",
            ),
        ]

        evaluator = OutputEvaluator(
            rubric="Score 1.0 if the output is a coherent response from an AI agent. "
            "Score 0.0 if the output is empty or clearly broken.",
        )

        experiment = Experiment(cases=cases, evaluators=[evaluator])
        reports = experiment.run_evaluations(task)

        assert len(reports) == 1
        report = reports[0]
        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.scores) == 1

    def test_coherence_evaluator_on_remote_trace(self, provider, session_id):
        """CoherenceEvaluator produces a valid score from a CloudWatch session."""

        def task(case: Case) -> dict:
            return provider.get_evaluation_data(case.input)

        cases = [
            Case(
                name="cloudwatch_session",
                input=session_id,
                expected_output="any agent response",
            ),
        ]

        evaluator = CoherenceEvaluator()

        experiment = Experiment(cases=cases, evaluators=[evaluator])
        reports = experiment.run_evaluations(task)

        assert len(reports) == 1
        report = reports[0]
        assert 0.0 <= report.overall_score <= 1.0

    def test_multiple_evaluators_on_remote_trace(self, provider, session_id):
        """Multiple evaluators can all run on the same CloudWatch session data."""

        def task(case: Case) -> dict:
            return provider.get_evaluation_data(case.input)

        cases = [
            Case(
                name="cloudwatch_session",
                input=session_id,
                expected_output="any agent response",
            ),
        ]

        evaluators = [
            OutputEvaluator(rubric="Score 1.0 if the output is coherent. Score 0.0 otherwise."),
            CoherenceEvaluator(),
            HelpfulnessEvaluator(),
        ]

        experiment = Experiment(cases=cases, evaluators=evaluators)
        reports = experiment.run_evaluations(task)

        assert len(reports) == 3
        for report in reports:
            assert 0.0 <= report.overall_score <= 1.0
            assert len(report.scores) == 1
