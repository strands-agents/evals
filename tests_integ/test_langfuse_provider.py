"""Integration tests for LangfuseProvider against a real Langfuse instance.

Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables.
Requires LANGFUSE_TEST_SESSION_ID environment variable pointing to a session
with convertible observations.

Run with: pytest tests_integ/test_langfuse_provider.py -v
"""

import os

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import (
    CoherenceEvaluator,
    HelpfulnessEvaluator,
    OutputEvaluator,
)
from strands_evals.providers.exceptions import (
    ProviderError,
    SessionNotFoundError,
)
from strands_evals.providers.langfuse_provider import LangfuseProvider
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    Session,
    ToolExecutionSpan,
    Trace,
)


@pytest.fixture(scope="module")
def provider():
    """Create a LangfuseProvider using env var credentials."""
    try:
        return LangfuseProvider()
    except ProviderError as e:
        pytest.skip(f"Langfuse credentials not available: {e}")


@pytest.fixture(scope="module")
def discovered_session_id():
    """Get a test session ID from environment variable."""
    session_id = os.environ.get("LANGFUSE_TEST_SESSION_ID")
    if not session_id:
        pytest.skip("LANGFUSE_TEST_SESSION_ID not set")
    return session_id


@pytest.fixture(scope="module")
def evaluation_data(provider, discovered_session_id):
    """Fetch evaluation data for the discovered session."""
    return provider.get_evaluation_data(discovered_session_id)


class TestGetEvaluationData:
    def test_returns_session_with_traces(self, evaluation_data, discovered_session_id):
        session = evaluation_data["trajectory"]
        assert isinstance(session, Session)
        assert session.session_id == discovered_session_id
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
        session = evaluation_data["trajectory"]
        for trace in session.traces:
            for span in trace.spans:
                if isinstance(span, AgentInvocationSpan):
                    assert isinstance(span.user_prompt, str)
                    assert isinstance(span.agent_response, str)
                    assert len(span.user_prompt) > 0
                    assert len(span.agent_response) > 0
                    return
        pytest.fail("No AgentInvocationSpan found")

    def test_output_is_nonempty_string(self, evaluation_data):
        assert isinstance(evaluation_data["output"], str)
        assert len(evaluation_data["output"]) > 0

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


# --- End-to-end: Langfuse → Evaluator pipeline ---


class TestEndToEnd:
    """Fetch traces from Langfuse and run real evaluators on them."""

    def test_output_evaluator_on_remote_trace(self, provider, discovered_session_id):
        """OutputEvaluator produces a valid score from a Langfuse session."""

        def task(case: Case) -> dict:
            return provider.get_evaluation_data(case.input)

        cases = [
            Case(
                name="langfuse_session",
                input=discovered_session_id,
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
        assert report.score is not None
        assert 0.0 <= report.score <= 1.0
        assert len(report.case_results) == 1

    def test_coherence_evaluator_on_remote_trace(self, provider, discovered_session_id):
        """CoherenceEvaluator produces a valid score from a Langfuse session."""

        def task(case: Case) -> dict:
            return provider.get_evaluation_data(case.input)

        cases = [
            Case(
                name="langfuse_session",
                input=discovered_session_id,
                expected_output="any agent response",
            ),
        ]

        evaluator = CoherenceEvaluator()

        experiment = Experiment(cases=cases, evaluators=[evaluator])
        reports = experiment.run_evaluations(task)

        assert len(reports) == 1
        report = reports[0]
        assert report.score is not None
        assert 0.0 <= report.score <= 1.0

    def test_multiple_evaluators_on_remote_trace(self, provider, discovered_session_id):
        """Multiple evaluators can all run on the same Langfuse session data."""

        def task(case: Case) -> dict:
            return provider.get_evaluation_data(case.input)

        cases = [
            Case(
                name="langfuse_session",
                input=discovered_session_id,
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
            assert report.score is not None
            assert 0.0 <= report.score <= 1.0
            assert len(report.case_results) == 1
