"""Integration tests for HoneyHiveProvider against a real HoneyHive project.

Requires the HH_API_KEY environment variable (a project API key with read access).
HH_API_URL is optional and defaults to the HoneyHive cloud data plane.
Requires the HH_TEST_SESSION_ID environment variable pointing to a session
that contains Strands agent spans.

Run with: pytest tests_integ/test_honeyhive_provider.py -v
"""

import os
import uuid

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import CoherenceEvaluator, OutputEvaluator
from strands_evals.providers.exceptions import ProviderError, SessionNotFoundError
from strands_evals.providers.honeyhive_provider import HoneyHiveProvider
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    Session,
    ToolExecutionSpan,
    Trace,
)


@pytest.fixture(scope="module")
def provider():
    """Create a HoneyHiveProvider using env var credentials."""
    try:
        return HoneyHiveProvider()
    except ProviderError as e:
        pytest.skip(f"HoneyHive credentials not available: {e}")


@pytest.fixture(scope="module")
def session_id():
    """Get a test session ID from environment variable."""
    session_id = os.environ.get("HH_TEST_SESSION_ID")
    if not session_id:
        pytest.skip("HH_TEST_SESSION_ID not set")
    return session_id


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
            assert len(trace.trace_id) > 0
            assert len(trace.spans) > 0

    def test_spans_are_typed(self, evaluation_data):
        """All spans should be one of the three known types."""
        valid_types = (AgentInvocationSpan, InferenceSpan, ToolExecutionSpan)
        session = evaluation_data["trajectory"]
        for trace in session.traces:
            for span in trace.spans:
                assert isinstance(span, valid_types), f"Unexpected span type: {type(span).__name__}"

    def test_agent_invocation_has_prompt_and_response(self, evaluation_data):
        session = evaluation_data["trajectory"]
        agent_spans = [
            span for trace in session.traces for span in trace.spans if isinstance(span, AgentInvocationSpan)
        ]
        assert agent_spans, "Expected at least one AgentInvocationSpan"
        for span in agent_spans:
            assert len(span.user_prompt) > 0
            assert len(span.agent_response) > 0

    def test_root_agent_span_has_no_parent(self, evaluation_data):
        """Each trace has a root agent span, which Trace finds by a parent_span_id of None."""
        session = evaluation_data["trajectory"]
        for trace in session.traces:
            roots = [
                s for s in trace.spans if isinstance(s, AgentInvocationSpan) and s.span_info.parent_span_id is None
            ]
            assert roots, f"Trace {trace.trace_id} has no root AgentInvocationSpan"

    def test_span_info_populated(self, evaluation_data):
        session = evaluation_data["trajectory"]
        for trace in session.traces:
            for span in trace.spans:
                si = span.span_info
                assert si.trace_id == trace.trace_id
                assert si.span_id
                assert si.session_id == session.session_id
                assert si.start_time <= si.end_time

    def test_output_is_last_agent_response(self, evaluation_data):
        output = evaluation_data["output"]
        assert isinstance(output, str)
        assert len(output) > 0

    def test_nonexistent_session_raises(self, provider):
        # HoneyHive session IDs are UUIDs. The events API returns a server error for other values.
        with pytest.raises(SessionNotFoundError):
            provider.get_evaluation_data(str(uuid.uuid4()))

    def test_invalid_api_key_raises(self, session_id):
        bad_provider = HoneyHiveProvider(api_key="invalid-key", api_url=os.environ.get("HH_API_URL"))
        with pytest.raises(ProviderError):
            bad_provider.get_evaluation_data(session_id)


class TestEndToEnd:
    """Fetch a session from HoneyHive and run real evaluators on it."""

    def test_evaluators_on_remote_trace(self, provider, session_id):
        def task(case: Case) -> dict:
            return provider.get_evaluation_data(case.input)

        cases = [Case(name="honeyhive_session", input=session_id, expected_output="any agent response")]
        evaluators = [
            OutputEvaluator(
                rubric="Score 1.0 if the output is a coherent response from an AI agent. "
                "Score 0.0 if the output is empty or clearly broken.",
            ),
            CoherenceEvaluator(),
        ]

        report = Experiment(cases=cases, evaluators=evaluators).run_evaluations(task)

        # A failed judge call still produces a score of 0.0, so check the reasons for errors.
        assert not [r for r in report.reasons if r.startswith("Evaluator error")], report.reasons
        assert len(report.scores) == 2
        assert all(0.0 <= s <= 1.0 for s in report.scores)
