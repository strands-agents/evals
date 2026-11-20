"""Tests for evaluate_sessions batch evaluation."""

from collections.abc import Iterator

from strands_evals.batch import evaluate_sessions
from strands_evals.evaluators.deterministic.output import Equals
from strands_evals.providers.exceptions import SessionNotFoundError
from strands_evals.providers.trace_provider import SessionFilter, TraceProvider
from strands_evals.types.evaluation import TaskOutput


class FakeProvider(TraceProvider):
    """Provider with discovery over an in-memory session_id -> output map."""

    def __init__(self, outputs: dict[str, str]):
        self._outputs = outputs
        self.fetched: list[str] = []
        self.last_filter: SessionFilter | None = None

    def list_sessions(self, session_filter: SessionFilter | None = None) -> Iterator[str]:
        self.last_filter = session_filter
        limit = session_filter.limit if session_filter else None
        for i, session_id in enumerate(self._outputs):
            if limit is not None and i >= limit:
                return
            yield session_id

    def get_evaluation_data(self, session_id: str) -> TaskOutput:
        if session_id not in self._outputs:
            raise SessionNotFoundError(session_id)
        self.fetched.append(session_id)
        return TaskOutput(output=self._outputs[session_id])


class NoDiscoveryProvider(TraceProvider):
    """Provider that does not override list_sessions."""

    def get_evaluation_data(self, session_id: str) -> TaskOutput:
        return TaskOutput(output="x")


def test_evaluate_sessions_evaluates_every_discovered_session():
    provider = FakeProvider({"s1": "hello", "s2": "world"})
    # Equals compares the task output against each case's expected_output. With no
    # expected_output set (None), both cases fail, but the report still has one row
    # per session and each session was fetched.
    report = evaluate_sessions(provider, evaluators=[Equals()])

    assert len(report.cases) == 2
    assert sorted(provider.fetched) == ["s1", "s2"]
    assert {case["name"] for case in report.cases} == {"s1", "s2"}


def test_evaluate_sessions_passes_output_to_evaluator():
    provider = FakeProvider({"s1": "match", "s2": "other"})

    # Equals("match") passes only for the session whose output is exactly "match".
    report = evaluate_sessions(provider, evaluators=[Equals("match")])

    passes = {case["name"]: passed for case, passed in zip(report.cases, report.test_passes, strict=True)}
    assert passes["s1"] is True
    assert passes["s2"] is False


def test_evaluate_sessions_forwards_filter():
    provider = FakeProvider({"s1": "a", "s2": "b", "s3": "c"})
    f = SessionFilter(limit=2)

    report = evaluate_sessions(provider, evaluators=[Equals()], session_filter=f)

    assert provider.last_filter is f
    assert len(report.cases) == 2
    assert sorted(provider.fetched) == ["s1", "s2"]


def test_evaluate_sessions_empty_discovery_yields_empty_report():
    provider = FakeProvider({})

    report = evaluate_sessions(provider, evaluators=[Equals()])

    assert report.cases == []
    assert provider.fetched == []


def test_evaluate_sessions_without_discovery_raises_not_implemented():
    provider = NoDiscoveryProvider()

    try:
        evaluate_sessions(provider, evaluators=[Equals()])
    except NotImplementedError as err:
        assert "does not support session discovery" in str(err)
    else:
        raise AssertionError("expected NotImplementedError")
