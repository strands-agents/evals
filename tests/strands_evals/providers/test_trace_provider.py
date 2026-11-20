"""Tests for TraceProvider ABC and exception hierarchy."""

from collections.abc import Iterator
from datetime import datetime

import pytest

from strands_evals.providers.exceptions import (
    ProviderError,
    SessionNotFoundError,
    TraceProviderError,
)
from strands_evals.providers.trace_provider import (
    SessionFilter,
    TraceProvider,
)
from strands_evals.types.evaluation import TaskOutput
from strands_evals.types.trace import Session


class ConcreteProvider(TraceProvider):
    """Minimal concrete implementation for testing the ABC."""

    def __init__(self, session: Session | None = None):
        self._session = session

    def get_evaluation_data(self, session_id: str) -> TaskOutput:
        if self._session is None:
            raise SessionNotFoundError(f"No session found: {session_id}")
        return TaskOutput(
            output="test response",
            trajectory=self._session,
        )


class DiscoverableProvider(ConcreteProvider):
    """Provider that overrides list_sessions to support discovery."""

    def __init__(self, session_ids: list[str], session: Session | None = None):
        super().__init__(session=session)
        self._session_ids = session_ids
        self.last_filter: SessionFilter | None = None

    def list_sessions(self, session_filter: SessionFilter | None = None) -> Iterator[str]:
        self.last_filter = session_filter
        limit = session_filter.limit if session_filter else None
        for i, session_id in enumerate(self._session_ids):
            if limit is not None and i >= limit:
                return
            yield session_id


class TestExceptionHierarchy:
    def test_trace_provider_error_is_exception(self):
        assert issubclass(TraceProviderError, Exception)

    def test_session_not_found_is_trace_provider_error(self):
        assert issubclass(SessionNotFoundError, TraceProviderError)

    def test_provider_error_is_trace_provider_error(self):
        assert issubclass(ProviderError, TraceProviderError)

    def test_exceptions_carry_message(self):
        err = SessionNotFoundError("session-123 not found")
        assert "session-123 not found" in str(err)

    def test_catching_base_catches_all(self):
        """All provider exceptions can be caught with TraceProviderError."""
        for exc_class in (SessionNotFoundError, ProviderError):
            with pytest.raises(TraceProviderError):
                raise exc_class("test")


class TestTraceProviderABC:
    def test_cannot_instantiate_without_get_evaluation_data(self):
        with pytest.raises(TypeError):
            TraceProvider()  # type: ignore[abstract]

    def test_concrete_provider_instantiates(self):
        provider = ConcreteProvider()
        assert isinstance(provider, TraceProvider)

    def test_get_evaluation_data_returns_task_output(self):
        session = Session(session_id="s1", traces=[])
        provider = ConcreteProvider(session=session)
        result = provider.get_evaluation_data("s1")
        assert result["output"] == "test response"
        assert result["trajectory"] == session

    def test_get_evaluation_data_raises_session_not_found(self):
        provider = ConcreteProvider(session=None)
        with pytest.raises(SessionNotFoundError, match="No session found"):
            provider.get_evaluation_data("missing")

    def test_as_task_returns_callable(self):
        session = Session(session_id="s1", traces=[])
        provider = ConcreteProvider(session=session)
        task = provider.as_task()
        assert callable(task)

    def test_as_task_callable_delegates_to_get_evaluation_data(self):
        """as_task() callable should pass case.session_id to get_evaluation_data."""
        session = Session(session_id="s1", traces=[])
        provider = ConcreteProvider(session=session)
        task = provider.as_task()

        class FakeCase:
            session_id = "s1"

        result = task(FakeCase())
        assert result["output"] == "test response"
        assert result["trajectory"] == session


class TestSessionFilter:
    def test_defaults_are_none_and_empty(self):
        f = SessionFilter()
        assert f.start_time is None
        assert f.end_time is None
        assert f.limit is None
        assert f.additional_fields == {}

    def test_accepts_universal_and_additional_fields(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        f = SessionFilter(start_time=start, end_time=end, limit=5, additional_fields={"env": "prod"})
        assert f.start_time == start
        assert f.end_time == end
        assert f.limit == 5
        assert f.additional_fields == {"env": "prod"}


class TestListSessions:
    def test_default_list_sessions_raises_not_implemented(self):
        provider = ConcreteProvider()
        with pytest.raises(NotImplementedError, match="does not support session discovery"):
            list(provider.list_sessions())

    def test_overridden_list_sessions_yields_ids(self):
        provider = DiscoverableProvider(["s1", "s2", "s3"])
        assert list(provider.list_sessions()) == ["s1", "s2", "s3"]

    def test_list_sessions_receives_filter(self):
        provider = DiscoverableProvider(["s1", "s2", "s3"])
        f = SessionFilter(limit=2)
        result = list(provider.list_sessions(f))
        assert result == ["s1", "s2"]
        assert provider.last_filter is f
