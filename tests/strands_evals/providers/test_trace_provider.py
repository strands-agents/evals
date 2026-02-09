"""Tests for TraceProvider ABC, SessionFilter, and exception hierarchy."""

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from strands_evals.providers.exceptions import (
    ProviderError,
    SessionNotFoundError,
    TraceNotFoundError,
    TraceProviderError,
)
from strands_evals.providers.trace_provider import (
    SessionFilter,
    TraceProvider,
)
from strands_evals.types.trace import Session


class ConcreteProvider(TraceProvider):
    """Minimal concrete implementation for testing the ABC."""

    def __init__(self, session: Session | None = None):
        self._session = session

    def get_session(self, session_id: str) -> Session:
        if self._session is None:
            raise SessionNotFoundError(f"No session found: {session_id}")
        return self._session


class FullProvider(TraceProvider):
    """Provider that overrides all optional methods."""

    def __init__(self, sessions: dict[str, Session] | None = None, session_ids: list[str] | None = None):
        self._sessions = sessions or {}
        self._session_ids = session_ids or []

    def get_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"No session found: {session_id}")
        return self._sessions[session_id]

    def list_sessions(self, session_filter: SessionFilter | None = None) -> Iterator[str]:
        yield from self._session_ids

    def get_session_by_trace_id(self, trace_id: str) -> Session:
        for session in self._sessions.values():
            for trace in session.traces:
                if trace.trace_id == trace_id:
                    return session
        raise TraceNotFoundError(f"No trace found: {trace_id}")


# --- SessionFilter tests ---


class TestSessionFilter:
    def test_defaults(self):
        f = SessionFilter()
        assert f.start_time is None
        assert f.end_time is None
        assert f.limit is None
        assert f.additional_fields == {}

    def test_with_all_fields(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 31, tzinfo=timezone.utc)
        f = SessionFilter(
            start_time=start,
            end_time=end,
            limit=50,
            additional_fields={"environment": "production"},
        )
        assert f.start_time == start
        assert f.end_time == end
        assert f.limit == 50
        assert f.additional_fields == {"environment": "production"}

    def test_additional_fields_default_is_independent(self):
        """Each instance gets its own dict (no shared mutable default)."""
        f1 = SessionFilter()
        f2 = SessionFilter()
        f1.additional_fields["key"] = "value"
        assert "key" not in f2.additional_fields


# --- Exception hierarchy tests ---


class TestExceptionHierarchy:
    def test_trace_provider_error_is_exception(self):
        assert issubclass(TraceProviderError, Exception)

    def test_session_not_found_is_trace_provider_error(self):
        assert issubclass(SessionNotFoundError, TraceProviderError)

    def test_trace_not_found_is_trace_provider_error(self):
        assert issubclass(TraceNotFoundError, TraceProviderError)

    def test_provider_error_is_trace_provider_error(self):
        assert issubclass(ProviderError, TraceProviderError)

    def test_exceptions_carry_message(self):
        err = SessionNotFoundError("session-123 not found")
        assert "session-123 not found" in str(err)

    def test_catching_base_catches_all(self):
        """All provider exceptions can be caught with TraceProviderError."""
        for exc_class in (SessionNotFoundError, TraceNotFoundError, ProviderError):
            with pytest.raises(TraceProviderError):
                raise exc_class("test")


# --- TraceProvider ABC tests ---


class TestTraceProviderABC:
    def test_cannot_instantiate_without_get_session(self):
        with pytest.raises(TypeError):
            TraceProvider()  # type: ignore[abstract]

    def test_concrete_provider_instantiates(self):
        provider = ConcreteProvider()
        assert isinstance(provider, TraceProvider)

    def test_get_session_returns_session(self):
        session = Session(session_id="s1", traces=[])
        provider = ConcreteProvider(session=session)
        result = provider.get_session("s1")
        assert result == session

    def test_get_session_raises_session_not_found(self):
        provider = ConcreteProvider(session=None)
        with pytest.raises(SessionNotFoundError, match="No session found"):
            provider.get_session("missing")

    def test_list_sessions_default_raises_not_implemented(self):
        provider = ConcreteProvider()
        with pytest.raises(NotImplementedError, match="does not support session discovery"):
            list(provider.list_sessions())

    def test_get_session_by_trace_id_default_raises_not_implemented(self):
        provider = ConcreteProvider()
        with pytest.raises(NotImplementedError, match="does not support trace-level retrieval"):
            provider.get_session_by_trace_id("trace-123")


class TestFullProvider:
    def test_list_sessions_yields_ids(self):
        provider = FullProvider(session_ids=["s1", "s2", "s3"])
        result = list(provider.list_sessions())
        assert result == ["s1", "s2", "s3"]

    def test_list_sessions_with_filter(self):
        provider = FullProvider(session_ids=["s1"])
        f = SessionFilter(limit=10)
        result = list(provider.list_sessions(session_filter=f))
        assert result == ["s1"]

    def test_get_session_by_trace_id(self):
        from strands_evals.types.trace import Trace

        session = Session(
            session_id="s1",
            traces=[Trace(trace_id="t1", session_id="s1", spans=[])],
        )
        provider = FullProvider(sessions={"s1": session})
        result = provider.get_session_by_trace_id("t1")
        assert result == session

    def test_get_session_by_trace_id_not_found(self):
        provider = FullProvider(sessions={})
        with pytest.raises(TraceNotFoundError, match="No trace found"):
            provider.get_session_by_trace_id("missing")
