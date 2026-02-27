"""Tests for TraceProvider ABC and exception hierarchy."""

import pytest

from strands_evals.providers.exceptions import (
    ProviderError,
    SessionNotFoundError,
    TraceProviderError,
)
from strands_evals.providers.trace_provider import (
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
