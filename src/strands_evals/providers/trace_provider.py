"""TraceProvider interface for retrieving agent trace data from observability backends."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..types.trace import Session


@dataclass
class SessionFilter:
    """Filter criteria for discovering sessions.

    Universal fields are defined here. Provider-specific parameters
    go in `additional_fields`.
    """

    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int | None = None
    additional_fields: dict[str, Any] = field(default_factory=dict)


class TraceProvider(ABC):
    """Retrieves agent trace data from observability backends for evaluation.

    Implementations handle authentication, pagination, and conversion from
    provider-native formats to the Session/Trace types the evals system consumes.
    """

    @abstractmethod
    def get_session(self, session_id: str) -> Session:
        """Retrieve all traces for a session.

        Args:
            session_id: The session identifier (maps to Strands session_id)

        Returns:
            Session object containing all traces for the session

        Raises:
            SessionNotFoundError: If no traces found for session_id
            ProviderError: If the provider is unreachable or returns an error
        """
        ...

    def list_sessions(
        self,
        session_filter: SessionFilter | None = None,
    ) -> Iterator[str]:
        """Discover session IDs matching filter criteria.

        Returns session IDs that can be fed to get_session().
        Not abstract — providers override to enable session discovery.

        Args:
            session_filter: Optional filter. If None, provider-specific defaults apply.

        Yields:
            Session ID strings

        Raises:
            NotImplementedError: If the provider does not support session discovery
            ProviderError: If the provider is unreachable or returns an error
        """
        raise NotImplementedError(
            "This provider does not support session discovery. Use get_session() with a known session_id instead."
        )

    def get_session_by_trace_id(self, trace_id: str) -> Session:
        """Fetch a single trace and wrap it in a Session.

        Useful when someone has a trace_id but not a session_id, or for
        single-shot agent runs without sessions.
        Not abstract — providers override to enable trace-level retrieval.

        Args:
            trace_id: The unique trace identifier

        Returns:
            Session object containing the single trace

        Raises:
            NotImplementedError: If the provider does not support trace-level retrieval
            TraceNotFoundError: If trace doesn't exist
            ProviderError: If the provider is unreachable or returns an error
        """
        raise NotImplementedError(
            "This provider does not support trace-level retrieval. Use get_session() with a session_id instead."
        )
