"""TraceProvider interface for retrieving agent trace data from observability backends."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..case import Case
from ..types.evaluation import TaskOutput


class SessionFilter(BaseModel):
    """Filter criteria for discovering sessions from a provider.

    Universal fields (`start_time`, `end_time`, `limit`) are defined here.
    Provider-specific parameters that do not generalize go in `additional_fields`.

    Attributes:
        start_time: Only include sessions at or after this time. None means no lower bound.
        end_time: Only include sessions at or before this time. None means no upper bound.
        limit: Maximum number of sessions to return. None means no limit.
        additional_fields: Provider-specific filter parameters that have no universal field.
    """

    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int | None = None
    additional_fields: dict[str, Any] = Field(default_factory=dict)


class TraceProvider(ABC):
    """Retrieves agent trace data from observability backends for evaluation.

    Implementations handle authentication, pagination, and conversion from
    provider-native formats to the types the evals system consumes.
    """

    @abstractmethod
    def get_evaluation_data(self, session_id: str) -> TaskOutput:
        """Retrieve all data needed to evaluate a session.

        This is the primary access pattern — given a session ID, fetch all
        traces, extract the agent output and trajectory, and return them
        in a format ready for evaluation.

        Args:
            session_id: The session identifier (maps to Strands session_id)

        Returns:
            TaskOutput with 'output' (final agent response) and
            'trajectory' (Session containing all traces/spans)

        Raises:
            SessionNotFoundError: If no traces found for session_id
            ProviderError: If the provider is unreachable or returns an error
        """
        ...

    def as_task(self) -> Callable[[Case], TaskOutput]:
        """Return a task callable that fetches evaluation data by session_id.

        Returns:
            A callable that takes a single Case and returns the TaskOutput
            for that case's session.
        """
        return lambda case: self.get_evaluation_data(case.session_id)

    def list_sessions(
        self,
        session_filter: SessionFilter | None = None,
    ) -> Iterator[str]:
        """Discover session IDs matching filter criteria.

        Returns session IDs that can be fed to `get_evaluation_data()`. This
        method is intentionally not abstract: providers only override it when
        their backend supports session discovery. The default raises
        `NotImplementedError` with a message pointing at the known-session-id
        access pattern.

        Args:
            session_filter: Optional filter criteria. If None, provider-specific
                defaults apply.

        Yields:
            Session ID strings.

        Raises:
            NotImplementedError: If the provider does not support session discovery.
            ProviderError: If the provider is unreachable or returns an error.
        """
        raise NotImplementedError(
            "this provider does not support session discovery; "
            "use get_evaluation_data() with a known session_id instead"
        )
