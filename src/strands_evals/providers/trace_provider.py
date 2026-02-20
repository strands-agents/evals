"""TraceProvider interface for retrieving agent trace data from observability backends."""

from abc import ABC, abstractmethod

from ..types.evaluation import TaskOutput


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