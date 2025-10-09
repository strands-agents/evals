"""
SessionMapper - Base class for mapping telemetry data to Session format
"""

from abc import ABC, abstractmethod

from typing_extensions import Any

from ..types.trace import Session


class SessionMapper(ABC):
    """Base class for mapping telemetry data to Session format for evaluation."""

    @abstractmethod
    def map_to_session(self, spans: list[Any], session_id: str) -> Session:
        """
        Map spans to Session format.

        Args:
            spans: List of span objects
            session_id: Session identifier

        Returns:
            Session object ready for evaluation
        """
        pass
