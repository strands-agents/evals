from typing import Protocol, runtime_checkable

from .types.evaluation import EvaluationData


@runtime_checkable
class TaskResultStore(Protocol):
    """Protocol for loading and saving task results.

    Implementations can use any storage backend (local files, S3, databases, etc.)
    as long as they implement the load and save methods.
    """

    def load(self, case_name: str) -> EvaluationData | None:
        """Load a cached task result by case name.

        Args:
            case_name: The name of the case to load results for.

        Returns:
            The cached EvaluationData if found, None otherwise.
        """
        ...

    def save(self, case_name: str, result: EvaluationData) -> None:
        """Save a task result for a case.

        Args:
            case_name: The name of the case to save results for.
            result: The EvaluationData to save.
        """
        ...
