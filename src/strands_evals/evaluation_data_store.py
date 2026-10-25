from typing import Protocol, runtime_checkable

from .types.evaluation import EvaluationData


@runtime_checkable
class EvaluationDataStore(Protocol):
    """Protocol for loading and saving evaluation data.

    Implementations can use any storage backend (local files, S3, databases, etc.)
    as long as they implement the load and save methods.

    A case may be evaluated more than once — for statistical confidence you run the same case
    N times and keep every result. `run_index` selects which run to read or write; it defaults
    to `0`, so single-run callers can ignore it entirely and see the same behavior as before.
    `completed_run_count` reports how many runs a case already has, which lets a caller resume an
    interrupted N-run sweep by running only the runs that are missing.
    """

    def load(self, case_name: str, run_index: int = 0) -> EvaluationData | None:
        """Load cached evaluation data by case name and run index.

        Args:
            case_name: The name of the case to load results for.
            run_index: Which run of the case to load. Defaults to `0` (the first/only run).

        Returns:
            The cached EvaluationData if found, None otherwise.
        """
        ...

    def save(self, case_name: str, result: EvaluationData, run_index: int = 0) -> None:
        """Save evaluation data for one run of a case.

        Args:
            case_name: The name of the case to save results for.
            result: The EvaluationData to save.
            run_index: Which run of the case this result is. Defaults to `0`.
        """
        ...

    def completed_run_count(self, case_name: str) -> int:
        """Return how many consecutive runs of a case are already stored.

        Counts stored runs starting at `run_index=0` and stopping at the first gap, so the
        result is the next `run_index` a caller should write when topping up to N runs.

        Args:
            case_name: The name of the case to count runs for.

        Returns:
            The number of consecutive stored runs, or `0` if none exist.
        """
        ...
