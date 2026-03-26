from pathlib import Path

from .types.evaluation import EvaluationData


class LocalFileTaskResultStore:
    """Task result store backed by local JSON files.

    Saves one JSON file per case in the specified directory,
    using the case name as the filename.
    """

    def __init__(self, directory: str | Path):
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def load(self, case_name: str) -> EvaluationData | None:
        """Load a cached task result from a JSON file.

        Args:
            case_name: The name of the case to load results for.

        Returns:
            The cached EvaluationData if the file exists, None otherwise.
        """
        path = self._directory / f"{case_name}.json"
        if not path.exists():
            return None
        return EvaluationData.model_validate_json(path.read_text())

    def save(self, case_name: str, result: EvaluationData) -> None:
        """Save a task result to a JSON file.

        Args:
            case_name: The name of the case to save results for.
            result: The EvaluationData to save.
        """
        path = self._directory / f"{case_name}.json"
        path.write_text(result.model_dump_json(indent=2))
