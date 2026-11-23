import json
from pathlib import Path

from .types.evaluation import EvaluationData

# A case's runs live in a per-case subdirectory as run_0.json, run_1.json, .... A pre-existing
# flat `{case_name}.json` from the single-run layout is treated as that case's run 0, so old
# stores keep loading without a migration step.
_RUN_PREFIX = "run_"
_MANIFEST_NAME = "_manifest.json"


class ConfigDriftError(RuntimeError):
    """Raised when a store's recorded config no longer matches the current experiment.

    Mixing results produced under different evaluators, cases, or model config into one report
    silently compares apples to oranges. When the store already holds results written under a
    different `config_hash`, resuming raises this instead of appending, unless the caller passes
    `overwrite=True` to discard the stale results and start fresh.
    """


class LocalFileTaskResultStore:
    """Task result store backed by local JSON files.

    Each case gets its own subdirectory holding one JSON file per run (`run_0.json`,
    `run_1.json`, ...), so a case can be evaluated N times and every result is kept. A legacy
    flat `{case_name}.json` file from the original single-run layout is read as that case's
    run 0, so existing stores keep working unchanged.

    An optional `config_hash` guards against config drift: pass the hash of the current
    experiment's evaluators/cases/model config and the store records it in a `_manifest.json`.
    A later run whose hash differs raises `ConfigDriftError` rather than mixing incompatible
    results, unless the store is opened with `overwrite=True`.
    """

    def __init__(self, directory: str | Path, config_hash: str | None = None, overwrite: bool = False):
        """Open (or create) a store rooted at `directory`.

        Args:
            directory: Directory that holds the per-case result subdirectories.
            config_hash: Optional hash of the current experiment config. When given, it is
                checked against the stored manifest (drift raises unless `overwrite=True`) and
                then recorded.
            overwrite: When True, discard any existing results and manifest before use. Use this
                to start fresh after an intentional config change.

        Raises:
            ConfigDriftError: If `config_hash` is given, differs from the stored manifest, and
                `overwrite` is False.
        """
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

        if overwrite:
            self._clear()

        if config_hash is not None:
            self._guard_config(config_hash, overwrite=overwrite)

    def _clear(self) -> None:
        """Remove all stored case results and the manifest, leaving an empty store directory."""
        for child in self._directory.iterdir():
            if child.is_dir():
                for f in child.glob(f"{_RUN_PREFIX}*.json"):
                    f.unlink()
                # Drop the now-empty case directory; ignore if other files remain.
                if not any(child.iterdir()):
                    child.rmdir()
            elif child.suffix == ".json":
                child.unlink()

    def _manifest_path(self) -> Path:
        return self._directory / _MANIFEST_NAME

    def _guard_config(self, config_hash: str, overwrite: bool) -> None:
        """Check the stored config hash against `config_hash`, then record the current one."""
        manifest_path = self._manifest_path()
        if manifest_path.exists() and not overwrite:
            stored = json.loads(manifest_path.read_text()).get("config_hash")
            if stored is not None and stored != config_hash:
                raise ConfigDriftError(
                    "Experiment configuration changed since results were last written "
                    f"(stored config_hash={stored}, current={config_hash}). "
                    "Open the store with overwrite=True to discard cached results and start fresh."
                )

        manifest_path.write_text(json.dumps({"config_hash": config_hash}, indent=2))

    def _case_dir(self, case_name: str) -> Path:
        return self._directory / case_name

    def _run_path(self, case_name: str, run_index: int) -> Path:
        return self._case_dir(case_name) / f"{_RUN_PREFIX}{run_index}.json"

    def _legacy_path(self, case_name: str) -> Path:
        return self._directory / f"{case_name}.json"

    def load(self, case_name: str, run_index: int = 0) -> EvaluationData | None:
        """Load a cached task result for one run of a case.

        Args:
            case_name: The name of the case to load results for.
            run_index: Which run to load. Defaults to `0`.

        Returns:
            The cached EvaluationData if the run exists, None otherwise. Run 0 also reads a
            legacy flat `{case_name}.json` file when no per-run file is present.
        """
        path = self._run_path(case_name, run_index)
        if path.exists():
            return EvaluationData.model_validate_json(path.read_text())

        if run_index == 0:
            legacy = self._legacy_path(case_name)
            if legacy.exists():
                return EvaluationData.model_validate_json(legacy.read_text())

        return None

    def save(self, case_name: str, result: EvaluationData, run_index: int = 0) -> None:
        """Save a task result for one run of a case.

        Args:
            case_name: The name of the case to save results for.
            result: The EvaluationData to save.
            run_index: Which run this result is. Defaults to `0`.
        """
        path = self._run_path(case_name, run_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2))

    def completed_run_count(self, case_name: str) -> int:
        """Return the number of consecutive stored runs for a case.

        Counts from `run_index=0` upward and stops at the first missing run, so the return value
        is the next index to write. A legacy flat file counts as run 0.
        """
        count = 0
        while self.load(case_name, count) is not None:
            count += 1
        return count
