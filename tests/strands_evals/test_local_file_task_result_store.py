import json

import pytest

from strands_evals.local_file_task_result_store import ConfigDriftError, LocalFileTaskResultStore
from strands_evals.types import EvaluationData


@pytest.fixture
def evaluation_data():
    return EvaluationData(
        input="What is 2+2?",
        actual_output="4",
        name="math_case",
        expected_output="4",
        metadata={"difficulty": "easy"},
    )


@pytest.fixture
def store(tmp_path):
    return LocalFileTaskResultStore(directory=tmp_path / "results")


class TestLocalFileTaskResultStore:
    def test_save_and_load(self, store, evaluation_data):
        store.save("math_case", evaluation_data)
        loaded = store.load("math_case")

        assert loaded is not None
        assert loaded.input == evaluation_data.input
        assert loaded.actual_output == evaluation_data.actual_output
        assert loaded.expected_output == evaluation_data.expected_output
        assert loaded.name == evaluation_data.name
        assert loaded.metadata == evaluation_data.metadata

    def test_load_missing_returns_none(self, store):
        result = store.load("nonexistent_case")
        assert result is None

    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "nested" / "results"
        assert not new_dir.exists()
        LocalFileTaskResultStore(directory=new_dir)
        assert new_dir.exists()

    def test_save_writes_run_file(self, store, evaluation_data, tmp_path):
        store.save("math_case", evaluation_data)
        file_path = tmp_path / "results" / "math_case" / "run_0.json"
        assert file_path.exists()
        data = json.loads(file_path.read_text())
        assert data["input"] == "What is 2+2?"
        assert data["actual_output"] == "4"


class TestPerRunKeying:
    """A case can be evaluated N times; each run is stored and read independently."""

    def _data(self, name, output):
        return EvaluationData(input="q", actual_output=output, name=name, expected_output="q")

    def test_runs_are_stored_and_loaded_independently(self, store):
        store.save("c", self._data("c", "run-a"), run_index=0)
        store.save("c", self._data("c", "run-b"), run_index=1)

        assert store.load("c", 0).actual_output == "run-a"
        assert store.load("c", 1).actual_output == "run-b"
        assert store.load("c", 2) is None

    def test_completed_run_count_counts_consecutive_runs(self, store):
        assert store.completed_run_count("c") == 0
        store.save("c", self._data("c", "0"), run_index=0)
        store.save("c", self._data("c", "1"), run_index=1)
        assert store.completed_run_count("c") == 2

    def test_completed_run_count_stops_at_first_gap(self, store):
        store.save("c", self._data("c", "0"), run_index=0)
        store.save("c", self._data("c", "2"), run_index=2)  # gap at 1
        assert store.completed_run_count("c") == 1

    def test_run_files_live_in_per_case_subdir(self, store, tmp_path):
        store.save("c", self._data("c", "x"), run_index=3)
        assert (tmp_path / "results" / "c" / "run_3.json").exists()

    def test_legacy_flat_file_is_read_as_run_zero(self, tmp_path):
        # A store written by the old single-run layout: flat {case_name}.json.
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        legacy = self._data("legacy", "old")
        (results_dir / "legacy.json").write_text(legacy.model_dump_json())

        store = LocalFileTaskResultStore(directory=results_dir)
        assert store.load("legacy", 0).actual_output == "old"
        assert store.completed_run_count("legacy") == 1


class TestConfigDrift:
    """An optional config hash prevents mixing results from incompatible experiment configs."""

    def _data(self, name):
        return EvaluationData(input="q", actual_output="a", name=name, expected_output="q")

    def test_first_open_records_hash(self, tmp_path):
        d = tmp_path / "results"
        LocalFileTaskResultStore(directory=d, config_hash="hash-1")
        manifest = json.loads((d / "_manifest.json").read_text())
        assert manifest["config_hash"] == "hash-1"

    def test_reopen_with_same_hash_is_allowed(self, tmp_path):
        d = tmp_path / "results"
        s1 = LocalFileTaskResultStore(directory=d, config_hash="hash-1")
        s1.save("c", self._data("c"))
        s2 = LocalFileTaskResultStore(directory=d, config_hash="hash-1")
        assert s2.load("c") is not None

    def test_reopen_with_different_hash_raises(self, tmp_path):
        d = tmp_path / "results"
        LocalFileTaskResultStore(directory=d, config_hash="hash-1")
        with pytest.raises(ConfigDriftError, match="configuration changed"):
            LocalFileTaskResultStore(directory=d, config_hash="hash-2")

    def test_overwrite_discards_results_and_updates_hash(self, tmp_path):
        d = tmp_path / "results"
        s1 = LocalFileTaskResultStore(directory=d, config_hash="hash-1")
        s1.save("c", self._data("c"))
        assert s1.load("c") is not None

        s2 = LocalFileTaskResultStore(directory=d, config_hash="hash-2", overwrite=True)
        assert s2.load("c") is None
        manifest = json.loads((d / "_manifest.json").read_text())
        assert manifest["config_hash"] == "hash-2"

    def test_no_hash_skips_the_guard(self, tmp_path):
        d = tmp_path / "results"
        LocalFileTaskResultStore(directory=d, config_hash="hash-1")
        # Opening without a hash must not raise even though a manifest exists.
        store = LocalFileTaskResultStore(directory=d)
        assert store is not None
