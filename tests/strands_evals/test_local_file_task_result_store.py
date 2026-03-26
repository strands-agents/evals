import json

import pytest

from strands_evals.local_file_task_result_store import LocalFileTaskResultStore
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

    def test_save_writes_json_file(self, store, evaluation_data, tmp_path):
        store.save("math_case", evaluation_data)
        file_path = tmp_path / "results" / "math_case.json"
        assert file_path.exists()
        data = json.loads(file_path.read_text())
        assert data["input"] == "What is 2+2?"
        assert data["actual_output"] == "4"
