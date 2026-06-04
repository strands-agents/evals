"""Tests for ``strands-evals validate``."""

from __future__ import annotations

import json
from pathlib import Path

from strands_evals.cli.main import main


def test_validate_valid_experiment_rich(experiment_file: Path, capsys):
    exit_code = main(["--rich", "validate", str(experiment_file)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "1 case" in captured.err
    assert "OutputEvaluator" in captured.err


def test_validate_valid_experiment_json(experiment_file: Path, capsys):
    exit_code = main(["--json", "validate", str(experiment_file)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload == {
        "experiment_file": str(experiment_file),
        "case_count": 1,
        "evaluator_count": 1,
        "evaluators": ["OutputEvaluator"],
        "valid": True,
    }


def test_validate_missing_file_exits_2(tmp_path: Path, capsys):
    missing = tmp_path / "nope.json"
    exit_code = main(["validate", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "error" in captured.err.lower()


def test_validate_malformed_json_exits_2(tmp_path: Path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    exit_code = main(["validate", str(bad)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "error" in captured.err.lower()


def test_validate_unknown_evaluator_exits_3(tmp_path: Path, capsys):
    """A serialized experiment referring to an unknown evaluator type raises a
    plain Exception inside Experiment.from_dict; the CLI maps it to exit 3."""
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "cases": [{"input": "x", "expected_output": "y"}],
                "evaluators": [{"evaluator_type": "DoesNotExist"}],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate", str(bad)])
    captured = capsys.readouterr()

    assert exit_code == 3
    assert "DoesNotExist" in captured.err
