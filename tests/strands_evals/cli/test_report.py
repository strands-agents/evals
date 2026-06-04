"""Tests for ``strands-evals report``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from strands_evals.cli.main import main


def test_report_json_to_stdout(reports_file: Path, capsys):
    exit_code = main(["--json", "report", str(reports_file)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["evaluator_name"] == "OutputEvaluator"
    assert payload["overall_score"] == 0.75
    assert payload["scores"] == [1.0, 0.5]


def test_report_output_file_always_json(reports_file: Path, tmp_path: Path):
    out_path = tmp_path / "out.json"
    exit_code = main(["--rich", "report", str(reports_file), "-o", str(out_path)])

    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload["overall_score"] == 0.75


def test_report_rich_calls_display(reports_file: Path):
    with patch("strands_evals.types.evaluation_report.EvaluationReport.display") as mock_display:
        exit_code = main(["--rich", "report", str(reports_file)])

    assert exit_code == 0
    mock_display.assert_called_once_with(include_recommendations=False)


def test_report_recommendations_flag_threads_through(reports_file: Path):
    with patch("strands_evals.types.evaluation_report.EvaluationReport.display") as mock_display:
        exit_code = main(["--rich", "report", str(reports_file), "--recommendations"])

    assert exit_code == 0
    mock_display.assert_called_once_with(include_recommendations=True)


def test_report_stdin(reports_file: Path, capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(reports_file.read_text()))
    exit_code = main(["--json", "report", "-"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["overall_score"] == 0.75
