"""Tests for ``strands-evals report``."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from strands_evals.cli.main import main
from strands_evals.display import display_console
from strands_evals.types.evaluation import EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport


def test_report_json_to_stdout(reports_file: Path, capsys):
    exit_code = main(["--json", "report", str(reports_file)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["overall_score"] == 0.75
    assert payload["scores"] == [1.0, 0.5]
    assert payload["cases"][0]["evaluator"] == "OutputEvaluator"


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


def test_report_interactive_calls_run_display(reports_file: Path):
    with patch("strands_evals.types.evaluation_report.EvaluationReport.run_display") as mock_run:
        exit_code = main(["report", str(reports_file), "--interactive"])

    assert exit_code == 0
    mock_run.assert_called_once_with(include_recommendations=False)


def test_report_interactive_short_flag_with_recommendations(reports_file: Path):
    with patch("strands_evals.types.evaluation_report.EvaluationReport.run_display") as mock_run:
        exit_code = main(["report", str(reports_file), "-i", "--recommendations"])

    assert exit_code == 0
    mock_run.assert_called_once_with(include_recommendations=True)


def test_report_interactive_overrides_json(reports_file: Path):
    with patch("strands_evals.types.evaluation_report.EvaluationReport.run_display") as mock_run:
        exit_code = main(["--json", "report", str(reports_file), "--interactive"])

    assert exit_code == 0
    mock_run.assert_called_once_with(include_recommendations=False)


def test_report_output_file_skips_interactive(reports_file: Path, tmp_path: Path):
    out_path = tmp_path / "out.json"
    with patch("strands_evals.types.evaluation_report.EvaluationReport.run_display") as mock_run:
        exit_code = main(["report", str(reports_file), "-i", "-o", str(out_path)])

    assert exit_code == 0
    mock_run.assert_not_called()
    payload = json.loads(out_path.read_text())
    assert payload["overall_score"] == 0.75


def test_report_stdin(reports_file: Path, capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(reports_file.read_text()))
    exit_code = main(["--json", "report", "-"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["overall_score"] == 0.75


def test_report_rich_renders_saved_rubric_and_cascade_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = EvaluationOutput(
        score=0.8,
        test_pass=True,
        label="false",
        metadata={
            "evaluated_by": "TypedRubricEvaluator",
            "question_id": "safety",
            "answer": True,
            "selected_option": "false",
            "probabilities": {"true": 0.2, "false": 0.8},
            "provider_name": "stub",
            "model_id": "checkpoint-1",
            "path": "fast",
            "p_gate": 0.2,
        },
    )
    report = EvaluationReport(
        overall_score=0.8, scores=[0.8], test_passes=[True], cases=[{"name": "case"}], detailed_results=[[row]]
    )
    report_file = tmp_path / "rubric-report.json"
    report.to_file(str(report_file))
    console = Console(file=StringIO(), record=True, width=180, color_system=None)
    monkeypatch.setattr(display_console, "console", console)

    assert main(["--rich", "report", str(report_file)]) == 0

    text = console.export_text()
    assert "stub / checkpoint-1" in text
    assert "path" in text and "fast" in text and "p_gate" in text
    assert "Raw answer: True" in text
    assert "Selected option: false" in text
    assert "Probabilities: true=0.2, false=0.8" in text
    assert EvaluationReport.from_dict(json.loads(report_file.read_text())).model_dump() == report.model_dump()
