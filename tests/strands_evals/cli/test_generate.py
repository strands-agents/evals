"""Tests for ``strands-evals generate``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from strands_evals.case import Case
from strands_evals.cli.main import main
from strands_evals.evaluators.output_evaluator import OutputEvaluator
from strands_evals.experiment import Experiment


def _experiment_with(num_cases: int = 2, evaluator: bool = False) -> Experiment:
    cases = [Case[str, str](name=f"c{i}", input=f"q{i}", expected_output=f"a{i}") for i in range(num_cases)]
    evaluators = [OutputEvaluator(rubric="generated rubric")] if evaluator else []
    return Experiment(cases=cases, evaluators=evaluators)


def test_generate_writes_to_stdout(tmp_path: Path, capsys):
    fake = AsyncMock(return_value=_experiment_with(num_cases=3))
    with patch(
        "strands_evals.cli.commands.generate.ExperimentGenerator.from_context_async",
        new=fake,
    ):
        exit_code = main(["generate", "--context", "ctx", "--num-cases", "3"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert len(payload["cases"]) == 3
    assert "generated 3 case(s)" in captured.err


def test_generate_writes_to_file_via_to_file(tmp_path: Path, capsys):
    out_path = tmp_path / "experiment.json"
    fake = AsyncMock(return_value=_experiment_with(num_cases=1, evaluator=True))
    with patch(
        "strands_evals.cli.commands.generate.ExperimentGenerator.from_context_async",
        new=fake,
    ):
        exit_code = main(
            [
                "generate",
                "--context",
                "ctx",
                "--num-cases",
                "1",
                "--evaluator",
                "OutputEvaluator",
                "-o",
                str(out_path),
            ]
        )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert len(payload["cases"]) == 1
    assert payload["evaluators"][0]["evaluator_type"] == "OutputEvaluator"
    assert "1 case(s), 1 evaluator(s)" in captured.err
    # Stdout is empty when -o is set.
    assert captured.out == ""


def test_generate_to_file_appends_json_extension(tmp_path: Path):
    out_path = tmp_path / "experiment"  # no extension
    fake = AsyncMock(return_value=_experiment_with(num_cases=1))
    with patch(
        "strands_evals.cli.commands.generate.ExperimentGenerator.from_context_async",
        new=fake,
    ):
        exit_code = main(["generate", "--context", "ctx", "-o", str(out_path)])

    assert exit_code == 0
    # Experiment.to_file appends .json automatically.
    assert (tmp_path / "experiment.json").exists()


def test_generate_passes_through_args(capsys):
    fake = AsyncMock(return_value=_experiment_with(num_cases=2, evaluator=True))
    with patch(
        "strands_evals.cli.commands.generate.ExperimentGenerator.from_context_async",
        new=fake,
    ):
        exit_code = main(
            [
                "generate",
                "--context",
                "ctx",
                "--num-cases",
                "2",
                "--task-description",
                "answer factual qs",
                "--num-topics",
                "3",
                "--evaluator",
                "TrajectoryEvaluator",
                "--model",
                "anthropic.claude-test",
            ]
        )

    assert exit_code == 0
    kwargs = dict(fake.await_args.kwargs)
    # The class object is passed; assert by name to avoid importing it again.
    assert kwargs.pop("evaluator").__name__ == "TrajectoryEvaluator"
    assert kwargs == {
        "context": "ctx",
        "num_cases": 2,
        "task_description": "answer factual qs",
        "num_topics": 3,
    }


def test_generate_unknown_evaluator_choice_exits_2(capsys):
    """argparse rejects --evaluator values outside the supported set."""
    with pytest.raises(SystemExit):
        main(["generate", "--context", "ctx", "--evaluator", "Helpfulness"])


def test_generate_missing_source_exits_2(capsys):
    """argparse requires exactly one of --context / --experiment."""
    with pytest.raises(SystemExit):
        main(["generate"])


def test_generate_context_and_experiment_mutually_exclusive(tmp_path: Path):
    src = tmp_path / "src.json"
    src.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["generate", "--context", "ctx", "--experiment", str(src)])


def test_generate_from_experiment_calls_from_experiment_async(experiment_file: Path, capsys):
    fake = AsyncMock(return_value=_experiment_with(num_cases=4, evaluator=True))
    with patch(
        "strands_evals.cli.commands.generate.ExperimentGenerator.from_experiment_async",
        new=fake,
    ):
        exit_code = main(
            [
                "generate",
                "--experiment",
                str(experiment_file),
                "--num-cases",
                "4",
                "--task-description",
                "rephrased task",
            ]
        )

    captured = capsys.readouterr()
    assert exit_code == 0
    fake.assert_awaited_once()
    kwargs = dict(fake.await_args.kwargs)
    assert isinstance(kwargs.pop("source_experiment"), Experiment)
    assert kwargs == {
        "task_description": "rephrased task",
        "num_cases": 4,
        "extra_information": None,
    }
    payload = json.loads(captured.out)
    assert len(payload["cases"]) == 4


def test_generate_extra_information_passed_through(experiment_file: Path):
    fake = AsyncMock(return_value=_experiment_with(num_cases=1))
    with patch(
        "strands_evals.cli.commands.generate.ExperimentGenerator.from_experiment_async",
        new=fake,
    ):
        exit_code = main(
            [
                "generate",
                "--experiment",
                str(experiment_file),
                "--extra-information",
                "Use a friendly tone.",
            ]
        )

    assert exit_code == 0
    assert fake.await_args.kwargs["extra_information"] == "Use a friendly tone."


def test_generate_experiment_rejects_evaluator(experiment_file: Path, capsys):
    """--evaluator only applies to --context; rejected with --experiment."""
    exit_code = main(
        [
            "generate",
            "--experiment",
            str(experiment_file),
            "--evaluator",
            "OutputEvaluator",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--evaluator" in captured.err


def test_generate_experiment_rejects_num_topics(experiment_file: Path, capsys):
    exit_code = main(
        [
            "generate",
            "--experiment",
            str(experiment_file),
            "--num-topics",
            "3",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--num-topics" in captured.err


def test_generate_extra_information_requires_experiment(capsys):
    exit_code = main(
        [
            "generate",
            "--context",
            "ctx",
            "--extra-information",
            "more",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--extra-information" in captured.err


def test_generate_custom_evaluator_requires_experiment(capsys):
    """--custom-evaluator only applies to --experiment; rejected with --context."""
    exit_code = main(
        [
            "generate",
            "--context",
            "ctx",
            "--custom-evaluator",
            "pkg.mod:MyEvaluator",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--custom-evaluator" in captured.err


def test_generate_experiment_missing_source_exits_2(tmp_path: Path, capsys):
    exit_code = main(["generate", "--experiment", str(tmp_path / "missing.json")])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "error" in captured.err.lower()
