"""Tests for ``strands-evals run``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_evals.cli.main import main


def test_run_with_task_smoke(experiment_file: Path, capsys, tmp_path: Path):
    """End-to-end: --task path executes the experiment and writes a flattened report."""
    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            str(experiment_file),
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer",
            "--fail-on",
            "none",
            "-o",
            str(out_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload["evaluator_name"] == "Combined"
    assert len(payload["cases"]) == 1
    summary = capsys.readouterr().err
    assert "strands-evals run" in summary
    assert "overall:" in summary


def test_run_display_flag_renders_with_rows_expanded(experiment_file: Path, tmp_path: Path, monkeypatch):
    """`--display` should hand a fully-expanded items dict to the renderer."""
    from strands_evals.cli.commands import run as run_module

    captured: dict = {}

    class _FakeDisplay:
        def __init__(self, items: dict, overall_score: float) -> None:
            captured["items"] = items
            captured["overall_score"] = overall_score

        def run(self, static: bool = True) -> None:
            captured["static"] = static

    monkeypatch.setattr(run_module, "CollapsibleTableReportDisplay", _FakeDisplay)

    exit_code = main(
        [
            "run",
            str(experiment_file),
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer",
            "--fail-on",
            "none",
            "--display",
        ]
    )

    assert exit_code == 0
    assert captured["static"] is True
    items = captured["items"]
    assert items, "expected at least one row in the items dict"
    for row in items.values():
        assert row["expanded"] is True
        details = row["details"]
        assert "input" in details
        assert "actual_output" in details
        assert "expected_output" in details


def test_run_without_display_does_not_render(experiment_file: Path, tmp_path: Path, monkeypatch):
    """Default `run` (no --display) keeps the one-line summary contract."""
    from strands_evals.cli.commands import run as run_module

    calls: list[tuple] = []

    class _FakeDisplay:
        def __init__(self, *args, **kwargs) -> None:
            calls.append((args, kwargs))

        def run(self, static: bool = True) -> None:
            pass

    monkeypatch.setattr(run_module, "CollapsibleTableReportDisplay", _FakeDisplay)

    exit_code = main(
        [
            "--json",
            "run",
            str(experiment_file),
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer",
            "--fail-on",
            "none",
            "-o",
            str(tmp_path / "reports.json"),
        ]
    )

    assert exit_code == 0
    assert calls == []


def test_run_with_agent_callable_smoke(experiment_file: Path, capsys, tmp_path: Path):
    """The --agent callable factory path also runs end-to-end."""
    out_path = tmp_path / "reports.json"
    # Using --fail-on none because the real OutputEvaluator would call Bedrock;
    # we rely on Experiment._run_evaluator's exception isolation to capture the
    # ValidationError and record a failed result, then bail with exit 0.
    exit_code = main(
        [
            "--json",
            "run",
            str(experiment_file),
            "--agent",
            "tests.strands_evals.cli.fixtures.agents:build_agent",
            "--fail-on",
            "none",
            "-o",
            str(out_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload["cases"][0]["actual_output"] == "hello from build_agent"


def test_run_requires_agent_or_task(experiment_file: Path):
    with pytest.raises(SystemExit):
        main(["run", str(experiment_file)])


def test_run_agent_and_task_mutually_exclusive(experiment_file: Path):
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                str(experiment_file),
                "--agent",
                "x:y",
                "--task",
                "x:y",
            ]
        )


def test_run_invalid_entry_point_exits_3(experiment_file: Path, capsys):
    exit_code = main(
        [
            "run",
            str(experiment_file),
            "--task",
            "nonexistent_module_xyz:foo",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 3
    assert "cannot import module" in captured.err


def test_run_fail_on_threshold_invalid(experiment_file: Path):
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                str(experiment_file),
                "--task",
                "tests.strands_evals.cli.fixtures.tasks:answer",
                "--fail-on",
                "threshold:9.9",
            ]
        )


def test_run_trace_attributes_parsed(experiment_file: Path, tmp_path: Path):
    """--trace-attributes accepts repeatable KEY=VALUE."""
    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            str(experiment_file),
            "--agent",
            "tests.strands_evals.cli.fixtures.agents:build_agent",
            "--trace-attributes",
            "foo=bar",
            "--trace-attributes",
            "baz=qux",
            "--fail-on",
            "none",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0


def test_run_trace_attributes_malformed(experiment_file: Path):
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                str(experiment_file),
                "--task",
                "tests.strands_evals.cli.fixtures.tasks:answer",
                "--trace-attributes",
                "no_equals_sign",
            ]
        )


def test_run_data_store_creates_directory(experiment_file: Path, tmp_path: Path):
    """--data-store points at a directory that LocalFileTaskResultStore creates."""
    store_dir = tmp_path / "cache"
    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            str(experiment_file),
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer",
            "--data-store",
            str(store_dir),
            "--fail-on",
            "none",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0
    assert store_dir.is_dir()
    # One file per case (case 'c1' in the fixture).
    assert (store_dir / "c1.json").exists()


def test_run_exit_zero_overrides_fail_on(experiment_file: Path, tmp_path: Path):
    """--exit-zero forces exit 0 even when --fail-on=any would fail."""
    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            str(experiment_file),
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer",
            "--fail-on",
            "any",
            "--exit-zero",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0


def test_run_custom_evaluator_threading(experiment_file: Path, tmp_path: Path):
    """--custom-evaluator lets from_file resolve user-defined evaluator types."""
    # Author a tweaked experiment that references the AlwaysPasses evaluator.
    tweaked = tmp_path / "experiment.json"
    data = json.loads(experiment_file.read_text())
    data["evaluators"] = [{"evaluator_type": "AlwaysPasses", "label": "stub"}]
    tweaked.write_text(json.dumps(data))

    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            str(tweaked),
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer",
            "--custom-evaluator",
            "tests.strands_evals.cli.fixtures.evaluators:AlwaysPasses",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload["cases"][0]["evaluator"] == "AlwaysPasses"
    assert payload["test_passes"] == [True]


def test_validate_custom_evaluator(tmp_path: Path, capsys):
    """validate also accepts --custom-evaluator."""
    tweaked = tmp_path / "experiment.json"
    tweaked.write_text(
        json.dumps(
            {
                "cases": [{"input": "x", "expected_output": "y"}],
                "evaluators": [{"evaluator_type": "AlwaysPasses"}],
            }
        )
    )
    exit_code = main(
        [
            "--json",
            "validate",
            str(tweaked),
            "--custom-evaluator",
            "tests.strands_evals.cli.fixtures.evaluators:AlwaysPasses",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["evaluators"] == ["AlwaysPasses"]
