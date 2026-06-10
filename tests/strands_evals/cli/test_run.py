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
    assert len(payload["cases"]) == 1
    assert payload["cases"][0]["evaluator"]
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


def test_run_interactive_renders_collapsed(experiment_file: Path, monkeypatch):
    """`--interactive` should hand a collapsed items dict to the renderer in interactive mode."""
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
            "--interactive",
        ]
    )

    assert exit_code == 0
    assert captured["static"] is False
    items = captured["items"]
    assert items, "expected at least one row in the items dict"
    for row in items.values():
        assert row["expanded"] is False


def test_run_display_and_interactive_combined(experiment_file: Path, monkeypatch):
    """`--display --interactive` is allowed; interactive wins (collapsed + interactive run)."""
    from strands_evals.cli.commands import run as run_module

    captured: dict = {}

    class _FakeDisplay:
        def __init__(self, items: dict, overall_score: float) -> None:
            captured["items"] = items

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
            "-i",
        ]
    )

    assert exit_code == 0
    assert captured["static"] is False
    for row in captured["items"].values():
        assert row["expanded"] is False


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


def test_run_invalid_entry_point_exits_bad_input(experiment_file: Path, capsys):
    """Bad --task spec is a user input error, so the CLI exits 2 (EXIT_BAD_INPUT)."""
    exit_code = main(
        [
            "run",
            str(experiment_file),
            "--task",
            "nonexistent_module_xyz:foo",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
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


def test_run_ad_hoc_with_expected_output_passes(capsys, tmp_path: Path):
    """Ad-hoc mode: --input + --expected-output defaults to Contains and passes."""
    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--expected-output",
            "answered: hi",
            "-o",
            str(out_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert len(payload["cases"]) == 1
    assert payload["cases"][0]["name"] == "ad_hoc"
    assert payload["cases"][0]["input"] == "hi"
    assert payload["cases"][0]["evaluator"] == "Contains"
    assert payload["test_passes"] == [True]


def test_run_ad_hoc_explicit_evaluator_shortname(tmp_path: Path):
    """Ad-hoc mode: --evaluator equals (shortname) wires up Equals against expected_output."""
    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--expected-output",
            "answered: hi",
            "--evaluator",
            "equals",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload["cases"][0]["evaluator"] == "Equals"
    assert payload["test_passes"] == [True]


def test_run_ad_hoc_custom_evaluator_via_module_class(tmp_path: Path):
    """Ad-hoc mode: --evaluator MODULE:CLASS resolves a custom Evaluator subclass."""
    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--evaluator",
            "tests.strands_evals.cli.fixtures.evaluators:AlwaysPasses",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload["cases"][0]["evaluator"] == "AlwaysPasses"
    assert payload["test_passes"] == [True]


def test_run_ad_hoc_input_file_from_stdin(monkeypatch, tmp_path: Path):
    """Ad-hoc mode: --input-file - reads case input from stdin."""
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("hi"))
    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input-file",
            "-",
            "--expected-output",
            "answered: hi",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload["cases"][0]["input"] == "hi"


def test_run_ad_hoc_name_override(tmp_path: Path):
    """Ad-hoc mode: --name overrides the default 'ad_hoc' case name."""
    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--expected-output",
            "answered: hi",
            "--name",
            "smoke",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload["cases"][0]["name"] == "smoke"


def test_run_experiment_file_and_ad_hoc_flags_conflict(experiment_file: Path, capsys):
    """EXPERIMENT_FILE and ad-hoc flags are mutually exclusive — exit 2 with a clear error."""
    exit_code = main(
        [
            "run",
            str(experiment_file),
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "mutually exclusive" in captured.err
    assert "--input" in captured.err


def test_run_no_experiment_file_and_no_input_errors(capsys):
    """Without EXPERIMENT_FILE or --input, the run has nothing to execute — exit 2."""
    exit_code = main(
        [
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "EXPERIMENT_FILE or --input" in captured.err


def test_run_ad_hoc_input_without_evaluator_errors(capsys):
    """--input alone (no --evaluator, --expected-output, or --rubric) is invalid — exit 2."""
    exit_code = main(
        [
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--evaluator" in captured.err
    assert "--expected-output" in captured.err
    assert "--rubric" in captured.err


def test_run_ad_hoc_rubric_auto_wires_output_evaluator(monkeypatch, tmp_path: Path):
    """--rubric TEXT alone is enough; auto-wires OutputEvaluator(rubric=TEXT)."""
    from strands_evals.cli.commands import run as run_module
    from strands_evals.evaluators.output_evaluator import OutputEvaluator
    from strands_evals.types.evaluation import EvaluationOutput

    captured: dict = {}

    async def _fake_evaluate(self, evaluation_case):
        captured["rubric"] = self.rubric
        return [EvaluationOutput(score=1.0, test_pass=True, reason="ok")]

    monkeypatch.setattr(OutputEvaluator, "evaluate_async", _fake_evaluate)
    monkeypatch.setattr(run_module.OutputEvaluator, "evaluate_async", _fake_evaluate)

    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--rubric",
            "Output should be cheerful.",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0
    assert captured["rubric"] == "Output should be cheerful."
    payload = json.loads(out_path.read_text())
    assert payload["cases"][0]["evaluator"] == "OutputEvaluator"
    assert payload["test_passes"] == [True]


def test_run_ad_hoc_rubric_and_expected_output_compose(monkeypatch, tmp_path: Path):
    """--rubric + --expected-output (no --evaluator) auto-wires BOTH evaluators."""
    from strands_evals.cli.commands import run as run_module
    from strands_evals.evaluators.output_evaluator import OutputEvaluator
    from strands_evals.types.evaluation import EvaluationOutput

    async def _fake_evaluate(self, evaluation_case):
        return [EvaluationOutput(score=1.0, test_pass=True, reason="ok")]

    monkeypatch.setattr(OutputEvaluator, "evaluate_async", _fake_evaluate)
    monkeypatch.setattr(run_module.OutputEvaluator, "evaluate_async", _fake_evaluate)

    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--expected-output",
            "answered: hi",
            "--rubric",
            "Output should mirror the input.",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    evaluators = sorted(c["evaluator"] for c in payload["cases"])
    assert evaluators == ["Contains", "OutputEvaluator"]


def test_run_ad_hoc_explicit_evaluator_suppresses_rubric_auto_wire(tmp_path: Path):
    """--evaluator wins over --rubric: the explicit list is never silently extended."""
    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--rubric",
            "ignored",
            "--evaluator",
            "tests.strands_evals.cli.fixtures.evaluators:AlwaysPasses",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    evaluators = [c["evaluator"] for c in payload["cases"]]
    assert evaluators == ["AlwaysPasses"]


def test_run_ad_hoc_unknown_evaluator_shortname_exits_bad_input(capsys):
    """Unknown shortname surfaces an EntryPointError message and exit 2."""
    exit_code = main(
        [
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--evaluator",
            "not-a-real-evaluator",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "not a known built-in shortname" in captured.err


def test_run_input_and_input_file_mutually_exclusive(experiment_file: Path):
    """argparse enforces --input vs --input-file mutual exclusion (exits via SystemExit)."""
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--task",
                "tests.strands_evals.cli.fixtures.tasks:answer_string",
                "--input",
                "hi",
                "--input-file",
                "-",
                "--expected-output",
                "x",
            ]
        )


def test_run_ad_hoc_trace_level_evaluator_with_task_errors(capsys):
    """Trace-level shortnames (faithfulness, helpfulness, ...) need --agent's Session — fail fast under --task."""
    exit_code = main(
        [
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--evaluator",
            "faithfulness",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "FaithfulnessEvaluator" in captured.err
    assert "--agent" in captured.err


def test_run_ad_hoc_tool_level_evaluator_with_task_errors(capsys):
    """Tool-level shortnames also need a Session — fail fast under --task."""
    exit_code = main(
        [
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--evaluator",
            "tool-selection-accuracy",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "ToolSelectionAccuracyEvaluator" in captured.err


def test_run_ad_hoc_equals_without_expected_output_errors(capsys):
    """`--evaluator equals` without `--expected-output` would compare against None — reject up front."""
    exit_code = main(
        [
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--evaluator",
            "equals",
            "--rubric",
            "anything",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--expected-output" in captured.err


def test_run_ad_hoc_equals_with_expected_output_passes(tmp_path: Path):
    """`equals` paired with `--expected-output` is the supported one-liner shape."""
    out_path = tmp_path / "reports.json"
    exit_code = main(
        [
            "--json",
            "run",
            "--task",
            "tests.strands_evals.cli.fixtures.tasks:answer_string",
            "--input",
            "hi",
            "--evaluator",
            "equals",
            "--expected-output",
            "answered: hi",
            "-o",
            str(out_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload["test_passes"] == [True]


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
