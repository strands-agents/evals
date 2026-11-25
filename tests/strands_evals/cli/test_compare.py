"""Tests for `strands-evals compare`."""

import builtins
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from strands_evals import Case, Experiment
from strands_evals.cli.main import main
from strands_evals.evaluators import Equals
from strands_evals.experimental import (
    CalibrationProfile,
    CalibrationTarget,
    CalibrationTargetDefinition,
    Calibrator,
)
from strands_evals.experimental.calibrator import CalibratedEqualAlgorithm
from strands_evals.types.evaluation_report import EvaluationReport


def _write_report(
    path: Path,
    delta: float,
    *,
    case_count: int = 6,
    evaluator: str = "correctness",
) -> None:
    scores = [0.2 + index * 0.05 + delta for index in range(case_count)]
    report = EvaluationReport(
        overall_score=sum(scores) / len(scores),
        scores=scores,
        cases=[
            {
                "name": f"case-{index}",
                "input": f"input-{index}",
                "evaluator": evaluator,
            }
            for index in range(case_count)
        ],
        test_passes=[True] * case_count,
        detailed_results=[[] for _ in scores],
    )
    path.write_text(report.model_dump_json(), encoding="utf-8")


def _write_calibration_profile(
    path: Path,
    *,
    target_name: str = "task_success",
    target_version: str = "1",
) -> CalibrationProfile:
    profile = CalibrationProfile.create(
        algorithm_id="calibrated_equal",
        algorithm_version=1,
        model_parameters={
            "weights": {"quality": 1.0},
            "intercept": 0.0,
            "scale": 1.0,
        },
        target_definition={"name": target_name, "version": target_version},
        weights={"quality": 1.0},
        status="calibrated",
        step_sizes={"quality": 0.1},
        variance_components={
            "quality": {"case": 0.1, "agent": 0.01, "judge": 0.01},
        },
        repeatabilities={"quality": 0.9},
        case_count=6,
        agent_runs=2,
        judge_replays=2,
        calibration_context=None,
        fit_structure_fingerprint="a" * 64,
    )
    profile.to_file(str(path))
    return profile


def test_importing_cli_does_not_load_comparison_dependencies():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import strands_evals.cli.main; "
                "assert 'scipy' not in sys.modules; "
                "assert 'strands_evals.experimental.comparison' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_compare_json_output(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0)
    _write_report(candidate, 0.1)

    exit_code = main(
        [
            "--json",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["verdict"] == "improved"
    assert payload["trial_count"] == 1
    assert payload["evaluator_summaries"][0]["evaluator"] == "correctness"


def test_compare_applies_calibration_profile_to_both_sides(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    profile_path = tmp_path / "profile.json"
    _write_report(baseline, 0.0, evaluator="quality")
    _write_report(candidate, 0.1, evaluator="quality")
    profile = _write_calibration_profile(profile_path)

    exit_code = main(
        [
            "--json",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--calibration-profile",
            str(profile_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["verdict"] == "improved"
    assert payload["profile_id"] == profile.profile_id
    assert payload["target_definition"] == {"name": "task_success", "version": "1"}
    assert payload["evaluator_summaries"][0]["evaluator"] == "target:task_success@1"
    assert payload["evaluator_summaries"][0]["mean_delta"] == pytest.approx(0.0241, abs=0.0001)


def test_compare_calibration_rich_output_names_target(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    profile_path = tmp_path / "profile.json"
    _write_report(baseline, 0.0, evaluator="quality")
    _write_report(candidate, 0.1, evaluator="quality")
    profile = _write_calibration_profile(profile_path)

    exit_code = main(
        [
            "--rich",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--calibration-profile",
            str(profile_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"Target comparison (task_success@1, profile {profile.profile_id[:12]}): improved" in output


@pytest.mark.parametrize("target_name", ["pass[strict]", "a[/b]"])
def test_compare_calibration_rich_output_treats_target_name_as_text(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    profile_path = tmp_path / "profile.json"
    _write_report(baseline, 0.0, evaluator="quality")
    _write_report(candidate, 0.1, evaluator="quality")
    _write_calibration_profile(profile_path, target_name=target_name)
    monkeypatch.setenv("COLUMNS", "120")

    exit_code = main(
        [
            "--rich",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--calibration-profile",
            str(profile_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert target_name in output


def test_compare_calibration_rejects_lower_is_better(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    profile_path = tmp_path / "profile.json"
    _write_report(baseline, 0.0, evaluator="quality")
    _write_report(candidate, 0.1, evaluator="quality")
    _write_calibration_profile(profile_path)

    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--calibration-profile",
            str(profile_path),
            "--lower-is-better",
            "quality",
        ]
    )

    assert exit_code == 2
    assert "--lower-is-better cannot be used with --calibration-profile" in capsys.readouterr().err


def test_compare_calibration_passes_statistical_options(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    profile_path = tmp_path / "profile.json"
    _write_report(baseline, 0.0, case_count=5, evaluator="quality")
    _write_report(candidate, 0.1, case_count=5, evaluator="quality")
    _write_calibration_profile(profile_path)

    exit_code = main(
        [
            "--json",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--calibration-profile",
            str(profile_path),
            "--confidence-level",
            "0.9",
            "--minimum-effect",
            "0.03",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    summary = payload["evaluator_summaries"][0]

    assert exit_code == 0
    assert summary["confidence_level"] == 0.9
    assert summary["minimum_effect"] == 0.03
    assert summary["verdict"] == "no_clear_change"


def test_compare_calibration_excludes_corrupted_trial_before_scoring(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    profile_path = tmp_path / "profile.json"
    _write_report(baseline, 0.0, evaluator="quality")
    _write_report(candidate, 0.1, evaluator="quality")
    _write_calibration_profile(profile_path)

    exit_code = main(
        [
            "--json",
            "compare",
            "--baseline",
            str(tmp_path / "missing-baseline.json"),
            "--baseline",
            str(baseline),
            "--candidate",
            str(tmp_path / "missing-candidate.json"),
            "--candidate",
            str(candidate),
            "--calibration-profile",
            str(profile_path),
            "--corrupted-trial",
            "0",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["trial_count"] == 1
    assert payload["corrupted_trial_indices"] == [0]
    assert payload["evaluator_summaries"][0]["mean_delta"] == pytest.approx(0.0241, abs=0.0001)


def test_compare_calibration_wraps_profile_load_error(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    missing_profile = tmp_path / "missing-profile.json"
    _write_report(baseline, 0.0, evaluator="quality")
    _write_report(candidate, 0.1, evaluator="quality")

    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--calibration-profile",
            str(missing_profile),
        ]
    )

    assert exit_code == 2
    assert f"invalid calibration profile at {missing_profile}" in capsys.readouterr().err


def test_compare_calibration_accepts_profile_fitted_from_contextual_reports(tmp_path: Path, capsys):
    cases = [Case(name=f"training-case-{index}", input=str(index), expected_output="pass") for index in range(6)]
    experiment = Experiment(cases=cases, evaluators=[Equals(name="quality")])
    targets_by_case = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    targets = [
        CalibrationTarget(case_name=case.name, agent_run=agent_run, value=targets_by_case[index])
        for agent_run in range(2)
        for index, case in enumerate(cases)
        if case.name is not None
    ]
    calibrator = Calibrator(
        step_sizes={"quality": 1.0},
        algorithm=CalibratedEqualAlgorithm(),
    )
    calibrator.run_evaluations(
        experiment,
        lambda case: "pass" if int(case.input) % 2 else "fail",
        agent_runs=2,
        judge_replays=2,
        targets=targets,
        target_definition=CalibrationTargetDefinition(name="task_success", version="1"),
    )
    assert calibrator.profile is not None
    profile_path = tmp_path / "contextual-profile.json"
    calibrator.profile.to_file(str(profile_path))
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0, evaluator="quality")
    _write_report(candidate, 0.1, evaluator="quality")

    exit_code = main(
        [
            "--json",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--calibration-profile",
            str(profile_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["profile_id"] == calibrator.profile.profile_id


def test_compare_multiple_trials_and_exclusion(tmp_path: Path, capsys):
    baseline_0 = tmp_path / "baseline-0.json"
    baseline_1 = tmp_path / "baseline-1.json"
    candidate_0 = tmp_path / "candidate-0.json"
    candidate_1 = tmp_path / "candidate-1.json"
    _write_report(baseline_0, 0.0)
    _write_report(baseline_1, 0.02)
    _write_report(candidate_0, 0.1)
    candidate_1.write_text("{broken", encoding="utf-8")

    exit_code = main(
        [
            "--json",
            "compare",
            "--baseline",
            str(baseline_0),
            "--baseline",
            str(baseline_1),
            "--candidate",
            str(candidate_0),
            "--candidate",
            str(candidate_1),
            "--corrupted-trial",
            "1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["verdict"] == "improved"
    assert payload["trial_count"] == 1
    assert payload["corrupted_trial_indices"] == [1]


def test_compare_rich_output(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0)
    _write_report(candidate, 0.1)

    exit_code = main(
        [
            "--rich",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Comparison: improved" in output
    assert "correctness" in output


def test_compare_rich_output_preserves_values_at_narrow_width(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0, evaluator="HelpfulnessEvaluator")
    _write_report(candidate, 0.1, evaluator="HelpfulnessEvaluator")
    monkeypatch.setenv("COLUMNS", "80")

    exit_code = main(
        [
            "--rich",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "…" not in output
    assert "[0.1000, 0.1000]" in output
    assert "0.03125" in output
    assert "improved" in output


def test_compare_rich_output_treats_evaluator_names_as_text(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0, evaluator="score[/v2]")
    _write_report(candidate, 0.1, evaluator="score[/v2]")

    exit_code = main(
        [
            "--rich",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "score[/v2]" in output


def test_compare_output_file_is_json(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    _write_report(baseline, 0.0)
    _write_report(candidate, 0.1)

    exit_code = main(
        [
            "--rich",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["verdict"] == "improved"


@pytest.mark.parametrize(
    ("extra_args", "expected_error"),
    [
        (["--confidence-level", "95"], "--confidence-level"),
        (["--corrupted-trial", "5"], "--corrupted-trial"),
    ],
)
def test_compare_invalid_options_return_bad_input(
    tmp_path: Path,
    capsys,
    extra_args: list[str],
    expected_error: str,
):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0)
    _write_report(candidate, 0.1)

    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            *extra_args,
        ]
    )
    error = capsys.readouterr().err

    assert exit_code == 2
    assert expected_error in error


def test_compare_unequal_report_counts_return_bad_input(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0)
    _write_report(candidate, 0.1)

    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ]
    )
    error = capsys.readouterr().err

    assert exit_code == 2
    assert "--baseline and --candidate" in error


def test_compare_invalid_report_identifies_trial_and_source(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0)
    candidate.write_text("{broken", encoding="utf-8")

    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ]
    )
    error = capsys.readouterr().err

    assert exit_code == 2
    assert "trial 0" in error
    assert str(candidate) in error


def test_compare_error_displays_windows_path_without_repr_escaping(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = r"C:\reports\candidate.json"
    _write_report(baseline, 0.0)

    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            candidate,
        ]
    )
    error = capsys.readouterr().err

    assert exit_code == 2
    assert f"source={candidate}:" in error


def test_compare_supports_lower_is_better(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0)
    _write_report(candidate, -0.1)

    exit_code = main(
        [
            "--json",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--lower-is-better",
            "correctness",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["verdict"] == "improved"
    assert payload["evaluator_summaries"][0]["higher_is_better"] is False


def test_compare_rejects_unknown_lower_is_better_name(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0)
    _write_report(candidate, -0.1)

    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--lower-is-better",
            "correctnes",
        ]
    )
    error = capsys.readouterr().err

    assert exit_code == 2
    assert "--lower-is-better names unknown evaluators" in error
    assert "correctnes" in error
    assert "correctness" in error


def test_compare_passes_statistical_options(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0, case_count=5)
    _write_report(candidate, 0.1, case_count=5)

    exit_code = main(
        [
            "--json",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--confidence-level",
            "0.9",
            "--minimum-effect",
            "0.05",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    summary = payload["evaluator_summaries"][0]

    assert exit_code == 0
    assert summary["confidence_level"] == 0.9
    assert summary["minimum_effect"] == 0.05
    assert summary["verdict"] == "improved"


def test_compare_reads_report_from_stdin(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0)
    _write_report(candidate, 0.1)
    monkeypatch.setattr(sys, "stdin", StringIO(baseline.read_text(encoding="utf-8")))

    exit_code = main(
        [
            "--json",
            "compare",
            "--baseline",
            "-",
            "--candidate",
            str(candidate),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["verdict"] == "improved"


def test_compare_missing_optional_dependencies_returns_bad_input(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, 0.0)
    _write_report(candidate, 0.1)
    original_import = builtins.__import__

    def import_without_comparison(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.endswith("experimental.comparison"):
            raise ImportError("missing comparison extra")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_comparison)

    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ]
    )
    error = capsys.readouterr().err

    assert exit_code == 2
    assert "missing comparison extra" in error


def test_compare_help_marks_command_experimental_and_documents_stdin(capsys):
    with pytest.raises(SystemExit) as root_exit:
        main(["--help"])
    root_help = capsys.readouterr().out

    with pytest.raises(SystemExit) as compare_exit:
        main(["compare", "--help"])
    compare_help = capsys.readouterr().out

    assert root_exit.value.code == 0
    assert "experimentally compare paired evaluation reports" in root_help
    assert compare_exit.value.code == 0
    assert "or - for stdin" in compare_help
    assert "JSON calibration profile path" in compare_help
    assert "target-probability units" in compare_help


def test_compare_accepts_experiment_reports(tmp_path: Path, capsys):
    cases = [Case(name=f"case-{index}", input=f"question-{index}", expected_output="answer") for index in range(6)]
    experiment = Experiment(cases=cases, evaluators=[Equals(name="equals")])
    baseline_report = experiment.run_evaluations(lambda case: case.expected_output)
    candidate_report = experiment.run_evaluations(lambda _: "wrong")
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline_report.to_file(str(baseline))
    candidate_report.to_file(str(candidate))

    exit_code = main(
        [
            "--json",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["verdict"] == "regressed"
    assert payload["evaluator_summaries"][0]["evaluator"] == "equals"
