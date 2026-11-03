"""Command integration tests with deterministic local runners and judges."""

import argparse
import json
from unittest.mock import Mock

import pytest

from strands_evals.cli._common import make_global_parent
from strands_evals.cli.commands import optimize
from strands_evals.types.evaluation_report import EvaluationReport


def arguments(tmp_path, *extra):
    parser = argparse.ArgumentParser()
    optimize.add_subparser(parser.add_subparsers(dest="command"), make_global_parent())
    (tmp_path / "agent.py").write_text("# source\n")
    (tmp_path / "instructions.md").write_text("original\n")
    (tmp_path / "cases").mkdir(exist_ok=True)
    for i in range(8):
        (tmp_path / "cases" / f"{i}.json").write_text(
            json.dumps({"id": str(i), "userTurns": [{"content": f"question {i}"}], "assertion": "answer"})
        )
    (tmp_path / "proposals.json").write_text(json.dumps(["candidate one\n", "candidate two\n"]))
    return parser.parse_args(
        [
            "optimize",
            str(tmp_path),
            "--cases",
            str(tmp_path / "cases"),
            "--instructions-file",
            str(tmp_path / "instructions.md"),
            "--budget",
            "2",
            "--proposal-file",
            str(tmp_path / "proposals.json"),
            *extra,
        ]
    )


def test_dry_run_never_loads_source_or_calls_models(tmp_path, monkeypatch, capsys):
    runner = Mock(side_effect=AssertionError("must not load source"))
    judge = Mock(side_effect=AssertionError("must not evaluate"))
    monkeypatch.setattr("strands_evals.cli._project.ProjectRunner", runner)
    monkeypatch.setattr("strands_evals.cli._project.evaluate_project", judge)
    monkeypatch.setattr(optimize, "generate_proposal", Mock(side_effect=AssertionError("must not propose")))
    assert optimize._run(arguments(tmp_path, "--dry-run")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"]
    assert report["max_rollouts"] == 19
    assert report["proposal_budget"] == 2
    assert report["monetary_estimate"] is None
    assert not (tmp_path / ".strands").exists()
    runner.assert_not_called()
    judge.assert_not_called()


def test_command_uses_runtime_overrides_and_resumes(tmp_path, monkeypatch, capsys):
    runner = Mock()
    runner.get_instructions.return_value = "original\n"
    monkeypatch.setattr("strands_evals.cli._project.ProjectRunner", Mock(return_value=runner))
    observed = []

    def evaluate(actual_runner, cases, overrides):
        assert actual_runner is runner
        assert set(overrides) == {"instructions"}
        observed.append(([case.name for case in cases], overrides["instructions"]))
        score = 0.2 if overrides["instructions"] == "original\n" else 0.8
        return EvaluationReport(
            overall_score=score,
            scores=[score] * len(cases),
            cases=[{} for _ in cases],
            test_passes=[True] * len(cases),
        )

    monkeypatch.setattr("strands_evals.cli._project.evaluate_project", evaluate)
    args = arguments(tmp_path)
    assert optimize._run(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["accepted"]
    assert len(observed) == 19
    assert [entry[0] for entry in observed[:3]] == [entry[0] for entry in observed[3:6]]
    assert {entry[0][0] for entry in observed[:3]}.isdisjoint(entry[0][0] for entry in observed[9:14])
    assert (tmp_path / "instructions.md").read_text() == "original\n"
    args.resume = first["run_id"]
    assert optimize._run(args) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == first["run_id"]
    assert len(observed) == 19
    # A changed source must not silently reuse cached results.
    (tmp_path / "agent.py").write_text("# changed\n")
    assert optimize._run(args) == 2
    assert "changed" in capsys.readouterr().err


def test_source_extraction_mismatch_fails_before_evaluation(tmp_path, monkeypatch, capsys):
    runner = Mock()
    runner.get_instructions.return_value = "unrelated source prompt"
    monkeypatch.setattr("strands_evals.cli._project.ProjectRunner", Mock(return_value=runner))
    judge = Mock(side_effect=AssertionError("must not evaluate"))
    monkeypatch.setattr("strands_evals.cli._project.evaluate_project", judge)
    assert optimize._run(arguments(tmp_path)) == 2
    assert "does not exactly match" in capsys.readouterr().err
    judge.assert_not_called()


def test_explicit_instructions_must_be_inside_project(tmp_path, capsys):
    args = arguments(tmp_path)
    args.instructions_file = str(tmp_path.parent / "outside.md")
    assert optimize._run(args) == 2
    assert "inside the project" in capsys.readouterr().err


@pytest.mark.parametrize("contents", ["[]", '["one"]', "[1,2]", "{}"])
def test_offline_proposals_validate_budget_and_shape(tmp_path, contents):
    file = tmp_path / "proposals.json"
    file.write_text(contents)
    with pytest.raises(ValueError):
        optimize._load_proposals(str(file), 2)


def test_real_python_runner_uses_instructions_override(tmp_path, monkeypatch, capsys):
    from strands_evals.cli._project import AssertionContains, evaluate_project

    args = arguments(tmp_path)
    (tmp_path / "agent.py").write_text(
        "class Agent:\n"
        "    system_prompt = 'original\\n'\n"
        "    def __call__(self, prompt):\n"
        "        return 'answer' if self.system_prompt.startswith('candidate') else 'wrong'\n"
        "agent = Agent()\n"
    )
    monkeypatch.setattr(
        "strands_evals.cli._project.evaluate_project",
        lambda runner, cases, overrides: evaluate_project(runner, cases, [AssertionContains()], overrides),
    )
    assert optimize._run(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["accepted"]
    assert report["evaluations"]["baseline_training"]["overall_score"] == 0
    assert report["evaluations"]["winner_holdout"]["overall_score"] == 1
    assert (tmp_path / "instructions.md").read_text() == "original\n"
