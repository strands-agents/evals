"""Tests for ``strands-evals optimize``: run, resume, apply, on scripted agents (no network)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from strands_evals import Case, Experiment
from strands_evals.cli import _optimize
from strands_evals.cli._entrypoint import import_attr
from strands_evals.cli.main import main
from strands_evals.optimization import (
    AtMost,
    Block,
    Budget,
    Maximize,
    OptimizationResult,
    Optimizer,
    Patch,
    Proposer,
    Search,
)
from tests.strands_evals.cli.fixtures.optimize_agents import EXACT, PROMPT_AGENT, Correct

AGENTS = "tests.strands_evals.cli.fixtures.optimize_agents"
CORRECT = f"{AGENTS}:Correct"


@pytest.fixture
def experiment_path(tmp_path: Path) -> Path:
    cases = [Case(name=f"q{i:02d}", input=f"question {i}", expected_output=f"answer {i}") for i in range(20)]
    path = tmp_path / "experiment.json"
    Experiment(cases=cases, evaluators=[Correct()]).to_file(str(path))
    return path


def _run_args(experiment_path: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--json",
        "optimize",
        "run",
        str(experiment_path),
        "--agent",
        f"{AGENTS}:manager_agent",
        "--custom-evaluator",
        CORRECT,
        "--objective",
        "maximize:Correct",
        "--constraint",
        "tokens<=450",
        "--max-iterations",
        "2",
        "--proposals-per-iteration",
        "2",
        "--max-workers",
        "40",
        "--base-dir",
        str(tmp_path / "runs"),
        *extra,
    ]


def _only_run(tmp_path: Path) -> Path:
    (root,) = (tmp_path / "runs").iterdir()
    return root


# run


def test_run_saves_its_inputs_and_prints_the_result(experiment_path, tmp_path, capsys):
    assert main(_run_args(experiment_path, tmp_path, "--name", "cli")) == 0
    out, err = capsys.readouterr()
    root = _only_run(tmp_path)
    assert root.name.startswith("cli-")
    assert json.loads(out) == json.loads((root / "result.json").read_text())
    assert f"run folder {root}" in err and "iteration 1" in err
    saved = json.loads((root / "cli.json").read_text())
    assert saved == {
        "schema_version": 1,
        "experiment_file": str(experiment_path.resolve()),
        "agent": f"{AGENTS}:manager_agent",
        "custom_evaluators": [CORRECT],
        "rollout": None,
        "environment": None,
        "objective": ["maximize:Correct"],
        "pareto": False,
        "constraints": ["tokens<=450"],
        "reflection_model": None,
    }
    run = json.loads((root / "run.json").read_text())
    assert run["objective"] == "Maximize(Correct)" and run["budget"]["max_iterations"] == 2


def test_run_rejects_a_factory_that_isnt_tunable(experiment_path, tmp_path, capsys):
    args = _run_args(experiment_path, tmp_path)
    args[args.index(f"{AGENTS}:manager_agent")] = f"{AGENTS}:plain_agent"
    assert main(args) == 2
    assert "not a tunable factory" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--objective", "most:Correct", "--objective must be maximize:METRIC"),
        ("--objective", "maximize:Correct:size=2", "must be weight=X or scale=X"),
        ("--constraint", "tokens<4000", "--constraint must be METRIC<=X"),
        ("--split", "0.5,0.5", "--split must be three numbers"),
        ("--agent-failure-score", "bad", "--agent-failure-score must be"),
        ("--objective", "maximize:Nope", "Nope"),  # an unknown metric, rejected by the Optimizer
    ],
)
def test_bad_flags_are_usage_errors(experiment_path, tmp_path, capsys, flag, value, message):
    assert main([*_run_args(experiment_path, tmp_path), flag, value]) == 2
    assert message in capsys.readouterr().err


def test_pareto_needs_two_objectives(experiment_path, tmp_path, capsys):
    assert main([*_run_args(experiment_path, tmp_path), "--pareto"]) == 2
    assert "--pareto needs at least two --objective" in capsys.readouterr().err


# resume


def test_resume_finishes_a_killed_run_from_its_folder_alone(experiment_path, tmp_path, capsys):
    assert main(_run_args(experiment_path, tmp_path)) == 0
    original = OptimizationResult.from_file(_only_run(tmp_path) / "result.json")
    root = original.run_dir
    lines = (root / "events.jsonl").read_text().splitlines()
    names = [json.loads(line)["event"] for line in lines]
    cut = next(i for i, n in enumerate(names) if n == "iteration_end") + 1  # killed after iteration 1
    killed = tmp_path / "killed" / root.name
    shutil.copytree(root, killed)
    (killed / "events.jsonl").write_text("\n".join(lines[:cut]) + "\n")
    (killed / "result.json").unlink()
    capsys.readouterr()

    assert main(["--json", "optimize", "resume", str(killed)]) == 0
    resumed = OptimizationResult.from_dict(json.loads(capsys.readouterr().out))
    assert resumed.best.id == original.best.id and resumed.candidates == original.candidates
    assert resumed.rollouts_spent == original.rollouts_spent


def test_resume_needs_cli_json(tmp_path, capsys):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "run.json").write_text("{}")
    assert main(["optimize", "resume", str(tmp_path / "run")]) == 2
    assert "has no cli.json" in capsys.readouterr().err


# apply


class AddInstruction(Proposer):
    """Appends the missing instruction to the prompt, once."""

    kinds = frozenset({"text"})

    def __init__(self):
        self.done = False

    async def propose(self, request):
        if self.done:
            return []
        self.done = True
        key = request.keys[0]
        return [Patch(updates={key: request.parent.values[key] + EXACT + "\n"}, rationale="ask for the exact answer")]

    def observe(self, observation):
        pass


@pytest.fixture
def finished_prompt_run(experiment_path, tmp_path, monkeypatch):
    """A finished run that improves `prompt.md`, started from Python with the inputs the CLI would have saved."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "prompt.md").write_text("You answer questions.\n")
    (project / "agent.py").write_text(PROMPT_AGENT)
    monkeypatch.chdir(project)
    spec = "./agent.py:prompt_agent"
    result = Optimizer(
        Experiment.from_file(str(experiment_path), custom_evaluators=[Correct]),
        import_attr(spec),
        objective=Maximize("Correct"),
        constraints=[AtMost("tokens", 450)],
        search=Search([Block(AddInstruction(), "prompt")]),
        budget=Budget(max_iterations=6),
        proposals_per_iteration=2,
        max_workers=40,
        base_dir=tmp_path / "runs",
    ).run()
    inputs = _optimize.Inputs(experiment_file=str(experiment_path), agent=spec, custom_evaluators=[CORRECT])
    _optimize.save_inputs(result.run_dir, inputs)
    _forget_agent_modules()  # `apply` is a new process: it must read the files as they are now
    assert result.improved
    return project, result


def _forget_agent_modules() -> None:
    for name in [name for name in sys.modules if name.startswith("_strands_evals_cli_agent_")]:
        del sys.modules[name]


def test_apply_writes_the_champions_file_and_lists_numbers_to_edit(finished_prompt_run, capsys, monkeypatch):
    project, result = finished_prompt_run
    monkeypatch.chdir(project.parent)  # apply runs in the run's working directory, wherever it is called from
    assert main(["optimize", "apply", str(result.run_dir), "--yes"]) == 0
    out = capsys.readouterr().out
    assert (project / "prompt.md").read_text() == result.best.values["prompt"]
    assert f"+{EXACT}" in out and f"wrote {project / 'prompt.md'}" in out
    if "temperature" in result.best.changed_values():
        assert "edit them by hand" in out and "temperature:" in out


def test_apply_dry_run_writes_nothing(finished_prompt_run, capsys):
    project, result = finished_prompt_run
    assert main(["optimize", "apply", str(result.run_dir), "--dry-run"]) == 0
    assert "dry run: nothing written" in capsys.readouterr().out
    assert (project / "prompt.md").read_text() == "You answer questions.\n"


def test_apply_refuses_a_file_changed_since_the_run(finished_prompt_run, capsys):
    project, result = finished_prompt_run
    (project / "prompt.md").write_text("Edited by hand.\n")
    assert main(["optimize", "apply", str(result.run_dir), "--yes"]) == 2
    assert "changed since the run" in capsys.readouterr().err
    assert (project / "prompt.md").read_text() == "Edited by hand.\n"


def test_apply_asks_unless_yes(finished_prompt_run, capsys, monkeypatch):
    project, result = finished_prompt_run
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert main(["optimize", "apply", str(result.run_dir)]) == 2
    assert "pass --yes" in capsys.readouterr().err
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert main(["optimize", "apply", str(result.run_dir)]) == 0
    assert (project / "prompt.md").read_text() == "You answer questions.\n"


def test_apply_the_baseline_by_number_changes_nothing(finished_prompt_run, capsys):
    _, result = finished_prompt_run
    assert main(["optimize", "apply", str(result.run_dir), "--candidate", "0"]) == 0
    assert "nothing to apply" in capsys.readouterr().out


def test_apply_needs_a_finished_run(finished_prompt_run, capsys):
    _, result = finished_prompt_run
    (result.run_dir / "result.json").unlink()
    assert main(["optimize", "apply", str(result.run_dir)]) == 2
    assert "the run hasn't finished" in capsys.readouterr().err
