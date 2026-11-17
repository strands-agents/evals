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
        "config": None,
        "config_sha256": None,
        "tune": [],
        "harness_version": None,
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


# --config: a strands-harness portable config


def _write_config(folder: Path, **extra) -> Path:
    config = {
        "name": "qa",
        "instructions": "You answer questions.",
        "modelModule": {"module": AGENTS, "export": "world"},
        "builtinTools": [],
        "builtinPlugins": [],
        "skills": False,
        **extra,
    }
    path = folder / "agent.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def _config_args(experiment_path: Path, config: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--json",
        "optimize",
        "run",
        str(experiment_path),
        "--config",
        str(config),
        "--custom-evaluator",
        CORRECT,
        "--objective",
        "maximize:Correct",
        "--max-iterations",
        "4",
        "--proposals-per-iteration",
        "2",
        "--max-workers",
        "40",
        "--base-dir",
        str(tmp_path / "runs"),
        *extra,
    ]


@pytest.fixture
def scripted_text_proposer(monkeypatch):
    """The default search's text block on a scripted proposer instead of Reflection (no model calls)."""
    monkeypatch.setattr("strands_evals.optimization.search.blocks.Reflection", AddInstruction)


def test_config_run_tunes_instructions_and_a_choice_then_apply_writes_both(
    experiment_path, tmp_path, monkeypatch, capsys, scripted_text_proposer
):
    project = tmp_path / "project"
    project.mkdir()
    config = _write_config(project, effort="auto")
    monkeypatch.chdir(project)
    args = _config_args(experiment_path, config, tmp_path, "--tune", 'effort=["auto","low"]')
    assert main(args) == 0
    root = _only_run(tmp_path)
    result = OptimizationResult.from_file(root / "result.json")
    assert result.improved and EXACT in result.best.values["instructions"]
    assert result.best.values["effort"] in ("auto", "low")
    assert "--- a/agent.json" in result.best.diff()  # the text change lands on the config file
    saved = json.loads((root / "cli.json").read_text())
    assert saved["config"] == str(config.resolve()) and saved["agent"] is None
    assert saved["tune"] == ['effort=["auto","low"]'] and saved["harness_version"]
    assert not (project / ".agent").exists()  # every trial's session and memory lived in its own temp folder
    capsys.readouterr()

    assert main(["optimize", "apply", str(root), "--yes"]) == 0
    written = json.loads(config.read_text())
    assert written["instructions"] == result.best.values["instructions"]
    assert written["effort"] == result.best.values["effort"]
    assert "edit them by hand" not in capsys.readouterr().out  # the config is the CLI's: choices are written too


def test_apply_refuses_a_config_changed_since_the_run(
    experiment_path, tmp_path, monkeypatch, capsys, scripted_text_proposer
):
    config = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(_config_args(experiment_path, config, tmp_path)) == 0
    config.write_text(config.read_text().replace('"qa"', '"renamed"'))
    assert main(["optimize", "apply", str(_only_run(tmp_path)), "--yes"]) == 2
    assert "changed since the run" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("config", "tune", "message"),
    [
        ({}, ["subagents.0.instructions"], "the config has nothing at that path"),
        ({}, ["name=[1, 2]"], "must be one of the options"),
        ({}, ["builtinTools"], "is a list, not text"),
        ({}, ["effort=[]"], "non-empty JSON list"),
        ({"a_b": "x", "a": {"b": "y"}}, ["a_b", "a.b"], "also named 'a_b'"),
        ({"skills": ["one", "two"]}, ["skills"], "skills must be one local folder path"),
    ],
)
def test_bad_tune_paths_are_usage_errors(experiment_path, tmp_path, capsys, config, tune, message):
    path = _write_config(tmp_path, **config)
    args = _config_args(experiment_path, path, tmp_path)
    for spec in tune:
        args += ["--tune", spec]
    assert main(args) == 2
    assert message in capsys.readouterr().err


def test_a_config_without_instructions_asks_for_one(experiment_path, tmp_path, capsys):
    path = _write_config(tmp_path)
    data = json.loads(path.read_text())
    del data["instructions"]
    path.write_text(json.dumps(data))
    assert main(_config_args(experiment_path, path, tmp_path)) == 2
    assert 'has no "instructions"; add one' in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--tune", "instructions"], "--tune needs --config"),
        (["--rollout", f"{AGENTS}:manager_agent"], None),
    ],
)
def test_agent_flags_go_with_their_agent(experiment_path, tmp_path, capsys, extra, message):
    if message is None:  # --rollout with --config
        args = [*_config_args(experiment_path, _write_config(tmp_path), tmp_path), *extra]
        message = "--rollout needs --agent"
    else:
        args = [*_run_args(experiment_path, tmp_path), *extra]
    assert main(args) == 2
    assert message in capsys.readouterr().err


def test_each_trial_gets_an_empty_session_and_a_copy_of_the_memory(tmp_path):
    from strands_evals.cli import _harness

    memory = tmp_path / "memories"
    memory.mkdir()
    (memory / "note.md").write_text("remember this")
    trial = _harness._fresh_state(
        {"session": {"id": "fixed"}, "memory": {"dir": "memories"}}, tmp_path, tmp_path / "trial"
    )
    assert trial["session"] == {"id": "fixed", "dir": str(tmp_path / "trial" / "session")}
    assert trial["memory"] == {"dir": str(tmp_path / "trial" / "memory")}
    assert (tmp_path / "trial" / "memory" / "note.md").read_text() == "remember this"
    assert _harness._fresh_state({"session": False, "memory": False}, tmp_path, tmp_path / "t2") == {
        "session": False,
        "memory": False,
    }


def test_apply_lists_a_key_active_on_one_side_as_inactive(experiment_path, tmp_path, capsys):
    assert main(_run_args(experiment_path, tmp_path)) == 0
    root = _only_run(tmp_path)
    result = OptimizationResult.from_file(root / "result.json")
    assert result.best.values["manager"] == "summary"  # every sliding window is over the token limit
    capsys.readouterr()
    assert main(["optimize", "apply", str(root), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "  manager: 'window' -> 'summary'\n" in out
    assert "  manager.window.window_size: 40 -> (inactive)\n" in out
