"""Tests for ``strands-evals optimize``: run and resume, on scripted agents (no network)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from strands_evals import Case, Experiment
from strands_evals.cli.main import main
from strands_evals.optimization import OptimizationResult, Patch, Proposer
from tests.strands_evals.cli.fixtures.optimize_agents import EXACT, Correct

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


# --config: a strands-harness portable config


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


def test_config_run_tunes_instructions_and_a_choice(
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


def test_best_values_json_shows_a_key_active_on_one_side(experiment_path, tmp_path):
    assert main(_run_args(experiment_path, tmp_path)) == 0
    root = _only_run(tmp_path)
    result = OptimizationResult.from_file(root / "result.json")
    assert result.best.values["manager"] == "summary"  # every sliding window is over the token limit
    values = json.loads((root / "best.values.json").read_text())
    assert values["manager"] == {"old": "window", "new": "summary"}
    assert values["manager.window.window_size"] == {"old": 40}
