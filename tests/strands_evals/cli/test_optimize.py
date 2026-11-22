"""Tests for ``strands-evals optimize``: run and resume, on scripted agents (no network)."""

from __future__ import annotations

import importlib
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
        "build_task": None,
        "space": None,
        "config": None,
        "config_sha256": None,
        "tune": [],
        "tune_instructions": True,
        "harness_version": None,
        "custom_evaluators": [CORRECT],
        "rollout": None,
        "environment": None,
        "objective": ["maximize:Correct"],
        "pareto": False,
        "constraints": ["tokens<=450"],
        "reflection_model": None,
        "search": None,
        "selection": None,
        "score_model": None,
        "on_iteration": None,
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


def test_resume_runs_the_test_of_a_run_whose_test_was_not_saved_as_done(experiment_path, tmp_path, capsys):
    # A run with result.json but no `test` event is what a test report skipped for the budget or for
    # infrastructure failures leaves behind: resume must run the test, not print the saved result.
    assert main(_run_args(experiment_path, tmp_path)) == 0
    original = OptimizationResult.from_file(_only_run(tmp_path) / "result.json")
    root = original.run_dir
    lines = (root / "events.jsonl").read_text().splitlines()
    skipped = tmp_path / "skipped" / root.name
    shutil.copytree(root, skipped)
    kept = [line for line in lines if json.loads(line)["event"] != "test"]
    (skipped / "events.jsonl").write_text("\n".join(kept) + "\n")
    capsys.readouterr()

    assert main(["--json", "optimize", "resume", str(skipped)]) == 0
    out, err = capsys.readouterr()
    assert "the run has finished" not in err
    resumed = OptimizationResult.from_dict(json.loads(out))
    assert resumed.test is not None and resumed.best.id == original.best.id
    events = [json.loads(line)["event"] for line in (skipped / "events.jsonl").read_text().splitlines()]
    assert events.count("test") == 1  # the resume ran the test and saved it as done


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
    json.loads(capsys.readouterr().out)  # stdout is the result alone: trials don't print their answers
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
        (["--environment", f"{AGENTS}:manager_agent"], None),
    ],
)
def test_agent_flags_go_with_their_agent(experiment_path, tmp_path, capsys, extra, message):
    if message is None:  # --environment with --config
        args = [*_config_args(experiment_path, _write_config(tmp_path), tmp_path), *extra]
        message = "--environment needs --agent"
    else:
        args = [*_run_args(experiment_path, tmp_path), *extra]
    assert main(args) == 2
    assert message in capsys.readouterr().err


def test_config_run_takes_a_rollout(experiment_path, tmp_path, monkeypatch, scripted_text_proposer):
    from tests.strands_evals.cli.fixtures import optimize_agents

    monkeypatch.setattr(optimize_agents, "ROLLOUTS", [])
    config = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(_config_args(experiment_path, config, tmp_path, "--rollout", f"{AGENTS}:harness_rollout")) == 0
    assert optimize_agents.ROLLOUTS  # every trial went through the rollout
    assert OptimizationResult.from_file(_only_run(tmp_path) / "result.json").improved


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


# MODULE:ATTR objects and the --build-task route


def _hook_args(experiment_path: Path, tmp_path: Path, *extra: str) -> list[str]:
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
        f"{AGENTS}:objective",
        "--constraint",
        f"{AGENTS}:constraints",
        "--search",
        f"{AGENTS}:search",
        "--selection",
        f"{AGENTS}:first_ranked",
        "--score-model",
        f"{AGENTS}:scores",
        "--final-pick",
        f"{AGENTS}:final_pick",
        "--split",
        f"{AGENTS}:split",
        "--on-iteration",
        f"{AGENTS}:record_iteration",
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


def _iterations(root: Path) -> list[int]:
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    return [event["iteration"] for event in events if event["event"] == "iteration_end"]


def _fresh_fixtures():
    """The fixture module, imported again: its objects as a new process would see them (a `Search` keeps its
    proposers' state from the run that used it)."""
    from tests.strands_evals.cli.fixtures import optimize_agents

    return importlib.reload(optimize_agents)


@pytest.fixture
def hooks():
    """The fixture module's objects, fresh."""
    return _fresh_fixtures()


def test_module_attr_objects_reach_the_run(experiment_path, tmp_path, capsys, hooks):
    assert main(_hook_args(experiment_path, tmp_path)) == 0
    root = _only_run(tmp_path)
    run = json.loads((root / "run.json").read_text())
    assert run["objective"] == "Maximize(Correct)" and run["constraints"] == ["tokens <= 450"]
    assert [block["config"]["n_startup_trials"] for block in run["search"]["blocks"] if block["proposer"] == "TPE"]
    assert run["search"]["blocks"][0]["keys"] == ["temperature"]
    assert run["selection"]["name"] == "first_ranked" and run["score_model"]["name"] == "scores"
    assert run["final_pick"] == {"n_finalists": 2, "rule": {"kind": "LowerBound", "q": 0.2}}
    assert run["split"]["cases"]["train"] == [f"q{i:02d}" for i in range(8)]
    assert hooks.ITERATIONS == _iterations(root)  # the async callback ran after every iteration
    saved = json.loads((root / "cli.json").read_text())
    assert saved["search"] == f"{AGENTS}:search" and saved["on_iteration"] == f"{AGENTS}:record_iteration"
    assert saved["objective"] == [f"{AGENTS}:objective"] and saved["constraints"] == [f"{AGENTS}:constraints"]


def test_resume_reimports_the_objects(experiment_path, tmp_path, capsys, hooks):
    assert main(_hook_args(experiment_path, tmp_path)) == 0
    original = OptimizationResult.from_file(_only_run(tmp_path) / "result.json")
    root = original.run_dir
    lines = (root / "events.jsonl").read_text().splitlines()
    cut = next(i for i, line in enumerate(lines) if json.loads(line)["event"] == "iteration_end") + 1
    killed = tmp_path / "killed" / root.name
    shutil.copytree(root, killed)
    (killed / "events.jsonl").write_text("\n".join(lines[:cut]) + "\n")
    (killed / "result.json").unlink()
    capsys.readouterr()
    hooks = _fresh_fixtures()  # resume is a new process

    assert main(["--json", "optimize", "resume", str(killed)]) == 0
    resumed = OptimizationResult.from_dict(json.loads(capsys.readouterr().out))
    assert resumed.best.id == original.best.id and resumed.candidates == original.candidates
    assert hooks.ITERATIONS == _iterations(root)[1:]  # only the iterations run after the resume


@pytest.mark.parametrize(
    ("flag", "attr", "message"),
    [
        ("--search", "final_pick", "is a FinalPick, not a Search"),
        ("--selection", "search", "is a Search, not a Selection"),
        ("--score-model", "search", "is a Search, not a ScoreModel"),
        ("--final-pick", "search", "is a Search, not a FinalPick"),
        ("--split", "search", "is a Search, not a Split"),
        ("--on-iteration", "EXACT", "is a str, not a callable"),
        ("--constraint", "search", "is a Search, not a constraint"),
    ],
)
def test_a_hook_of_the_wrong_type_is_a_usage_error(experiment_path, tmp_path, capsys, flag, attr, message):
    assert main([*_run_args(experiment_path, tmp_path), flag, f"{AGENTS}:{attr}"]) == 2
    assert message in capsys.readouterr().err


def test_an_objective_hook_must_be_an_objective_and_alone(experiment_path, tmp_path, capsys):
    args = [*_run_args(experiment_path, tmp_path), "--objective", f"{AGENTS}:objective"]
    assert main(args) == 2
    assert "names the whole objective" in capsys.readouterr().err  # with _run_args' own --objective
    args = [a for a in _run_args(experiment_path, tmp_path) if a != "maximize:Correct"]
    args.remove("--objective")
    assert main([*args, "--objective", f"{AGENTS}:search"]) == 2
    assert "is a Search, not an objective" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--final-pick", f"{AGENTS}:final_pick", "--n-finalists", "2"], "--final-pick and --n-finalists"),
        (["--search", f"{AGENTS}:search", "--reflection-model", "m"], "--search and --reflection-model"),
        (["--space", f"{AGENTS}:space"], "--build-task and --space go together"),
    ],
)
def test_flags_that_exclude_or_need_each_other(experiment_path, tmp_path, capsys, extra, message):
    assert main([*_run_args(experiment_path, tmp_path), *extra]) == 2
    assert message in capsys.readouterr().err


def _build_task_args(experiment_path: Path, tmp_path: Path, *extra: str) -> list[str]:
    args = _run_args(experiment_path, tmp_path)
    at = args.index("--agent")
    return [*args[:at], "--build-task", f"{AGENTS}:build_task", "--space", f"{AGENTS}:space", *args[at + 2 :], *extra]


def test_build_task_route_runs_and_resumes(experiment_path, tmp_path, capsys):
    assert main(_build_task_args(experiment_path, tmp_path)) == 0
    root = _only_run(tmp_path)
    result = OptimizationResult.from_file(root / "result.json")
    assert set(result.best.values) >= {"manager", "temperature"}
    saved = json.loads((root / "cli.json").read_text())
    assert saved["build_task"] == f"{AGENTS}:build_task" and saved["space"] == f"{AGENTS}:space"
    assert saved["agent"] is None
    capsys.readouterr()
    assert main(["--json", "optimize", "resume", str(root)]) == 0  # a finished run prints its result again
    assert OptimizationResult.from_dict(json.loads(capsys.readouterr().out)).best.id == result.best.id


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--rollout", f"{AGENTS}:harness_rollout"], "--rollout doesn't go with --build-task"),
        (["--environment", f"{AGENTS}:harness_rollout"], "--environment needs --agent"),
    ],
)
def test_build_task_route_takes_no_rollout_or_environment(experiment_path, tmp_path, capsys, extra, message):
    assert main(_build_task_args(experiment_path, tmp_path, *extra)) == 2
    assert message in capsys.readouterr().err


def test_space_must_be_a_mapping(experiment_path, tmp_path, capsys):
    args = _build_task_args(experiment_path, tmp_path)
    args[args.index(f"{AGENTS}:space")] = f"{AGENTS}:EXACT"
    assert main(args) == 2
    assert "is a str, not a mapping" in capsys.readouterr().err


# the .toml run file


def _write_run_file(folder: Path, body: str) -> Path:
    path = folder / "optimize.toml"
    path.write_text(body)
    return path


RUN_FILE = f"""
experiment = "experiment.json"
agent = "{AGENTS}:manager_agent"
custom_evaluator = ["{CORRECT}"]
objective = ["maximize:Correct"]
constraint = ["tokens<=450"]
max_iterations = 2
proposals_per_iteration = 2
max_workers = 40
base_dir = "runs"
"""


def test_a_run_file_gives_the_same_inputs_as_the_flags(experiment_path, tmp_path, monkeypatch, capsys):
    assert main(_run_args(experiment_path, tmp_path / "flags")) == 0
    from_flags = json.loads((_only_run(tmp_path / "flags") / "cli.json").read_text())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # paths in the file resolve against its folder, not the working directory
    assert main(["--json", "optimize", "run", str(_write_run_file(tmp_path, RUN_FILE))]) == 0
    root = _only_run(tmp_path)
    assert json.loads((root / "cli.json").read_text()) == from_flags
    flags_run = json.loads((_only_run(tmp_path / "flags") / "run.json").read_text())
    file_run = json.loads((root / "run.json").read_text())
    for key in ("objective", "constraints", "budget", "search", "split", "proposals_per_iteration"):
        assert file_run[key] == flags_run[key]


def test_command_line_flags_override_the_run_file(experiment_path, tmp_path, capsys):
    path = _write_run_file(tmp_path, RUN_FILE)
    assert main(["--json", "optimize", "run", str(path), "--max-iterations", "1", "--name", "override"]) == 0
    root = _only_run(tmp_path)
    assert root.name.startswith("override-")
    assert json.loads((root / "run.json").read_text())["budget"]["max_iterations"] == 1


def test_an_agent_flag_replaces_the_run_files_agent_route(experiment_path, tmp_path, capsys):
    path = _write_run_file(tmp_path, RUN_FILE)
    args = [
        "--json",
        "optimize",
        "run",
        str(path),
        "--build-task",
        f"{AGENTS}:build_task",
        "--space",
        f"{AGENTS}:space",
    ]
    assert main(args) == 0
    saved = json.loads((_only_run(tmp_path) / "cli.json").read_text())
    assert saved["agent"] is None and saved["build_task"] == f"{AGENTS}:build_task"


def test_a_path_form_hook_resolves_against_the_run_files_folder(experiment_path, tmp_path, monkeypatch, capsys):
    (tmp_path / "agent.py").write_text(f"from {AGENTS} import manager_agent  # noqa: F401\n")
    body = RUN_FILE.replace(f'agent = "{AGENTS}:manager_agent"', 'agent = "agent.py:manager_agent"')
    path = _write_run_file(tmp_path, body)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main(["--json", "optimize", "run", str(path)]) == 0
    saved = json.loads((_only_run(tmp_path) / "cli.json").read_text())
    assert saved["agent"] == f"{tmp_path / 'agent.py'}:manager_agent"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (RUN_FILE + "json = true\n", "unknown key 'json'"),
        (RUN_FILE + "nope = 1\n", "unknown key 'nope'"),
        (
            RUN_FILE.replace("constraint = [", "constraints = ["),
            "unknown key 'constraints'; did you mean 'constraint'?",
        ),
        (RUN_FILE.replace("max_iterations", "max_iteration"), "did you mean 'max_iterations'?"),
        (RUN_FILE.replace("max_iterations = 2", 'max_iterations = "two"'), "max_iterations must be an integer"),
        (RUN_FILE.replace('objective = ["maximize:Correct"]', 'objective = "maximize:Correct"'), "list of strings"),
        (RUN_FILE + "pareto = 1\n", "pareto must be true or false"),
        (RUN_FILE.replace('experiment = "experiment.json"\n', ""), "has no experiment"),
        (RUN_FILE.replace(f'agent = "{AGENTS}:manager_agent"\n', ""), "give exactly one of --agent"),
        (RUN_FILE + 'config = "agent.json"\n', "give exactly one of --agent"),
        ("experiment = [", "isn't valid TOML"),
    ],
)
def test_a_bad_run_file_is_a_usage_error(experiment_path, tmp_path, capsys, body, message):
    assert main(["optimize", "run", str(_write_run_file(tmp_path, body))]) == 2
    assert message in capsys.readouterr().err


OBJECTIVE_SECTIONS = (
    RUN_FILE.replace('objective = ["maximize:Correct"]\n', "")
    + """
[objective.correct]
maximize = "Correct"
weight = 2

[objective.cost]
minimize = "tokens"
scale = 1000
"""
)

NO_CONSTRAINT = OBJECTIVE_SECTIONS.replace('constraint = ["tokens<=450"]\n', "")


def test_named_objective_tables_become_specs(experiment_path, tmp_path, capsys):
    assert main(["--json", "optimize", "run", str(_write_run_file(tmp_path, OBJECTIVE_SECTIONS))]) == 0
    root = _only_run(tmp_path)
    assert json.loads((root / "cli.json").read_text())["objective"] == [
        "maximize:Correct:weight=2",
        "minimize:tokens:scale=1000",
    ]
    assert (
        json.loads((root / "run.json").read_text())["objective"]
        == "WeightedSum(Maximize(Correct, weight=2), Minimize(tokens, scale=1000))"
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (OBJECTIVE_SECTIONS + "max_rollouts = 60\n", "every key after a [objective] or [objective.NAME] header"),
        (
            NO_CONSTRAINT + '[constraint.c]\nmetric = "tokens"\nat_most = 450\nat_least = 1\n',
            "needs one of at_most = X or at_least = X",
        ),
        (NO_CONSTRAINT + "[constraint.c]\nat_most = 450\n", 'needs metric = "METRIC"'),
        (NO_CONSTRAINT + '[constraint]\nmetric = "tokens"\nat_most = 450\n', "give each constraint a named table"),
        (OBJECTIVE_SECTIONS + '[tune.e]\npath = "effort"\nchoices = "low"\n', "choices must be a list"),
        (
            OBJECTIVE_SECTIONS.replace('maximize = "Correct"', 'maximize = "Correct"\nminimize = "tokens"'),
            "needs one of",
        ),
        (OBJECTIVE_SECTIONS.replace("weight = 2", 'weight = "two"'), "weight must be a number"),
        (OBJECTIVE_SECTIONS.replace("weight = 2", "size = 2"), "unknown key 'size'"),
        (OBJECTIVE_SECTIONS.replace("weight = 2", "weights = 2"), "unknown key 'weights'; did you mean 'weight'?"),
        (  # [[objective]] sections and inline tables: not accepted
            RUN_FILE.replace('objective = ["maximize:Correct"]\n', "") + '[[objective]]\nmaximize = "Correct"\n',
            "[[objective]] sections and inline tables aren't accepted",
        ),
        (
            RUN_FILE.replace('objective = ["maximize:Correct"]', 'objective = [{ maximize = "Correct" }]'),
            "[[objective]] sections and inline tables aren't accepted",
        ),
        (NO_CONSTRAINT + '[[constraint]]\nmetric = "tokens"\nat_most = 450\n', "[[constraint]] sections"),
        (OBJECTIVE_SECTIONS + "[constraint]\nat_most = 450\n", "a key given twice"),
    ],
)
def test_a_bad_objective_table_is_a_usage_error(experiment_path, tmp_path, capsys, body, message):
    assert main(["optimize", "run", str(_write_run_file(tmp_path, body))]) == 2
    assert message in capsys.readouterr().err


def test_a_constraint_table_becomes_a_spec(experiment_path, tmp_path, capsys):
    body = RUN_FILE.replace('constraint = ["tokens<=450"]\n', "") + '[constraint.c]\nmetric = "tokens"\nat_most = 450\n'
    assert main(["--json", "optimize", "run", str(_write_run_file(tmp_path, body))]) == 0
    root = _only_run(tmp_path)
    assert json.loads((root / "cli.json").read_text())["constraints"] == ["tokens<=450"]
    assert json.loads((root / "run.json").read_text())["constraints"] == ["tokens <= 450"]


def test_tune_tables_become_specs(experiment_path, tmp_path, monkeypatch, capsys, scripted_text_proposer):
    config = _write_config(tmp_path, effort="auto")
    monkeypatch.chdir(tmp_path)
    body = f"""
experiment = "experiment.json"
config = "{config.name}"
custom_evaluator = ["{CORRECT}"]
objective = ["maximize:Correct"]
max_iterations = 4
proposals_per_iteration = 2
max_workers = 40
base_dir = "runs"

[tune.effort]
path = "effort"
choices = ["auto", "low"]

[tune.description]
path = "description"
"""
    _write_config(tmp_path, effort="auto", description="A question answerer.")
    assert main(["--json", "optimize", "run", str(_write_run_file(tmp_path, body))]) == 0
    saved = json.loads((_only_run(tmp_path) / "cli.json").read_text())
    assert saved["tune"] == ['effort=["auto", "low"]', "description"]


SETTINGS = f"""
experiment = "experiment.json"
custom_evaluator = ["{CORRECT}"]
max_iterations = 2
proposals_per_iteration = 2
max_workers = 40
base_dir = "runs"
"""


def test_named_objective_tables_in_an_agent_section_run(experiment_path, tmp_path, capsys):
    objectives = (
        '[objective.correct]\nmaximize = "Correct"\nweight = 2\n\n[objective.cost]\nminimize = "tokens"\nscale = 1000\n'
    )
    body = SETTINGS + f'\n[agent]\nfactory = "{AGENTS}:manager_agent"\n\n' + objectives
    assert main(["--json", "optimize", "run", str(_write_run_file(tmp_path, body))]) == 0
    saved = json.loads((_only_run(tmp_path) / "cli.json").read_text())
    assert saved["agent"] == f"{AGENTS}:manager_agent"  # [agent] factory is --agent
    assert saved["objective"] == ["maximize:Correct:weight=2", "minimize:tokens:scale=1000"]


def test_an_agent_section_holds_a_config_route(experiment_path, tmp_path, monkeypatch, capsys, scripted_text_proposer):
    _write_config(tmp_path, effort="auto")
    monkeypatch.chdir(tmp_path)
    body = (
        SETTINGS
        + 'objective = ["maximize:Correct"]\n\n[agent]\nconfig = "agent.json"\n\n'
        + '[agent.tune.effort]\npath = "effort"\nchoices = ["auto", "low"]\n'
    )
    assert main(["--json", "optimize", "run", str(_write_run_file(tmp_path, body))]) == 0
    saved = json.loads((_only_run(tmp_path) / "cli.json").read_text())
    assert saved["config"] == str((tmp_path / "agent.json").resolve()) and saved["tune"] == ['effort=["auto", "low"]']


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            SETTINGS + f'[agent]\nfactory = "{AGENTS}:manager_agent"\nmax_rollouts = 60\n',
            "every key after the [agent] header belongs to it",
        ),
        (SETTINGS + f'[agent]\nfactory = "{AGENTS}:manager_agent"\nmodel = "x"\n', "[agent] has no key 'model'"),
        (SETTINGS + '[agent]\nfactori = "x:y"\n', "[agent] has no key 'factori'; did you mean 'factory'?"),
        (
            SETTINGS + f'config = "agent.json"\n[agent]\nconfig = "agent.json"\nfactory = "{AGENTS}:manager_agent"\n',
            "config is given both in [agent] and at the top level",
        ),
        (SETTINGS + f'agent = "{AGENTS}:manager_agent"\n[agent]\nconfig = "x"\n', "use one form"),
        (
            SETTINGS + '[objective]\nmaximize = "Correct"\n[objective]\nminimize = "tokens"\n',
            "a key given twice",
        ),
        (
            SETTINGS + '[objective]\nmaximize = "Correct"\n[objective.cost]\nminimize = "tokens"\n',
            "[objective] holds 'maximize' directly; give each objective a named table",
        ),
    ],
)
def test_a_bad_section_is_a_usage_error(experiment_path, tmp_path, capsys, body, message):
    assert main(["optimize", "run", str(_write_run_file(tmp_path, body))]) == 2
    assert message in capsys.readouterr().err


# PR #553 review fixes


def _killed_copy(root: Path, tmp_path: Path) -> Path:
    """A copy of finished run `root` cut after its first iteration, as a kill would leave it."""
    lines = (root / "events.jsonl").read_text().splitlines()
    cut = next(i for i, line in enumerate(lines) if json.loads(line)["event"] == "iteration_end") + 1
    killed = tmp_path / "killed" / root.name
    shutil.copytree(root, killed)
    (killed / "events.jsonl").write_text("\n".join(lines[:cut]) + "\n")
    (killed / "result.json").unlink()
    return killed


def test_a_zero_flag_overrides_the_run_file(experiment_path, tmp_path, capsys):
    path = _write_run_file(tmp_path, RUN_FILE + "seed = 7\nrerun_passes = 2\n")
    assert main(["--json", "optimize", "run", str(path), "--seed", "0", "--rerun-passes", "0"]) == 0
    run = json.loads((_only_run(tmp_path) / "run.json").read_text())
    assert run["seed"] == 0 and run["rerun_passes"] == 0


def test_resume_refuses_a_config_changed_since_the_run(
    experiment_path, tmp_path, monkeypatch, capsys, scripted_text_proposer
):
    config = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(_config_args(experiment_path, config, tmp_path)) == 0
    killed = _killed_copy(_only_run(tmp_path), tmp_path)
    capsys.readouterr()
    assert main(["--json", "optimize", "resume", str(killed)]) == 0  # unchanged: resumes
    killed = _killed_copy(_only_run(tmp_path), tmp_path / "again")
    config.write_text(config.read_text().replace('"qa"', '"renamed"'))  # a key that isn't tuned
    capsys.readouterr()
    assert main(["optimize", "resume", str(killed)]) == 2
    assert "changed since the run" in capsys.readouterr().err


class Unreachable(Proposer):
    """A text proposer whose model can't be reached."""

    kinds = frozenset({"text"})

    async def propose(self, request):
        raise RuntimeError("no model access")

    def observe(self, observation):
        pass


def test_a_search_stopped_by_a_proposer_error_exits_3(experiment_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("strands_evals.optimization.search.blocks.Reflection", Unreachable)
    config = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(_config_args(experiment_path, config, tmp_path)) == 3
    out, err = capsys.readouterr()
    assert out == ""  # the search stopped before the end phase, so there is no result to print
    assert "raised RuntimeError: no model access" in err
    assert "optimize resume" in err  # how to continue the search with the CLI


def test_debug_prints_the_traceback_of_a_bad_hook(experiment_path, tmp_path, capsys):
    args = _run_args(experiment_path, tmp_path, "--search", f"{AGENTS}:final_pick")
    assert main(["--debug", *args]) == 2
    assert "Traceback" in capsys.readouterr().err


def test_ctrl_c_says_how_to_resume(experiment_path, tmp_path, monkeypatch, capsys):
    async def interrupted(optimizer, on_folder):
        raise KeyboardInterrupt

    monkeypatch.setattr(_optimize_module(), "drive", interrupted)
    assert main(_run_args(experiment_path, tmp_path)) == 130
    assert "continue with `strands-evals optimize resume" in capsys.readouterr().err


def _optimize_module():
    from strands_evals.cli import _optimize

    return _optimize


def test_resume_overrides_a_killed_run_and_warns_on_a_finished_one(experiment_path, tmp_path, monkeypatch, capsys):
    from strands_evals.optimization import Optimizer

    assert main(_run_args(experiment_path, tmp_path)) == 0
    root = _only_run(tmp_path)
    killed = _killed_copy(root, tmp_path)
    capsys.readouterr()
    real, given = Optimizer.resume.__func__, {}

    def spy(cls, *args, **kwargs):
        given.update(kwargs)
        return real(cls, *args, **kwargs)

    monkeypatch.setattr(Optimizer, "resume", classmethod(spy))
    assert main(["--json", "optimize", "resume", str(killed), "--max-workers", "8", "--max-rollouts", "400"]) == 0
    assert "change nothing" not in capsys.readouterr().err
    events = [json.loads(line) for line in (killed / "events.jsonl").read_text().splitlines()]
    (resumed,) = [event for event in events if event["event"] == "resume"]
    assert resumed["budget"]["max_rollouts"] == 400  # the override reached the run
    assert given["max_workers"] == 8
    # a finished run only prints its result: an override that wouldn't fit the spend is ignored, not an error
    assert main(["--json", "optimize", "resume", str(root), "--max-rollouts", "1"]) == 0
    assert "the run has finished, so --max-rollouts change nothing" in capsys.readouterr().err


def test_a_finished_config_run_prints_its_result_whatever_happened_to_the_config(
    experiment_path, tmp_path, monkeypatch, capsys, scripted_text_proposer
):
    config = _write_config(tmp_path, effort="auto")
    monkeypatch.chdir(tmp_path)
    assert main(_config_args(experiment_path, config, tmp_path, "--tune", 'effort=["auto","low"]')) == 0
    root = _only_run(tmp_path)
    expected = json.loads(capsys.readouterr().out)
    config.write_text(config.read_text().replace('"auto"', '"low"'))  # a tuned value applied, as from the champion
    assert main(["--json", "optimize", "resume", str(root)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    config.unlink()
    (tmp_path / "experiment.json").unlink()  # a finished run reads neither
    assert main(["--json", "optimize", "resume", str(root)]) == 0
    capsys.readouterr()
    moved = tmp_path / "elsewhere" / root.name
    shutil.move(root, moved)
    assert main(["--json", "optimize", "resume", str(moved)]) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["run_id"] == root.name and f"run folder {moved}" in err


def test_ctrl_c_while_printing_says_the_run_finished(experiment_path, tmp_path, monkeypatch, capsys):
    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr("strands_evals.cli.commands.optimize._print_result", interrupted)
    assert main(_run_args(experiment_path, tmp_path)) == 130
    err = capsys.readouterr().err
    assert "the run has finished and its result is in" in err and "nothing was spent" not in err


@pytest.mark.parametrize(
    ("file_key", "flag"),
    [
        ('reflection_model = "some-model"\n', ["--search", f"{AGENTS}:search"]),
        ("n_finalists = 3\n", ["--final-pick", f"{AGENTS}:final_pick"]),
    ],
)
def test_a_flag_replaces_the_file_value_it_excludes(experiment_path, tmp_path, capsys, hooks, file_key, flag):
    path = _write_run_file(tmp_path, RUN_FILE + file_key)
    assert main(["--json", "optimize", "run", str(path), *flag]) == 0, capsys.readouterr().err


def test_n_finalists_replaces_the_files_final_pick(experiment_path, tmp_path, capsys, hooks):
    path = _write_run_file(tmp_path, RUN_FILE + f'final_pick = "{AGENTS}:final_pick"\n')
    assert main(["--json", "optimize", "run", str(path), "--n-finalists", "1"]) == 0, capsys.readouterr().err
    assert json.loads((_only_run(tmp_path) / "run.json").read_text())["final_pick"]["n_finalists"] == 1


def test_reflection_model_replaces_the_files_search(experiment_path, tmp_path, monkeypatch, capsys):
    models = []

    def reflection(model=None):
        models.append(model)
        return AddInstruction()

    monkeypatch.setattr(_optimize_module(), "Reflection", reflection)
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    body = (
        SETTINGS + f'search = "{AGENTS}:search"\nobjective = ["maximize:Correct"]\n\n[agent]\nconfig = "agent.json"\n'
    )
    path = _write_run_file(tmp_path, body)
    assert main(["--json", "optimize", "run", str(path), "--reflection-model", "some-model"]) == 0
    assert models == ["some-model"]


def test_ctrl_c_before_the_run_starts(experiment_path, tmp_path, monkeypatch, capsys):
    def interrupted(inputs):
        raise KeyboardInterrupt

    monkeypatch.setattr(_optimize_module(), "load_experiment", interrupted)
    assert main(_run_args(experiment_path, tmp_path)) == 130
    assert "interrupted before the run started; nothing was spent" in capsys.readouterr().err


def test_resume_a_config_run(experiment_path, tmp_path, monkeypatch, capsys):
    config = _write_config(tmp_path, effort="auto")
    monkeypatch.chdir(tmp_path)
    tune = ("--no-tune-instructions", "--tune", 'effort=["auto","low"]')  # TPE only: its state is replayed on resume
    assert main(_config_args(experiment_path, config, tmp_path, *tune)) == 0
    root = _only_run(tmp_path)
    original = OptimizationResult.from_file(root / "result.json")
    killed = _killed_copy(root, tmp_path)
    capsys.readouterr()
    assert main(["--json", "optimize", "resume", str(killed)]) == 0
    assert OptimizationResult.from_dict(json.loads(capsys.readouterr().out)).best.id == original.best.id


def test_a_folder_stopped_before_cli_json_says_nothing_was_spent(tmp_path, capsys):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "run.json").write_text("{}")
    (tmp_path / "run" / "events.jsonl").write_text('{"event": "config"}\n')
    assert main(["optimize", "resume", str(tmp_path / "run")]) == 2
    assert "nothing was spent; start a new run" in capsys.readouterr().err


def test_resume_ignores_cli_json_keys_it_does_not_know(experiment_path, tmp_path, capsys, caplog):
    assert main(_run_args(experiment_path, tmp_path)) == 0
    killed = _killed_copy(_only_run(tmp_path), tmp_path)
    saved = json.loads((killed / "cli.json").read_text())
    (killed / "cli.json").write_text(json.dumps({**saved, "from_a_newer_version": 1}))
    capsys.readouterr()
    assert main(["--json", "optimize", "resume", str(killed)]) == 0
    assert "from_a_newer_version" in caplog.text


def test_a_harness_version_change_warns(caplog):
    from strands_evals.cli import _optimize

    _optimize.check_harness_version(_optimize.Inputs(experiment_file="x", config="agent.json", harness_version="0.0.1"))
    assert "strands-harness version differs" in caplog.text


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--constraint", "tokens"], "--constraint must be METRIC<=X or METRIC>=X"),
        (["--no-tune-instructions"], "--tune-instructions / --no-tune-instructions go with --config"),
    ],
)
def test_more_usage_errors(experiment_path, tmp_path, capsys, extra, message):
    assert main([*_run_args(experiment_path, tmp_path), *extra]) == 2
    assert message in capsys.readouterr().err


def test_agent_key_in_the_agent_section_points_to_factory(experiment_path, tmp_path, capsys):
    body = SETTINGS + 'objective = ["maximize:Correct"]\n\n[agent]\nagent = "x:y"\n'
    assert main(["optimize", "run", str(_write_run_file(tmp_path, body))]) == 2
    assert 'the factory is factory = "...", not agent = "..."' in capsys.readouterr().err


def test_no_tune_instructions_tunes_only_the_tune_paths(
    experiment_path, tmp_path, monkeypatch, capsys, scripted_text_proposer
):
    config = _write_config(tmp_path, effort="auto")
    monkeypatch.chdir(tmp_path)
    args = _config_args(experiment_path, config, tmp_path, "--no-tune-instructions", "--tune", 'effort=["auto","low"]')
    assert main(args) == 0
    root = _only_run(tmp_path)
    assert json.loads((root / "cli.json").read_text())["tune_instructions"] is False
    assert [entry["key"] for entry in json.loads((root / "run.json").read_text())["space"]] == ["effort"]
    capsys.readouterr()
    assert main(_config_args(experiment_path, config, tmp_path / "none", "--no-tune-instructions")) == 2
    assert "nothing is left to tune" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("section", "flags", "expected"),
    [
        (  # --agent over a config route: the config's tune doesn't fit, so it's dropped (no "--tune needs --config")
            '[agent]\nconfig = "agent.json"\ntune = [\'effort=["auto","low"]\']\n',
            ["--agent", f"{AGENTS}:manager_agent"],
            {"agent": f"{AGENTS}:manager_agent", "config": None, "tune": []},
        ),
        (  # --agent over another factory: the file's rollout and environment fit, so they're kept
            f'[agent]\nfactory = "{AGENTS}:plain_agent"\nrollout = "{AGENTS}:harness_rollout"\n'
            f'environment = "{AGENTS}:no_environment"\n',
            ["--agent", f"{AGENTS}:manager_agent"],
            {
                "agent": f"{AGENTS}:manager_agent",
                "rollout": f"{AGENTS}:harness_rollout",
                "environment": f"{AGENTS}:no_environment",
            },
        ),
        (  # --build-task + --space over a factory: its rollout doesn't fit, so it's dropped
            f'[agent]\nfactory = "{AGENTS}:manager_agent"\nrollout = "{AGENTS}:harness_rollout"\n',
            ["--build-task", f"{AGENTS}:build_task", "--space", f"{AGENTS}:space"],
            {"agent": None, "build_task": f"{AGENTS}:build_task", "space": f"{AGENTS}:space", "rollout": None},
        ),
        (  # --build-task alone: the file's space is the other half of the pair, so it's kept
            f'[agent]\nbuild_task = "{AGENTS}:plain_agent"\nspace = "{AGENTS}:space"\n',
            ["--build-task", f"{AGENTS}:build_task"],
            {"build_task": f"{AGENTS}:build_task", "space": f"{AGENTS}:space"},
        ),
        (  # --config over a factory: its environment doesn't fit, so it's dropped; its rollout fits, so it's kept
            f'[agent]\nfactory = "{AGENTS}:manager_agent"\nrollout = "{AGENTS}:harness_rollout"\n'
            f'environment = "{AGENTS}:no_environment"\n',
            ["--config", "agent.json"],
            {"agent": None, "environment": None, "rollout": f"{AGENTS}:harness_rollout"},
        ),
        (  # --config over a config: the file's tune and tune_instructions fit, so they're kept
            '[agent]\nconfig = "other.json"\ntune_instructions = false\ntune = [\'effort=["auto","low"]\']\n',
            ["--config", "agent.json"],
            {"tune": ['effort=["auto","low"]'], "tune_instructions": False},
        ),
    ],
    ids=[
        "agent-drops-tune",
        "agent-keeps-rollout-environment",
        "build-task-drops-rollout",
        "build-task-keeps-space",
        "config-drops-environment-keeps-rollout",
        "config-keeps-tune",
    ],
)
def test_a_route_flag_drops_only_the_file_keys_that_conflict(
    experiment_path, tmp_path, monkeypatch, capsys, scripted_text_proposer, section, flags, expected
):
    _write_config(tmp_path, effort="auto")
    monkeypatch.chdir(tmp_path)
    body = SETTINGS + 'objective = ["maximize:Correct"]\n\n' + section
    assert main(["--json", "optimize", "run", str(_write_run_file(tmp_path, body)), *flags]) == 0, (
        capsys.readouterr().err
    )
    saved = json.loads((_only_run(tmp_path) / "cli.json").read_text())
    assert {key: saved[key] for key in expected} == expected


@pytest.mark.parametrize(
    ("builtin_tools", "warned"),
    [
        (["read"], []),
        (["shell", "read"], ["shell"]),
        ({"web_search": False}, ["shell", "write", "edit"]),  # a map edits the defaults: the host tools stay on
        ({"shell": False, "write": False, "edit": False}, []),
        ({"*": False, "read": True}, []),
        ({"*": False, "edit": True}, ["edit"]),
    ],
)
def test_the_host_tools_warning_reads_builtin_tools_as_the_harness_does(tmp_path, caplog, builtin_tools, warned):
    from strands_evals.cli import _harness

    caplog.clear()
    _harness.warn_host_tools(_write_config(tmp_path, builtinTools=builtin_tools))
    assert (f"tools=<{', '.join(warned)}>" in caplog.text) if warned else ("tools=<" not in caplog.text)


def _other_experiment(tmp_path: Path) -> Path:
    cases = [Case(name=f"other{i:02d}", input=f"question {i}", expected_output=f"answer {i}") for i in range(20)]
    path = tmp_path / "other" / "experiment.json"
    path.parent.mkdir()
    Experiment(cases=cases, evaluators=[Correct()]).to_file(str(path))
    return path


def test_experiment_flag_overrides_the_run_files(experiment_path, tmp_path, capsys):
    other = _other_experiment(tmp_path)
    assert (
        main(["--json", "optimize", "run", str(_write_run_file(tmp_path, RUN_FILE)), "--experiment", str(other)]) == 0
    )
    root = _only_run(tmp_path)
    assert json.loads((root / "cli.json").read_text())["experiment_file"] == str(other.resolve())
    assert json.loads((root / "run.json").read_text())["cases"][0].startswith("other")


def test_experiment_flag_stands_in_for_a_missing_experiment_key(experiment_path, tmp_path, capsys):
    body = RUN_FILE.replace('experiment = "experiment.json"\n', "")
    path = _write_run_file(tmp_path, body)
    assert main(["--json", "optimize", "run", str(path), "--experiment", str(experiment_path)]) == 0


def test_experiment_flag_needs_a_run_file(experiment_path, tmp_path, capsys):
    args = [*_run_args(experiment_path, tmp_path), "--experiment", str(experiment_path)]
    assert main(args) == 2
    assert "--experiment goes with a .toml run file" in capsys.readouterr().err


def test_an_uppercase_toml_suffix_is_a_run_file(experiment_path, tmp_path, capsys):
    path = tmp_path / "OPTIMIZE.TOML"
    path.write_text(RUN_FILE)
    assert main(["--json", "optimize", "run", str(path)]) == 0


DEPTH = "builtinTools.subagent.maxDepth"
TIMEOUT = "builtinTools.programmatic_tool_caller.timeout"


def _depth_config(folder: Path) -> Path:
    """A config whose only built-in tool is the subagent one, with a numeric setting to tune."""
    return _write_config(folder, builtinTools={"*": False, "subagent": {"maxDepth": 1}})


@pytest.mark.parametrize("form", ["flag", "table"])
def test_an_int_range_tunes_a_config_number(experiment_path, tmp_path, monkeypatch, capsys, form):
    config = _depth_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    if form == "flag":
        args = _config_args(experiment_path, config, tmp_path, "--no-tune-instructions", "--tune", f"{DEPTH}=int:0..3")
    else:
        body = (
            SETTINGS
            + 'objective = ["maximize:Correct"]\n\n[agent]\nconfig = "agent.json"\ntune_instructions = false\n\n'
            + f'[agent.tune.depth]\npath = "{DEPTH}"\nint = [0, 3]\n'
        )
        args = ["--json", "optimize", "run", str(_write_run_file(tmp_path, body))]
    assert main(args) == 0, capsys.readouterr().err
    root = _only_run(tmp_path)
    assert json.loads((root / "cli.json").read_text())["tune"] == [f"{DEPTH}=int:0..3"]
    (entry,) = json.loads((root / "run.json").read_text())["space"]
    assert (entry["kind"], entry["min_value"], entry["max_value"], entry["default"]) == ("int", 0, 3, 1)


def _timeout_config(folder: Path) -> Path:
    """A config whose only built-in tool is the programmatic tool caller, with a float setting to tune."""
    return _write_config(folder, builtinTools={"*": False, "programmatic_tool_caller": {"timeout": 30}})


def test_a_float_range_declares_a_float_around_the_config_value(tmp_path):
    from strands_evals.cli import _harness
    from strands_evals.optimization import Float

    markers, tuned = _harness.declare(_timeout_config(tmp_path), [f"{TIMEOUT}=float:1..60"], tune_instructions=False)
    marker = markers[TIMEOUT.replace(".", "_")]
    assert isinstance(marker, Float) and (marker.min_value, marker.max_value, marker.default) == (1.0, 60.0, 30.0)
    assert tuned[0].kind == "number" and tuned[0].path == ("builtinTools", "programmatic_tool_caller", "timeout")


@pytest.mark.parametrize(
    ("path", "kind", "low", "high", "write", "tune"),
    [
        (DEPTH, int, 0, 3, _depth_config, ("--tune", f"{DEPTH}=int:0..3")),
        (TIMEOUT, float, 1.0, 60.0, _timeout_config, f'[agent.tune.timeout]\npath = "{TIMEOUT}"\nfloat = [1, 60]\n'),
    ],
    ids=["int-flag", "float-table"],
)
def test_a_ranged_number_reaches_each_trials_config(
    experiment_path, tmp_path, monkeypatch, capsys, path, kind, low, high, write, tune
):
    from strands_evals.cli import _harness

    seen = []
    real = _harness.harness_agent_kwargs_from_config

    def spy(config, **kwargs):
        seen.append(_harness._get(config, _harness._path(path), path))
        return real(config, **kwargs)

    monkeypatch.setattr(_harness, "harness_agent_kwargs_from_config", spy)
    config = write(tmp_path)
    monkeypatch.chdir(tmp_path)
    if isinstance(tune, tuple):
        args = _config_args(experiment_path, config, tmp_path, "--no-tune-instructions", *tune)
    else:
        body = SETTINGS + 'objective = ["maximize:Correct"]\n\n[agent]\nconfig = "agent.json"\n'
        args = [
            "--json",
            "optimize",
            "run",
            str(_write_run_file(tmp_path, body + "tune_instructions = false\n\n" + tune)),
        ]
    assert main(args) == 0, capsys.readouterr().err
    assert seen and all(type(value) is kind and low <= value <= high for value in seen)
    assert len(set(seen)) > 1  # proposals, not only the baseline, reached the trials


@pytest.mark.parametrize(
    ("tune", "config", "message"),
    [
        (
            f"{DEPTH}=int:0..3",
            {"builtinTools": {"*": False, "subagent": {"maxDepth": 5}}},
            "must be an integer in 0..3",
        ),
        ("contextManager.ratio=int:0..3", {"contextManager": {"ratio": 0.5}}, "must be an integer"),
        ("instructions=int:0..3", {}, "must be an integer"),
        (f"{DEPTH}=int:3..1", {}, "LO < HI"),
        (f"{DEPTH}=int:a..b", {}, "two ints"),
        (f"{DEPTH}=float:0.1", {}, "two finite floats"),
        (f"{DEPTH}=float:0...95", {}, "two finite floats"),
        (f"{DEPTH}=float:nan..1", {}, "two finite floats"),
        (f"{DEPTH}=float:-inf..inf", {}, "two finite floats"),
        (f"{DEPTH}=int:3..3", {}, "LO < HI"),
        (f"{DEPTH}=int:0..3", {}, "write the field in the config"),  # builtinTools is a list: no subagent object
        (
            f"{DEPTH}=float:0..3",
            {"builtinTools": {"*": False, "subagent": {"maxDepth": 1}}},
            "the harness doesn't take 1.5 there",  # an integer field: no trial could take a fraction
        ),
        (
            f"{TIMEOUT}=float:0..60",
            {"builtinTools": {"*": False, "programmatic_tool_caller": {"timeout": 30}}},
            "the harness doesn't take 0.0 there",  # timeout must be positive
        ),
        ('effort=["auto","bogus"]', {"effort": "auto"}, "the harness doesn't take 'bogus' there"),
        ("contextManager.ratio=float:0.1..0.4", {"contextManager": {"ratio": 0.5}}, "must be a number in 0.1..0.4"),
    ],
)
def test_a_bad_range_is_a_usage_error(experiment_path, tmp_path, capsys, tune, config, message):
    path = _write_config(tmp_path, **config)
    assert main(_config_args(experiment_path, path, tmp_path, "--tune", tune)) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ("int = [0, 3]\nchoices = [0, 1]\n", "give one of choices, int, float"),
        ("int = [0]\n", "int must be [LO, HI], two numbers"),
        ('float = ["a", "b"]\n', "float must be [LO, HI], two numbers"),
        ("int = [0.5, 3]\n", "int = [0.5, 3] needs two integers"),
        ("float = [1.0, 1.0]\n", "needs two finite numbers with LO < HI"),
    ],
)
def test_a_bad_range_table_is_a_usage_error(experiment_path, tmp_path, capsys, fields, message):
    body = SETTINGS + 'objective = ["maximize:Correct"]\n\n[agent]\nconfig = "agent.json"\n\n'
    body += f'[agent.tune.depth]\npath = "{DEPTH}"\n' + fields
    _depth_config(tmp_path)
    assert main(["optimize", "run", str(_write_run_file(tmp_path, body))]) == 2
    assert message in capsys.readouterr().err


def test_a_list_choice_declares_a_json_choice_on_the_config(tmp_path):
    from strands_evals.cli import _harness
    from strands_evals.optimization import JsonChoice, Space

    config = _write_config(tmp_path)
    markers, tuned = _harness.declare(config, ['builtinTools=[[],["read"]]'], tune_instructions=False)
    marker = markers["builtinTools"]
    assert isinstance(marker, JsonChoice) and marker.options == ("[]", '["read"]') and marker.default == "[]"
    assert tuned[0].kind == "choice" and tuned[0].path == ("builtinTools",)
    # `build_task` reads the decoded value off the space, so each trial's config gets the real list.
    assert Space(markers).resolve("builtinTools", {"builtinTools": '["read"]'}, None) == ["read"]


def test_a_list_choice_missing_the_configs_value_is_rejected(tmp_path):
    from strands_evals.cli import _harness

    config = _write_config(tmp_path, builtinTools=["read"])
    with pytest.raises(ValueError, match="the config's value .* must be one of the options"):
        _harness.declare(config, ['builtinTools=[[],["write"]]'], tune_instructions=False)
