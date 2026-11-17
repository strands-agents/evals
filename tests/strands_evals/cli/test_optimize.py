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
            NO_CONSTRAINT + '[constraint]\nmetric = "tokens"\nat_most = 450\nat_least = 1\n',
            "needs one of at_most = X or at_least = X",
        ),
        (NO_CONSTRAINT + "[constraint]\nat_most = 450\n", 'needs metric = "METRIC"'),
        (OBJECTIVE_SECTIONS + '[tune]\npath = "effort"\nchoices = "low"\n', "choices must be a list"),
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
    body = RUN_FILE.replace('constraint = ["tokens<=450"]\n', "") + '[constraint]\nmetric = "tokens"\nat_most = 450\n'
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


@pytest.mark.parametrize(
    "objectives",
    [
        '[objective]\nmaximize = "Correct"\n',  # a lone objective takes no weight
        (
            '[objective.correct]\nmaximize = "Correct"\nweight = 2\n\n'
            '[objective.cost]\nminimize = "tokens"\nscale = 1000\n'
        ),
    ],
    ids=["single", "named"],
)
def test_single_and_named_objective_tables(experiment_path, tmp_path, capsys, objectives):
    body = SETTINGS + f'\n[agent]\nfactory = "{AGENTS}:manager_agent"\n\n' + objectives
    assert main(["--json", "optimize", "run", str(_write_run_file(tmp_path, body))]) == 0
    saved = json.loads((_only_run(tmp_path) / "cli.json").read_text())
    assert saved["agent"] == f"{AGENTS}:manager_agent"  # [agent] factory is --agent
    named = "cost" in objectives
    expected = ["maximize:Correct:weight=2", "minimize:tokens:scale=1000"] if named else ["maximize:Correct"]
    assert saved["objective"] == expected


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
            "mixes fields with named [objective.NAME] tables",
        ),
    ],
)
def test_a_bad_section_is_a_usage_error(experiment_path, tmp_path, capsys, body, message):
    assert main(["optimize", "run", str(_write_run_file(tmp_path, body))]) == 2
    assert message in capsys.readouterr().err
