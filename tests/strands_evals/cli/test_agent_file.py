"""Tests for `strands-evals optimize --agent agent.py`: an agent file whose agent `create_harness` builds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_evals import Case, Experiment
from strands_evals.cli import _agent_file
from strands_evals.cli.main import main
from strands_evals.optimization import OptimizationResult
from tests.strands_evals.cli.fixtures.optimize_agents import BUILT, EXACT, INSTRUCTIONS, Correct
from tests.strands_evals.cli.test_optimize import AddInstruction

FIXTURES = Path(__file__).parent / "fixtures"
AGENT_FILE = FIXTURES / "harness_agent.py"
AGENTS = "tests.strands_evals.cli.fixtures.optimize_agents"


@pytest.fixture
def experiment_path(tmp_path: Path) -> Path:
    cases = [Case(name=f"q{i:02d}", input=f"question {i}", expected_output=f"answer {i}") for i in range(20)]
    path = tmp_path / "experiment.json"
    Experiment(cases=cases, evaluators=[Correct()]).to_file(str(path))
    return path


def _args(experiment_path: Path, base_dir: Path, agent: str, *extra: str) -> list[str]:
    return [
        "--json",
        "optimize",
        "run",
        str(experiment_path),
        "--agent",
        agent,
        "--custom-evaluator",
        f"{AGENTS}:Correct",
        "--objective",
        "maximize:Correct",
        "--max-iterations",
        "3",
        "--proposals-per-iteration",
        "2",
        "--max-workers",
        "40",
        "--base-dir",
        str(base_dir),
        *extra,
    ]


def _run(base_dir: Path) -> OptimizationResult:
    (root,) = base_dir.iterdir()
    return OptimizationResult.from_file(root / "result.json")


def _comparable(result: OptimizationResult) -> dict:
    """What two runs of the same agent and seed must agree on (not timings, run ids, or the agent's name)."""
    candidates = sorted(result.candidates.values(), key=lambda c: c.number)
    volatile = ("seconds", "run_id", "time")
    validation = result.validation.model_dump()
    for metrics in validation["measured"].values():
        metrics.pop("seconds", None)
    return {
        "candidates": [(c.number, c.id, c.parent_id, c.values) for c in candidates],
        "best": result.best.id,
        "rollouts": result.rollouts_spent,
        "stop": result.stop_reason,
        "tokens": {split: (u.tokens, u.model_calls) for split, u in result.agent_cost.items()},
        "validation": validation,
        "rows": [
            {key: value for key, value in row.items() if not any(word in key for word in volatile)}
            for row in result.to_rows("proposal")
        ],
    }


def test_an_agent_file_runs_as_its_tunable_factory_does(experiment_path, tmp_path, capsys):
    tune = [
        "--no-tune-instructions",
        "--tune",
        f"instructions={json.dumps(INSTRUCTIONS)}",
        "--tune",
        'effort=["auto","low"]',
    ]
    assert main(_args(experiment_path, tmp_path / "file", str(AGENT_FILE), *tune)) == 0
    assert main(_args(experiment_path, tmp_path / "module", f"{AGENTS}:harness_factory")) == 0
    capsys.readouterr()

    from_file, from_module = _run(tmp_path / "file"), _run(tmp_path / "module")
    assert _comparable(from_file) == _comparable(from_module)
    assert len(from_file.candidates) > 2 and EXACT in from_file.best.values["instructions"]


def test_the_default_space_is_the_instructions_as_text(experiment_path, tmp_path, monkeypatch, capsys):
    factory = _agent_file.load(str(AGENT_FILE), ['effort=["auto","low"]'])
    instructions, effort = factory.space.param("instructions"), factory.space.param("effort")
    assert (instructions.kind, instructions.default) == ("text", "You answer questions.")
    assert (effort.kind, effort.options, effort.default) == ("one_of", ("auto", "low"), "auto")

    monkeypatch.setattr("strands_evals.optimization.search.blocks.Reflection", AddInstruction)  # no model calls
    runs = tmp_path / "runs"
    assert main(_args(experiment_path, runs, str(AGENT_FILE))) == 0
    capsys.readouterr()
    result = _run(runs)
    assert result.improved and EXACT in result.best.values["instructions"]
    assert "--- instructions" in result.best.diff()  # the text lives in code: a hunk headed by its key
    saved = json.loads(next(runs.iterdir()).joinpath("cli.json").read_text())
    assert saved["agent"] == str(AGENT_FILE) and saved["tune"] == [] and saved["tune_instructions"] is True
    assert saved["harness_version"]


def test_each_build_runs_the_file_again_with_the_values_and_fresh_state():
    factory = _agent_file.load(f"{AGENT_FILE}:agent", ['effort=["auto","low"]'])
    BUILT.clear()
    first = factory.bind({"instructions": "Be brief.", "effort": "low"})()
    second = factory.bind({"instructions": "Be brief.", "effort": "low"})()
    assert BUILT == ["file", "file"]  # the file ran once per build
    assert first is not second and first.model is not second.model  # fresh objects, the model included
    assert "Be brief." in first.system_prompt
    assert "You answer questions." not in first.system_prompt
    baseline = factory()
    assert "You answer questions." in baseline.system_prompt


def test_each_create_harness_call_gets_its_own_session_and_memory(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "notes.md").write_text("remembered")
    folder = tmp_path / "trial"
    folder.mkdir()
    kwargs = _agent_file._fresh_state({"memory": {"dir": str(tmp_path / "memory")}, "session": True}, folder)
    assert kwargs["session"] == {"dir": str(folder / "session")}
    assert kwargs["memory"] == {"dir": str(folder / "memory")}
    assert (folder / "memory" / "notes.md").read_text() == "remembered"
    assert _agent_file._fresh_state({"session": False, "memory": False}, folder) == {"session": False, "memory": False}


@pytest.mark.parametrize(
    ("body", "spec_attr", "tune", "message"),
    [
        ("agent = 3\n", "agent", [], "neither a tunable factory"),
        ("x = 1\n", "agent", [], "has no attribute 'agent'"),
        ("raise RuntimeError('boom')\n", "agent", [], "boom"),
        (
            "from tests.strands_evals.cli.fixtures.optimize_agents import manager_agent\n",
            "manager_agent",
            ["temperature"],
            "is a tunable factory",
        ),
    ],
)
def test_files_that_cant_be_tuned_are_rejected(tmp_path, body, spec_attr, tune, message):
    path = tmp_path / "agent.py"
    path.write_text(body)
    with pytest.raises((ValueError, Exception), match=message):
        _agent_file.load(f"{path}:{spec_attr}", tune)


@pytest.mark.parametrize(
    ("tune", "message"),
    [
        (['effort=["low","high"]'], "must be one of the options"),
        (["builtin_tools.subagent"], "nothing at that path"),
        (["model"], "not text"),
        (["nope"], "no argument 'nope'"),
    ],
)
def test_bad_tune_paths_are_rejected(tune, message):
    with pytest.raises(ValueError, match=message):
        _agent_file.load(str(AGENT_FILE), tune)


def test_a_path_to_a_tunable_factory_still_loads_as_one(experiment_path, tmp_path, capsys):
    spec = f"{FIXTURES / 'optimize_agents.py'}:manager_agent"
    args = _args(experiment_path, tmp_path / "runs", spec, "--max-iterations", "1")
    assert main(args) == 0
    capsys.readouterr()
    assert set(_run(tmp_path / "runs").best.values) >= {"manager", "temperature"}
