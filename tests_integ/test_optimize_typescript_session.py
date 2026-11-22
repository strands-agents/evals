"""The session protocol end to end: a TypeScript agent held open across turns, and an optimize run over it.

`typescript_session/` holds an agent (`agent.ts`) on a scripted model (`world.ts`) that answers only from the
second user turn of one conversation, and a two-turn rollout (`rollout.py`) that opens a session per case. A
fresh agent per call never answers, so every right answer proves the worker held one conversation open. The
fixture is copied and its npm packages installed (the public registry, no install scripts); skipped without
Node.js 22.18+ and npm.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from strands_evals import Case, Experiment
from strands_evals.cli.main import main
from strands_evals.optimization import OptimizationResult
from strands_evals.optimization.execution.node import NodeAgent, NodeRunner
from tests.strands_evals.cli.fixtures.optimize_agents import Correct

FIXTURE = Path(__file__).parent / "typescript_session"
INSTRUCTIONS = ["You answer questions.", "You answer questions. Give the exact answer."]


def _node_ok() -> bool:
    node, npm = shutil.which("node"), shutil.which("npm")
    if node is None or npm is None:
        return False
    major, minor, *_ = subprocess.run([node, "--version"], capture_output=True, text=True).stdout[1:].split(".")
    return (int(major), int(minor)) >= (22, 18)


pytestmark = pytest.mark.skipif(not _node_ok(), reason="needs Node.js 22.18+ and npm")


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("project") / "agent"
    shutil.copytree(FIXTURE, folder, ignore=shutil.ignore_patterns("node_modules", "__pycache__"))
    subprocess.run(
        ["npm", "install", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=folder, check=True, capture_output=True
    )
    return folder


@pytest.fixture
def experiment_path(tmp_path: Path) -> Path:
    cases = [Case(name=f"q{i:02d}", input=f"question {i}", expected_output=f"answer {i}") for i in range(1, 13)]
    path = tmp_path / "experiment.json"
    Experiment(cases=cases, evaluators=[Correct()]).to_file(str(path))
    return path


def test_a_session_holds_one_conversation_open(project):
    """Over the real runner: `invoke_async` never reaches a second turn, a session's second turn does."""
    runner = NodeRunner(project / "agent.ts")
    try:
        described = runner.call({"type": "describe", "attr": "agent", "paths": [["instructions"]]})
        # The exact instruction makes "question 7" answerable, so the start's values reach the one build too.
        agent = NodeAgent(runner, "agent", int(described["index"]), [(["instructions"], INSTRUCTIONS[1])])

        async def conversation() -> tuple[str, str, str]:
            single = await agent.invoke_async("question 7")
            async with agent.session() as session:
                first = await session.turn("here it comes: question 7")
                second = await session.turn("question 7")
                assert session.last_result["stop_reason"] == "endTurn"
                assert "answer 7" in json.dumps(session.last_result["last_message"])
            return single, first, second

        single, first, second = asyncio.run(conversation())
        assert single == "I'm not sure."  # a fresh agent per call is always on its first turn
        assert first == "I'm not sure."  # and so is a session's opening turn
        assert second == "answer 7"  # the second turn saw the first: the same agent answered both
    finally:
        runner.close()


def test_optimize_runs_a_two_turn_rollout_over_sessions(project, experiment_path, tmp_path, capsys):
    base_dir = tmp_path / "runs"
    args = [
        "--json",
        "optimize",
        "run",
        str(experiment_path),
        "--agent",
        str(project / "agent.ts"),
        "--rollout",
        f"{project / 'rollout.py'}:two_turns",
        "--custom-evaluator",
        "tests.strands_evals.cli.fixtures.optimize_agents:Correct",
        "--objective",
        "maximize:Correct",
        "--no-tune-instructions",
        "--tune",
        f"instructions={json.dumps(INSTRUCTIONS)}",
        "--max-iterations",
        "2",
        "--proposals-per-iteration",
        "2",
        "--max-workers",
        "8",
        "--base-dir",
        str(base_dir),
    ]
    assert main(args) == 0
    capsys.readouterr()
    (root,) = base_dir.iterdir()
    result = OptimizationResult.from_file(root / "result.json")

    # The exact instruction answers 11 of 12 from the second turn; the plain one 4 of 12. Only a search whose
    # every trial really spoke twice to one held agent can see that difference.
    assert "Give the exact answer." in result.best.values["instructions"]
    assert result.agent_cost["train"].tokens > 0  # the turns' spans landed in the trials
    # The exact instruction answers all but every 10th question; the plain one every 3rd. One conversation
    # never held open would answer nothing at all (`rollout.py` raises when an opening turn answers).
    best_train = result.reports[result.best.id]["train"]
    assert best_train.overall_score is not None and best_train.overall_score >= 0.8
