"""`strands-evals optimize --agent agent.ts` against `--agent agent.py`: the same harness agent, in Node and in Python.

`typescript_agent/` holds the agent twice (`agent.ts`, `agent.py`) on one deterministic scripted model (`world.ts`,
`world_py.py`), so no model is called. The fixture is copied and its npm packages installed (the public registry,
no install scripts); skipped without Node.js 22.18+ and npm.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from strands_evals import Case, Experiment
from strands_evals.cli.main import main
from strands_evals.optimization import OptimizationResult
from strands_evals.types.trace import Session
from tests.strands_evals.cli.fixtures.optimize_agents import Correct

FIXTURE = Path(__file__).parent / "typescript_agent"
INSTRUCTIONS = ["You answer questions.", "You answer questions. Give the exact answer."]


def _node_ok() -> bool:
    node, npm = shutil.which("node"), shutil.which("npm")
    if node is None or npm is None:
        return False
    try:
        done = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=30)
        major, minor, *_ = done.stdout.strip().lstrip("v").split(".")
        return done.returncode == 0 and (int(major), int(minor)) >= (22, 18)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False  # a broken shim or a version manager's stub: skip rather than error at collection


pytestmark = pytest.mark.skipif(not _node_ok(), reason="needs Node.js 22.18+ and npm")


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("project") / "agent"
    shutil.copytree(FIXTURE, folder)
    subprocess.run(
        ["npm", "install", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=folder, check=True, capture_output=True
    )
    return folder


@pytest.fixture
def experiment_path(tmp_path: Path) -> Path:
    cases = [Case(name=f"q{i:02d}", input=f"question {i}", expected_output=f"answer {i}") for i in range(20)]
    path = tmp_path / "experiment.json"
    Experiment(cases=cases, evaluators=[Correct()]).to_file(str(path))
    return path


def _optimize(experiment_path: Path, agent: Path, base_dir: Path) -> OptimizationResult:
    args = [
        "--json",
        "optimize",
        "run",
        str(experiment_path),
        "--agent",
        str(agent),
        "--custom-evaluator",
        "tests.strands_evals.cli.fixtures.optimize_agents:Correct",
        "--objective",
        "maximize:Correct",
        "--no-tune-instructions",
        "--tune",
        f"instructions={json.dumps(INSTRUCTIONS)}",
        "--tune",
        'effort=["auto","low"]',
        "--max-iterations",
        "3",
        "--proposals-per-iteration",
        "2",
        "--max-workers",
        "8",
        "--base-dir",
        str(base_dir),
    ]
    assert main(args) == 0
    (root,) = base_dir.iterdir()
    return OptimizationResult.from_file(root / "result.json")


def _comparable(result: OptimizationResult) -> dict:
    """What two runs of the same agent and seed must agree on (not timings or run ids)."""
    validation = result.validation.model_dump()
    for metrics in validation["measured"].values():
        metrics.pop("seconds", None)
    return {
        "candidates": [
            (c.number, c.id, c.parent_id, c.values) for c in sorted(result.candidates.values(), key=lambda c: c.number)
        ],
        "best": result.best.id,
        "rollouts": result.rollouts_spent,
        "stop": result.stop_reason,
        "usage": {
            s: (u.tokens, u.input_tokens, u.model_calls, u.models)
            for s, u in ((s, a.usage) for s, a in result.cost.agent.items())
        },
        "benchmark": sum(u.tokens for u in result.cost.benchmark.values()),
        "validation": validation,
        "rows": [
            {key: value for key, value in row.items() if "seconds" not in key and "run_id" not in key}
            for row in result.to_rows("proposal")
        ],
    }


def _invocations(result: OptimizationResult) -> list[tuple[str, str, str]]:
    """Each best-candidate train trial's (case, prompt, answer), from its mapped trajectory."""
    found = []
    for case in result.reports[result.best.id]["train"].cases:
        session = result.load_transcript(case["transcript"])
        assert isinstance(session, Session)
        spans = session.traces[0].spans
        invocation = next(span for span in spans if type(span).__name__ == "AgentInvocationSpan")
        found.append((case["name"], invocation.user_prompt, invocation.agent_response.strip()))
    return sorted(found)


def test_a_typescript_agent_optimizes_as_its_python_twin(project, experiment_path, tmp_path, capsys):
    from_ts = _optimize(experiment_path, project / "agent.ts", tmp_path / "ts")
    from_py = _optimize(experiment_path, project / "agent.py", tmp_path / "py")
    capsys.readouterr()

    assert _comparable(from_ts) == _comparable(from_py)
    assert from_ts.cost.agent["train"].usage.tokens > 0  # a TypeScript trial never reads as free
    assert "Give the exact answer." in from_ts.best.values["instructions"]
    assert _invocations(from_ts) == _invocations(from_py)  # the trajectory evaluators and Reflection read
