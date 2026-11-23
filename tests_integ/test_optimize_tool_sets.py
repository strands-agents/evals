"""`strands-evals optimize --tune 'builtinTools=[[],["read"]]'`: tool sets as a searched choice.

The agent is `typescript_agent/agent_tools.ts` on the deterministic scripted model (`world.ts`), so no
model service is called and the `read` tool is never invoked: what is tested is that a list-valued
`--tune` declares a `JsonChoice`, the search draws both tool sets, each trial's `createHarness` call
gets the decoded list, and the run's records carry the canonical JSON text. The fixture is copied and
its npm packages installed (the public registry, no install scripts); skipped without Node.js 22.18+
and npm.
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
from tests.strands_evals.cli.fixtures.optimize_agents import Correct

FIXTURE = Path(__file__).parent / "typescript_agent"


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
    shutil.copytree(FIXTURE, folder)
    subprocess.run(
        ["npm", "install", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=folder, check=True, capture_output=True
    )
    return folder


def test_tool_sets_are_searched_and_each_trial_gets_the_decoded_list(project, tmp_path, capsys):
    cases = [Case(name=f"q{i:02d}", input=f"question {i}", expected_output=f"answer {i}") for i in range(8)]
    experiment = tmp_path / "experiment.json"
    Experiment(cases=cases, evaluators=[Correct()]).to_file(str(experiment))
    base_dir = tmp_path / "runs"
    args = [
        "--json",
        "optimize",
        "run",
        str(experiment),
        "--agent",
        str(project / "agent_tools.ts"),
        "--custom-evaluator",
        "tests.strands_evals.cli.fixtures.optimize_agents:Correct",
        "--objective",
        "maximize:Correct",
        "--no-tune-instructions",
        "--tune",
        'builtinTools=[[],["read"]]',
        "--max-iterations",
        "2",
        "--proposals-per-iteration",
        "2",
        "--max-workers",
        "8",
        "--seed",
        "0",
        "--base-dir",
        str(base_dir),
    ]
    assert main(args) == 0, capsys.readouterr().err
    (root,) = base_dir.iterdir()
    result = OptimizationResult.from_file(root / "result.json")

    # The space read off the createHarness call: one choice key over the canonical JSON of the tool sets.
    (entry,) = json.loads((root / "run.json").read_text())["space"]
    assert entry["key"] == "builtinTools" and entry["kind"] == "one_of"
    assert (entry["default"], tuple(entry["options"])) == ("[]", ("[]", '["read"]'))

    # The search drew both tool sets, each trial built its agent (no candidate died on the value), and
    # every recorded value is the canonical JSON of a list.
    drawn = {candidate.values["builtinTools"] for candidate in result.candidates.values()}
    assert drawn == {"[]", '["read"]'}
    assert all(
        isinstance(json.loads(candidate.values["builtinTools"]), list) for candidate in result.candidates.values()
    )
    moved = [candidate for candidate in result.candidates.values() if candidate.changed]
    assert moved, "at least one candidate moved the tool set off the baseline"
    assert all(candidate.changed["builtinTools"] == {"old": "[]", "new": '["read"]'} for candidate in moved)

    # Two options are the whole space, so the search runs out of new candidates before the iteration
    # cap, and the champion's value decodes to a list.
    assert result.stop_reason == "exhausted"
    assert isinstance(json.loads(result.best.values["builtinTools"]), list)
