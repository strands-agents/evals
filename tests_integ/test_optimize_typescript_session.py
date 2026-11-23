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
import time
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
    shutil.copytree(FIXTURE, folder, ignore=shutil.ignore_patterns("node_modules", "__pycache__", ".agent"))
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


def _fragile_agent(project: Path, name: str, log: Path, *, build_ms: int = 0, shutdown_throws: bool = False) -> Path:
    """An `agent.ts` variant whose build takes `build_ms` and whose shutdown writes "released" to `log`, then throws
    when `shutdown_throws`."""
    path = project / f"{name}.ts"
    path.write_text(
        f"""import {{ appendFileSync }} from 'node:fs'
import {{ createHarness }} from '@strands-agents/harness'
import {{ TwoTurnDeterministic }} from './world.ts'

await new Promise((done) => setTimeout(done, {build_ms}))
export const agent = await createHarness({{
  model: new TwoTurnDeterministic(),
  instructions: 'You answer questions.',
  effort: 'auto',
  builtinTools: [],
  builtinPlugins: [],
  skills: false,
}})
const shutdown = agent.shutdown?.bind(agent)
agent.shutdown = async () => {{
  await shutdown?.()
  appendFileSync({json.dumps(str(log))}, 'released\\n')
  if ({json.dumps(shutdown_throws)}) throw new Error('shutdown failed')
}}
"""
    )
    return path


def _released(log: Path, *, within: float = 10.0) -> int:
    """How many agents `log` says were released, waiting up to `within` seconds for at least one."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline and not log.exists():
        time.sleep(0.05)
    return log.read_text().count("released") if log.exists() else 0


def _start(runner: NodeRunner, attr_file_index: int) -> tuple[int, str]:
    """Send a raw `start`; returns `(request id, session id)`."""
    request_id, future = runner._send({"type": "start", "attr": "agent", "index": attr_file_index, "values": []})
    reply = future.result(timeout=60)
    assert reply["ok"], reply
    return request_id, reply["session"]


def _write(runner: NodeRunner, message: dict) -> None:
    """Write one protocol line to the worker, as `NodeRunner.call_async` writes a cancel."""
    with runner._lock:
        runner._process.stdin.write(json.dumps(message) + "\n")
        runner._process.stdin.flush()


def test_a_session_id_from_before_a_restart_names_no_session(project):
    runner = NodeRunner(project / "agent.ts")
    try:
        _, before = _start(runner, 0)
        runner._process.kill()
        runner._process.wait(timeout=10)
        _, after = _start(runner, 0)  # the next request restarts the worker
        assert before != after
        reply = runner.call({"type": "turn", "session": before, "input": "question 7"})
        assert (reply["status"], reply["error_type"]) == ("infrastructure_invalid", "NodeSessionGone")
        runner.call({"type": "release", "session": after})
    finally:
        runner.close()


def test_an_agent_whose_shutdown_throws_still_answers(project, tmp_path):
    log = tmp_path / "released.log"
    runner = NodeRunner(_fragile_agent(project, "throws_on_shutdown", log, shutdown_throws=True))
    try:
        agent = NodeAgent(runner, "agent", 0, [(["instructions"], INSTRUCTIONS[1])])

        async def both() -> tuple[str, str]:
            single = await agent.invoke_async("question 7")
            async with agent.session() as session:
                await session.turn("here it comes: question 7")
                second = await session.turn("question 7")
            return single, second

        assert asyncio.run(both()) == ("I'm not sure.", "answer 7")  # neither release failed the answer
        assert _released(log) == 2
    finally:
        runner.close()


def test_a_start_cancelled_while_building_releases_its_agent(project, tmp_path):
    log = tmp_path / "released.log"
    runner = NodeRunner(_fragile_agent(project, "slow_build", log, build_ms=2000))
    try:
        runner.call({"type": "describe", "attr": "agent", "paths": []})  # the worker is up before the clock starts

        async def cancel_the_start() -> None:
            entering = asyncio.ensure_future(NodeAgent(runner, "agent", 0, []).session().__aenter__())
            await asyncio.sleep(0.5)
            entering.cancel()
            with pytest.raises(asyncio.CancelledError):
                await entering

        log.unlink(missing_ok=True)  # `describe` built and released one too
        asyncio.run(cancel_the_start())
        assert _released(log) == 1
    finally:
        runner.close()


def test_a_start_cancelled_after_it_replied_releases_its_agent(project, tmp_path):
    log = tmp_path / "released.log"
    runner = NodeRunner(_fragile_agent(project, "cancel_after_reply", log))
    try:
        request_id, session = _start(runner, 0)
        _write(runner, {"type": "cancel", "target": request_id})  # the caller never learned `session`
        assert _released(log) == 1
        reply = runner.call({"type": "turn", "session": session, "input": "question 7"})
        assert reply["error_type"] == "NodeSessionGone"
    finally:
        runner.close()


def test_a_turn_after_a_cancelled_one_waits_for_the_agent(project):
    """A timed-out turn keeps the agent busy in Node until it notices; the next turn, and two at once, wait for it."""
    runner = NodeRunner(project / "agent.ts")
    try:

        async def converse() -> tuple[str, list[str]]:
            async with NodeAgent(runner, "agent", 0, []).session() as session:
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(session.turn("slow 1500"), 0.3)
                after = await session.turn("question 3")  # sent while the slow turn is still running
                both = await asyncio.gather(session.turn("slow 301"), session.turn("question 6"))
            return after, list(both)

        after, both = asyncio.run(converse())
        assert after == "answer 3"  # the cancelled turn's prompt counts: this is the conversation's second turn
        assert both == ["I'm not sure.", "answer 6"]
    finally:
        runner.close()


def test_requests_sharing_a_trace_each_get_their_spans(project):
    """Two sessions of one trial at once share its trace; the one that finishes first doesn't take the other's spans
    away (they finish later, after its reply), so every span comes back once."""
    runner = NodeRunner(project / "agent.ts")
    trace, parent = "0123456789abcdef0123456789abcdef", "0123456789abcdef"
    try:

        async def spans_of_one_session(slow: int) -> int:
            traced = {"trace_id": trace, "span_id": parent}
            start = await runner.call_async({"type": "start", "attr": "agent", "index": 0, "values": [], **traced})
            count = len(start["spans"])
            for prompt in (f"slow {slow}", "question 3"):
                reply = await runner.call_async(
                    {"type": "turn", "session": start["session"], "input": prompt, **traced}
                )
                count += len(reply["spans"])
            await runner.call_async({"type": "release", "session": start["session"]})
            return count

        async def together() -> list[int]:
            return list(await asyncio.gather(spans_of_one_session(50), spans_of_one_session(800)))

        alone = asyncio.run(spans_of_one_session(50))
        assert alone > 0 and asyncio.run(spans_of_one_session(800)) == alone
        assert sum(asyncio.run(together())) == 2 * alone
    finally:
        runner.close()
