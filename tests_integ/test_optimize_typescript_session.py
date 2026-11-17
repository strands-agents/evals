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
import logging
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.context import Context

from strands_evals import Case, Experiment
from strands_evals.cli.main import main
from strands_evals.optimization import OptimizationResult
from strands_evals.optimization.execution.node import NodeAgent, NodeRunner
from strands_evals.optimization.execution.rollout import TrialLog, instrument
from strands_evals.optimization.execution.traces import TraceCapture, ingest
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
    assert result.cost.agent["train"].usage.tokens > 0  # the turns' spans landed in the trials
    # The exact instruction answers all but every 10th question; the plain one every 3rd. One conversation
    # never held open would answer nothing at all (`rollout.py` raises when an opening turn answers).
    best_train = result.reports[result.best.id]["train"]
    assert best_train.overall_score is not None and best_train.overall_score >= 0.8


def _fragile_agent(
    project: Path, name: str, log: Path, *, build_ms: int = 0, shutdown_throws: bool = False, memory: bool = True
) -> Path:
    """An `agent.ts` variant whose build takes `build_ms` and whose shutdown writes "released" to `log`, then throws
    when `shutdown_throws`; `memory=False` turns the harness's memory off."""
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
  skills: false,{"" if memory else " memory: false,"}
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


def _trial_folders(tmp: Path) -> list[Path]:
    return sorted(tmp.glob("strands-evals-trial-*"))


def test_a_session_cancelled_mid_turn_leaves_no_folder(project, tmp_path, monkeypatch):
    """Leaving a session waits for a cancelled turn still running before its folders go, so the turn's session
    snapshot isn't written back after them; a session nobody released is cleaned up when the worker closes."""
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    for name in ("TMPDIR", "TEMP", "TMP"):  # the worker's os.tmpdir() (TEMP / TMP on Windows), read when it starts
        monkeypatch.setenv(name, str(node_tmp))
    runner = NodeRunner(_fragile_agent(project, "no_memory", tmp_path / "released.log", memory=False))
    try:
        agent = NodeAgent(runner, "agent", 0, [])

        async def cancelled_mid_turn() -> None:
            async with agent.session() as session:
                await session.turn("hello")
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(session.turn("slow 1500"), 0.3)

        asyncio.run(cancelled_mid_turn())
        time.sleep(0.5)  # anything the cancelled turn still had to write
        assert _trial_folders(node_tmp) == []  # `release` removed them, after the turn ended

        async def left_open() -> None:
            await agent.session().__aenter__()  # never released

        asyncio.run(left_open())
        assert len(_trial_folders(node_tmp)) == 1
    finally:
        runner.close()
    assert _trial_folders(node_tmp) == []  # the worker let it go on close


def test_a_cancelled_turn_stops_the_agent_rather_than_waiting_it_out(project):
    """The cancel reaches the agent's model call, so the next turn doesn't wait for the cancelled one's full length."""
    runner = NodeRunner(project / "agent.ts")
    try:

        async def converse() -> float:
            async with NodeAgent(runner, "agent", 0, []).session() as session:
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(session.turn("abortable 10000"), 0.3)
                started = time.monotonic()
                await session.turn("question 3")
                return time.monotonic() - started

        assert asyncio.run(converse()) < 5  # the abortable call would take 10 s left alone
    finally:
        runner.close()


def test_a_cancelled_request_leaves_the_others_spans_to_them(project):
    """A cancelled turn's reply is thrown away; the spans a run of the same trial finished meanwhile still reach the
    run's reply."""
    runner = NodeRunner(project / "agent.ts")
    traced = {"trace_id": "fedcba9876543210fedcba9876543210", "span_id": "fedcba9876543210"}
    run = {"type": "run", "attr": "agent", "index": 0, "values": [], "input": "slow 3000", **traced}
    try:

        async def names_of_a_run(*, beside_a_cancelled_turn: bool) -> Counter:
            if not beside_a_cancelled_turn:
                return Counter(span["name"] for span in (await runner.call_async(run))["spans"])
            start = await runner.call_async({"type": "start", "attr": "agent", "index": 0, "values": [], **traced})
            running = asyncio.ensure_future(runner.call_async(run))
            await asyncio.sleep(0.2)  # the run's memory spans have finished; its model call takes 3 s
            turn = {"type": "turn", "session": start["session"], "input": "slow 1500", **traced}
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(runner.call_async(turn), 0.3)  # finishes in Node at 1.5 s, while the run runs
            spans = (await running)["spans"]
            await runner.call_async({"type": "release", "session": start["session"]})
            return Counter(span["name"] for span in spans)

        alone = asyncio.run(names_of_a_run(beside_a_cancelled_turn=False))
        together = asyncio.run(names_of_a_run(beside_a_cancelled_turn=True))
        assert alone and not alone - together  # every span of the run alone is still there (the turn's join them)
    finally:
        runner.close()


def test_a_cancelled_last_turn_still_brings_its_spans_back(project):
    """A rollout that times out its last turn and leaves the session: the turn's reply comes back before the release's,
    and its spans are read off it, so the turn's spend still counts."""
    runner = NodeRunner(project / "agent.ts")
    traced = {"trace_id": "00112233445566778899aabbccddeeff", "span_id": "0011223344556677"}
    try:

        async def cancel_the_last_turn() -> list[dict]:
            late: list[dict] = []
            start = await runner.call_async({"type": "start", "attr": "agent", "index": 0, "values": [], **traced})
            turn = {"type": "turn", "session": start["session"], "input": "slow 1000", **traced}
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(runner.call_async(turn, on_late=late.append), 0.2)
            await runner.call_async({"type": "release", "session": start["session"]})
            return late  # read as soon as the release replied

        (reply,) = asyncio.run(cancel_the_last_turn())
        assert "chat" in {span["name"] for span in reply["spans"]}  # the cancelled turn's model call
    finally:
        runner.close()


def test_a_release_waits_for_a_stuck_turn_only_so_long(project, tmp_path, monkeypatch):
    """A turn that ignores the cancel holds `release` up for `wait_ms` at most; its folders still go once it ends."""
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(node_tmp))
    runner = NodeRunner(_fragile_agent(project, "stuck_turn", tmp_path / "released.log", memory=False))
    try:

        async def release_beside_a_stuck_turn() -> float:
            start = await runner.call_async({"type": "start", "attr": "agent", "index": 0, "values": []})
            turn = {"type": "turn", "session": start["session"], "input": "slow 3000"}
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(runner.call_async(turn), 0.2)
            began = time.monotonic()
            await runner.call_async({"type": "release", "session": start["session"], "wait_ms": 300})
            return time.monotonic() - began

        assert asyncio.run(release_beside_a_stuck_turn()) < 2  # not the turn's 3 s
        deadline = time.monotonic() + 10
        while _trial_folders(node_tmp) and time.monotonic() < deadline:
            time.sleep(0.1)
        time.sleep(0.5)  # anything the turn still wrote as it ended
        assert _trial_folders(node_tmp) == []
    finally:
        runner.close()


def _model_calls_of_a_long_conversation(project: Path, turns: int) -> list[dict]:
    """The `chat` spans of one conversation of `turns` turns over the real runner, as its replies carry them."""
    runner = NodeRunner(project / "agent.ts")
    traced = {"trace_id": "0123456789abcdef0123456789abcdef", "span_id": "0123456789abcdef"}
    try:

        async def talk() -> list[dict]:
            start = await runner.call_async({"type": "start", "attr": "agent", "index": 0, "values": [], **traced})
            spans = list(start["spans"])
            for i in range(turns):
                turn = {"type": "turn", "session": start["session"], "input": f"question {i}", **traced}
                spans += (await runner.call_async(turn))["spans"]
            await runner.call_async({"type": "release", "session": start["session"]})
            return [span for span in spans if span["attributes"].get("gen_ai.operation.name") == "chat"]

        return asyncio.run(talk())
    finally:
        runner.close()


def test_a_long_conversations_model_calls_keep_every_message(project, monkeypatch):
    """OpenTelemetry's default of 128 events per span would drop the start of the conversation."""
    monkeypatch.delenv("OTEL_SPAN_EVENT_COUNT_LIMIT", raising=False)
    last = _model_calls_of_a_long_conversation(project, 70)[-1]  # 70 user turns and 69 replies before it

    assert len(last["events"]) > 128
    assert last["dropped_events"] == 0


def test_a_limit_the_user_set_is_kept_and_what_it_dropped_is_reported(project, monkeypatch, caplog):
    monkeypatch.setenv("OTEL_SPAN_EVENT_COUNT_LIMIT", "50")
    spans = _model_calls_of_a_long_conversation(project, 70)
    last = spans[-1]
    assert len(last["events"]) == 50
    assert last["dropped_events"] > 0

    traces = TraceCapture()
    with trace_api.get_tracer("optimization-integ").start_as_current_span("execute_case", context=Context()) as root:
        trace_id = format(root.get_span_context().trace_id, "032x")
        parent = format(root.get_span_context().span_id, "016x")
    traces.register(trace_id)
    for span in spans:
        span.update(trace_id=trace_id, parent_span_id=parent)  # each model call straight under the trial
    ingest(spans)
    with caplog.at_level(logging.WARNING, logger="strands_evals.optimization.execution.traces"):
        traces.session(trace_id, parent)

    assert f"dropped_events=<{sum(span['dropped_events'] for span in spans)}>" in caplog.text


def test_a_cancelled_calls_late_reply_lands_in_the_trials_usage(project):
    """A rollout that gives up on a call returns before the worker's late reply; the trial still counts its spend."""
    traces = TraceCapture()
    runner = NodeRunner(project / "agent.ts")
    agent = NodeAgent(runner, "agent", 0, [])

    async def gives_up(case: Case) -> str:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(agent.invoke_async("slow 600"), 0.2)  # the worker finishes it at about 0.6 s
        return "gave up"

    async def trial() -> TrialLog:
        await agent.invoke_async("question 1")  # the worker is up before the trial starts
        log = TrialLog()
        with trace_api.get_tracer("optimization-integ").start_as_current_span("execute_case", context=Context()):
            await instrument(gives_up, log, traces)(Case(name="c", input="q"))
        return log

    try:
        log = asyncio.run(trial())
    finally:
        runner.close()
    assert log.records()["c"].usage.model_calls == 1
