"""Agents that run in a Node.js worker (`node/runner.mjs`): a TypeScript or JavaScript agent file.

One `NodeRunner` (a Node process) per agent file serves every trial over JSON lines on stdin / stdout. A trial's agent
is a `NodeAgent`: `invoke_async` sends the case to the worker with the current span's trace context, and the worker
runs the file again, builds a fresh agent, runs it inside that context, and replies with its output, its status, and
its spans, which `traces.ingest` adds to the trial's trace. So usage, the trajectory, and the candidate's spans read
as a Python agent's.

A rollout that speaks more than once opens a `NodeSession` (`NodeAgent.session()`): the worker builds the agent once
and keeps it, every `turn` is answered by that same agent with the conversation so far, and leaving the session
releases it.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import shutil
import subprocess
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import opentelemetry.trace as trace_api

from .status import InfrastructureError
from .traces import ingest

logger = logging.getLogger(__name__)

RUNNER = Path(__file__).parent / "node" / "runner.mjs"
SUFFIXES = frozenset({".ts", ".mts", ".cts", ".js", ".mjs", ".cjs"})
_SETUP_TIMEOUT = 300.0


class NodeRunnerGone(RuntimeError):
    """The Node worker exited or stopped reading requests."""


class NodeError(Exception):
    """An error the agent raised in Node; `node_error` names a subclass after the JavaScript error's type."""


_ERROR_TYPES: dict[str, type[NodeError]] = {}


def node_error(error_type: str, message: str) -> NodeError:
    """A `NodeError` whose type is named `error_type`, so a trial records the JavaScript error's type."""
    name = error_type if error_type.isidentifier() else "NodeError"
    kind = _ERROR_TYPES.get(name)
    if kind is None:
        kind = _ERROR_TYPES[name] = type(name, (NodeError,), {"__module__": "node"})
    return kind(message)


class NodeRunner:
    """One Node worker running `agent_file`; restarted on the next request after it exits."""

    def __init__(self, agent_file: Path, *, node: str | None = None) -> None:
        found = node or shutil.which("node")
        if found is None:
            raise FileNotFoundError("TypeScript agents need Node.js 22.18 or later on PATH (`node`)")
        self._command = [found, str(RUNNER), str(agent_file)]
        self._lock = threading.Lock()
        self._pending: dict[int, Future[dict[str, Any]]] = {}
        self._next_id = 0
        self._process: subprocess.Popen[str] | None = None
        atexit.register(self.close)

    def _start(self) -> subprocess.Popen[str]:
        # The harness's own trace export would compete with the runner's collector; trials are traced by the runner.
        env = {**os.environ, "OTEL_TRACES_EXPORTER": "none"}
        process = subprocess.Popen(  # noqa: S603 - the command is node and this package's runner
            self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", env=env
        )
        threading.Thread(target=self._read, args=(process,), name="strands-evals-node-runner", daemon=True).start()
        return process

    def _read(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                reply = json.loads(line)
            except ValueError:
                logger.warning("line=<%s> | ignored a node runner line that is not json", line.strip()[:200])
                continue
            with self._lock:
                future = self._pending.pop(reply.get("id"), None)
            if future is not None and not future.done():
                future.set_result(reply)
        code = process.wait()
        with self._lock:
            if self._process is process:
                self._process = None
            gone, self._pending = self._pending, {}
        for future in gone.values():
            if not future.done():
                future.set_exception(NodeRunnerGone(f"the Node agent runner exited with code {code}"))

    def _send(self, message: dict[str, Any]) -> tuple[int, Future[dict[str, Any]]]:
        future: Future[dict[str, Any]] = Future()
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._process = self._start()
            self._next_id += 1
            request_id = self._next_id
            self._pending[request_id] = future
            try:
                assert self._process.stdin is not None
                self._process.stdin.write(json.dumps({**message, "id": request_id}) + "\n")
                self._process.stdin.flush()
            except OSError as error:
                self._pending.pop(request_id, None)
                raise NodeRunnerGone(f"the Node agent runner stopped reading requests: {error}") from error
        return request_id, future

    def call(self, message: dict[str, Any], *, timeout: float | None = _SETUP_TIMEOUT) -> dict[str, Any]:
        """Send a request and wait for its reply (`describe`, `probe`)."""
        return _checked(self._send(message)[1].result(timeout=timeout))

    async def call_async(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a request and await its reply; a cancelled call cancels the request in Node."""
        request_id, future = self._send(message)
        try:
            return _checked(await asyncio.wrap_future(future))
        except asyncio.CancelledError:
            with self._lock:
                self._pending.pop(request_id, None)
                process = self._process
            if process is not None and process.stdin is not None:
                try:
                    with self._lock:
                        process.stdin.write(json.dumps({"type": "cancel", "target": request_id}) + "\n")
                        process.stdin.flush()
                except OSError:
                    pass
            raise

    def close(self) -> None:
        """Close the worker's stdin (it finishes what is in flight, then exits), killing it if it doesn't."""
        with self._lock:
            process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        try:
            assert process.stdin is not None
            process.stdin.close()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()


def _checked(reply: dict[str, Any]) -> dict[str, Any]:
    if "setup_error" in reply:
        raise RuntimeError(f"the Node agent runner couldn't start: {reply['setup_error']}")
    return reply


def _answered(reply: dict[str, Any]) -> str:
    """The reply's output when it completed; otherwise the failure it reports, raised.

    Ingests the reply's spans either way (a crash's spend is real), names the error after the JavaScript
    error's type, and raises `InfrastructureError` for a reply the worker classified as infrastructure.
    """
    ingest(reply.get("spans") or [])
    status = reply.get("status")
    if status == "completed":
        return str(reply.get("output", ""))
    failure = node_error(str(reply.get("error_type") or "Error"), str(reply.get("error") or ""))
    if status == "infrastructure_invalid":
        raise InfrastructureError(str(failure)) from failure
    raise failure


def _traced(request: dict[str, Any]) -> dict[str, Any]:
    """`request` with the current span's trace context, when there is one, so the agent's spans parent to it."""
    context = trace_api.get_current_span().get_span_context()
    if context.is_valid:
        request["trace_id"] = format(context.trace_id, "032x")
        request["span_id"] = format(context.span_id, "016x")
    return request


class NodeAgent:
    """A trial's agent in the Node worker: the file's `attr`, its `index`-th `createHarness` call given `values`.

    `invoke_async(prompt)` asks a fresh agent once and returns its answer as text; `session()` opens a
    conversation whose turns share one agent. `trace_attributes` are set on the agent's spans in Node (the
    optimizer tags the factory's agent through them).
    """

    def __init__(self, runner: NodeRunner, attr: str, index: int, values: list[tuple[list[str | int], Any]]) -> None:
        self._runner, self._attr, self._index, self._values = runner, attr, index, values
        self.trace_attributes: dict[str, Any] = {}

    async def invoke_async(self, prompt: Any) -> str:
        request = _traced(
            {
                "type": "run",
                "attr": self._attr,
                "index": self._index,
                "values": self._values,
                "input": prompt,
                "trace_attributes": self.trace_attributes,
            }
        )
        try:
            reply = await self._runner.call_async(request)
        except NodeRunnerGone as error:
            raise InfrastructureError(str(error)) from error
        return _answered(reply)

    def session(self) -> NodeSession:
        """A conversation with one agent, for a rollout that speaks more than once.

        Use as an async context manager: entering builds the agent in the worker and holds it open, each
        `turn(prompt)` asks that same agent (the conversation so far included), and leaving releases it::

            async with agent.session() as session:
                first = await session.turn("hello")
                second = await session.turn("and again?")
        """
        return NodeSession(self._runner, self._attr, self._index, self._values, self.trace_attributes)


class NodeSession:
    """One agent held open in the Node worker, answering every `turn` with the conversation so far.

    Built by `NodeAgent.session()`. Entering sends `start` (the agent is built once there; a build the
    values break fails here), each `turn` sends the prompt to that agent, and leaving sends `release`.
    `last_result` is the latest completed turn's structured result as the worker could write it to JSON
    (the stop reason, the final message, any structured output), or None.
    """

    def __init__(
        self,
        runner: NodeRunner,
        attr: str,
        index: int,
        values: list[tuple[list[str | int], Any]],
        trace_attributes: dict[str, Any],
    ) -> None:
        self._runner, self._attr, self._index, self._values = runner, attr, index, values
        self._trace_attributes = dict(trace_attributes)
        self._session: str | None = None
        self.last_result: Any = None

    async def __aenter__(self) -> NodeSession:
        request = {
            "type": "start",
            "attr": self._attr,
            "index": self._index,
            "values": self._values,
            "trace_attributes": self._trace_attributes,
        }
        try:
            reply = await self._runner.call_async(request)
        except NodeRunnerGone as error:
            raise InfrastructureError(str(error)) from error
        if not reply.get("ok"):
            failure = node_error(str(reply.get("error_type") or "Error"), str(reply.get("error") or ""))
            if reply.get("status") == "infrastructure_invalid":
                raise InfrastructureError(str(failure)) from failure
            raise failure
        self._session = str(reply["session"])
        return self

    async def turn(self, prompt: Any) -> str:
        """Ask the held agent `prompt` and return its answer as text; `last_result` holds the structured result."""
        if self._session is None:
            raise RuntimeError("the session is not open: enter it first (`async with agent.session() as session`)")
        request = _traced({"type": "turn", "session": self._session, "input": prompt})
        try:
            reply = await self._runner.call_async(request)
        except NodeRunnerGone as error:
            # The worker restarting lost the conversation: later turns would silently start one afresh.
            self._session = None
            raise InfrastructureError(str(error)) from error
        answer = _answered(reply)
        self.last_result = reply.get("result")
        return answer

    async def __aexit__(self, *exc_info: Any) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        try:
            await self._runner.call_async({"type": "release", "session": session})
        except NodeRunnerGone:
            pass  # the worker is gone, and with it everything the session held open
