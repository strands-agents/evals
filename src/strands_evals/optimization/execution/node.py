"""Agents that run in a Node.js worker (`node/runner.mjs`): a TypeScript or JavaScript agent file.

One `NodeRunner` (a Node process) per agent file serves every trial over JSON lines on stdin / stdout. A trial's agent
is a `NodeAgent`: `invoke_async` sends the case to the worker with the current span's trace context, and the worker
runs the file again, builds a fresh agent, runs it inside that context, and replies with its output, its status, and
its spans, which `traces.ingest` adds to the trial's trace. So usage, the trajectory, and the candidate's spans read
as a Python agent's.
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


class NodeAgent:
    """A trial's agent in the Node worker: the file's `attr`, its `index`-th `createHarness` call given `values`.

    Only `invoke_async(prompt)` is supported; it returns the agent's answer as text. `trace_attributes` are set on
    the agent's spans in Node (the optimizer tags the factory's agent through them).
    """

    def __init__(self, runner: NodeRunner, attr: str, index: int, values: list[tuple[list[str | int], Any]]) -> None:
        self._runner, self._attr, self._index, self._values = runner, attr, index, values
        self.trace_attributes: dict[str, Any] = {}

    async def invoke_async(self, prompt: Any) -> str:
        context = trace_api.get_current_span().get_span_context()
        request: dict[str, Any] = {
            "type": "run",
            "attr": self._attr,
            "index": self._index,
            "values": self._values,
            "input": prompt,
            "trace_attributes": self.trace_attributes,
        }
        if context.is_valid:
            request["trace_id"] = format(context.trace_id, "032x")
            request["span_id"] = format(context.span_id, "016x")
        try:
            reply = await self._runner.call_async(request)
        except NodeRunnerGone as error:
            raise InfrastructureError(str(error)) from error
        ingest(reply.get("spans") or [])
        status = reply.get("status")
        if status == "completed":
            return str(reply.get("output", ""))
        failure = node_error(str(reply.get("error_type") or "Error"), str(reply.get("error") or ""))
        if status == "infrastructure_invalid":
            raise InfrastructureError(str(failure)) from failure
        raise failure
