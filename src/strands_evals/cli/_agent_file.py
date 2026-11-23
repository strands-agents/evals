"""The agent-file route of `strands-evals optimize --agent`: a Python file that builds its agent with `create_harness`.

`/export` writes such a file (`agent = create_harness(...)`). Its space is `--tune` paths over the keyword arguments of
the `create_harness` call that built the agent, `instructions` by default, as `--config`'s are over a portable config.
The file becomes a `TunableFactory` whose every build runs the file again with that call given the candidate's values,
so each trial gets a fresh agent and fresh objects (model clients, tools, hooks). Modules the file imports are
imported once and shared, as for any factory: build anything that keeps state in the file itself.

Every `create_harness` call the file makes gets its own fresh `session` and `memory` folders (the memory folder a copy
of the configured one), as `--config` trials do, and `callback_handler=None` unless the call gives one.
"""

from __future__ import annotations

import importlib.util
import inspect
import itertools
import json
import logging
import re
import shutil
import sys
import tempfile
import threading
import weakref
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import strands_harness
import strands_harness.agent
from strands_harness.defaults import DEFAULT_MEMORY_DIR

from ..optimization import Marker, SkillLibrary, TextValue, TunableFactory, tunable
from ..optimization.execution import node
from ..optimization.execution.status import InfrastructureError
from ._entrypoint import EntryPointError, _looks_like_path, import_attr
from ._harness import DEFAULT_TUNED, Tuned, _choice, _number, _path, _Range, _set, _split

logger = logging.getLogger(__name__)

DEFAULT_ATTR = "agent"  # what `/export` names the agent

_REAL_CREATE_HARNESS = strands_harness.agent.create_harness
_SIGNATURE = inspect.signature(_REAL_CREATE_HARNESS)
_PATCH_LOCK = threading.Lock()  # one file runs at a time while `create_harness` is swapped
_RUNS = itertools.count()
_WARNED: set[str] = set()


@dataclass
class _Capture:
    """The `create_harness` calls of one run of the file, and the values to give the call that builds the agent."""

    target: int | None = None
    values: dict[tuple[str | int, ...], Any] = field(default_factory=dict)
    calls: list[tuple[dict[str, Any], Any]] = field(default_factory=list)  # (kwargs as the file wrote them, agent)


_CAPTURE: ContextVar[_Capture | None] = ContextVar("agent_file_capture", default=None)


def _create_harness(**kwargs: Any) -> Any:
    """`create_harness` while a file runs: fresh state for every call, the values for the agent's; else the real one."""
    capture = _CAPTURE.get()
    if capture is None:
        return _REAL_CREATE_HARNESS(**kwargs)
    written = dict(kwargs)
    trial = dict(kwargs)
    if len(capture.calls) == capture.target:
        for path, value in capture.values.items():
            _set(trial, path, value)
    folder = Path(tempfile.mkdtemp(prefix="strands-evals-trial-"))
    try:
        trial = _fresh_state(trial, folder)
        trial.setdefault("callback_handler", None)  # Strands' default prints every streamed answer
        agent = _REAL_CREATE_HARNESS(**trial)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    weakref.finalize(agent, shutil.rmtree, folder, True)  # also runs at exit
    capture.calls.append((written, agent))
    return agent


def load(spec: str, tune: list[str], *, tune_instructions: bool = True) -> TunableFactory:
    """The `--agent` file as a `TunableFactory`: its own when it defines one, else its `create_harness` agent's.

    Args:
        spec: `FILE.py` (the attribute `agent`) or `FILE[.py]:ATTR`; or a TypeScript / JavaScript file (`agent.ts`,
            `agent.mjs`, ...), with or without `:ATTR`, whose agent `createHarness` builds, run in a Node worker.
        tune: The `--tune` specs, as `--config` takes them, over the agent's `create_harness` keyword arguments.
        tune_instructions: Whether `instructions` is tuned without a `--tune` naming it.

    Raises:
        EntryPointError: The file is missing, fails to run, lacks the attribute, or the attribute is neither a
            tunable factory nor an agent a `create_harness` call in the file built.
        ValueError: A bad `--tune` (as `--config`'s), or `--tune` given for a tunable factory.
    """
    found = _file_and_attr(spec)
    if found is None:
        raise EntryPointError(f"--agent {spec}: not a Python file")
    path, attr = found
    if path.suffix in node.SUFFIXES:
        return _load_node(path, attr, tune, tune_instructions=tune_instructions)
    if spec.endswith(".py"):  # no attribute given: new with agent files
        module, capture = _run_file(path, _Capture())
        target = getattr(module, attr, _MISSING)
        if target is _MISSING:
            raise EntryPointError(f"--agent {spec}: {path.name} has no attribute {attr!r}")
    else:  # `FILE:ATTR` is imported as `--agent` always imported it; a file is run again only for an agent
        target = import_attr(spec)
        capture = None
    if isinstance(target, TunableFactory):
        if tune or not tune_instructions:
            raise ValueError(
                f"--agent {spec} is a tunable factory, which declares its space with markers; --tune and "
                "--no-tune-instructions go with --config or an agent built by create_harness"
            )
        return target
    if capture is None:
        module, capture = _run_file(path, _Capture())
        target = getattr(module, attr, _MISSING)
    index = next((i for i, (_, agent) in enumerate(capture.calls) if agent is target), None)
    if index is None:
        raise EntryPointError(
            f"--agent {spec}: {attr} is a {type(target).__name__}, neither a tunable factory (@tunable) nor an agent "
            f"built by a create_harness call in {path.name}"
        )
    kwargs = capture.calls[index][0]
    markers, tuning = _declare(lambda p, t: _current(kwargs, p, t), tune, tune_instructions=tune_instructions)
    return tunable(_AgentFile(path, attr, index, tuning.leaving_out(kwargs)).build, inject=markers)


@dataclass(frozen=True)
class _AgentFile:
    """Builds the agent by running the file again, its `index`-th `create_harness` call given the values."""

    path: Path
    attr: str
    index: int
    tuning: _Tuning

    def build(self, **values: Any) -> Any:
        overrides = dict(self.tuning.overrides(values))
        module, capture = _run_file(self.path, _Capture(target=self.index, values=overrides))
        agent = getattr(module, self.attr, None)
        if len(capture.calls) <= self.index or capture.calls[self.index][1] is not agent:
            raise RuntimeError(
                f"{self.path.name} built {self.attr} from another create_harness call this time; build it from the "
                "same call on every run of the file"
            )
        return agent


_MISSING = object()


@dataclass(frozen=True)
class _Tuning:
    """How each tuned value reaches the agent's call.

    Attributes:
        tuned: Each declared name's path and kind.
        listed: Names of a `skills` the call gave as a list of one folder (as `/export` writes it), passed as one.
        baselines: Each name's value in the call (or the default it was left at).
        left_out: Names whose argument the call leaves out: passed only when changed, so the baseline builds the call
            as written (an unset `caching` and `caching=True` aren't the same to the harness).
    """

    tuned: list[Tuned]
    listed: frozenset[str]
    baselines: dict[str, Any]
    left_out: frozenset[str] = frozenset()

    def leaving_out(self, written: Iterable[str]) -> _Tuning:
        """This tuning, with the names whose top-level argument isn't in `written` left out at their baseline."""
        given = set(written)
        left_out = frozenset(item.name for item in self.tuned if item.path[0] not in given)
        return _Tuning(self.tuned, self.listed, self.baselines, left_out)

    def overrides(self, values: Mapping[str, Any]) -> list[tuple[tuple[str | int, ...], Any]]:
        """`(path, value)` for the candidate's values, as the call takes them: a skill folder (the candidate's copy)
        as text or a list of one; a left-out argument still at its baseline is skipped."""
        found = []
        for item in self.tuned:
            if item.name not in values:
                continue
            value = values[item.name]
            baseline = self.baselines.get(item.name)
            if item.name in self.left_out and type(value) is type(baseline) and value == baseline:
                continue
            if item.kind == "skills":
                value = [str(value)] if item.name in self.listed else str(value)
            found.append((item.path, value))
        return found


def _file_and_attr(spec: str) -> tuple[Path, str] | None:
    if spec.endswith((".py", *node.SUFFIXES)):
        file, attr = spec, DEFAULT_ATTR
    elif ":" in spec:
        file, attr = spec.rsplit(":", 1)
        if not (_looks_like_path(file) or file.endswith(tuple(node.SUFFIXES))) or not attr:
            return None
    else:
        return None
    path = Path(file).expanduser()
    return (path if path.suffix in {".py", *node.SUFFIXES} else path.with_suffix(".py")).resolve(), attr


def _run_file(path: Path, capture: _Capture) -> tuple[ModuleType, _Capture]:
    """Run the file as a fresh module, with `create_harness` swapped for `_create_harness` while it runs."""
    if not path.is_file():
        raise EntryPointError(f"--agent: cannot find {path}")
    name = f"_strands_evals_agent_file_{path.stem}_{next(_RUNS)}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EntryPointError(f"--agent: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    with _PATCH_LOCK:
        token = _CAPTURE.set(capture)
        strands_harness.create_harness = strands_harness.agent.create_harness = _create_harness  # type: ignore[assignment]
        sys.modules[name] = module  # dataclasses and typing look a class's module up while the file runs
        try:
            spec.loader.exec_module(module)
        except EntryPointError:
            raise
        except Exception as error:
            raise EntryPointError(f"--agent: running {path.name} failed: {type(error).__name__}: {error}") from error
        finally:
            sys.modules.pop(name, None)
            strands_harness.create_harness = strands_harness.agent.create_harness = _REAL_CREATE_HARNESS
            _CAPTURE.reset(token)
    return module, capture


def _declare(
    current: Callable[[tuple[str | int, ...], str], Any], tune: list[str], *, tune_instructions: bool
) -> tuple[dict[str, Marker], _Tuning]:
    """The space over the agent's `create_harness` arguments, as `_harness.declare` over a config; `current(path,
    text)` gives the agent's value at a path, or raises `ValueError` naming the `--tune` `text`."""
    specs = [_split(spec) for spec in tune]
    if tune_instructions and DEFAULT_TUNED not in {path for path, _ in specs}:
        specs.insert(0, (DEFAULT_TUNED, None))
    if not specs:
        raise ValueError("--no-tune-instructions with no --tune: nothing is left to tune")
    markers: dict[str, Marker] = {}
    tuned: list[Tuned] = []
    listed: set[str] = set()
    baselines: dict[str, Any] = {}
    kind: Literal["text", "skills", "choice", "number"]
    for text, options in specs:
        path = _path(text)
        name = text.replace(".", "_")
        if name in markers:
            raise ValueError(f"--tune {text}: another --tune path is also named {name!r}")
        value = baselines[name] = current(path, text)
        if isinstance(options, _Range):
            markers[name], kind = _number(text, options, value), "number"
        elif options is not None:
            markers[name], kind = _choice(text, "the agent's", options, value), "choice"
        elif path == ("skills",):
            if isinstance(value, list) and len(value) == 1:  # /export writes skills=[project_path(...)]
                listed.add(name)
                value = value[0]
            if not isinstance(value, str) or value.startswith(("http://", "https://")):
                raise ValueError("--tune skills: the agent's skills must be one local folder path (or a list of one)")
            markers[name], kind = SkillLibrary(Path(value).expanduser().resolve()), "skills"
        elif not isinstance(value, str):
            hint = 'add instructions="" (it may be empty) to tune it' if path == (DEFAULT_TUNED,) else ""
            raise ValueError(
                f"--tune {text} is {'unset' if value is None else 'a ' + type(value).__name__}, not text: "
                + (hint or f"give --tune {text}=JSON_LIST for a choice, or {text}=int:LO..HI / float:LO..HI")
            )
        else:
            markers[name], kind = TextValue(default=value), "text"
        tuned.append(Tuned(name, path, kind))
    return markers, _Tuning(tuned, frozenset(listed), baselines)


def _current(kwargs: dict[str, Any], path: tuple[str | int, ...], text: str) -> Any:
    """The value at `path` in the call; a top-level argument left out is at `create_harness`'s default, or the
    harness's default config's when that default isn't a plain value (`caching`), as `/export` leaves defaults out."""
    head, rest = path[0], path[1:]
    if head in kwargs:
        value = kwargs[head]
    elif isinstance(head, str) and head in _SIGNATURE.parameters and head != "agent_kwargs":
        value = _SIGNATURE.parameters[head].default
        if not isinstance(value, (str, int, float, bool, type(None))):  # e.g. caching's "not given" sentinel
            config_key = re.sub(r"_([a-z])", lambda match: match.group(1).upper(), head)
            value = strands_harness.DEFAULT_HARNESS_AGENT_CONFIG.get(config_key, inspect.Parameter.empty)
        if value is inspect.Parameter.empty or not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError(
                f"--tune {text}: write {head}= in the create_harness call (its value there is the baseline)"
            )
    else:
        raise ValueError(f"--tune {text}: the create_harness call has no argument {head!r}; write it in the call")
    for part in rest:
        try:
            value = value[part]
        except (KeyError, IndexError, TypeError):
            raise ValueError(
                f"--tune {text}: the create_harness call has nothing at that path; write it in the call "
                "(its value there is the baseline)"
            ) from None
    return value


def _fresh_state(kwargs: dict[str, Any], folder: Path) -> dict[str, Any]:
    """`kwargs` with `session` in an empty folder and `memory` in a copy of its folder, both under `folder`.

    A session or memory given as an object (a `SessionManager`, a `MemoryManager`) is left as it is, with a warning:
    trials share whatever it keeps.
    """
    trial = dict(kwargs)
    for key, kind in (("session", "session manager"), ("memory", "memory manager")):
        value = trial.get(key, True)
        if value is False:
            continue
        if not isinstance(value, (bool, dict, type(None))):
            if key not in _WARNED:
                _WARNED.add(key)
                logger.warning(
                    "argument=<%s> | the agent's %s is an object, so every trial shares its state | pass a config "
                    "(%s={...}) or %s=False instead",
                    key,
                    kind,
                    key,
                    key,
                )
            continue
        config = dict(value) if isinstance(value, dict) else {}
        if key == "memory":
            source = Path(config["dir"]).expanduser() if "dir" in config else Path(DEFAULT_MEMORY_DIR)
            if source.is_dir():
                shutil.copytree(source, folder / key)
        config["dir"] = str(folder / key)
        trial[key] = config
    return trial


_RUNNERS: dict[Path, node.NodeRunner] = {}


def _load_node(path: Path, attr: str, tune: list[str], *, tune_instructions: bool) -> TunableFactory:
    """A TypeScript / JavaScript agent file as a `TunableFactory` of `NodeAgent`s, its space from the Node worker."""
    if not path.is_file():
        raise EntryPointError(f"--agent: cannot find {path}")
    try:
        runner = _RUNNERS.get(path) or _RUNNERS.setdefault(path, node.NodeRunner(path))
    except FileNotFoundError as error:
        raise EntryPointError(f"--agent {path.name}: {error}") from error
    specs = [_split(spec) for spec in tune]
    paths = [_path(text) for text, _ in specs]
    if tune_instructions and (DEFAULT_TUNED,) not in paths:
        paths.insert(0, (DEFAULT_TUNED,))
    try:
        described = runner.call({"type": "describe", "attr": attr, "paths": [list(p) for p in paths]})
    except RuntimeError as error:
        raise EntryPointError(f"--agent {path.name}: {error}") from error
    if "error" in described:
        raise EntryPointError(f"--agent {path.name}: {described['error']}")
    found = dict(zip(paths, described["values"], strict=True))

    def current(at: tuple[str | int, ...], text: str) -> Any:
        entry = found[at]
        if "missing" in entry:
            raise ValueError(f"--tune {text}: {entry['missing']}; write it in the createHarness call to tune it")
        return entry["value"]

    markers, tuning = _declare(current, tune, tune_instructions=tune_instructions)
    tuning = tuning.leaving_out(described.get("written", []))
    return tunable(_NodeAgentFile(runner, attr, int(described["index"]), tuning).build, inject=markers)


@dataclass(frozen=True)
class _NodeAgentFile:
    """Builds a `NodeAgent` for the values, after the worker built that agent once without errors."""

    runner: node.NodeRunner
    attr: str
    index: int
    tuning: _Tuning
    _built: set[str] = field(default_factory=set)

    def build(self, **values: Any) -> node.NodeAgent:
        overrides = [(list(path), value) for path, value in self.tuning.overrides(values)]
        key = json.dumps(overrides, sort_keys=True)
        if key not in self._built:  # once per candidate: a value the agent can't be built with fails here
            reply = self.runner.call({"type": "probe", "attr": self.attr, "index": self.index, "values": overrides})
            if not reply.get("ok"):
                error = node.node_error(str(reply.get("error_type") or "Error"), str(reply.get("error") or ""))
                if reply.get("status") == "infrastructure_invalid":
                    raise InfrastructureError(str(error)) from error
                raise error
            self._built.add(key)
        return node.NodeAgent(self.runner, self.attr, self.index, overrides)
