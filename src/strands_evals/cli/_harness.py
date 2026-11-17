"""The `--config` agent of `strands-evals optimize`: a strands-harness portable config.

The space comes from the config's paths (`instructions`, plus every `--tune`): a text field is a `JsonField` on the
config, a `skills` path a `SkillLibrary`, a choice a `OneOf`. Each candidate's values are set at their paths in a copy
of the config held in memory; each trial builds a fresh harness from it, with its own empty session folder and a copy
of the configured memory folder, so no trial sees another's state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from strands_harness import (
    create_harness,
    define_harness_agent_config,
    harness_agent_kwargs_from_config,
)
from strands_harness.defaults import DEFAULT_MEMORY_DIR

from ..case import Case
from ..optimization import BuildContext, JsonField, Marker, OneOf, SkillLibrary

logger = logging.getLogger(__name__)

DEFAULT_TUNED = "instructions"
_HOST_TOOLS = ("shell", "write", "edit")


@dataclass(frozen=True)
class Tuned:
    """One tuned config path: its declared name, its path as keys and indexes, and what it becomes."""

    name: str
    path: tuple[str | int, ...]
    kind: Literal["text", "skills", "choice"]


def harness_version() -> str:
    """The installed strands-harness version, recorded in `cli.json`."""
    return metadata.version("strands-harness")


def file_sha256(path: Path) -> str:
    """sha256 of the config file's bytes, recorded in `cli.json`."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_config(path: Path) -> dict[str, Any]:
    """The config file as written (no defaults merged)."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"--config {path}: a portable config is a JSON object")
    return config


def declare(config_path: Path, tune: list[str]) -> tuple[dict[str, Marker], list[Tuned]]:
    """The space and the tuned paths: `instructions` (unless a `--tune` names it), then each `--tune` in order.

    Raises:
        ValueError: A path the config doesn't have, a text path that isn't a string, a bad choice list or one
            missing the config's current value, `skills` not given as one path, or two paths with one name.
    """
    raw = read_config(config_path)
    current = define_harness_agent_config(copy.deepcopy(raw))  # defaults merged: a choice may be left at one
    specs = [_split(spec) for spec in tune]
    if DEFAULT_TUNED not in {path for path, _ in specs}:
        specs.insert(0, (DEFAULT_TUNED, None))
    markers: dict[str, Marker] = {}
    tuned: list[Tuned] = []
    kind: Literal["text", "skills", "choice"]
    for text, options in specs:
        path = _path(text)
        name = text.replace(".", "_")
        if name in markers:
            raise ValueError(f"--tune {text}: another --tune path is also named {name!r}")
        if options is not None:
            value = _get(current, path, text)
            if value not in options:
                raise ValueError(f"--tune {text}: the config's value {value!r} must be one of the options")
            markers[name], kind = OneOf(*options, default=value), "choice"
        elif path == ("skills",):
            skills = raw.get("skills")
            if not isinstance(skills, str) or skills.startswith(("http://", "https://")):
                raise ValueError("--tune skills: the config's skills must be one local folder path")
            folder = Path(skills).expanduser()
            markers[name], kind = (
                SkillLibrary(folder if folder.is_absolute() else config_path.parent / folder),
                "skills",
            )
        else:
            if path == (DEFAULT_TUNED,) and DEFAULT_TUNED not in raw:
                raise ValueError(f'--config {config_path} has no "instructions"; add one (it may be empty) to tune it')
            value = _get(raw, path, text)
            if not isinstance(value, str):
                raise ValueError(
                    f"--tune {text} is a {type(value).__name__}, not text: give --tune {text}=JSON_LIST for a choice"
                )
            markers[name], kind = JsonField(config_path, _pointer(path)), "text"
        tuned.append(Tuned(name, path, kind))
    return markers, tuned


def warn_host_tools(config_path: Path) -> None:
    """Warn once when built-in tools that act on this machine are on and no sandbox is set."""
    config = define_harness_agent_config(read_config(config_path))
    tools = config["builtinTools"]
    names = list(tools) if isinstance(tools, (list, dict)) else []
    on = [tool for tool in _HOST_TOOLS if tool in names]
    if on and config["sandbox"] is None:
        logger.warning(
            "tools=<%s> | these built-in tools run on this machine with no sandbox, and concurrent trials share the "
            "working directory; set sandbox or builtinTools in the config to limit them",
            ", ".join(on),
        )


def build_task(config_path: Path, tuned: list[Tuned]) -> Callable[[BuildContext], Callable[[Case], Any]]:
    """`build_task` for `Optimizer`: a candidate's values set at their paths, a fresh harness per trial."""
    raw = read_config(config_path)
    base_dir = config_path.parent

    def build(ctx: BuildContext) -> Callable[[Case], Any]:
        config = copy.deepcopy(raw)
        for item in tuned:
            value = ctx.resolve(item.name)
            _set(config, item.path, str(value) if item.kind == "skills" else value)

        async def task(case: Case) -> dict[str, Any]:
            with tempfile.TemporaryDirectory(prefix="strands-evals-trial-") as folder:
                trial = _fresh_state(config, base_dir, Path(folder))
                agent = create_harness(**harness_agent_kwargs_from_config(trial, base_dir=base_dir))
                result = await agent.invoke_async(case.input)
            return {"output": str(result)}

        return task

    return build


def _fresh_state(config: dict[str, Any], base_dir: Path, folder: Path) -> dict[str, Any]:
    """`config` with its session in an empty folder and its memory in a copy, both under `folder`."""
    trial = copy.deepcopy(config)
    session = trial.get("session", True)
    if session is not False:
        trial["session"] = {**(session if isinstance(session, dict) else {}), "dir": str(folder / "session")}
    memory = trial.get("memory", True)
    if memory is not False:
        memory = dict(memory) if isinstance(memory, dict) else {}
        source = Path(memory["dir"]).expanduser() if "dir" in memory else Path(DEFAULT_MEMORY_DIR)
        if "dir" in memory and not source.is_absolute():
            source = base_dir / source  # the harness resolves a configured folder against the config's
        target = folder / "memory"
        if source.is_dir():
            shutil.copytree(source, target)
        memory["dir"] = str(target)
        trial["memory"] = memory
    return trial


def _split(spec: str) -> tuple[str, list[Any] | None]:
    """`PATH` or `PATH=JSON_LIST`."""
    path, sep, raw = spec.partition("=")
    if not path:
        raise ValueError(f"--tune must be PATH or PATH=JSON_LIST (got '{spec}')")
    if not sep:
        return path, None
    try:
        options = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"--tune {path}: the options must be a JSON list (got '{raw}')") from error
    if not isinstance(options, list) or not options:
        raise ValueError(f"--tune {path}: the options must be a non-empty JSON list (got '{raw}')")
    return path, options


def _path(text: str) -> tuple[str | int, ...]:
    """A dotted path's keys, digits as list indexes."""
    parts = text.split(".")
    if any(not part for part in parts):
        raise ValueError(f"--tune {text}: not a dotted path")
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _pointer(path: tuple[str | int, ...]) -> str:
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)


def _get(document: Any, path: tuple[str | int, ...], text: str) -> Any:
    for part in path:
        try:
            document = document[part]
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"--tune {text}: the config has nothing at that path") from None
    return document


def _set(document: Any, path: tuple[str | int, ...], value: Any) -> None:
    """Set `value` at `path`; a missing object on the way (a choice left at the harness's default) is created."""
    for part in path[:-1]:
        document = document.setdefault(part, {}) if isinstance(part, str) else document[part]
    document[path[-1]] = value
