"""The `--config` agent of `strands-evals optimize`: a strands-harness portable config.

The space comes from the config's paths (`instructions`, plus every `--tune`): a text field is a `JsonField` on the
config, a `skills` path a `SkillLibrary`, a choice a `OneOf` (a `JsonChoice` when a list or object is among the
options, e.g. tool sets). Each candidate's values are set at their paths in a copy
of the config held in memory; each trial builds a fresh harness from it, with its own empty session folder and a copy
of the configured memory folder, so no trial sees another's state. A `--rollout` runs each trial instead of the
default `invoke_async(case.input)`, given a factory that builds such a fresh harness.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
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
from ..optimization import BuildContext, Float, Int, JsonChoice, JsonField, Marker, OneOf, RolloutFn, SkillLibrary
from ..optimization.declaration.markers import _canonical_json
from ..optimization.execution.rollout import _call

logger = logging.getLogger(__name__)

DEFAULT_TUNED = "instructions"
_HOST_TOOLS = ("shell", "write", "edit")


@dataclass(frozen=True)
class Tuned:
    """One tuned config path: its declared name, its path as keys and indexes, and what it becomes."""

    name: str
    path: tuple[str | int, ...]
    kind: Literal["text", "skills", "choice", "number"]


@dataclass(frozen=True)
class _Range:
    """A `--tune PATH=int:LO..HI` / `float:LO..HI` range."""

    kind: Literal["int", "float"]
    low: float
    high: float


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


def declare(
    config_path: Path, tune: list[str], *, tune_instructions: bool = True
) -> tuple[dict[str, Marker], list[Tuned]]:
    """Declare a portable config's space: `instructions` (unless a `--tune` names it, or `tune_instructions` is
    False), then each `--tune` in order.

    Args:
        config_path: The portable config (JSON).
        tune: The `--tune` specs: `PATH` (a text field, or `skills`), `PATH=JSON_LIST` (a choice), or
            `PATH=int:LO..HI` / `PATH=float:LO..HI` (a number in an inclusive range).
        tune_instructions: Whether `instructions` is tuned without a `--tune` naming it (`--no-tune-instructions`).

    Returns:
        `(markers, tuned)`: the space, declared name -> marker (`JsonField`, `SkillLibrary`, `OneOf`, `Int`, or
        `Float`), and for
        each name where its value goes in the config (`Tuned`: name, path, kind), which `build_task` reads.

    Raises:
        ValueError: A path the config doesn't have, a text path that isn't a string, a bad choice list or one
            missing the config's current value, a bad range or one the config's value is outside of (or not a
            number of that type), an option or range value the harness rejects at that path, `skills` not given as
            one path, two paths with one name, or nothing to tune.
    """
    raw = read_config(config_path)
    current = define_harness_agent_config(copy.deepcopy(raw))  # defaults merged: a choice may be left at one
    specs = [_split(spec) for spec in tune]
    if tune_instructions and DEFAULT_TUNED not in {path for path, _ in specs}:
        specs.insert(0, (DEFAULT_TUNED, None))
    if not specs:
        raise ValueError("--no-tune-instructions with no --tune: nothing is left to tune")
    markers: dict[str, Marker] = {}
    tuned: list[Tuned] = []
    kind: Literal["text", "skills", "choice", "number"]
    for text, options in specs:
        path = _path(text)
        name = text.replace(".", "_")
        if name in markers:
            raise ValueError(f"--tune {text}: another --tune path is also named {name!r}")
        if isinstance(options, _Range):
            number = _number(text, options, _get(current, path, text))
            _check_accepted(raw, path, text, _samples(options, number.default))
            markers[name], kind = number, "number"
        elif options is not None:
            value = _get(current, path, text)
            marker = _choice(text, "the config's", options, value)
            _check_accepted(raw, path, text, options)
            markers[name], kind = marker, "choice"
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
                    f"--tune {text} is a {type(value).__name__}, not text: give --tune {text}=JSON_LIST for a "
                    f"choice, or {text}=int:LO..HI / {text}=float:LO..HI for a number"
                )
            markers[name], kind = JsonField(config_path, _pointer(path)), "text"
        tuned.append(Tuned(name, path, kind))
    return markers, tuned


def warn_host_tools(config_path: Path) -> None:
    """Warn once when built-in tools that act on this machine (`shell`, `write`, `edit`) are on and no sandbox is set.

    `builtinTools` as a list names exactly the tools that are on; as a map it edits the default set (every built-in
    tool): `"*"` picks the starting set, and each key turns its tool on (true or a config) or off (false).
    """
    config = define_harness_agent_config(read_config(config_path))
    tools = config["builtinTools"]
    if isinstance(tools, dict):  # edits to the default set: "*" picks the start, a key turns its tool on or off
        start = tools.get("*", True)
        on = [tool for tool in _HOST_TOOLS if tools.get(tool, start) is not False]
    else:  # a list: exactly these tools
        on = [tool for tool in _HOST_TOOLS if tool in tools]
    if on and config["sandbox"] is None:
        logger.warning(
            "tools=<%s> | these built-in tools run on this machine with no sandbox, and concurrent trials share the "
            "working directory; set sandbox or builtinTools in the config to limit them",
            ", ".join(on),
        )


def build_task(
    config_path: Path, tuned: list[Tuned], rollout_fn: RolloutFn | None = None
) -> Callable[[BuildContext], Callable[[Case], Any]]:
    """`build_task` for `Optimizer`: a candidate's values set at their paths, a fresh harness per trial.

    Without `rollout_fn`, each trial is `invoke_async(case.input)` on one harness. With it, each trial is
    `rollout_fn(make_harness, case)`, where every `make_harness()` call builds a fresh harness from the candidate's
    config, sharing the trial's session and memory folders; a sync `rollout_fn` runs in a worker thread. Every
    harness gets `callback_handler=None` unless the config's kwargs give one.

    Args:
        config_path: The portable config, read once; its folder is the harness's `base_dir`.
        tuned: Where each declared name's value goes in the config, from `declare`.
        rollout_fn: `--rollout`, or None for the default trial.

    Returns:
        The `build_task` for `Optimizer(build_task=..., space=...)`: called once per candidate with its
        `BuildContext`, it returns that candidate's `task(case)`.
    """
    raw = read_config(config_path)
    base_dir = config_path.parent

    def build(ctx: BuildContext) -> Callable[[Case], Any]:
        config = copy.deepcopy(raw)
        for item in tuned:
            value = ctx.resolve(item.name)
            _set(config, item.path, str(value) if item.kind == "skills" else value)

        async def task(case: Case) -> Any:
            with tempfile.TemporaryDirectory(prefix="strands-evals-trial-") as folder:
                trial = _fresh_state(config, base_dir, Path(folder))

                def make_harness() -> Any:
                    kwargs = harness_agent_kwargs_from_config(trial, base_dir=base_dir)
                    kwargs.setdefault("callback_handler", None)  # Strands' default prints every streamed answer
                    return create_harness(**kwargs)

                if rollout_fn is not None:
                    return await _call(rollout_fn, make_harness, case)
                result = await make_harness().invoke_async(case.input)
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


def _choice(text: str, whose: str, options: list[Any], value: Any) -> OneOf:
    """The marker of a `--tune PATH=JSON_LIST`: a `OneOf` over plain options, a `JsonChoice` when a list or
    object is among them.

    Args:
        text: The `--tune` path, for errors.
        whose: Whose value `value` is, for errors: `"the config's"` or `"the agent's"`.
        options: The parsed JSON list.
        value: The baseline (the config's or the agent call's value), which must be one of the options.

    Raises:
        ValueError: The baseline isn't one of the options (canonical JSON compared for a `JsonChoice`),
            or an option or the baseline isn't a JSON value.
    """
    if all(isinstance(option, (str, int, float, bool)) or option is None for option in options):
        if value not in options:
            raise ValueError(f"--tune {text}: {whose} value {value!r} must be one of the options")
        return OneOf(*options, default=value)
    try:
        held = _canonical_json(f"--tune {text}: {whose} value", value)
        if held not in {_canonical_json(f"--tune {text}: the options", option) for option in options}:
            raise ValueError(f"--tune {text}: {whose} value {value!r} must be one of the options")
        return JsonChoice(*options, default=value)
    except TypeError as error:
        raise ValueError(str(error)) from None


def _check_accepted(raw: dict[str, Any], path: tuple[str | int, ...], text: str, values: list[Any]) -> None:
    """Each of `values` set at `path` must pass the harness's own checks, so a value no trial could run (an option
    the field doesn't take, a fraction for an integer field) is a usage error here, before any spend."""
    for value in values:
        config = copy.deepcopy(raw)
        _set(config, path, value)
        try:
            define_harness_agent_config(config)
        except ValueError as error:
            raise ValueError(f"--tune {text}: the harness doesn't take {value!r} there: {error}") from None


def _samples(bounds: _Range, default: float) -> list[Any]:
    """The values to check a range with: for a float range first a fraction inside it, as most proposals are (so an
    integer field is reported on that), then both bounds and the default, as the marker gives them."""
    if bounds.kind == "int":
        return [int(bounds.low), int(bounds.high), default]
    step, inside = bounds.high - bounds.low, None
    for _ in range(60):
        step /= 2
        if not (bounds.low + step).is_integer():
            inside = bounds.low + step
            break
    return [*([inside] if inside is not None else []), float(bounds.low), float(bounds.high), float(default)]


def _number(text: str, bounds: _Range, value: Any) -> Int | Float:
    """`Int` / `Float` over `bounds`, its default the config's `value`, which must be a number of that type inside."""
    numeric = not isinstance(value, bool) and isinstance(value, int if bounds.kind == "int" else (int, float))
    if not numeric or not bounds.low <= value <= bounds.high:
        what = "an integer" if bounds.kind == "int" else "a number"
        raise ValueError(
            f"--tune {text}: the config's value {value!r} must be {what} in {bounds.low!r}..{bounds.high!r}"
        )
    if bounds.kind == "int":
        return Int(int(bounds.low), int(bounds.high), default=value)
    return Float(bounds.low, bounds.high, default=float(value))


def _range(path: str, raw: str) -> _Range:
    """`int:LO..HI` or `float:LO..HI` (inclusive, LO < HI)."""
    kind, _, bounds = raw.partition(":")
    parts = bounds.split("..")
    try:
        if len(parts) != 2 or "..." in bounds:
            raise ValueError
        parse = int if kind == "int" else float
        lo, hi = parse(parts[0]), parse(parts[1])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError
    except ValueError as error:
        raise ValueError(
            f"--tune {path}: a range is {kind}:LO..HI with two {'ints' if kind == 'int' else 'finite floats'} "
            f"(got '{raw}')"
        ) from error
    if lo >= hi:
        raise ValueError(f"--tune {path}: a range is {kind}:LO..HI with LO < HI (got '{raw}')")
    return _Range("int" if kind == "int" else "float", lo, hi)


def _split(spec: str) -> tuple[str, list[Any] | _Range | None]:
    """`PATH`, `PATH=JSON_LIST`, or `PATH=int:LO..HI` / `PATH=float:LO..HI`."""
    path, sep, raw = spec.partition("=")
    if not path:
        raise ValueError(f"--tune must be PATH, PATH=JSON_LIST, or PATH=int:LO..HI / float:LO..HI (got '{spec}')")
    if not sep:
        return path, None
    if raw.startswith(("int:", "float:")):
        return path, _range(path, raw)
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
            raise ValueError(
                f"--tune {text}: the config has nothing at that path; write the field in the config (its value "
                "there is the baseline) to tune it"
            ) from None
    return document


def _set(document: Any, path: tuple[str | int, ...], value: Any) -> None:
    """Set `value` at `path`; a missing object on the way (a choice left at the harness's default) is created."""
    for part in path[:-1]:
        document = document.setdefault(part, {}) if isinstance(part, str) else document[part]
    document[path[-1]] = value
