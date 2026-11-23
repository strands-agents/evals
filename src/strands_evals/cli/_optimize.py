"""What `strands-evals optimize` builds from its flags: the `Optimizer` and `cli.json`.

`cli.json` holds the inputs of `optimize run`, as given, so `resume RUN_DIR` rebuilds the same experiment, agent,
objective, constraints, search, selection, score model, and progress callback from the run folder alone. A flag that
takes an object takes `MODULE:ATTR` (`import_attr`), checked against the type the `Optimizer` expects.

A `.toml` run file holds the same flags: each key is a `run` flag (without `--`, `-` -> `_`), each value what the
flag takes; flags given on the command line win (`apply_run_file`).
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import inspect
import json
import logging
import math
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from ..experiment import Experiment
from ..optimization import (
    AtLeast,
    AtMost,
    Block,
    Constraint,
    FinalPick,
    IterationRecord,
    Maximize,
    Minimize,
    Objective,
    OptimizationResult,
    Optimizer,
    Pareto,
    Reflection,
    ScoreModel,
    Search,
    Selection,
    Space,
    Split,
    TunableFactory,
    WeightedSum,
)
from ..optimization.declaration.files import _write_text_atomically
from ._entrypoint import EntryPointError, _looks_like_path, import_attr, resolve_custom_evaluators

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

logger = logging.getLogger(__name__)

CLI_FILE = "cli.json"
# Bump SCHEMA_VERSION for a field that changes what runs (an older CLI ignoring it would resume a different run), so
# an older CLI refuses the file; a field that only records something stays at the same version, and an older CLI
# ignores it with a warning (`load_inputs`).
SCHEMA_VERSION = 1

_CONSTRAINT = re.compile(r"^\s*(?P<metric>[^<>=\s][^<>=]*?)\s*(?P<op><=|>=)\s*(?P<bound>\S+)\s*$")


@dataclass(frozen=True)
class Inputs:
    """What `optimize run` was given, as `cli.json` stores it; specs are kept as typed and parsed again on load.

    Exactly one agent route is set: `agent` (a `MODULE:ATTR` tunable factory), `build_task` with `space` (two
    `MODULE:ATTR`s), or `config` (a portable config's absolute path, with its sha256, its `--tune` specs, and the
    strands-harness version). `objective` and `constraints` hold specs or `MODULE:ATTR`s, as typed.
    """

    experiment_file: str
    agent: str | None = None
    build_task: str | None = None
    space: str | None = None
    config: str | None = None
    config_sha256: str | None = None
    tune: list[str] = field(default_factory=list)
    tune_instructions: bool = True
    harness_version: str | None = None
    custom_evaluators: list[str] = field(default_factory=list)
    rollout: str | None = None
    environment: str | None = None
    objective: list[str] = field(default_factory=list)
    pareto: bool = False
    constraints: list[str] = field(default_factory=list)
    reflection_model: str | None = None
    search: str | None = None
    selection: str | None = None
    score_model: str | None = None
    on_iteration: str | None = None


def save_inputs(run_dir: Path, inputs: Inputs) -> Path:
    """Write `<run_dir>/cli.json`; returns its path."""
    path = run_dir / CLI_FILE
    _write_text_atomically(path, json.dumps({"schema_version": SCHEMA_VERSION, **asdict(inputs)}, indent=2) + "\n")
    return path


def load_inputs(run_dir: Path) -> Inputs:
    """Read `<run_dir>/cli.json`.

    Raises:
        ValueError: The folder has no `cli.json` (it wasn't started by `optimize run`, or was stopped before any
            spend), or a newer schema. Unknown keys (from a newer strands-evals) are ignored with a warning.
    """
    path = run_dir / CLI_FILE
    if not path.is_file():
        events = run_dir / "events.jsonl"
        if events.is_file() and '"event": "trial"' not in events.read_text(encoding="utf-8"):
            raise ValueError(
                f"{run_dir} has no {CLI_FILE}: the run was stopped before it was written, so nothing was spent; "
                "start a new run"
            )
        raise ValueError(
            f"{run_dir} has no {CLI_FILE}: it wasn't started by `strands-evals optimize run`; "
            "resume it from Python with Optimizer.resume"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.pop("schema_version", None)
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path} has schema_version {version!r}; this strands-evals reads {SCHEMA_VERSION}")
    known = {f.name for f in fields(Inputs)}
    unknown = sorted(set(data) - known)
    if unknown:  # written by a newer strands-evals: what it added is ignored, the rest still resumes
        logger.warning("file=<%s>, keys=<%s> | ignoring cli.json keys this strands-evals doesn't know", path, unknown)
    return Inputs(**{key: value for key, value in data.items() if key in known})


def read_run_json(run_dir: Path) -> dict[str, Any]:
    """`<run_dir>/run.json`; a folder without one isn't a run folder (`ValueError`)."""
    path = run_dir / "run.json"
    if not path.is_file():
        raise ValueError(f"{run_dir} is not a run folder: it has no run.json")
    return dict(json.loads(path.read_text(encoding="utf-8")))


@contextmanager
def working_directory(path: str | Path) -> Iterator[None]:
    """Run the block in `path`, as the run did, so relative agent specs and file markers resolve alike."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


# Parsing flags; a bad value raises `ValueError` (exit 2).


def is_objective_spec(spec: str) -> bool:
    """`maximize:...` / `minimize:...`, or a value without `:` (a malformed spec); anything else is `MODULE:ATTR`."""
    return spec.partition(":")[0] in ("maximize", "minimize") or ":" not in spec


def parse_objective(spec: str) -> Maximize | Minimize:
    """`maximize:METRIC[:weight=X][:scale=X]` or `minimize:...`."""
    direction, _, rest = spec.partition(":")
    parts = rest.split(":") if rest else []
    if direction not in ("maximize", "minimize") or not parts or not parts[0]:
        raise ValueError(
            f"--objective must be maximize:METRIC or minimize:METRIC, optionally :weight=X / :scale=X (got '{spec}')"
        )
    options: dict[str, float] = {}
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if not sep or key not in ("weight", "scale") or key in options:
            raise ValueError(f"--objective option '{part}' must be weight=X or scale=X, once each")
        try:
            options[key] = float(value)
        except ValueError as error:
            raise ValueError(f"--objective {key} must be a number (got '{value}')") from error
    kind = Maximize if direction == "maximize" else Minimize
    try:
        return kind(parts[0], **options)
    except (TypeError, ValueError) as error:
        raise ValueError(f"--objective '{spec}': {error}") from error


def parse_constraints(spec: str) -> list[Constraint]:
    """A `--constraint`: `METRIC<=X` / `METRIC>=X`, or `MODULE:ATTR` naming a constraint or a list of them (a value
    with `<`, `>`, or `=` is a spec)."""
    if any(char in spec for char in "<>=") or ":" not in spec:
        return [parse_constraint(spec)]
    target = load_hook(spec, "--constraint", (AtMost, AtLeast, list, tuple), "a constraint (AtMost or AtLeast)")
    targets = list(target) if isinstance(target, (list, tuple)) else [target]
    wrong = [type(item).__name__ for item in targets if not isinstance(item, (AtMost, AtLeast))]
    if wrong:
        raise EntryPointError(f"--constraint '{spec}' holds a {wrong[0]}, not a constraint (AtMost or AtLeast)")
    return targets


def parse_constraint(spec: str) -> Constraint:
    """`METRIC<=X` (`AtMost`) or `METRIC>=X` (`AtLeast`)."""
    match = _CONSTRAINT.match(spec)
    if match is None:
        raise ValueError(f"--constraint must be METRIC<=X or METRIC>=X (got '{spec}')")
    try:
        bound = float(match["bound"])
    except ValueError as error:
        raise ValueError(f"--constraint bound must be a number (got '{match['bound']}')") from error
    kind = AtMost if match["op"] == "<=" else AtLeast
    return kind(match["metric"], bound)


def parse_split(spec: str) -> Split:
    """`TRAIN,VALIDATION,TEST` shares (a test share of 0 means no test cases), or `MODULE:ATTR` naming a `Split`
    (case lists, a split seed)."""
    if ":" in spec:
        return load_hook(spec, "--split", Split, "a Split")  # type: ignore[no-any-return]
    parts = spec.split(",")
    try:
        shares = [float(part) for part in parts]
    except ValueError as error:
        raise ValueError(f"--split must be three numbers, e.g. 0.4,0.3,0.3 (got '{spec}')") from error
    if len(shares) != 3:
        raise ValueError(f"--split must be three numbers, e.g. 0.4,0.3,0.3 (got '{spec}')")
    train, validation, test = shares
    try:
        return Split(train=train, validation=validation, test=test if test else None)
    except (TypeError, ValueError) as error:
        raise ValueError(f"--split '{spec}': {error}") from error


def parse_agent_failure_score(spec: str) -> Literal["worst"] | float | None:
    """`worst`, `none` (missing data), or a number."""
    if spec in ("worst", "none"):
        return None if spec == "none" else "worst"
    try:
        return float(spec)
    except ValueError as error:
        raise ValueError(f"--agent-failure-score must be 'worst', 'none', or a number (got '{spec}')") from error


# Building what the Optimizer takes.


def load_experiment(inputs: Inputs) -> Experiment:
    """The experiment file with the custom evaluators registered first, as `strands-evals run` loads it."""
    custom = resolve_custom_evaluators(inputs.custom_evaluators)
    return Experiment.from_file(inputs.experiment_file, custom_evaluators=custom)


def load_hook(spec: str, flag: str, kind: Any, what: str) -> Any:
    """Import `flag`'s `MODULE:ATTR` target and check its type.

    Args:
        spec: `pkg.module:attr` or `path/to/file.py:attr`, resolved by `import_attr`.
        flag: The flag's name, for the error (`"--search"`).
        kind: What the target must be an instance of: a type, a tuple of types, or a union; None means callable.
        what: The expected kind in words, for the error (`"a Search"`).

    Returns:
        The imported target.

    Raises:
        EntryPointError: `spec` can't be imported, or the target isn't of `kind` (a `ValueError`, so exit 2).
    """
    target = import_attr(spec)
    if not (callable(target) if kind is None else isinstance(target, kind)):
        raise EntryPointError(f"{flag} '{spec}' is a {type(target).__name__}, not {what}")
    return target


_AGENT_FILE_SUFFIXES = (".py", ".ts", ".mts", ".cts", ".js", ".mjs", ".cjs")


def is_agent_file(spec: str) -> bool:
    """Whether `--agent` names a file (`agent.py`, `./agent.py:agent`, `agent.ts`), loaded by `_agent_file.load`,
    rather than a dotted `module:attr`. Doesn't import strands-harness."""
    if spec.endswith(_AGENT_FILE_SUFFIXES):
        return True
    module, sep, attr = spec.rpartition(":")
    return bool(sep and module and attr and (_looks_like_path(module) or module.endswith(_AGENT_FILE_SUFFIXES)))


def load_factory(spec: str) -> TunableFactory:
    """The `--agent` target, which must be a `TunableFactory`."""
    target = import_attr(spec)
    if not isinstance(target, TunableFactory):
        raise EntryPointError(
            f"--agent '{spec}' is a {type(target).__name__}, not a tunable factory: declare what may change "
            "with @tunable(...) or tunable(fn, inject={...}) from strands_evals.optimization"
        )
    return target


def build_objective(inputs: Inputs) -> Objective | None:
    """None (the default objective), one objective, a `WeightedSum`, or with `--pareto` a `Pareto`; or the whole
    objective from one `--objective MODULE:ATTR`."""
    hooks = [spec for spec in inputs.objective if not is_objective_spec(spec)]
    if hooks:
        try:
            objective = load_hook(
                hooks[0], "--objective", Objective, "an objective (Maximize, Minimize, WeightedSum, Pareto)"
            )
        except EntryPointError as error:
            raise ValueError(
                f"--objective must be maximize:METRIC or minimize:METRIC (optionally :weight=X / :scale=X), "
                f"or MODULE:ATTR naming an objective: {error}"
            ) from error
        if len(inputs.objective) > 1 or inputs.pareto:
            raise ValueError(
                "--objective MODULE:ATTR names the whole objective: give it alone, with no other --objective, --pareto"
            )
        return objective  # type: ignore[no-any-return]
    objectives = [parse_objective(spec) for spec in inputs.objective]
    if inputs.pareto:
        if len(objectives) < 2:
            raise ValueError("--pareto needs at least two --objective")
        return Pareto(*objectives)
    if not objectives:
        return None
    return objectives[0] if len(objectives) == 1 else WeightedSum(*objectives)


def build_search(space: Space, inputs: Inputs) -> Search | None:
    """`--search`'s `Search`; else the default search, with `Reflection(model=...)` on the text keys when
    `--reflection-model` is given."""
    if inputs.search is not None:
        if inputs.reflection_model is not None:
            raise ValueError("--search and --reflection-model: set the model in the search's Reflection(model=...)")
        return load_hook(inputs.search, "--search", Search, "a Search")  # type: ignore[no-any-return]
    reflection_model = inputs.reflection_model
    if reflection_model is None:
        return None
    names = list(dict.fromkeys(info.name for info in space.params() if info.kind == "text"))
    if not names:
        raise ValueError("--reflection-model is set, but nothing tuned is text")
    return Search([Block(Reflection(model=reflection_model), *names)])


def agent_arguments(inputs: Inputs) -> dict[str, Any]:
    """Build the `Optimizer` / `Optimizer.resume` arguments that `cli.json` records, shared by `run` and `resume`.

    Args:
        inputs: The run's inputs, from the flags (`run`) or `cli.json` (`resume`).

    Returns:
        Keyword arguments for `Optimizer` and `Optimizer.resume`: always `experiment`, `objective`, `constraints`,
        `selection`, `score_model`, `on_iteration`, `search`; then the agent route's own: `factory`, `rollout_fn`,
        `environment` for `--agent`; `build_task`, `space` for `--build-task` and for `--config` (whose
        `build_task` is `_harness.build_task`). Settings `run.json` stores (split, final pick, budget, ...) aren't
        here: `run` adds them, `resume` reads them from `run.json`.

    Raises:
        ValueError: A spec or hook is invalid (`EntryPointError` included), or `--config`'s `--tune` is.
        TypeError: The space holds something that isn't a marker.
    """
    rollout = (
        load_hook(inputs.rollout, "--rollout", None, "a rollout_fn(agent_factory, case)") if inputs.rollout else None
    )
    common: dict[str, Any] = {
        "experiment": load_experiment(inputs),
        "objective": build_objective(inputs),
        "constraints": [c for spec in inputs.constraints for c in parse_constraints(spec)],
        "selection": load_hook(inputs.selection, "--selection", Selection, "a Selection") if inputs.selection else None,
        "score_model": (
            load_hook(inputs.score_model, "--score-model", ScoreModel, "a ScoreModel") if inputs.score_model else None
        ),
        "on_iteration": progress(inputs.on_iteration),
    }
    if inputs.config is not None:
        from . import _harness  # needs strands-harness, in the `optimization` extra

        config = Path(inputs.config)
        markers, tuned = _harness.declare(config, inputs.tune, tune_instructions=inputs.tune_instructions)
        _harness.warn_host_tools(config)
        return {
            **common,
            "build_task": _harness.build_task(config, tuned, rollout),
            "space": markers,
            "search": build_search(Space(markers), inputs),
        }
    if inputs.build_task is not None:
        assert inputs.space is not None
        build = load_hook(inputs.build_task, "--build-task", None, "a function build_task(ctx) returning a task")
        space = load_hook(inputs.space, "--space", Mapping, "a mapping of declared names to markers")
        return {**common, "build_task": build, "space": space, "search": build_search(Space(space), inputs)}
    assert inputs.agent is not None
    if is_agent_file(inputs.agent):
        from . import _agent_file  # needs strands-harness, in the `optimization` extra

        factory = _agent_file.load(inputs.agent, inputs.tune, tune_instructions=inputs.tune_instructions)
    else:
        factory = load_factory(inputs.agent)
    environment = inputs.environment
    return {
        **common,
        "factory": factory,
        "rollout_fn": rollout,
        "environment": load_hook(environment, "--environment", None, "an environment(case)") if environment else None,
        "search": build_search(factory.space, inputs),
    }


def progress(spec: str | None) -> Callable[[IterationRecord], Any]:
    """The `on_iteration` callback: the CLI's progress line, then `--on-iteration`'s callback (awaited when async)."""
    if spec is None:
        return print_iteration
    user = load_hook(spec, "--on-iteration", None, "a callable taking an IterationRecord")

    async def on_iteration(record: IterationRecord) -> None:
        print_iteration(record)
        result = user(record)
        if inspect.isawaitable(result):
            await result

    return on_iteration


def load_final_pick(spec: str) -> FinalPick:
    """`--final-pick`'s `FinalPick`."""
    return load_hook(spec, "--final-pick", FinalPick, "a FinalPick")  # type: ignore[no-any-return]


def check_config_unchanged(inputs: Inputs) -> None:
    """Refuse to resume a `--config` run whose config file changed since the run: a change to a key that isn't
    tuned would otherwise mix trials of two agents (a tuned key's change is caught by `Optimizer.resume`). Only the
    config file itself is hashed, not folders it names (skills, memory); a tuned skill's `SKILL.md` is checked by
    `Optimizer.resume` with the space.

    Raises:
        ValueError: The config file is gone, or its sha256 isn't the one `cli.json` recorded.
    """
    if inputs.config is None:
        return
    from . import _harness

    config = Path(inputs.config)
    if not config.is_file():
        raise ValueError(f"--config {config} no longer exists; restore it to resume, or start a new run")
    if _harness.file_sha256(config) != inputs.config_sha256:
        raise ValueError(
            f"the config file {config} changed since the run, so resumed trials would run a different agent; "
            "restore it to resume, or start a new run"
        )


def check_harness_version(inputs: Inputs) -> None:
    """Warn when the installed strands-harness isn't the run's: a 0.x release may rename config keys."""
    if inputs.config is None:
        return
    from . import _harness

    installed = _harness.harness_version()
    if installed != inputs.harness_version:
        logger.warning(
            "run=<%s>, now=<%s> | strands-harness version differs from the run's; a 0.x release may change config keys",
            inputs.harness_version,
            installed,
        )


def print_iteration(record: IterationRecord) -> None:
    """One progress line per iteration, on stderr."""
    print(  # noqa: T201
        f"strands-evals optimize: iteration {record.iteration} ({', '.join(record.blocks)}): "
        f"{len(record.proposals)} proposals, {record.rollouts_spent} rollouts",
        file=sys.stderr,
    )


async def drive(optimizer: Optimizer, on_folder: Callable[[Path], None]) -> OptimizationResult:
    """Run `optimizer`, calling `on_folder` with its run folder as soon as it exists (before the baseline is
    scored), then wait for the result."""
    task = asyncio.ensure_future(optimizer.run_async())
    while optimizer.run_dir is None and not task.done():
        await asyncio.sleep(0)
    if optimizer.run_dir is not None:
        on_folder(optimizer.run_dir)
    return await task


# The .toml run file.

RUN_FILE_SUFFIX = ".toml"
# Each route's keys: the route flag(s), then the agent keys the CLI accepts with that route.
_ROUTE_KEYS = {
    "agent": ("agent", "rollout", "environment"),
    "build_task": ("build_task", "space"),
    "config": ("config", "tune", "tune_instructions", "rollout"),
}
_AGENT_KEYS = frozenset(key for keys in _ROUTE_KEYS.values() for key in keys)
_PATH_KEYS = ("experiment", "config", "base_dir")
# Flags that exclude each other: given on the command line, one replaces the file's value of the other.
_EXCLUSIVE = (
    ("search", "reflection_model"),
    ("reflection_model", "search"),
    ("final_pick", "n_finalists"),
    ("n_finalists", "final_pick"),
)
_HOOK_KEYS = (
    "agent",
    "build_task",
    "space",
    "rollout",
    "environment",
    "custom_evaluator",
    "search",
    "selection",
    "score_model",
    "final_pick",
    "on_iteration",
)
_OUTPUT_FLAGS = ("format", "verbose", "debug", "help")


def apply_run_file(path: Path, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fill `args` from the `.toml` run file at `path`, for every flag not given on the command line.

    Keys are the `run` flags without `--` (`-` -> `_`), so the experiment file is `experiment` (`--experiment`); the
    output flags (`--json`, `--rich`, `-v`, `--debug`) aren't keys. A repeatable flag's value is a list, `--pareto`'s a
    boolean, every other flag's a string or number. The agent route may sit in an `[agent]` section (`factory` for
    `--agent`, then `build_task`, `space`, `config`, `tune`, `tune_instructions`, `rollout`, `environment`).
    `objective`, `constraint`, and `tune` take the flag's strings or named `[key.NAME]` tables (not one `[key]`,
    `[[key]]`, or inline tables); a table's fields are `{maximize | minimize = METRIC, weight?, scale?}`,
    `{metric = METRIC, at_most | at_least = X}`, `{path = PATH, choices? | int? | float?}`, and each becomes the
    flag's spec. A
    flag on the command line replaces the file's value (for a repeatable flag, its whole list). A route flag on the
    command line
    (`--agent`, `--build-task` / `--space`, `--config`) chooses the route: the file's agent keys that don't fit it
    are dropped, those that do are kept (with `--agent`: `rollout`, `environment`; with `--build-task`: the other
    of `build_task` / `space`; with `--config`, or `--agent` naming an agent file: `tune`, `tune_instructions`,
    `rollout`). Likewise a flag that excludes
    another replaces the file's value of it (`--search` / `reflection_model`, `--final-pick` / `n_finalists`). Paths
    (`experiment`, `config`, `base_dir`) and path-form `MODULE:ATTR`s resolve against the file's folder; dotted
    modules import from the working directory, as on the command line.

    Args:
        path: The run file.
        parser: The `run` subparser; its arguments give the file's keys and their types.
        args: The parsed command line, filled in place: a flag already given there is left as is.

    Raises:
        ValueError: The file can't be read or parsed, has an unknown key, a value of the wrong type, a misplaced
            or bad section, or no `experiment`.
    """
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"run file {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        hint = next((h for text, h in _TOML_HINTS if text in str(error)), "")
        raise ValueError(f"run file {path} isn't valid TOML: {error}{hint}") from error
    actions = _run_file_actions(parser)
    data = _flatten_sections(path, data, actions)
    unknown = sorted(set(data) - set(actions))
    if unknown:
        raise ValueError(
            f"run file {path}: unknown key {unknown[0]!r}{_did_you_mean(unknown[0], actions)}; keys are the run "
            f"flags without --, e.g. {', '.join(list(actions)[:4])}"
        )
    if "experiment" not in data and args.experiment is None:
        raise ValueError(f'run file {path} has no experiment = "<experiment.json>" (or pass --experiment)')
    base = path.resolve().parent
    on_command_line = {action.dest for action in actions.values() if getattr(args, action.dest) != action.default}
    for flag, exclusive in _EXCLUSIVE:  # a flag given here replaces the file's value of the flag it excludes
        if flag in on_command_line:
            data.pop(exclusive, None)
    route = _command_line_route(on_command_line)
    if route is not None:  # the command line chose the route: drop the file's agent keys that don't fit it
        fits = set(_ROUTE_KEYS[route])
        if route == "agent" and is_agent_file(args.agent):  # an agent file takes --tune, as --config does
            fits |= {"tune", "tune_instructions"}
        data = {key: value for key, value in data.items() if key not in _AGENT_KEYS or key in fits}
    for key, value in data.items():
        action = actions[key]
        if action.dest in on_command_line:
            continue
        if key in _TABLES and isinstance(value, list):
            value = [_table_entry(path, key, i, entry, actions) for i, entry in enumerate(value)]
        value = _run_file_value(path, key, action, value)
        setattr(args, action.dest, _resolve_against(base, action.dest, value))


def _command_line_route(on_command_line: set[str]) -> str | None:
    """The route the command line's route flags choose (`agent`, `build_task`, `config`), or None; with several, the
    first, and the route check reports the conflict."""
    if "agent" in on_command_line:
        return "agent"
    if on_command_line & {"build_task", "space"}:
        return "build_task"
    return "config" if "config" in on_command_line else None


def _did_you_mean(name: str, options: Any) -> str:
    """`; did you mean 'x'?` for the closest of `options` (a plural, a typo), else nothing."""
    close = difflib.get_close_matches(name, list(options), n=1, cutoff=0.6)
    return f"; did you mean {close[0]!r}?" if close else ""


def _run_file_actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Run-file key -> the `run` argument it sets."""
    actions: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if action.dest in _OUTPUT_FLAGS:
            continue
        if not action.option_strings:  # the positional names the run file itself; `experiment` is --experiment
            continue
        flag = next(option for option in action.option_strings if option.startswith("--"))
        actions[flag[2:].replace("-", "_")] = action
    return actions


def _flatten_sections(path: Path, data: dict[str, Any], actions: Mapping[str, argparse.Action]) -> dict[str, Any]:
    """`data` with an `[agent]` section's keys moved to the top level (`factory` -> `agent`), and every `objective` /
    `constraint` / `tune` value as a list of entries (strings or tables)."""
    data = dict(data)
    section = data.get("agent")
    if isinstance(section, dict):
        del data["agent"]
        for name, value in section.items():
            if name == "agent":
                raise ValueError(f'run file {path}: in [agent], the factory is factory = "...", not agent = "..."')
            if name not in _AGENT_SECTION:
                if name in actions:
                    raise ValueError(
                        f"run file {path}: [agent] holds {name!r}: every key after the [agent] header belongs to it, "
                        "so move the run's other keys above it"
                    )
                raise ValueError(
                    f"run file {path}: [agent] has no key {name!r}{_did_you_mean(name, _AGENT_SECTION)}; "
                    f"it takes {', '.join(_AGENT_SECTION)}"
                )
            key = "agent" if name == "factory" else name
            if key in data:
                raise ValueError(f"run file {path}: {key} is given both in [agent] and at the top level")
            data[key] = value
    for key in _TABLES:
        if key in data:
            data[key] = _table_entries(path, key, data[key])
    return data


def _table_entries(path: Path, key: str, value: Any) -> Any:
    """`key = ["...", ...]` (the flag's strings) as given; named `[key.NAME]` tables as their values in file order
    (the names are labels only). One `[key]` table with the fields directly under it, `[[key]]` sections, and
    inline tables in the list are refused, so each key has one string form and one table form."""
    if isinstance(value, list):
        if any(isinstance(entry, dict) for entry in value):
            raise ValueError(
                f"run file {path}: {key} takes strings (the --{key.replace('_', '-')} specs) or named [{key}.NAME] "
                f"tables; [[{key}]] sections and inline tables aren't accepted"
            )
        return value
    if not isinstance(value, dict):
        return value
    fields_here = sorted(name for name, entry in value.items() if not isinstance(entry, dict))
    if fields_here:
        raise ValueError(
            f"run file {path}: [{key}] holds {fields_here[0]!r} directly; give each {key} a named table, "
            f"[{key}.NAME], with its fields under it (even a single one)"
        )
    return list(value.values())


def _table_entry(path: Path, key: str, index: int, entry: Any, actions: Mapping[str, argparse.Action]) -> Any:
    """An `objective` / `constraint` / `tune` table as the flag's spec string; a string entry as is."""
    if not isinstance(entry, dict):
        return entry
    where = f"run file {path}: {key}[{index}]"
    misplaced = sorted(set(entry) & set(actions))
    if misplaced:
        raise ValueError(
            f"{where} holds {misplaced[0]!r}: every key after a [{key}] or [{key}.NAME] header belongs to that {key}, "
            "so move the run's other keys above the first [...] header"
        )
    allowed = _TABLES[key]
    unknown = sorted(set(entry) - set(allowed))
    if unknown:
        raise ValueError(
            f"{where}: unknown key {unknown[0]!r}{_did_you_mean(unknown[0], allowed)}; "
            f"a {key} takes {', '.join(allowed)}"
        )
    return _TABLE_SPECS[key](where, entry)


def _objective_spec(where: str, entry: dict[str, Any]) -> str:
    """`{maximize | minimize = METRIC, weight?, scale?}` -> `maximize:METRIC[:weight=X][:scale=X]`."""
    direction = _one_of(where, entry, ("maximize", "minimize"), str, '"METRIC"')
    spec = f"{direction}:{entry[direction]}"
    for option in ("weight", "scale"):
        if option in entry:
            spec += f":{option}={_number(where, entry, option)}"
    return spec


def _constraint_spec(where: str, entry: dict[str, Any]) -> str:
    """`{metric = METRIC, at_most | at_least = X}` -> `METRIC<=X` / `METRIC>=X`."""
    if not isinstance(entry.get("metric"), str):
        raise ValueError(f'{where} needs metric = "METRIC"')
    bound = _one_of(where, entry, ("at_most", "at_least"), (int, float), "X")
    return f"{entry['metric']}{'<=' if bound == 'at_most' else '>='}{_number(where, entry, bound)}"


def _tune_spec(where: str, entry: dict[str, Any]) -> str:
    """`{path = PATH}` -> `PATH` (a text field); with `choices = [...]` -> `PATH=JSON_LIST` (a choice); with
    `int = [LO, HI]` / `float = [LO, HI]` -> `PATH=int:LO..HI` / `PATH=float:LO..HI` (a number)."""
    if not isinstance(entry.get("path"), str):
        raise ValueError(f'{where} needs path = "PATH"')
    given = [name for name in ("choices", "int", "float") if name in entry]
    if len(given) > 1:
        raise ValueError(f"{where}: give one of choices, int, float (got {', '.join(given)})")
    if not given:
        return str(entry["path"])
    if given[0] == "choices":
        if not isinstance(entry["choices"], list):
            raise ValueError(f"{where}: choices must be a list")
        return f"{entry['path']}={json.dumps(entry['choices'])}"
    kind, bounds = given[0], entry[given[0]]
    numbers = isinstance(bounds, list) and len(bounds) == 2
    if not numbers or any(isinstance(b, bool) or not isinstance(b, (int, float)) for b in bounds):
        raise ValueError(f"{where}: {kind} must be [LO, HI], two numbers")
    if kind == "int" and not all(isinstance(b, int) for b in bounds):
        raise ValueError(f"{where}: int = {bounds} needs two integers (float = [LO, HI] for a number with a fraction)")
    if not all(math.isfinite(b) for b in bounds) or bounds[0] >= bounds[1]:
        raise ValueError(f"{where}: {kind} = {bounds} needs two finite numbers with LO < HI")
    return f"{entry['path']}={kind}:{bounds[0]}..{bounds[1]}"


def _one_of(where: str, entry: dict[str, Any], names: tuple[str, ...], kind: Any, placeholder: str) -> str:
    """The one of `names` that `entry` has, holding a value of `kind`."""
    present = [name for name in names if name in entry]
    if len(present) != 1 or isinstance(entry[present[0]], bool) or not isinstance(entry[present[0]], kind):
        raise ValueError(f"{where} needs one of {' or '.join(f'{name} = {placeholder}' for name in names)}")
    return present[0]


def _number(where: str, entry: dict[str, Any], name: str) -> int | float:
    value = entry[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {name} must be a number")
    return value


_AGENT_SECTION = ("factory", "build_task", "space", "config", "tune", "tune_instructions", "rollout", "environment")
_TOML_HINTS = (
    ("Cannot overwrite a value", " (a key given both as a value and as a [section]: use one form)"),
    ("immutable namespace", " ([[key]] sections aren't accepted: use the key's strings or [key.NAME] tables)"),
    (
        "twice",
        " (a key given twice: as both `key = ...` and a [key] table, or as two [key] tables; use one form, and "
        "[key.NAME] tables for several)",
    ),
)
_TABLES: dict[str, tuple[str, ...]] = {
    "objective": ("maximize", "minimize", "weight", "scale"),
    "constraint": ("metric", "at_most", "at_least"),
    "tune": ("path", "choices", "int", "float"),
}
_TABLE_SPECS: dict[str, Callable[[str, dict[str, Any]], str]] = {
    "objective": _objective_spec,
    "constraint": _constraint_spec,
    "tune": _tune_spec,
}


def _run_file_value(path: Path, key: str, action: argparse.Action, value: Any) -> Any:
    """`value` checked against what the flag takes, converted as argparse would convert it."""
    where = f"run file {path}: {key}"
    if isinstance(action, argparse._AppendAction):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{where} must be a list of strings, as the flag is repeatable")
        return value
    if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be true or false")
        return value
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{where} must be a string or a number")
    if action.type in (int, float):
        if isinstance(value, str) or (action.type is int and not isinstance(value, int)):
            raise ValueError(f"{where} must be {'an integer' if action.type is int else 'a number'}")
        return action.type(value)
    return str(value)


def _resolve_against(base: Path, dest: str, value: Any) -> Any:
    """A path or path-form `MODULE:ATTR` from the run file, made absolute against the file's folder."""
    if dest in _PATH_KEYS:
        target = Path(value).expanduser()
        return str(target if target.is_absolute() else base / target)
    if dest == "agent" and value.endswith(_AGENT_FILE_SUFFIXES):  # an agent file named without :ATTR
        target = Path(value).expanduser()
        return str(target if target.is_absolute() else base / target)
    if dest in _HOOK_KEYS:
        return [_hook_against(base, v) for v in value] if isinstance(value, list) else _hook_against(base, value)
    if dest == "objective_specs":
        return [spec if is_objective_spec(spec) else _hook_against(base, spec) for spec in value]
    if dest == "constraint_specs":
        return [spec if any(c in spec for c in "<>=") else _hook_against(base, spec) for spec in value]
    if dest == "split":
        return _hook_against(base, value) if ":" in value else value
    return value


def _hook_against(base: Path, spec: str) -> str:
    """`spec` with a path-form module made absolute against `base`; a dotted module is left as is."""
    module, sep, attr = spec.rpartition(":")
    if not sep or not _looks_like_path(module):
        return spec
    target = Path(module).expanduser()
    return f"{target if target.is_absolute() else base / target}:{attr}"
