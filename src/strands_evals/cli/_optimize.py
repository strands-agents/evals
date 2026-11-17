"""What `strands-evals optimize` builds from its flags: the `Optimizer`, `cli.json`, and applying a result.

`cli.json` holds the inputs of `optimize run`, as given, so `resume RUN_DIR` and `apply RUN_DIR` rebuild the same
experiment, agent, objective, constraints, and search from the run folder alone.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..experiment import Experiment
from ..optimization import (
    AtLeast,
    AtMost,
    Block,
    Constraint,
    IterationRecord,
    Maximize,
    Minimize,
    Objective,
    OptimizationResult,
    Optimizer,
    Pareto,
    Reflection,
    Search,
    Split,
    TunableFactory,
    WeightedSum,
)
from ..optimization.declaration.files import _write_text_atomically
from ._entrypoint import EntryPointError, import_attr, resolve_custom_evaluators

CLI_FILE = "cli.json"
SCHEMA_VERSION = 1

_CONSTRAINT = re.compile(r"^\s*(?P<metric>[^<>=\s][^<>=]*?)\s*(?P<op><=|>=)\s*(?P<bound>\S+)\s*$")


@dataclass(frozen=True)
class Inputs:
    """What `optimize run` was given, as `cli.json` stores it; specs are kept as typed and parsed again on load."""

    experiment_file: str
    agent: str
    custom_evaluators: list[str] = field(default_factory=list)
    rollout: str | None = None
    environment: str | None = None
    objective: list[str] = field(default_factory=list)
    pareto: bool = False
    constraints: list[str] = field(default_factory=list)
    reflection_model: str | None = None


def save_inputs(run_dir: Path, inputs: Inputs) -> Path:
    """Write `<run_dir>/cli.json`; returns its path."""
    path = run_dir / CLI_FILE
    _write_text_atomically(path, json.dumps({"schema_version": SCHEMA_VERSION, **asdict(inputs)}, indent=2) + "\n")
    return path


def load_inputs(run_dir: Path) -> Inputs:
    """Read `<run_dir>/cli.json`.

    Raises:
        ValueError: The folder has no `cli.json` (it wasn't started by `optimize run`), or a newer schema.
    """
    path = run_dir / CLI_FILE
    if not path.is_file():
        raise ValueError(
            f"{run_dir} has no {CLI_FILE}: it wasn't started by `strands-evals optimize run`; "
            "resume it from Python with Optimizer.resume"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.pop("schema_version", None)
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path} has schema_version {version!r}; this strands-evals reads {SCHEMA_VERSION}")
    return Inputs(**data)


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
    """`TRAIN,VALIDATION,TEST` shares; a test share of 0 means no test cases."""
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
    """None (the default objective), one objective, a `WeightedSum`, or with `--pareto` a `Pareto`."""
    objectives = [parse_objective(spec) for spec in inputs.objective]
    if inputs.pareto:
        if len(objectives) < 2:
            raise ValueError("--pareto needs at least two --objective")
        return Pareto(*objectives)
    if not objectives:
        return None
    return objectives[0] if len(objectives) == 1 else WeightedSum(*objectives)


def build_search(factory: TunableFactory, reflection_model: str | None) -> Search | None:
    """The default search, with `Reflection(model=...)` on the text keys when `--reflection-model` is given."""
    if reflection_model is None:
        return None
    names = list(dict.fromkeys(info.name for info in factory.space.params() if info.kind == "text"))
    if not names:
        raise ValueError("--reflection-model is set, but the factory declares no text to rewrite")
    return Search([Block(Reflection(model=reflection_model), *names)])


def agent_arguments(inputs: Inputs) -> dict[str, Any]:
    """What `Optimizer` and `Optimizer.resume` take about the agent and the goal, built from `inputs`."""
    factory = load_factory(inputs.agent)
    return {
        "experiment": load_experiment(inputs),
        "factory": factory,
        "rollout_fn": import_attr(inputs.rollout) if inputs.rollout else None,
        "environment": import_attr(inputs.environment) if inputs.environment else None,
        "objective": build_objective(inputs),
        "constraints": [parse_constraint(spec) for spec in inputs.constraints],
        "search": build_search(factory, inputs.reflection_model),
    }


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


# Applying a result.


@dataclass(frozen=True)
class Plan:
    """What `apply` would do: the files to write, and the numbers and choices to edit by hand."""

    candidate_number: int
    candidate_id: str
    diff: str
    files: dict[Path, str]
    by_hand: dict[str, dict[str, Any]]


def plan_apply(run_dir: Path, candidate: str | None) -> Plan:
    """Work out what applying `candidate` (an id or a display number; default the champion) writes.

    Must run in the run's working directory (`working_directory`), so the factory reads the files as they are now.

    Raises:
        ValueError: The run hasn't finished (no `result.json`), `candidate` names no candidate, or the
            declaration or a tuned file changed since the run.
    """
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        raise ValueError(
            f"{run_dir} has no result.json: the run hasn't finished; run `strands-evals optimize resume {run_dir}`"
        )
    result = OptimizationResult.from_file(result_path)
    view = result.best if candidate is None else _find(result, candidate)
    saved = {entry["key"]: entry for entry in read_run_json(run_dir)["space"]}
    space = load_factory(load_inputs(run_dir).agent).space
    current = {info.key: info for info in space.params()}
    if set(saved) != set(current):
        raise ValueError("the factory's declaration changed since the run; nothing written")
    changed = [
        str(info.source.path if info.source is not None else key)
        for key, info in current.items()
        if info.kind == "text" and saved[key]["default"]["sha256"] != _sha256(str(info.default))
    ]
    if changed:
        raise ValueError(f"changed since the run: {', '.join(changed)}; nothing written")
    before = space._render_files(result.baseline.values)
    after = space._render_files(view.values)
    files = {path: text for path, text in after.items() if before.get(path) != text}
    return Plan(view.number, view.id, view.diff(), files, view.changed_values())


def write_files(files: dict[Path, str]) -> None:
    """Write each file in place, atomically."""
    for path, text in files.items():
        _write_text_atomically(path, text)


def _find(result: OptimizationResult, candidate: str) -> Any:
    if candidate in result.candidates:
        return result.candidates[candidate]
    matches = [view for view in result.candidates.values() if str(view.number) == candidate]
    if not matches:
        raise ValueError(f"--candidate '{candidate}' is neither a candidate id nor a display number of this run")
    return matches[0]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
