"""What `strands-evals optimize` builds from its flags: the `Optimizer` and `cli.json`.

`cli.json` holds the inputs of `optimize run`, as given, so `resume RUN_DIR` rebuilds the same experiment, agent,
objective, constraints, search, selection, score model, and progress callback from the run folder alone. A flag that
takes an object takes `MODULE:ATTR` (`import_attr`), checked against the type the `Optimizer` expects.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
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
from ._entrypoint import EntryPointError, import_attr, resolve_custom_evaluators

logger = logging.getLogger(__name__)

CLI_FILE = "cli.json"
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
    if any(char in spec for char in "<>="):
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
    """`flag`'s `MODULE:ATTR` target, which must be an instance of `kind` (a type, a tuple of types, or a union), or
    callable when `kind` is None; `what` names the expected kind in the error."""
    target = import_attr(spec)
    if not (callable(target) if kind is None else isinstance(target, kind)):
        raise EntryPointError(f"{flag} '{spec}' is a {type(target).__name__}, not {what}")
    return target


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
    """What `Optimizer` and `Optimizer.resume` take about the agent, the goal, the search, the selection, the score
    model, and progress, built from `inputs`."""
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
        markers, tuned = _harness.declare(config, inputs.tune)
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
