"""`strands-evals optimize` — search for a better agent configuration, or resume a search that was killed or
stopped on a proposer error."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._common import EXIT_BAD_INPUT, EXIT_OK, EXIT_RUNTIME_ERROR, emit_error, resolve_format

if TYPE_CHECKING:
    from ...optimization import OptimizationResult

EXIT_INTERRUPTED = 130  # 128 + SIGINT, as shells report Ctrl-C

# The optimizer is imported inside each subcommand, never at module level: `strands-evals` imports every command at
# startup, and `strands_evals.optimization` needs the `optimization` extra.


def _print_result(result: OptimizationResult, args: argparse.Namespace, run_dir: Path) -> None:
    """`summary()` on a TTY (or with `--rich`), else the whole result as JSON on stdout; the folder on stderr."""
    if resolve_format(args) == "json":
        sys.stdout.write(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(result.summary() + "\n")
    print(f"strands-evals optimize: run folder {run_dir}", file=sys.stderr)  # noqa: T201


def _finish(result: OptimizationResult, args: argparse.Namespace, run_dir: Path) -> int:
    """Print the result.

    `run_dir` is the folder as found now, not `result.run_dir`, which is where the run was written (a folder may
    have moved since)."""
    try:
        _print_result(result, args, run_dir)
    except KeyboardInterrupt:
        print(  # noqa: T201
            f"strands-evals optimize: interrupted while printing; the run has finished and its result is in "
            f"{run_dir / 'result.json'} (`strands-evals optimize resume {run_dir}` prints it again)",
            file=sys.stderr,
        )
        return EXIT_INTERRUPTED
    return EXIT_OK


def _proposer_stopped(error: Exception, run_dir: Path, *, debug: bool) -> int:
    """`ProposerError`: the search stopped before the end phase, so there is no result to print; exit 3 with the
    proposer's error (a script checking the exit code shouldn't read a dead search as success) and say how to
    continue with the CLI."""
    emit_error(str(error), debug=debug, exc=error)
    print(  # noqa: T201
        f"strands-evals optimize: continue with `strands-evals optimize resume {run_dir}`", file=sys.stderr
    )
    return EXIT_RUNTIME_ERROR


def _interrupted(run_dir: Path | None) -> int:
    """Ctrl-C: say how to continue instead of printing a traceback."""
    where = f" {run_dir}" if run_dir is not None else " RUN_DIR"
    print(  # noqa: T201
        f"strands-evals optimize: interrupted; continue with `strands-evals optimize resume{where}`", file=sys.stderr
    )
    return EXIT_INTERRUPTED


def _given(args: argparse.Namespace, *names: str) -> dict[str, Any]:
    """The flags among `names` that were passed (left None otherwise), so the `Optimizer`'s defaults stay the
    only defaults."""
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def _before_start(subcommand: Any) -> Any:
    """Wrap a subcommand so Ctrl-C before the run starts (imports, loading the experiment and hooks) exits 130
    with one line instead of a traceback; nothing has been spent then. A Ctrl-C once the run has started is handled
    where it happens (`_interrupted`, `_finish`), so it never reaches this."""

    def guarded(args: argparse.Namespace) -> int:
        try:
            return int(subcommand(args))
        except KeyboardInterrupt:
            print("strands-evals optimize: interrupted before the run started; nothing was spent", file=sys.stderr)  # noqa: T201
            return EXIT_INTERRUPTED

    guarded.__doc__ = subcommand.__doc__
    return guarded


def _run(args: argparse.Namespace) -> int:
    """`optimize run`: read a run file if given, check the flags, build the `Optimizer`, run it, and print the
    result; `cli.json` is written as soon as the run folder exists. Returns the exit code (2 on bad input, 3 when
    the search stopped on a proposer error, 130 on Ctrl-C)."""
    try:
        from ...optimization import Budget, FinalPick, Optimizer, ProposerError
        from .. import _optimize
    except ImportError as missing:  # the `optimization` extra isn't installed
        emit_error(str(missing))
        return EXIT_BAD_INPUT
    debug = getattr(args, "debug", False)

    if args.experiment_file.lower().endswith(_optimize.RUN_FILE_SUFFIX):
        try:
            _optimize.apply_run_file(Path(args.experiment_file), args.run_parser, args)
        except ValueError as run_file_error:
            emit_error(str(run_file_error), debug=debug, exc=run_file_error)
            return EXIT_BAD_INPUT
        experiment_file = args.experiment
    elif args.experiment is not None:
        emit_error(f"--experiment goes with a .toml run file; {args.experiment_file} is already the experiment")
        return EXIT_BAD_INPUT
    else:
        experiment_file = args.experiment_file
    problem = _check_agent_flags(args)
    if problem is not None:
        emit_error(problem)
        return EXIT_BAD_INPUT
    config = Path(args.config).resolve() if args.config is not None else None
    harness: dict[str, Any] = {}
    if config is not None:
        from .. import _harness  # needs strands-harness, in the `optimization` extra

        if not config.is_file():
            emit_error(f"--config {args.config}: no such file")
            return EXIT_BAD_INPUT
        harness = {
            "config": str(config),
            "config_sha256": _harness.file_sha256(config),
            "tune": args.tune or [],
            "tune_instructions": args.tune_instructions is not False,
            "harness_version": _harness.harness_version(),
        }
    elif args.agent is not None and _optimize.is_agent_file(args.agent):
        from .. import _harness  # needs strands-harness, in the `optimization` extra

        harness = {
            "tune": args.tune or [],
            "tune_instructions": args.tune_instructions is not False,
            "harness_version": _harness.harness_version(),
        }
    inputs = _optimize.Inputs(
        experiment_file=str(Path(experiment_file).resolve()),
        agent=args.agent,
        build_task=args.build_task,
        space=args.space,
        **harness,
        custom_evaluators=args.custom_evaluator or [],
        rollout=args.rollout,
        environment=args.environment,
        objective=args.objective_specs or [],
        pareto=args.pareto,
        constraints=args.constraint_specs or [],
        reflection_model=args.reflection_model,
        search=args.search,
        selection=args.selection,
        score_model=args.score_model,
        on_iteration=args.on_iteration,
    )
    settings = _given(
        args,
        "seed",
        "runs_per_candidate",
        "proposals_per_iteration",
        "feasibility_threshold",
        "rerun_passes",
        "max_workers",
        "base_dir",
        "name",
    )
    budget = _given(args, "max_rollouts", "max_iterations")
    try:
        if args.split is not None:
            settings["split"] = _optimize.parse_split(args.split)
        if args.agent_failure_score is not None:
            settings["agent_failure_score"] = _optimize.parse_agent_failure_score(args.agent_failure_score)
        if budget:
            settings["budget"] = Budget(**budget)
        if args.final_pick is not None:
            settings["final_pick"] = _optimize.load_final_pick(args.final_pick)
        elif args.n_finalists is not None:
            settings["final_pick"] = FinalPick(n_finalists=args.n_finalists)
        optimizer = Optimizer(**_optimize.agent_arguments(inputs), **settings)
    except (TypeError, ValueError) as error:
        emit_error(str(error), debug=debug, exc=error)
        return EXIT_BAD_INPUT

    def on_folder(run_dir: Path) -> None:
        _optimize.save_inputs(run_dir, inputs)
        print(f"strands-evals optimize: run folder {run_dir}", file=sys.stderr)  # noqa: T201

    try:
        result = asyncio.run(_optimize.drive(optimizer, on_folder))
    except KeyboardInterrupt:
        return _interrupted(optimizer.run_dir)
    except ProposerError as error:
        return _proposer_stopped(error, error.run_dir, debug=debug)
    return _finish(result, args, Path(result.run_dir))


def _check_agent_flags(args: argparse.Namespace) -> str | None:
    """Flags that need another flag, or exclude one: exactly one agent route (`--agent`, `--build-task`, `--config`,
    on the command line or in a run file); `--tune` goes with `--config`; `--build-task` with `--space`; `--rollout`
    with `--agent` or `--config`; `--environment` with `--agent`; `--final-pick` replaces `--n-finalists`,
    `--search` replaces `--reflection-model`."""
    routes = [
        flag
        for flag, value in (("--agent", args.agent), ("--build-task", args.build_task), ("--config", args.config))
        if value
    ]
    if len(routes) != 1:
        given = f"got {', '.join(routes)}" if routes else "got none"
        return f"give exactly one of --agent, --build-task (with --space), --config ({given})"
    from .._optimize import is_agent_file

    tunes = args.config is not None or (args.agent is not None and is_agent_file(args.agent))
    if not tunes and args.tune:
        return (
            "--tune needs --config or an agent file (--agent agent.py); with a tunable factory or --space, declare "
            "what may change with markers"
        )
    if not tunes and args.tune_instructions is not None:
        return "--tune-instructions / --no-tune-instructions go with --config or an agent file (--agent agent.py)"
    if (args.build_task is None) != (args.space is None):
        return "--build-task and --space go together: the task builder and the space it reads values from"
    if args.build_task is not None and args.rollout:
        return "--rollout doesn't go with --build-task: the task build_task returns is the rollout"
    if args.environment and args.agent is None:
        return "--environment needs --agent: it passes resources to the factory's parameters"
    if args.final_pick is not None and args.n_finalists is not None:
        return "--final-pick and --n-finalists: set n_finalists in the FinalPick"
    if args.search is not None and args.reflection_model is not None:
        return "--search and --reflection-model: set the model in the search's Reflection(model=...)"
    return None


def _resume(args: argparse.Namespace) -> int:
    """`optimize resume`: print a finished run's saved result, or rebuild the inputs of a run that was killed or
    stopped on a proposer error from `cli.json`, reopen it with `Optimizer.resume` in its working directory, and
    print the result. Returns the exit code, as `_run`'s."""
    try:
        from ...optimization import Budget, Optimizer, ProposerError
        from .. import _optimize
    except ImportError as missing:  # the `optimization` extra isn't installed
        emit_error(str(missing))
        return EXIT_BAD_INPUT
    debug = getattr(args, "debug", False)

    run_dir = Path(args.run_dir).resolve()
    if (run_dir / "result.json").is_file() and _test_saved_as_done(run_dir):
        # finished: print the saved result; no agent, experiment, or config
        return _print_finished(run_dir, args)
    try:
        saved = _optimize.read_run_json(run_dir)
        inputs = _optimize.load_inputs(run_dir)
        _optimize.check_config_unchanged(inputs)
    except ValueError as error:
        emit_error(str(error), debug=debug, exc=error)
        return EXIT_BAD_INPUT
    _optimize.check_harness_version(inputs)
    with _optimize.working_directory(saved["workdir"]):
        try:
            budget = None
            if args.max_rollouts is not None or args.max_iterations is not None:
                given = _given(args, "max_rollouts", "max_iterations")
                budget = Budget(**{**saved["budget"], **given})
            optimizer = Optimizer.resume(
                run_dir, **_optimize.agent_arguments(inputs), budget=budget, max_workers=args.max_workers
            )
        except (TypeError, ValueError) as error:
            emit_error(str(error), debug=debug, exc=error)
            return EXIT_BAD_INPUT
        try:
            result = asyncio.run(_optimize.drive(optimizer, lambda _: None))
        except KeyboardInterrupt:
            return _interrupted(run_dir)
        except ProposerError as error:
            return _proposer_stopped(error, run_dir, debug=debug)
    return _finish(result, args, run_dir)


def _test_saved_as_done(run_dir: Path) -> bool:
    """Whether the run's `test` event was written, so its result is final.

    A test report skipped for the budget, or left to re-runs of infrastructure failures, writes `result.json`
    with no `test` event, and `Optimizer.resume` can still run the test; printing the saved result instead
    would dead-end the resume the report's reason asks for. A missing or unreadable events.jsonl prints the
    saved result, as before."""
    try:
        lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return True
    for line in reversed(lines):
        try:
            if json.loads(line).get("event") == "test":
                return True
        except json.JSONDecodeError:  # a kill can truncate the last line
            continue
    return False


def _print_finished(run_dir: Path, args: argparse.Namespace) -> int:
    """A finished run: print its `result.json`, building no agent, so a config edited (or deleted) since, for
    instance by applying the champion, doesn't matter. `--max-*` overrides change nothing, and say so."""
    from ...optimization import OptimizationResult

    overrides = [
        flag
        for flag, value in (
            ("--max-rollouts", args.max_rollouts),
            ("--max-iterations", args.max_iterations),
            ("--max-workers", args.max_workers),
        )
        if value is not None
    ]
    if overrides:
        print(  # noqa: T201
            f"strands-evals optimize: the run has finished, so {', '.join(overrides)} change nothing; "
            "start a new run to search more",
            file=sys.stderr,
        )
    try:
        result = OptimizationResult.from_file(run_dir / "result.json")
    except (OSError, ValueError) as error:
        emit_error(f"{run_dir / 'result.json'}: {error}", debug=getattr(args, "debug", False), exc=error)
        return EXIT_BAD_INPUT
    return _finish(result, args, run_dir)


def _add_run(subparsers: argparse._SubParsersAction, parent: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "run",
        parents=[parent],
        help="search for a better configuration of a tunable agent",
        description=(
            "Load an Experiment (its cases and evaluators), take the agent from --agent (a @tunable factory, or an "
            "agent.py / agent.ts file whose agent create_harness builds), "
            "--build-task with --space (your task builder and its space), or --config (a strands-harness portable "
            "config), and run the optimizer: the search on the train cases, the champion picked on validation, "
            "and the champion compared with the baseline on the test cases. Writes the run folder (events, best.diff, "
            "best.values.json, result.json, cli.json); nothing in your project is changed."
        ),
    )
    parser.add_argument(
        "experiment_file",
        metavar="EXPERIMENT_FILE",
        help=(
            "a serialized Experiment JSON file, or a .toml run file holding the run's flags (keys are the flag names "
            "with _ for -; flags given here override it)"
        ),
    )
    parser.add_argument(
        "--experiment",
        metavar="PATH",
        default=None,
        help="with a .toml run file: the experiment file, instead of the file's experiment",
    )
    agent = parser.add_mutually_exclusive_group()
    agent.add_argument(
        "--agent",
        metavar="MODULE:ATTR | FILE.py[:ATTR] | FILE.ts[:ATTR]",
        help=(
            "a tunable factory (@tunable or tunable(fn, inject=...)), as 'module:attr' or 'path/to/file.py:attr'; or "
            "an agent file whose agent (the attribute or export named 'agent' unless :ATTR says) is built by "
            "create_harness / createHarness, as /export writes it: agent.py, or agent.ts / .mts / .js / .mjs (run "
            "in Node.js 22.18+, with @strands-agents/harness installed). Its instructions are tuned, plus each --tune "
            "over that call's arguments, and each trial runs the file again"
        ),
    )
    agent.add_argument(
        "--build-task",
        metavar="MODULE:ATTR",
        help="a build_task(ctx) returning the candidate's task(case); needs --space",
    )
    agent.add_argument(
        "--config",
        metavar="PATH",
        help="a strands-harness portable config (JSON); its instructions are tuned, plus each --tune",
    )
    parser.add_argument(
        "--space", metavar="MODULE:ATTR", default=None, help="with --build-task: a mapping of declared names to markers"
    )
    parser.add_argument(
        "--tune",
        metavar="PATH[=JSON_LIST | =int:LO..HI | =float:LO..HI]",
        action="append",
        default=None,
        help=(
            "with --config or an agent file, repeatable: a dotted config path (or create_harness / createHarness "
            "argument) to tune, as text (e.g. 'description', or "
            "'skills' for its skill folder), with =JSON_LIST as a choice of JSON values, lists and objects "
            'included (e.g. \'effort=["low","high"]\', or \'builtinTools=[[],["read"]]\' for tool sets), or '
            "with =int:LO..HI / =float:LO..HI as a number in a range (e.g. 'builtinTools.subagent.maxDepth=int:0..3')"
        ),
    )
    parser.add_argument(
        "--tune-instructions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "with --config or an agent file: tune the instructions (default); --no-tune-instructions tunes only --tune"
        ),
    )
    parser.add_argument(
        "--rollout",
        metavar="MODULE:ATTR",
        default=None,
        help=(
            "a rollout_fn(agent_factory, case); with --config, agent_factory builds a fresh harness; with a .ts or .js "
            "--agent, it returns an agent whose invoke_async asks a fresh agent each call, and whose session() keeps "
            "one conversation across turns"
        ),
    )
    parser.add_argument(
        "--environment", metavar="MODULE:ATTR", default=None, help="with --agent: an environment(case) function"
    )
    parser.add_argument(
        "--custom-evaluator",
        metavar="MODULE:CLASS",
        action="append",
        default=None,
        help="custom Evaluator subclass to register before loading the experiment (repeatable)",
    )

    goal = parser.add_argument_group("what to optimize (default: maximize the mean of every evaluator)")
    goal.add_argument(
        "--objective",
        dest="objective_specs",
        metavar="{maximize,minimize}:METRIC[:weight=X][:scale=X] | MODULE:ATTR",
        action="append",
        default=None,
        help=(
            "repeatable; several are combined as a weighted sum (or a Pareto front with --pareto); "
            "MODULE:ATTR names a whole objective, given alone (use it for a metric whose name contains ':')"
        ),
    )
    goal.add_argument("--pareto", action="store_true", help="combine the --objective's as a Pareto front")
    goal.add_argument(
        "--constraint",
        dest="constraint_specs",
        metavar="METRIC<=X | METRIC>=X | MODULE:ATTR",
        action="append",
        default=None,
        help="repeatable, e.g. 'tokens<=4000'; MODULE:ATTR names a constraint or a list of them",
    )

    objects = parser.add_argument_group("objects from your code (MODULE:ATTR; default: the Optimizer's)")
    objects.add_argument("--search", metavar="MODULE:ATTR", default=None, help="a Search: blocks, proposers, schedule")
    objects.add_argument("--selection", metavar="MODULE:ATTR", default=None, help="a Selection choosing each parent")
    objects.add_argument("--score-model", metavar="MODULE:ATTR", default=None, help="a ScoreModel")
    objects.add_argument(
        "--final-pick", metavar="MODULE:ATTR", default=None, help="a FinalPick (finalists and the pick rule)"
    )
    objects.add_argument(
        "--on-iteration",
        metavar="MODULE:ATTR",
        default=None,
        help="a callback (sync or async) called with each IterationRecord, after the progress line",
    )

    settings = parser.add_argument_group("settings (default: the Optimizer's)")
    settings.add_argument(
        "--split",
        default=None,
        metavar="TRAIN,VALIDATION,TEST | MODULE:ATTR",
        help="shares (test 0 = none), or a Split (case lists, a split seed)",
    )
    settings.add_argument(
        "--seed", type=int, default=None, metavar="N", help="the run's seed: split, TPE, Thompson, ... (default 0)"
    )
    settings.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        metavar="N",
        help="the budget, in rollouts (one case run once); the end phase's share is reserved up front (default 500)",
    )
    settings.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        metavar="N",
        help="stop after N x --proposals-per-iteration candidate slots (default: no limit)",
    )
    settings.add_argument(
        "--runs-per-candidate", type=int, default=None, metavar="N", help="repetitions per case (default 1)"
    )
    settings.add_argument(
        "--proposals-per-iteration", type=int, default=None, metavar="N", help="candidates per step (default 4)"
    )
    settings.add_argument(
        "--n-finalists",
        type=int,
        default=None,
        metavar="N",
        help="finalists measured on validation before the pick (default 3)",
    )
    settings.add_argument(
        "--agent-failure-score",
        default=None,
        metavar="{worst,none,X}",
        help="what a crashed rollout scores: each metric's worst end, missing, or a number (default worst)",
    )
    settings.add_argument(
        "--feasibility-threshold",
        type=float,
        default=None,
        metavar="X",
        help="P(constraint met) a candidate needs to count as feasible (default 0.9)",
    )
    settings.add_argument(
        "--rerun-passes",
        type=int,
        default=None,
        metavar="N",
        help="re-runs of infrastructure failures and failed judges (default 2)",
    )
    settings.add_argument("--max-workers", type=int, default=None, metavar="N", help="parallel trials (default: 4)")
    settings.add_argument(
        "--reflection-model", metavar="MODEL_ID", default=None, help="the model rewriting text values"
    )
    settings.add_argument("--base-dir", metavar="DIR", default=None, help="parent of run folders")
    settings.add_argument("--name", default=None, help="prefix of the run folder's name")
    parser.set_defaults(func=_before_start(_run), run_parser=parser)


def add_subparser(subparsers: argparse._SubParsersAction, parent: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "optimize",
        parents=[parent],
        help="search for a better agent configuration, or resume a search",
        description=(
            "Search for a better configuration of a tunable agent, or resume a search that was killed or "
            "stopped on a proposer error."
        ),
    )
    commands = parser.add_subparsers(dest="optimize_command", metavar="SUBCOMMAND")
    commands.required = True
    _add_run(commands, parent)

    resume = commands.add_parser(
        "resume",
        parents=[parent],
        help="continue a run that was killed or stopped on an error, or print a finished run's result",
        description=(
            "Rebuild a run's inputs from its cli.json and continue it, whether it was killed or stopped on a "
            "proposer error; a finished run prints its result."
        ),
    )
    resume.add_argument("run_dir", metavar="RUN_DIR", help="the run folder `optimize run` printed")
    resume.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        metavar="N",
        help="a new budget for the continued run (default: the run's)",
    )
    resume.add_argument(
        "--max-iterations", type=int, default=None, metavar="N", help="a new iteration limit (default: the run's)"
    )
    resume.add_argument(
        "--max-workers", type=int, default=None, metavar="N", help="parallel trials (default: the run's)"
    )
    resume.set_defaults(func=_before_start(_resume))
    return parser
