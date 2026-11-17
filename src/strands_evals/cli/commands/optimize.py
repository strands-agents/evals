"""`strands-evals optimize` — search for a better agent configuration, resume a killed search, apply a result."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._common import EXIT_BAD_INPUT, EXIT_OK, emit_error, resolve_format

if TYPE_CHECKING:
    from ...optimization import OptimizationResult

# The optimizer is imported inside each subcommand, never at module level: `strands-evals` imports every command at
# startup, and `strands_evals.optimization` needs the `optimization` extra.


def _print_result(result: OptimizationResult, args: argparse.Namespace) -> None:
    """`summary()` on a TTY (or with `--rich`), else the whole result as JSON on stdout; the folder on stderr."""
    if resolve_format(args) == "json":
        sys.stdout.write(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(result.summary() + "\n")
    print(f"strands-evals optimize: run folder {result.run_dir}", file=sys.stderr)  # noqa: T201


def _given(args: argparse.Namespace, *names: str) -> dict[str, Any]:
    """The flags among `names` that were passed (left None otherwise), so the `Optimizer`'s defaults stay the
    only defaults."""
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def _run(args: argparse.Namespace) -> int:
    from ...optimization import Budget, FinalPick, Optimizer
    from .. import _optimize

    error = _check_agent_flags(args)
    if error is not None:
        emit_error(error)
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
            "harness_version": _harness.harness_version(),
        }
    inputs = _optimize.Inputs(
        experiment_file=str(Path(args.experiment_file).resolve()),
        agent=args.agent,
        **harness,
        custom_evaluators=args.custom_evaluator or [],
        rollout=args.rollout,
        environment=args.environment,
        objective=args.objective_specs or [],
        pareto=args.pareto,
        constraints=args.constraint_specs or [],
        reflection_model=args.reflection_model,
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
        if args.n_finalists is not None:
            settings["final_pick"] = FinalPick(n_finalists=args.n_finalists)
        optimizer = Optimizer(**_optimize.agent_arguments(inputs), **settings, on_iteration=_optimize.print_iteration)
    except (TypeError, ValueError) as error:
        emit_error(str(error))
        return EXIT_BAD_INPUT

    def on_folder(run_dir: Path) -> None:
        _optimize.save_inputs(run_dir, inputs)
        print(f"strands-evals optimize: run folder {run_dir}", file=sys.stderr)  # noqa: T201

    _print_result(asyncio.run(_optimize.drive(optimizer, on_folder)), args)
    return EXIT_OK


def _check_agent_flags(args: argparse.Namespace) -> str | None:
    """`--tune` goes with `--config`; `--rollout` and `--environment` with `--agent`."""
    if args.config is None and args.tune:
        return "--tune needs --config (with --agent, declare what may change with @tunable)"
    if args.config is not None:
        given = [flag for flag, value in (("--rollout", args.rollout), ("--environment", args.environment)) if value]
        if given:
            return f"{', '.join(given)} needs --agent: with --config, each trial runs the configured harness"
    return None


def _resume(args: argparse.Namespace) -> int:
    from ...optimization import Budget, Optimizer
    from .. import _optimize

    run_dir = Path(args.run_dir).resolve()
    try:
        saved = _optimize.read_run_json(run_dir)
        inputs = _optimize.load_inputs(run_dir)
    except ValueError as error:
        emit_error(str(error))
        return EXIT_BAD_INPUT
    _optimize.check_harness_version(inputs)
    with _optimize.working_directory(saved["workdir"]):
        try:
            budget = None
            if args.max_rollouts is not None or args.max_iterations is not None:
                given = _given(args, "max_rollouts", "max_iterations")
                budget = Budget(**{**saved["budget"], **given})
            optimizer = Optimizer.resume(
                run_dir,
                **_optimize.agent_arguments(inputs),
                budget=budget,
                max_workers=args.max_workers,
                on_iteration=_optimize.print_iteration,
            )
        except (TypeError, ValueError) as error:
            emit_error(str(error))
            return EXIT_BAD_INPUT
        result = asyncio.run(_optimize.drive(optimizer, lambda _: None))
    _print_result(result, args)
    return EXIT_OK


def _apply(args: argparse.Namespace) -> int:
    from .. import _optimize

    run_dir = Path(args.run_dir).resolve()
    try:
        workdir = _optimize.read_run_json(run_dir)["workdir"]
        with _optimize.working_directory(workdir):
            plan = _optimize.plan_apply(run_dir, args.candidate)
    except ValueError as error:
        emit_error(str(error))
        return EXIT_BAD_INPUT

    out = sys.stdout
    if not plan.files and not plan.by_hand:
        out.write(f"candidate {plan.candidate_number} changes nothing against the baseline: nothing to apply\n")
        return EXIT_OK
    out.write(f"candidate {plan.candidate_number} ({plan.candidate_id})\n")
    if plan.diff:
        out.write(plan.diff if plan.diff.endswith("\n") else plan.diff + "\n")
    if args.dry_run:
        out.write("dry run: nothing written\n")
    elif plan.files:
        if not args.yes:
            if not sys.stdin.isatty():
                emit_error("stdin is not a terminal: pass --yes to write the files above, or --dry-run")
                return EXIT_BAD_INPUT
            answer = input(f"Write {len(plan.files)} file(s)? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                out.write("nothing written\n")
                return EXIT_OK
        _optimize.write_files(plan.files)
        for path in plan.files:
            out.write(f"wrote {path}\n")
    if plan.by_hand:
        out.write("numbers and choices live in your code; edit them by hand:\n")
        for key, sides in plan.by_hand.items():
            old = repr(sides["old"]) if "old" in sides else "(inactive)"
            new = repr(sides["new"]) if "new" in sides else "(inactive)"
            out.write(f"  {key}: {old} -> {new}\n")
    return EXIT_OK


def _add_run(subparsers: argparse._SubParsersAction, parent: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "run",
        parents=[parent],
        help="search for a better configuration of a tunable agent",
        description=(
            "Load an Experiment (its cases and evaluators), take the agent from --agent (a @tunable factory) or "
            "--config (a strands-harness portable config), and run the "
            "optimizer: the search on the train cases, the champion picked on validation, and the champion "
            "compared with the baseline on the test cases. Writes the run folder (events, best.diff, "
            "best.values.json, result.json, cli.json); nothing in your project is changed until `optimize apply`."
        ),
    )
    parser.add_argument("experiment_file", metavar="EXPERIMENT_FILE", help="path to a serialized Experiment JSON file")
    agent = parser.add_mutually_exclusive_group(required=True)
    agent.add_argument(
        "--agent",
        metavar="MODULE:ATTR",
        help="a tunable factory (@tunable or tunable(fn, inject=...)), as 'module:attr' or 'path/to/file.py:attr'",
    )
    agent.add_argument(
        "--config",
        metavar="PATH",
        help="a strands-harness portable config (JSON); its instructions are tuned, plus each --tune",
    )
    parser.add_argument(
        "--tune",
        metavar="PATH[=JSON_LIST]",
        action="append",
        default=None,
        help=(
            "with --config, repeatable: a dotted config path to tune, as text (e.g. 'subagents.0.instructions', "
            "or 'skills' for its skill folder) or, with =JSON_LIST, as a choice (e.g. 'effort=[\"low\",\"high\"]')"
        ),
    )
    parser.add_argument("--rollout", metavar="MODULE:ATTR", default=None, help="a custom rollout_fn")
    parser.add_argument("--environment", metavar="MODULE:ATTR", default=None, help="an environment(case) function")
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
        metavar="{maximize,minimize}:METRIC[:weight=X][:scale=X]",
        action="append",
        default=None,
        help="repeatable; several are combined as a weighted sum (or a Pareto front with --pareto)",
    )
    goal.add_argument("--pareto", action="store_true", help="combine the --objective's as a Pareto front")
    goal.add_argument(
        "--constraint",
        dest="constraint_specs",
        metavar="METRIC<=X|METRIC>=X",
        action="append",
        default=None,
        help="repeatable, e.g. 'tokens<=4000'",
    )

    settings = parser.add_argument_group("settings (default: the Optimizer's)")
    settings.add_argument("--split", default=None, metavar="TRAIN,VALIDATION,TEST", help="shares; test 0 = none")
    settings.add_argument("--seed", type=int, default=None, metavar="N")
    settings.add_argument("--max-rollouts", type=int, default=None, metavar="N")
    settings.add_argument("--max-iterations", type=int, default=None, metavar="N")
    settings.add_argument("--runs-per-candidate", type=int, default=None, metavar="N")
    settings.add_argument("--proposals-per-iteration", type=int, default=None, metavar="N")
    settings.add_argument("--n-finalists", type=int, default=None, metavar="N")
    settings.add_argument("--agent-failure-score", default=None, metavar="{worst,none,X}")
    settings.add_argument("--feasibility-threshold", type=float, default=None, metavar="X")
    settings.add_argument("--rerun-passes", type=int, default=None, metavar="N")
    settings.add_argument("--max-workers", type=int, default=None, metavar="N", help="parallel trials (default: 4)")
    settings.add_argument(
        "--reflection-model", metavar="MODEL_ID", default=None, help="the model rewriting text values"
    )
    settings.add_argument("--base-dir", metavar="DIR", default=None, help="parent of run folders")
    settings.add_argument("--name", default=None, help="prefix of the run folder's name")
    parser.set_defaults(func=_run)


def add_subparser(subparsers: argparse._SubParsersAction, parent: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "optimize",
        parents=[parent],
        help="search for a better agent configuration, resume a search, or apply its result",
        description="Search for a better configuration of a tunable agent, resume a killed search, or apply a result.",
    )
    commands = parser.add_subparsers(dest="optimize_command", metavar="SUBCOMMAND")
    commands.required = True
    _add_run(commands, parent)

    resume = commands.add_parser(
        "resume",
        parents=[parent],
        help="continue a killed run, or print a finished run's result",
        description="Rebuild a run's inputs from its cli.json and continue it; a finished run prints its result.",
    )
    resume.add_argument("run_dir", metavar="RUN_DIR")
    resume.add_argument("--max-rollouts", type=int, default=None, metavar="N")
    resume.add_argument("--max-iterations", type=int, default=None, metavar="N")
    resume.add_argument("--max-workers", type=int, default=None, metavar="N")
    resume.set_defaults(func=_resume)

    apply = commands.add_parser(
        "apply",
        parents=[parent],
        help="write a finished run's champion (or another candidate) into your files",
        description=(
            "Write a candidate's text values into the original files, after confirmation. Refuses when a tuned "
            "file or the declaration changed since the run. Numbers and choices are printed for you to edit."
        ),
    )
    apply.add_argument("run_dir", metavar="RUN_DIR")
    apply.add_argument("--candidate", metavar="ID_OR_NUMBER", default=None, help="default: the champion")
    apply.add_argument("--dry-run", action="store_true", help="print what would change; write nothing")
    apply.add_argument("--yes", action="store_true", help="write without asking")
    apply.set_defaults(func=_apply)
    return parser
