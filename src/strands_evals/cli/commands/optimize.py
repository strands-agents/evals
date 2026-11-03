"""`strands-evals optimize`: search on train, pick once on validation, and report test.

The command never edits project source. `--apply` writes the picked candidate's file texts only when asked.
"""

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...case import Case
from ...evaluators.deterministic import Contains
from ...evaluators.evaluator import Evaluator
from ...experiment import Experiment
from ...optimization.execution.status import InfrastructureError
from ...optimization.setup.budget import Budget
from ...types.evaluation import EvaluationOutput
from ...types.optimization import CaseSplit, Value
from .._common import EXIT_BAD_INPUT, EXIT_OK, EXIT_RUNTIME_ERROR, emit_error
from .._optimize.apply import apply_candidate
from .._optimize.engine import (
    REDRAWS,
    SCHEDULE,
    SELECTION,
    SERVICE_ERROR_CODES,
    EvaluatorRow,
    InterimOptimizer,
    RolloutOutput,
    Settings,
)
from .._optimize.proposers import FileProposer, Proposer, RandomProposer, ReflectionProposer, load_proposal_file
from .._optimize.record import RUNS_DIRECTORY, SCHEMA_VERSION, RunDirectory, canonical_json, check_name
from .._optimize.scoring import CrashScore, Criteria, MetricRegistry, build_objective, require_optimization_extra
from .._optimize.space import CliSpace, instructions_space, load_space
from .._optimize.splits import DEFAULT_SHARES, assign_split, parse_shares
from .._project import (
    AssertionContains,
    ProjectError,
    ProjectRunner,
    default_project_evaluator,
    load_project_cases,
    resolve_project,
)

ENGINE = "cli-interim"
DEFAULT_FINALISTS = 3
DEFAULT_PROPOSALS_PER_ITERATION = 4
DEFAULT_RERUN_PASSES = 2
DEFAULT_SEED = 0
CASE_DIGEST_LENGTH = 16
READ_ON_RESUME = (
    "split",
    "seed",
    "finalists",
    "runs_per_candidate",
    "crash_score",
    "proposals_per_iteration",
    "rerun_passes",
    "name",
    "base_dir",
)
COMPARED_ON_RESUME = (
    "project",
    "runner",
    "cases",
    "space",
    "objective",
    "constraints",
    "evaluators",
    "judge",
    "search",
    "selection",
    "score_model",
)
# An external runner reports provider and network failures only as text, so these markers mark them for a re-run.
TRANSIENT_RUNNER_MARKERS = (
    *SERVICE_ERROR_CODES,
    "throttl",
    "too many requests",
    "rate exceeded",
    "serviceunavailable",
    "service unavailable",
    "modelnotready",
    "econnreset",
    "econnrefused",
    "enotfound",
    "etimedout",
    "eai_again",
    "socket hang up",
    "connection reset",
    "connection refused",
)
_POSITIVE = ("max_rollouts", "max_iterations", "finalists", "runs_per_candidate", "proposals_per_iteration")


def _run(args: argparse.Namespace) -> int:
    try:
        with _stdout_to_stderr():
            result = _apply(args) if args.apply else _optimize(args)
    except (ValueError, OSError) as exc:
        emit_error(str(exc))
        return EXIT_BAD_INPUT
    except Exception as exc:
        emit_error(f"optimization failed: {exc}", debug=getattr(args, "debug", False), exc=exc)
        return EXIT_RUNTIME_ERROR
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return EXIT_OK


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Send everything written to stdout, by Python or by child processes, to stderr; stdout gets only the JSON."""
    sys.stdout.flush()
    try:
        saved = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        saved = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stderr.flush()
        if saved is not None:
            os.dup2(saved, 1)
            os.close(saved)


def _apply(args: argparse.Namespace) -> dict[str, Any]:
    extra = [
        flag for flag in ("project", "project_option", "cases", "space", "instructions_file") if getattr(args, flag)
    ]
    if extra:
        raise ValueError("--apply takes only RUN_DIR, --candidate, and --dry-run")
    return apply_candidate(Path(args.apply), args.candidate, dry_run=args.dry_run)


def _optimize(args: argparse.Namespace) -> dict[str, Any]:
    _check_flags(args)
    entry = resolve_project(_project_argument(args))
    root = entry.parent
    space = _space(args, root)
    cases = load_project_cases(args.cases)
    evaluators = _evaluators(args)
    registry = MetricRegistry({evaluator.get_name(): _score_range(evaluator) for evaluator in evaluators})
    run = RunDirectory.open(Path(args.resume)) if args.resume else None
    saved = {**run.config(), "budget": _saved_budget(run)} if run else {}
    criteria = Criteria(build_objective(args.objective or [], registry), registry, _crash_score(args, saved))
    seed = saved.get("seed", args.seed if args.seed is not None else DEFAULT_SEED)
    proposers = _proposers(args, space, seed, required=not args.dry_run)
    settings = _settings(args, cases, seed, saved)
    config = _config(args, entry, cases, space, criteria, proposers, settings, seed)
    if args.dry_run:
        return _plan(config, space, settings, proposers, _base_dir(args, root))
    require_optimization_extra()
    runner = ProjectRunner(entry, runner=args.runner, collector_endpoint=args.collector_endpoint)
    _check_instructions(runner, space)
    if run is None:
        run = RunDirectory.create(_base_dir(args, root), config, name=args.name)
    else:
        _compare(saved, config)
    progress(f"run directory {run.root}")
    engine = InterimOptimizer(
        space=space,
        cases=cases,
        criteria=criteria,
        settings=settings,
        proposers=proposers,
        rollout=_rollout(runner),
        evaluate=_evaluate(evaluators),
        run=run,
        progress=progress,
    )
    if args.resume:
        engine.restore(run.events.read())
        if not engine.finished:
            code = {"strands_evals": importlib.metadata.version("strands-agents-evals")}
            budget = {"max_rollouts": settings.budget.max_rollouts, "max_iterations": settings.budget.max_iterations}
            run.events.append("resume", engine.iteration, spent=engine.spent, budget=budget, code=code)
    return engine.run()


def progress(message: str) -> None:
    """One progress line on stderr; stdout carries only the result JSON."""
    print(f"strands-evals optimize: {message}", file=sys.stderr)  # noqa: T201


def _check_flags(args: argparse.Namespace) -> None:
    if args.candidate is not None:
        raise ValueError("--candidate needs --apply RUN_DIR")
    if not args.cases:
        raise ValueError("optimize needs --cases DIRECTORY")
    if not args.space and not args.instructions_file:
        raise ValueError("optimize needs --space FILE or --instructions-file FILE")
    for flag in _POSITIVE:
        value = getattr(args, flag)
        if value is not None and value < 1:
            raise ValueError(f"--{flag.replace('_', '-')} must be at least 1")
    if args.rerun_passes is not None and args.rerun_passes < 0:
        raise ValueError("--rerun-passes must be at least 0")
    if args.resume:
        explicit = [f"--{flag.replace('_', '-')}" for flag in READ_ON_RESUME if getattr(args, flag) is not None]
        if explicit:
            raise ValueError(f"{', '.join(explicit)}: read from run.json on resume, so omit them")
        if args.dry_run:
            raise ValueError("--dry-run cannot be combined with --resume")


def _project_argument(args: argparse.Namespace) -> str:
    if bool(args.project_option) == bool(args.project):
        raise ValueError("provide exactly one project via --project PATH or positional PATH")
    return str(args.project_option or args.project)


def _space(args: argparse.Namespace, root: Path) -> CliSpace:
    if args.space:
        return load_space(Path(args.space).expanduser(), root)
    return instructions_space(Path(args.instructions_file).expanduser().resolve(), root)


def _evaluators(args: argparse.Namespace) -> list[Evaluator]:
    if args.contains is not None:
        return [Contains(value=args.contains)]
    if args.evaluator == "contains":
        return [AssertionContains()]
    return [default_project_evaluator()]


def _judge_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.contains is not None:
        return {"kind": "contains", "value": args.contains}
    if args.evaluator == "contains":
        return {"kind": "assertion_contains"}
    rubric = default_project_evaluator().rubric
    return {"kind": "assertion_judge", "rubric_sha256": hashlib.sha256(rubric.encode("utf-8")).hexdigest()}


def _score_range(evaluator: Evaluator) -> tuple[float | None, float | None]:
    score_range = getattr(evaluator, "score_range", None)
    return (0.0, 1.0) if score_range is None else (score_range[0], score_range[1])


def _crash_score(args: argparse.Namespace, saved: Mapping[str, Any]) -> CrashScore:
    if saved:
        return saved["crash_score"]  # type: ignore[no-any-return]
    text = args.crash_score or "worst"
    if text in ("worst", "none"):
        return "worst" if text == "worst" else None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"--crash-score must be worst, none, or a number, got {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"--crash-score must be a finite number, got {text!r}")
    return value


def _proposers(args: argparse.Namespace, space: CliSpace, seed: int, *, required: bool) -> list[Proposer]:
    if args.proposal_file:
        path = Path(args.proposal_file).expanduser().resolve()
        keys = tuple(param.key for param in space.params)
        return [FileProposer(patches=load_proposal_file(path, space), keys=keys, source=str(path), name="file")]
    blocks: list[Proposer] = []
    if space.text_keys:
        if not args.proposal_model and required:
            raise ValueError("text keys need --proposal-model MODEL_ID, or --proposal-file FILE for an offline run")
        blocks.append(ReflectionProposer(model=args.proposal_model or "", keys=space.text_keys))
    if space.value_keys:
        blocks.append(RandomProposer(keys=space.value_keys, seed=seed))
    return blocks


def _saved_budget(run: RunDirectory) -> dict[str, Any]:
    """The budget of the last `resume` event, else the one in `run.json`: a resume keeps the last budget given."""
    resumed = [event["budget"] for event in run.events.read() if event["event"] == "resume" and "budget" in event]
    return dict(resumed[-1]) if resumed else dict(run.config()["budget"])


def _settings(args: argparse.Namespace, cases: Sequence[Case], seed: int, saved: Mapping[str, Any]) -> Settings:
    if saved:
        budget = saved["budget"]
        if args.max_rollouts is not None and args.max_rollouts < budget["max_rollouts"]:
            raise ValueError(
                f"--max-rollouts on resume must be at least the saved {budget['max_rollouts']}: "
                "a resume can only move where the search stops"
            )
        return Settings(
            split=CaseSplit.model_validate(saved["split"]["cases"]),
            budget=Budget(
                max_rollouts=args.max_rollouts or budget["max_rollouts"],
                max_iterations=args.max_iterations if args.max_iterations is not None else budget["max_iterations"],
            ),
            finalists=saved["final_pick"]["finalists"],
            runs_per_candidate=saved["runs_per_candidate"],
            proposals_per_iteration=saved["proposals_per_iteration"],
            rerun_passes=saved["rerun_passes"],
        )
    shares = parse_shares(args.split) if args.split else DEFAULT_SHARES
    settings = Settings(
        split=assign_split([case.name for case in cases if case.name], shares, seed),
        budget=Budget(max_rollouts=args.max_rollouts or Budget().max_rollouts, max_iterations=args.max_iterations),
        finalists=args.finalists or DEFAULT_FINALISTS,
        runs_per_candidate=args.runs_per_candidate or 1,
        proposals_per_iteration=args.proposals_per_iteration or DEFAULT_PROPOSALS_PER_ITERATION,
        rerun_passes=args.rerun_passes if args.rerun_passes is not None else DEFAULT_RERUN_PASSES,
    )
    settings.check_budget()
    return settings


def _config(
    args: argparse.Namespace,
    entry: Path,
    cases: Sequence[Case],
    space: CliSpace,
    criteria: Criteria,
    proposers: Sequence[Proposer],
    settings: Settings,
    seed: int,
) -> dict[str, Any]:
    check_name(args.name)
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "workdir": str(entry.parent),
        "project": str(entry),
        "runner": str(Path(shutil.which(args.runner) or args.runner).resolve()) if args.runner else None,
        "cases": [{"name": case.name, "digest": _case_digest(case)} for case in cases],
        "space": space.config(),
        "objective": str(criteria.objective),
        "constraints": [],
        "evaluators": list(criteria.registry.evaluators),
        "judge": _judge_config(args),
        "search": {
            "schedule": {"name": SCHEDULE},
            "blocks": [proposer.config() for proposer in proposers],
            "redraws": REDRAWS,
        },
        "selection": {"name": SELECTION},
        "score_model": None,
        "split": {
            "shares": list(parse_shares(args.split) if args.split else DEFAULT_SHARES),
            "seed": seed,
            "cases": settings.split.model_dump(mode="json"),
        },
        "final_pick": {"finalists": settings.finalists, "rule": {"kind": "Mean"}},
        "budget": {"max_rollouts": settings.budget.max_rollouts, "max_iterations": settings.budget.max_iterations},
        "runs_per_candidate": settings.runs_per_candidate,
        "crash_score": criteria.crash_score,
        "seed": seed,
        "proposals_per_iteration": settings.proposals_per_iteration,
        "rerun_passes": settings.rerun_passes,
        "code": {"strands_evals": importlib.metadata.version("strands-agents-evals")},
    }


def _compare(saved: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    for field in COMPARED_ON_RESUME:
        if canonical_json(saved.get(field)) != canonical_json(config.get(field)):
            raise ValueError(
                f"cannot resume: {field} changed since the run started; start a new run to use the new {field}"
            )


def _plan(
    config: Mapping[str, Any], space: CliSpace, settings: Settings, proposers: Sequence[Proposer], base_dir: Path
) -> dict[str, Any]:
    validation_reserve, test_reserve = settings.reserves()
    per_candidate = len(settings.split.train) * settings.runs_per_candidate
    search = settings.budget.max_rollouts - per_candidate - validation_reserve - test_reserve
    blocks = [{key: value for key, value in p.config().items() if key != "patches"} for p in proposers]
    warnings = [
        f"the {block['block']} block has no model: pass --proposal-model before a real run"
        for block in blocks
        if block["proposer"] == "Reflection" and not block["model"]
    ]
    return {
        "dry_run": True,
        "engine": ENGINE,
        "space": [_plan_param(param) for param in config["space"]],
        "split": config["split"]["cases"],
        "objective": config["objective"],
        "evaluators": config["evaluators"],
        "budget": config["budget"],
        "rollouts": {
            "baseline": per_candidate,
            "per_candidate": per_candidate,
            "validation_reserve": validation_reserve,
            "test_reserve": test_reserve,
            "search": search,
            "max_candidates": search // per_candidate,
        },
        "final_pick": config["final_pick"],
        "runs_per_candidate": settings.runs_per_candidate,
        "proposals_per_iteration": settings.proposals_per_iteration,
        "seed": config["seed"],
        "search": blocks,
        "base_dir": str(base_dir),
        "monetary_estimate": None,
        "warnings": warnings,
    }


def _plan_param(param: Mapping[str, Any]) -> dict[str, Any]:
    if param["kind"] != "text":
        return dict(param)
    return {key: value for key, value in param.items() if key != "default"} | {"chars": len(param["default"])}


def _base_dir(args: argparse.Namespace, root: Path) -> Path:
    return Path(args.base_dir).expanduser().resolve() if args.base_dir else root / RUNS_DIRECTORY


def _case_digest(case: Case) -> str:
    data = {
        "input": case.input,
        "expected_output": case.expected_output,
        "expected_assertion": case.expected_assertion,
        "metadata": case.metadata,
    }
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:CASE_DIGEST_LENGTH]


def _check_instructions(runner: ProjectRunner, space: CliSpace) -> None:
    param = next((p for p in space.params if p.key == "instructions" and p.source is not None), None)
    if param is None or param.source is None:
        return
    try:
        runtime = runner.get_instructions()
    except TypeError as exc:
        progress(
            f"cannot compare {param.source.path.name} with the runtime instructions ({exc}); give every "
            "create_agent parameter a default to enable this check"
        )
        return
    if runtime != param.default:
        raise ValueError(
            f"{param.source.path} does not match the instructions that the project uses at runtime: "
            "declare the file that the agent reads"
        )


def _rollout(runner: ProjectRunner) -> Any:
    def rollout(values: Mapping[str, Value], case: Case) -> RolloutOutput:
        try:
            response = runner.run(case, overrides=dict(values))
        except ProjectError as exc:
            if runner.runner and any(marker in str(exc).lower() for marker in TRANSIENT_RUNNER_MARKERS):
                raise InfrastructureError(str(exc)) from exc
            raise
        return RolloutOutput(output=response["output"], trajectory=response.get("trajectory"))

    return rollout


def _evaluate(evaluators: Sequence[Evaluator]) -> Any:
    def evaluate(case: Case, result: RolloutOutput) -> list[EvaluatorRow]:
        experiment = Experiment(cases=[case], evaluators=list(evaluators))
        task_output = {"output": result.output, "trajectory": result.trajectory}
        report = asyncio.run(experiment.run_evaluations_async(lambda _: task_output, max_workers=1))
        rows = []
        for index, row in enumerate(report.cases):
            reason = report.reasons[index] if index < len(report.reasons) else ""
            output = EvaluationOutput(score=report.scores[index], test_pass=report.test_passes[index], reason=reason)
            details = report.detailed_results[index] if index < len(report.detailed_results) else []
            error = reason if reason.startswith("Evaluator error:") else None
            rows.append(EvaluatorRow(name=row["evaluator"], output=output, details=details, error=error))
        return rows

    return evaluate


def add_subparser(subparsers: Any, parent: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register `optimize`."""
    parser = subparsers.add_parser(
        "optimize",
        parents=[parent],
        help="optimize a source project on its cases: search on train, pick on validation, report test",
        description=(
            "Search for changes to the keys that a space declares. Score every candidate on the train cases, "
            "measure the top finalists and the baseline on validation, ship the best of them, and compare it "
            "with the baseline on test, for the report only. The run writes best.diff for review and never "
            "edits source; --apply writes the picked file texts when you ask. Source loading executes trusted "
            "project code. Each rollout, judge, and proposal can incur provider costs."
        ),
    )
    parser.add_argument("project", nargs="?", help="project directory or source entry point")
    parser.add_argument("--project", dest="project_option", help="project directory or source entry point")
    parser.add_argument("--cases", metavar="DIRECTORY", help="directory of capture-v1 JSON cases")
    tunables = parser.add_mutually_exclusive_group()
    tunables.add_argument(
        "--space", metavar="FILE", help="JSON file of markers: FileText, TextValue, Int, Float, OneOf"
    )
    tunables.add_argument("--instructions-file", metavar="FILE", help="shortcut: one FileText key named instructions")
    parser.add_argument(
        "--objective",
        action="append",
        metavar="SPEC",
        help="maximize:METRIC or minimize:METRIC[:weight=N][:scale=N]; repeat for a weighted sum "
        "(default: maximize every evaluator)",
    )
    judge = parser.add_mutually_exclusive_group()
    judge.add_argument("--evaluator", choices=["contains"], help="use each case assertion as a substring check")
    judge.add_argument("--contains", metavar="TEXT", help="use one fixed substring check for every case")
    proposal = parser.add_mutually_exclusive_group()
    proposal.add_argument("--proposal-model", metavar="MODEL_ID", help="model of the Reflection proposer")
    proposal.add_argument("--proposal-file", metavar="FILE", help="fixed patches for an offline run")
    parser.add_argument("--split", metavar="TRAIN,VALIDATION[,TEST]", help="case shares (default 0.4,0.3,0.3)")
    parser.add_argument("--seed", type=int, help="run seed for the split and the random draws (default 0)")
    parser.add_argument("--max-rollouts", type=int, help="rollout cap; one rollout runs one case once (default 500)")
    parser.add_argument("--max-iterations", type=int, help="iteration cap (default: no cap)")
    parser.add_argument("--finalists", type=int, help="candidates measured on validation (default 3)")
    parser.add_argument("--runs-per-candidate", type=int, help="runs of each case per candidate (default 1)")
    parser.add_argument("--proposals-per-iteration", type=int, help="patches per proposer turn (default 4)")
    parser.add_argument("--crash-score", metavar="worst|none|NUMBER", help="score of an agent failure (default worst)")
    parser.add_argument("--rerun-passes", type=int, help="re-runs after an infrastructure failure (default 2)")
    parser.add_argument("--runner", metavar="EXECUTABLE", help="external runner executable (required for JS/TS)")
    parser.add_argument("--collector-endpoint", metavar="URL", help="OTLP collector HTTP(S) endpoint")
    parser.add_argument("--base-dir", metavar="DIRECTORY", help="parent of run folders (default .strands-evals/...)")
    parser.add_argument("--name", help="run folder prefix: lowercase letters, digits, '-', and '_'")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", metavar="RUN_DIR", help="continue a killed run with the settings in its run.json")
    modes.add_argument("--apply", metavar="RUN_DIR", help="write the picked candidate's file texts into the project")
    parser.add_argument("--candidate", metavar="ID", help="with --apply: apply this candidate, not the pick")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, or the files --apply would change")
    parser.set_defaults(func=_run)
    return parser
