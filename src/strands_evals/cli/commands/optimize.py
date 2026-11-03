"""`strands-evals optimize` — a bounded, eval-gated instructions-only POC."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from .._common import EXIT_BAD_INPUT, EXIT_OK, EXIT_RUNTIME_ERROR, emit_error
from .._optimization import InstructionOptimizer, digest, generate_proposal


def _source_digest(root: Path) -> str:
    """Fingerprint project files, excluding generated state and dependency trees."""
    excluded = {".strands", ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "dist", "build"}
    files = {
        path.relative_to(root).as_posix(): path.read_bytes().hex()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and not any(part in excluded for part in path.relative_to(root).parts)
    }
    return digest(files)


def _load_proposals(path: str | None, budget: int) -> list[str] | None:
    if path is None:
        return None
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        proposals = json.loads(text)
        if not isinstance(proposals, list) or not all(isinstance(item, str) and item.strip() for item in proposals):
            raise ValueError("--proposal-file JSON must be a nonempty list of instructions strings")
    else:
        proposals = [text]
    if len(proposals) < budget:
        raise ValueError("--proposal-file must contain at least --budget proposals (plain text provides one)")
    if any(not item.strip() for item in proposals):
        raise ValueError("proposals must not be empty")
    return proposals


def _run(args: argparse.Namespace) -> int:
    # Project source and runner output must not corrupt machine-readable stdout.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = _execute(args)
    except (ValueError, OSError) as exc:
        emit_error(str(exc))
        return EXIT_BAD_INPUT
    except Exception as exc:
        emit_error(f"optimization failed: {exc}", debug=getattr(args, "debug", False), exc=exc)
        return EXIT_RUNTIME_ERROR
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return EXIT_OK


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    from .._project import ProjectRunner, evaluate_project, load_project_cases

    project_arg = args.project_option or args.project
    if not project_arg or (args.project_option and args.project):
        raise ValueError("provide exactly one project via --project PATH or positional PATH")
    project = Path(project_arg).expanduser().resolve()
    if not project.exists():
        raise ValueError(f"project does not exist: {project}")
    root = project if project.is_dir() else project.parent
    instructions_path = Path(args.instructions_file).expanduser().resolve()
    if not instructions_path.is_relative_to(root) or not instructions_path.is_file():
        raise ValueError("--instructions-file must be an existing file inside the project")
    original = instructions_path.read_text(encoding="utf-8")
    if not original.strip():
        raise ValueError("--instructions-file must contain nonempty instructions")
    cases = load_project_cases(args.cases)
    proposals = _load_proposals(args.proposal_file, args.budget)
    if not args.dry_run and proposals is None and not args.proposal_model:
        raise ValueError("provide --proposal-model explicitly, or --proposal-file for offline proposals")
    # Case generates a random session ID when captures do not provide one.
    # Stabilize it so both split membership and resume identity derive from data.
    serialized = []
    for case in cases:
        data = case.model_dump(mode="json", exclude={"session_id"})
        data["session_id"] = digest(data)
        serialized.append(data)
    source_digest = _source_digest(root)
    options = {
        "project": str(project),
        "instructions_file": str(instructions_path),
        "dataset_files": {
            path.name: digest(path.read_bytes().hex()) for path in sorted(Path(args.cases).expanduser().iterdir())
        },
        "runner": args.runner,
        "runner_digest": digest(Path(args.runner).read_bytes().hex()) if args.runner else None,
        "proposal_model": args.proposal_model,
        "proposals": proposals,
        "allow_small_holdout": args.allow_small_holdout,
        "contains": args.contains,
        "evaluator": args.evaluator,
    }
    runner: Any = None

    def evaluate(instructions: str, subset: list[Any]) -> dict[str, Any]:
        # Subsets retain the exact original Case objects and ordering.
        from ...case import Case

        if _source_digest(root) != source_digest:
            raise ValueError("project source changed during optimization")
        kwargs: dict[str, Any] = {"overrides": {"instructions": instructions}}
        if args.contains is not None:
            from ...evaluators.deterministic import Contains

            kwargs["evaluators"] = [Contains(value=args.contains)]
        elif args.evaluator == "contains":
            from .._project import AssertionContains

            kwargs["evaluators"] = [AssertionContains()]
        report = evaluate_project(runner, [Case.model_validate(case) for case in subset], **kwargs)
        return report.model_dump(mode="json")

    def propose(instructions: str, feedback: list[dict[str, Any]], attempt: int) -> str:
        if proposals is not None:
            return proposals[attempt]
        return generate_proposal(args.proposal_model, instructions, feedback, attempt)

    optimizer = InstructionOptimizer(
        root=root,
        instructions_path=instructions_path,
        original=original,
        cases=serialized,
        source_digest=source_digest,
        options=options,
        evaluate=evaluate,
        propose=propose,
        budget=args.budget,
        holdout_size=args.holdout_size,
        allow_small_holdout=args.allow_small_holdout,
        per_case=True,
    )
    if args.dry_run:
        return {
            **optimizer.plan(),
            "dry_run": True,
            "source_extraction": "not executed: source import is deferred until a real run",
            "proposal_model": args.proposal_model,
            "instructions_file": str(instructions_path),
        }
    runner = ProjectRunner(project, runner=args.runner, collector_endpoint=args.collector_endpoint)
    extracted = runner.get_instructions()
    if extracted != original:
        raise ValueError("--instructions-file does not exactly match the instructions extracted from the project")
    return optimizer.run(args.resume)


def add_subparser(subparsers: Any, parent: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register an explicitly scoped optimizer rather than a general config tuner."""
    parser = subparsers.add_parser(
        "optimize",
        parents=[parent],
        help="bounded, eval-gated instructions-only optimization (POC; never edits source)",
        description=(
            "Propose replacement instructions, require strict training improvement, then confirm on a "
            "disjoint holdout. Writes a manual-apply patch, never project source. Source loading executes "
            "trusted project code. Each rollout and judge may incur provider costs."
        ),
    )
    parser.add_argument("project", nargs="?", help="project directory or source entrypoint")
    parser.add_argument(
        "--evaluator",
        choices=["contains"],
        default=None,
        help="deterministic gate using each case assertion as a substring",
    )
    parser.add_argument("--project", dest="project_option", help="project directory or source entrypoint")
    parser.add_argument(
        "--contains", default=None, help="use a deterministic output substring gate instead of LLM judge"
    )
    parser.add_argument("--cases", required=True, help="directory containing capture-v1 JSON cases")
    parser.add_argument(
        "--instructions-file", required=True, help="existing project file matching runtime instructions"
    )
    parser.add_argument("--collector-endpoint", default=None)
    parser.add_argument(
        "--runner", default=None, help="external runner executable (must support instruction extraction)"
    )
    parser.add_argument("--budget", type=int, default=3, help="maximum number of proposed candidates (default: 3)")
    parser.add_argument("--proposal-model", default=None, help="explicit Strands proposal model ID")
    parser.add_argument("--proposal-file", default=None, help="offline plain-text candidate or JSON list of candidates")
    parser.add_argument("--holdout-size", type=int, default=5, help="disjoint holdout cases (default/minimum: 5)")
    parser.add_argument(
        "--allow-small-holdout", action="store_true", help="testing only: allow fewer than 5 holdout cases"
    )
    parser.add_argument("--resume", default=None, help="run ID or path directly under this project's .strands/runs")
    parser.add_argument(
        "--dry-run", action="store_true", help="show bounded plan without importing source or invoking models"
    )
    parser.set_defaults(func=_run)
    return parser
