"""`strands-evals run` — execute an experiment against an agent or task callable."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from ...display.display_console import CollapsibleTableReportDisplay
from ...experiment import Experiment
from ...local_file_task_result_store import LocalFileTaskResultStore
from ...types.detector import ConfidenceLevel, DiagnosisConfig, DiagnosisTrigger
from ...types.evaluation_report import EvaluationReport
from .._agent_task import synthesize_task_function
from .._common import EXIT_FAILURES, EXIT_OK, resolve_format
from .._entrypoint import (
    resolve_agent,
    resolve_custom_evaluators,
    resolve_task,
)
from .._io import write_text_output

logger = logging.getLogger(__name__)


def _parse_trace_attribute(raw: str) -> tuple[str, str]:
    """Parse a `KEY=VALUE` pair from a `--trace-attributes` argument."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--trace-attributes value must be KEY=VALUE, got '{raw}'")
    key, value = raw.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"--trace-attributes KEY may not be empty (got '{raw}')")
    return key, value


def _parse_fail_on(raw: str) -> tuple[str, float | None]:
    """Parse `--fail-on={any|none|threshold:0.X}` into `(mode, threshold)`."""
    if raw in ("any", "none"):
        return raw, None
    if raw.startswith("threshold:"):
        try:
            threshold = float(raw.split(":", 1)[1])
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"invalid threshold in '{raw}'") from e
        if not 0.0 <= threshold <= 1.0:
            raise argparse.ArgumentTypeError(f"threshold in '{raw}' must be between 0.0 and 1.0")
        return "threshold", threshold
    raise argparse.ArgumentTypeError(f"--fail-on must be 'any', 'none', or 'threshold:0.X' (got '{raw}')")


def _build_diagnosis_config(args: argparse.Namespace) -> DiagnosisConfig | None:
    if args.diagnose is None:
        return None
    return DiagnosisConfig(
        trigger=DiagnosisTrigger(args.diagnose),
        confidence_threshold=ConfidenceLevel(args.confidence),
    )


def _decide_exit_code(report: EvaluationReport, fail_on: tuple[str, float | None]) -> int:
    """Map `--fail-on` to an exit code based on the flattened report."""
    mode, threshold = fail_on
    if mode == "none":
        return EXIT_OK

    if mode == "any":
        return EXIT_FAILURES if any(not p for p in report.test_passes) else EXIT_OK

    # mode == "threshold"
    if threshold is None:
        raise ValueError("--fail-on=threshold:X requires a numeric threshold")
    return EXIT_FAILURES if report.overall_score < threshold else EXIT_OK


def _print_summary(report: EvaluationReport) -> None:
    """Emit the stable per-evaluator summary line on stderr.

    `run_evaluations_async` returns a single flattened report whose `cases`
    rows are tagged with an `evaluator` key; regroup by that tag so the
    summary still reads `<evaluator>: P/T passed (avg)` per evaluator.
    """
    by_eval: dict[str, list[int]] = {}
    by_eval_scores: dict[str, list[float]] = {}
    for i, case in enumerate(report.cases):
        name = case.get("evaluator", "unknown")
        by_eval.setdefault(name, []).append(int(report.test_passes[i]))
        by_eval_scores.setdefault(name, []).append(report.scores[i])

    parts: list[str] = ["strands-evals run"]
    for name, passes in by_eval.items():
        passed = sum(passes)
        total = len(passes)
        scores = by_eval_scores[name]
        avg = sum(scores) / len(scores) if scores else 0.0
        parts.append(f"{name}: {passed}/{total} passed ({avg:.2f})")

    parts.append(f"overall: {report.overall_score:.2f}")
    print(" | ".join(parts), file=sys.stderr)  # noqa: T201


def _trace_attrs_dict(pairs: list[tuple[str, str]] | None) -> dict[str, Any]:
    if not pairs:
        return {}
    merged: dict[str, Any] = {}
    for key, value in pairs:
        merged[key] = value
    return merged


def _display_expanded(
    report: EvaluationReport,
    *,
    include_recommendations: bool,
    interactive: bool = False,
) -> None:
    """Render the flattened report as a Rich table.

    Static mode (default) pre-expands every row so values are visible without
    dropping into a prompt — `EvaluationReport.display()` would collapse the
    non-core columns to literal "...". Interactive mode (`--interactive`) seeds
    rows collapsed and hands control to `CollapsibleTableReportDisplay.run`'s
    expand/collapse loop, matching `EvaluationReport.run_display()`.
    """
    items: dict[str, dict[str, Any]] = {}
    for i in range(len(report.scores)):
        case = report.cases[i] if i < len(report.cases) else {}
        details: dict[str, Any] = {"name": case.get("name", f"Test {i + 1}")}
        if "evaluator" in case:
            details["evaluator"] = case["evaluator"]
        details["score"] = f"{report.scores[i]:.2f}"
        details["test_pass"] = report.test_passes[i] if i < len(report.test_passes) else False
        details["reason"] = report.reasons[i] if i < len(report.reasons) else ""
        details["input"] = EvaluationReport.format_input_for_display(case.get("input"))
        details["actual_output"] = str(case.get("actual_output"))
        details["expected_output"] = str(case.get("expected_output"))
        if include_recommendations:
            rec = report.recommendations[i] if i < len(report.recommendations) else None
            if rec is not None:
                details["recommendation"] = rec

        items[str(i)] = {
            "details": details,
            "detailed_results": report.detailed_results[i] if i < len(report.detailed_results) else [],
            "expanded": not interactive,
        }

    CollapsibleTableReportDisplay(items=items, overall_score=report.overall_score).run(static=not interactive)


def _run(args: argparse.Namespace) -> int:
    custom_evaluators = resolve_custom_evaluators(args.custom_evaluator or [])
    loaded = Experiment.from_file(args.experiment_file, custom_evaluators=custom_evaluators)

    diagnosis_config = _build_diagnosis_config(args)
    # Experiment.from_file does not accept diagnosis_config; rebuild via the
    # public constructor so we don't poke private attrs. Cases are deep-copied
    # by the property accessor, which is fine for a one-shot CLI run.
    experiment = Experiment(
        cases=loaded.cases,
        evaluators=loaded.evaluators,
        diagnosis_config=diagnosis_config,
    )

    if args.agent is not None:
        entry = resolve_agent(args.agent)
        task_function = synthesize_task_function(
            entry,
            extra_trace_attributes=_trace_attrs_dict(args.trace_attributes),
        )
    else:
        entry = resolve_task(args.task)
        if args.trace_attributes:
            logger.warning(
                "trace_attributes=<%s> | --trace-attributes ignored when using --task; "
                "the user owns agent instantiation",
                args.trace_attributes,
            )
        task_function = entry.obj

    data_store = LocalFileTaskResultStore(args.data_store) if args.data_store else None

    report = asyncio.run(
        experiment.run_evaluations_async(
            task_function,
            max_workers=args.max_workers,
            evaluation_data_store=data_store,
        )
    )

    if args.output is not None:
        write_text_output(report.model_dump_json(indent=2), args.output)

    if args.display or args.interactive:
        _display_expanded(
            report,
            include_recommendations=args.diagnose is not None,
            interactive=args.interactive,
        )
    elif args.output is None:
        # `run` intentionally only emits JSON to stdout (or nothing, when -o is set).
        # Rich rendering is opt-in via `--display`: building the table eagerly walks
        # every case row and is wasteful on large experiments where users typically
        # pipe to `strands-evals report` or a file.
        fmt = resolve_format(args)
        if fmt == "json":
            sys.stdout.write(report.model_dump_json(indent=2))
            sys.stdout.write("\n")

    _print_summary(report)

    if args.exit_zero:
        return EXIT_OK
    return _decide_exit_code(report, args.fail_on)


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "run",
        parents=[parent],
        help="run an experiment against an agent or task callable",
        description=(
            "Load an Experiment, resolve --agent or --task, execute "
            "run_evaluations_async, and write a flattened EvaluationReport. "
            "--agent synthesizes the standard task wrapper (telemetry → invoke "
            "→ Session mapping); --task is the escape hatch for custom shapes."
        ),
    )
    parser.add_argument(
        "experiment_file",
        metavar="EXPERIMENT_FILE",
        help="path to a serialized Experiment JSON file",
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--agent",
        metavar="MODULE:ATTR",
        help=(
            "entry point for a factory callable that returns a strands.Agent. "
            "Zero-arg ('def build_agent() -> Agent: ...') or one-arg "
            "('def build_agent(case: Case) -> Agent: ...'). The factory is "
            "invoked once per case so each case runs against a fresh Agent."
        ),
    )
    target.add_argument(
        "--task",
        metavar="MODULE:ATTR",
        help="entry point for a Callable[[Case], dict|str] (escape hatch)",
    )

    parser.add_argument(
        "--trace-attributes",
        metavar="KEY=VALUE",
        action="append",
        type=_parse_trace_attribute,
        default=None,
        help=(
            "extra OTel trace attributes (repeatable). Set as W3C Baggage on "
            "the per-case context and stamped on every span emitted by the "
            "agent. session.id and gen_ai.conversation.id are always set "
            "from the case; --trace-attributes is for additional keys. "
            "No-op when --task is used (the user owns agent instantiation)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="max parallel workers for run_evaluations_async (default: 1)",
    )
    parser.add_argument(
        "--data-store",
        metavar="DIR",
        default=None,
        help="LocalFileTaskResultStore directory; cached task outputs short-circuit reruns",
    )
    parser.add_argument(
        "--diagnose",
        choices=["on_failure", "always"],
        default=None,
        help="run diagnose_session on cases (on failure or always); requires Session trajectories",
    )
    parser.add_argument(
        "--confidence",
        choices=[level.value for level in ConfidenceLevel],
        default=ConfidenceLevel.MEDIUM.value,
        help="confidence threshold passed to DiagnosisConfig (default: medium)",
    )
    parser.add_argument(
        "--custom-evaluator",
        metavar="MODULE:CLASS",
        action="append",
        default=None,
        help="custom Evaluator subclass to register before from_file (repeatable)",
    )
    parser.add_argument(
        "--fail-on",
        type=_parse_fail_on,
        default=("any", None),
        help="exit-code rule: any | none | threshold:0.X (default: any)",
    )
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="always exit 0 regardless of --fail-on result",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="write the flattened reports JSON to PATH instead of stdout",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "render the flattened report as a Rich table on stdout "
            "(includes input, expected_output, actual_output; recommendations "
            "when --diagnose is set)"
        ),
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "render the flattened report as an interactive Rich table where "
            "rows can be expanded/collapsed (implies --display)"
        ),
    )

    parser.set_defaults(func=_run)
    return parser
