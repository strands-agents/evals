"""`strands-evals run` — execute an experiment against an agent or task callable."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from ...display.display_console import CollapsibleTableReportDisplay
from ...evaluators.evaluator import Evaluator
from ...experiment import Experiment
from ...local_file_task_result_store import LocalFileTaskResultStore
from ...types.detector import ConfidenceLevel, DiagnosisConfig, DiagnosisTrigger
from ...types.evaluation_report import EvaluationReport
from .._agent_task import synthesize_task_function
from .._common import EXIT_FAILURES, EXIT_OK, resolve_format
from .._entrypoint import (
    EntryPointError,
    import_attr,
    resolve_agent,
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


def _resolve_custom_evaluators(specs: list[str]) -> list[type[Evaluator]]:
    classes: list[type[Evaluator]] = []
    for spec in specs:
        obj = import_attr(spec)
        if not isinstance(obj, type) or not issubclass(obj, Evaluator):
            raise EntryPointError(
                f"--custom-evaluator '{spec}' must reference a subclass of "
                f"strands_evals.evaluators.Evaluator (got {type(obj).__name__})"
            )
        classes.append(obj)
    return classes


def _build_diagnosis_config(args: argparse.Namespace) -> DiagnosisConfig | None:
    if args.diagnose is None:
        return None
    return DiagnosisConfig(
        trigger=DiagnosisTrigger(args.diagnose),
        confidence_threshold=ConfidenceLevel(args.confidence),
    )


def _decide_exit_code(reports: list[EvaluationReport], fail_on: tuple[str, float | None]) -> int:
    """Map `--fail-on` to an exit code based on the flattened reports."""
    mode, threshold = fail_on
    if mode == "none":
        return EXIT_OK

    flattened = EvaluationReport.flatten(reports)
    if mode == "any":
        return EXIT_FAILURES if any(not p for p in flattened.test_passes) else EXIT_OK

    # mode == "threshold"
    assert threshold is not None
    return EXIT_FAILURES if flattened.overall_score < threshold else EXIT_OK


def _print_summary(reports: list[EvaluationReport]) -> None:
    """Emit the stable per-evaluator summary line on stderr."""
    parts: list[str] = ["strands-evals run"]
    for report in reports:
        passed = sum(1 for p in report.test_passes if p)
        total = len(report.test_passes)
        parts.append(f"{report.evaluator_name}: {passed}/{total} passed ({report.overall_score:.2f})")

    flattened = EvaluationReport.flatten(reports)
    parts.append(f"overall: {flattened.overall_score:.2f}")
    print(" | ".join(parts), file=sys.stderr)  # noqa: T201


def _trace_attrs_dict(pairs: list[tuple[str, str]] | None) -> dict[str, Any]:
    if not pairs:
        return {}
    merged: dict[str, Any] = {}
    for key, value in pairs:
        merged[key] = value
    return merged


def _display_expanded(report: EvaluationReport, *, include_recommendations: bool) -> None:
    """Render the flattened report with every row pre-expanded.

    `EvaluationReport.display()` collapses non-core columns to literal "..."
    in static mode. For `run --display` we want the values visible without
    requiring the user to drop into the interactive prompt, so we build the
    same items dict the library does and flip `expanded=True`.
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
        details["input"] = EvaluationReport._format_input_for_display(case.get("input"))
        details["actual_output"] = str(case.get("actual_output"))
        details["expected_output"] = str(case.get("expected_output"))
        if include_recommendations:
            rec = report.recommendations[i] if i < len(report.recommendations) else None
            if rec is not None:
                details["recommendation"] = rec

        items[str(i)] = {
            "details": details,
            "detailed_results": report.detailed_results[i] if i < len(report.detailed_results) else [],
            "expanded": True,
        }

    CollapsibleTableReportDisplay(items=items, overall_score=report.overall_score).run(static=True)


def _run(args: argparse.Namespace) -> int:
    custom_evaluators = _resolve_custom_evaluators(args.custom_evaluator or [])
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
        if entry.kind == "agent_instance" and args.trace_attributes:
            logger.warning(
                "trace_attributes=<%s> | --trace-attributes ignored: --agent resolved to a prebuilt Agent instance",
                args.trace_attributes,
            )
        task_function = synthesize_task_function(
            entry,
            extra_trace_attributes=_trace_attrs_dict(args.trace_attributes),
        )
    else:
        entry = resolve_task(args.task)
        if args.trace_attributes:
            logger.warning("--trace-attributes is ignored when using --task; the user owns agent instantiation")
        task_function = entry.obj

    data_store = LocalFileTaskResultStore(args.data_store) if args.data_store else None

    reports = asyncio.run(
        experiment.run_evaluations_async(
            task_function,
            max_workers=args.max_workers,
            evaluation_data_store=data_store,
        )
    )

    flattened = EvaluationReport.flatten(reports)

    if args.output is not None:
        write_text_output(flattened.model_dump_json(indent=2), args.output)

    if args.display:
        _display_expanded(flattened, include_recommendations=args.diagnose is not None)
    elif args.output is None:
        fmt = resolve_format(args)
        if fmt == "json":
            sys.stdout.write(flattened.model_dump_json(indent=2))
            sys.stdout.write("\n")

    _print_summary(reports)

    if args.exit_zero:
        return EXIT_OK
    return _decide_exit_code(reports, args.fail_on)


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
        help="entry point for a strands.Agent instance, subclass, or factory callable",
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
        help="extra OTel trace attributes (repeatable). No-op for --agent <instance> and --task.",
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

    parser.set_defaults(func=_run)
    return parser
