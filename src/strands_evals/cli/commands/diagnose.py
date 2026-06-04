"""`strands-evals diagnose` — failure detection and/or root cause analysis."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...detectors import analyze_root_cause, detect_failures, diagnose_session
from ...types.detector import ConfidenceLevel, DiagnosisResult, FailureOutput, RCAOutput
from ...types.trace import Session
from .._common import EXIT_OK, resolve_format
from .._io import read_text_input, write_text_output


def _render_diagnosis(result: DiagnosisResult) -> None:
    console = Console()
    console.print(Panel(f"Diagnosis for session [bold]{result.session_id}[/bold]"))

    if not result.failures:
        console.print("[green]No failures detected.[/green]")
        return

    failures_table = Table(title="Failures", show_lines=True)
    failures_table.add_column("span_id", style="cyan")
    failures_table.add_column("category", style="magenta")
    failures_table.add_column("confidence", style="yellow")
    failures_table.add_column("evidence")
    for failure in result.failures:
        failures_table.add_row(
            failure.span_id,
            ", ".join(failure.category),
            ", ".join(f"{c:.2f}" for c in failure.confidence),
            "\n".join(failure.evidence),
        )
    console.print(failures_table)

    if result.root_causes:
        rca_table = Table(title="Root causes", show_lines=True)
        rca_table.add_column("failure_span_id", style="cyan")
        rca_table.add_column("location", style="magenta")
        rca_table.add_column("causality", style="yellow")
        rca_table.add_column("explanation")
        rca_table.add_column("recommendation", style="green")
        for rca in result.root_causes:
            rca_table.add_row(
                rca.failure_span_id,
                rca.location,
                rca.causality,
                rca.root_cause_explanation,
                rca.fix_recommendation,
            )
        console.print(rca_table)


def _render_failure_output(output: FailureOutput) -> None:
    console = Console()
    console.print(Panel(f"Failures for session [bold]{output.session_id}[/bold]"))
    if not output.failures:
        console.print("[green]No failures detected.[/green]")
        return
    table = Table(show_lines=True)
    table.add_column("span_id", style="cyan")
    table.add_column("category", style="magenta")
    table.add_column("confidence", style="yellow")
    table.add_column("evidence")
    for failure in output.failures:
        table.add_row(
            failure.span_id,
            ", ".join(failure.category),
            ", ".join(f"{c:.2f}" for c in failure.confidence),
            "\n".join(failure.evidence),
        )
    console.print(table)


def _render_rca_output(output: RCAOutput) -> None:
    console = Console()
    if not output.root_causes:
        console.print("[yellow]No root causes produced.[/yellow]")
        return
    table = Table(title="Root causes", show_lines=True)
    table.add_column("failure_span_id", style="cyan")
    table.add_column("location", style="magenta")
    table.add_column("causality", style="yellow")
    table.add_column("explanation")
    table.add_column("recommendation", style="green")
    for rca in output.root_causes:
        table.add_row(
            rca.failure_span_id,
            rca.location,
            rca.causality,
            rca.root_cause_explanation,
            rca.fix_recommendation,
        )
    console.print(table)


def _run(args: argparse.Namespace) -> int:
    raw = read_text_input(args.session_file)
    session = Session.model_validate_json(raw)

    confidence = ConfidenceLevel(args.confidence)

    if args.detect_only:
        result: DiagnosisResult | FailureOutput | RCAOutput = detect_failures(
            session,
            confidence_threshold=confidence,
            model=args.model,
        )
    elif args.rca_only:
        result = analyze_root_cause(session, model=args.model)
    else:
        result = diagnose_session(
            session,
            model=args.model,
            confidence_threshold=confidence,
        )

    fmt = resolve_format(args)
    if fmt == "json" or args.output is not None:
        write_text_output(result.model_dump_json(indent=2), args.output)
    else:
        if isinstance(result, DiagnosisResult):
            _render_diagnosis(result)
        elif isinstance(result, FailureOutput):
            _render_failure_output(result)
        else:
            _render_rca_output(result)

        # Always also emit a one-line summary on stderr for scripting.
        if isinstance(result, DiagnosisResult):
            print(  # noqa: T201
                f"diagnosis: {len(result.failures)} failure(s), {len(result.root_causes)} root cause(s)",
                file=sys.stderr,
            )
        elif isinstance(result, FailureOutput):
            print(f"failures: {len(result.failures)}", file=sys.stderr)  # noqa: T201
        else:
            print(f"root_causes: {len(result.root_causes)}", file=sys.stderr)  # noqa: T201

    return EXIT_OK


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "diagnose",
        parents=[parent],
        help="detect failures and/or analyze root causes for a session",
        description=(
            "Load a Session JSON file (path or '-' for stdin) and run failure detection, "
            "root cause analysis, or the full diagnose_session pipeline."
        ),
    )
    parser.add_argument(
        "session_file",
        metavar="SESSION_FILE",
        help="path to a Session JSON file, or '-' to read from stdin",
    )
    parser.add_argument(
        "--confidence",
        choices=[level.value for level in ConfidenceLevel],
        default=ConfidenceLevel.LOW.value,
        help="minimum confidence threshold for failure detection (default: low)",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        default=None,
        help="override the judge model for detection and RCA",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--detect-only",
        action="store_true",
        help="run only detect_failures and skip root cause analysis",
    )
    mode.add_argument(
        "--rca-only",
        action="store_true",
        help="run only analyze_root_cause (failures are inferred from the session)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="write the diagnosis JSON to PATH instead of rendering to the console",
    )
    parser.set_defaults(func=_run)
    return parser
