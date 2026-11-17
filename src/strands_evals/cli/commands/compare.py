"""`strands-evals compare` — statistically compare paired evaluation reports."""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ...types.evaluation_report import EvaluationReport
from .._common import EXIT_BAD_INPUT, EXIT_OK, emit_error, resolve_format
from .._io import read_text_input, write_text_output

if TYPE_CHECKING:
    from ...experimental.comparison import PairedComparisonResult


def _load_reports(paths: list[str], excluded_indices: set[int]) -> list[EvaluationReport]:
    reports: list[EvaluationReport] = []
    for index, path in enumerate(paths):
        if index in excluded_indices:
            reports.append(
                EvaluationReport(
                    overall_score=0.0,
                    scores=[],
                    cases=[],
                    test_passes=[],
                    detailed_results=[],
                )
            )
            continue

        try:
            report = EvaluationReport.model_validate_json(read_text_input(path))
        except (OSError, ValueError) as error:
            source = "stdin" if path == "-" else path
            raise ValueError(f"invalid report at trial {index}, source={source}: {error}") from error
        reports.append(report)
    return reports


def _render_result(result: PairedComparisonResult) -> None:
    console = Console()
    trial_label = "trial" if result.trial_count == 1 else "trials"
    title = f"Comparison: {result.verdict.replace('_', ' ')} ({result.trial_count} {trial_label})"
    if console.width < 100:
        console.print(Text(title, style="bold"))
        for summary in result.evaluator_summaries:
            interval = (
                f"[{summary.confidence_lower:.4f}, {summary.confidence_upper:.4f}]"
                if summary.confidence_lower is not None and summary.confidence_upper is not None
                else "N/A"
            )
            adjusted_p = f"{summary.adjusted_p_value:.4g}" if summary.adjusted_p_value is not None else "N/A"
            details = Table.grid(padding=(0, 1))
            details.add_column(style="bold", no_wrap=True)
            details.add_column(overflow="fold")
            details.add_row("Evaluator", Text(summary.evaluator))
            details.add_row("Cases", str(summary.case_count))
            details.add_row("Changed", str(summary.changed_case_count))
            details.add_row("Delta", f"{summary.mean_delta:+.4f}")
            details.add_row("CI", interval)
            details.add_row("Adj. p", adjusted_p)
            details.add_row("Verdict", summary.verdict.replace("_", " "))
            console.print(details)
        return

    table = Table(title=title)
    table.add_column("Evaluator", overflow="fold")
    table.add_column("Cases", justify="right", no_wrap=True)
    table.add_column("Changed", justify="right", no_wrap=True)
    table.add_column("Delta", justify="right", no_wrap=True)
    table.add_column("CI", justify="right", no_wrap=True)
    table.add_column("Adj. p", justify="right", no_wrap=True)
    table.add_column("Verdict", overflow="fold")

    for summary in result.evaluator_summaries:
        interval = (
            f"[{summary.confidence_lower:.4f}, {summary.confidence_upper:.4f}]"
            if summary.confidence_lower is not None and summary.confidence_upper is not None
            else "N/A"
        )
        adjusted_p = f"{summary.adjusted_p_value:.4g}" if summary.adjusted_p_value is not None else "N/A"
        table.add_row(
            Text(summary.evaluator),
            str(summary.case_count),
            str(summary.changed_case_count),
            f"{summary.mean_delta:+.4f}",
            interval,
            adjusted_p,
            summary.verdict.replace("_", " "),
        )

    console.print(table)


def _run(args: argparse.Namespace) -> int:
    try:
        corrupted = _validate_args(args)

        # optional extra: comparison
        from ...experimental.comparison import compare_paired

        result = compare_paired(
            baseline=_load_reports(args.baseline, corrupted),
            candidate=_load_reports(args.candidate, corrupted),
            corrupted_trial_indices=corrupted,
            confidence_level=args.confidence_level,
            minimum_effect=args.minimum_effect,
            lower_is_better=args.lower_is_better,
        )
    except (ImportError, ValueError) as error:
        message = str(error)
        if message.startswith("lower_is_better "):
            message = f"--lower-is-better {message.removeprefix('lower_is_better ')}"
        emit_error(message, debug=getattr(args, "debug", False), exc=error)
        return EXIT_BAD_INPUT

    if args.output is not None or resolve_format(args) == "json":
        write_text_output(result.model_dump_json(indent=2), args.output)
    else:
        _render_result(result)

    return EXIT_OK


def _validate_args(args: argparse.Namespace) -> set[int]:
    if len(args.baseline) != len(args.candidate):
        raise ValueError(
            "--baseline and --candidate must be provided the same number of times: "
            f"baseline={len(args.baseline)}, candidate={len(args.candidate)}"
        )
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence-level must be greater than 0 and less than 1")
    if not math.isfinite(args.minimum_effect) or args.minimum_effect < 0.0:
        raise ValueError("--minimum-effect must be finite and greater than or equal to 0")

    trial_indices = set(range(len(args.baseline)))
    corrupted = set(args.corrupted_trial)
    unknown_corrupted = corrupted - trial_indices
    if unknown_corrupted:
        raise ValueError(
            "--corrupted-trial contains unknown indices: "
            f"unknown={sorted(unknown_corrupted)}, valid={sorted(trial_indices)}"
        )
    if corrupted == trial_indices:
        raise ValueError("--corrupted-trial excludes every supplied trial")
    return corrupted


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "compare",
        parents=[parent],
        help="experimentally compare paired evaluation reports",
        description=(
            "Experimental: compare baseline and candidate EvaluationReport JSON files. "
            "Cases are the statistical unit; repeated trials only reduce per-case noise. "
            "Repeat --baseline and --candidate in matching trial order."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        metavar="REPORT",
        help="baseline EvaluationReport JSON file, or - for stdin; repeat once per trial",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="REPORT",
        help="candidate EvaluationReport JSON file, or - for stdin; repeat once per trial",
    )
    parser.add_argument(
        "--corrupted-trial",
        action="append",
        default=[],
        type=int,
        metavar="INDEX",
        help="zero-based trial index to skip before reading either report; may be repeated",
    )
    parser.add_argument(
        "--confidence-level",
        default=0.95,
        type=float,
        help=(
            "confidence interval coverage and significance threshold complement; "
            "at 0.95, at least 6 changed cases are needed before adjustment (default: 0.95)"
        ),
    )
    parser.add_argument(
        "--minimum-effect",
        default=0.0,
        type=float,
        help="require the full confidence interval beyond this effect threshold (default: 0.0)",
    )
    parser.add_argument(
        "--lower-is-better",
        action="append",
        default=[],
        metavar="EVALUATOR",
        help="evaluator whose lower score is beneficial; may be repeated",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="PATH",
        help="write comparison JSON to PATH instead of stdout",
    )
    parser.set_defaults(func=_run)
    return parser
