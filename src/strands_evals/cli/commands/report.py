"""`strands-evals report` — render or dump an existing EvaluationReport."""

from __future__ import annotations

import argparse

from ...types.evaluation_report import EvaluationReport
from .._common import EXIT_OK, resolve_format
from .._io import read_text_input, write_text_output


def _run(args: argparse.Namespace) -> int:
    raw = read_text_input(args.reports_file)
    report = EvaluationReport.model_validate_json(raw)

    # `-o PATH` always writes JSON to disk regardless of --interactive/--rich,
    # so callers piping through `report` to persist a file get a stable format.
    if args.output is not None:
        write_text_output(report.model_dump_json(indent=2), args.output)
        return EXIT_OK

    # --interactive forces the interactive rich path (run_display), overriding
    # an explicit --json since the two are incompatible — interactive mode owns
    # stdout for the live table.
    if args.interactive:
        report.run_display(include_recommendations=args.recommendations)
        return EXIT_OK

    fmt = resolve_format(args)
    if fmt == "json":
        write_text_output(report.model_dump_json(indent=2), None)
    else:
        report.display(include_recommendations=args.recommendations)

    return EXIT_OK


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "report",
        parents=[parent],
        help="render an existing EvaluationReport",
        description=(
            "Load an EvaluationReport JSON document (path or '-' for stdin) and either "
            "render it via Rich or dump JSON. With -o, always writes JSON to the file."
        ),
    )
    parser.add_argument(
        "reports_file",
        metavar="REPORTS_FILE",
        help="path to an EvaluationReport JSON file, or '-' to read from stdin",
    )
    parser.add_argument(
        "--recommendations",
        action="store_true",
        help="include diagnosis recommendations in the rich-rendered table",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "render an interactive rich table where rows can be expanded/collapsed "
            "(EvaluationReport.run_display). Implies rich format; ignored when -o is set."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="write the report JSON to PATH instead of stdout/console",
    )
    parser.set_defaults(func=_run)
    return parser
