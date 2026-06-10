"""Top-level argparse setup and dispatch for `strands-evals`."""

from __future__ import annotations

import argparse
import sys

from . import _common
from .commands import diagnose as diagnose_cmd
from .commands import report as report_cmd
from .commands import run as run_cmd
from .commands import validate as validate_cmd


def _build_parser() -> argparse.ArgumentParser:
    parent = _common.make_global_parent()

    parser = argparse.ArgumentParser(
        prog="strands-evals",
        description="Evaluate, diagnose, and inspect Strands Agents experiments.",
        parents=[parent],
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    validate_cmd.add_subparser(subparsers, parent)
    report_cmd.add_subparser(subparsers, parent)
    diagnose_cmd.add_subparser(subparsers, parent)
    run_cmd.add_subparser(subparsers, parent)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `strands-evals` console script.

    Returns the process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    _common.configure_logging(args)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return _common.EXIT_BAD_INPUT

    return _common.run_command(func, args)
