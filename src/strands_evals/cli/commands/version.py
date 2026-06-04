"""`strands-evals version` — print the package and Python versions."""

from __future__ import annotations

import argparse
import platform
from importlib.metadata import PackageNotFoundError, version


def _package_version() -> str:
    try:
        return version("strands-agents-evals")
    except PackageNotFoundError:
        return "unknown"


def _run(args: argparse.Namespace) -> int:
    del args
    print(f"strands-agents-evals {_package_version()} (python {platform.python_version()})")  # noqa: T201
    return 0


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "version",
        parents=[parent],
        help="print package and Python versions",
        description="Print the installed strands-agents-evals version and the Python version.",
    )
    parser.set_defaults(func=_run)
    return parser
