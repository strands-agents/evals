"""Shared CLI helpers: exit codes, argument parents, format/verbosity handling."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from typing import TextIO

from pydantic import ValidationError

from ._entrypoint import EntryPointError

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_BAD_INPUT = 2
EXIT_RUNTIME_ERROR = 3


def make_global_parent() -> argparse.ArgumentParser:
    """Return the parent parser carrying flags that every subcommand accepts.

    The parent uses `add_help=False` so it can be passed via `parents=[...]`
    to subcommand parsers without colliding with their own `-h`.
    """
    parent = argparse.ArgumentParser(add_help=False)

    # Use SUPPRESS as the default so subparsers built with parents=[parent]
    # do not overwrite a root-level value when the flag isn't repeated.
    fmt = parent.add_mutually_exclusive_group()
    fmt.add_argument(
        "--json",
        dest="format",
        action="store_const",
        const="json",
        default=argparse.SUPPRESS,
        help="emit machine-readable JSON to stdout",
    )
    fmt.add_argument(
        "--rich",
        dest="format",
        action="store_const",
        const="rich",
        default=argparse.SUPPRESS,
        help="emit Rich-rendered output to stdout (default when stdout is a TTY)",
    )

    parent.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="increase log verbosity (repeat for DEBUG)",
    )
    parent.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="DEBUG logging plus full tracebacks on errors",
    )

    return parent


def resolve_format(args: argparse.Namespace, stream: TextIO | None = None) -> str:
    """Resolve the effective output format.

    Explicit `--json`/`--rich` always win. Otherwise default to `rich`
    when the stream is a TTY and `json` when it isn't.
    """
    explicit = getattr(args, "format", None)
    if explicit is not None:
        return str(explicit)
    target = stream if stream is not None else sys.stdout
    return "rich" if target.isatty() else "json"


def configure_logging(args: argparse.Namespace) -> None:
    """Set the `strands_evals` logger level from `-v`/`--debug` flags.

    Defaults to WARNING. Logs always go to stderr.
    """
    debug = getattr(args, "debug", False)
    verbose = getattr(args, "verbose", 0)
    if debug or verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logger = logging.getLogger("strands_evals")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)


def emit_error(message: str, *, debug: bool = False, exc: BaseException | None = None) -> None:
    """Print an error to stderr; include a traceback when `debug` is set."""
    print(f"strands-evals: error: {message}", file=sys.stderr)  # noqa: T201
    if debug and exc is not None:
        import traceback

        traceback.print_exception(exc.__class__, exc, exc.__traceback__, file=sys.stderr)


def run_command(func: Callable[[argparse.Namespace], int | None], args: argparse.Namespace) -> int:
    """Invoke a command callback, mapping exceptions to standard exit codes.

    - `FileNotFoundError` / `json.JSONDecodeError` / pydantic `ValidationError` /
      `EntryPointError` → 2
    - other exceptions → 3
    - command may also return an int directly (`None` is treated as `EXIT_OK`).
    """
    debug = getattr(args, "debug", False)
    try:
        result = func(args)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError, EntryPointError) as e:
        emit_error(str(e), debug=debug, exc=e)
        return EXIT_BAD_INPUT
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - top-level CLI boundary
        emit_error(str(e), debug=debug, exc=e)
        return EXIT_RUNTIME_ERROR

    return int(result) if isinstance(result, int) else EXIT_OK
