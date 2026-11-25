"""Anonymous usage reporting for the `strands-evals` console script.

One invocation sends one event: which subcommand ran, the version and platform that ran
it, whether its output went to a terminal, and how the process ended. The event carries
no identifier of any kind and nothing about what was evaluated, so no two stored events
can be tied to each other, to a machine, or to a person. `STRANDS_EVALS_TELEMETRY=0` or
`DO_NOT_TRACK` stops it.

This module is not `strands_evals.telemetry`, which is OpenTelemetry tracing of an
evaluation. Nothing here is observability: an event that cannot be sent is dropped, never
retried, and never raised to the caller.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import sys
import urllib.request
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from typing import TextIO

from . import _common

logger = logging.getLogger(__name__)

ENDPOINT = "https://telemetry.strandsagents.com/strands-evals/ping"
EVENT = "command/1"
TIMEOUT_SECONDS = 1.0

_DISTRIBUTION = "strands-agents-evals"

UNKNOWN_COMMAND = "unknown"

# Mirrors the subcommands that `main._build_parser` registers, which `test_usage.py` pins. It is a copy because the
# server holds the same list as an enum and rejects a name it does not know, so a new subcommand is reported as
# `unknown` until the server is updated.
COMMANDS: frozenset[str] = frozenset({"validate", "report", "diagnose", "run", "generate", "fetch", "optimize", "ui"})

# `started` is for a subcommand that does not exit on its own, so it counts a server that opened rather than an
# invocation that ended. Never add it to the other three.
OUTCOMES: frozenset[str] = frozenset({"completed", "failed", "interrupted", "started"})

# `EXIT_FAILURES` means the evaluation found failures, which is a working CLI, so it reports `completed` like
# `EXIT_OK`. Only a code that says the CLI itself could not do its work reports `failed`.
_OUTCOME_BY_EXIT_CODE: dict[int, str] = {
    _common.EXIT_OK: "completed",
    _common.EXIT_FAILURES: "completed",
    _common.EXIT_BAD_INPUT: "failed",
    _common.EXIT_RUNTIME_ERROR: "failed",
}

_OS_FAMILIES: dict[str, str] = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}
_OTHER_OS = "other"

# The server takes MAJOR.MINOR.PATCH and at most four digits in each part. A prerelease label or build metadata can
# carry a branch, a user, or a host name, so this reads the numbers at the front and drops the rest.
_VERSION_PREFIX = re.compile(r"^(\d{1,4})\.(\d{1,4})(?:\.(\d{1,4}))?")

# An environment variable with any of these values is off. Everything else, including a value another tool gives its
# own meaning, is on.
_OFF_VALUES = frozenset({"", "0", "false", "no", "off"})


def enabled(env: Mapping[str, str] | None = None) -> bool:
    """Report whether usage reporting is on.

    Two switches turn it off, and either one is enough: `STRANDS_EVALS_TELEMETRY` set to
    `0`, or the cross-tool `DO_NOT_TRACK` set to anything that is not `0`. The CLI runs
    mostly with no terminal, so an environment variable is the whole mechanism: there is
    no config file to hold a preference and often no session in which to show a notice.

    Args:
        env: The environment to read. Defaults to `os.environ`.

    Returns:
        True when an event may be sent.
    """
    source = env if env is not None else os.environ
    if source.get("STRANDS_EVALS_TELEMETRY", "1").strip().lower() in _OFF_VALUES:
        return False
    if source.get("DO_NOT_TRACK", "").strip().lower() not in _OFF_VALUES:
        return False
    return True


def command_name(command: str | None) -> str:
    """Map the parsed subcommand to a name the server accepts.

    Args:
        command: The `command` attribute argparse set, or None.

    Returns:
        The subcommand name, or `unknown` for anything the server does not know yet.
    """
    return command if command in COMMANDS else UNKNOWN_COMMAND


def outcome_for(exit_code: int) -> str:
    """Map an exit code of the CLI to the outcome of the process.

    Args:
        exit_code: The code `_common.run_command` returned.

    Returns:
        `completed` when the CLI did its work, whatever the evaluation found, and `failed`
        when the CLI itself could not finish.
    """
    return _OUTCOME_BY_EXIT_CODE.get(exit_code, "failed")


def cli_version() -> str | None:
    """Read the installed version, normalized to MAJOR.MINOR.PATCH.

    Returns:
        The normalized version, or None when the package is not installed or its version
        has no such prefix. The server rejects any other shape, so there is nothing to
        gain by sending one.
    """
    try:
        raw = distribution_version(_DISTRIBUTION)
    except PackageNotFoundError:
        return None

    match = _VERSION_PREFIX.match(raw)
    if match is None:
        return None

    major, minor, patch = match.group(1), match.group(2), match.group(3) or "0"
    return f"{int(major)}.{int(minor)}.{int(patch)}"


def os_family() -> str:
    """Return the operating system family, as one of the four names the server accepts."""
    return _OS_FAMILIES.get(platform.system(), _OTHER_OS)


def build_event(command: str, outcome: str, *, stream: TextIO | None = None) -> dict[str, object] | None:
    """Build the event for one invocation.

    Args:
        command: A name from `command_name`.
        outcome: A name from `OUTCOMES`.
        stream: The stream to test for a terminal. Defaults to `sys.stdout`, which is what
            `_common.resolve_format` reads to choose the output format.

    Returns:
        The event, or None when the installed version cannot be normalized.
    """
    version = cli_version()
    if version is None:
        return None

    target = stream if stream is not None else sys.stdout
    return {
        "v": 1,
        "event": EVENT,
        "context": {"cli_version": version, "os": os_family(), "interactive": bool(target.isatty())},
        "data": {"command": command, "outcome": outcome},
    }


def report(
    command: str,
    outcome: str,
    *,
    env: Mapping[str, str] | None = None,
    stream: TextIO | None = None,
) -> None:
    """Send one event for this invocation, and drop it if anything goes wrong.

    Blocks for up to `TIMEOUT_SECONDS`. A CLI process that returns can be gone before a
    background thread flushes its socket, which loses most events, so the wait is paid
    here on the exit path instead.

    Never raises, and never changes what the user's command returns.

    Args:
        command: A name from `command_name`.
        outcome: A name from `OUTCOMES`.
        env: The environment to read the switches from. Defaults to `os.environ`.
        stream: The stream to test for a terminal. Defaults to `sys.stdout`.
    """
    if not enabled(env):
        return

    try:
        event = build_event(command, outcome, stream=stream)
        if event is None:
            logger.debug("command=<%s> | version is not MAJOR.MINOR.PATCH so no event is sent", command)
            return
        _send(event)
    except Exception as e:  # noqa: BLE001 - no telemetry failure may reach the user's command
        logger.debug("command=<%s>, error=<%s> | usage event dropped", command, e)


def _send(event: dict[str, object]) -> None:
    body = json.dumps(event, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Connection": "close"},
    )

    # A refused event is dropped and never retried. The endpoint rate limits per source IP and answers a blocked
    # request with 403, so a retry from a fleet of CI jobs behind one address would extend its own block.
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        logger.debug("status=<%s> | usage event sent", response.status)
