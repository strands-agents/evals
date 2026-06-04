"""I/O helpers: read positional path-or-stdin, write structured output."""

from __future__ import annotations

import sys
from pathlib import Path


def read_text_input(path_or_dash: str) -> str:
    """Read a JSON document from a path or from stdin when given `-`."""
    if path_or_dash == "-":
        return sys.stdin.read()
    return Path(path_or_dash).expanduser().read_text(encoding="utf-8")


def write_text_output(text: str, path: str | None) -> None:
    """Write `text` to `path` (or stdout when `path` is None).

    Creates parent directories as needed.
    """
    if path is None:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return

    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")
