"""The append-only `lineage.jsonl` event log."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

EventName = Literal[
    "config",
    "propose",
    "score",
    "select",
    "merge",
    "admit",
    "dominated",
    "infeasible",
    "duplicate",
    "unscored",
    "accept",
    "reject",
    "proposer_error",
    "resume",
    "stop",
    "observe",
    "screened",
    "iteration_end",
]


class Lineage:
    """One JSON event per line; the first line is the run's config."""

    def __init__(self, path: Path):
        raise NotImplementedError

    def append(self, event: EventName, **fields: Any) -> None:
        raise NotImplementedError

    def read(self) -> list[dict[str, Any]]:
        """Every event, dropping a truncated last line."""
        raise NotImplementedError
