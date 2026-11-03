"""The run directory: `run.json`, the append-only `events.jsonl`, transcripts, and the review files."""

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RUNS_DIRECTORY = Path(".strands-evals") / "optimization" / "runs"
TRANSCRIPT_HASH_LENGTH = 16
_CREATE_ATTEMPTS = 5
_NAME = re.compile(r"[a-z0-9_-]+")


def check_name(name: str | None) -> None:
    """Raise `ValueError` unless `name` is None or lowercase letters, digits, `-`, and `_`."""
    if name is not None and not _NAME.fullmatch(name):
        raise ValueError(f"--name must contain only lowercase letters, digits, '-', and '_': {name!r}")


def new_run_id(name: str | None, now: datetime) -> str:
    """`[<name>-]<YYYYMMDD-HHMMSS>-<6 hex>`, the start time in UTC."""
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(3)
    return f"{name}-{stamp}-{suffix}" if name else f"{stamp}-{suffix}"


def write_json(path: Path, value: Any) -> None:
    """Write JSON through a temporary file and an atomic rename."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_json(value: Any) -> str:
    """Sorted keys, no whitespace: the form that content hashes use."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=str)


class EventLog:
    """One JSON event per line; every event has `event`, `iteration`, and `time` (for reading only)."""

    def __init__(self, path: Path):
        """Open the log at `path`; the file is created by the first append."""
        self.path = path

    def append(self, event: str, iteration: int, **fields: Any) -> None:
        """Append one event with one write, then flush it to disk."""
        line = json.dumps(
            {"event": event, "iteration": iteration, "time": datetime.now(timezone.utc).isoformat(), **fields},
            ensure_ascii=False,
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def read(self) -> list[dict[str, Any]]:
        """Every event, without a truncated last line.

        Raises:
            ValueError: A line other than a truncated last line is not a JSON object.
        """
        text = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        lines = text.split("\n")
        complete = lines[:-1]
        events = []
        for number, line in enumerate(complete, start=1):
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"corrupt event log {self.path} at line {number}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"corrupt event log {self.path} at line {number}: not an object")
            events.append(event)
        return events

    def drop_truncated_tail(self) -> None:
        """Cut a partial last line, so that the next append starts on a new line."""
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            with self.path.open("r+b") as stream:
                stream.truncate(data.rfind(b"\n") + 1)


@dataclass(frozen=True)
class RunDirectory:
    """`<base>/<run_id>/` with `run.json`, `events.jsonl`, `transcripts/`, and the review files."""

    root: Path
    events: EventLog

    @property
    def run_id(self) -> str:
        """The folder name."""
        return self.root.name

    @classmethod
    def create(cls, base_dir: Path, config: dict[str, Any], *, name: str | None) -> "RunDirectory":
        """Make a new run folder, then write `config` as `run.json` and as the first event.

        Raises:
            ValueError: `name` has characters other than lowercase letters, digits, `-`, and `_`.
            RuntimeError: No unique folder name was found.
        """
        check_name(name)
        base_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(_CREATE_ATTEMPTS):
            root = base_dir / new_run_id(name, datetime.now(timezone.utc))
            try:
                root.mkdir()
            except FileExistsError:
                continue
            run = cls(root=root.resolve(), events=EventLog(root.resolve() / "events.jsonl"))
            stored = {**config, "run_id": run.run_id}
            write_json(run.root / "run.json", stored)
            run.events.append("config", 0, **stored)
            return run
        raise RuntimeError(f"cannot create a unique run directory under {base_dir}")

    @classmethod
    def open(cls, path: Path) -> "RunDirectory":
        """Reopen an existing run folder and cut a truncated last event.

        Raises:
            ValueError: `path` is not a run folder.
        """
        root = path.expanduser().resolve()
        if not (root / "run.json").is_file() or not (root / "events.jsonl").is_file():
            raise ValueError(f"not an optimization run directory: {root}")
        run = cls(root=root, events=EventLog(root / "events.jsonl"))
        run.events.drop_truncated_tail()
        return run

    def config(self) -> dict[str, Any]:
        """The `run.json` contents."""
        data = json.loads((self.root / "run.json").read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported run.json schema in {self.root}")
        return data

    def save_transcript(self, trajectory: Any) -> dict[str, str] | None:
        """Store a trajectory once under `transcripts/<hash>.json`; return its reference for the trial event."""
        if trajectory is None:
            return None
        text = canonical_json(trajectory)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:TRANSCRIPT_HASH_LENGTH]
        path = self.root / "transcripts" / f"{digest}.json"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            path.write_text(text + "\n", encoding="utf-8")
        return {"hash": digest, "format": "json"}

    def load_transcript(self, reference: dict[str, str] | None) -> Any:
        """Read a trajectory back from its reference; None stays None."""
        if reference is None:
            return None
        path = self.root / "transcripts" / f"{reference['hash']}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def write_review(self, stem: str, diff: str, values: dict[str, Any]) -> tuple[Path, Path]:
        """Write `<stem>.diff` and `<stem>.values.json`; both files exist even when they are empty."""
        diff_path = self.root / f"{stem}.diff"
        values_path = self.root / f"{stem}.values.json"
        diff_path.write_text(diff, encoding="utf-8")
        write_json(values_path, values)
        return diff_path, values_path

    def write_result(self, result: dict[str, Any]) -> Path:
        """Write `result.json`."""
        path = self.root / "result.json"
        write_json(path, result)
        return path
