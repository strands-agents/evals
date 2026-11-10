"""The run directory: `run.json`, the event log, transcripts, effects, and per-candidate file copies."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
import urllib.parse
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from ...types.optimization import TrialRecord
from ...types.trace import Session
from ..candidate import Candidate
from .events import EventLog, _encode, closed_events

SCHEMA_VERSION = 1

_NAME = re.compile(r"[a-z0-9_-]+")


class RunDirectory:
    """`<base>/<run_id>/`; implements `execution.trials.TrialCache`. Original user files are never written.

    `run_id` is `[<name>-]<YYYYMMDD-HHMMSS>-<6 hex>`, the start time in UTC. Layout: `run.json`,
    `events.jsonl` (the only source of truth on resume), `transcripts/<hash>.json` (each trajectory once,
    named by the first 16 hex of the sha256 of its canonical JSON), `effects.json`, the review diffs,
    `result.json` (the whole result, written last), and `candidates/<id>/optimization_files/` (only when a
    value reaches the agent as a path). No per-candidate values, trial, or report files.

    Attributes:
        config: The `config` event's contents (= `run.json`), as read back.
    """

    run_id: str
    root: Path
    events: EventLog
    config: dict[str, Any]

    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = root
        self.run_id = root.name
        self.events = EventLog(root / "events.jsonl")
        self.config = config
        # Candidate id -> its latest full (not rescore) propose event; insertion order = display number.
        self._candidates: dict[str, dict[str, Any]] = {}
        # Ids whose latest full propose event was in a killed iteration (dropped on resume).
        self._killed: set[str] = set()
        # (candidate id, run index, case name) -> the last trial event's record.
        self._trials: dict[tuple[str, int, str], dict[str, Any]] = {}

    @classmethod
    def create(cls, base_dir: Path, config: Mapping[str, Any], *, name: str | None = None) -> RunDirectory:
        """Make `<base_dir>/<run_id>/` and write `config` as the first event and as `run.json`.

        `schema_version` and `code` (the strands-evals version, git commit, and dirty flag writing it;
        each None where it can't be found) are added to `config`.

        Args:
            base_dir: The parent of all run directories.
            config: The run's config, as compared on resume.
            name: The run folder's prefix, `Optimizer(name=)`; a collision draws new hex.

        Raises:
            ValueError: `name` holds anything but lowercase letters, digits, `-`, and `_`.
        """
        if name is not None and not _NAME.fullmatch(name):
            raise ValueError(f"name={name!r}: use only lowercase letters, digits, '-' and '_'")
        base_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        prefix = f"{name}-" if name is not None else ""
        while True:
            root = base_dir / f"{prefix}{stamp}-{secrets.token_hex(3)}"
            try:
                root.mkdir()
                break
            except FileExistsError:
                continue
        contents = json.loads(
            json.dumps(
                {**config, "schema_version": SCHEMA_VERSION, "code": code_version()},
                default=_encode,
                allow_nan=False,
            )
        )
        (root / "run.json").write_text(json.dumps(contents, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        run = cls(root, contents)
        run.events.append("config", iteration=0, config=contents)
        return run

    @classmethod
    def open(cls, path: Path) -> RunDirectory:
        """Read `events.jsonl` once and index candidates and the last trial event per (candidate, run, case).

        Every candidate is indexed, a killed iteration's too (so its number and values stay); those are
        marked, so proposing one again writes a full `propose` event (`save_candidate`).

        Raises:
            ValueError: The log doesn't start with a `config` event, or was written by another
                `schema_version`.
        """
        log = EventLog(path / "events.jsonl")
        events = log.read()
        if not events or events[0]["event"] != "config":
            raise ValueError(f"{path} is not a run directory: events.jsonl doesn't start with a config event")
        config = events[0]["config"]
        if config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path} was written with schema_version {config.get('schema_version')!r}; "
                f"this version reads {SCHEMA_VERSION}"
            )
        run = cls(path, config)
        run.events = log
        closed = {id(event) for event in closed_events(events)}
        for event in events:
            if event["event"] == "propose" and not event["rescore"]:
                candidate_id = event["candidate_id"]
                run._candidates[candidate_id] = event
                if id(event) in closed:
                    run._killed.discard(candidate_id)
                else:
                    run._killed.add(candidate_id)
            elif event["event"] == "trial":
                record = event["record"]
                run._trials[(record["candidate_id"], record["run_index"], record["case_name"])] = record
        return run

    def workspace(self, candidate_id: str) -> Path:
        """`<root>/candidates/<id>/optimization_files/`."""
        return self.root / "candidates" / candidate_id / "optimization_files"

    def save_candidate(self, candidate: Candidate, **fields: Any) -> int:
        """Append the `propose` event and return the candidate's display number.

        0 for the baseline, then 1, 2, ... in proposal order; for people only, never a key. An id already
        indexed (an ineligible candidate proposed again) gets `rescore=true`, no values, the same number;
        except one from a killed iteration, which resume dropped: it gets a full event with its old number,
        replacing it in the index.

        Args:
            candidate: The candidate.
            **fields: The optimizer's fields: `changed_keys`, `round`, `role`, `patch`.
        """
        first = self._candidates.get(candidate.id)
        if first is not None and candidate.id not in self._killed:
            self.events.append(
                "propose",
                candidate_id=candidate.id,
                number=first["number"],
                parent_id=first["parent_id"],
                origin=candidate.origin,
                rationale=candidate.rationale,
                rescore=True,
                **fields,
            )
            return int(first["number"])
        number = len(self._candidates) if first is None else int(first["number"])
        self._killed.discard(candidate.id)
        event = {
            "candidate_id": candidate.id,
            "number": number,
            "parent_id": candidate.parent_id,
            "origin": candidate.origin,
            "values": dict(candidate.values),
            "rationale": candidate.rationale,
            "inherited": dict(candidate.inherited),
            "defaulted": list(candidate.defaulted),
            "rescore": False,
            **fields,
        }
        self.events.append("propose", **event)
        self._candidates[candidate.id] = event
        return number

    def number(self, candidate_id: str) -> int:
        """The display number from the index.

        Raises:
            KeyError: The candidate was never saved.
        """
        return int(self._candidates[candidate_id]["number"])

    def load_candidate(self, candidate_id: str) -> Candidate:
        """From the index (its first `propose` event).

        Raises:
            KeyError: The candidate was never saved.
            ValueError: The saved values no longer hash to `candidate_id`.
        """
        event = self._candidates[candidate_id]
        candidate = Candidate(
            values=event["values"],
            parent_id=event["parent_id"],
            origin=event["origin"],
            rationale=event["rationale"],
            inherited=event["inherited"],
            defaulted=tuple(event["defaulted"]),
        )
        if candidate.id != candidate_id:
            raise ValueError(f"candidate {candidate_id}: its saved values hash to {candidate.id}")
        return candidate

    def load_trial(self, candidate_id: str, run_index: int, case_name: str) -> TrialRecord | None:
        """The last trial event for the key, its transcript read back as trajectory."""
        saved = self._trials.get((candidate_id, run_index, case_name))
        if saved is None:
            return None
        record = dict(saved)
        transcript = record.pop("transcript")
        trajectory: Any = None
        if transcript is not None:
            data = json.loads((self.root / "transcripts" / f"{transcript['hash']}.json").read_text(encoding="utf-8"))
            trajectory = Session.model_validate(data) if transcript["format"] == "session" else data
        return TrialRecord.model_validate({**record, "trajectory": trajectory})

    def save_trial(self, record: TrialRecord) -> None:
        """Write the transcript (if new), then append the trial event and update the index."""
        saved = record.model_dump(mode="json")
        trajectory = saved.pop("trajectory")
        transcript = None
        if trajectory is not None:
            text = json.dumps(trajectory, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            _write_once(self.root / "transcripts" / f"{digest}.json", text)
            transcript = {"hash": digest, "format": "session" if isinstance(record.trajectory, Session) else "list"}
        saved["transcript"] = transcript
        self.events.append("trial", record=saved)
        self._trials[(record.candidate_id, record.run_index, record.case_name)] = saved

    def save_effects(self, effects: Mapping[str, Any]) -> None:
        """`<root>/effects.json`."""
        text = json.dumps(effects, indent=2, default=_encode, ensure_ascii=False, allow_nan=False)
        (self.root / "effects.json").write_text(text + "\n", encoding="utf-8")

    def save_diff(self, stem: str, diff: str, values: Mapping[str, Any]) -> None:
        """`<root>/<stem>.diff` and `<root>/<stem>.values.json`, written at the end after `effects.json`.

        For review only. Derived from the events, so a resumed finished run rewrites them with the same
        content.

        Args:
            stem: `"best"` for the champion, or `"front/<number>-<id>"` for a `Pareto` front member.
            diff: The review diff (`diff(...)`).
            values: The changed numbers and choices, `{key: {"old": ..., "new": ...}}`.
        """
        path = self.root / stem
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_name(f"{path.name}.diff").write_text(diff, encoding="utf-8")
        text = json.dumps(values, indent=2, default=_encode, ensure_ascii=False, allow_nan=False)
        path.with_name(f"{path.name}.values.json").write_text(text + "\n", encoding="utf-8")

    def save_result(self, result: Mapping[str, Any]) -> None:
        """`<root>/result.json`: the whole `OptimizationResult` (`to_dict()`), written last, after the diffs.

        Derived from the events, so a resumed finished run rewrites it; never read on resume.
        """
        text = json.dumps(result, indent=2, default=_encode, ensure_ascii=False, allow_nan=False)
        (self.root / "result.json").write_text(text + "\n", encoding="utf-8")


def _write_once(path: Path, text: str) -> None:
    """Write `path` unless it exists, through a temp file renamed into place, so an existing file is complete."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def code_version() -> dict[str, Any]:
    """Which strands-evals is running: `{"version", "commit", "dirty"}`, each None where it can't be found.

    `version` is the installed distribution's, only when the running code is that installation (installed,
    or installed editable from this checkout). `commit` and `dirty` come from git, only when this file is
    tracked in the checkout git finds (so a virtual environment inside another repository isn't read as
    it). Not the user's agent code.
    """
    here = Path(__file__).resolve()
    out: dict[str, Any] = {"version": None, "commit": None, "dirty": None}
    try:
        dist = metadata.distribution("strands-agents-evals")
        url = json.loads(dist.read_text("direct_url.json") or "{}").get("url", "")
        located = Path(str(dist.locate_file("strands_evals/optimization/record/run_directory.py"))).resolve()
        if located == here or (
            url.startswith("file://") and here.is_relative_to(Path(urllib.parse.unquote(url[7:])).resolve())
        ):
            out["version"] = dist.version
    except (metadata.PackageNotFoundError, OSError, ValueError):
        pass
    try:

        def git(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(["git", *args], cwd=here.parent, capture_output=True, text=True, timeout=10)

        if git("ls-files", "--error-unmatch", here.name).returncode == 0:
            head = git("rev-parse", "HEAD")
            if head.returncode == 0:
                out["commit"] = head.stdout.strip()
                changes = git("status", "--porcelain", "--untracked-files=no")
                out["dirty"] = bool(changes.stdout.strip()) if changes.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return out
