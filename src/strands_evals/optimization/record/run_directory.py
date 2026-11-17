"""The run directory: `run.json`, the event log, transcripts, effects, and per-candidate file copies."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import logging
import re
import secrets
import subprocess
import sys
import urllib.parse
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import TypeAdapter

from ...types.optimization import TrialRecord, TrialStatus
from ...types.trace import MessageUnion, Session, SpanType
from ..candidate import Candidate
from ..declaration.files import _write_text_atomically
from .events import EventLog, _encode, closed_events

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)

_NAME = re.compile(r"[a-z0-9_-]+")
_LOCK_NAME = "run.lock"
# Windows locks are mandatory, so the locked byte sits far past the end of the empty lock file, where reading or
# copying the file through another handle never touches it.
_LOCK_OFFSET = 2**30
# The type of `InferenceSpan.messages`, so a transcript's distinct messages validate as they would in a span.
_MESSAGES: TypeAdapter[list[MessageUnion]] = TypeAdapter(list[MessageUnion])


class RunDirectory:
    """`<base>/<run_id>/`; implements `execution.trials.TrialCache`. Original user files are never written.

    `run_id` is `[<name>-]<YYYYMMDD-HHMMSS>-<6 hex>`, the start time in UTC. Layout: `run.json`,
    `events.jsonl` (the only source of truth on resume), `transcripts/<hash>.json` (each trajectory once,
    a `Session` with each distinct message stored once, named by the first 16 hex of the sha256 of the
    file's canonical JSON; `load_transcript` reads one back), `effects.json`, the review diffs,
    `result.json` (the whole result, written last; it names transcripts by hash),
    `candidates/<id>/optimization_files/` (only when a value reaches the agent as a path), and `run.lock`
    (empty; made by the first `lock()`). No per-candidate values, trial, or report files.

    One writer per run. `open` only reads (`EventLog` cuts off a killed write's last line at its first append,
    not when it opens the log), and a writer calls `lock()` before it appends to a run it opened or created.
    `lock()` takes an exclusive OS lock on `run.lock`, held until `unlock()` or until the process exits, so a
    killed run needs no cleanup. A second writer, in another process or in this one, gets a `RuntimeError`; so
    does one whose log another writer changed after it was read. The lock file is never deleted.

    Attributes:
        config: The `config` event's contents (= `run.json`), as read back.
    """

    run_id: str
    root: Path
    events: EventLog
    config: dict[str, Any]

    def __init__(self, root: Path, config: dict[str, Any], events: EventLog):
        self.root = root
        self.run_id = root.name
        self.events = events
        self.config = config
        self._held: BinaryIO | None = None  # the open lock file while this object holds the lock
        # Candidate id -> its latest full (not rescore) propose event; insertion order = display number.
        self._candidates: dict[str, dict[str, Any]] = {}
        # Ids whose latest full propose event was in a killed iteration (dropped on resume).
        self._killed: set[str] = set()
        # (candidate id, run index, case name) -> the last trial event's record.
        self._trials: dict[tuple[str, int, str], dict[str, Any]] = {}

    @classmethod
    def create(cls, base_dir: Path, config: Mapping[str, Any], *, name: str | None = None) -> RunDirectory:
        """Make `<base_dir>/<run_id>/` and write `config` as the first event and as `run.json`.

        `schema_version` and `code` (`code_version()`: the strands-evals version, git commit, and dirty flag
        writing it, and the numpy and optuna versions; each None where it can't be found) are added to `config`.

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
        _write_text_atomically(root / "run.json", json.dumps(contents, indent=2, ensure_ascii=False) + "\n")
        run = cls(root, contents, EventLog(root / "events.jsonl"))
        run.events.append("config", iteration=0, config=contents)
        return run

    @classmethod
    def open(cls, path: Path) -> RunDirectory:
        """Read `events.jsonl` once and index candidates and the last trial event per (candidate, run, case).

        Every candidate is indexed, a killed iteration's too (so its number and values stay); those are
        marked, so proposing one again writes a full `propose` event (`save_candidate`).

        Every trial event's transcript file must exist (each is checked, not read), since transcripts are read
        only later: when a step asks for proposals, and when the result is built.

        Raises:
            ValueError: The log doesn't start with a `config` event, or was written by another
                `schema_version`.
            FileNotFoundError: A transcript file a trial event refers to is missing.
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
        run = cls(path, config, log)
        closed = {id(event) for event in closed_events(events)}
        missing: set[str] = set()
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
                transcript = record.get("transcript")
                if transcript is not None and not (path / "transcripts" / f"{transcript['hash']}.json").exists():
                    missing.add(transcript["hash"])
        if missing:
            raise FileNotFoundError(
                f"{path}: {len(missing)} transcript file(s) its trial events refer to are missing from transcripts/"
                f" (e.g. {sorted(missing)[0]}.json)"
            )
        return run

    def lock(self) -> None:
        """Take the run's write lock without waiting; a no-op while this object holds it.

        On a filesystem that can't lock files, a warning is logged and the run goes on unlocked. A child
        forked from the holder without exec (`multiprocessing` with the fork start method) shares the lock,
        so after the holder is killed it stays held until that child exits too.

        Raises:
            RuntimeError: Another process, or another `RunDirectory` in this one, holds the lock; or
                `events.jsonl` changed since this object read it or last wrote it, so another writer ran in
                between and this object's state is stale.
        """
        if self._held is not None:
            return
        held = _acquire(self.root)
        if self.events.changed():
            if held is not None:
                _release(held)
            raise RuntimeError(
                f"run {self.run_id}: events.jsonl changed after it was read, so another writer ran in between;"
                " resume the run again (Optimizer.resume) to continue it"
            )
        self._held = held

    def unlock(self) -> None:
        """Release the write lock, if this object holds it."""
        held, self._held = self._held, None
        if held is not None:
            _release(held)

    def workspace(self, candidate_id: str) -> Path:
        """`<root>/candidates/<id>/optimization_files/`."""
        return self.root / "candidates" / candidate_id / "optimization_files"

    def save_candidate(self, candidate: Candidate, **fields: Any) -> int:
        """Append the `propose` event and return the candidate's display number.

        0 for the baseline, then 1, 2, ... in proposal order; for people only, never a key. An id already
        indexed (an ineligible candidate proposed again, which the optimizer re-scores only when a re-score can
        change it) gets `rescore=true`, no values, the same number;
        except one from a killed iteration, which resume dropped: it gets a full event with its old number,
        replacing it in the index.

        Args:
            candidate: The candidate.
            **fields: The optimizer's fields: `changed_keys`, `round`, `role`, `deferred_patches`.
        """
        first = self._candidates.get(candidate.id)
        if first is not None and candidate.id not in self._killed:
            self.events.append(
                "propose",
                candidate_id=candidate.id,
                number=first["number"],
                parent_id=first["parent_id"],
                origin=list(candidate.origin),
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
            "origin": list(candidate.origin),
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
        """From the index (its latest full `propose` event).

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
        return None if saved is None else self.record_from(saved)

    def record_from(self, saved: Mapping[str, Any], *, load_trajectory: bool = True) -> TrialRecord:
        """A `trial` event's record, its transcript read back as trajectory.

        Args:
            saved: The event's `record`.
            load_trajectory: False leaves `trajectory` None and reads no file; `load_transcript` reads it later.
        """
        record = dict(saved)
        transcript = record.pop("transcript")
        trajectory = self.load_transcript(transcript) if load_trajectory and transcript is not None else None
        return TrialRecord.model_validate({**record, "trajectory": trajectory})

    def transcript(self, candidate_id: str, run_index: int, case_name: str) -> dict[str, str] | None:
        """The last trial event's `transcript` for the key, from the index: `{hash, format}`, or None when the
        record has no trajectory.

        Raises:
            KeyError: The trial was never saved.
        """
        transcript = self._trials[(candidate_id, run_index, case_name)]["transcript"]
        return None if transcript is None else dict(transcript)

    def load_transcript(self, transcript: Mapping[str, str]) -> list[Any] | Session:
        """A `trial` event's `transcript` (`{hash, format}`) read back: the `Session`, or the rollout's own list.

        A `packed_session` transcript's messages are built once each, so spans that repeat a message share
        one object.

        Raises:
            FileNotFoundError: The transcript file is missing.
            ValueError: An unknown `format`, or a hash that isn't 16 lowercase hex digits.
        """
        return read_transcript(self.root, transcript["hash"], transcript["format"])

    def trial_status(self, candidate_id: str, run_index: int, case_name: str) -> TrialStatus | None:
        """The last trial event's status for the key, from the index (no transcript read)."""
        saved = self._trials.get((candidate_id, run_index, case_name))
        return None if saved is None else TrialStatus(saved["status"])

    def save_trial(self, record: TrialRecord) -> None:
        """Write the transcript (if new), then append the trial event and update the index."""
        saved = record.model_dump(mode="json")
        trajectory = saved.pop("trajectory")
        transcript = None
        if trajectory is not None:
            packed = isinstance(record.trajectory, Session)
            text = _canonical(_pack_session(trajectory) if packed else trajectory)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            path = self.root / "transcripts" / f"{digest}.json"
            if not path.exists():
                _write_text_atomically(path, text)
            transcript = {"hash": digest, "format": "packed_session" if packed else "list"}
        saved["transcript"] = transcript
        self.events.append("trial", record=saved)
        self._trials[(record.candidate_id, record.run_index, record.case_name)] = saved

    def save_effects(self, effects: Mapping[str, Any]) -> None:
        """`<root>/effects.json`."""
        text = json.dumps(effects, indent=2, default=_encode, ensure_ascii=False, allow_nan=False)
        _write_text_atomically(self.root / "effects.json", text + "\n")

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
        _write_text_atomically(path.with_name(f"{path.name}.diff"), diff)
        text = json.dumps(values, indent=2, default=_encode, ensure_ascii=False, allow_nan=False)
        _write_text_atomically(path.with_name(f"{path.name}.values.json"), text + "\n")

    def save_result(self, result: Mapping[str, Any]) -> None:
        """`<root>/result.json`: the whole `OptimizationResult` (`to_dict()`) as compact JSON (no indentation),
        written last, after the diffs.

        Derived from the events, so a resumed finished run rewrites it; never read on resume.
        """
        text = json.dumps(result, separators=(",", ":"), default=_encode, ensure_ascii=False, allow_nan=False)
        _write_text_atomically(self.root / "result.json", text + "\n")


def _acquire(root: Path) -> BinaryIO | None:
    """`<root>/run.lock`, open and exclusively locked; None when the filesystem can't lock files.

    Raises:
        RuntimeError: Another open file holds the lock, in this process or another.
    """
    file = (root / _LOCK_NAME).open("a+b")  # opened for writing: NFS grants an exclusive lock only to a writer
    try:
        if sys.platform == "win32":
            file.seek(_LOCK_OFFSET)  # `msvcrt.locking` locks bytes from the current position
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        file.close()
        if isinstance(error, (BlockingIOError, PermissionError)):  # held: EWOULDBLOCK (flock), EACCES (Windows)
            raise RuntimeError(
                f"run {root.name} is open for writing in another process (or a child process it forked), or by"
                " another Optimizer in this one; wait for it to finish, or stop it, then resume the run"
            ) from error
        if error.errno in (errno.ENOLCK, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP):
            logger.warning(
                "path=<%s>, error=<%s> | the filesystem can't lock the run | nothing stops a second writer",
                root,
                error,
            )
            return None
        raise
    return file


def _release(file: BinaryIO) -> None:
    """Release a lock `_acquire` took, then close its file."""
    with contextlib.suppress(OSError):  # closing the file releases the lock too
        if sys.platform == "win32":
            file.seek(_LOCK_OFFSET)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    file.close()


def _canonical(value: Any) -> str:
    """Compact JSON with sorted keys: a transcript's text, and the identity of one of its messages."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _pack_session(session: dict[str, Any]) -> dict[str, Any]:
    """A dumped `Session` with each distinct message stored once: `{"messages": [...], "session": {...}}`.

    `messages` holds every distinct message (by `_canonical` text) in first-use order over the spans. In
    `session`, each inference span's `messages` becomes half-open `[start, stop]` ranges into that list, in
    order. A model call's messages repeat the conversation so far, so each span is usually one range and the
    transcript grows with the conversation, not with its square.
    """
    messages: list[Any] = []
    positions: dict[str, int] = {}
    traces = []
    for trace in session["traces"]:
        spans = []
        for span in trace["spans"]:
            if span["span_type"] == SpanType.INFERENCE.value:
                ranges: list[list[int]] = []
                for message in span["messages"]:
                    position = positions.setdefault(_canonical(message), len(messages))
                    if position == len(messages):
                        messages.append(message)
                    if ranges and ranges[-1][1] == position:
                        ranges[-1][1] = position + 1
                    else:
                        ranges.append([position, position + 1])
                span = {**span, "messages": ranges}
            spans.append(span)
        traces.append({**trace, "spans": spans})
    return {"messages": messages, "session": {**session, "traces": traces}}


def _unpack_session(data: Mapping[str, Any]) -> Session:
    """The `Session` `_pack_session` stored; each message is validated once and shared by the spans that hold it."""
    messages = _MESSAGES.validate_python(data["messages"])
    session = data["session"]
    traces = [
        {
            **trace,
            "spans": [
                {**span, "messages": [messages[i] for start, stop in span["messages"] for i in range(start, stop)]}
                if span["span_type"] == SpanType.INFERENCE.value
                else span
                for span in trace["spans"]
            ],
        }
        for trace in session["traces"]
    ]
    return Session.model_validate({**session, "traces": traces})


_TRANSCRIPT_HASH = re.compile(r"[0-9a-f]{16}")


def read_transcript(root: Path, digest: str, format: str | None = None) -> list[Any] | Session:
    """`<root>/transcripts/<digest>.json` read back: the `Session`, or the rollout's own list.

    `format` is the trial event's (`packed_session`, `list`, or `session`, a whole `Session` dump as runs before
    packing saved it); None tells it by the file's shape (a list; a dict with `messages` and `session`; else a
    whole dump), for a caller that has only the hash.

    Raises:
        FileNotFoundError: The transcript file is missing.
        ValueError: A `digest` that isn't 16 lowercase hex digits (as `save_trial` names files), or an unknown
            `format`.
    """
    if not _TRANSCRIPT_HASH.fullmatch(digest):
        raise ValueError(f"{digest!r} is not a transcript hash (16 lowercase hex digits)")
    data = json.loads((root / "transcripts" / f"{digest}.json").read_text(encoding="utf-8"))
    if format is None:
        if isinstance(data, list):
            format = "list"
        else:
            format = "packed_session" if "messages" in data and "session" in data else "session"
    if format == "packed_session":
        return _unpack_session(data)
    if format == "session":
        return Session.model_validate(data)
    if format == "list":
        return data
    raise ValueError(f"transcript {digest}: unknown format {format!r}")


def code_version() -> dict[str, Any]:
    """Which code is running: `{"version", "commit", "dirty", "numpy", "optuna"}`, each None where it can't be found.

    `version` is the installed distribution's, only when the running code is that installation (installed,
    or installed editable from this checkout). `commit` and `dirty` come from git, only when this file is
    tracked in the checkout git finds (so a virtual environment inside another repository isn't read as
    it). `numpy` and `optuna` are the installed versions. Not the user's agent code.
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
    # A release can change TPE's, Thompson's, or the fit's draws for one seed.
    for package in ("numpy", "optuna"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out
