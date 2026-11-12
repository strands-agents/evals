"""Local HTTP server backing `strands-evals ui`.

Discovery, pairing and writes all live here rather than in the browser, so the logic that
decides *which file is what* is covered by the Python test suite and the page stays a view.

The server is a development tool bound to loopback, but it reads and writes files in the
directory it was pointed at, so it is treated as an authenticated surface:

- a per-process token, printed in the URL, gates every `/api` route and the page itself.
  A random web page in another tab cannot guess it, so it cannot read or overwrite files.
- the `Host` header must name a loopback address, which is what stops DNS rebinding from
  turning a public hostname into a request to this port.
- every path from the client is resolved and checked to be inside the root, so `..` and
  symlinks pointing out of the tree are rejected rather than followed.
- writes carry the `modified` stamp the client loaded with, and a file that changed on disk
  since then is a 409 instead of a silent overwrite of someone else's edit.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import secrets
import socket
import tempfile
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..case import Case
from ..experiment import builtin_evaluator_types

logger = logging.getLogger(__name__)

DEFAULT_DEPTH = 3
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_SCAN_FILES = 2000
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})

# One optimizer trial is one event, so a long run's log is mostly trials. The cap is on lines
# rather than bytes because a half-read log has to be reported as partial, and lines are what the
# page counts.
MAX_EVENT_LINES = 50_000

EXPERIMENT = "experiment"
REPORT = "report"

EVENT_LOG_NAME = "events.jsonl"
RUN_CONFIG_NAME = "run.json"


@dataclass
class ScannedFile:
    """One JSON file in the tree that parsed as an experiment or a report."""

    path: str  # root-relative, posix-style
    kind: str
    obj: dict
    size: int
    modified: float

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def directory(self) -> str:
        return self.path.rsplit("/", 1)[0] if "/" in self.path else "."


@dataclass
class Discovery:
    runs: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    truncated: bool = False
    optimizer_runs: list[dict] = field(default_factory=list)


@dataclass
class EventLog:
    """A parsed `events.jsonl`: the events, the lines that would not parse, and whether it was cut."""

    events: list[dict] = field(default_factory=list)
    bad_lines: list[dict] = field(default_factory=list)
    truncated: bool = False


def classify(obj: Any) -> str | None:
    """Identify a parsed JSON document by its shape.

    Filenames are not evidence: `nightly-eval.json` and `out.json` are both things people
    actually commit. Both shapes carry `cases`, so the discriminator is `evaluators` for an
    experiment against the report's own required fields.
    """
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("cases"), list) and isinstance(obj.get("evaluators"), list):
        return EXPERIMENT
    if (
        isinstance(obj.get("overall_score"), (int, float))
        and not isinstance(obj.get("overall_score"), bool)
        and isinstance(obj.get("scores"), list)
        and isinstance(obj.get("test_passes"), list)
        and isinstance(obj.get("cases"), list)
    ):
        return REPORT
    return None


def _dirname(path: str) -> str:
    """The directory part of a root-relative path, `.` at the root; mirrors `ScannedFile.directory`."""
    return path.rsplit("/", 1)[0] if "/" in path else "."


_STEM_SUFFIX = re.compile(r"(^|[._-])(experiment|exp|report|rep)$", re.IGNORECASE)


def _stem(name: str) -> str:
    """Filename with the .json suffix and a trailing experiment/report token removed.

    Lets `nightly.experiment.json` pair with `nightly.report.json` while leaving `nightly.json`
    and `smoke.json` apart. Plain `experiment.json` and `report.json` both reduce to the empty
    stem, which is what pairs them — that is the layout the CLI's own `--output` produces, so it
    has to work without relying on there being exactly one of each in the directory.

    The token only comes off at a separator or the whole name, so `myexperiment.json` keeps its
    name instead of becoming `my`.
    """
    base = name[:-5] if name.lower().endswith(".json") else name
    return _STEM_SUFFIX.sub("", base)


def _case_keys(report_obj: dict) -> set[str]:
    """Identity of each row in a report, preferring session_id and falling back to name."""
    keys = set()
    for row in report_obj.get("cases", []):
        if isinstance(row, dict):
            key = row.get("session_id") or row.get("name")
            if key is not None:
                keys.add(str(key))
    return keys


def link_status(experiment_obj: dict, report_obj: dict) -> dict:
    """Whether a report plausibly came from the experiment it is being shown beside.

    Pairing by directory and filename is a guess. A report whose rows name cases the
    experiment does not contain is a guess that went wrong, and showing its score next to
    those cases would be a quietly wrong answer to "how did this experiment do". The rows
    carry `session_id`, so the overlap is checkable; the UI reports what this returns rather
    than implying the pair is certain.
    """
    exp_keys = set()
    for case in experiment_obj.get("cases", []):
        if isinstance(case, dict):
            if case.get("session_id") is not None:
                exp_keys.add(str(case["session_id"]))
            if case.get("name") is not None:
                exp_keys.add(str(case["name"]))

    rep_keys = _case_keys(report_obj)
    if not exp_keys or not rep_keys:
        return {"status": "unverified", "detail": "no case identifiers to compare"}

    missing = rep_keys - exp_keys
    if not missing:
        return {"status": "verified", "detail": f"all {len(rep_keys)} report case(s) exist in this experiment"}
    if len(missing) == len(rep_keys):
        return {
            "status": "mismatch",
            "detail": f"none of the {len(rep_keys)} report case(s) appear in this experiment",
        }
    return {
        "status": "partial",
        "detail": f"{len(missing)} of {len(rep_keys)} report case(s) are not in this experiment",
    }


def _report_summary(obj: dict) -> dict:
    """Headline numbers for the run list, counted the way the library counts them."""
    detailed = obj.get("detailed_results") or []
    passes = obj.get("test_passes") or []
    failing = 0
    not_applicable = 0
    for index, _ in enumerate(obj.get("cases", [])):
        outputs = detailed[index] if index < len(detailed) else []
        # Mirrors EvaluationReport.is_applicable: a row is droppable only when it has outputs
        # and every one of them both declined to judge and passed.
        declined = bool(outputs) and all(
            isinstance(o, dict) and o.get("label") == "not_applicable" and o.get("test_pass") for o in outputs
        )
        if declined:
            not_applicable += 1
        elif index < len(passes) and not passes[index]:
            failing += 1
    return {
        "overall_score": obj.get("overall_score"),
        "rows": len(obj.get("cases", [])),
        "failing": failing,
        "not_applicable": not_applicable,
        "evaluators": sorted({str(row.get("evaluator")) for row in obj.get("cases", []) if isinstance(row, dict)}),
    }


def _experiment_summary(obj: dict) -> dict:
    return {
        "cases": len(obj.get("cases", [])),
        "evaluators": [e for e in obj.get("evaluators", []) if isinstance(e, dict)],
        "evaluator_notes": check_evaluators(obj.get("evaluators")),
    }


def _file_entry(scanned: ScannedFile) -> dict:
    entry = {
        "path": scanned.path,
        "name": scanned.name,
        "size": scanned.size,
        "modified": scanned.modified,
    }
    if scanned.kind == EXPERIMENT:
        entry.update(_experiment_summary(scanned.obj))
    else:
        entry.update(_report_summary(scanned.obj))
    return entry


def scan(root: Path, depth: int = DEFAULT_DEPTH) -> tuple[list[ScannedFile], list[dict], bool]:
    """Read every JSON file under `root` up to `depth` directories deep.

    Returns the files that classified, the ones that did not with the reason, and whether the
    file cap was hit. Unreadable and unparseable files are reported rather than dropped: a
    folder where the thing you are looking for failed to parse should not look like a folder
    where it is absent.
    """
    found: list[ScannedFile] = []
    skipped: list[dict] = []
    truncated = False
    root = root.resolve()
    seen = 0

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        level = len(here.relative_to(root).parts)
        if level >= depth:
            dirnames[:] = []
        # Scanning a repo's dependencies is slow and finds nothing a user is looking for.
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in {"node_modules", "__pycache__"})

        for filename in sorted(filenames):
            if not filename.lower().endswith(".json"):
                continue
            if seen >= MAX_SCAN_FILES:
                truncated = True
                break
            seen += 1
            full = here / filename
            rel = full.relative_to(root).as_posix()
            try:
                stat = full.stat()
                obj = json.loads(full.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as exc:
                skipped.append({"path": rel, "why": f"could not read: {exc.__class__.__name__}"})
                continue
            except json.JSONDecodeError as exc:
                skipped.append({"path": rel, "why": f"invalid JSON at line {exc.lineno}: {exc.msg}"})
                continue

            kind = classify(obj)
            if kind is None:
                skipped.append({"path": rel, "why": "not an experiment or a report"})
                continue
            found.append(ScannedFile(path=rel, kind=kind, obj=obj, size=stat.st_size, modified=stat.st_mtime))
        if truncated:
            break

    return found, skipped, truncated


def discover(root: Path, depth: int = DEFAULT_DEPTH) -> Discovery:
    """Group the tree into runs: an experiment, the report beside it, and how sure that is."""
    found, skipped, truncated = scan(root, depth)

    by_dir: dict[str, list[ScannedFile]] = {}
    for scanned in found:
        by_dir.setdefault(scanned.directory, []).append(scanned)

    runs: list[dict] = []
    for directory in sorted(by_dir):
        files = by_dir[directory]
        experiments = [f for f in files if f.kind == EXPERIMENT]
        reports = [f for f in files if f.kind == REPORT]
        taken: set[str] = set()

        for experiment in experiments:
            match = next(
                (r for r in reports if r.path not in taken and _stem(r.name) == _stem(experiment.name)),
                None,
            )
            if match is None and len(experiments) == 1 and len(reports) == 1:
                match = reports[0]
            if match is not None:
                taken.add(match.path)
            runs.append(
                {
                    "id": experiment.path,
                    "dir": directory,
                    "experiment": _file_entry(experiment),
                    "report": _file_entry(match) if match else None,
                    "link": link_status(experiment.obj, match.obj) if match else None,
                }
            )

        for report in reports:
            if report.path in taken:
                continue
            runs.append(
                {"id": report.path, "dir": directory, "experiment": None, "report": _file_entry(report), "link": None}
            )

    runs.sort(
        key=lambda r: (
            -max(
                (r["experiment"] or {}).get("modified", 0),
                (r["report"] or {}).get("modified", 0),
            )
        )
    )

    optimizer_runs = discover_optimizer_runs(root, depth)

    # An optimizer run's `run.json` is not an experiment or a report, so the scan files it under
    # "did not classify". Once it has been recognised as a run config, saying so again as a
    # problem would send someone looking for a fault that isn't there.
    optimizer_dirs = {r["dir"] for r in optimizer_runs}
    skipped = [
        s
        for s in skipped
        if not (s["path"].rsplit("/", 1)[-1] == RUN_CONFIG_NAME and _dirname(s["path"]) in optimizer_dirs)
    ]

    return Discovery(runs=runs, skipped=skipped, truncated=truncated, optimizer_runs=optimizer_runs)


def read_event_log(path: Path, max_lines: int = MAX_EVENT_LINES) -> EventLog:
    """Read an optimizer `events.jsonl`.

    A truncated last line is dropped rather than reported: the optimizer appends with one write
    per event, so the only way to see a partial line is to read the log while a run is still
    writing, and that is normal rather than a problem worth showing. Any *earlier* line that
    fails to parse is real corruption, so it is reported with its line number.
    """
    log = EventLog()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.bad_lines.append({"line": 0, "why": f"could not read: {exc.__class__.__name__}"})
        return log

    lines = text.split("\n")
    # Every line but the last is known complete, because a complete line is one with a newline
    # after it. That holds either way the file ends: with a newline the split leaves a trailing
    # empty string (skipped as blank below), and without one the last element is the partial line.
    complete = len(lines) - 1
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if len(log.events) >= max_lines:
            log.truncated = True
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            # The last line of a log being written is expected to be partial; drop it quietly.
            if number > complete:
                continue
            log.bad_lines.append({"line": number, "why": f"invalid JSON: {exc.msg}"})
            continue
        if isinstance(event, dict):
            log.events.append(event)
        else:
            log.bad_lines.append({"line": number, "why": "not a JSON object"})

    return log


def _optimizer_run_summary(
    directory: str,
    config: dict,
    events_stat: os.stat_result,
    config_stat: os.stat_result,
    log: EventLog,
) -> dict:
    """What the runs list shows for one optimizer run.

    Only things that can be *read* out of the log, never derived: counts, the last stop reason,
    the champion the end phase recorded. Every derived series — best-so-far, the budget curve —
    is computed in the page from the same event log, so the rule for it lives in exactly one
    place instead of being reimplemented here and drifting.
    """
    # Distinct iteration numbers rather than a count of `iteration_end` events, so the same
    # iteration recorded twice reads as one iteration rather than as progress.
    iterations = {e.get("iteration") for e in log.events if e.get("event") == "iteration_end"}
    champion = None
    stop_reason = None
    for event in log.events:
        name = event.get("event")
        if name in ("picked", "test"):
            report = event.get("report")
            if isinstance(report, dict) and report.get("champion_id"):
                champion = report["champion_id"]
        elif name == "stop":
            stop_reason = event.get("reason")

    return {
        "id": directory,
        "dir": directory,
        "run_id": config.get("run_id") or directory.rsplit("/", 1)[-1],
        "name": config.get("name"),
        "objective": config.get("objective"),
        "constraints": config.get("constraints") or [],
        "is_pareto": bool(config.get("is_pareto")),
        "budget": config.get("budget") or {},
        "config": config,
        "events_path": f"{directory}/{EVENT_LOG_NAME}" if directory != "." else EVENT_LOG_NAME,
        "events": len(log.events),
        "events_size": events_stat.st_size,
        "modified": max(events_stat.st_mtime, config_stat.st_mtime),
        "iterations": len(iterations),
        "candidates": sum(1 for e in log.events if e.get("event") == "propose"),
        "trials": sum(1 for e in log.events if e.get("event") == "trial"),
        "champion_id": champion,
        "stop_reason": stop_reason,
        "bad_lines": log.bad_lines,
        "truncated": log.truncated,
    }


def discover_optimizer_runs(root: Path, depth: int = DEFAULT_DEPTH) -> list[dict]:
    """Find optimizer run directories: a `run.json` with an `events.jsonl` beside it.

    Both are required. `run.json` alone is far too common a filename to treat as evidence of an
    optimizer run, and an event log with no config cannot be read (the objective and the budget
    live in the config).
    """
    found: list[dict] = []
    root = root.resolve()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        level = len(here.relative_to(root).parts)
        if level >= depth:
            dirnames[:] = []
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in {"node_modules", "__pycache__"})

        names = set(filenames)
        if RUN_CONFIG_NAME not in names or EVENT_LOG_NAME not in names:
            continue

        config_path, events_path = here / RUN_CONFIG_NAME, here / EVENT_LOG_NAME
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config_stat, events_stat = config_path.stat(), events_path.stat()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(config, dict):
            continue

        relative = here.relative_to(root).as_posix() or "."
        log = read_event_log(events_path)
        found.append(_optimizer_run_summary(relative, config, events_stat, config_stat, log))

    found.sort(key=lambda r: -r["modified"])
    return found


class PathOutsideRoot(ValueError):
    """A client-supplied path resolved outside the served directory."""


def resolve_within(root: Path, relative: str) -> Path:
    """Resolve a client path against the root, refusing anything that escapes it.

    Resolution happens before the check so a symlink pointing out of the tree is caught too,
    not just a literal `..`.
    """
    if not relative:
        raise PathOutsideRoot("no path given")
    candidate = (root / relative).resolve()
    root = root.resolve()
    if candidate != root and root not in candidate.parents:
        raise PathOutsideRoot(f"{relative} is outside {root}")
    return candidate


def validate_experiment_payload(payload: Any) -> dict:
    """Check an experiment document well enough to refuse to write a broken one.

    Cases go through `Case.model_validate`, which is the same gate `Experiment.from_file` uses,
    so a save cannot produce a file the library then refuses to load.

    Evaluators are deliberately *not* instantiated. `Experiment.from_dict` resolves
    `evaluator_type` against a registry of built-ins, so validating them here would reject
    every experiment built on a custom evaluator subclass — which is most real ones. They are
    inspected without being constructed by `check_evaluators`, and reported as advice.
    """
    if not isinstance(payload, dict):
        raise ValueError("experiment must be a JSON object")
    if not isinstance(payload.get("cases"), list):
        raise ValueError("experiment is missing a 'cases' list")
    if not isinstance(payload.get("evaluators"), list):
        raise ValueError("experiment is missing an 'evaluators' list")

    for index, case in enumerate(payload["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} is not a JSON object")
        try:
            Case.model_validate(case)
        except Exception as exc:  # pydantic ValidationError and anything it wraps
            name = case.get("name", f"index {index}")
            raise ValueError(f"case {name!r} is not valid: {exc}") from exc

    for index, evaluator in enumerate(payload["evaluators"]):
        if not isinstance(evaluator, dict) or "evaluator_type" not in evaluator:
            raise ValueError(f"evaluator {index} has no 'evaluator_type'")

    return payload


def check_evaluators(evaluators: Any) -> list[dict]:
    """Say whether each `evaluator_type` would survive `Experiment.from_dict`, without running it.

    A page that can save an experiment the CLI then refuses to load is a trap, and the failure
    arrives later and somewhere else — `strands-evals validate` reporting a `TypeError` from a
    constructor. So the settings are checked against the real registry and the real signature,
    and reported as advice rather than enforced: nothing here blocks a save.

    `from_dict` constructs the evaluator, so checking by construction would mean building models
    and whatever else an `__init__` does. `inspect.signature` answers the same question for the
    two ways a built-in actually fails — a required setting left out and a setting that is not a
    parameter — without that.

    An unregistered name is only a warning. Custom evaluator subclasses are passed to
    `from_dict` by the caller, not named in the file, and most real experiments have one.
    """
    if not isinstance(evaluators, list):
        return []

    registry = builtin_evaluator_types()
    notes: list[dict] = []
    for index, evaluator in enumerate(evaluators):
        if not isinstance(evaluator, dict):
            continue
        name = evaluator.get("evaluator_type")
        if not isinstance(name, str):
            continue

        def note(level: str, message: str, index: int = index, name: str = name) -> None:
            notes.append({"index": index, "evaluator_type": name, "level": level, "message": message})

        target = registry.get(name)
        if target is None:
            note("warning", "not a built-in evaluator; `run` and `validate` need it passed in as a custom evaluator")
            continue

        # `from_dict` renames this one before constructing, so the file is allowed to say either.
        settings = {("model" if key == "model_id" else key) for key in evaluator if key != "evaluator_type"}
        try:
            parameters = inspect.signature(target).parameters
        except (TypeError, ValueError):  # a builtin or C-implemented __init__
            continue

        accepts_anything = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        named = {
            key
            for key, parameter in parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        required = {key for key in named if parameters[key].default is inspect.Parameter.empty}

        for missing in sorted(required - settings):
            note("error", f"needs a '{missing}' setting; {name} cannot be constructed without it")
        if not accepts_anything:
            for extra in sorted(settings - named):
                note("error", f"'{extra}' is not a setting of {name}")

    return notes


def write_json_atomic(target: Path, payload: dict) -> float:
    """Serialize, then replace the file in one step; returns the new mtime.

    Serializing first means a document that cannot become valid JSON leaves the existing file
    untouched, matching `Experiment.to_file`. `allow_nan=False` is deliberate: NaN and Infinity
    are not JSON, and the library's own writer rejects them rather than emitting something only
    Python can read back.
    """
    data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", dir=directory, prefix=f".{target.name}.", suffix=".tmp", delete=False)
    try:
        with handle as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target.stat().st_mtime


@dataclass
class UIConfig:
    root: Path
    token: str
    depth: int = DEFAULT_DEPTH
    read_only: bool = False
    asset: str = ""


class UIHandler(BaseHTTPRequestHandler):
    """Routes for the UI. `config` is attached to the server by `build_server`."""

    server_version = "strands-evals-ui"
    protocol_version = "HTTP/1.1"

    @property
    def config(self) -> UIConfig:
        return self.server.config  # type: ignore[attr-defined]

    # ── plumbing ──

    def log_message(self, fmt: str, *args: Any) -> None:
        # BaseHTTPRequestHandler writes to stderr unconditionally; route it through logging so
        # the CLI's -v/--debug flags control it like everything else.
        logger.debug("client=<%s> | %s", self.address_string(), fmt % args)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page is served from, and only talks to, this origin.
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        self._send(status, json.dumps(payload, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    def _host_is_loopback(self) -> bool:
        host = (self.headers.get("Host") or "").strip()
        if not host:
            return False
        name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        return name.lower() in LOOPBACK_HOSTS

    def _authorized(self, query: dict[str, list[str]]) -> bool:
        supplied = (self.headers.get("X-Evals-Token") or "") or (query.get("t", [""])[0])
        # Compared as bytes: compare_digest rejects non-ASCII str, and a header is attacker-chosen.
        return secrets.compare_digest(supplied.encode("utf-8", "replace"), self.config.token.encode("utf-8"))

    # ── routes ──

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if not self._host_is_loopback():
            self._error(HTTPStatus.FORBIDDEN, "this server only answers requests addressed to localhost")
            return

        if parsed.path in ("/", "/index.html"):
            if not self._authorized(query):
                self._send(
                    HTTPStatus.UNAUTHORIZED,
                    b"<!doctype html><meta charset=utf-8><title>strands-evals</title>"
                    b"<body style='font:14px system-ui;padding:40px'>"
                    b"<h3>Open the link the CLI printed</h3>"
                    b"<p>This page needs the one-time token from <code>strands-evals ui</code>.</p>",
                    "text/html; charset=utf-8",
                )
                return
            self._send(HTTPStatus.OK, self.config.asset.encode("utf-8"), "text/html; charset=utf-8")
            return

        if not self._authorized(query):
            self._error(HTTPStatus.UNAUTHORIZED, "missing or wrong token")
            return

        if parsed.path == "/api/config":
            self._json(
                HTTPStatus.OK,
                {
                    "root": str(self.config.root),
                    "depth": self.config.depth,
                    "read_only": self.config.read_only,
                },
            )
            return

        if parsed.path == "/api/runs":
            result = discover(self.config.root, self.config.depth)
            self._json(
                HTTPStatus.OK,
                {
                    "root": str(self.config.root),
                    "depth": self.config.depth,
                    "read_only": self.config.read_only,
                    "runs": result.runs,
                    "skipped": result.skipped,
                    "truncated": result.truncated,
                    "optimizer_runs": result.optimizer_runs,
                },
            )
            return

        if parsed.path == "/api/events":
            rel = query.get("path", [""])[0]
            try:
                target = resolve_within(self.config.root, rel)
            except PathOutsideRoot as exc:
                self._error(HTTPStatus.FORBIDDEN, str(exc))
                return
            if not target.is_file():
                self._error(HTTPStatus.NOT_FOUND, f"{rel} is not a file")
                return
            log = read_event_log(target)
            self._json(
                HTTPStatus.OK,
                {
                    "path": rel,
                    "modified": target.stat().st_mtime,
                    "events": log.events,
                    "bad_lines": log.bad_lines,
                    "truncated": log.truncated,
                },
            )
            return

        if parsed.path == "/api/file":
            rel = query.get("path", [""])[0]
            try:
                target = resolve_within(self.config.root, rel)
            except PathOutsideRoot as exc:
                self._error(HTTPStatus.FORBIDDEN, str(exc))
                return
            if not target.is_file():
                self._error(HTTPStatus.NOT_FOUND, f"{rel} is not a file")
                return
            try:
                obj = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{rel} could not be parsed: {exc}")
                return
            self._json(
                HTTPStatus.OK,
                {
                    "path": rel,
                    "modified": target.stat().st_mtime,
                    "document": obj,
                    "evaluator_notes": check_evaluators(obj.get("evaluators") if isinstance(obj, dict) else None),
                },
            )
            return

        self._error(HTTPStatus.NOT_FOUND, f"no route for {parsed.path}")

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if not self._host_is_loopback():
            self._error(HTTPStatus.FORBIDDEN, "this server only answers requests addressed to localhost")
            return
        if not self._authorized(query):
            self._error(HTTPStatus.UNAUTHORIZED, "missing or wrong token")
            return
        if parsed.path != "/api/experiment":
            self._error(HTTPStatus.NOT_FOUND, f"no route for {parsed.path}")
            return
        if self.config.read_only:
            self._error(HTTPStatus.FORBIDDEN, "server started with --read-only")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "bad Content-Length")
            return
        if length <= 0:
            self._error(HTTPStatus.BAD_REQUEST, "empty body")
            return
        if length > MAX_UPLOAD_BYTES:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body over {MAX_UPLOAD_BYTES} bytes")
            return

        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"body is not JSON: {exc}")
            return
        if not isinstance(body, dict) or "document" not in body:
            self._error(HTTPStatus.BAD_REQUEST, "expected {'document': ..., 'modified': ...}")
            return

        rel = query.get("path", [""])[0]
        try:
            target = resolve_within(self.config.root, rel)
        except PathOutsideRoot as exc:
            self._error(HTTPStatus.FORBIDDEN, str(exc))
            return
        if target.suffix != ".json":
            self._error(HTTPStatus.BAD_REQUEST, "only .json files can be written")
            return

        try:
            document = validate_experiment_payload(body["document"])
        except ValueError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            return

        # Lost-update guard: the client sends the mtime it loaded. A file that moved under it
        # means someone else (an editor, another tab, a `run`) wrote in the meantime, and
        # overwriting would throw their work away without anyone noticing.
        expected = body.get("modified")
        if target.exists() and expected is not None:
            actual = target.stat().st_mtime
            if abs(actual - float(expected)) > 1e-6:
                self._error(
                    HTTPStatus.CONFLICT,
                    f"{rel} changed on disk since it was loaded; reload before saving",
                )
                return

        try:
            modified = write_json_atomic(target, document)
        except (OSError, ValueError) as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"could not write {rel}: {exc}")
            return

        logger.info("path=<%s>, cases=<%d> | wrote experiment", rel, len(document["cases"]))
        self._json(
            HTTPStatus.OK,
            {
                "path": rel,
                "modified": modified,
                "cases": len(document["cases"]),
                "evaluator_notes": check_evaluators(document.get("evaluators")),
            },
        )


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], config: UIConfig):
        self.config = config
        super().__init__(address, handler)


def load_asset() -> str:
    """The single-file page shipped beside this module."""
    from importlib import resources

    return resources.files(__package__).joinpath("assets/ui.html").read_text(encoding="utf-8")


def build_server(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    depth: int = DEFAULT_DEPTH,
    read_only: bool = False,
    token: str | None = None,
    asset: str | None = None,
) -> _Server:
    """Bind a server for `root`. Port 0 picks a free one, which the tests rely on."""
    config = UIConfig(
        root=root.resolve(),
        token=token or secrets.token_urlsafe(32),
        depth=depth,
        read_only=read_only,
        asset=asset if asset is not None else load_asset(),
    )
    try:
        return _Server((host, port), UIHandler, config)
    except OSError as exc:
        raise OSError(f"could not bind {host}:{port}: {exc}") from exc


def server_url(server: _Server) -> str:
    host, port = server.server_address[:2]
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    if isinstance(shown, bytes):  # pragma: no cover - AF_UNIX never happens here
        shown = shown.decode()
    if ":" in str(shown) and not str(shown).startswith("["):
        shown = f"[{shown}]"
    return f"http://{shown}:{port}/?t={server.config.token}"


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])
