"""Local HTTP server backing `strands-evals ui`.

Bound to loopback, but it reads and writes files under the served directory, so it is an authenticated surface:

- a per-process token, printed in the URL, gates the page and every `/api` route.
- the `Host` header must name a loopback address, which blocks DNS rebinding.
- every client path is resolved and must stay inside the root, so `..` and outbound symlinks are rejected.
- a write carries the `modified` stamp the client loaded, and a file changed on disk since then is a 409.
"""

from __future__ import annotations

import functools
import hashlib
import html
import inspect
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import socket
import stat
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import parse_qs, quote, unquote, urlparse

from ..case import Case
from ..experiment import _builtin_evaluator_types
from ._ui_markdown import diagnosis_markdown, experiment_markdown, llms_text, report_markdown, session_markdown

logger = logging.getLogger(__name__)

DEFAULT_DEPTH = 3
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_SCAN_FILES = 2000
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})
WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::"})

# Where the served page finds its token. The address bar no longer carries it past the first request.
TOKEN_SLOT = '<meta name="strands-evals-token" content="">'
COOKIE_PREFIX = "strands_evals_ui_"

# A refused request's body is read and dropped only within these bounds; past them the connection closes.
DRAIN_LIMIT = 1024 * 1024
DRAIN_TIMEOUT = 2.0

# Capped on lines, not bytes: a cut log is reported as partial, and lines are what the page counts.
MAX_EVENT_LINES = 50_000

EXPERIMENT = "experiment"
REPORT = "report"
SESSION = "session"
DIAGNOSIS = "diagnosis"

EVENT_LOG_NAME = "events.jsonl"
RUN_CONFIG_NAME = "run.json"

SKIPPED_DIRS = frozenset({"node_modules", "__pycache__"})
# Where the page keeps its views: `/reports/<file>`, `/experiments/<file>`, `/optimization/<directory>/overview`.
PAGE_ROOTS = ("/reports", "/experiments", "/sessions", "/optimization")
# A report, an experiment or a session is also served as data: its page address with one of these forms added.
DATA_ROOTS = {"/reports/": REPORT, "/experiments/": EXPERIMENT, "/sessions/": SESSION}
DATA_FORMS = (".md", ".json")
PRINTABLE = re.compile(r"[!-~]+")

# A run directory's own storage. Scanning it would fill the skipped list and spend the file budget.
RUN_OWN_DIRS = frozenset({"transcripts", "candidates", "front"})
# Files a run writes beside its log; none is an experiment or a report.
RUN_OWN_FILES = frozenset({"effects.json", "best.values.json", "result.json"})

# How a saved task result opens: `EvaluationData` writes `input` first. Enough to tell one without parsing it.
TASK_RESULT_HEAD = re.compile(rb'\s*\{\s*"input"\s*:')
TASK_RESULT_KEYS = frozenset({"input", "actual_output", "expected_output", "actual_trajectory"})


def _prune_run_storage(filenames: list[str], dirnames: list[str]) -> None:
    """Drop a run directory's own storage from a walk, in place.

    Applies only where `run.json` and `events.jsonl` sit together; a `candidates` folder elsewhere is walked.
    """
    names = set(filenames)
    if RUN_CONFIG_NAME in names and EVENT_LOG_NAME in names:
        dirnames[:] = [d for d in dirnames if d not in RUN_OWN_DIRS]
        filenames[:] = [f for f in filenames if f.lower() not in RUN_OWN_FILES]


def _walk(root: Path, depth: int) -> Iterator[tuple[Path, list[str]]]:
    """Yield each directory under `root`, at most `depth` levels down, with its sorted filenames.

    Skips dot-directories, dependency folders, and a run directory's own storage.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        if len(here.relative_to(root).parts) >= depth:
            dirnames[:] = []
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIPPED_DIRS)
        _prune_run_storage(filenames, dirnames)
        yield here, sorted(filenames)


def is_page_path(path: str) -> bool:
    """Whether a request path is one of the page's own addresses. They all serve the one document."""
    return path in ("/", "/index.html") or any(path == root or path.startswith(f"{root}/") for root in PAGE_ROOTS)


def data_form(path: str) -> tuple[str, str, str] | None:
    """`(kind, file, form)` when a request path is the address of a file of that kind with a form added."""
    for root, kind in DATA_ROOTS.items():
        for form in DATA_FORMS:
            if path.startswith(root) and path.lower().endswith(f".json{form}"):
                return kind, unquote(path[len(root) : -len(form)]), form
    return None


def fingerprint(root: Path, depth: int = DEFAULT_DEPTH) -> str:
    """Digest the name, size and mtime of every file the page can show, so a change to the tree changes it."""
    digest = hashlib.sha256()
    for here, filenames in _walk(root, depth):
        names = [name for name in filenames if name.lower().endswith((".json", ".jsonl"))]
        if RUN_CONFIG_NAME in filenames and EVENT_LOG_NAME in filenames:
            names += sorted(RUN_OWN_FILES)
        for name in names:
            try:
                info = (here / name).stat()
            except OSError:
                continue
            where = (here / name).relative_to(root).as_posix()
            digest.update(f"{where}\0{info.st_size}\0{info.st_mtime_ns}\n".encode())
    return digest.hexdigest()


@dataclass
class ScannedFile:
    """One JSON file in the tree that parsed as something the page shows."""

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
        return _dirname(self.path)


@dataclass
class Discovery:
    experiments: list[dict] = field(default_factory=list)
    reports: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    truncated: bool = False
    optimizer_runs: list[dict] = field(default_factory=list)
    sessions: list[dict] = field(default_factory=list)
    diagnoses: list[dict] = field(default_factory=list)
    stores: list[dict] = field(default_factory=list)


@dataclass
class EventLog:
    """A parsed `events.jsonl`: the events, the lines that would not parse, and whether it was cut."""

    events: list[dict] = field(default_factory=list)
    bad_lines: list[dict] = field(default_factory=list)
    truncated: bool = False


# `classify`, `_stem`, `link_status`, `_report_summary`, `read_event_log` and `tuned_against` are mirrored in
# `assets/ui-folder.js` for the page's no-server mode, and `_session_summary`, `_diagnosis_summary`, `session_of`
# and `is_task_result` in `assets/ui-sessions.js`. Change one side and the other must follow.
def classify(obj: Any) -> str | None:
    """Identify a parsed JSON document by its shape, never by its filename.

    Both shapes carry `cases`: `evaluators` marks an experiment, the report's required fields mark a report.
    A session is traces of spans, as `fetch -o` writes one; a diagnosis is what `diagnose -o` writes about one.
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
    traces = obj.get("traces")
    if isinstance(obj.get("session_id"), str) and isinstance(traces, list):
        return SESSION if all(isinstance(t, dict) and isinstance(t.get("spans"), list) for t in traces) else None
    failures, causes = obj.get("failures"), obj.get("root_causes")
    found = isinstance(failures, list) and all(
        isinstance(f, dict) and isinstance(f.get("span_id"), str) for f in failures
    )
    explained = isinstance(causes, list) and all(
        isinstance(c, dict) and isinstance(c.get("failure_span_id"), str) for c in causes
    )
    if found and isinstance(obj.get("session_id"), str) and (causes is None or explained):
        return DIAGNOSIS
    # `--rca-only` writes root causes alone, with no session named.
    if failures is None and explained and causes:
        return DIAGNOSIS
    return None


def is_task_result(obj: Any) -> bool:
    """Whether a parsed document is one case's saved task output, as `LocalFileTaskResultStore` writes it."""
    return isinstance(obj, dict) and TASK_RESULT_KEYS <= obj.keys()


def _dirname(path: str) -> str:
    """The directory part of a root-relative path, `.` at the root; mirrors `ScannedFile.directory`."""
    return path.rsplit("/", 1)[0] if "/" in path else "."


_STEM_SUFFIX = re.compile(r"(^|[._-])(experiment|exp|report|rep)$", re.IGNORECASE)


def _stem(name: str) -> str:
    """Filename with the .json suffix and a trailing experiment/report token removed.

    The token comes off only at a separator or as the whole name: `nightly.experiment.json` gives `nightly`,
    `myexperiment.json` gives `myexperiment`, and `experiment.json` and `report.json` both give the empty stem.
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


def _experiment_case_ids(experiment_obj: dict) -> set[str]:
    """Every identifier a case in an experiment can be referred to by.

    Both `session_id` and `name`: report rows prefer `session_id`, optimizer splits hold whichever the run was given.
    """
    ids: set[str] = set()
    for case in experiment_obj.get("cases", []):
        if isinstance(case, dict):
            for key in ("session_id", "name"):
                if case.get(key) is not None:
                    ids.add(str(case[key]))
    return ids


def _split_case_ids(split: Any) -> set[str]:
    """Every case an optimizer run's split names, across train, validation and test.

    `run.json` has two shapes: names under `split.cases.train` with the fractions at `split.train`, or, in
    earlier records, names at `split.train` directly.
    """
    ids: set[str] = set()
    if not isinstance(split, dict):
        return ids
    nested = split.get("cases")
    for part in ("train", "validation", "test"):
        cases = (nested or {}).get(part) if isinstance(nested, dict) else None
        if not isinstance(cases, list):
            cases = split.get(part)
        if isinstance(cases, list):
            ids.update(str(case) for case in cases if isinstance(case, (str, int)))
    return ids


def tuned_against(split: Any, experiments: Sequence[ScannedFile]) -> dict | None:
    """Which experiment, if any, supplied the cases an optimizer run split up.

    `run.json` does not name its experiment, so this is a guess from case-id overlap and is reported as one.
    Returns `None` when nothing overlaps. The largest overlap wins, and ties go to the first path.
    """
    wanted = _split_case_ids(split)
    if not wanted:
        return None

    best: dict | None = None
    for experiment in sorted(experiments, key=lambda f: f.path):
        matched = len(wanted & _experiment_case_ids(experiment.obj))
        if matched and (best is None or matched > best["matched"]):
            best = {
                "path": experiment.path,
                "matched": matched,
                "split_cases": len(wanted),
                "experiment_cases": len(experiment.obj.get("cases", [])),
                "status": "verified" if matched == len(wanted) else "partial",
            }
    return best


def link_status(experiment_obj: dict, report_obj: dict) -> dict:
    """Whether a report plausibly came from the experiment it is being shown beside.

    Pairing by directory and filename is a guess, so the case-id overlap is returned as a `status`
    (`verified`, `partial`, `mismatch` or `unverified`) with a `detail` string.
    """
    exp_keys = _experiment_case_ids(experiment_obj)
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
    """Headline numbers for the Reports list, counted the way the library counts them."""
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


def _millis(stamp: Any) -> int | None:
    """A span's timestamp in whole milliseconds, which is all the page's `Date.parse` keeps of one."""
    if not isinstance(stamp, str):
        return None
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(moment.replace(microsecond=0).timestamp()) * 1000 + moment.microsecond // 1000


def _session_summary(obj: dict) -> dict:
    """How much a session recorded, and how long its traces ran for. The time between traces is not counted."""
    traces = [trace["spans"] for trace in obj["traces"]]
    millis: int | None = None
    for spans in traces:
        infos = [span.get("span_info") for span in spans if isinstance(span, dict)]
        ends = [info.get(key) for info in infos if isinstance(info, dict) for key in ("start_time", "end_time")]
        times = [millis_at for millis_at in map(_millis, ends) if millis_at is not None]
        if len(times) > 1:
            millis = (millis or 0) + max(times) - min(times)
    return {
        "session_id": obj["session_id"],
        "traces": len(traces),
        "spans": sum(len(spans) for spans in traces),
        "seconds": None if millis is None else millis / 1000,
    }


def _diagnosis_summary(obj: dict) -> dict:
    session_id = obj.get("session_id")
    return {
        "session_id": session_id if isinstance(session_id, str) else None,
        "failures": len(obj.get("failures") or []),
        "root_causes": len(obj.get("root_causes") or []),
    }


_SUMMARY: dict[str, Callable[[dict], dict]] = {
    EXPERIMENT: _experiment_summary,
    REPORT: _report_summary,
    SESSION: _session_summary,
    DIAGNOSIS: _diagnosis_summary,
}


def _file_entry(scanned: ScannedFile) -> dict:
    return {
        "path": scanned.path,
        "name": scanned.name,
        "size": scanned.size,
        "modified": scanned.modified,
        **_SUMMARY[scanned.kind](scanned.obj),
    }


def _opens_as_task_result(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return TASK_RESULT_HEAD.match(handle.read(64)) is not None
    except OSError:
        return False


def scan(root: Path, depth: int = DEFAULT_DEPTH) -> tuple[list[ScannedFile], list[dict], bool, dict[str, int]]:
    """Read every JSON file under `root` up to `depth` directories deep.

    Returns the files that classified, the ones that did not with the reason, whether the file cap was hit, and
    how many saved task results each task result store holds. Those are neither listed nor counted against the cap.
    """
    found: list[ScannedFile] = []
    skipped: list[dict] = []
    truncated = False
    stores: dict[str, int] = {}
    root = root.resolve()
    seen = 0

    for here, filenames in _walk(root, depth):
        for filename in filenames:
            if not filename.lower().endswith(".json"):
                continue
            full = here / filename
            rel = full.relative_to(root).as_posix()
            # One result read in full shows the directory is a store; the rest are told by how they open.
            if _dirname(rel) in stores and _opens_as_task_result(full):
                stores[_dirname(rel)] += 1
                continue
            if seen >= MAX_SCAN_FILES:
                truncated = True
                break
            seen += 1
            if full.is_symlink() and not _inside(root, full.resolve()):
                skipped.append({"path": rel, "why": "a link to a file outside this folder"})
                continue
            try:
                info = full.stat()
                obj = json.loads(full.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as exc:
                skipped.append({"path": rel, "why": f"could not read: {exc.__class__.__name__}"})
                continue
            except json.JSONDecodeError as exc:
                skipped.append({"path": rel, "why": f"invalid JSON at line {exc.lineno}: {exc.msg}"})
                continue

            kind = classify(obj)
            if kind is None and is_task_result(obj):
                stores[_dirname(rel)] = stores.get(_dirname(rel), 0) + 1
                seen -= 1
                continue
            if kind is None:
                skipped.append({"path": rel, "why": "not an experiment or a report"})
                continue
            found.append(ScannedFile(path=rel, kind=kind, obj=obj, size=info.st_size, modified=info.st_mtime))
        if truncated:
            break

    return found, skipped, truncated, stores


def _steps_apart(left: str, right: str) -> int:
    """How many directory steps separate two root-relative directories."""
    a = [] if left == "." else left.split("/")
    b = [] if right == "." else right.split("/")
    shared = 0
    while shared < min(len(a), len(b)) and a[shared] == b[shared]:
        shared += 1
    return len(a) + len(b) - 2 * shared


def experiment_of(report: ScannedFile, experiments: Sequence[ScannedFile], alone: bool) -> ScannedFile | None:
    """The experiment a report was made from, or `None` when that cannot be told.

    Beside it under the same name first, or as the only pair in the directory when `alone`. Failing that, the
    nearest experiment holding every case the report has. Two equally near are a tie, and a tie is not guessed.
    """
    beside = sorted((e for e in experiments if e.directory == report.directory), key=lambda e: e.path)
    named = [e for e in beside if _stem(e.name) == _stem(report.name)]
    if named:
        return named[0]
    if alone and len(beside) == 1:
        return beside[0]

    holding = [e for e in experiments if link_status(e.obj, report.obj)["status"] == "verified"]
    if not holding:
        return None
    nearest = min(_steps_apart(e.directory, report.directory) for e in holding)
    closest = [e for e in holding if _steps_apart(e.directory, report.directory) == nearest]
    return closest[0] if len(closest) == 1 else None


def session_of(diagnosis: ScannedFile, sessions: Sequence[ScannedFile]) -> ScannedFile | None:
    """The session a diagnosis is of: the nearest one with its `session_id`. A tie is not guessed."""
    named = [s for s in sessions if s.obj["session_id"] == diagnosis.obj.get("session_id")]
    if not named:
        return None
    nearest = min(_steps_apart(s.directory, diagnosis.directory) for s in named)
    closest = [s for s in named if _steps_apart(s.directory, diagnosis.directory) == nearest]
    return closest[0] if len(closest) == 1 else None


def _newest_first(scanned: ScannedFile) -> tuple[float, str]:
    """Sort key for a list: most recently changed first, and by path where that ties."""
    return (-scanned.modified, scanned.path)


def discover(root: Path, depth: int = DEFAULT_DEPTH) -> Discovery:
    """List the experiments and the reports in the tree, each report with the experiment it was made from.

    A report made in code has no experiment file, and one experiment can have many reports. Sessions are listed
    too, each with the diagnoses made of it, and a diagnosis whose session is not in the tree stands on its own.
    """
    found, skipped, truncated, stores = scan(root, depth)
    scanned_experiments = [f for f in found if f.kind == EXPERIMENT]
    scanned_reports = [f for f in found if f.kind == REPORT]

    reports: list[dict] = []
    made_from: dict[str, list[str]] = {}
    for report in sorted(scanned_reports, key=_newest_first):
        alone = sum(1 for other in scanned_reports if other.directory == report.directory) == 1
        experiment = experiment_of(report, scanned_experiments, alone)
        if experiment is not None:
            made_from.setdefault(experiment.path, []).append(report.path)
        reports.append(
            {
                **_file_entry(report),
                "dir": report.directory,
                "experiment": experiment.path if experiment else None,
                "link": link_status(experiment.obj, report.obj) if experiment else None,
            }
        )

    experiments = [
        {**_file_entry(experiment), "dir": experiment.directory, "reports": made_from.get(experiment.path, [])}
        for experiment in sorted(scanned_experiments, key=_newest_first)
    ]

    scanned_sessions = [f for f in found if f.kind == SESSION]
    diagnoses: list[dict] = []
    diagnosed: dict[str, list[str]] = {}
    for diagnosis in sorted((f for f in found if f.kind == DIAGNOSIS), key=_newest_first):
        session = session_of(diagnosis, scanned_sessions)
        if session is not None:
            diagnosed.setdefault(session.path, []).append(diagnosis.path)
        diagnoses.append(
            {**_file_entry(diagnosis), "dir": diagnosis.directory, "session": session.path if session else None}
        )
    sessions = [
        {**_file_entry(session), "dir": session.directory, "diagnoses": diagnosed.get(session.path, [])}
        for session in sorted(scanned_sessions, key=_newest_first)
    ]

    optimizer_runs = discover_optimizer_runs(root, depth)

    for optimizer_run in optimizer_runs:
        split = (optimizer_run.get("config") or {}).get("split")
        optimizer_run["tuned_against"] = tuned_against(split, scanned_experiments)

    # A `run.json` recognised as an optimizer run config is not a scan failure.
    optimizer_dirs = {r["dir"] for r in optimizer_runs}
    skipped = [
        s
        for s in skipped
        if not (s["path"].rsplit("/", 1)[-1] == RUN_CONFIG_NAME and _dirname(s["path"]) in optimizer_dirs)
    ]

    return Discovery(
        experiments=experiments,
        reports=reports,
        skipped=skipped,
        truncated=truncated,
        optimizer_runs=optimizer_runs,
        sessions=sessions,
        diagnoses=diagnoses,
        stores=[{"dir": directory, "files": files} for directory, files in sorted(stores.items())],
    )


def read_event_log(path: Path, max_lines: int = MAX_EVENT_LINES) -> EventLog:
    """Read an optimizer `events.jsonl`.

    A partial last line is dropped silently, because a live run is still appending. Any earlier line that
    fails to parse is reported with its line number.
    """
    log = EventLog()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.bad_lines.append({"line": 0, "why": f"could not read: {exc.__class__.__name__}"})
        return log

    lines = text.split("\n")
    # Only the last element can be a partial line: every earlier one ended in a newline.
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

    Only values read from the log. Derived series (best-so-far, the budget curve) are computed in the page.
    """
    # Distinct numbers, so an iteration recorded twice counts once.
    iterations = {e.get("iteration") for e in log.events if e.get("event") == "iteration_end"}
    champion = None
    stop_reason = None
    end_reports: dict[str, dict] = {}
    for event in log.events:
        name = event.get("event")
        if name in ("picked", "test"):
            report = event.get("report")
            if isinstance(report, dict):
                if report.get("champion_id"):
                    champion = report["champion_id"]
                end_reports[name] = report
        elif name == "stop":
            stop_reason = event.get("reason")

    # `test` (held-out split) outranks `picked` (validation split), whatever order the log has them in.
    end_report = end_reports.get("test") or end_reports.get("picked")
    objective = (end_report or {}).get("objective")
    result = {"objective": objective, "held_out": "test" in end_reports} if isinstance(objective, dict) else None

    return {
        "id": directory,
        "dir": directory,
        "run_id": config.get("run_id") or directory.rsplit("/", 1)[-1],
        "name": config.get("name"),
        "objective": config.get("objective"),
        "constraints": config.get("constraints") or [],
        # A multi-objective run says so in its objective string; older records carried a flag.
        "is_pareto": bool(config.get("is_pareto")) or str(config.get("objective") or "").startswith("Pareto("),
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
        "result": result,
        "stop_reason": stop_reason,
        "bad_lines": log.bad_lines,
        "truncated": log.truncated,
        # Filled in by `discover`, which holds the experiments to match the split against.
        "tuned_against": None,
    }


def discover_optimizer_runs(root: Path, depth: int = DEFAULT_DEPTH) -> list[dict]:
    """Find optimizer run directories: a `run.json` with an `events.jsonl` beside it.

    Both are required: `run.json` alone is too common a filename, and the log cannot be read without the config.
    """
    found: list[dict] = []
    root = root.resolve()

    for here, filenames in _walk(root, depth):
        if RUN_CONFIG_NAME not in filenames or EVENT_LOG_NAME not in filenames:
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
        summary = _optimizer_run_summary(relative, config, events_stat, config_stat, log)
        # Most runs have no effects model, so the page is told rather than left to ask for a file that is not there.
        summary["has_effects"] = (here / "effects.json").is_file()
        found.append(summary)

    found.sort(key=lambda r: -r["modified"])
    return found


class PathOutsideRoot(ValueError):
    """A client-supplied path resolved outside the served directory."""


def resolve_within(root: Path, relative: str) -> Path:
    """Resolve a client path against the root, refusing anything that escapes it.

    Resolves before checking, so a symlink pointing out of the tree is caught as well as a literal `..`.
    """
    if not relative:
        raise PathOutsideRoot("no path given")
    if "\x00" in relative:
        raise PathOutsideRoot(f"{relative!r} is not a path")
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not _inside(root, candidate):
        raise PathOutsideRoot(f"{relative} is outside {root}")
    return candidate


def _inside(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _same_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address, bound: str) -> bool:
    try:
        return address == ipaddress.ip_address(bound)
    except ValueError:  # bound by name, e.g. a hostname: only loopback names are answered
        return False


def is_loopback(host: str) -> bool:
    """Whether binding `host` keeps the server on this machine."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_timestamp(value: Any) -> TypeGuard[float]:
    """Whether a client-sent `modified` can be compared with an mtime."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) < 2**53
    return isinstance(value, float) and math.isfinite(value)


def _finite(value: Any) -> Any:
    """`value` with NaN and Infinity replaced by None, which is how JSON can carry them."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def validate_experiment_payload(payload: Any) -> dict:
    """Check an experiment document well enough to refuse to write a broken one.

    Cases go through `Case.model_validate`, the same gate `Experiment.from_file` uses. Evaluators are never
    instantiated, because that would reject custom evaluator subclasses; `check_evaluators` inspects them.
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

    Uses `inspect.signature` on the built-in registry, never construction. The notes are advice and block nothing.
    An unregistered name is only a warning: custom evaluators are passed to `from_dict` by the caller.
    """
    if not isinstance(evaluators, list):
        return []

    registry = _builtin_evaluator_types()
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


@functools.cache
def evaluator_catalog() -> dict[str, dict]:
    """What the page says about each built-in evaluator: its summary and its settings, read off the class.

    Read from the library so that the page describes an evaluator added later without being edited.
    """
    catalog: dict[str, dict] = {}
    for name, target in _builtin_evaluator_types().items():
        try:
            parameters = list(inspect.signature(target).parameters.values())
        except (TypeError, ValueError):  # a builtin or C-implemented __init__
            parameters = []
        settings = [
            {
                "name": parameter.name,
                "required": parameter.default is inspect.Parameter.empty,
                "default": None if parameter.default is inspect.Parameter.empty else _shown(parameter.default),
            }
            for parameter in parameters
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        summary = (inspect.getdoc(target) or "").strip().split("\n", 1)[0]
        catalog[name] = {"summary": summary, "settings": settings}
    return catalog


def _shown(default: Any) -> str:
    """A setting's default as a user would write it in the file, cut short when it is a long prompt."""
    text = json.dumps(default) if isinstance(default, (str, int, float, bool, type(None))) else repr(default)
    return text if len(text) <= 48 else text[:45] + "..."


def write_json_atomic(target: Path, payload: dict) -> float:
    """Serialize, then replace the file in one step; returns the new mtime.

    Serializing first leaves the existing file untouched when the document is not valid JSON
    (NaN and Infinity included), matching `Experiment.to_file`.
    """
    data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        mode = None
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
    )
    try:
        with handle as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # The temp file is created 0600; without this a save would narrow the file's permissions.
        if mode is not None:
            os.chmod(handle.name, mode)
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target.stat().st_mtime


UNAUTHORIZED_PAGE = (
    b"<!doctype html><meta charset=utf-8><title>strands-evals</title>"
    b"<body style='font:14px \"Space Grotesk\",system-ui,sans-serif;padding:40px'>"
    b"<h3 style='font:600 18px \"JetBrains Mono\",ui-monospace,monospace'>Open the link the CLI printed</h3>"
    b"<p>This page needs the token in the URL that <code>strands-evals ui</code> printed.</p>"
)


@dataclass
class UIConfig:
    root: Path
    token: str
    depth: int = DEFAULT_DEPTH
    read_only: bool = False
    asset: str = ""
    # True when `asset` came from the shipped file rather than a caller, so it can be re-read.
    reload_asset: bool = False
    # The address the server was asked to bind, which decides which `Host` headers it answers.
    bound_host: str = "127.0.0.1"
    # Carried by the page's own <style> and <script> tags; markup injected from a file cannot know it.
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(16))

    def page(self) -> str:
        """The page to serve, re-read from disk on every request unless a caller pinned one."""
        if not self.reload_asset:
            return self.asset
        try:
            return load_asset(self.nonce)
        except OSError as exc:
            logger.warning("error=<%s> | could not re-read the ui page, serving the copy read at startup", exc)
            return self.asset

    def content_security_policy(self) -> str:
        own = f"'nonce-{self.nonce}'"
        return (
            f"default-src 'none'; script-src {own}; style-src {own}; style-src-attr 'unsafe-inline'; "
            "img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        )


class UIHandler(BaseHTTPRequestHandler):
    """Routes for the UI. `config` is attached to the server by `build_server`."""

    server_version = "strands-evals-ui"
    protocol_version = "HTTP/1.1"
    timeout = 60  # an idle or stalled connection gives its thread back

    @property
    def config(self) -> UIConfig:
        return self.server.config  # type: ignore[attr-defined]

    # ── plumbing ──

    def log_message(self, fmt: str, *args: Any) -> None:
        # The base class writes to stderr; logging puts it under the CLI's -v/--debug flags.
        line = _TOKEN_IN_URL.sub(r"\1<redacted>", fmt % args)
        logger.debug("client=<%s> | %s", self.address_string(), line)

    def _send(
        self, status: HTTPStatus, body: bytes, content_type: str, *, headers: dict[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        try:
            text = json.dumps(payload, allow_nan=False)
        except ValueError:  # a file on disk held NaN or Infinity, which Python reads and JSON forbids
            text = json.dumps(_finite(payload), allow_nan=False)
        self._send(status, text.encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    def _host_is_admitted(self) -> bool:
        """Whether `Host` names this server, which is what stops DNS rebinding.

        A rebinding request carries the attacker's hostname. Loopback names are always answered; a server
        bound beyond loopback also answers to the address it was bound to, or to any IP literal when it was
        bound to every interface.
        """
        match = _HOST_HEADER.fullmatch((self.headers.get("Host") or "").strip())
        if match is None:
            return False
        name = match.group(1).lower()
        if name in LOOPBACK_HOSTS:
            return True
        try:
            address = ipaddress.ip_address(name.strip("[]"))
        except ValueError:
            return False
        bound = self.config.bound_host
        return bound in WILDCARD_HOSTS or _same_address(address, bound)

    def _is_token(self, supplied: str) -> bool:
        # Compared as bytes: compare_digest rejects non-ASCII str, and the value is attacker-chosen.
        return secrets.compare_digest(supplied.encode("utf-8", "replace"), self.config.token.encode("utf-8"))

    def _authorized(self, query: dict[str, list[str]]) -> bool:
        return self._is_token((self.headers.get("X-Evals-Token") or "") or (query.get("t", [""])[0]))

    def _cookie_name(self) -> str:
        # Cookies ignore the port, so two servers on one machine need two names.
        return f"{COOKIE_PREFIX}{self.connection.getsockname()[1]}"

    def _has_page_cookie(self) -> bool:
        try:
            morsel = SimpleCookie(self.headers.get("Cookie") or "").get(self._cookie_name())
        except CookieError:
            return False
        return morsel is not None and self._is_token(unquote(morsel.value))

    def _refusal(self, query: dict[str, list[str]]) -> tuple[HTTPStatus, str] | None:
        """Why this request is not admitted, if it is not: the wrong Host or the wrong token."""
        if not self._host_is_admitted():
            return HTTPStatus.FORBIDDEN, "this server only answers requests addressed to the host it is serving on"
        if not self._authorized(query):
            return HTTPStatus.UNAUTHORIZED, "missing or wrong token"
        return None

    def _settings(self) -> dict:
        return {"root": str(self.config.root), "depth": self.config.depth, "read_only": self.config.read_only}

    def _requested_file(
        self, query: dict[str, list[str]], missing: str = "{rel} is not a file"
    ) -> tuple[str, Path] | None:
        """The `path` parameter resolved to a file inside the root, or None once the error is sent."""
        rel = query.get("path", [""])[0]
        try:
            target = resolve_within(self.config.root, rel)
        except PathOutsideRoot as exc:
            self._error(HTTPStatus.FORBIDDEN, str(exc))
            return None
        if not target.is_file():
            self._error(HTTPStatus.NOT_FOUND, missing.format(rel=rel))
            return None
        return rel, target

    def _body_length(self) -> int | None:
        """The declared body size, or None when it cannot be read by length.

        That is a missing or malformed `Content-Length`, a chunked body, or a size over the cap.
        """
        declared = self.headers.get("Content-Length")
        if declared is None or self.headers.get("Transfer-Encoding"):
            return None
        try:
            length = int(declared)
        except ValueError:
            return None
        return length if 0 <= length <= MAX_UPLOAD_BYTES else None

    def _reject_unread(self, status: HTTPStatus, message: str) -> None:
        """Refuse a request whose body has not been read.

        A small body is drained first: left unread, a kept-alive connection would parse it as the next
        request. Anything else closes the connection, so a client cannot hold a thread by withholding a body.
        """
        remaining = self._body_length()
        if remaining is None or remaining > DRAIN_LIMIT:
            self.close_connection = True
        else:
            self.connection.settimeout(DRAIN_TIMEOUT)
            try:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 65536))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except OSError:  # the declared body never arrived
                self.close_connection = True
            finally:
                self.connection.settimeout(self.timeout)
        self._error(status, message)

    def _json_body(self) -> dict | None:
        """The request body as a JSON object, or None once the error is sent."""
        length = self._body_length()
        if length is None:
            declared = self.headers.get("Content-Length") or ""
            if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
                self._reject_unread(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body over {MAX_UPLOAD_BYTES} bytes")
            else:
                self._reject_unread(HTTPStatus.LENGTH_REQUIRED, "send the body with a Content-Length")
            return None
        if length == 0:
            self._error(HTTPStatus.BAD_REQUEST, "empty body")
            return None

        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"body is not JSON: {exc}")
            return None
        if not isinstance(body, dict):
            self._error(HTTPStatus.BAD_REQUEST, "body is not a JSON object")
            return None
        return body

    # ── routes ──

    def _get_config(self, query: dict[str, list[str]]) -> None:
        """The served folder, the scan depth, whether saving is off, and the evaluator catalogue."""
        self._json(HTTPStatus.OK, {**self._settings(), "evaluators": evaluator_catalog()})

    def _get_scan(self, query: dict[str, list[str]]) -> None:
        """Every experiment, report, session, diagnosis and optimizer run under the served folder, and a fingerprint."""
        # Taken before the scan, so a file written during it reads as a change afterwards.
        mark = fingerprint(self.config.root, self.config.depth)
        result = discover(self.config.root, self.config.depth)
        self._json(
            HTTPStatus.OK,
            {
                **self._settings(),
                "fingerprint": mark,
                "experiments": result.experiments,
                "reports": result.reports,
                "skipped": result.skipped,
                "truncated": result.truncated,
                "optimizer_runs": result.optimizer_runs,
                "sessions": result.sessions,
                "diagnoses": result.diagnoses,
                "stores": result.stores,
            },
        )

    def _get_changes(self, query: dict[str, list[str]]) -> None:
        """The fingerprint now. It differs from a scan's once something the scan listed has changed."""
        self._json(HTTPStatus.OK, {"fingerprint": fingerprint(self.config.root, self.config.depth)})

    def _get_events(self, query: dict[str, list[str]]) -> None:
        """With `?path=<events.jsonl>`: the events an optimizer run logged, one per line of the file."""
        requested = self._requested_file(query)
        if requested is None:
            return
        rel, target = requested
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

    def _get_file(self, query: dict[str, list[str]]) -> None:
        """With `?path=<file>`: one JSON file under the served folder, with the time it was last modified."""
        requested = self._requested_file(query)
        if requested is None:
            return
        rel, target = requested
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

    def _get_llms(self, query: dict[str, list[str]]) -> None:
        """This description of what the server answers."""
        routes = {route: inspect.getdoc(handler) or "" for route, handler in self._GET_ROUTES.items()}
        text = llms_text(routes, PAGE_ROOTS, DATA_ROOTS, DATA_FORMS)
        self._send(HTTPStatus.OK, text.encode("utf-8"), "text/plain; charset=utf-8")

    def _get_data_form(self, wanted: str, rel: str, form: str) -> None:
        """A report, an experiment or a session as Markdown or as JSON, for a reader with no browser."""
        requested = self._requested_file({"path": [rel]})
        if requested is None:
            return
        target = requested[1]
        try:
            obj = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{rel} could not be parsed: {exc}")
            return
        kind = classify(obj)
        # A session's address also names a diagnosis: the page opens one there when its session is not in the folder.
        if kind != wanted and (wanted, kind) != (SESSION, DIAGNOSIS):
            self._error(HTTPStatus.NOT_FOUND, f"{rel} is not {'an' if wanted == EXPERIMENT else 'a'} {wanted}")
            return
        if form == ".json":
            self._json(HTTPStatus.OK, obj)
            return
        obj = _finite(obj)  # as the page is sent it, so both write the same Markdown
        # The path as the scan writes it, so the file is found in the listing however it was asked for.
        root = self.config.root.resolve()
        path = target.relative_to(root).as_posix()
        found = discover(self.config.root, self.config.depth)

        def listed(entries: list[dict]) -> dict:
            return next((entry for entry in entries if entry["path"] == path), {})

        if kind == REPORT:
            text = report_markdown(path, obj, listed(found.reports).get("experiment"))
        elif kind == EXPERIMENT:
            text = experiment_markdown(path, obj, listed(found.experiments).get("reports") or [])
        elif kind == DIAGNOSIS:
            text = diagnosis_markdown(path, obj, listed(found.diagnoses).get("session"))
        else:
            diagnoses = []
            for made in listed(found.sessions).get("diagnoses") or []:
                try:
                    diagnoses.append((made, _finite(json.loads((root / made).read_text(encoding="utf-8")))))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{made} could not be parsed: {exc}")
                    return
            text = session_markdown(path, obj, _session_summary(obj)["seconds"], diagnoses)
        self._send(HTTPStatus.OK, text.encode("utf-8", "replace"), "text/markdown; charset=utf-8")

    def _put_experiment(self, query: dict[str, list[str]], body: dict) -> None:
        if "document" not in body:
            self._error(HTTPStatus.BAD_REQUEST, "expected {'document': ..., 'modified': ...}")
            return
        requested = self._requested_file(query, missing="{rel} is no longer on disk, so it cannot be saved in place")
        if requested is None:
            return
        rel, target = requested
        if target.suffix.lower() != ".json":
            self._error(HTTPStatus.BAD_REQUEST, "only .json files can be written")
            return

        try:
            document = validate_experiment_payload(body["document"])
        except ValueError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            return

        expected = body.get("modified")
        if not _is_timestamp(expected):
            self._error(HTTPStatus.BAD_REQUEST, "'modified' must be the number the file was loaded with")
            return
        # The page saves back to the experiment it loaded; it never creates a file or replaces another kind.
        try:
            current = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            current = None
        if classify(current) != EXPERIMENT:
            self._error(HTTPStatus.CONFLICT, f"{rel} is not an experiment file")
            return
        # Lost-update guard: a file whose mtime differs from the one the client loaded was written by someone else.
        if abs(target.stat().st_mtime - expected) > 1e-6:
            self._error(HTTPStatus.CONFLICT, f"{rel} changed on disk since it was loaded; reload before saving")
            return
        if not os.access(target, os.W_OK):
            self._error(HTTPStatus.FORBIDDEN, f"{rel} is read-only")
            return

        try:
            modified = write_json_atomic(target, document)
        except ValueError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{rel} is not valid JSON: {exc}")
            return
        except OSError as exc:
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

    def _get_page(self, query: dict[str, list[str]]) -> None:
        """Serve the page to a request that proves it has the token.

        The printed link carries the token once. It is answered with a cookie and a redirect to the bare
        address, so the token is not left in the address bar, the history or a shared link. The cookie
        only opens the page: every `/api` route still needs the token itself, which the page reads out
        of the document it was served.
        """
        html_type = "text/html; charset=utf-8"
        if "t" in query and self._authorized(query):
            cookie = f"{self._cookie_name()}={quote(self.config.token, safe='')}; Path=/; HttpOnly; SameSite=Lax"
            # The same address without the token, so a link into the page lands where it pointed.
            link = urlparse(self.path)
            rest = "&".join(pair for pair in link.query.split("&") if pair and pair.split("=")[0] != "t")
            path = link.path + (f"?{rest}" if rest else "")
            where = path if PRINTABLE.fullmatch(path) else "/"
            self._send(HTTPStatus.SEE_OTHER, b"", html_type, headers={"Location": where, "Set-Cookie": cookie})
            return
        if not (self._authorized(query) or self._has_page_cookie()):
            self._send(HTTPStatus.UNAUTHORIZED, UNAUTHORIZED_PAGE, html_type)
            return
        slot = TOKEN_SLOT.replace('""', f'"{html.escape(self.config.token, quote=True)}"')
        page = self.config.page().replace(TOKEN_SLOT, slot, 1)
        policy = {"Content-Security-Policy": self.config.content_security_policy()}
        self._send(HTTPStatus.OK, page.encode("utf-8"), html_type, headers=policy)

    _GET_ROUTES: dict[str, Callable[[UIHandler, dict[str, list[str]]], None]] = {
        "/api/config": _get_config,
        "/api/scan": _get_scan,
        "/api/changes": _get_changes,
        "/api/events": _get_events,
        "/api/file": _get_file,
        "/llms.txt": _get_llms,
    }

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        form = data_form(parsed.path)
        is_page = form is None and is_page_path(parsed.path)

        if is_page and self._host_is_admitted():
            self._get_page(query)
            return
        refusal = self._refusal(query)
        if refusal:
            self._error(*refusal)
            return
        if form:
            self._get_data_form(*form)
            return

        route = self._GET_ROUTES.get(parsed.path)
        if route is None:
            self._error(HTTPStatus.NOT_FOUND, f"no route for {parsed.path}")
            return
        route(self, query)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        refusal = self._refusal(query)
        if refusal is None and parsed.path != "/api/experiment":
            refusal = HTTPStatus.NOT_FOUND, f"no route for {parsed.path}"
        if refusal is None and self.config.read_only:
            refusal = HTTPStatus.FORBIDDEN, "server started with --read-only"
        if refusal:
            self._reject_unread(*refusal)
            return

        body = self._json_body()
        if body is not None:
            self._put_experiment(query, body)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], config: UIConfig):
        self.config = config
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, handler)


_LOCAL_ASSET = re.compile(r'<link rel="stylesheet" href="([\w.-]+)">|<script src="([\w.-]+)"></script>')
_ANY_ASSET = re.compile(r"<script[^>]*\ssrc=|<link[^>]*\srel=\"stylesheet\"")
_HOST_HEADER = re.compile(r"(\[[0-9A-Fa-f:.]+\]|[^:\[\]]+)(?::\d+)?")
_TOKEN_IN_URL = re.compile(r"([?&]t=)[^&\s\"]+")


def load_asset(nonce: str = "") -> str:
    """The page shipped beside this module, with its stylesheet and scripts inlined.

    One document keeps every byte of the page behind the token and needs no static routes.
    `tests/ui/harness.mjs` assembles the page the same way for the browser tests.

    Raises:
        ValueError: If `ui.html` references an asset in a form this cannot inline.
    """
    assets = resources.files(__package__).joinpath("assets")
    source = assets.joinpath("ui.html").read_text(encoding="utf-8")
    if len(_ANY_ASSET.findall(source)) != len(_LOCAL_ASSET.findall(source)):
        raise ValueError(
            'ui.html may only reference assets as <link rel="stylesheet" href="NAME"> or <script src="NAME"></script>'
        )
    attribute = f' nonce="{nonce}"' if nonce else ""

    def inline(match: re.Match[str]) -> str:
        stylesheet, script = match.groups()
        if stylesheet:
            return f"<style{attribute}>\n{assets.joinpath(stylesheet).read_text(encoding='utf-8')}</style>"
        return f"<script{attribute}>\n{assets.joinpath(script).read_text(encoding='utf-8')}</script>"

    return _LOCAL_ASSET.sub(inline, source)


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
        reload_asset=asset is None,
        bound_host=host,
    )
    config.asset = asset if asset is not None else load_asset(config.nonce)
    try:
        return _Server((host, port), UIHandler, config)
    except OSError as exc:
        raise OSError(f"could not bind {host}:{port}: {exc}") from exc


def server_url(server: _Server) -> str:
    host, port = server.server_address[:2]
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else "::1" if host == "::" else host
    if isinstance(shown, bytes):  # pragma: no cover - AF_UNIX never happens here
        shown = shown.decode()
    if ":" in str(shown) and not str(shown).startswith("["):
        shown = f"[{shown}]"
    return f"http://{shown}:{port}/?t={server.config.token}"
