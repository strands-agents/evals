"""What `strands-evals ui` finds in a folder: experiments, reports, sessions, diagnoses and optimizer runs.

The page reads a folder the same way when it has no server (`assets/ui-folder.js` and `assets/ui-sessions.js`),
and `tests/ui/parity.mjs` holds the two to the same listing.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ._ui_evaluators import check_evaluators

DEFAULT_DEPTH = 3
MAX_SCAN_FILES = 2000

# Capped on lines, not bytes: a cut log is reported as partial, and lines are what the page counts.
MAX_EVENT_LINES = 50_000

EXPERIMENT = "experiment"
REPORT = "report"
SESSION = "session"
DIAGNOSIS = "diagnosis"

EVENT_LOG_NAME = "events.jsonl"
RUN_CONFIG_NAME = "run.json"

SKIPPED_DIRS = frozenset({"node_modules", "__pycache__"})

# A run directory's own storage. Scanning it would fill the skipped list and spend the file budget.
RUN_OWN_DIRS = frozenset({"transcripts", "candidates", "front"})
# Files a run writes beside its log; none is an experiment or a report.
RUN_OWN_FILES = frozenset({"effects.json", "best.values.json", "result.json"})

# How a saved task result opens: `EvaluationData` writes `input` first. Enough to tell one without parsing it.
TASK_RESULT_HEAD = re.compile(rb'\s*\{\s*"input"\s*:')
TASK_RESULT_KEYS = frozenset({"input", "actual_output", "expected_output", "actual_trajectory"})

# What reading a file as JSON can raise. `RecursionError` is a document nested too deeply for the reader.
NOT_JSON = (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError)


def _prune_run_storage(filenames: list[str], dirnames: list[str]) -> None:
    """Drop a run directory's own storage from a walk, in place.

    Applies only where `run.json` and `events.jsonl` sit together; a `candidates` folder elsewhere is walked.
    """
    names = set(filenames)
    if RUN_CONFIG_NAME in names and EVENT_LOG_NAME in names:
        dirnames[:] = [d for d in dirnames if d not in RUN_OWN_DIRS]
        filenames[:] = [f for f in filenames if f.lower() not in RUN_OWN_FILES]


def _walk(root: Path, depth: int) -> Iterator[tuple[Path, str, list[str]]]:
    """Yield each directory under `root`, `depth` levels down at most: its path, its root-relative name, its files.

    The root's own name is `.`. Skips dot-directories, dependency folders, and a run directory's own storage.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        where = here.relative_to(root)
        if len(where.parts) >= depth:
            dirnames[:] = []
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIPPED_DIRS)
        _prune_run_storage(filenames, dirnames)
        yield here, where.as_posix(), sorted(filenames)


def _under(where: str, name: str) -> str:
    """The root-relative path of the file `name` in the directory `where`."""
    return name if where == "." else f"{where}/{name}"


def fingerprint(root: Path, depth: int = DEFAULT_DEPTH) -> str:
    """Digest the name, size and mtime of every file the page can show, so a change to the tree changes it."""
    digest = hashlib.sha256()
    for here, where, filenames in _walk(root, depth):
        names = [name for name in filenames if name.lower().endswith((".json", ".jsonl"))]
        if RUN_CONFIG_NAME in filenames and EVENT_LOG_NAME in filenames:
            names += sorted(RUN_OWN_FILES)
        for name in names:
            try:
                info = (here / name).stat()
            except OSError:
                continue
            digest.update(f"{_under(where, name)}\0{info.st_size}\0{info.st_mtime_ns}\n".encode())
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

    @functools.cached_property
    def case_ids(self) -> set[str]:
        """For an experiment: every identifier one of its cases can be referred to by."""
        return _experiment_case_ids(self.obj)


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
    """The directory part of a root-relative path, `.` at the root."""
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
        matched = len(wanted & experiment.case_ids)
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
    detailed = obj.get("detailed_results")
    detailed = detailed if isinstance(detailed, list) else []
    passes = obj.get("test_passes") or []
    failing = 0
    not_applicable = 0
    for index, _ in enumerate(obj.get("cases", [])):
        outputs = detailed[index] if index < len(detailed) and isinstance(detailed[index], list) else []
        # Mirrors EvaluationReport.is_applicable: a row is droppable only when it has outputs
        # and every one of them both declined to judge and passed.
        declined = bool(outputs) and all(
            isinstance(o, dict) and o.get("label") == "not_applicable" and o.get("test_pass") for o in outputs
        )
        if declined:
            not_applicable += 1
        elif index >= len(passes) or not passes[index]:
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
        seconds = int(moment.replace(microsecond=0).timestamp())
    except (ValueError, OverflowError, OSError):  # not a time, or one the platform's clock cannot place
        return None
    return seconds * 1000 + moment.microsecond // 1000


def session_seconds(obj: dict) -> float | None:
    """How long a session's traces ran for. The time between traces is not counted."""
    millis: int | None = None
    for trace in obj["traces"]:
        infos = [span.get("span_info") for span in trace["spans"] if isinstance(span, dict)]
        ends = [info.get(key) for info in infos if isinstance(info, dict) for key in ("start_time", "end_time")]
        times = [millis_at for millis_at in map(_millis, ends) if millis_at is not None]
        if len(times) > 1:
            millis = (millis or 0) + max(times) - min(times)
    return None if millis is None else millis / 1000


def _session_summary(obj: dict) -> dict:
    return {
        "session_id": obj["session_id"],
        "traces": len(obj["traces"]),
        "spans": sum(len(trace["spans"]) for trace in obj["traces"]),
        "seconds": session_seconds(obj),
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


def _file_entry(scanned: ScannedFile, **links: Any) -> dict:
    """What a list shows for one file. `links` are the files it was paired with."""
    return {
        "path": scanned.path,
        "name": scanned.name,
        "size": scanned.size,
        "modified": scanned.modified,
        **_SUMMARY[scanned.kind](scanned.obj),
        "dir": scanned.directory,
        **links,
    }


def _opens_as_task_result(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return TASK_RESULT_HEAD.match(handle.read(64)) is not None
    except OSError:
        return False


def _json_files(root: Path, depth: int) -> Iterator[tuple[Path, str, str]]:
    """Each `.json` file the walk reaches: its path, its directory under the root and its root-relative path."""
    for here, where, filenames in _walk(root, depth):
        for filename in filenames:
            if filename.lower().endswith(".json"):
                yield here / filename, where, _under(where, filename)


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

    for full, where, rel in _json_files(root, depth):
        # One result read in full shows the directory is a store; the rest are told by how they open.
        if where in stores and not full.is_symlink() and _opens_as_task_result(full):
            stores[where] += 1
            continue
        if len(found) + len(skipped) >= MAX_SCAN_FILES:
            truncated = True
            break
        if full.is_symlink() and not is_inside(root, real_path(full)):
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
        except RecursionError:
            skipped.append({"path": rel, "why": "nested too deeply to read"})
            continue

        kind = classify(obj)
        if kind is not None:
            found.append(ScannedFile(path=rel, kind=kind, obj=obj, size=info.st_size, modified=info.st_mtime))
        elif is_task_result(obj):
            stores[where] = stores.get(where, 0) + 1
        else:
            skipped.append({"path": rel, "why": "not an experiment or a report"})

    return found, skipped, truncated, stores


def _steps_apart(left: str, right: str) -> int:
    """How many directory steps separate two root-relative directories."""
    a = [] if left == "." else left.split("/")
    b = [] if right == "." else right.split("/")
    shared = 0
    while shared < min(len(a), len(b)) and a[shared] == b[shared]:
        shared += 1
    return len(a) + len(b) - 2 * shared


def _nearest(candidates: Sequence[ScannedFile], directory: str) -> ScannedFile | None:
    """The one of `candidates` fewest steps from `directory`. Two equally near are a tie, and a tie is not guessed."""
    if not candidates:
        return None
    steps = [_steps_apart(candidate.directory, directory) for candidate in candidates]
    closest = [candidate for candidate, apart in zip(candidates, steps, strict=True) if apart == min(steps)]
    return closest[0] if len(closest) == 1 else None


def experiment_of(report: ScannedFile, experiments: Sequence[ScannedFile], alone: bool) -> ScannedFile | None:
    """The experiment a report was made from, or `None` when that cannot be told.

    The one beside it under the same name first, or the only pair in the directory when `alone`. Failing that,
    the nearest experiment holding every case the report has.
    """
    beside = sorted((e for e in experiments if e.directory == report.directory), key=lambda e: e.path)
    named = [e for e in beside if _stem(e.name) == _stem(report.name)]
    if len(named) == 1:
        return named[0]
    if alone and len(beside) == 1:
        return beside[0]

    # The experiments `link_status` would call verified.
    wanted = _case_keys(report.obj)
    return _nearest([e for e in experiments if wanted and wanted <= e.case_ids], report.directory)


def session_of(diagnosis: ScannedFile, sessions: Sequence[ScannedFile]) -> ScannedFile | None:
    """The session a diagnosis is of: the nearest one with its `session_id`."""
    named = [s for s in sessions if s.obj["session_id"] == diagnosis.obj.get("session_id")]
    return _nearest(named, diagnosis.directory)


def _newest_first(scanned: ScannedFile) -> tuple[float, str]:
    """Sort key for a list: most recently changed first, and by path where that ties."""
    return (-scanned.modified, scanned.path)


def discover(root: Path, depth: int = DEFAULT_DEPTH) -> Discovery:
    """List the experiments and the reports in the tree, each report with the experiment it was made from.

    A report made in code has no experiment file, and one experiment can have many reports. Sessions are listed
    too, each with the diagnoses made of it, and a diagnosis whose session is not in the tree stands on its own.
    """
    found, skipped, truncated, stores = scan(root, depth)
    of_kind: dict[str, list[ScannedFile]] = {kind: [] for kind in _SUMMARY}
    for scanned in sorted(found, key=_newest_first):
        of_kind[scanned.kind].append(scanned)

    reports: list[dict] = []
    made_from: dict[str, list[str]] = {}
    reports_in = Counter(report.directory for report in of_kind[REPORT])
    for report in of_kind[REPORT]:
        experiment = experiment_of(report, of_kind[EXPERIMENT], alone=reports_in[report.directory] == 1)
        if experiment is not None:
            made_from.setdefault(experiment.path, []).append(report.path)
        reports.append(
            _file_entry(
                report,
                experiment=experiment.path if experiment else None,
                link=link_status(experiment.obj, report.obj) if experiment else None,
            )
        )

    diagnoses: list[dict] = []
    diagnosed: dict[str, list[str]] = {}
    for diagnosis in of_kind[DIAGNOSIS]:
        session = session_of(diagnosis, of_kind[SESSION])
        if session is not None:
            diagnosed.setdefault(session.path, []).append(diagnosis.path)
        diagnoses.append(_file_entry(diagnosis, session=session.path if session else None))

    optimizer_runs = discover_optimizer_runs(root, depth)
    for optimizer_run in optimizer_runs:
        split = (optimizer_run.get("config") or {}).get("split")
        optimizer_run["tuned_against"] = tuned_against(split, of_kind[EXPERIMENT])

    # A `run.json` recognised as an optimizer run config is not a scan failure.
    run_configs = {_under(run["dir"], RUN_CONFIG_NAME) for run in optimizer_runs}
    skipped = [skip for skip in skipped if skip["path"] not in run_configs] + _lone_event_logs(root, depth)

    return Discovery(
        experiments=[_file_entry(e, reports=made_from.get(e.path, [])) for e in of_kind[EXPERIMENT]],
        reports=reports,
        skipped=skipped,
        truncated=truncated,
        optimizer_runs=optimizer_runs,
        sessions=[_file_entry(s, diagnoses=diagnosed.get(s.path, [])) for s in of_kind[SESSION]],
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
        except RecursionError:
            log.bad_lines.append({"line": number, "why": "nested too deeply to read"})
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
    # Distinct, so an iteration recorded twice counts once. As text, so one that is not a number still counts.
    iterations = {str(e.get("iteration")) for e in log.events if e.get("event") == "iteration_end"}
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
        "run_id": config.get("run_id") or ("this folder" if directory == "." else directory.rsplit("/", 1)[-1]),
        "name": config.get("name"),
        "objective": config.get("objective"),
        "constraints": config.get("constraints") or [],
        # A multi-objective run says so in its objective string; older records carried a flag.
        "is_pareto": bool(config.get("is_pareto")) or str(config.get("objective") or "").startswith("Pareto("),
        "budget": config.get("budget") or {},
        "config": config,
        "events_path": _under(directory, EVENT_LOG_NAME),
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


def _lone_event_logs(root: Path, depth: int) -> list[dict]:
    """An `events.jsonl` with no `run.json` beside it is not a run: each is skipped, saying what it lacks."""
    return [
        {"path": _under(where, EVENT_LOG_NAME), "why": "no run.json beside it"}
        for _, where, filenames in _walk(root, depth)
        if EVENT_LOG_NAME in filenames and RUN_CONFIG_NAME not in filenames
    ]


def discover_optimizer_runs(root: Path, depth: int = DEFAULT_DEPTH) -> list[dict]:
    """Find optimizer run directories: a `run.json` with an `events.jsonl` beside it.

    Both are required: `run.json` alone is too common a filename, and the log cannot be read without the config.
    """
    found: list[dict] = []
    root = root.resolve()

    for here, where, filenames in _walk(root, depth):
        if RUN_CONFIG_NAME not in filenames or EVENT_LOG_NAME not in filenames:
            continue

        config_path, events_path = here / RUN_CONFIG_NAME, here / EVENT_LOG_NAME
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config_stat, events_stat = config_path.stat(), events_path.stat()
        except NOT_JSON:
            continue
        if not isinstance(config, dict):
            continue

        log = read_event_log(events_path)
        summary = _optimizer_run_summary(where, config, events_stat, config_stat, log)
        # Most runs have no effects model, so the page is told rather than left to ask for a file that is not there.
        summary["has_effects"] = (here / "effects.json").is_file()
        found.append(summary)

    found.sort(key=lambda r: -r["modified"])
    return found


def real_path(path: Path) -> Path:
    """`path.resolve()`, which before Python 3.13 raises on a link that leads back to itself."""
    return Path(os.path.realpath(path))


def is_inside(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents
