"""Tests for `strands-evals ui`: what the scan finds, what the server admits, and what a save may overwrite."""

from __future__ import annotations

import json
import os
import re
import runpy
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import Message
from http import HTTPStatus
from http.client import HTTPConnection
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from strands_evals.cli import _ui_discovery, _ui_server
from strands_evals.cli._ui_discovery import (
    EventLog,
    classify,
    discover,
    discover_optimizer_runs,
    fingerprint,
    link_status,
    read_event_log,
)
from strands_evals.cli._ui_evaluators import check_evaluators, evaluator_catalog
from strands_evals.cli._ui_markdown import DEPTH_LIMIT
from strands_evals.cli._ui_server import (
    MAX_UPLOAD_BYTES,
    PAGE_ROOTS,
    UNAUTHORIZED_PAGE,
    PathOutsideRoot,
    UIHandler,
    build_server,
    load_asset,
    resolve_within,
    server_url,
    validate_experiment_payload,
    write_json_atomic,
)
from strands_evals.cli.commands import ui as ui_command
from strands_evals.cli.main import main
from strands_evals.experiment import _builtin_evaluator_types
from strands_evals.local_file_task_result_store import LocalFileTaskResultStore
from strands_evals.types.detector import DiagnosisResult, FailureItem, FailureOutput, RCAItem, RCAOutput
from strands_evals.types.evaluation import EvaluationData
from strands_evals.types.trace import InferenceSpan, Session, SpanInfo, Trace

NOT_RECOGNISED = "not an experiment, a report, a session or a diagnosis"
# More arrays inside one another than the JSON reader will follow.
TOO_DEEP = "[" * 100_000


def _reads_too_deep() -> bool:
    """Whether this Python reads `TOO_DEEP` to its end, where it is a document cut short, instead of giving up."""
    try:
        json.loads(TOO_DEEP)
    except RecursionError:
        return False
    except ValueError:
        return True
    return True


READS_TOO_DEEP = _reads_too_deep()
# Fewer than that, and more than a copy made level by level will follow.
DEEP = "[" * 700 + "1" + "]" * 700


def _deepest_read() -> str:
    """Arrays inside one another, as many as this Python reads with room to spare, up to 5,000.

    On every version that is more than `dataclasses.asdict` follows, which is what a scan was once copied with.
    """
    levels = 5000
    while True:
        nested = "[" * levels + "1" + "]" * levels
        try:
            json.loads(nested)
            return nested
        except RecursionError:
            levels //= 2


DEEPEST_READ = _deepest_read()
MISSING_RUBRIC = {
    "index": 0,
    "evaluator_type": "OutputEvaluator",
    "level": "error",
    "message": "needs a 'rubric' setting; OutputEvaluator cannot be constructed without it",
}

# ── documents ──


def _experiment(*names: str) -> dict:
    return {
        "cases": [{"name": name, "session_id": f"sid-{name}", "input": f"ask about {name}"} for name in names],
        "evaluators": [{"evaluator_type": "OutputEvaluator"}],
    }


def _report(*names: str, failing: tuple[str, ...] = (), declined: tuple[str, ...] = ()) -> dict:
    cases, scores, passes, detailed = [], [], [], []
    for name in names:
        cases.append({"name": name, "input": f"ask about {name}", "evaluator": "OutputEvaluator"})
        if name in declined:
            scores.append(0.0)
            passes.append(True)
            detailed.append([{"score": 0.0, "test_pass": True, "label": "not_applicable"}])
        elif name in failing:
            scores.append(0.0)
            passes.append(False)
            detailed.append([{"score": 0.0, "test_pass": False, "label": "no_match"}])
        else:
            scores.append(1.0)
            passes.append(True)
            detailed.append([{"score": 1.0, "test_pass": True, "label": "match"}])
    applicable = [s for n, s in zip(names, scores, strict=True) if n not in declined]
    return {
        "overall_score": sum(applicable) / len(applicable) if applicable else 0.0,
        "cases": cases,
        "scores": scores,
        "test_passes": passes,
        "detailed_results": detailed,
    }


def _deep_experiment(nested: str = DEEP) -> str:
    """An experiment file whose evaluator has a setting nested `DEEP`, or as `nested` is."""
    return json.dumps(_experiment("a")).replace('"OutputEvaluator"', f'"OutputEvaluator", "extra": {nested}')


def _session(session_id: str, *traces: list[tuple[float, float]]) -> dict:
    """A session as `fetch -o` writes it: a trace per argument, holding a span per (start, end) in seconds."""
    began = datetime(2026, 1, 5, 9, tzinfo=timezone.utc)

    def span(start: float, end: float) -> InferenceSpan:
        times = {"start_time": began + timedelta(seconds=start), "end_time": began + timedelta(seconds=end)}
        return InferenceSpan(span_info=SpanInfo(session_id=session_id, **times), messages=[])

    recorded = Session(
        session_id=session_id,
        traces=[
            Trace(trace_id=f"{session_id}-{number}", session_id=session_id, spans=[span(*times) for times in spans])
            for number, spans in enumerate(traces)
        ],
    )
    return json.loads(recorded.model_dump_json(indent=2))


_FAILURE = FailureItem(span_id="span-3", category=["wrong_tool"], confidence=[0.9], evidence=["looked it up twice"])
_CAUSE = RCAItem(
    failure_span_id="span-3",
    location="span-2",
    causality="PRIMARY_FAILURE",
    failure_detection_timing="ONLY_AT_TASK_END",
    completion_status="PARTIAL_SUCCESS",
    root_cause_explanation="no refund tool is described",
    fix_type="TOOL_DESCRIPTION_FIX",
    fix_recommendation="describe one",
)


def _diagnosis(session_id: str) -> dict:
    """A diagnosis as `diagnose -o` writes it."""
    return DiagnosisResult(session_id=session_id, failures=[_FAILURE], root_causes=[_CAUSE]).model_dump(mode="json")


def _write(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def _event_log(path: Path, *events: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path


def _optimizer_run(directory: Path, *events: dict, config: dict | None = None) -> Path:
    _write(directory / "run.json", config or {"run_id": "r-1", "objective": "Maximize(Helpfulness)"})
    _event_log(directory / "events.jsonl", *events)
    return directory


def _listed(root: Path, relative: str) -> dict:
    """The part of a file's entry that comes from the file itself."""
    info = (root / relative).stat()
    return {"path": relative, "name": relative.rsplit("/", 1)[-1], "size": info.st_size, "modified": info.st_mtime}


def _pairs(root: Path) -> set[tuple[str | None, str | None]]:
    """Each (experiment path, report path) discovery pairs, with `None` on the side that has no partner."""
    found = discover(root)
    made = {(report["experiment"], report["path"]) for report in found.reports}
    unrun = {(experiment["path"], None) for experiment in found.experiments if not experiment["reports"]}
    return made | unrun


def _found(root: Path) -> list[str]:
    """Every experiment and report path discovery lists."""
    found = discover(root)
    return sorted(entry["path"] for entry in found.experiments + found.reports)


def _cases(table: dict[str, tuple]) -> list:
    """Parametrize rows from a table keyed by test id."""
    return [pytest.param(*row, id=name) for name, row in table.items()]


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


# ── classification and scanning ──


@pytest.mark.parametrize(
    "obj,expected",
    _cases(
        {
            "experiment": ({"cases": [], "evaluators": []}, "experiment"),
            "report": ({"overall_score": 0.5, "cases": [], "scores": [], "test_passes": []}, "report"),
            "cases alone": ({"cases": []}, None),
            "score alone": ({"overall_score": 0.5, "cases": []}, None),
            "boolean score": ({"overall_score": True, "cases": [], "scores": [], "test_passes": []}, None),
            "unrelated object": ({"name": "something else"}, None),
            "list": ([], None),
            "string": ("a string", None),
            "a fetched session": (_session("s", [(0, 1)]), "session"),
            "a session with no traces": ({"session_id": "s", "traces": []}, "session"),
            "traces that hold no spans": ({"session_id": "s", "traces": [{"trace_id": "t"}]}, None),
            "traces with no session id": ({"traces": [{"spans": []}]}, None),
            "a diagnosis": (_diagnosis("s"), "diagnosis"),
            "detect-only": (FailureOutput(session_id="s", failures=[_FAILURE]).model_dump(mode="json"), "diagnosis"),
            "rca-only": (RCAOutput(root_causes=[_CAUSE]).model_dump(mode="json"), "diagnosis"),
            "a diagnosis that found nothing": (DiagnosisResult(session_id="s").model_dump(mode="json"), "diagnosis"),
            "failures that name no span": ({"session_id": "s", "failures": ["it broke"]}, None),
            "rca-only that found nothing": (RCAOutput().model_dump(mode="json"), "diagnosis"),
            "no root causes among other things": ({"root_causes": [], "notes": "none yet"}, None),
        }
    ),
)
def test_classify_goes_by_shape_not_filename(obj, expected):
    assert classify(obj) == expected


def test_scan_names_the_files_it_could_not_use(tmp_path: Path):
    _write(tmp_path / "experiment.json", _experiment("a"))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "binary.json").write_bytes(b"\xff\xfe")
    (tmp_path / "unrelated.json").write_text('{"hello": "world"}')
    (tmp_path / "deep.json").write_text(TOO_DEEP)

    result = discover(tmp_path)

    assert _found(tmp_path) == ["experiment.json"]
    reasons = {skip["path"]: skip["why"] for skip in result.skipped}
    # The tail of the invalid-JSON reason is the json module's wording.
    assert reasons.pop("broken.json").startswith("invalid JSON at line 1: ")
    assert reasons == {
        "binary.json": "could not read: UnicodeDecodeError",
        "unrelated.json": NOT_RECOGNISED,
        "deep.json": "invalid JSON at line 1: Expecting value" if READS_TOO_DEEP else "nested too deeply to read",
    }


def test_discover_describes_an_experiment_and_the_report_made_from_it(tmp_path: Path):
    _write(tmp_path / "nightly" / "experiment.json", _experiment("a", "b"))
    _write(tmp_path / "nightly" / "report.json", _report("a", "b"))

    found = discover(tmp_path)

    assert found.experiments == [
        {
            **_listed(tmp_path, "nightly/experiment.json"),
            "cases": 2,
            "evaluators": [{"evaluator_type": "OutputEvaluator"}],
            "evaluator_notes": [MISSING_RUBRIC],
            "dir": "nightly",
            "reports": ["nightly/report.json"],
        }
    ]
    assert found.reports == [
        {
            **_listed(tmp_path, "nightly/report.json"),
            "overall_score": 1.0,
            "rows": 2,
            "failing": 0,
            "not_applicable": 0,
            "evaluators": ["OutputEvaluator"],
            "dir": "nightly",
            "experiment": "nightly/experiment.json",
            "link": {"status": "verified", "detail": "all 2 report case(s) exist in this experiment"},
        }
    ]


def test_discover_lists_a_report_made_in_code_with_no_experiment(tmp_path: Path):
    _write(tmp_path / "nightly.report.json", _report("a"))

    found = discover(tmp_path)

    assert found.experiments == []
    assert [(report["path"], report["experiment"], report["link"]) for report in found.reports] == [
        ("nightly.report.json", None, None)
    ]


def test_discover_lists_newest_first_and_gives_an_experiment_every_report_made_from_it(tmp_path: Path):
    _write(tmp_path / "support.experiment.json", _experiment("a", "b"))
    for age, name in enumerate(("support.report.json", "second-try.json", "runs/third.json")):
        made = _write(tmp_path / name, _report("a", "b"))
        os.utime(made, (1_000 - age, 1_000 - age))

    found = discover(tmp_path)

    assert [report["path"] for report in found.reports] == ["support.report.json", "second-try.json", "runs/third.json"]
    assert {report["experiment"] for report in found.reports} == {"support.experiment.json"}
    assert found.experiments[0]["reports"] == ["support.report.json", "second-try.json", "runs/third.json"]


def test_report_summary_counts_declined_rows_apart_from_failing_ones(tmp_path: Path):
    report = _report("ok", "bad", "skipped", failing=("bad",), declined=("skipped",))
    # Declined without passing is a failure to judge, so it counts as failing.
    report["detailed_results"][1] = [{"score": 0.0, "test_pass": False, "label": "not_applicable"}]
    _write(tmp_path / "report.json", report)

    assert discover(tmp_path).reports[0] == {
        **_listed(tmp_path, "report.json"),
        "overall_score": 0.5,
        "rows": 3,
        "failing": 1,
        "not_applicable": 1,
        "evaluators": ["OutputEvaluator"],
        "dir": ".",
        "experiment": None,
        "link": None,
    }


def test_report_summary_counts_a_row_with_no_verdict_as_failing(tmp_path: Path):
    report = _report("ok", "cut-off")
    report["test_passes"] = [True]
    _write(tmp_path / "report.json", report)

    assert discover(tmp_path).reports[0]["failing"] == 1


def test_discover_lists_a_session_with_what_it_recorded(tmp_path: Path):
    # Two traces ten minutes apart, and one with nothing in it: the wait between them is not the session's time.
    _write(tmp_path / "recorded" / "any-name.json", _session("s-1", [(0, 1.5), (2, 4)], [(600, 600.25)], []))
    untimed = _session("s-2", [(0, 1)])
    untimed["traces"][0]["spans"][0]["span_info"].update(start_time="yesterday", end_time=None)
    _write(tmp_path / "untimed.json", untimed)

    found = discover(tmp_path)

    by_path = {session["path"]: session for session in found.sessions}
    assert by_path["recorded/any-name.json"] == {
        **_listed(tmp_path, "recorded/any-name.json"),
        "session_id": "s-1",
        "traces": 3,
        "spans": 3,
        "seconds": 4.25,
        "dir": "recorded",
        "diagnoses": [],
    }
    assert (by_path["untimed.json"]["spans"], by_path["untimed.json"]["seconds"]) == (1, None)
    assert (found.skipped, found.experiments, found.reports) == ([], [], [])


@pytest.mark.parametrize(
    "document,listed",
    _cases(
        {
            "a report whose results for a row are not a list": (
                {**_report("a"), "detailed_results": [5]},
                {"rows": 1, "failing": 0, "not_applicable": 0},
            ),
            "a report whose detailed results are not a list": (
                {**_report("a", failing=("a",)), "detailed_results": {"a": 1}},
                {"rows": 1, "failing": 1, "not_applicable": 0},
            ),
            "a session timed before any clock counts from": (
                {
                    "session_id": "s",
                    "traces": [{"spans": [{"span_info": {"start_time": "0001-01-01T00:00:00", "end_time": "x"}}]}],
                },
                {"spans": 1, "seconds": None},
            ),
        }
    ),
)
def test_a_file_of_a_known_shape_is_listed_whatever_its_values_are(tmp_path: Path, document, listed):
    _write(tmp_path / "odd.json", document)

    found = discover(tmp_path)

    (entry,) = found.reports + found.sessions
    assert {key: entry[key] for key in listed} == listed
    assert found.skipped == []


def test_discover_stops_at_the_depth_limit(tmp_path: Path):
    deep = tmp_path / "one" / "two" / "three"
    _write(deep / "experiment.json", _experiment("deep"))
    _optimizer_run(deep / "opt")

    shallow, full = discover(tmp_path, depth=2), discover(tmp_path, depth=4)

    assert (shallow.experiments, shallow.optimizer_runs) == ([], [])
    assert [experiment["path"] for experiment in full.experiments] == ["one/two/three/experiment.json"]
    assert [run["dir"] for run in full.optimizer_runs] == ["one/two/three/opt"]


def test_discover_skips_hidden_and_dependency_directories(tmp_path: Path):
    for directory in ("kept", ".git", "node_modules/pkg", "__pycache__"):
        _write(tmp_path / directory / "experiment.json", _experiment("a"))
        (tmp_path / directory / "broken.json").write_text("{not json")

    result = discover(tmp_path)

    assert [experiment["path"] for experiment in result.experiments] == ["kept/experiment.json"]
    assert [skip["path"] for skip in result.skipped] == ["kept/broken.json"]


def test_scan_stops_at_the_file_cap_and_says_so(tmp_path: Path, monkeypatch):
    for name in ("a.json", "b.json", "c.json", "sub/d.json"):
        _write(tmp_path / name, _experiment("a"))

    monkeypatch.setattr(_ui_discovery, "MAX_SCAN_FILES", 2)
    capped = discover(tmp_path)
    monkeypatch.setattr(_ui_discovery, "MAX_SCAN_FILES", 4)
    whole = discover(tmp_path)

    assert (capped.truncated, {experiment["path"] for experiment in capped.experiments}) == (True, {"a.json", "b.json"})
    assert (whole.truncated, len(whole.experiments)) == (False, 4)


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_scan_does_not_read_through_a_link_that_leaves_the_folder(tmp_path: Path):
    outside = _write(tmp_path / "outside" / "experiment.json", _experiment("secret"))
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.json").symlink_to(outside)

    found = discover(root)

    assert found.experiments == []
    assert found.skipped == [{"path": "link.json", "why": "a link to a file outside this folder"}]


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_scan_sets_aside_a_link_that_leads_back_to_itself(tmp_path: Path):
    _write(tmp_path / "experiment.json", _experiment("a"))
    (tmp_path / "loop.json").symlink_to(tmp_path / "loop.json")

    found = discover(tmp_path)

    assert [experiment["path"] for experiment in found.experiments] == ["experiment.json"]
    assert found.skipped == [{"path": "loop.json", "why": "could not read: OSError"}]


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
@pytest.mark.parametrize("linked", ["run.json", "events.jsonl"])
def test_an_optimizer_run_is_not_read_through_a_link_that_leaves_the_folder(tmp_path: Path, linked):
    outside = _optimizer_run(tmp_path / "outside", {"event": "propose"}, config={"run_id": "not-for-this-folder"})
    run = _optimizer_run(tmp_path / "root" / "tuning", {"event": "config"})
    (run / linked).unlink()
    (run / linked).symlink_to(outside / linked)

    found = discover(tmp_path / "root")

    assert found.optimizer_runs == []
    assert {"path": f"tuning/{linked}", "why": "a link to a file outside this folder"} in found.skipped


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a named pipe")
def test_scan_sets_aside_a_pipe_instead_of_waiting_on_it(tmp_path: Path):
    """Nothing writes to these pipes, so reading one would never return."""
    LocalFileTaskResultStore(tmp_path / "store").save("a", EvaluationData(input="q", actual_output="a"))
    _optimizer_run(tmp_path / "tuning", {"event": "config"})
    (tmp_path / "tuning" / "events.jsonl").unlink()
    for pipe in ("pipe.json", "store/b.json", "tuning/events.jsonl"):
        os.mkfifo(tmp_path / pipe)
    scanned: list = []

    worker = threading.Thread(target=lambda: scanned.append(discover(tmp_path)), daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert scanned, "the scan is still waiting on a pipe"
    assert scanned[0].optimizer_runs == []
    assert {skip["path"]: skip["why"] for skip in scanned[0].skipped if skip["path"] != "tuning/run.json"} == {
        "pipe.json": "not a regular file",
        "store/b.json": "not a regular file",
        "tuning/events.jsonl": "not a regular file",
    }


@pytest.mark.skipif(not hasattr(sys, "get_int_max_str_digits"), reason="this Python reads a number of any length")
def test_scan_sets_aside_a_file_holding_a_number_too_long_to_read(tmp_path: Path):
    _write(tmp_path / "experiment.json", _experiment("a"))
    (tmp_path / "huge.json").write_text('{"cases": [], "evaluators": [], "n": 1' + "0" * 5000 + "}")
    _write(tmp_path / "tuning" / "run.json", {"run_id": "r"})
    (tmp_path / "tuning" / "events.jsonl").write_text(
        '{"event": "trial", "tokens": 1' + "0" * 5000 + '}\n{"event": "stop"}\n'
    )

    found = discover(tmp_path)

    assert [experiment["path"] for experiment in found.experiments] == ["experiment.json"]
    assert found.skipped == [{"path": "huge.json", "why": "holds a number too long to read"}]
    assert [(run["events"], run["bad_lines"]) for run in found.optimizer_runs] == [
        (1, [{"line": 1, "why": "holds a number too long to read"}])
    ]


def test_fingerprint_takes_a_file_name_that_is_not_utf8(tmp_path: Path):
    try:
        (tmp_path / os.fsdecode(b"caf\xe9.json")).write_text("{}")
    except (OSError, UnicodeDecodeError):
        pytest.skip("this platform has no file name that is not UTF-8")

    assert len(fingerprint(tmp_path)) == 64


def test_scan_sets_aside_what_is_in_a_folder_it_can_list_but_not_enter(tmp_path: Path):
    _write(tmp_path / "experiment.json", _experiment("a"))
    _write(tmp_path / "locked" / "inside.json", _experiment("b"))
    _optimizer_run(tmp_path / "sealed")
    for name in ("locked", "sealed"):
        (tmp_path / name).chmod(0o444)
    try:
        if os.access(tmp_path / "locked" / "inside.json", os.R_OK):
            pytest.skip("this user, or this platform, reads a file whatever its folder allows")
        found = discover(tmp_path)
        mark = fingerprint(tmp_path)
    finally:
        for name in ("locked", "sealed"):
            (tmp_path / name).chmod(0o755)

    assert [experiment["path"] for experiment in found.experiments] == ["experiment.json"]
    assert found.optimizer_runs == []
    assert {skip["path"]: skip["why"] for skip in found.skipped} == {
        "locked/inside.json": "could not read: PermissionError",
        "sealed/run.json": "could not read: PermissionError",
        "sealed/events.jsonl": "could not read: PermissionError",
    }
    assert len(mark) == 64


def test_a_task_result_store_is_reported_and_its_files_are_neither_skipped_nor_counted(tmp_path: Path, monkeypatch):
    store = LocalFileTaskResultStore(tmp_path / "store")
    for name in ("a", "b", "c", "d", "e"):
        store.save(name, EvaluationData(input=f"ask {name}", actual_output="answer", name=name))
    _write(tmp_path / "store" / "notes.json", {"hello": "world"})
    _write(tmp_path / "experiment.json", _experiment("a"))
    _write(tmp_path / "report.json", _report("a"))
    # Opens the way a saved result does, and is not one.
    _write(tmp_path / "inputs" / "row.json", {"input": "q", "output": "a"})

    # Four files besides the five results: the cap is reached only if a result is counted.
    monkeypatch.setattr(_ui_discovery, "MAX_SCAN_FILES", 4)
    found = discover(tmp_path)

    assert found.stores == [{"dir": "store", "files": 5}]
    assert found.truncated is False
    assert {skip["path"]: skip["why"] for skip in found.skipped} == {
        "inputs/row.json": NOT_RECOGNISED,
        "store/notes.json": NOT_RECOGNISED,
    }
    assert _found(tmp_path) == ["experiment.json", "report.json"]

    monkeypatch.setattr(_ui_discovery, "MAX_SCAN_FILES", 3)
    assert discover(tmp_path).truncated is True


def test_a_store_counts_what_opens_like_a_result_and_a_report_in_it_is_still_listed(tmp_path: Path):
    store = LocalFileTaskResultStore(tmp_path / "cached")
    for name in ("a", "b"):
        store.save(name, EvaluationData(input=f"ask {name}", actual_output="answer", name=name))
    _write(tmp_path / "cached" / "c.report.json", _report("c"))
    # Read after the results, so only how it opens is looked at. The same file elsewhere is read in full.
    _write(tmp_path / "cached" / "z-notes.json", {"input": "opens like a result"})
    _write(tmp_path / "loose" / "row.json", {"input": "opens like a result"})

    found = discover(tmp_path)

    assert found.stores == [{"dir": "cached", "files": 3}]
    assert [report["path"] for report in found.reports] == ["cached/c.report.json"]
    assert found.skipped == [{"path": "loose/row.json", "why": NOT_RECOGNISED}]


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_a_link_out_of_the_folder_is_not_counted_as_a_task_result(tmp_path: Path):
    result = EvaluationData(input="q", actual_output="a")
    LocalFileTaskResultStore(tmp_path / "outside").save("secret", result)
    LocalFileTaskResultStore(tmp_path / "root" / "store").save("a", result)
    (tmp_path / "root" / "store" / "b.json").symlink_to(tmp_path / "outside" / "secret.json")

    found = discover(tmp_path / "root")

    assert found.stores == [{"dir": "store", "files": 1}]
    assert found.skipped == [{"path": "store/b.json", "why": "a link to a file outside this folder"}]


# ── pairing ──


@pytest.mark.parametrize(
    "experiments,reports,expected",
    _cases(
        {
            "default names": (
                ["experiment.json", "experiment.saved.json"],
                ["report.json"],
                {("experiment.json", "report.json"), ("experiment.saved.json", None)},
            ),
            "shared stem": (
                ["nightly.experiment.json", "smoke.experiment.json"],
                ["nightly.report.json"],
                {("nightly.experiment.json", "nightly.report.json"), ("smoke.experiment.json", None)},
            ),
            "token needs a separator": (
                ["myexperiment.json", "second.json"],
                ["my.report.json"],
                {("myexperiment.json", None), ("second.json", None), (None, "my.report.json")},
            ),
            "a lone pair needs no shared stem": (["cases.json"], ["results.json"], {("cases.json", "results.json")}),
            "two experiments get no fallback": (
                ["cases.json", "more.json"],
                ["results.json"],
                {("cases.json", None), ("more.json", None), (None, "results.json")},
            ),
            "two reports get none either": (
                ["unrelated.json"],
                ["first.json", "second.json"],
                {("unrelated.json", None), (None, "first.json"), (None, "second.json")},
            ),
            "elsewhere in the tree, by the cases it holds": (
                ["a/experiment.json"],
                ["b/report.json"],
                {("a/experiment.json", "b/report.json")},
            ),
            "a second report of one experiment": (
                ["support.experiment.json", "other/unrelated.json"],
                ["support.report.json", "support-v2.json"],
                {
                    ("support.experiment.json", "support.report.json"),
                    ("support.experiment.json", "support-v2.json"),
                    ("other/unrelated.json", None),
                },
            ),
            "the nearer of two that hold its cases": (
                ["near/experiment.json", "far/away/experiment.json"],
                ["near/runs/latest.json"],
                {("near/experiment.json", "near/runs/latest.json"), ("far/away/experiment.json", None)},
            ),
            "equally near is a tie, and a tie is not guessed": (
                ["left/experiment.json", "right/experiment.json"],
                ["latest.json"],
                {("left/experiment.json", None), ("right/experiment.json", None), (None, "latest.json")},
            ),
            "the one holding its cases, over the only one beside it": (
                ["unrelated.json", "other/checkout.experiment.json"],
                ["nightly.json"],
                {("other/checkout.experiment.json", "nightly.json"), ("unrelated.json", None)},
            ),
        }
    ),
)
def test_discover_pairs_a_report_with_the_experiment_it_was_made_from(tmp_path: Path, experiments, reports, expected):
    for name in experiments:
        _write(tmp_path / name, _experiment("b" if "unrelated" in name else "a"))
    for name in reports:
        _write(tmp_path / name, _report("a"))

    assert _pairs(tmp_path) == expected


def test_discover_does_not_guess_between_two_experiments_of_a_reports_name(tmp_path: Path):
    _write(tmp_path / "nightly.exp.json", _experiment("z"))
    _write(tmp_path / "nightly.experiment.json", _experiment("a"))
    _write(tmp_path / "nightly.report.json", _report("a"))

    assert discover(tmp_path).reports[0]["experiment"] == "nightly.experiment.json"

    _write(tmp_path / "nightly.exp.json", _experiment("a"))

    assert discover(tmp_path).reports[0]["experiment"] is None


def test_discover_says_which_experiments_tie_for_a_report(tmp_path: Path):
    _write(tmp_path / "left" / "experiment.json", _experiment("a"))
    _write(tmp_path / "right" / "experiment.json", _experiment("a"))
    _write(tmp_path / "latest.json", _report("a"))

    report = discover(tmp_path).reports[0]

    assert (report["experiment"], report["link"]) == (
        None,
        {
            "status": "tie",
            "detail": "2 experiments hold every case of this report and are equally near: "
            "left/experiment.json, right/experiment.json",
        },
    )


@pytest.mark.parametrize(
    "cases,rows,status",
    [(["a"], ["a", "b"], "partial"), ([None], [None], "unverified"), (["a"], ["z"], "mismatch")],
    ids=["some of its cases", "cases with no names", "none of its cases"],
)
def test_discover_pairs_the_only_report_with_the_only_experiment_beside_it_and_says_how_they_fit(
    tmp_path: Path, cases, rows, status
):
    _write(tmp_path / "cases.json", _experiment(*cases))
    _write(tmp_path / "results.json", _report(*rows))

    report = discover(tmp_path).reports[0]

    assert (report["experiment"], report["link"]["status"]) == ("cases.json", status)


def test_discover_pairs_the_reports_of_an_experiment_written_by_hand_by_their_case_names(tmp_path: Path):
    """Its cases have no session_id, so each run gives them one, which the row of a task that raised records."""
    by_hand = {"cases": [{"name": name, "input": "q"} for name in "abc"], "evaluators": []}
    _write(tmp_path / "hand.experiment.json", by_hand)
    for run in ("first", "second"):
        report = _report("a", "b", "c", failing=("b",))
        report["cases"][1]["session_id"] = f"given-in-the-{run}-run"
        _write(tmp_path / f"{run}.json", report)
    # Where the experiment does name its cases by session_id, a row is still told by the one it carries.
    _write(tmp_path / "ids" / "experiment.json", _experiment("z"))
    _write(tmp_path / "ids" / "other.json", {**_report("z"), "cases": [{"name": "z", "session_id": "another"}]})
    _write(tmp_path / "ids" / "third.json", _report("z"))

    paired = {r["path"]: (r["experiment"], r["link"] and r["link"]["status"]) for r in discover(tmp_path).reports}

    assert paired == {
        "first.json": ("hand.experiment.json", "verified"),
        "second.json": ("hand.experiment.json", "verified"),
        "ids/other.json": (None, None),
        "ids/third.json": ("ids/experiment.json", "verified"),
    }


def test_discover_leaves_a_report_unpaired_when_no_experiment_holds_all_of_its_cases(tmp_path: Path):
    _write(tmp_path / "suite" / "experiment.json", _experiment("a"))
    _write(tmp_path / "runs" / "latest.json", _report("a", "b"))

    assert _pairs(tmp_path) == {("suite/experiment.json", None), (None, "runs/latest.json")}


def test_discover_pairs_a_report_beside_an_experiment_of_the_same_name_even_when_the_cases_differ(tmp_path: Path):
    _write(tmp_path / "nightly.experiment.json", _experiment("a"))
    _write(tmp_path / "nightly.report.json", _report("z"))
    _write(tmp_path / "elsewhere" / "experiment.json", _experiment("z"))

    report = discover(tmp_path).reports[0]

    assert (report["experiment"], report["link"]["status"]) == ("nightly.experiment.json", "mismatch")


@pytest.mark.parametrize(
    "experiment,report,expected",
    [
        (_experiment("a", "b"), _report("a", "b"), ("verified", "all 2 report case(s) exist in this experiment")),
        (_experiment("a"), _report("a", "b"), ("partial", "1 of 2 report case(s) are not in this experiment")),
        (
            _experiment("a", "b"),
            _report("x", "y"),
            ("mismatch", "none of the 2 report case(s) appear in this experiment"),
        ),
        ({"cases": [{}]}, {"cases": [{}]}, ("unverified", "no case identifiers to compare")),
    ],
    ids=["verified", "partial", "mismatch", "unverified"],
)
def test_link_status_compares_case_identifiers(experiment, report, expected):
    status, detail = expected
    assert link_status(experiment, report) == {"status": status, "detail": detail}


def test_a_diagnosis_goes_to_the_nearest_session_with_its_id_or_stands_alone(tmp_path: Path):
    _write(tmp_path / "runs" / "checkout.json", _session("s-1", [(0, 1)]))
    _write(tmp_path / "copies" / "checkout.json", _session("s-1", [(0, 1)]))
    _write(tmp_path / "runs" / "found" / "checkout.diagnosis.json", _diagnosis("s-1"))
    _write(tmp_path / "runs" / "elsewhere.diagnosis.json", _diagnosis("s-9"))
    _write(tmp_path / "runs" / "causes.json", RCAOutput(root_causes=[_CAUSE]).model_dump(mode="json"))
    # Two sessions with the id, equally near: which one was diagnosed is not guessed.
    _write(tmp_path / "left" / "twin.json", _session("twin", [(0, 1)]))
    _write(tmp_path / "right" / "twin.json", _session("twin", [(0, 1)]))
    _write(tmp_path / "twin.diagnosis.json", _diagnosis("twin"))

    found = discover(tmp_path)

    assert {d["path"]: d["session"] for d in found.diagnoses} == {
        "runs/found/checkout.diagnosis.json": "runs/checkout.json",
        "runs/elsewhere.diagnosis.json": None,
        "runs/causes.json": None,
        "twin.diagnosis.json": None,
    }
    assert {s["path"]: s["diagnoses"] for s in found.sessions if s["diagnoses"]} == {
        "runs/checkout.json": ["runs/found/checkout.diagnosis.json"]
    }
    entry = next(d for d in found.diagnoses if d["path"] == "runs/causes.json")
    assert (entry["session_id"], entry["failures"], entry["root_causes"], entry["dir"]) == (None, 0, 1, "runs")
    assert found.skipped == []


# ── optimizer runs ──


PICKED_OBJECTIVE = {"mean_diff": 0.30, "ci_low": 0.18, "ci_high": 0.42, "n_cases": 8}
TESTED_OBJECTIVE = {"mean_diff": 0.14, "ci_low": 0.02, "ci_high": 0.26, "n_cases": 12}
PICKED = {"event": "picked", "report": {"champion_id": "c3", "baseline_id": "c0", "objective": PICKED_OBJECTIVE}}
TESTED = {"event": "test", "report": {"champion_id": "c3", "baseline_id": "c0", "objective": TESTED_OBJECTIVE}}


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            '{"event": "config"}\n\n{"event": "stop"}\n',
            EventLog(events=[{"event": "config"}, {"event": "stop"}]),
        ),
        ('{"event": "config"}\n{"event": "tri', EventLog(events=[{"event": "config"}])),
        (
            '{"event": "config"}\n{"event": "tri\n{"event": "stop"}\n',
            EventLog(
                events=[{"event": "config"}, {"event": "stop"}],
                bad_lines=[{"line": 2, "why": "invalid JSON: Unterminated string starting at"}],
            ),
        ),
        (
            '{"event": "config"}\n[1, 2, 3]\n',
            EventLog(events=[{"event": "config"}], bad_lines=[{"line": 2, "why": "not a JSON object"}]),
        ),
        (
            f'{{"event": "config"}}\n{TOO_DEEP}\n',
            EventLog(
                events=[{"event": "config"}],
                bad_lines=[
                    {
                        "line": 2,
                        "why": "invalid JSON: Expecting value" if READS_TOO_DEEP else "nested too deeply to read",
                    }
                ],
            ),
        ),
    ],
    ids=[
        "complete lines",
        "partial last line dropped",
        "broken middle line reported",
        "line that is not an object",
        "line nested too deeply",
    ],
)
def test_read_event_log_reads_one_event_per_line(tmp_path: Path, text, expected):
    path = tmp_path / "events.jsonl"
    path.write_text(text)

    assert read_event_log(path) == expected


def test_read_event_log_stops_at_the_line_cap_and_says_so(tmp_path: Path):
    events = [{"event": "trial", "n": n} for n in range(10)]

    log = read_event_log(_event_log(tmp_path / "events.jsonl", *events), max_lines=4)

    assert log == EventLog(events=events[:4], truncated=True)


def test_read_event_log_reports_a_file_it_cannot_read(tmp_path: Path):
    assert read_event_log(tmp_path / "missing.jsonl") == EventLog(
        bad_lines=[{"line": 0, "why": "could not read: FileNotFoundError"}]
    )


def test_discover_optimizer_runs_summarises_what_the_log_records(tmp_path: Path):
    config = {"run_id": "r-7", "name": "tuning", "objective": "Maximize(H)", "budget": {"max_rollouts": 20}}
    events = [
        {"event": "config"},
        {"event": "propose", "candidate_id": "c0", "iteration": 0},
        {"event": "trial", "iteration": 0, "record": {"status": "completed"}},
        {"event": "iteration_end", "iteration": 0, "record": {"iteration": 0}},
        {"event": "iteration_end", "iteration": 0, "record": {"iteration": 0}},
        {"event": "picked", "iteration": 1, "report": {"champion_id": "c0"}},
        {"event": "stop", "iteration": 1, "reason": "iterations"},
    ]
    run = _optimizer_run(tmp_path / "run-a", *events, config=config)
    log, record = (run / "events.jsonl").stat(), (run / "run.json").stat()

    assert discover_optimizer_runs(tmp_path) == [
        {
            "id": "run-a",
            "dir": "run-a",
            "run_id": "r-7",
            "name": "tuning",
            "objective": "Maximize(H)",
            "constraints": [],
            "is_pareto": False,
            "budget": {"max_rollouts": 20},
            "config": config,
            "events_path": "run-a/events.jsonl",
            "events": 7,
            "events_size": log.st_size,
            "modified": max(log.st_mtime, record.st_mtime),
            "iterations": 1,  # recorded twice, counted once
            "candidates": 1,
            "trials": 1,
            "champion_id": "c0",
            "result": None,
            "stop_reason": "iterations",
            "state": "unknown",  # no run.lock to tell
            "bad_lines": [],
            "truncated": False,
            "tuned_against": None,
            "has_effects": False,
        }
    ]


@pytest.mark.parametrize(
    "events,expected",
    _cases(
        {
            "held-out split outranks validation": ([PICKED, TESTED], {"objective": TESTED_OBJECTIVE, "held_out": True}),
            "whatever order the log has them in": ([TESTED, PICKED], {"objective": TESTED_OBJECTIVE, "held_out": True}),
            "validation split alone": ([PICKED], {"objective": PICKED_OBJECTIVE, "held_out": False}),
            "still searching": ([{"event": "config"}, {"event": "trial", "iteration": 0}], None),
            "champion without a comparison": ([{"event": "picked", "report": {"champion_id": "c0"}}], None),
        }
    ),
)
def test_discover_optimizer_runs_reads_the_result_off_the_end_phase(tmp_path: Path, events, expected):
    _optimizer_run(tmp_path / "run-a", *events)

    assert discover_optimizer_runs(tmp_path)[0]["result"] == expected


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"objective": "Pareto(Maximize(H), Minimize(tokens))"}, True),
        ({"objective": "WeightedSum(Maximize(H), Minimize(tokens))"}, False),
        ({"objective": "Maximize(H)", "is_pareto": True}, True),
    ],
    ids=["Pareto objective", "single objective", "flag in the record"],
)
def test_discover_optimizer_runs_knows_a_pareto_run(tmp_path: Path, config, expected):
    _optimizer_run(tmp_path / "run-a", config=config)

    assert discover_optimizer_runs(tmp_path)[0]["is_pareto"] is expected


@pytest.mark.parametrize(
    "files",
    [
        {"run.json": '{"run_id": "r-1"}'},
        {"events.jsonl": '{"event": "config"}\n'},
        {"run.json": "[1, 2, 3]", "events.jsonl": '{"event": "config"}\n'},
        {"run.json": "{not json", "events.jsonl": '{"event": "config"}\n'},
    ],
    ids=["no event log", "no run.json", "run.json is not an object", "run.json is not JSON"],
)
def test_discover_optimizer_runs_needs_a_config_object_beside_an_event_log(tmp_path: Path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text)

    assert discover_optimizer_runs(tmp_path) == []


def test_discover_says_what_an_event_log_with_no_run_json_lacks(tmp_path: Path):
    _event_log(tmp_path / "half" / "events.jsonl", {"event": "config"})
    (tmp_path / "half" / "notes.jsonl").write_text("{}\n")

    assert discover(tmp_path).skipped == [{"path": "half/events.jsonl", "why": "no run.json beside it"}]


def test_a_run_at_the_root_that_names_itself_nothing_is_called_this_folder(tmp_path: Path):
    _optimizer_run(tmp_path, {"event": "config"}, config={"objective": "Maximize(Helpfulness)"})
    _optimizer_run(tmp_path / "nested", {"event": "config"}, config={"objective": "Maximize(Helpfulness)"})

    assert {run["dir"]: run["run_id"] for run in discover_optimizer_runs(tmp_path)} == {
        ".": "this folder",
        "nested": "nested",
    }


def test_discover_optimizer_runs_counts_iterations_that_are_not_numbers(tmp_path: Path):
    ends = [{"event": "iteration_end", "iteration": value} for value in (0, 0, {"round": 1}, {"round": 1}, None)]
    _optimizer_run(tmp_path / "run-a", *ends)

    assert discover_optimizer_runs(tmp_path)[0]["iterations"] == 3


@pytest.mark.parametrize(
    ("events", "result", "state"),
    [
        # Killed or cancelled mid-search: no `stop`, nobody holds the lock.
        ([{"event": "config"}, {"event": "iteration_end", "iteration": 0}], False, "interrupted"),
        # Killed in its end phase, after the `stop` event.
        ([{"event": "config"}, {"event": "stop", "reason": "budget"}, {"event": "finalists"}], False, "interrupted"),
        ([{"event": "config"}, {"event": "proposer_error", "block": "text"}], False, "halted"),
        # The baseline couldn't be measured: the run is over, with no end phase.
        ([{"event": "config"}, {"event": "stop", "reason": "setup"}], False, "finished"),
        # Resumed after a proposer error, then killed again.
        ([{"event": "proposer_error"}, {"event": "resume"}, {"event": "iteration_end"}], False, "interrupted"),
        ([{"event": "config"}, {"event": "stop", "reason": "budget"}, {"event": "test"}], False, "finished"),
        # The test report skipped for the budget: no `test` event, but the end phase wrote its result.
        ([{"event": "config"}, {"event": "stop", "reason": "budget"}, {"event": "picked"}], True, "finished"),
    ],
)
def test_a_run_no_process_holds_reads_from_its_log(tmp_path: Path, events, result, state):
    from strands_evals.optimization.record.run_directory import _acquire, _release

    run = _optimizer_run(tmp_path / "run-a", *events)
    _release(_acquire(run))  # the run was locked once, as every run the optimizer writes is
    if result:
        _write(run / "result.json", {})

    assert discover_optimizer_runs(tmp_path)[0]["state"] == state


@pytest.mark.parametrize(
    ("events", "once_freed"),
    [
        ([{"event": "config"}], "interrupted"),
        # Resumed to finish its end phase.
        ([{"event": "stop", "reason": "budget"}, {"event": "test"}], "finished"),
        # Resumed after a proposer error, before it writes its `resume` event.
        ([{"event": "proposer_error"}], "halted"),
    ],
)
def test_a_run_whose_lock_is_held_is_running_whatever_its_log_says(tmp_path: Path, events, once_freed):
    from strands_evals.optimization.record.run_directory import _acquire, _release

    run = _optimizer_run(tmp_path / "run-a", *events)
    held = _acquire(run)
    try:
        assert discover_optimizer_runs(tmp_path)[0]["state"] == "running"
        # The probe took nothing: the writer still holds the lock.
        with pytest.raises(RuntimeError, match="open for writing"):
            _acquire(run)
    finally:
        _release(held)
    assert discover_optimizer_runs(tmp_path)[0]["state"] == once_freed


def test_a_run_with_no_lock_file_is_unknown_until_it_finishes(tmp_path: Path):
    _optimizer_run(tmp_path / "old", {"event": "config"})
    _optimizer_run(tmp_path / "done", {"event": "stop", "reason": "budget"}, {"event": "test"})
    (tmp_path / "linked").mkdir()
    _optimizer_run(tmp_path / "linked", {"event": "config"})
    (tmp_path / "elsewhere.lock").write_text("")
    (tmp_path / "linked" / "run.lock").symlink_to(tmp_path / "elsewhere.lock")

    found = {run["id"]: run["state"] for run in discover_optimizer_runs(tmp_path)}

    assert found == {"old": "unknown", "done": "finished", "linked": "unknown"}
    assert not (tmp_path / "old" / "run.lock").exists()  # the probe never makes one


def test_fingerprint_moves_when_a_runs_process_exits(tmp_path: Path):
    from strands_evals.optimization.record.run_directory import _acquire, _release

    run = _optimizer_run(tmp_path / "run-a", {"event": "config"})
    held = _acquire(run)
    while_running = fingerprint(tmp_path)
    _release(held)  # as the OS frees it when the process dies: no file changes

    assert fingerprint(tmp_path) != while_running


def test_discover_optimizer_runs_says_whether_an_effects_model_sits_beside_the_run(tmp_path: Path):
    _optimizer_run(tmp_path / "fitted", {"event": "config"})
    _optimizer_run(tmp_path / "plain", {"event": "config"})
    _write(tmp_path / "fitted" / "effects.json", {"effects": []})

    found = {run["id"]: run["has_effects"] for run in discover_optimizer_runs(tmp_path)}

    assert found == {"fitted": True, "plain": False}


def test_discover_keeps_a_recognised_run_json_out_of_skipped(tmp_path: Path):
    _optimizer_run(tmp_path / "opt")
    _write(tmp_path / "elsewhere" / "run.json", {"not": "an optimizer run"})

    result = discover(tmp_path)

    assert [run["dir"] for run in result.optimizer_runs] == ["opt"]
    assert result.skipped == [{"path": "elsewhere/run.json", "why": NOT_RECOGNISED}]


def test_discover_keeps_a_runs_own_storage_out_of_skipped(tmp_path: Path):
    run = _optimizer_run(tmp_path / "opt")
    for name in ("result.json", "effects.json", "best.values.json"):
        _write(run / name, {"whatever": 1})
    for directory in ("transcripts", "candidates", "front"):
        _write(run / directory / "c0.json", {"values": {}})
    _write(tmp_path / "candidates" / "c0.json", {"values": {}})

    result = discover(tmp_path)

    assert [run["dir"] for run in result.optimizer_runs] == ["opt"]
    # Only a run directory's own `candidates` is its storage.
    assert result.skipped == [{"path": "candidates/c0.json", "why": NOT_RECOGNISED}]


def _tuned(path: str, matched: int, split_cases: int, experiment_cases: int) -> dict:
    return {
        "path": path,
        "matched": matched,
        "split_cases": split_cases,
        "experiment_cases": experiment_cases,
        "status": "verified" if matched == split_cases else "partial",
    }


@pytest.mark.parametrize(
    "experiments,split,expected",
    _cases(
        {
            "names at split.train": (
                {"experiment.json": "abc"},
                {"train": ["a", "b"], "validation": ["c"], "seed": 7},
                _tuned("experiment.json", 3, 3, 3),
            ),
            "names under split.cases": (
                {"experiment.json": "abc"},
                {"train": 0.6, "test": 0.4, "cases": {"train": ["a", "b"], "test": ["c"]}},
                _tuned("experiment.json", 3, 3, 3),
            ),
            "session ids": (
                {"experiment.json": "ab"},
                {"train": ["sid-a", "sid-b"]},
                _tuned("experiment.json", 2, 2, 2),
            ),
            "partial match": (
                {"experiment.json": "ab"},
                {"train": ["a", "b"], "test": ["gone"]},
                _tuned("experiment.json", 2, 3, 2),
            ),
            "largest overlap wins": (
                {"few/experiment.json": "a", "most/experiment.json": "abc"},
                {"train": ["a", "b", "c"]},
                _tuned("most/experiment.json", 3, 3, 3),
            ),
            "of two that fit equally, the nearer": (
                {"a/far/experiment.json": "ab", "near/experiment.json": "ab"},
                {"train": ["a", "b"]},
                _tuned("near/experiment.json", 2, 2, 2),
            ),
            "a tie is not guessed": (
                {"b-copy/experiment.json": "ab", "a-copy/experiment.json": "ab"},
                {"train": ["a", "b", "gone"]},
                {
                    "path": None,
                    "matched": 2,
                    "split_cases": 3,
                    "status": "tie",
                    "detail": "2 experiments hold 2 of its 3 cases and are equally near: "
                    "a-copy/experiment.json, b-copy/experiment.json",
                },
            ),
            "nothing overlaps": ({"experiment.json": "xy"}, {"train": ["a", "b"]}, None),
            "no split recorded": ({"experiment.json": "a"}, None, None),
            "split is not a mapping": ({"experiment.json": "a"}, "all of them", None),
            "entries that are not case ids are dropped": (
                {"experiment.json": "a"},
                {"train": "a", "validation": [None, {"a": 1}, "a"]},
                _tuned("experiment.json", 1, 1, 1),
            ),
        }
    ),
)
def test_tuned_against_names_the_experiment_whose_cases_the_split_holds(tmp_path: Path, experiments, split, expected):
    for path, case_names in experiments.items():
        _write(tmp_path / path, _experiment(*case_names))
    _optimizer_run(tmp_path / "opt", config={"run_id": "r-1", "split": split})

    assert discover(tmp_path).optimizer_runs[0]["tuned_against"] == expected


def test_browser_fixture_run_is_readable(tmp_path: Path):
    """The run the browser tests chart, read by the real reader."""
    generator = Path(__file__).parents[2] / "ui" / "fixtures" / "make_optimizer_run.py"
    runpy.run_path(str(generator))["main"](tmp_path)

    log = read_event_log(tmp_path / "events.jsonl")

    assert (log.bad_lines, log.truncated) == ([], False)
    assert log.events[0]["event"] == "config"
    # The invalid patch never became a candidate, so it has a proposal but no `propose` event.
    assert sum(1 for e in log.events if e["event"] == "propose") == 10
    assert sum(len(e["record"]["proposals"]) for e in log.events if e["event"] == "iteration_end") == 11


# ── evaluators ──


def test_evaluator_notes_say_what_the_library_would_refuse_to_construct():
    notes = check_evaluators(
        [
            {"evaluator_type": "OutputEvaluator", "rubric": "Score it.", "model_id": "a-model"},
            {"evaluator_type": "Contains", "value": "refund", "threshold": 1, "colour": "red"},
            {"evaluator_type": "MyTeamsOwnEvaluator", "anything": 1},
            {"evaluator_type": 7},
            "not an evaluator",
            {"evaluator_type": "OutputEvaluator"},
        ]
    )

    assert notes == [
        {
            "index": 1,
            "evaluator_type": "Contains",
            "level": "error",
            "message": "'colour' is not a setting of Contains",
        },
        {
            "index": 1,
            "evaluator_type": "Contains",
            "level": "error",
            "message": "'threshold' is not a setting of Contains",
        },
        {
            "index": 2,
            "evaluator_type": "MyTeamsOwnEvaluator",
            "level": "warning",
            "message": "not a built-in evaluator; `run` and `validate` need it passed in as a custom evaluator",
        },
        {**MISSING_RUBRIC, "index": 5},
    ]


def test_evaluator_catalog_describes_each_built_in_from_the_class():
    catalog = evaluator_catalog()

    assert catalog["Contains"] == {
        "summary": "Checks if actual_output contains a substring.",
        "settings": [
            {"name": "value", "required": True, "default": None},
            {"name": "case_sensitive", "required": False, "default": "true"},
            {"name": "name", "required": False, "default": "null"},
        ],
    }
    assert set(catalog) == set(_builtin_evaluator_types())
    assert all(len(setting["default"] or "") <= 48 for entry in catalog.values() for setting in entry["settings"])


# ── paths ──


def test_resolve_within_allows_a_file_in_the_tree(tmp_path: Path):
    target = _write(tmp_path / "nested" / "experiment.json", _experiment("a"))

    assert resolve_within(tmp_path, "nested/experiment.json") == target.resolve()


@pytest.mark.parametrize(
    "attempt,message",
    [
        ("../outside.json", "../outside.json is outside {root}"),
        ("nested/../../outside.json", "nested/../../outside.json is outside {root}"),
        ("/etc/passwd", "/etc/passwd is outside {root}"),
        ("", "no path given"),
    ],
    ids=["parent", "parent through a child", "absolute", "empty"],
)
def test_resolve_within_rejects_escapes(tmp_path: Path, attempt: str, message: str):
    with pytest.raises(PathOutsideRoot) as refused:
        resolve_within(tmp_path, attempt)

    assert str(refused.value) == message.replace("{root}", str(tmp_path.resolve()))


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_resolve_within_rejects_a_symlink_out_of_the_tree(tmp_path: Path):
    secret = _write(tmp_path / "outside" / "secret.json", {"shh": True})
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.json").symlink_to(secret)

    with pytest.raises(PathOutsideRoot):
        resolve_within(root, "link.json")


# ── writes ──


def test_validate_experiment_payload_rejects_a_case_the_library_would_refuse():
    # The rest of the message is pydantic's.
    with pytest.raises(ValueError, match="^case 'no input' is not valid: "):
        validate_experiment_payload({"cases": [{"name": "no input"}], "evaluators": []})


def test_validate_experiment_payload_keeps_an_evaluator_it_cannot_resolve():
    payload = {
        "cases": [{"name": "a", "input": "hi"}],
        "evaluators": [{"evaluator_type": "MyTeamsOwnEvaluator", "threshold": 0.9}],
    }

    assert validate_experiment_payload(payload) is payload


@pytest.mark.parametrize(
    "payload,message",
    [
        ("not a document", "experiment must be a JSON object"),
        ({"evaluators": []}, "experiment is missing a 'cases' list"),
        ({"cases": []}, "experiment is missing an 'evaluators' list"),
        ({"cases": ["nope"], "evaluators": []}, "case 0 is not a JSON object"),
        ({"cases": [], "evaluators": [{"name": "x"}]}, "evaluator 0 has no 'evaluator_type'"),
    ],
    ids=["not an object", "no cases", "no evaluators", "case is not an object", "evaluator has no type"],
)
def test_validate_experiment_payload_rejects_a_broken_shape(payload, message):
    with pytest.raises(ValueError) as refused:
        validate_experiment_payload(payload)

    assert str(refused.value) == message


def test_write_json_atomic_leaves_the_old_file_when_serialization_fails(tmp_path: Path):
    target = _write(tmp_path / "experiment.json", _experiment("original"))
    before = target.read_text()

    with pytest.raises(ValueError):
        write_json_atomic(target, {"cases": [{"score": float("nan")}]})

    assert target.read_text() == before
    assert list(tmp_path.iterdir()) == [target]


def test_write_json_atomic_creates_a_file_that_is_not_there(tmp_path: Path):
    target = tmp_path / "experiment.json"

    modified = write_json_atomic(target, _experiment("a", "b"))

    assert json.loads(target.read_text()) == _experiment("a", "b")
    assert modified == target.stat().st_mtime
    assert list(tmp_path.iterdir()) == [target]


# ── the server ──

PAGE = "<html>page</html>"
HOST_REFUSAL = "this server only answers requests addressed to the host it is serving on"
TOKEN_REFUSAL = (
    "missing or wrong token: send it as the X-Evals-Token header or as ?t=<token>; /llms.txt says what is answered"
)
LENGTH_REFUSAL = "send the body with a Content-Length"
WRONG_TOKEN = {"X-Evals-Token": "wrong"}
NO_TOKEN = {"X-Evals-Token": None}
LONG_NAME = "a" * 300 + ".json"
# Stands in a PUT body for the mtime its target has when the request is sent.
LOADED = "the stamp the file was loaded with"
SAVE = {"document": _experiment("a"), "modified": LOADED}


@dataclass
class Reply:
    status: int
    body: Any = ""  # parsed when the response is JSON, text otherwise
    headers: Message = field(default_factory=Message, compare=False, repr=False)


def _refused(status: HTTPStatus, message: str) -> Reply:
    return Reply(status, {"error": message})


class Client:
    """Talks to one running server, with its token and a loopback `Host` unless a test says otherwise."""

    def __init__(self, server, root: Path):
        self.server = server
        self.root = root
        self.host, self.port = server.server_address[:2]
        self.token = server.config.token
        self.settings = {"root": str(root.resolve()), "depth": 3, "read_only": False}
        # What `/api/config` answers: the settings and the evaluator catalogue the page explains settings from.
        self.config = {**self.settings, "evaluators": evaluator_catalog()}
        # What the printed link leaves in a browser.
        self.cookie = f"strands_evals_ui_{self.port}={self.token}"

    def connect(self) -> HTTPConnection:
        return HTTPConnection(self.host, self.port, timeout=5)

    def send(self, method: str, path: str, *, body: Any = None, headers: dict | None = None, conn=None) -> Reply:
        """Send one request, on `conn` when a test needs to keep the connection.

        A `body` that is not bytes goes as JSON. A header set to None is left out and `headers` wins over
        the computed `Content-Length`, so a test can omit the token or lie about the body.
        """
        payload = body if isinstance(body, bytes | None) else json.dumps(body).encode()
        sent: dict[str, str | None] = {
            "Host": f"127.0.0.1:{self.port}",
            "X-Evals-Token": self.token,
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",  # what a browser asks a page for
        }
        if payload is not None:
            sent["Content-Length"] = str(len(payload))
        sent.update(headers or {})

        connection = conn or self.connect()
        try:
            connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            for name, value in sent.items():
                if value is not None:
                    connection.putheader(name, value)
            connection.endheaders(payload)
            response = connection.getresponse()
            text = response.read().decode("utf-8")
        finally:
            if conn is None:
                connection.close()
        is_json = text and response.headers.get_content_type() == "application/json"
        return Reply(response.status, json.loads(text) if is_json else text, response.headers)

    def put(self, path: str, body: Any, **how: Any) -> Reply:
        """PUT to an experiment path, with `LOADED` in the body replaced by the target's current mtime."""
        if isinstance(body, dict) and body.get("modified") == LOADED:
            body = {**body, "modified": (self.root / path).stat().st_mtime}
        return self.send("PUT", f"/api/experiment?path={path}", body=body, **how)


@pytest.fixture
def serve(tmp_path: Path):
    """Start servers over a tree holding one paired run; `asset=None` serves the shipped page."""
    _write(tmp_path / "experiment.json", _experiment("alpha", "beta"))
    _write(tmp_path / "report.json", _report("alpha", "beta", failing=("beta",)))
    started = []

    def start(**options) -> Client:
        server = build_server(tmp_path, **{"asset": PAGE, **options})
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        started.append((server, thread))
        return Client(server, tmp_path)

    yield start
    for server, thread in started:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def client(serve) -> Client:
    return serve()


# ── the server: who is admitted ──


@pytest.mark.parametrize(
    "path,token",
    [
        ("/api/scan", "guessed"),
        ("/api/scan?t=guessed", None),
        ("/api/scan?t=%C3%A9", None),
        ("/api/scan", None),
        ("/api/changes", None),
    ],
    ids=["wrong header", "wrong query", "not ASCII", "none", "none, asking what changed"],
)
def test_api_refuses_a_request_without_the_token(client: Client, path, token):
    assert client.send("GET", path, headers={"X-Evals-Token": token}) == _refused(
        HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL
    )


def test_api_takes_the_token_from_the_query_too(client: Client):
    reply = client.send("GET", f"/api/config?t={client.token}", headers=NO_TOKEN)

    assert reply == Reply(HTTPStatus.OK, client.config)


@pytest.mark.parametrize(
    "bound,host,admitted",
    [
        ("127.0.0.1", "localhost", True),
        ("127.0.0.1", "LOCALHOST:8080", True),
        ("127.0.0.1", "[::1]:9", True),
        ("127.0.0.1", "[::1].evil.example", False),
        ("127.0.0.1", "127.0.0.1.evil.example", False),
        ("127.0.0.1", "localhost:notaport", False),
        ("127.0.0.1", "", False),
        ("127.0.0.1", "evals.evil.example.com", False),
        ("127.0.0.1", "192.168.1.5:8000", False),
        ("0.0.0.0", "192.168.1.5:8000", True),
        ("0.0.0.0", "[fe80::1]:8000", True),
        ("0.0.0.0", "evals.evil.example.com", False),
        ("192.168.1.5", "192.168.1.5:8000", True),
        ("192.168.1.5", "192.168.1.6:8000", False),
        ("192.168.1.5", "localhost:8000", True),
        ("evals.internal", "192.168.1.5:8000", False),
    ],
)
def test_host_header_must_name_loopback_or_the_bound_address(client: Client, bound, host, admitted):
    """An address cannot be rebound by an attacker's DNS, a name can."""
    client.server.config.bound_host = bound

    reply = client.send("GET", "/api/config", headers={"Host": host})

    assert reply == (Reply(HTTPStatus.OK, client.config) if admitted else _refused(HTTPStatus.FORBIDDEN, HOST_REFUSAL))


def test_an_ipv6_loopback_address_can_be_bound(serve):
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.bind(("::1", 0))
    except OSError:
        pytest.skip("this machine has no IPv6 loopback")

    client = serve(host="::1", token="chosen")

    assert server_url(client.server) == f"http://[::1]:{client.port}/?t=chosen"
    assert client.send("GET", "/api/config", headers={"Host": f"[::1]:{client.port}"}).status == HTTPStatus.OK


def test_server_url_carries_a_supplied_token(serve):
    client = serve(token="chosen-token")

    assert server_url(client.server) == f"http://127.0.0.1:{client.port}/?t=chosen-token"


def test_a_generated_token_is_long_and_differs_between_servers(serve):
    first, second = serve(), serve()

    assert first.token != second.token
    assert len(first.token) >= 32


# ── the server: the page ──


def test_page_needs_the_token(client: Client):
    assert client.send("GET", "/", headers=NO_TOKEN) == Reply(HTTPStatus.UNAUTHORIZED, UNAUTHORIZED_PAGE.decode())
    assert client.send("GET", "/") == Reply(HTTPStatus.OK, PAGE)


def test_a_refusal_says_how_the_token_is_sent_and_nothing_of_the_folder(client: Client):
    page = client.send("GET", "/reports/report.json", headers=NO_TOKEN)
    data = client.send("GET", "/reports/report.json.md", headers=WRONG_TOKEN)

    assert data == _refused(HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)
    for body in (page.body, data.body["error"]):
        assert "X-Evals-Token" in body and "/llms.txt" in body
        assert client.token not in body and str(client.root) not in body and "report.json" not in body


@pytest.mark.parametrize(
    "path,told",
    [
        ("/reports", "/reports is a page"),
        ("/reports/report.md", "Did you mean /reports/report.json.md?"),
        (
            "/experiments/experiment",
            "Did you mean /experiments/experiment.json.md or /experiments/experiment.json.json?",
        ),
        ("/reports/elsewhere.md?r=report.json", "/reports/elsewhere.md is a page"),
    ],
    ids=["a list", "a form with the file's .json left out", "a file with its .json left out", "an .md with no file"],
)
@pytest.mark.parametrize("accept", ["*/*", "application/json", None], ids=["anything", "json", "no header"])
def test_a_page_address_tells_a_reader_with_no_browser_where_the_data_is(client: Client, path, told, accept):
    reply = client.send("GET", path, headers={"Accept": accept})

    assert (reply.status, reply.headers.get_content_type()) == (HTTPStatus.NOT_ACCEPTABLE, "text/plain")
    assert told in reply.body and "/llms.txt" in reply.body and ".json.md" in reply.body
    assert len(reply.body) < 1000


def test_a_reader_with_no_browser_is_refused_like_any_other_without_the_token(client: Client):
    plain = {"Accept": "*/*"}

    assert client.send("GET", "/reports", headers={**NO_TOKEN, **plain}).status == HTTPStatus.UNAUTHORIZED
    assert client.send("GET", "/reports/report.md", headers={**WRONG_TOKEN, **plain}).status == HTTPStatus.UNAUTHORIZED
    with_link = client.send("GET", f"/reports?t={client.token}", headers={**NO_TOKEN, **plain})
    assert (with_link.status, with_link.headers["Set-Cookie"]) == (HTTPStatus.NOT_ACCEPTABLE, None)


def test_a_link_with_the_wrong_token_gets_no_cookie(client: Client):
    reply = client.send("GET", "/reports?t=guessed", headers=NO_TOKEN)

    assert reply == Reply(HTTPStatus.UNAUTHORIZED, UNAUTHORIZED_PAGE.decode())
    assert reply.headers["Set-Cookie"] is None


def test_the_page_is_not_served_to_another_host_even_with_the_token(client: Client):
    reply = client.send("GET", "/reports", headers={"Host": "evals.evil.example.com", "Cookie": client.cookie})

    assert reply == _refused(HTTPStatus.FORBIDDEN, HOST_REFUSAL)


def test_the_printed_link_trades_its_token_for_a_cookie_and_a_clean_address(client: Client):
    redirect = client.send("GET", f"/?t={client.token}", headers=NO_TOKEN)
    page = client.send("GET", "/", headers={**NO_TOKEN, "Cookie": client.cookie})

    assert redirect == Reply(HTTPStatus.SEE_OTHER)
    assert redirect.headers["Location"] == "/"
    assert redirect.headers["Set-Cookie"] == f"{client.cookie}; Path=/; HttpOnly; SameSite=Lax"
    assert page == Reply(HTTPStatus.OK, PAGE)


@pytest.mark.parametrize(
    "cookie",
    ["strands_evals_ui_{port}=wrong", "strands_evals_ui_1={token}", 'strands_evals_ui_{port}="\\012"'],
    ids=["wrong value", "another port's cookie", "malformed"],
)
def test_the_page_is_refused_without_this_servers_cookie(client: Client, cookie):
    sent = cookie.format(port=client.port, token=client.token)

    reply = client.send("GET", "/", headers={**NO_TOKEN, "Cookie": sent})

    assert reply == Reply(HTTPStatus.UNAUTHORIZED, UNAUTHORIZED_PAGE.decode())


@pytest.mark.parametrize(
    "path",
    [
        "/reports",
        "/reports/support/support.report.json",
        "/experiments/support/support.experiment.json/cases/a%20case",
        "/sessions/recorded/checkout.session.json",
        "/optimization/tuning/run-1/candidates/c8/report",
    ],
)
def test_every_address_of_the_page_serves_it_and_survives_the_token_redirect(client: Client, path):
    redirect = client.send("GET", f"{path}?t={client.token}", headers=NO_TOKEN)

    assert redirect.status == HTTPStatus.SEE_OTHER
    assert redirect.headers["Location"] == path
    assert client.send("GET", path, headers={**NO_TOKEN, "Cookie": client.cookie}) == Reply(HTTPStatus.OK, PAGE)
    assert client.send("GET", path, headers=NO_TOKEN).status == HTTPStatus.UNAUTHORIZED


def test_the_token_redirect_keeps_the_rest_of_the_query(client: Client):
    query = "r=a%2Fone.report.json&r=b%2Ftwo.report.json"

    first = client.send("GET", f"/reports?t={client.token}&{query}", headers=NO_TOKEN)
    last = client.send("GET", f"/reports?{query}&t={client.token}", headers=NO_TOKEN)

    assert first.headers["Location"] == last.headers["Location"] == f"/reports?{query}"


def test_the_token_redirect_drops_a_token_whose_name_is_encoded(client: Client):
    redirect = client.send("GET", f"/reports?%74={client.token}&r=one.json", headers=NO_TOKEN)

    assert redirect.headers["Location"] == "/reports?r=one.json"


@pytest.mark.parametrize("path", ["/reportsheet", "/runs", "/optimizations", "/favicon.ico", "/api"])
def test_an_address_that_is_not_the_pages_is_not_found(client: Client, path):
    assert client.send("GET", path) == _refused(HTTPStatus.NOT_FOUND, f"no route for {path}")


@pytest.mark.parametrize("path", ["/nope", "/reportsheet", "/runs/old-link", "/favicon.ico"])
def test_a_browser_at_an_address_that_is_not_the_pages_gets_the_page_to_say_so(client: Client, path):
    held = {**NO_TOKEN, "Cookie": client.cookie}
    link = f"{path}?t={client.token}"

    assert client.send("GET", path, headers=held) == Reply(HTTPStatus.NOT_FOUND, PAGE)
    # The token in such a link is not traded for a cookie: the redirect would echo a path the server does not know.
    assert client.send("GET", link, headers=held) == Reply(HTTPStatus.NOT_FOUND, PAGE)
    assert client.send("GET", link, headers=NO_TOKEN) == _refused(HTTPStatus.NOT_FOUND, f"no route for {path}")


@pytest.mark.parametrize(
    ("path", "headers", "refusal"),
    [
        ("/nope", {}, (HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)),
        ("/nope?t=guessed", {}, (HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)),
        ("/nope", {"Cookie": "strands_evals_ui_{port}=wrong"}, (HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)),
        ("/nope", {"Cookie": "{cookie}", "Host": "evals.evil.example.com"}, (HTTPStatus.FORBIDDEN, HOST_REFUSAL)),
        ("/api", {"Cookie": "{cookie}"}, (HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)),
        ("/api/nothing", {"Cookie": "{cookie}"}, (HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)),
        ("/reports/gone.json.md", {"Cookie": "{cookie}"}, (HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)),
    ],
    ids=["no cookie", "wrong token", "wrong cookie", "another host", "the api root", "an api route", "a data form"],
)
def test_the_not_found_page_is_for_a_browser_this_server_opened_and_never_for_the_api(
    client: Client, path, headers, refusal
):
    sent = {name: value.format(port=client.port, cookie=client.cookie) for name, value in headers.items()}

    assert client.send("GET", path, headers={**NO_TOKEN, **sent}) == _refused(*refusal)


def test_the_token_redirect_never_points_off_the_server(client: Client):
    redirect = client.send("GET", f"//elsewhere.example/reports?t={client.token}", headers=NO_TOKEN)

    # Refused outright where the standard library collapses the leading slashes, and a local path where it does not.
    assert redirect.headers.get("Location") in (None, "/reports")


def test_the_token_redirect_does_not_echo_a_path_that_is_not_printable(client: Client):
    asked = f"GET /reports/\u00e9?t={client.token} HTTP/1.1\r\nHost: 127.0.0.1\r\nAccept: text/html\r\n"
    request = f"{asked}Connection: close\r\n\r\n"

    with socket.create_connection((client.host, client.port), timeout=5) as raw:
        raw.sendall(request.encode("latin-1"))
        head = raw.makefile("rb").read().decode("latin-1")

    assert "\r\nLocation: /\r\n" in head


def test_the_cookie_opens_the_page_but_not_the_api(client: Client):
    reply = client.send("GET", "/api/config", headers={**NO_TOKEN, "Cookie": client.cookie})

    assert reply == _refused(HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)


def test_the_shipped_page_is_served_under_a_policy_that_runs_only_its_own_scripts(serve):
    client = serve(asset=None)
    nonce = client.server.config.nonce

    reply = client.send("GET", "/")

    assert reply.headers["Content-Security-Policy"] == (
        f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
        "style-src-attr 'unsafe-inline'; img-src data:; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    assert reply.headers["Referrer-Policy"] == "no-referrer"
    assert f'<script nonce="{nonce}">' in reply.body
    assert "<script>" not in reply.body
    assert f'<meta name="strands-evals-token" content="{client.token}">' in reply.body


@pytest.mark.parametrize(
    "asset,expected",
    [(None, "<html>edited</html>"), (PAGE, PAGE)],
    ids=["the shipped page is", "a pinned page is not"],
)
def test_the_page_is_re_read_per_request(serve, monkeypatch, asset, expected):
    client = serve(asset=asset)
    monkeypatch.setattr(_ui_server, "load_asset", lambda nonce="": "<html>edited</html>")

    assert client.send("GET", "/") == Reply(HTTPStatus.OK, expected)


def test_an_unreadable_page_falls_back_to_the_one_read_at_startup(serve, monkeypatch, caplog):
    client = serve(asset=None)
    at_startup = client.send("GET", "/")

    def vanished(nonce: str = "") -> str:
        raise OSError("vanished mid-save")

    monkeypatch.setattr(_ui_server, "load_asset", vanished)
    with caplog.at_level("WARNING", logger=_ui_server.logger.name):
        reply = client.send("GET", "/")

    assert reply == at_startup
    assert [record.getMessage() for record in caplog.records if record.name == _ui_server.logger.name] == [
        "error=<vanished mid-save> | could not re-read the ui page, serving the copy read at startup"
    ]


def _assets():
    return resources.files("strands_evals.cli").joinpath("assets")


def test_load_asset_inlines_every_asset_under_the_nonce():
    source = _assets().joinpath("ui.html").read_text(encoding="utf-8")
    scripts = source.count("<script src=")

    page = load_asset("n0nce")

    # Nothing is left for the browser to fetch: a second request would not carry the token.
    assert (page.count("<script src="), page.count('rel="stylesheet"')) == (0, 0)
    assert (page.count("<script>"), page.count("<style>")) == (0, 0)
    assert (page.count('<script nonce="n0nce">'), page.count('<style nonce="n0nce">')) == (scripts, 1)
    assert scripts > 0
    for name in ("ui.css", "ui-core.js", "ui-boot.js"):
        assert _assets().joinpath(name).read_text(encoding="utf-8") in page


def test_no_asset_can_close_the_tag_it_is_inlined_into():
    for asset in _assets().iterdir():
        if asset.name.endswith((".js", ".css")):
            assert "</script" not in asset.read_text(encoding="utf-8").lower(), asset.name


def test_a_reference_load_asset_cannot_inline_is_an_error(tmp_path: Path, monkeypatch):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "ui.html").write_text('<script defer src="ui-core.js"></script>')
    monkeypatch.setattr(_ui_server.resources, "files", lambda package: tmp_path)

    with pytest.raises(ValueError) as refused:
        load_asset()

    assert str(refused.value) == (
        'ui.html may only reference assets as <link rel="stylesheet" href="NAME"> or <script src="NAME"></script>'
    )


# ── noticing that the tree changed ──


@pytest.mark.parametrize(
    "change,noticed",
    _cases(
        {
            "a report written again": (lambda root: _write(root / "report.json", _report("alpha")), True),
            "the same bytes at a later time": (lambda root: os.utime(root / "report.json", ns=(1, 1)), True),
            "a file added": (lambda root: _write(root / "new" / "experiment.json", _experiment("a")), True),
            "a file removed": (lambda root: (root / "report.json").unlink(), True),
            "an event appended to a run's log": (
                lambda root: (root / "opt" / "events.jsonl").write_text('{"event": "config"}\n{"event": "stop"}\n'),
                True,
            ),
            "an effects model fitted beside a run": (lambda root: _write(root / "opt" / "effects.json", {}), True),
            "a file the page never reads": (lambda root: (root / "notes.txt").write_text("x"), False),
            "a dependency folder": (lambda root: _write(root / "node_modules" / "p" / "report.json", {}), False),
            "below the depth limit": (lambda root: _write(root / "a" / "b" / "c" / "d" / "report.json", {}), False),
        }
    ),
)
def test_fingerprint_moves_when_something_the_page_shows_does(tmp_path: Path, change, noticed):
    _write(tmp_path / "report.json", _report("alpha", "beta"))
    _optimizer_run(tmp_path / "opt", {"event": "config"})
    before = fingerprint(tmp_path)

    change(tmp_path)

    assert (fingerprint(tmp_path) != before) is noticed


def test_changes_endpoint_answers_the_fingerprint_a_scan_was_taken_at(client: Client):
    scanned = client.send("GET", "/api/scan").body["fingerprint"]
    assert client.send("GET", "/api/changes") == Reply(HTTPStatus.OK, {"fingerprint": scanned})

    _write(client.root / "report.json", _report("alpha"))

    assert client.send("GET", "/api/changes").body["fingerprint"] != scanned


# ── the server: reading ──


def test_scan_endpoint_serves_the_discovery_with_the_settings(client: Client):
    _optimizer_run(client.root / "opt", {"event": "config"}, {"event": "stop", "reason": "budget"})
    _write(client.root / "recorded" / "checkout.json", _session("s-1", [(0, 1)]))
    _write(client.root / "recorded" / "checkout.diagnosis.json", _diagnosis("s-1"))
    LocalFileTaskResultStore(client.root / "store").save("a", EvaluationData(input="q", actual_output="a"))
    (client.root / "broken.json").write_text("{not json")

    reply = client.send("GET", "/api/scan")

    found = asdict(discover(client.root))
    assert {name: len(listed) for name, listed in found.items() if name != "truncated"} == dict.fromkeys(
        ("experiments", "reports", "skipped", "optimizer_runs", "sessions", "diagnoses", "stores"), 1
    )
    assert reply == Reply(
        HTTPStatus.OK, {"schema_version": 1, **client.settings, "fingerprint": fingerprint(client.root), **found}
    )


def test_scan_endpoint_survives_a_report_holding_nan(client: Client):
    (client.root / "odd").mkdir()
    (client.root / "odd" / "report.json").write_text(json.dumps({**_report("alpha"), "overall_score": float("nan")}))

    reply = client.send("GET", "/api/scan")

    assert reply.status == HTTPStatus.OK
    assert {report["path"]: report["overall_score"] for report in reply.body["reports"]} == {
        "report.json": 0.5,
        "odd/report.json": None,
    }


def test_scan_endpoint_lists_a_file_nested_deeper_than_a_copy_follows(client: Client):
    (client.root / "deep.json").write_text(_deep_experiment(DEEPEST_READ))
    (client.root / "odd.json").write_text(json.dumps({**_report("alpha"), "overall_score": float("nan")}))

    reply = client.send("GET", "/api/scan")

    assert reply.status == HTTPStatus.OK
    assert [experiment["path"] for experiment in reply.body["experiments"]] == ["deep.json", "experiment.json"]


@pytest.mark.parametrize("method", ["GET", "PUT", "POST"])
def test_a_request_for_what_is_not_an_address_is_refused_and_the_server_carries_on(client: Client, method):
    reply = client.send(method, "http://[/", body=b"{}" if method != "GET" else None)

    assert (reply.status, reply.body) == (HTTPStatus.BAD_REQUEST, {"error": "the request target is not an address"})
    assert client.send("GET", "/api/config").status == HTTPStatus.OK


@pytest.mark.parametrize(
    "method,path,broken",
    [("GET", "/api/scan", "discover"), ("PUT", "/api/experiment?path=experiment.json", "validate_experiment_payload")],
)
def test_a_route_that_raises_answers_with_an_error_and_the_server_carries_on(
    client: Client, monkeypatch, method, path, broken
):
    def fail(*args: Any) -> None:
        raise RuntimeError(f"could not read {client.root.parent}/elsewhere")

    monkeypatch.setattr(_ui_server, broken, fail)

    reply = client.send(method, path, body=SAVE if method == "PUT" else None)

    assert reply == _refused(HTTPStatus.INTERNAL_SERVER_ERROR, "the server could not answer this request: RuntimeError")
    assert client.send("GET", "/api/config") == Reply(HTTPStatus.OK, client.config)


@pytest.mark.parametrize("name,notes", [("experiment.json", [MISSING_RUBRIC]), ("report.json", [])])
def test_file_endpoint_returns_the_document_and_its_stamp(client: Client, name, notes):
    target = client.root / name

    assert client.send("GET", f"/api/file?path={name}") == Reply(
        HTTPStatus.OK,
        {
            "path": name,
            "modified": target.stat().st_mtime,
            "document": json.loads(target.read_text()),
            "evaluator_notes": notes,
        },
    )


def test_events_endpoint_returns_the_log(client: Client):
    events = [{"event": "config"}, {"event": "trial", "record": {"status": "completed"}}]
    run = _optimizer_run(client.root / "opt", *events)

    assert client.send("GET", "/api/events?path=opt/events.jsonl") == Reply(
        HTTPStatus.OK,
        {
            "path": "opt/events.jsonl",
            "modified": (run / "events.jsonl").stat().st_mtime,
            "events": events,
            "bad_lines": [],
            "truncated": False,
        },
    )


@pytest.mark.parametrize("route", ["/api/file", "/api/events"])
@pytest.mark.parametrize(
    "query,status,message",
    [
        ("?path=../../etc/passwd", HTTPStatus.FORBIDDEN, "../../etc/passwd is outside {root}"),
        ("?path=%00", HTTPStatus.FORBIDDEN, "'\\x00' is not a path"),
        ("", HTTPStatus.FORBIDDEN, "no path given"),
        ("?path=nowhere/events.jsonl", HTTPStatus.NOT_FOUND, "nowhere/events.jsonl is not a file"),
        (f"?path={LONG_NAME}", HTTPStatus.NOT_FOUND, f"{LONG_NAME} is not a file"),
        ("?path=loop.json", HTTPStatus.NOT_FOUND, "loop.json is not a file"),
    ],
    ids=["outside the root", "null byte", "no path", "missing file", "a name too long to be one", "a link to itself"],
)
def test_reading_refuses_a_path_it_cannot_serve(client: Client, route, query, status, message):
    (client.root / "loop.json").symlink_to(client.root / "loop.json")

    assert client.send("GET", route + query) == _refused(status, message.replace("{root}", client.settings["root"]))


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_an_unknown_route_is_not_found(client: Client, method):
    reply = client.send(method, "/api/nothing")

    assert reply.status == HTTPStatus.NOT_FOUND
    assert reply.body == ({"error": "no route for /api/nothing"} if method == "GET" else "")


# ── the server: a file as Markdown and as JSON, and what the server says of itself ──

REPORT_MARKDOWN = """# Report: report.json

- Overall score: 0.50
- Rows: 2 (1 passed, 1 failed, 0 not applicable)
- Cases: 2
- Evaluators: OutputEvaluator
- Experiment: experiment.json (verified: all 2 report case(s) exist in this experiment)

## Failing rows

### beta · OutputEvaluator

- Score: 0.00
- Verdict: failed
- Label: no_match

Input:

```text
ask about beta
```

## All rows

| Case | Evaluator | Score | Verdict |
| --- | --- | --- | --- |
| alpha | OutputEvaluator | 1.00 | passed |
| beta | OutputEvaluator | 0.00 | failed |
"""
EXPERIMENT_MARKDOWN = """# Experiment: experiment.json

- Cases: 2
- Evaluators: 1
- Reports: report.json

## Evaluators

### OutputEvaluator

## Cases

### alpha

Input:

```text
ask about alpha
```

### beta

Input:

```text
ask about beta
```
"""
DATA_ADDRESSES = [
    "/reports/report.json.md",
    "/reports/report.json.json",
    "/experiments/experiment.json.md",
    "/sessions/recorded/checkout.json.md",
    "/sessions/recorded/checkout.json.json",
    "/llms.txt",
]


def test_a_report_and_an_experiment_are_served_as_markdown_at_their_page_address(client: Client):
    report = client.send("GET", "/reports/report.json.md")
    experiment = client.send("GET", "/experiments/experiment.json.md")

    assert report == Reply(HTTPStatus.OK, REPORT_MARKDOWN)
    assert experiment == Reply(HTTPStatus.OK, EXPERIMENT_MARKDOWN)
    assert report.headers["Content-Type"] == experiment.headers["Content-Type"] == "text/markdown; charset=utf-8"


def test_a_report_and_an_experiment_are_served_as_json_at_their_page_address(client: Client):
    assert client.send("GET", "/reports/report.json.json") == Reply(
        HTTPStatus.OK, _report("alpha", "beta", failing=("beta",))
    )
    assert client.send("GET", "/experiments/experiment.json.json") == Reply(HTTPStatus.OK, _experiment("alpha", "beta"))


def test_a_session_is_served_as_markdown_and_as_json_at_its_page_address(client: Client):
    recorded = _session("s-1", [(0, 1.5)], [])
    _write(client.root / "recorded" / "checkout.json", recorded)
    _write(client.root / "recorded" / "checkout.diagnosis.json", _diagnosis("s-1"))
    _write(client.root / "lost.diagnosis.json", _diagnosis("s-gone"))

    session = client.send("GET", "/sessions/recorded/checkout.json.md")
    lost = client.send("GET", "/sessions/lost.diagnosis.json.md")

    assert session.headers["Content-Type"] == "text/markdown; charset=utf-8"
    assert session.body.startswith(
        "# Session: recorded/checkout.json\n\n- Session: s-1\n- Traces: 2\n- Spans: 1\n- Time: 1.50s\n"
        "- Diagnoses: recorded/checkout.diagnosis.json\n\n"
        "## Diagnosis: recorded/checkout.diagnosis.json\n\n```json\n{\n"
    )
    assert session.body.endswith(
        "## Trace 1 of 2: s-1-0\n\n### Span 0: inference\n\n## Trace 2 of 2: s-1-1\n\nNo spans.\n"
    )
    assert lost.body.startswith(
        "# Diagnosis: lost.diagnosis.json\n\n- Session: s-gone, which is not in the folder\n"
        "- Failures: 1\n- Root causes: 1\n\n```json\n"
    )
    assert client.send("GET", "/sessions/recorded/checkout.json.json") == Reply(HTTPStatus.OK, recorded)
    assert client.send("GET", "/sessions/lost.diagnosis.json.json") == Reply(HTTPStatus.OK, _diagnosis("s-gone"))


def test_markdown_names_a_file_as_the_scan_does_however_it_was_asked_for(client: Client):
    _write(client.root / "code" / "nightly.report.json", _report("gamma"))

    roundabout = client.send("GET", "/reports/code/../report.json.md")
    unpaired = client.send("GET", "/reports/code/nightly%2Ereport.json.md")

    assert roundabout == Reply(HTTPStatus.OK, REPORT_MARKDOWN)
    assert unpaired.body.startswith("# Report: code/nightly.report.json\n")
    assert (
        "\n- Experiment: none found in this folder; a report made in code has none\n\n## Failing rows" in unpaired.body
    )


def test_markdown_says_why_a_report_was_left_unpaired(client: Client):
    for side in ("left", "right"):
        _write(client.root / "tie" / side / "e.json", _experiment("t"))
    _write(client.root / "tie" / "shared.report.json", _report("t"))

    reply = client.send("GET", "/reports/tie/shared.report.json.md")

    assert (
        "\n- Experiment: not paired: 2 experiments hold every case of this report and are equally near: "
        "tie/left/e.json, tie/right/e.json\n"
    ) in reply.body


def test_a_file_whose_own_name_ends_in_a_form_has_a_page_and_its_data(client: Client):
    """`copy.json.json` is a file, not the JSON form of `copy.json`, when only the longer name is there."""
    _write(client.root / "copy.json.json", _report("gamma"))

    page = client.send("GET", "/reports/copy.json.json", headers={**NO_TOKEN, "Cookie": client.cookie})
    data = client.send("GET", "/reports/copy.json.json.json")
    unopened = client.send("GET", "/reports/copy.json.json", headers=NO_TOKEN)
    form = client.send("GET", "/reports/report.json.json", headers={**NO_TOKEN, "Cookie": client.cookie})

    assert (page.status, page.body) == (HTTPStatus.OK, PAGE)
    assert data == Reply(HTTPStatus.OK, _report("gamma"))
    assert unopened == _refused(HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)
    assert form == _refused(HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)


def test_markdown_survives_a_report_holding_nan(client: Client):
    (client.root / "odd.report.json").write_text(
        '{"overall_score": NaN, "cases": [{"name": "a", "evaluator": "E", "input": Infinity}],'
        ' "scores": [NaN], "test_passes": [false]}'
    )

    reply = client.send("GET", "/reports/odd.report.json.md")

    assert reply.status == HTTPStatus.OK
    assert "- Overall score: none\n" in reply.body
    assert reply.body.endswith(
        "### a · E\n\n- Score: none\n- Verdict: failed\n\n## All rows\n\n"
        + ("| Case | Evaluator | Score | Verdict |\n| --- | --- | --- | --- |\n| a | E | none | failed |\n")
    )


@pytest.mark.parametrize("address", DATA_ADDRESSES)
def test_the_data_at_a_page_address_needs_the_token_itself(client: Client, address):
    _write(client.root / "recorded" / "checkout.json", _session("s-1", [(0, 1)]))

    assert client.send("GET", address, headers=NO_TOKEN) == _refused(HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)
    assert client.send("GET", address, headers={**NO_TOKEN, "Cookie": client.cookie}) == _refused(
        HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL
    )
    assert client.send("GET", f"{address}?t=guessed", headers=NO_TOKEN) == _refused(
        HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL
    )
    assert client.send("GET", address, headers={"Host": "evals.evil.example.com"}) == _refused(
        HTTPStatus.FORBIDDEN, HOST_REFUSAL
    )
    assert client.send("GET", f"{address}?t={client.token}", headers=NO_TOKEN).status == HTTPStatus.OK


@pytest.mark.parametrize(
    "address,status,message",
    [
        ("/reports/../../etc/passwd.json.md", HTTPStatus.FORBIDDEN, "../../etc/passwd.json is outside {root}"),
        (
            "/reports/%2E%2E/%2E%2E/etc/passwd.json.json",
            HTTPStatus.FORBIDDEN,
            "../../etc/passwd.json is outside {root}",
        ),
        ("/experiments/..%2Foutside.json.md", HTTPStatus.FORBIDDEN, "../outside.json is outside {root}"),
        ("/reports/leak.json.md", HTTPStatus.FORBIDDEN, "leak.json is outside {root}"),
        ("/reports/%00.json.md", HTTPStatus.FORBIDDEN, "'\\x00.json' is not a path"),
        ("/reports/nowhere.json.md", HTTPStatus.NOT_FOUND, "nowhere.json is not a file"),
        ("/reports/experiment.json.md", HTTPStatus.NOT_FOUND, "experiment.json is not a report"),
        ("/experiments/report.json.json", HTTPStatus.NOT_FOUND, "report.json is not an experiment"),
        ("/sessions/report.json.md", HTTPStatus.NOT_FOUND, "report.json is not a session"),
        ("/reports/recorded/checkout.json.md", HTTPStatus.NOT_FOUND, "recorded/checkout.json is not a report"),
        ("/reports/notes.json.json", HTTPStatus.NOT_FOUND, "notes.json is not a report"),
    ],
    ids=[
        "outside the root",
        "outside the root, encoded",
        "outside the root, encoded slash",
        "a link out of the root",
        "null byte",
        "missing file",
        "an experiment asked for as a report",
        "a report asked for as an experiment",
        "a report asked for as a session",
        "a session asked for as a report",
        "neither",
    ],
)
def test_the_data_at_a_page_address_is_refused_for_a_file_it_cannot_serve(client: Client, address, status, message):
    outside = _write(client.root.parent / "outside.json", _report("secret"))
    (client.root / "leak.json").symlink_to(outside)
    _write(client.root / "notes.json", {"just": "notes"})
    _write(client.root / "recorded" / "checkout.json", _session("s-1", [(0, 1)]))

    reply = client.send("GET", address)

    assert reply == _refused(status, message.replace("{root}", client.settings["root"]))


@pytest.mark.parametrize("text", ['{"cases": [', TOO_DEEP], ids=["cut short", "nested too deeply"])
@pytest.mark.parametrize("address", ["/reports/odd.json.md", "/api/file?path=odd.json"])
def test_reading_says_when_the_file_is_not_json(client: Client, address, text):
    (client.root / "odd.json").write_text(text)

    reply = client.send("GET", address)

    assert reply.status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert reply.body["error"].startswith("odd.json could not be parsed: ")


def test_markdown_prints_a_score_too_large_for_two_decimals(client: Client):
    (client.root / "big.json").write_text(json.dumps({**_report("alpha"), "overall_score": 1e26, "scores": [1e26]}))

    reply = client.send("GET", "/reports/big.json.md")

    assert reply.status == HTTPStatus.OK
    assert "- Overall score: 100000000000000000000000000.00\n" in reply.body
    assert reply.body.endswith("| alpha | OutputEvaluator | 100000000000000000000000000.00 | passed |\n")


@pytest.mark.parametrize("levels", [DEPTH_LIMIT - 2, 5000], ids=["just past the depth written", "far past it"])
def test_markdown_says_when_a_file_is_nested_too_deeply_to_write(client: Client, levels):
    """The same on every Python, so that a small file cannot be laid out as hundreds of megabytes."""
    nested = "[" * levels + "1" + "]" * levels
    failing = _report("alpha", failing=("alpha",))
    (client.root / "deep.json").write_text(json.dumps(failing).replace('"ask about alpha"', nested))

    reply = client.send("GET", "/reports/deep.json.md")

    assert reply.status == HTTPStatus.UNPROCESSABLE_ENTITY
    # A Python that cannot read the file that deep says so first.
    assert reply.body["error"].startswith(
        ("deep.json is nested more than 200 levels", "deep.json could not be parsed: ")
    )


def test_markdown_writes_a_file_nested_as_deeply_as_is_written(client: Client):
    # The value sits three levels down: in the report, in its cases, in one case.
    nested = "[" * (DEPTH_LIMIT - 3) + "1" + "]" * (DEPTH_LIMIT - 3)
    failing = _report("alpha", failing=("alpha",))
    (client.root / "deep.json").write_text(json.dumps(failing).replace('"ask about alpha"', nested))

    reply = client.send("GET", "/reports/deep.json.md")

    assert reply.status == HTTPStatus.OK
    assert reply.body.startswith("# Report: deep.json\n")


# ── the server: Markdown of a document the page sends ──

RENDER = "/api/markdown"
LINKED = {"status": "verified", "detail": "all 2 report case(s) exist in this experiment"}
FAILING = _report("alpha", "beta", failing=("beta",))


@pytest.mark.parametrize(
    "body,written",
    [
        (
            {"kind": "report", "path": "tune · c3", "document": FAILING},
            "# Report: tune · c3\n\n- Overall score: 0.50\n- Rows: 2 (1 passed, 1 failed, 0 not applicable)\n",
        ),
        (
            {"kind": "report", "path": "r.json", "document": FAILING, "experiment": "e.json", "link": LINKED},
            "- Experiment: e.json (verified: all 2 report case(s) exist in this experiment)\n",
        ),
        (
            {"kind": "row", "row": 1, "document": FAILING, "report": "/served/r.json", "experiment": "/served/e.json"},
            "# beta · OutputEvaluator\n\n- Score: 0.00\n- Verdict: failed\n- Label: no_match\n"
            "- Report: `/served/r.json`\n- Experiment: `/served/e.json`\n",
        ),
        (
            {"kind": "failing", "document": FAILING, "report": "/served/r.json", "command": "strands-evals run e"},
            "This row fails in an evaluation report.",
        ),
        (
            {"kind": "experiment", "path": "e.json", "document": _experiment("edited"), "reports": ["r.json"]},
            "# Experiment: e.json\n\n- Cases: 1\n- Evaluators: 1\n- Reports: r.json\n",
        ),
        (
            {"kind": "case", "index": 0, "document": {"name": "edited", "input": "q"}, "experiment": "/served/e.json"},
            "# edited\n\n- Experiment: `/served/e.json`\n\nInput:\n\n```text\nq\n```\n",
        ),
        (
            {"kind": "trace", "document": [{"role": "user", "content": "hi"}], "cost": "1 assistant message"},
            "# Trace: 1 message\n\n1 assistant message\n\n## Message 1: user\n",
        ),
    ],
    ids=[
        "a report",
        "a report with its pairing",
        "one row",
        "the failing rows",
        "an experiment",
        "one case",
        "a trace",
    ],
)
def test_a_document_the_page_holds_is_written_as_markdown(client: Client, body, written):
    reply = client.send("POST", RENDER, body=body)

    assert reply.status == HTTPStatus.OK
    assert reply.headers["Content-Type"] == "text/markdown; charset=utf-8"
    assert written in reply.body


@pytest.mark.parametrize(
    "body,message",
    [
        ({"kind": "poem", "document": {}}, "'kind' must be report, row, failing, experiment, case or trace"),
        ({"document": FAILING}, "'kind' must be report, row, failing, experiment, case or trace"),
        ({"kind": ["report"]}, "'kind' must be report, row, failing, experiment, case or trace"),
        ({"kind": "report", "document": FAILING}, "'path' is missing or is not what a 'report' takes"),
        (
            {"kind": "report", "path": "r.json", "document": [1]},
            "'document' is missing or is not what a 'report' takes",
        ),
        (
            {"kind": "row", "row": True, "document": FAILING, "report": "r"},
            "'row' is missing or is not what a 'row' takes",
        ),
        ({"kind": "row", "row": 2, "document": FAILING, "report": "r"}, "the report has no row 2"),
        ({"kind": "case", "index": 0, "document": {}}, "'experiment' is missing or is not what a 'case' takes"),
        ({"kind": "trace", "document": "text"}, "'document' is missing or is not what a 'trace' takes"),
        (
            {"kind": "report", "path": "r.json", "document": FAILING, "experiment": "e.json", "link": {"status": "x"}},
            "'link' must hold a 'status' and a 'detail', as a scan gives them",
        ),
    ],
    ids=[
        "no such kind",
        "no kind",
        "a kind that is not a name, and nothing else",
        "no path",
        "not a document",
        "a row that is no number",
        "no such row",
        "no file",
        "text",
        "a pairing that does not say how the two fit",
    ],
)
def test_a_render_request_that_does_not_say_what_to_write_is_refused(client: Client, body, message):
    assert client.send("POST", RENDER, body=body) == _refused(HTTPStatus.BAD_REQUEST, message)


def test_rendering_needs_the_token_and_writes_nothing(serve):
    client = serve(read_only=True)
    before = _snapshot(client.root)
    body = {"kind": "report", "path": "r.json", "document": FAILING}

    assert client.send("POST", RENDER, body=body, headers=WRONG_TOKEN) == _refused(
        HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL
    )
    cookie_alone = {**NO_TOKEN, "Cookie": client.cookie}
    assert client.send("POST", RENDER, body=body, headers=cookie_alone).status == HTTPStatus.UNAUTHORIZED
    assert client.send("POST", RENDER, body=body, headers={"Host": "evil.example"}).status == HTTPStatus.FORBIDDEN
    assert client.send("POST", RENDER, body=body).status == HTTPStatus.OK
    assert client.send("POST", "/api/experiment?path=experiment.json", body=SAVE).status == HTTPStatus.NOT_FOUND
    assert _snapshot(client.root) == before


def test_rendering_reads_nothing_from_the_folder_on_the_word_of_the_body(client: Client, tmp_path_factory):
    outside = _write(tmp_path_factory.mktemp("outside") / "secret.json", _report("kept out"))
    named = {"path": "../secret.json", "report": str(outside), "experiment": "experiment.json"}

    report = client.send("POST", f"{RENDER}?path=experiment.json", body={"kind": "report", "document": {}, **named})
    row = client.send("POST", RENDER, body={"kind": "row", "row": 0, "document": FAILING, **named})

    assert report.body.startswith("# Report: ../secret.json\n\n- Overall score: none\n- Rows: 0 ")
    assert "- Experiment: experiment.json\n" in report.body
    assert f"- Report: `{outside}`\n- Experiment: `experiment.json`\n" in row.body
    assert "kept out" not in report.body + row.body and "ask about" not in report.body


def test_rendering_is_not_opened_to_a_page_of_another_origin(client: Client):
    body = {"kind": "report", "path": "r.json", "document": FAILING}
    elsewhere = {"Origin": "http://evil.example", "Content-Type": "text/plain"}
    asking = {**elsewhere, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-evals-token"}

    preflight = client.send("OPTIONS", RENDER, headers={**NO_TOKEN, **asking})
    form = client.send("POST", RENDER, body=body, headers={**NO_TOKEN, **elsewhere, "Cookie": client.cookie})
    answered = client.send("POST", RENDER, body=body, headers=elsewhere)

    assert preflight.status == HTTPStatus.NOT_IMPLEMENTED
    assert form == _refused(HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)
    assert answered.status == HTTPStatus.OK
    assert not any(
        name.lower().startswith("access-control-") for reply in (preflight, form, answered) for name in reply.headers
    )


def test_rendering_refuses_a_body_over_the_size_cap(client: Client):
    reply = client.send("POST", RENDER, headers={"Content-Length": str(MAX_UPLOAD_BYTES + 1)})

    assert reply == _refused(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body over {MAX_UPLOAD_BYTES} bytes")
    assert reply.headers["Connection"] == "close"


def test_rendering_writes_a_number_that_is_not_json_as_a_file_of_it_is_written(client: Client):
    document = (
        '{"overall_score": NaN, "cases": [{"name": "a", "evaluator": "E", "input": [Infinity]}],'
        ' "scores": [NaN], "test_passes": [false]}'
    )
    (client.root / "odd.report.json").write_text(document)
    body = f'{{"kind": "report", "path": "odd.report.json", "document": {document}}}'

    reply = client.send("POST", RENDER, body=body.encode())

    assert reply.status == HTTPStatus.OK
    assert "Input:\n\n```json\n[\n  null\n]\n```\n" in reply.body
    # The file's own Markdown says what the scan found of an experiment, which a posted document was not scanned for.
    unpaired = "\n- Experiment: none found in this folder; a report made in code has none"
    assert reply.body == client.send("GET", "/reports/odd.report.json.md").body.replace(unpaired, "")


@pytest.mark.parametrize("levels", [DEPTH_LIMIT, 5000], ids=["just past the depth written", "far past it"])
def test_rendering_says_when_a_document_is_nested_too_deeply_to_write(client: Client, levels):
    failing = {"cases": [{"name": "a", "evaluator": "E", "input": "deep"}], "scores": [0], "test_passes": [False]}
    body = json.dumps({"kind": "report", "path": "r.json", "document": failing})

    reply = client.send("POST", RENDER, body=body.replace('"deep"', "[" * levels + "1" + "]" * levels).encode())

    # A Python that cannot read the body that deep says so first.
    assert reply.status in (HTTPStatus.UNPROCESSABLE_ENTITY, HTTPStatus.BAD_REQUEST)
    assert reply.body["error"].startswith(("the document is nested more than 200 levels", "body is not JSON: "))


def test_rendering_says_nothing_of_a_folder_it_did_not_read(client: Client):
    """A candidate's report was paired with nothing, and a case being edited has no file that holds the rest."""
    report = client.send("POST", RENDER, body={"kind": "report", "path": "tune · c3", "document": FAILING})
    case = {"kind": "case", "index": 0, "document": {"name": "c", "input": "z" * 5000}, "experiment": "unsaved"}
    cut = client.send("POST", RENDER, body=case)

    assert "- Evaluators: OutputEvaluator\n\n## Failing rows" in report.body
    assert cut.body.endswith(f"```text\n{'z' * 4000}\n```\n\nCut after 4000 of 5000 characters.\n")


def test_llms_txt_describes_every_address_the_server_answers(client: Client):
    reply = client.send("GET", "/llms.txt")
    lines = reply.body.splitlines()
    route_script = resources.files("strands_evals.cli.assets").joinpath("ui-route.js").read_text()
    views = re.findall(r"'opt-[a-z]+': '([a-z-]+)'", route_script)

    assert reply.status == HTTPStatus.OK
    assert reply.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert lines[0] == "# strands-evals ui"
    assert "sent as the `X-Evals-Token` header or as `?t=<token>`" in reply.body
    assert "The cookie a browser holds opens the page only" in reply.body
    assert "In Markdown a value longer than 4000 characters is cut" in reply.body
    assert (
        "A diagnosis file is written whole; one held in a report row is cut like any other value. "
        "A document nested more than 200 levels deep is not written: its Markdown answers 422."
    ) in reply.body
    assert (
        "This description, the Markdown and JSON forms, `POST /api/markdown`, the shape of `/api/scan` and the "
        "prompt the page copies are experimental: they may change before a release."
    ) in lines
    assert "Asked for without `text/html` in `Accept`, one answers 406" in reply.body
    # Every route the handler has is listed with what it returns, so a route added later cannot go undescribed.
    for route, handler in UIHandler._GET_ROUTES.items():
        assert handler.__doc__
        assert f"- `{route}`: {' '.join(handler.__doc__.split())}" in lines
    assert (
        '- `PUT /api/experiment`: With `?path=<experiment file>` and a body of `{"document": ..., "modified": ...}`: '
        "saves the experiment over the file it was read from. `modified` is the time `/api/file` gave; a file changed "
        "since is refused, and so is every save when the server was started with `--read-only`."
    ) in lines
    (rendering,) = (line for line in lines if line.startswith("- `POST /api/markdown`: "))
    assert rendering.startswith('- `POST /api/markdown`: With a body of `{"kind": ..., "document": ...}`: ')
    assert rendering.endswith(
        "A path in the body is a name to write: nothing is read from the folder. A body over 32 MB is refused."
    )
    assert MAX_UPLOAD_BYTES == 32 * 1024 * 1024
    assert (
        "`POST /api/markdown` writes the same Markdown for a document sent to it. The page uses it for what has no "
        "file of its own: one row, one case, the failing rows as a prompt, a trace inside a report, a candidate's "
        "report, and an experiment with edits that are not saved."
    ) in lines
    for line in (
        "- `/reports/<report file>.md`: its figures, each failing row with its input, expected and actual output "
        "and the evaluator's reason, then every row",
        "- `/experiments/<experiment file>.md`: its evaluators with their settings, then every case",
        "- `/reports/<report file>.json`: the report file's document",
        "- `/experiments/<experiment file>.json`: the experiment file's document",
        "- `/sessions/<session file>.md`: what it recorded, each diagnosis made of it, then every span of every "
        "trace: the prompts, the model's messages, the tool calls and their results",
        "- `/sessions/<session file>.json`: the session file's document",
        "- `/reports/<report file>/rows/<case>/<evaluator>`: one row of it, opened and in view",
        "- `/reports/<report file>/rows/<case>`: the same, for a row whose case has no evaluator named",
        "- `/experiments/<experiment file>/cases/<name>`: the same, on one case",
        "- `/sessions/<session file>`: one session, trace by trace, under the diagnoses made of it",
        "- `/optimization/<run directory>/candidates/<id>`: one candidate of it, opened and in view",
        "- `/optimization/<run directory>/candidates/<id>/report`: the cases that candidate ran, as a report",
    ):
        assert line in lines
    # Every root the page's own route parser knows is listed, and none is listed with nothing said of it.
    assert sorted(f"/{root}" for root in re.findall(r"root === '([a-z]+)'", route_script)) == sorted(PAGE_ROOTS)
    for root in PAGE_ROOTS:
        assert any(line.startswith(f"- `{root}`: ") for line in lines)
    assert not any(line.endswith(": ") for line in lines)
    assert len(views) == 5
    view_line = next(line for line in lines if line.startswith("- `/optimization/<run directory>/<view>`: "))
    assert all(re.search(rf"\b{view}\b", view_line) for view in views)


# ── the server: saving ──


def test_put_writes_back_values_and_keys_the_editor_has_no_field_for(client: Client):
    """A red-team experiment carries strategies at the top, and any case can hold structured values."""
    document = {
        "schema_version": 2,
        "attack_strategies": [{"strategy_type": "crescendo", "label": "slow"}],
        "cases": [
            {
                "name": "a",
                "input": {"instruction": "Describe it", "media": ["q3.png"]},
                "expected_output": [{"role": "assistant", "content": "A chart."}],
                "risk": {"category": "pii"},
            }
        ],
        "evaluators": [{"evaluator_type": "Equals", "internal": {"keep": True}}],
        "model": "judge-model",
    }
    target = _write(client.root / "experiment.json", _experiment("alpha"))

    reply = client.put("experiment.json", {"document": document, "modified": LOADED})

    assert reply.status == HTTPStatus.OK
    assert json.loads(target.read_text()) == document
    assert list(json.loads(target.read_text())) == list(document)


@pytest.mark.parametrize("name", ["experiment.json", "UPPER.JSON"])
def test_put_writes_the_file_it_loaded(client: Client, name):
    target = _write(client.root / name, _experiment("alpha", "beta"))
    loaded = client.send("GET", f"/api/file?path={name}").body
    document = loaded["document"]
    document["cases"][0]["expected_output"] = "a new expectation"

    reply = client.put(name, {"document": document, "modified": loaded["modified"]})

    assert reply == Reply(
        HTTPStatus.OK,
        {"path": name, "modified": target.stat().st_mtime, "cases": 2, "evaluator_notes": [MISSING_RUBRIC]},
    )
    assert json.loads(target.read_text()) == document


def test_two_saves_from_one_loaded_copy_do_not_both_land(client: Client, monkeypatch):
    """The second would silently undo the first: it is told the file changed, as if it had arrived later."""
    write = _ui_server.write_json_atomic

    def slowly(target: Path, payload: dict) -> float:
        time.sleep(0.2)  # long enough for the other save to have checked the stamp, were nothing holding it back
        return write(target, payload)

    monkeypatch.setattr(_ui_server, "write_json_atomic", slowly)
    replies: list[Reply] = []
    savers = [threading.Thread(target=lambda: replies.append(client.put("experiment.json", SAVE))) for _ in range(2)]
    stamp = (client.root / "experiment.json").stat().st_mtime

    for saver in savers:
        saver.start()
    for saver in savers:
        saver.join(timeout=10)

    assert sorted(reply.status for reply in replies) == [HTTPStatus.OK, HTTPStatus.CONFLICT]
    assert (client.root / "experiment.json").stat().st_mtime != stamp


@pytest.mark.parametrize(
    "path,body,status,message",
    _cases(
        {
            "stale stamp": (
                "experiment.json",
                {**SAVE, "modified": 1.0},
                HTTPStatus.CONFLICT,
                "experiment.json changed on disk since it was loaded; reload before saving",
            ),
            "target is not an experiment": (
                "report.json",
                SAVE,
                HTTPStatus.CONFLICT,
                "report.json is not an experiment file",
            ),
            "target is not on disk": (
                "new/created.json",
                {**SAVE, "modified": 1.0},
                HTTPStatus.NOT_FOUND,
                "new/created.json is no longer on disk, so it cannot be saved in place",
            ),
            "target is outside the root": (
                "../escaped.json",
                {**SAVE, "modified": 1.0},
                HTTPStatus.FORBIDDEN,
                "../escaped.json is outside {root}",
            ),
            "target is not a .json file": (
                "notes.txt",
                SAVE,
                HTTPStatus.BAD_REQUEST,
                "only .json files can be written",
            ),
            "invalid case": (
                "experiment.json",
                {**SAVE, "document": {"cases": ["broken"], "evaluators": []}},
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "case 0 is not a JSON object",
            ),
            "no document": (
                "experiment.json",
                {"modified": LOADED},
                HTTPStatus.BAD_REQUEST,
                "expected {'document': ..., 'modified': ...}",
            ),
            "body is not an object": (
                "experiment.json",
                ["a list"],
                HTTPStatus.BAD_REQUEST,
                "body is not a JSON object",
            ),
            "empty body": ("experiment.json", b"", HTTPStatus.BAD_REQUEST, "empty body"),
        }
    ),
)
def test_put_refuses_and_leaves_the_tree_untouched(client: Client, path, body, status, message):
    (client.root / "notes.txt").write_text("{}")
    before = _snapshot(client.root)

    reply = client.put(path, body)

    assert reply == _refused(status, message.replace("{root}", client.settings["root"]))
    assert _snapshot(client.root) == before


@pytest.mark.parametrize(
    "stamp",
    ["yesterday", [1], True, None, 10**400, float("inf"), float("nan")],
    ids=["string", "list", "boolean", "null", "integer past 2**53", "infinite", "not a number"],
)
def test_put_needs_a_finite_number_for_the_modified_stamp(client: Client, stamp):
    before = _snapshot(client.root)

    reply = client.put("experiment.json", {**SAVE, "modified": stamp})

    assert reply == _refused(HTTPStatus.BAD_REQUEST, "'modified' must be the number the file was loaded with")
    assert _snapshot(client.root) == before


def test_put_reports_a_document_that_cannot_be_json_as_unprocessable(client: Client):
    before = _snapshot(client.root)
    document = _experiment("a")
    document["cases"][0]["metadata"] = {"score": float("nan")}

    reply = client.put("experiment.json", {"document": document, "modified": LOADED})

    assert reply.status == HTTPStatus.UNPROCESSABLE_ENTITY
    # The rest of the message is the json module's.
    assert reply.body["error"].startswith("experiment.json is not valid JSON: ")
    assert _snapshot(client.root) == before


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX permission bits")
def test_put_keeps_the_permissions_of_the_file_it_replaces(client: Client):
    target = client.root / "experiment.json"
    target.chmod(0o640)

    reply = client.put("experiment.json", SAVE)

    assert reply.status == HTTPStatus.OK
    assert target.stat().st_mode & 0o777 == 0o640


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs a file the test user cannot write")
def test_put_refuses_a_read_only_file(client: Client):
    (client.root / "experiment.json").chmod(0o444)
    before = _snapshot(client.root)

    reply = client.put("experiment.json", SAVE)

    assert reply == _refused(HTTPStatus.FORBIDDEN, "experiment.json is read-only")
    assert _snapshot(client.root) == before


def test_a_read_only_server_refuses_writes(serve):
    client = serve(read_only=True)
    before = _snapshot(client.root)

    reply = client.put("experiment.json", SAVE)

    assert reply == _refused(HTTPStatus.FORBIDDEN, "server started with --read-only")
    assert client.send("GET", "/api/config") == Reply(HTTPStatus.OK, {**client.config, "read_only": True})
    assert _snapshot(client.root) == before


def test_put_to_an_unknown_route_is_not_found(client: Client):
    assert client.send("PUT", "/api/other", body=SAVE) == _refused(HTTPStatus.NOT_FOUND, "no route for /api/other")


# ── the server: request bodies it refuses ──


def test_a_refused_put_does_not_leave_its_body_for_the_next_request(client: Client):
    conn = client.connect()
    try:
        refused = client.put("experiment.json", SAVE, headers=WRONG_TOKEN, conn=conn)
        connection = conn.sock
        followed = client.send("GET", "/api/config", conn=conn)
        reused = conn.sock is connection
    finally:
        conn.close()

    assert refused == _refused(HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL)
    assert followed == Reply(HTTPStatus.OK, client.config)
    assert reused


@pytest.mark.parametrize(
    "headers,status,message",
    _cases(
        {
            "over the size cap": (
                {"Content-Length": str(MAX_UPLOAD_BYTES + 1)},
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"body over {MAX_UPLOAD_BYTES} bytes",
            ),
            "no length": ({}, HTTPStatus.LENGTH_REQUIRED, LENGTH_REFUSAL),
            "length is not a number": ({"Content-Length": "lots"}, HTTPStatus.LENGTH_REQUIRED, LENGTH_REFUSAL),
            "chunked": ({"Transfer-Encoding": "chunked"}, HTTPStatus.LENGTH_REQUIRED, LENGTH_REFUSAL),
            "wrong token, no length": (WRONG_TOKEN, HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL),
            "wrong token, chunked": (
                {**WRONG_TOKEN, "Transfer-Encoding": "chunked"},
                HTTPStatus.UNAUTHORIZED,
                TOKEN_REFUSAL,
            ),
            "wrong token, body withheld": (
                {**WRONG_TOKEN, "Content-Length": "1000"},
                HTTPStatus.UNAUTHORIZED,
                TOKEN_REFUSAL,
            ),
        }
    ),
)
def test_a_put_whose_body_cannot_be_read_closes_the_connection(client: Client, monkeypatch, headers, status, message):
    # Only the withheld body waits, and only this long.
    monkeypatch.setattr(_ui_server, "DRAIN_TIMEOUT", 0.05)

    reply = client.put("experiment.json", None, headers=headers)

    assert reply == _refused(status, message)
    assert reply.headers["Connection"] == "close"


# ── the command ──


def _list_tree(root: Path) -> None:
    _write(root / "experiment.json", _experiment("alpha"))
    _write(root / "report.json", _report("alpha"))
    _write(root / "spare" / "unrun.json", _experiment("beta"))
    _write(root / "code" / "nightly.json", _report("gamma"))
    for copy in ("one", "two"):
        _write(root / "tied" / copy / "experiment.json", _experiment("delta"))
    _write(root / "tied" / "nightly.json", _report("delta"))
    _write(root / "notes.json", {"hello": "world"})
    _optimizer_run(
        root / "tuning",
        {"event": "propose", "iteration": 0, "candidate_id": "c0"},
        {"event": "iteration_end", "iteration": 0},
        {"event": "picked", "iteration": 0, "report": {"champion_id": "c0"}},
        {"event": "stop", "iteration": 0, "reason": "budget"},
    )
    _optimizer_run(root / "tuning" / "unfinished")
    _write(root / "recorded" / "checkout.json", _session("s-1", [(0, 1), (1, 2)], [(5, 6)]))
    _write(root / "recorded" / "checkout.diagnosis.json", _diagnosis("s-1"))
    _write(root / "recorded" / "quiet.json", _session("s-2", [(0, 1)]))
    _write(root / "lost.diagnosis.json", _diagnosis("s-9"))
    LocalFileTaskResultStore(root / "store").save("a", EvaluationData(input="q", actual_output="a"))


def test_ui_list_prints_what_it_found_as_json(tmp_path: Path, capsys):
    _list_tree(tmp_path)

    assert main(["--json", "ui", str(tmp_path), "--list"]) == 0

    found = asdict(discover(tmp_path))
    assert {name: len(listed) for name, listed in found.items() if name != "truncated"} == {
        "experiments": 4,
        "reports": 3,
        "skipped": 1,
        "optimizer_runs": 2,
        "sessions": 2,
        "diagnoses": 2,
        "stores": 1,
    }
    assert json.loads(capsys.readouterr().out) == {"schema_version": 1, "root": str(tmp_path.resolve()), **found}


def test_ui_list_prints_what_it_found_as_text(tmp_path: Path, capsys):
    _list_tree(tmp_path)

    # `--rich` because stdout is not a TTY under capsys, which would otherwise pick JSON.
    assert main(["--rich", "ui", str(tmp_path), "--list"]) == 0

    printed = capsys.readouterr()
    assert sorted(printed.out.splitlines()) == [
        "(no experiment)  ->  code/nightly.json (1.00)",
        "(no session)  ->  lost.diagnosis.json (1 failures)",
        "(not paired: 2 experiments hold every case of this report and are equally near: "
        "tied/one/experiment.json, tied/two/experiment.json)  ->  tied/nightly.json (1.00)",
        "experiment.json (1 cases)  ->  report.json (1.00)",
        "optimizer run tuning (1 candidates, 1 iterations)  ->  champion c0, not finished, stopped: budget",
        "optimizer run tuning/unfinished (0 candidates, 0 iterations)  ->  no champion, not finished",
        "session recorded/checkout.json (2 traces, 3 spans)  ->  recorded/checkout.diagnosis.json",
        "session recorded/quiet.json (1 traces, 1 spans)  ->  (not diagnosed)",
        "spare/unrun.json (1 cases)  ->  (not run)",
        "task result store store (1 results)",
        "tied/one/experiment.json (1 cases)  ->  (not run)",
        "tied/two/experiment.json (1 cases)  ->  (not run)",
    ]
    assert printed.err == f"skipped notes.json: {NOT_RECOGNISED}\n"


def test_ui_list_says_when_the_scan_stopped_at_the_file_cap(tmp_path: Path, capsys, monkeypatch):
    for name in ("a.json", "b.json"):
        _write(tmp_path / name, _experiment("a"))
    monkeypatch.setattr(_ui_discovery, "MAX_SCAN_FILES", 1)

    assert main(["--json", "ui", str(tmp_path), "--list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert main(["--rich", "ui", str(tmp_path), "--list"]) == 0
    printed = capsys.readouterr()

    assert (listed["truncated"], [experiment["path"] for experiment in listed["experiments"]]) == (True, ["a.json"])
    assert printed.out == "a.json (1 cases)  ->  (not run)\n"
    assert printed.err == "scan hit the file cap: the rest of DIRECTORY was not read\n"


@pytest.mark.parametrize("nested", [DEEP, "[" * 3000 + "1" + "]" * 3000], ids=["laid out", "on one line"])
def test_ui_list_prints_json_for_a_file_nested_deeper_than_a_copy_follows(tmp_path: Path, capsys, nested):
    (tmp_path / "deep.json").write_text(_deep_experiment().replace(DEEP, nested))
    _write(tmp_path / "plain.json", _experiment("a"))

    assert main(["--json", "ui", str(tmp_path), "--list"]) == 0

    listed = json.loads(capsys.readouterr().out)
    # A reader that stops at its recursion limit sets the deeper file aside, which is a listing too.
    assert {entry["path"] for entry in listed["experiments"] + listed["skipped"]} == {"deep.json", "plain.json"}


def test_ui_list_prints_json_for_a_report_holding_nan(tmp_path: Path, capsys):
    (tmp_path / "report.json").write_text(json.dumps({**_report("a"), "overall_score": float("nan")}))

    assert main(["--json", "ui", str(tmp_path), "--list"]) == 0

    def refuse(constant: str) -> None:
        raise AssertionError(f"{constant} is not JSON")

    listed = json.loads(capsys.readouterr().out, parse_constant=refuse)
    assert [report["overall_score"] for report in listed["reports"]] == [None]


def test_ui_rejects_a_path_that_is_not_a_directory(tmp_path: Path, capsys):
    target = _write(tmp_path / "experiment.json", _experiment("a"))

    assert main(["ui", str(target), "--list"]) == 2
    assert capsys.readouterr().err == f"strands-evals: error: {target} is not a directory\n"


class _FakeServer:
    config = type("Config", (), {"token": "fake-token"})()
    server_address = ("127.0.0.1", 9999)

    def serve_forever(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def server_close(self) -> None:
        return None


FAKE_URL = "http://127.0.0.1:9999/?t=fake-token"


@pytest.fixture
def launched(monkeypatch):
    """Run `strands-evals ui` without serving: what it asked `build_server` for and which links it opened."""
    seen: dict = {"opened": []}

    def fake_build(root, **kwargs):
        seen.update(kwargs, root=root)
        return _FakeServer()

    monkeypatch.setattr(ui_command, "build_server", fake_build)
    monkeypatch.setattr(ui_command.webbrowser, "open", seen["opened"].append)
    monkeypatch.delenv(ui_command.TOKEN_ENV, raising=False)
    return seen


@pytest.mark.parametrize(
    "flags,env,expected",
    [
        ([], None, None),
        (["--token", "bookmarkable"], None, "bookmarkable"),
        ([], "from-the-env", "from-the-env"),
        (["--token", "explicit"], "from-the-env", "explicit"),
        ([], "", None),
    ],
    ids=["default lets the server generate one", "flag", "environment", "flag beats environment", "empty is none"],
)
def test_ui_takes_the_token_from_the_flag_then_the_environment(
    tmp_path: Path, launched, monkeypatch, flags, env, expected
):
    if env is not None:
        monkeypatch.setenv(ui_command.TOKEN_ENV, env)

    assert main(["ui", str(tmp_path), "--no-browser", *flags]) == 0
    assert launched["token"] == expected


GENERATED = "  the token in that link is what authorizes the page; Ctrl-C to stop"
SUPPLIED = (
    "  that token was supplied, not generated: any process on this machine that guesses it can {} files under DIRECTORY"
)
BEYOND = (
    "  listening beyond this machine on {}: anyone who can reach it and has that link can {} files under DIRECTORY,"
    " over unencrypted HTTP"
)


@pytest.mark.parametrize(
    "flags,mode,notes",
    _cases(
        {
            "default": ([], "", [GENERATED]),
            "localhost": (["--host", "localhost"], "", [GENERATED]),
            "IPv6 loopback": (["--host", "::1"], "", [GENERATED]),
            "supplied token": (["--token", "short"], "", [SUPPLIED.format("read and write")]),
            "supplied token, read-only": (
                ["--token", "short", "--read-only"],
                " (read-only)",
                [SUPPLIED.format("read")],
            ),
            "every interface": (["--host", "0.0.0.0"], "", [GENERATED, BEYOND.format("0.0.0.0", "read and write")]),
            "a hostname, read-only": (
                ["--host", "evals.internal", "--read-only"],
                " (read-only)",
                [GENERATED, BEYOND.format("evals.internal", "read")],
            ),
        }
    ),
)
def test_ui_says_what_the_link_exposes(tmp_path: Path, launched, capsys, flags, mode, notes):
    assert main(["ui", str(tmp_path), "--no-browser", *flags]) == 0

    assert capsys.readouterr().err.splitlines() == [
        f"strands-evals ui{mode}: serving {tmp_path.resolve()}",
        f"  {FAKE_URL}",
        "  without a browser: send that token as the X-Evals-Token header; /llms.txt says what is answered",
        *notes,
    ]


@pytest.mark.parametrize("flags,opened", [([], [FAKE_URL]), (["--no-browser"], [])], ids=["default", "--no-browser"])
def test_ui_opens_the_printed_link_unless_told_not_to(tmp_path: Path, launched, flags, opened):
    assert main(["ui", str(tmp_path), *flags]) == 0
    assert launched["opened"] == opened
