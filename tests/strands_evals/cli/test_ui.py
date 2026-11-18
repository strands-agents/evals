"""Tests for `strands-evals ui`: what the scan finds, what the server admits, and what a save may overwrite."""

from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import asdict, dataclass, field
from email.message import Message
from http import HTTPStatus
from http.client import HTTPConnection
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from strands_evals.cli import _ui_server
from strands_evals.cli._ui_server import (
    MAX_UPLOAD_BYTES,
    UNAUTHORIZED_PAGE,
    EventLog,
    PathOutsideRoot,
    build_server,
    classify,
    discover,
    discover_optimizer_runs,
    evaluator_catalog,
    fingerprint,
    link_status,
    load_asset,
    read_event_log,
    resolve_within,
    server_url,
    validate_experiment_payload,
    write_json_atomic,
)
from strands_evals.cli.commands import ui as ui_command
from strands_evals.cli.main import main
from strands_evals.experiment import _builtin_evaluator_types

NOT_RECOGNISED = "not an experiment or a report"
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
        cases.append({"name": name, "session_id": f"sid-{name}", "evaluator": "OutputEvaluator"})
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


def _write(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def _listed(root: Path, relative: str) -> dict:
    """The part of a run's file entry that comes from the file itself."""
    info = (root / relative).stat()
    return {"path": relative, "name": relative.rsplit("/", 1)[-1], "size": info.st_size, "modified": info.st_mtime}


def _pairs(root: Path) -> set[tuple[str | None, str | None]]:
    """Each discovered run as (experiment path, report path)."""
    return {((run["experiment"] or {}).get("path"), (run["report"] or {}).get("path")) for run in discover(root).runs}


def _cases(table: dict[str, tuple]) -> list:
    """Parametrize rows from a table keyed by test id."""
    return [pytest.param(*row, id=name) for name, row in table.items()]


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


# ── classification and scanning ──


@pytest.mark.parametrize(
    "obj,expected",
    [
        ({"cases": [], "evaluators": []}, "experiment"),
        ({"overall_score": 0.5, "cases": [], "scores": [], "test_passes": []}, "report"),
        ({"cases": []}, None),
        ({"overall_score": 0.5, "cases": []}, None),
        ({"overall_score": True, "cases": [], "scores": [], "test_passes": []}, None),
        ({"name": "something else"}, None),
        ([], None),
        ("a string", None),
    ],
    ids=["experiment", "report", "cases alone", "score alone", "boolean score", "unrelated object", "list", "string"],
)
def test_classify_goes_by_shape_not_filename(obj, expected):
    assert classify(obj) == expected


def test_scan_names_the_files_it_could_not_use(tmp_path: Path):
    _write(tmp_path / "experiment.json", _experiment("a"))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "binary.json").write_bytes(b"\xff\xfe")
    (tmp_path / "unrelated.json").write_text('{"hello": "world"}')

    result = discover(tmp_path)

    assert [run["id"] for run in result.runs] == ["experiment.json"]
    reasons = {skip["path"]: skip["why"] for skip in result.skipped}
    # The tail of the invalid-JSON reason is the json module's wording.
    assert reasons.pop("broken.json").startswith("invalid JSON at line 1: ")
    assert reasons == {"binary.json": "could not read: UnicodeDecodeError", "unrelated.json": NOT_RECOGNISED}


def test_discover_describes_a_paired_run(tmp_path: Path):
    _write(tmp_path / "nightly" / "experiment.json", _experiment("a", "b"))
    _write(tmp_path / "nightly" / "report.json", _report("a", "b"))

    assert discover(tmp_path).runs == [
        {
            "id": "nightly/experiment.json",
            "dir": "nightly",
            "experiment": {
                **_listed(tmp_path, "nightly/experiment.json"),
                "cases": 2,
                "evaluators": [{"evaluator_type": "OutputEvaluator"}],
                "evaluator_notes": [MISSING_RUBRIC],
            },
            "report": {
                **_listed(tmp_path, "nightly/report.json"),
                "overall_score": 1.0,
                "rows": 2,
                "failing": 0,
                "not_applicable": 0,
                "evaluators": ["OutputEvaluator"],
            },
            "link": {"status": "verified", "detail": "all 2 report case(s) exist in this experiment"},
        }
    ]


def test_report_summary_counts_declined_rows_apart_from_failing_ones(tmp_path: Path):
    report = _report("ok", "bad", "skipped", failing=("bad",), declined=("skipped",))
    # Declined without passing is a failure to judge, so it counts as failing.
    report["detailed_results"][1] = [{"score": 0.0, "test_pass": False, "label": "not_applicable"}]
    _write(tmp_path / "report.json", report)

    assert discover(tmp_path).runs[0]["report"] == {
        **_listed(tmp_path, "report.json"),
        "overall_score": 0.5,
        "rows": 3,
        "failing": 1,
        "not_applicable": 1,
        "evaluators": ["OutputEvaluator"],
    }


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
            "never across directories": (
                ["a/experiment.json"],
                ["b/report.json"],
                {("a/experiment.json", None), (None, "b/report.json")},
            ),
        }
    ),
)
def test_discover_pairs_a_report_with_the_experiment_beside_it(tmp_path: Path, experiments, reports, expected):
    for name in experiments:
        _write(tmp_path / name, _experiment("a"))
    for name in reports:
        _write(tmp_path / name, _report("a"))

    assert _pairs(tmp_path) == expected


def test_discover_stops_at_the_depth_limit(tmp_path: Path):
    deep = tmp_path / "one" / "two" / "three"
    _write(deep / "experiment.json", _experiment("deep"))
    _optimizer_run(deep / "opt")

    shallow, full = discover(tmp_path, depth=2), discover(tmp_path, depth=4)

    assert (shallow.runs, shallow.optimizer_runs) == ([], [])
    assert [run["id"] for run in full.runs] == ["one/two/three/experiment.json"]
    assert [run["dir"] for run in full.optimizer_runs] == ["one/two/three/opt"]


def test_discover_skips_hidden_and_dependency_directories(tmp_path: Path):
    for directory in ("kept", ".git", "node_modules/pkg", "__pycache__"):
        _write(tmp_path / directory / "experiment.json", _experiment("a"))
        (tmp_path / directory / "broken.json").write_text("{not json")

    result = discover(tmp_path)

    assert [run["id"] for run in result.runs] == ["kept/experiment.json"]
    assert [skip["path"] for skip in result.skipped] == ["kept/broken.json"]


def test_scan_stops_at_the_file_cap_and_says_so(tmp_path: Path, monkeypatch):
    for name in ("a.json", "b.json", "c.json", "sub/d.json"):
        _write(tmp_path / name, _experiment("a"))

    monkeypatch.setattr(_ui_server, "MAX_SCAN_FILES", 2)
    capped = discover(tmp_path)
    monkeypatch.setattr(_ui_server, "MAX_SCAN_FILES", 4)
    whole = discover(tmp_path)

    assert (capped.truncated, {run["id"] for run in capped.runs}) == (True, {"a.json", "b.json"})
    assert (whole.truncated, len(whole.runs)) == (False, 4)


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_scan_does_not_read_through_a_link_that_leaves_the_folder(tmp_path: Path):
    outside = _write(tmp_path / "outside" / "experiment.json", _experiment("secret"))
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.json").symlink_to(outside)

    found = discover(root)

    assert found.runs == []
    assert found.skipped == [{"path": "link.json", "why": "a link to a file outside this folder"}]


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


# ── optimizer runs ──


def _event_log(path: Path, *events: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path


def _optimizer_run(directory: Path, *events: dict, config: dict | None = None) -> Path:
    _write(directory / "run.json", config or {"run_id": "r-1", "objective": "Maximize(Helpfulness)"})
    _event_log(directory / "events.jsonl", *events)
    return directory


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
    ],
    ids=["complete lines", "partial last line dropped", "broken middle line reported", "line that is not an object"],
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
            "a tie goes to the first path": (
                {"b-copy/experiment.json": "ab", "a-copy/experiment.json": "ab"},
                {"train": ["a", "b"]},
                _tuned("a-copy/experiment.json", 2, 2, 2),
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


def test_browser_fixture_run_is_readable():
    """The run the browser tests chart, read by the real reader."""
    fixture = Path(__file__).parents[2] / "ui" / "fixtures" / "optimizer-run"

    log = read_event_log(fixture / "events.jsonl")

    assert (log.bad_lines, log.truncated) == ([], False)
    assert log.events[0]["event"] == "config"
    # The invalid patch never became a candidate, so it has a proposal but no `propose` event.
    assert sum(1 for e in log.events if e["event"] == "propose") == 10
    assert sum(len(e["record"]["proposals"]) for e in log.events if e["event"] == "iteration_end") == 11


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
TOKEN_REFUSAL = "missing or wrong token"
LENGTH_REFUSAL = "send the body with a Content-Length"
WRONG_TOKEN = {"X-Evals-Token": "wrong"}
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

    def connect(self) -> HTTPConnection:
        return HTTPConnection(self.host, self.port, timeout=5)

    def send(self, method: str, path: str, *, body: Any = None, headers: dict | None = None, conn=None) -> Reply:
        """Send one request, on `conn` when a test needs to keep the connection.

        A `body` that is not bytes goes as JSON. A header set to None is left out and `headers` wins over
        the computed `Content-Length`, so a test can omit the token or lie about the body.
        """
        payload = body if isinstance(body, bytes | None) else json.dumps(body).encode()
        sent: dict[str, str | None] = {"Host": f"127.0.0.1:{self.port}", "X-Evals-Token": self.token}
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
        ("/api/runs", "guessed"),
        ("/api/runs?t=guessed", None),
        ("/api/runs?t=%C3%A9", None),
        ("/api/runs", None),
        ("/api/changes", None),
    ],
    ids=["wrong header", "wrong query", "not ASCII", "none", "none, asking what changed"],
)
def test_api_refuses_a_request_without_the_token(client: Client, path, token):
    assert client.send("GET", path, headers={"X-Evals-Token": token}) == _refused(
        HTTPStatus.UNAUTHORIZED, TOKEN_REFUSAL
    )


def test_api_takes_the_token_from_the_query_too(client: Client):
    reply = client.send("GET", f"/api/config?t={client.token}", headers={"X-Evals-Token": None})

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
    assert client.send("GET", "/", headers={"X-Evals-Token": None}) == Reply(
        HTTPStatus.UNAUTHORIZED, UNAUTHORIZED_PAGE.decode()
    )
    assert client.send("GET", "/") == Reply(HTTPStatus.OK, PAGE)


def test_the_printed_link_trades_its_token_for_a_cookie_and_a_clean_address(client: Client):
    cookie = f"strands_evals_ui_{client.port}={client.token}"

    redirect = client.send("GET", f"/?t={client.token}", headers={"X-Evals-Token": None})
    page = client.send("GET", "/", headers={"X-Evals-Token": None, "Cookie": cookie})

    assert redirect == Reply(HTTPStatus.SEE_OTHER)
    assert redirect.headers["Location"] == "/"
    assert redirect.headers["Set-Cookie"] == f"{cookie}; Path=/; HttpOnly; SameSite=Lax"
    assert page == Reply(HTTPStatus.OK, PAGE)


@pytest.mark.parametrize(
    "cookie",
    ["strands_evals_ui_{port}=wrong", "strands_evals_ui_1={token}", 'strands_evals_ui_{port}="\\012"'],
    ids=["wrong value", "another port's cookie", "malformed"],
)
def test_the_page_is_refused_without_this_servers_cookie(client: Client, cookie):
    sent = cookie.format(port=client.port, token=client.token)

    reply = client.send("GET", "/", headers={"X-Evals-Token": None, "Cookie": sent})

    assert reply == Reply(HTTPStatus.UNAUTHORIZED, UNAUTHORIZED_PAGE.decode())


@pytest.mark.parametrize(
    "path",
    ["/runs", "/runs/support/support.experiment.json/report", "/optimization/tuning/run-1/candidates/c8/report"],
)
def test_every_address_of_the_page_serves_it_and_survives_the_token_redirect(client: Client, path):
    cookie = f"strands_evals_ui_{client.port}={client.token}"

    redirect = client.send("GET", f"{path}?t={client.token}", headers={"X-Evals-Token": None})

    assert redirect.status == HTTPStatus.SEE_OTHER
    assert redirect.headers["Location"] == path
    assert client.send("GET", path, headers={"X-Evals-Token": None, "Cookie": cookie}) == Reply(HTTPStatus.OK, PAGE)
    assert client.send("GET", path, headers={"X-Evals-Token": None}).status == HTTPStatus.UNAUTHORIZED


@pytest.mark.parametrize("path", ["/runsheet", "/optimizations", "/favicon.ico", "/api"])
def test_an_address_that_is_not_the_pages_is_not_found(client: Client, path):
    assert client.send("GET", path) == _refused(HTTPStatus.NOT_FOUND, f"no route for {path}")


def test_the_token_redirect_never_points_off_the_server(client: Client):
    redirect = client.send("GET", f"//elsewhere.example/runs?t={client.token}", headers={"X-Evals-Token": None})

    # Refused outright where the standard library collapses the leading slashes, and a local path where it does not.
    assert redirect.headers.get("Location") in (None, "/runs")


def test_the_cookie_opens_the_page_but_not_the_api(client: Client):
    cookie = f"strands_evals_ui_{client.port}={client.token}"

    reply = client.send("GET", "/api/config", headers={"X-Evals-Token": None, "Cookie": cookie})

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
    scanned = client.send("GET", "/api/runs").body["fingerprint"]
    assert client.send("GET", "/api/changes") == Reply(HTTPStatus.OK, {"fingerprint": scanned})

    _write(client.root / "report.json", _report("alpha"))

    assert client.send("GET", "/api/changes").body["fingerprint"] != scanned


# ── the server: reading ──


def test_runs_endpoint_serves_the_discovery_with_the_settings(client: Client):
    _optimizer_run(client.root / "opt", {"event": "config"}, {"event": "stop", "reason": "budget"})
    (client.root / "broken.json").write_text("{not json")

    reply = client.send("GET", "/api/runs")

    found = discover(client.root)
    assert [len(found.runs), len(found.optimizer_runs), len(found.skipped)] == [1, 1, 1]
    assert reply == Reply(HTTPStatus.OK, {**client.settings, "fingerprint": fingerprint(client.root), **asdict(found)})


def test_runs_endpoint_survives_a_report_holding_nan(client: Client):
    (client.root / "odd").mkdir()
    (client.root / "odd" / "report.json").write_text(json.dumps({**_report("alpha"), "overall_score": float("nan")}))

    reply = client.send("GET", "/api/runs")

    assert reply.status == HTTPStatus.OK
    assert {run["id"]: run["report"]["overall_score"] for run in reply.body["runs"]} == {
        "experiment.json": 0.5,
        "odd/report.json": None,
    }


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
    ],
    ids=["outside the root", "null byte", "no path", "missing file"],
)
def test_reading_refuses_a_path_it_cannot_serve(client: Client, route, query, status, message):
    assert client.send("GET", route + query) == _refused(status, message.replace("{root}", client.settings["root"]))


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_an_unknown_route_is_not_found(client: Client, method):
    reply = client.send(method, "/api/nothing")

    assert reply.status == HTTPStatus.NOT_FOUND
    assert reply.body == ({"error": "no route for /api/nothing"} if method == "GET" else "")


# ── the server: saving ──


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


# ── the command ──


def _list_tree(root: Path) -> None:
    _write(root / "experiment.json", _experiment("alpha"))
    _write(root / "report.json", _report("alpha"))
    _write(root / "notes.json", {"hello": "world"})
    _optimizer_run(
        root / "tuning",
        {"event": "propose", "iteration": 0, "candidate_id": "c0"},
        {"event": "iteration_end", "iteration": 0},
        {"event": "picked", "iteration": 0, "report": {"champion_id": "c0"}},
        {"event": "stop", "iteration": 0, "reason": "budget"},
    )
    _optimizer_run(root / "tuning" / "unfinished")


def test_ui_list_prints_runs_and_optimizer_runs_as_json(tmp_path: Path, capsys):
    _list_tree(tmp_path)

    assert main(["--json", "ui", str(tmp_path), "--list"]) == 0

    found = discover(tmp_path)
    assert [len(found.runs), len(found.optimizer_runs), len(found.skipped)] == [1, 2, 1]
    assert json.loads(capsys.readouterr().out) == {
        "root": str(tmp_path.resolve()),
        "runs": found.runs,
        "optimizer_runs": found.optimizer_runs,
        "skipped": found.skipped,
    }


def test_ui_list_prints_runs_and_optimizer_runs_as_text(tmp_path: Path, capsys):
    _list_tree(tmp_path)

    # `--rich` because stdout is not a TTY under capsys, which would otherwise pick JSON.
    assert main(["--rich", "ui", str(tmp_path), "--list"]) == 0

    printed = capsys.readouterr()
    assert sorted(printed.out.splitlines()) == [
        "experiment.json (1 cases)  ->  report.json (1.00)",
        "optimizer run tuning (1 candidates, 1 iterations)  ->  champion c0, stopped: budget",
        "optimizer run tuning/unfinished (0 candidates, 0 iterations)  ->  no champion",
    ]
    assert printed.err == f"skipped notes.json: {NOT_RECOGNISED}\n"


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
        *notes,
    ]


@pytest.mark.parametrize("flags,opened", [([], [FAKE_URL]), (["--no-browser"], [])], ids=["default", "--no-browser"])
def test_ui_opens_the_printed_link_unless_told_not_to(tmp_path: Path, launched, flags, opened):
    assert main(["ui", str(tmp_path), *flags]) == 0
    assert launched["opened"] == opened
