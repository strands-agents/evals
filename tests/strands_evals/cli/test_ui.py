"""Tests for ``strands-evals ui``.

The UI's risky logic — deciding which file is an experiment, which report belongs to it, and
whether a write is safe — lives in Python precisely so it can be tested here rather than being
trusted in a browser. These cover the three ways it could be quietly wrong: misidentifying a
file, pairing a report with an experiment it did not come from, and overwriting someone's edit.
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path

import pytest

from strands_evals.cli._ui_server import (
    PathOutsideRoot,
    build_server,
    classify,
    discover,
    discover_optimizer_runs,
    link_status,
    read_event_log,
    resolve_within,
    server_url,
    validate_experiment_payload,
    write_json_atomic,
)
from strands_evals.cli.main import main

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


def _write(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


# ── classification ──


@pytest.mark.parametrize(
    "obj,expected",
    [
        ({"cases": [], "evaluators": []}, "experiment"),
        ({"overall_score": 0.5, "cases": [], "scores": [], "test_passes": []}, "report"),
        ({"cases": []}, None),
        ({"overall_score": True, "cases": [], "scores": [], "test_passes": []}, None),
        ({"name": "something else"}, None),
        ([], None),
        ("a string", None),
    ],
)
def test_classify_goes_by_shape_not_filename(obj, expected):
    assert classify(obj) == expected


def test_scan_names_the_files_it_could_not_use(tmp_path: Path):
    """A folder where the experiment failed to parse must not look like a folder without one."""
    _write(tmp_path / "experiment.json", _experiment("a"))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "unrelated.json").write_text('{"hello": "world"}')

    result = discover(tmp_path)

    assert len(result.runs) == 1
    reasons = {skip["path"]: skip["why"] for skip in result.skipped}
    assert "invalid JSON" in reasons["broken.json"]
    assert reasons["unrelated.json"] == "not an experiment or a report"


def test_discover_pairs_the_default_filenames(tmp_path: Path):
    """`experiment.json` + `report.json` is what the CLI's own --output produces.

    It has to pair on the names alone, not on there happening to be exactly one of each: a
    directory that also holds last week's experiment is normal.
    """
    _write(tmp_path / "experiment.json", _experiment("a"))
    _write(tmp_path / "report.json", _report("a"))
    _write(tmp_path / "experiment.saved.json", _experiment("a"))

    runs = {run["id"]: run for run in discover(tmp_path).runs}

    assert runs["experiment.json"]["report"]["name"] == "report.json"
    assert runs["experiment.json"]["link"]["status"] == "verified"
    assert runs["experiment.saved.json"]["report"] is None


def test_discover_keeps_a_name_that_merely_ends_in_the_token(tmp_path: Path):
    """`myexperiment.json` is not the experiment belonging to `other.report.json`."""
    _write(tmp_path / "myexperiment.json", _experiment("a"))
    _write(tmp_path / "other.report.json", _report("a"))
    _write(tmp_path / "second.json", _experiment("b"))  # defeats the single-each fallback

    runs = {run["id"]: run for run in discover(tmp_path).runs}

    assert runs["myexperiment.json"]["report"] is None


def test_discover_pairs_by_stem_within_a_directory(tmp_path: Path):
    _write(tmp_path / "nightly.experiment.json", _experiment("a"))
    _write(tmp_path / "nightly.report.json", _report("a"))
    _write(tmp_path / "smoke.experiment.json", _experiment("b"))

    runs = {run["id"]: run for run in discover(tmp_path).runs}

    assert runs["nightly.experiment.json"]["report"]["name"] == "nightly.report.json"
    assert runs["smoke.experiment.json"]["report"] is None


def test_discover_does_not_pair_across_directories(tmp_path: Path):
    _write(tmp_path / "a" / "experiment.json", _experiment("a"))
    _write(tmp_path / "b" / "report.json", _report("a"))

    runs = discover(tmp_path).runs

    assert len(runs) == 2
    by_id = {run["id"]: run for run in runs}
    assert by_id["a/experiment.json"]["report"] is None
    assert by_id["b/report.json"]["experiment"] is None


def test_discover_respects_depth(tmp_path: Path):
    _write(tmp_path / "one" / "two" / "three" / "experiment.json", _experiment("deep"))

    assert discover(tmp_path, depth=2).runs == []
    assert len(discover(tmp_path, depth=4).runs) == 1


def test_report_summary_counts_the_way_the_library_does(tmp_path: Path):
    """not_applicable with test_pass=false is a failure to judge, not a skipped row."""
    report = _report("ok", "bad", "skipped", failing=("bad",), declined=("skipped",))
    report["detailed_results"][1] = [{"score": 0.0, "test_pass": False, "label": "not_applicable"}]
    _write(tmp_path / "experiment.json", _experiment("ok", "bad", "skipped"))
    _write(tmp_path / "report.json", report)

    summary = discover(tmp_path).runs[0]["report"]

    assert summary["rows"] == 3
    assert summary["not_applicable"] == 1  # only the one that both declined and passed
    assert summary["failing"] == 1


# ── pairing is checked, not assumed ──


def test_link_status_verified_when_rows_match():
    assert link_status(_experiment("a", "b"), _report("a", "b"))["status"] == "verified"


def test_link_status_mismatch_when_no_row_matches():
    status = link_status(_experiment("a", "b"), _report("x", "y"))
    assert status["status"] == "mismatch"
    assert "none" in status["detail"]


def test_link_status_partial_when_some_rows_are_absent():
    status = link_status(_experiment("a"), _report("a", "b"))
    assert status["status"] == "partial"
    assert "1 of 2" in status["detail"]


def test_link_status_unverified_without_identifiers():
    assert link_status({"cases": [{}]}, {"cases": [{}]})["status"] == "unverified"


def test_discover_reports_a_mismatched_pair_rather_than_hiding_it(tmp_path: Path):
    """Same folder, matching stem, wrong report: the filename guess went wrong."""
    _write(tmp_path / "experiment.json", _experiment("a", "b"))
    _write(tmp_path / "report.json", _report("totally", "different"))

    run = discover(tmp_path).runs[0]

    assert run["report"] is not None
    assert run["link"]["status"] == "mismatch"


# ── optimizer runs ──


def _event_log(path: Path, *events: dict, trailing_newline: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(e) + "\n" for e in events)
    path.write_text(text if trailing_newline else text.rstrip("\n"))
    return path


def _optimizer_run(directory: Path, *events: dict, config: dict | None = None) -> Path:
    _write(directory / "run.json", config or {"run_id": "r-1", "objective": "Maximize(Helpfulness)"})
    _event_log(directory / "events.jsonl", *events)
    return directory


def test_read_event_log_reads_one_event_per_line(tmp_path: Path):
    log = read_event_log(_event_log(tmp_path / "events.jsonl", {"event": "config"}, {"event": "stop"}))

    assert [e["event"] for e in log.events] == ["config", "stop"]
    assert (log.bad_lines, log.truncated) == ([], False)


def test_read_event_log_drops_a_half_written_last_line(tmp_path: Path):
    """A run still appending: the tail is expected to be partial, so it is not an error."""
    path = _event_log(tmp_path / "events.jsonl", {"event": "config"})
    path.write_text(path.read_text() + '{"event": "tri')

    log = read_event_log(path)

    assert [e["event"] for e in log.events] == ["config"]
    assert log.bad_lines == []


def test_read_event_log_reports_a_broken_line_in_the_middle(tmp_path: Path):
    """Same text, but with a complete line after it, so it cannot be a log being written."""
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "config"}\n{"event": "tri\n{"event": "stop"}\n')

    log = read_event_log(path)

    assert [e["event"] for e in log.events] == ["config", "stop"]
    assert [b["line"] for b in log.bad_lines] == [2]


def test_read_event_log_reports_a_line_that_is_not_an_object(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "config"}\n[1, 2, 3]\n')

    log = read_event_log(path)

    assert len(log.events) == 1
    assert log.bad_lines == [{"line": 2, "why": "not a JSON object"}]


def test_read_event_log_stops_at_the_line_cap_and_says_so(tmp_path: Path):
    path = _event_log(tmp_path / "events.jsonl", *({"event": "trial", "n": i} for i in range(10)))

    log = read_event_log(path, max_lines=4)

    assert len(log.events) == 4
    assert log.truncated is True


def test_read_event_log_survives_an_unreadable_file(tmp_path: Path):
    log = read_event_log(tmp_path / "missing.jsonl")

    assert log.events == []
    assert log.bad_lines[0]["line"] == 0


def test_discover_optimizer_runs_summarises_what_the_log_records(tmp_path: Path):
    _optimizer_run(
        tmp_path / "run-a",
        {"event": "config"},
        {"event": "propose", "candidate_id": "c0", "iteration": 0},
        {"event": "trial", "iteration": 0, "record": {"status": "completed"}},
        {"event": "iteration_end", "iteration": 0, "record": {"iteration": 0}},
        {"event": "picked", "iteration": 1, "report": {"champion_id": "c0"}},
        {"event": "stop", "iteration": 1, "reason": "iterations"},
        config={"run_id": "r-7", "name": "tuning", "objective": "Maximize(H)", "budget": {"max_rollouts": 20}},
    )

    runs = discover_optimizer_runs(tmp_path)

    assert len(runs) == 1
    run = runs[0]
    assert (run["dir"], run["run_id"], run["name"]) == ("run-a", "r-7", "tuning")
    assert (run["candidates"], run["trials"], run["iterations"]) == (1, 1, 1)
    assert (run["champion_id"], run["stop_reason"]) == ("c0", "iterations")
    assert run["budget"] == {"max_rollouts": 20}
    assert run["events_path"] == "run-a/events.jsonl"


def test_discover_optimizer_runs_needs_both_files(tmp_path: Path):
    """`run.json` is far too ordinary a name to take as evidence on its own."""
    _write(tmp_path / "config-only" / "run.json", {"run_id": "r-1"})
    _event_log(tmp_path / "log-only" / "events.jsonl", {"event": "config"})

    assert discover_optimizer_runs(tmp_path) == []


def test_discover_optimizer_runs_ignores_a_run_json_that_is_not_an_object(tmp_path: Path):
    (tmp_path / "odd").mkdir()
    (tmp_path / "odd" / "run.json").write_text("[1, 2, 3]")
    _event_log(tmp_path / "odd" / "events.jsonl", {"event": "config"})

    assert discover_optimizer_runs(tmp_path) == []


def test_discover_keeps_a_recognised_run_json_out_of_skipped(tmp_path: Path):
    """It is reported as an optimizer run, so reporting it again as unusable would contradict that."""
    _optimizer_run(tmp_path / "opt", {"event": "config"})
    _write(tmp_path / "elsewhere" / "run.json", {"not": "an optimizer run"})

    result = discover(tmp_path)

    assert [r["dir"] for r in result.optimizer_runs] == ["opt"]
    assert [s["path"] for s in result.skipped] == ["elsewhere/run.json"]


def test_discover_optimizer_runs_respects_depth(tmp_path: Path):
    _optimizer_run(tmp_path / "a" / "b" / "deep", {"event": "config"})

    assert discover_optimizer_runs(tmp_path, depth=2) == []
    assert len(discover_optimizer_runs(tmp_path, depth=3)) == 1


def test_fixture_run_is_readable():
    """The synthetic run the browser tests draw their charts from, read by the real reader."""
    fixture = Path(__file__).parents[2] / "ui" / "fixtures" / "optimizer-run"
    if not fixture.is_dir():  # pragma: no cover - the fixture is committed, this is a clearer failure
        pytest.skip(f"missing fixture {fixture}")

    log = read_event_log(fixture / "events.jsonl")

    assert log.bad_lines == []
    assert log.truncated is False
    assert log.events[0]["event"] == "config"
    # 11 proposals in the plan, but the invalid patch never became a candidate and so never
    # reached `propose` — the page has to get its row from `iteration_end` instead.
    assert sum(1 for e in log.events if e["event"] == "propose") == 10
    assert sum(len(e["record"]["proposals"]) for e in log.events if e["event"] == "iteration_end") == 11


# ── paths ──


def test_resolve_within_allows_a_file_in_the_tree(tmp_path: Path):
    target = _write(tmp_path / "nested" / "experiment.json", _experiment("a"))
    assert resolve_within(tmp_path, "nested/experiment.json") == target.resolve()


@pytest.mark.parametrize("attempt", ["../outside.json", "nested/../../outside.json", "/etc/passwd", ""])
def test_resolve_within_rejects_escapes(tmp_path: Path, attempt: str):
    with pytest.raises(PathOutsideRoot):
        resolve_within(tmp_path, attempt)


def test_resolve_within_rejects_a_symlink_out_of_the_tree(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    _write(outside / "secret.json", {"shh": True})
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.json").symlink_to(outside / "secret.json")

    with pytest.raises(PathOutsideRoot):
        resolve_within(root, "link.json")


# ── writes ──


def test_validate_experiment_payload_rejects_a_case_the_library_would_refuse():
    payload = {"cases": [{"name": "no input"}], "evaluators": []}
    with pytest.raises(ValueError, match="no input"):
        validate_experiment_payload(payload)


def test_validate_experiment_payload_keeps_unknown_evaluators():
    """A custom evaluator subclass is not resolvable here, and rejecting it would be wrong."""
    payload = {
        "cases": [{"name": "a", "input": "hi"}],
        "evaluators": [{"evaluator_type": "MyTeamsOwnEvaluator", "threshold": 0.9}],
    }
    assert validate_experiment_payload(payload)["evaluators"][0]["threshold"] == 0.9


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"evaluators": []}, "cases"),
        ({"cases": []}, "evaluators"),
        ({"cases": ["nope"], "evaluators": []}, "not a JSON object"),
        ({"cases": [], "evaluators": [{"name": "x"}]}, "evaluator_type"),
        ("not a document", "JSON object"),
    ],
)
def test_validate_experiment_payload_shape_errors(payload, message):
    with pytest.raises(ValueError, match=message):
        validate_experiment_payload(payload)


def test_write_json_atomic_leaves_the_old_file_when_serialization_fails(tmp_path: Path):
    target = _write(tmp_path / "experiment.json", _experiment("original"))
    before = target.read_text()

    with pytest.raises(ValueError):
        write_json_atomic(target, {"cases": [{"score": float("nan")}]})

    assert target.read_text() == before
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".experiment")]


def test_write_json_atomic_round_trips_through_the_library(tmp_path: Path):
    from strands_evals.case import Case

    target = tmp_path / "experiment.json"
    document = _experiment("a", "b")
    write_json_atomic(target, document)

    reloaded = json.loads(target.read_text())
    assert [Case(**case).name for case in reloaded["cases"]] == ["a", "b"]


# ── the server ──


@pytest.fixture
def served(tmp_path: Path):
    """A running server over a tree with one paired run, plus a client helper."""
    _write(tmp_path / "experiment.json", _experiment("alpha", "beta"))
    _write(tmp_path / "report.json", _report("alpha", "beta", failing=("beta",)))

    server = build_server(tmp_path, port=0, asset="<html>page</html>")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    token = server.config.token

    def request(method: str, path: str, *, body: dict | None = None, headers: dict | None = None):
        conn = HTTPConnection(host, port, timeout=5)
        payload = json.dumps(body).encode() if body is not None else None
        sent = {"Host": f"127.0.0.1:{port}", "X-Evals-Token": token}
        if payload is not None:
            sent["Content-Type"] = "application/json"
        sent.update(headers or {})
        if headers and "X-Evals-Token" in headers and headers["X-Evals-Token"] is None:
            sent.pop("X-Evals-Token")
        conn.request(method, path, body=payload, headers={k: v for k, v in sent.items() if v is not None})
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        try:
            return response.status, json.loads(raw)
        except json.JSONDecodeError:
            return response.status, raw

    try:
        yield tmp_path, request, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_page_needs_the_token(served):
    _, request, server = served
    status, body = request("GET", "/", headers={"X-Evals-Token": None})
    assert status == HTTPStatus.UNAUTHORIZED
    assert b"Open the link the CLI printed" in body

    status, body = request("GET", f"/?t={server.config.token}", headers={"X-Evals-Token": None})
    assert (status, body) == (HTTPStatus.OK, b"<html>page</html>")


def test_api_rejects_a_wrong_token(served):
    _, request, _ = served
    status, body = request("GET", "/api/runs", headers={"X-Evals-Token": "guessed"})
    assert status == HTTPStatus.UNAUTHORIZED
    assert "token" in body["error"]


def test_api_rejects_a_non_loopback_host_header(served):
    """What stops a DNS-rebinding page from reaching this port as evil.example.com."""
    _, request, _ = served
    status, body = request("GET", "/api/runs", headers={"Host": "evals.evil.example.com"})
    assert status == HTTPStatus.FORBIDDEN
    assert "localhost" in body["error"]


def test_runs_endpoint_describes_the_tree(served):
    _, request, _ = served
    status, body = request("GET", "/api/runs")

    assert status == HTTPStatus.OK
    assert len(body["runs"]) == 1
    run = body["runs"][0]
    assert run["experiment"]["cases"] == 2
    assert run["report"]["failing"] == 1
    assert run["link"]["status"] == "verified"


def test_runs_endpoint_reports_optimizer_runs_alongside_experiments(served):
    root, request, _ = served
    _optimizer_run(root / "opt", {"event": "config"}, {"event": "stop", "reason": "budget"})

    status, body = request("GET", "/api/runs")

    assert status == HTTPStatus.OK
    assert len(body["runs"]) == 1  # the optimizer run is a different kind of thing, not an extra run
    assert [r["dir"] for r in body["optimizer_runs"]] == ["opt"]
    assert body["optimizer_runs"][0]["stop_reason"] == "budget"


def test_events_endpoint_returns_the_log(served):
    root, request, _ = served
    _optimizer_run(root / "opt", {"event": "config"}, {"event": "trial", "record": {"status": "completed"}})

    status, body = request("GET", "/api/events?path=opt/events.jsonl")

    assert status == HTTPStatus.OK
    assert [e["event"] for e in body["events"]] == ["config", "trial"]
    assert (body["bad_lines"], body["truncated"]) == ([], False)


def test_events_endpoint_refuses_to_leave_the_root(served):
    _, request, _ = served
    status, body = request("GET", "/api/events?path=../../etc/passwd")
    assert status == HTTPStatus.FORBIDDEN
    assert "outside" in body["error"]


def test_events_endpoint_reports_a_missing_log(served):
    _, request, _ = served
    status, body = request("GET", "/api/events?path=nowhere/events.jsonl")
    assert status == HTTPStatus.NOT_FOUND
    assert "not a file" in body["error"]


def test_file_endpoint_refuses_to_leave_the_root(served):
    _, request, _ = served
    status, body = request("GET", "/api/file?path=../../etc/passwd")
    assert status == HTTPStatus.FORBIDDEN
    assert "outside" in body["error"]


def test_put_writes_the_file_it_loaded(served):
    root, request, _ = served
    _, loaded = request("GET", "/api/file?path=experiment.json")
    document = loaded["document"]
    document["cases"][0]["expected_output"] = "a new expectation"

    status, body = request(
        "PUT",
        "/api/experiment?path=experiment.json",
        body={"document": document, "modified": loaded["modified"]},
    )

    assert (status, body["cases"]) == (HTTPStatus.OK, 2)
    on_disk = json.loads((root / "experiment.json").read_text())
    assert on_disk["cases"][0]["expected_output"] == "a new expectation"
    assert on_disk["cases"][0]["session_id"] == "sid-alpha"  # identity preserved
    assert on_disk["evaluators"] == document["evaluators"]


def test_put_refuses_a_stale_write(served):
    """Someone else wrote the file after the page loaded it; overwriting would lose their work."""
    root, request, _ = served
    _, loaded = request("GET", "/api/file?path=experiment.json")

    status, body = request(
        "PUT",
        "/api/experiment?path=experiment.json",
        body={"document": loaded["document"], "modified": loaded["modified"] - 60},
    )

    assert status == HTTPStatus.CONFLICT
    assert "changed on disk" in body["error"]
    assert json.loads((root / "experiment.json").read_text())["cases"][0]["name"] == "alpha"


def test_put_rejects_an_invalid_document(served):
    root, request, _ = served
    before = (root / "experiment.json").read_text()

    status, body = request(
        "PUT",
        "/api/experiment?path=experiment.json",
        body={"document": {"cases": [{"name": "broken"}], "evaluators": []}},
    )

    assert status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "broken" in body["error"]
    assert (root / "experiment.json").read_text() == before


def test_put_refuses_to_write_outside_the_root(served):
    _, request, _ = served
    status, _ = request(
        "PUT",
        "/api/experiment?path=../escaped.json",
        body={"document": _experiment("a")},
    )
    assert status == HTTPStatus.FORBIDDEN


def test_read_only_server_refuses_writes(tmp_path: Path):
    _write(tmp_path / "experiment.json", _experiment("a"))
    server = build_server(tmp_path, port=0, read_only=True, asset="<html></html>")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request(
            "PUT",
            "/api/experiment?path=experiment.json",
            body=json.dumps({"document": _experiment("a")}),
            headers={"Host": f"127.0.0.1:{port}", "X-Evals-Token": server.config.token},
        )
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()
        assert response.status == HTTPStatus.FORBIDDEN
        assert "read-only" in body["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_shipped_asset_is_the_real_page():
    from strands_evals.cli._ui_server import load_asset

    page = load_asset()
    assert page.startswith("<!DOCTYPE html>")
    assert 'id="runs-rows"' in page


def test_the_page_is_re_read_per_request(tmp_path: Path, monkeypatch):
    """Else every edit to ui.html needs a restart, which hands out a new token and new URL."""
    from strands_evals.cli import _ui_server

    server = build_server(tmp_path, port=0)
    assert server.config.reload_asset is True

    monkeypatch.setattr(_ui_server, "load_asset", lambda: "<html>edited</html>")
    assert server.config.page() == "<html>edited</html>"
    server.server_close()


def test_a_pinned_asset_is_not_re_read(tmp_path: Path, monkeypatch):
    """The test harness injects a page; re-reading would serve the real one instead."""
    from strands_evals.cli import _ui_server

    server = build_server(tmp_path, port=0, asset="<html>pinned</html>")
    monkeypatch.setattr(_ui_server, "load_asset", lambda: "<html>other</html>")

    assert server.config.reload_asset is False
    assert server.config.page() == "<html>pinned</html>"
    server.server_close()


def test_an_unreadable_page_falls_back_to_the_one_read_at_startup(tmp_path: Path, monkeypatch):
    """A half-written ui.html must not turn the whole server into a 500."""
    from strands_evals.cli import _ui_server

    server = build_server(tmp_path, port=0)
    startup = server.config.asset

    def boom() -> str:
        raise OSError("vanished mid-save")

    monkeypatch.setattr(_ui_server, "load_asset", boom)
    assert server.config.page() == startup
    server.server_close()


# ── the run explorer ──


@pytest.fixture
def explorer(tmp_path: Path):
    """A server with a runs directory and a stand-in bundle, plus a client helper.

    The bundle is faked rather than the shipped one so the assertions are about the routes and
    not about whatever the last `npm run build` happened to emit.
    """
    runs = tmp_path / "runs"
    _optimizer_run(runs / "tuning", {"event": "propose", "candidate": 0})
    (runs / "tuning" / "transcripts").mkdir()
    (runs / "tuning" / "transcripts" / "abc.json").write_text('{"turns": []}', encoding="utf-8")
    (runs / "not-a-run").mkdir()  # no events.jsonl, so not a run
    (runs / "stray.txt").write_text("not a directory either", encoding="utf-8")

    app = tmp_path / "app"
    (app / "assets").mkdir(parents=True)
    (app / "index.html").write_text("<html>explorer</html>", encoding="utf-8")
    (app / "assets" / "index.js").write_text("console.log(1)", encoding="utf-8")
    (app / "assets" / "index.css").write_text("body{}", encoding="utf-8")
    (app / "assets" / "index.js.map").write_text("{}", encoding="utf-8")
    (app / "secret.txt").write_text("above the assets directory", encoding="utf-8")

    served_root = tmp_path / "tree"
    served_root.mkdir()
    server = build_server(served_root, port=0, asset="<html>page</html>", opt_root=runs, opt_app=app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    token = server.config.token

    def request(method: str, path: str, *, token_header: str | None = token):
        conn = HTTPConnection(host, port, timeout=5)
        headers = {"Host": f"127.0.0.1:{port}"}
        if token_header is not None:
            headers["X-Evals-Token"] = token_header
        conn.request(method, path, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        kind = response.getheader("Content-Type")
        conn.close()
        return response.status, raw, kind

    try:
        yield runs, request, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_config_reports_no_optimization_without_a_runs_directory(served):
    _, request, _ = served
    _, body = request("GET", "/api/config")
    assert body["optimization"] is False


def test_config_reports_optimization_when_a_runs_directory_is_served(explorer):
    _, request, _ = explorer
    conn_status, raw, _ = request("GET", "/api/config")
    assert conn_status == HTTPStatus.OK
    assert json.loads(raw)["optimization"] is True


@pytest.mark.parametrize("path", ["/optimization", "/optimization/"])
def test_explorer_page_is_served_with_and_without_the_slash(explorer, path: str):
    _, request, _ = explorer
    status, raw, kind = request("GET", path)
    assert (status, raw) == (HTTPStatus.OK, b"<html>explorer</html>")
    assert kind == "text/html; charset=utf-8"


def test_explorer_page_needs_the_token(explorer):
    """It reads run records through the same API, so it is gated like the main page."""
    _, request, _ = explorer
    status, raw, _ = request("GET", "/optimization/", token_header=None)
    assert status == HTTPStatus.UNAUTHORIZED
    assert b"explorer" not in raw


def test_explorer_page_404s_when_no_bundle_is_installed(tmp_path: Path):
    """A source checkout with no `npm run build` must say so rather than serve an empty page."""
    server = build_server(tmp_path, port=0, asset="x", opt_root=tmp_path, opt_app=tmp_path / "never-built")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/optimization/", headers={"Host": f"127.0.0.1:{port}"})
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        # 404 before the token check: whether a bundle exists is not a secret, and a 401 here
        # would send someone hunting for the right token to a route that has nothing to serve.
        assert response.status == HTTPStatus.NOT_FOUND
        assert b"no optimization app is installed" in raw
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("name", "kind"),
    [("index.js", "text/javascript"), ("index.css", "text/css"), ("index.js.map", "application/json")],
)
def test_bundle_assets_are_served_untokened(explorer, name: str, kind: str):
    """A <script src> carries no header, and the bundle is code rather than anyone's data."""
    _, request, _ = explorer
    status, raw, served_kind = request("GET", f"/optimization/assets/{name}", token_header=None)
    assert status == HTTPStatus.OK
    assert raw
    assert served_kind == f"{kind}; charset=utf-8"


def test_an_unknown_asset_suffix_is_served_as_bytes(explorer):
    """Guessing a type is worse than declining to: a wrong one on a script makes it not run."""
    runs, request, server = explorer
    assert server.config.opt_app is not None
    (server.config.opt_app / "assets" / "font.woff2").write_bytes(b"\x00\x01")
    status, raw, kind = request("GET", "/optimization/assets/font.woff2")
    assert (status, raw) == (HTTPStatus.OK, b"\x00\x01")
    assert kind is not None and kind.startswith("application/octet-stream")


def test_a_missing_asset_404s(explorer):
    _, request, _ = explorer
    status, raw, _ = request("GET", "/optimization/assets/nope.js")
    assert status == HTTPStatus.NOT_FOUND
    assert b"no such asset" in raw


def test_the_asset_route_cannot_leave_the_assets_directory(explorer):
    """Untokened, so this is the one route a stranger on loopback can reach at all."""
    _, request, _ = explorer
    status, raw, _ = request("GET", "/optimization/assets/../secret.txt", token_header=None)
    assert status == HTTPStatus.FORBIDDEN
    assert b"above the assets directory" not in raw


def test_optimization_runs_lists_only_directories_holding_a_log(explorer):
    _, request, _ = explorer
    status, raw, _ = request("GET", "/api/optimization/runs")
    assert status == HTTPStatus.OK
    body = json.loads(raw)
    assert [run["id"] for run in body["runs"]] == ["tuning"]
    assert body["runs"][0]["modified"] > 0


def test_optimization_runs_404s_when_none_was_configured(served):
    _, request, _ = served
    status, body = request("GET", "/api/optimization/runs")
    assert status == HTTPStatus.NOT_FOUND
    assert "--runs-dir" in body["error"]


def test_optimization_file_returns_the_bytes_on_disk(explorer):
    """The explorer parses the record itself, so the route hands it over unchanged."""
    _, request, _ = explorer
    status, raw, kind = request("GET", "/api/optimization/file?run=tuning&path=events.jsonl")
    assert status == HTTPStatus.OK
    assert json.loads(raw.splitlines()[0])["event"] == "propose"
    assert kind == "text/plain; charset=utf-8"


def test_optimization_file_reads_below_the_run(explorer):
    _, request, _ = explorer
    status, raw, _ = request("GET", "/api/optimization/file?run=tuning&path=transcripts/abc.json")
    assert (status, raw) == (HTTPStatus.OK, b'{"turns": []}')


def test_optimization_file_404s_when_not_configured(served):
    _, request, _ = served
    status, body = request("GET", "/api/optimization/file?run=x&path=y")
    assert status == HTTPStatus.NOT_FOUND
    assert "--runs-dir" in body["error"]


def test_optimization_file_404s_on_a_missing_file(explorer):
    _, request, _ = explorer
    status, raw, _ = request("GET", "/api/optimization/file?run=tuning&path=best.diff")
    assert status == HTTPStatus.NOT_FOUND
    assert b"tuning/best.diff" in raw


@pytest.mark.parametrize(
    ("run", "rel"),
    [
        ("tuning", "../../outside.json"),
        ("..", "outside.json"),
        ("tuning", "/etc/passwd"),
        ("", "events.jsonl"),
        ("tuning", ""),
    ],
)
def test_optimization_file_refuses_to_leave_the_run(explorer, run: str, rel: str):
    runs, request, _ = explorer
    (runs.parent / "outside.json").write_text('{"secret": true}', encoding="utf-8")
    status, raw, _ = request("GET", f"/api/optimization/file?run={run}&path={rel}")
    assert status in (HTTPStatus.FORBIDDEN, HTTPStatus.NOT_FOUND)
    assert b"secret" not in raw


def test_the_shipped_bundle_is_found_by_default(tmp_path: Path):
    """`--runs-dir` alone has to be enough; nobody should have to point at the build output."""
    from strands_evals.cli._ui_server import _default_opt_app

    app = _default_opt_app()
    if app is None:  # pragma: no cover - the bundle is committed, this is a clearer failure
        pytest.skip("no optimization bundle is built into the package")
    assert (app / "index.html").is_file()

    server = build_server(tmp_path, port=0, asset="x")
    assert server.config.opt_app == app
    server.server_close()


# ── the command ──


def test_ui_token_defaults_to_a_fresh_random_one(tmp_path: Path):
    """Two servers over the same tree must not share a token."""
    first = build_server(tmp_path, port=0)
    second = build_server(tmp_path, port=0)
    try:
        assert first.config.token != second.config.token
        assert len(first.config.token) >= 32
    finally:
        first.server_close()
        second.server_close()


def test_ui_accepts_a_supplied_token(tmp_path: Path):
    """What makes the URL bookmarkable: it is the same after a restart."""
    server = build_server(tmp_path, port=0, token="chosen-token")
    try:
        assert server.config.token == "chosen-token"
        assert server_url(server).endswith("/?t=chosen-token")
    finally:
        server.server_close()


def test_ui_list_prints_the_discovered_runs(tmp_path: Path, capsys):
    _write(tmp_path / "experiment.json", _experiment("alpha"))
    _write(tmp_path / "report.json", _report("alpha"))

    exit_code = main(["--json", "ui", str(tmp_path), "--list"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"][0]["experiment"]["path"] == "experiment.json"
    assert payload["runs"][0]["link"]["status"] == "verified"


@pytest.fixture
def launched(monkeypatch):
    """Run `strands-evals ui` without serving: capture what it asked `build_server` for."""
    from strands_evals.cli.commands import ui as ui_command

    seen: dict = {}

    class _Fake:
        config = type("C", (), {"token": "fake-token"})()
        server_address = ("127.0.0.1", 9999)

        def serve_forever(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

        def server_close(self) -> None:
            return None

    def fake_build(root, **kwargs):
        seen.update(kwargs)
        seen["root"] = root
        return _Fake()

    monkeypatch.setattr(ui_command, "build_server", fake_build)
    monkeypatch.delenv(ui_command.TOKEN_ENV, raising=False)
    return seen


def test_ui_passes_no_token_by_default(tmp_path: Path, launched):
    assert main(["ui", str(tmp_path), "--no-browser"]) == 0
    assert launched["token"] is None, "a default run must let the server generate its own"


def test_ui_passes_the_token_flag_through(tmp_path: Path, launched):
    assert main(["ui", str(tmp_path), "--no-browser", "--token", "bookmarkable"]) == 0
    assert launched["token"] == "bookmarkable"


def test_ui_reads_the_token_from_the_environment(tmp_path: Path, launched, monkeypatch):
    """So it can be set once in a shell profile and never typed again."""
    from strands_evals.cli.commands import ui as ui_command

    monkeypatch.setenv(ui_command.TOKEN_ENV, "from-the-env")
    assert main(["ui", str(tmp_path), "--no-browser"]) == 0
    assert launched["token"] == "from-the-env"


def test_the_token_flag_beats_the_environment(tmp_path: Path, launched, monkeypatch):
    from strands_evals.cli.commands import ui as ui_command

    monkeypatch.setenv(ui_command.TOKEN_ENV, "from-the-env")
    assert main(["ui", str(tmp_path), "--no-browser", "--token", "explicit"]) == 0
    assert launched["token"] == "explicit"


def test_an_empty_token_is_not_a_token(tmp_path: Path, launched, monkeypatch):
    """`STRANDS_EVALS_UI_TOKEN=` unset in a profile must not authorize everything."""
    from strands_evals.cli.commands import ui as ui_command

    monkeypatch.setenv(ui_command.TOKEN_ENV, "")
    assert main(["ui", str(tmp_path), "--no-browser"]) == 0
    assert launched["token"] is None


def test_a_supplied_token_is_called_out_as_guessable(tmp_path: Path, launched, capsys):
    assert main(["ui", str(tmp_path), "--no-browser", "--token", "short"]) == 0
    err = capsys.readouterr().err
    assert "supplied, not generated" in err
    assert "read and write files" in err


def test_a_supplied_token_read_only_says_only_read(tmp_path: Path, launched, capsys):
    assert main(["ui", str(tmp_path), "--no-browser", "--read-only", "--token", "short"]) == 0
    err = capsys.readouterr().err
    assert "can read files" in err and "read and write" not in err


def test_ui_serves_no_runs_directory_by_default(tmp_path: Path, launched):
    assert main(["ui", str(tmp_path), "--no-browser"]) == 0
    assert launched["opt_root"] is None


def test_ui_passes_the_runs_directory_through(tmp_path: Path, launched):
    runs = tmp_path / "runs"
    runs.mkdir()
    assert main(["ui", str(tmp_path), "--no-browser", "--runs-dir", str(runs)]) == 0
    assert launched["opt_root"] == runs


def test_ui_prints_the_explorer_url(tmp_path: Path, launched, capsys):
    """The route is not guessable from the printed link, so it is printed too."""
    runs = tmp_path / "runs"
    runs.mkdir()
    assert main(["ui", str(tmp_path), "--no-browser", "--runs-dir", str(runs)]) == 0
    err = capsys.readouterr().err
    assert "/optimization/?t=fake-token" in err
    assert str(runs.resolve()) in err


def test_ui_rejects_a_runs_directory_that_is_not_one(tmp_path: Path, launched, capsys):
    """Caught before binding: a server that is up with a dead tab is the confusing outcome."""
    stray = tmp_path / "events.jsonl"
    stray.write_text("{}", encoding="utf-8")
    assert main(["ui", str(tmp_path), "--no-browser", "--runs-dir", str(stray)]) == 2
    assert "not a directory" in capsys.readouterr().err
    assert not launched, "it must not have reached build_server"


def test_ui_rejects_a_path_that_is_not_a_directory(tmp_path: Path, capsys):
    target = _write(tmp_path / "experiment.json", _experiment("a"))
    assert main(["ui", str(target), "--list"]) == 2
    assert "not a directory" in capsys.readouterr().err
