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
    link_status,
    resolve_within,
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


# ── the command ──


def test_ui_list_prints_the_discovered_runs(tmp_path: Path, capsys):
    _write(tmp_path / "experiment.json", _experiment("alpha"))
    _write(tmp_path / "report.json", _report("alpha"))

    exit_code = main(["--json", "ui", str(tmp_path), "--list"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"][0]["experiment"]["path"] == "experiment.json"
    assert payload["runs"][0]["link"]["status"] == "verified"


def test_ui_rejects_a_path_that_is_not_a_directory(tmp_path: Path, capsys):
    target = _write(tmp_path / "experiment.json", _experiment("a"))
    assert main(["ui", str(target), "--list"]) == 2
    assert "not a directory" in capsys.readouterr().err
