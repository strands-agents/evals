import json

import pytest

from strands_evals.cli._optimize.apply import apply_candidate
from strands_evals.cli._optimize.record import RunDirectory, write_json

SPACE = [
    {"key": "instructions", "kind": "text", "source": "instructions.md"},
    {"key": "tone", "kind": "text", "source": None},
    {"key": "turns", "kind": "int", "source": None},
]
BASELINE = {"instructions": "old\n", "tone": "warm", "turns": 3}
BETTER = {"instructions": "new\n", "tone": "cool", "turns": 5}


@pytest.fixture
def run(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "instructions.md").write_text("old\n")
    run = RunDirectory.create(
        tmp_path / "runs", {"schema_version": 1, "workdir": str(project), "space": SPACE}, name=None
    )
    run.events.append("propose", 0, candidate_id="base", origin="baseline", values=BASELINE)
    run.events.append("propose", 1, candidate_id="better", origin="file", values=BETTER)
    return run


def pick(run, candidate_id):
    run.events.append("selected", 1, report={"champion_id": candidate_id})


def test_applies_the_pick_to_its_files_and_reports_the_other_changes(run, tmp_path):
    pick(run, "better")

    result = apply_candidate(run.root, None, dry_run=False)

    assert (tmp_path / "project" / "instructions.md").read_text() == "new\n"
    assert result["files"] == [str(tmp_path / "project" / "instructions.md")]
    assert result["manual"] == {"tone": {"old": "warm", "new": "cool"}, "turns": {"old": 3, "new": 5}}
    assert apply_candidate(run.root, None, dry_run=False)["files"] == []


def test_dry_run_lists_the_files_without_writing(run, tmp_path):
    pick(run, "better")

    result = apply_candidate(run.root, None, dry_run=True)

    assert result["dry_run"] is True and len(result["files"]) == 1
    assert (tmp_path / "project" / "instructions.md").read_text() == "old\n"


def test_refuses_a_file_that_changed_since_the_run_started(run, tmp_path):
    pick(run, "better")
    (tmp_path / "project" / "instructions.md").write_text("edited by hand\n")

    with pytest.raises(ValueError, match="changed since the run started"):
        apply_candidate(run.root, None, dry_run=False)


def test_needs_a_pick_or_a_known_candidate(run):
    with pytest.raises(ValueError, match="no final pick yet"):
        apply_candidate(run.root, None, dry_run=False)
    with pytest.raises(ValueError, match="unknown candidate"):
        apply_candidate(run.root, "missing", dry_run=False)

    assert apply_candidate(run.root, "better", dry_run=True)["candidate_id"] == "better"


def test_refuses_a_path_outside_the_project(run):
    pick(run, "better")
    config = json.loads((run.root / "run.json").read_text())
    config["space"][0]["source"] = "../escape.md"
    write_json(run.root / "run.json", config)

    with pytest.raises(ValueError, match="outside the project"):
        apply_candidate(run.root, None, dry_run=False)
