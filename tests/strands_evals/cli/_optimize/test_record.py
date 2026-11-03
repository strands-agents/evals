import json
import re
from datetime import datetime, timezone

import pytest

from strands_evals.cli._optimize.record import EventLog, RunDirectory, check_name, new_run_id


def test_new_run_id_uses_the_utc_start_time_and_an_optional_prefix():
    now = datetime(2026, 11, 3, 12, 30, 45, tzinfo=timezone.utc)

    assert re.fullmatch(r"20261103-123045-[0-9a-f]{6}", new_run_id(None, now))
    assert re.fullmatch(r"demo-20261103-123045-[0-9a-f]{6}", new_run_id("demo", now))


@pytest.mark.parametrize("name", ["Demo", "a b", "a.b"])
def test_check_name_rejects_other_characters(name):
    with pytest.raises(ValueError, match="--name"):
        check_name(name)


def test_create_writes_run_json_and_the_config_event_first(tmp_path):
    run = RunDirectory.create(tmp_path / "runs", {"schema_version": 1, "seed": 4}, name="demo")

    assert run.run_id.startswith("demo-")
    assert run.config() == {"schema_version": 1, "seed": 4, "run_id": run.run_id}
    first = run.events.read()[0]
    assert first["event"] == "config" and first["iteration"] == 0 and first["seed"] == 4


def test_read_drops_a_truncated_last_line_and_the_tail_is_cut_before_appending(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.append("config", 0, seed=1)
    with log.path.open("a") as stream:
        stream.write('{"event": "tri')

    assert [event["event"] for event in log.read()] == ["config"]

    log.drop_truncated_tail()
    log.append("stop", 2, reason="budget")
    assert [event["event"] for event in log.read()] == ["config", "stop"]


def test_read_rejects_a_corrupt_middle_line(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "config"}\nnot json\n{"event": "stop"}\n')

    with pytest.raises(ValueError, match="line 2"):
        EventLog(path).read()


def test_transcripts_are_stored_once_by_content_hash(tmp_path):
    run = RunDirectory.create(tmp_path, {"schema_version": 1}, name=None)
    trajectory = [{"role": "assistant", "content": "hi"}]

    first = run.save_transcript(trajectory)
    second = run.save_transcript(list(trajectory))

    assert first == second and first is not None
    assert len(list((run.root / "transcripts").iterdir())) == 1
    assert run.load_transcript(first) == trajectory
    assert run.save_transcript(None) is None


def test_open_rejects_a_folder_without_a_run(tmp_path):
    with pytest.raises(ValueError, match="not an optimization run directory"):
        RunDirectory.open(tmp_path)


def test_config_rejects_another_schema_version(tmp_path):
    run = RunDirectory.create(tmp_path, {"schema_version": 1}, name=None)
    (run.root / "run.json").write_text(json.dumps({"schema_version": 99}))

    with pytest.raises(ValueError, match="unsupported run.json schema"):
        run.config()
