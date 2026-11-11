import json
import math

import pytest

from strands_evals.optimization.record.events import EventLog, closed_events
from strands_evals.types.optimization import Patch


def test_append_and_read_round_trip(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.append("config", iteration=0, config={"seed": 0})
    log.append("select", iteration=1, parent_id="a", strategy="top_ranked")
    log.append("trial", record={"case_name": "c"})

    events = EventLog(tmp_path / "events.jsonl").read()

    assert [e["event"] for e in events] == ["config", "select", "trial"]
    assert events[1]["parent_id"] == "a"
    # A trial saved during a step carries the step's iteration.
    assert events[2]["iteration"] == 1
    assert all("time" in e for e in events)


def test_pydantic_records_and_tuples_are_written_as_json(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.append("observe", block="b", observation=Patch(updates={"x": 1}, rationale="r"), round=(2, 0))

    event = log.read()[0]

    assert event["observation"] == {"updates": {"x": 1}, "rationale": "r", "state": None, "usage": None}
    assert event["round"] == [2, 0]


def test_a_field_that_is_not_json_raises(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")

    with pytest.raises(TypeError, match="object"):
        log.append("stop", reason=object())
    with pytest.raises(ValueError):
        log.append("stop", reason=math.nan)
    with pytest.raises(ValueError, match="'time'"):
        log.append("stop", reason="budget", time="2026-01-01")
    assert log.read() == []


def test_a_truncated_last_line_is_dropped_and_cut_on_reopen(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append("config", config={})
    log.append("select", parent_id="a")
    path.write_bytes(path.read_bytes()[:-10])  # killed mid-write

    assert [e["event"] for e in EventLog(path).read()] == ["config"]

    reopened = EventLog(path)
    reopened.append("select", parent_id="b")
    events = reopened.read()
    assert [e.get("parent_id") for e in events] == [None, "b"]


def test_an_unreadable_complete_last_line_is_dropped(tmp_path):
    path = tmp_path / "events.jsonl"
    EventLog(path).append("config", config={})
    with path.open("a") as file:
        file.write("not json\n")

    assert [e["event"] for e in EventLog(path).read()] == ["config"]


def test_a_bad_line_before_the_last_means_a_corrupt_log(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"event": "config"}) + "\nnot json\n" + json.dumps({"event": "stop"}) + "\n")

    with pytest.raises(ValueError, match="line 2 is not a JSON event"):
        EventLog(path)


def _events(*names):
    return [{"event": name, "n": i} for i, name in enumerate(names)]


def test_closed_events_keep_closed_iterations_and_drop_a_killed_one():
    events = _events("config", "propose", "trial", "iteration_end", "select", "propose", "trial")

    kept = closed_events(events)

    # The killed iteration's select and propose go; its trial stays (the cache).
    assert [e["n"] for e in kept] == [0, 1, 2, 3, 6]


def test_closed_events_drop_a_killed_iteration_at_the_next_resume():
    events = _events("config", "iteration_end", "select", "propose", "resume", "select", "propose", "iteration_end")

    kept = closed_events(events)

    assert [e["n"] for e in kept] == [0, 1, 4, 5, 6, 7]


def test_closed_events_keep_the_iteration_a_stop_ends():
    events = _events("config", "iteration_end", "select", "propose", "stop", "finalists", "label", "resume", "picked")

    kept = closed_events(events)

    assert [e["n"] for e in kept] == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_closed_events_without_a_closed_iteration_keep_only_the_setup_outside_iterations():
    events = _events("config", "propose", "score", "trial")

    assert [e["n"] for e in closed_events(events)] == [0, 3]
