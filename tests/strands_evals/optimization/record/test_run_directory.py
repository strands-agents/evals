import json
import re
from datetime import datetime

import pytest

from strands_evals.optimization.candidate import Candidate
from strands_evals.optimization.record import run_directory as run_directory_module
from strands_evals.optimization.record.run_directory import SCHEMA_VERSION, RunDirectory
from strands_evals.types.evaluation import EvaluationOutput
from strands_evals.types.optimization import AttemptRecord, TrialRecord, TrialStatus, Usage
from strands_evals.types.trace import Session

CONFIG = {"cases": ["a", "b"], "seed": 0}


def _usage(tokens=10):
    return Usage(
        tokens=tokens,
        input_tokens=tokens - 2,
        output_tokens=2,
        cache_read_tokens=None,
        cache_write_tokens=None,
        model_calls=1,
        seconds=0.5,
        models=["m"],
    )


def _trial(candidate_id, case_name="a", run_index=0, *, trajectory=None, score=1.0, previous_attempts=()):
    output = EvaluationOutput(score=score, test_pass=score >= 0.5, reason="ok")
    return TrialRecord(
        candidate_id=candidate_id,
        case_name=case_name,
        run_index=run_index,
        split="train",
        status=TrialStatus.COMPLETED,
        error_type=None,
        error=None,
        attempts=len(previous_attempts) + 1,
        previous_attempts=list(previous_attempts),
        output="out",
        trajectory=trajectory,
        interactions=None,
        environment_state=None,
        evaluations={"Helpfulness": output},
        details={"Helpfulness": [output]},
        evaluator_errors={},
        usage=_usage(),
        judge_usage={"Helpfulness": _usage(3)},
        rollouts=1,
    )


def _baseline():
    return Candidate(values={"x": 1, "prompt": "hi"}, parent_id=None, origin="baseline", rationale="")


def _child(parent, x=2):
    return Candidate(values={"x": x, "prompt": "hi"}, parent_id=parent.id, origin="numbers", rationale="try")


def test_create_writes_run_json_and_the_config_event(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG, name="quickstart")

    assert re.fullmatch(r"quickstart-\d{8}-\d{6}-[0-9a-f]{6}", run.run_id)
    saved = json.loads((run.root / "run.json").read_text())
    assert saved["schema_version"] == SCHEMA_VERSION
    assert set(saved["code"]) == {"version", "commit", "dirty"}
    events = run.events.read()
    assert events[0]["event"] == "config"
    assert events[0]["config"] == saved == run.config


@pytest.mark.parametrize("name", ["Quick", "a b", "a.b", "a/b", ""])
def test_create_rejects_a_bad_name(tmp_path, name):
    with pytest.raises(ValueError, match="lowercase letters"):
        RunDirectory.create(tmp_path, CONFIG, name=name)


def test_two_runs_created_in_the_same_second_get_different_folders(tmp_path, monkeypatch):
    class Frozen(datetime):  # both runs start in one second, however slow the machine
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 10, 12, 0, 0, tzinfo=tz)

    draws = iter(["aaaaaa", "aaaaaa", "bbbbbb"])
    monkeypatch.setattr(run_directory_module, "datetime", Frozen)
    monkeypatch.setattr(run_directory_module.secrets, "token_hex", lambda _: next(draws))

    first = RunDirectory.create(tmp_path, CONFIG)
    second = RunDirectory.create(tmp_path, CONFIG)

    assert first.run_id == "20261110-120000-aaaaaa"
    assert second.run_id == "20261110-120000-bbbbbb"  # the second draw collided, so it drew again


def test_open_rejects_a_folder_that_is_not_a_run(tmp_path):
    (tmp_path / "events.jsonl").write_text(json.dumps({"event": "trial"}) + "\n")

    with pytest.raises(ValueError, match="not a run directory"):
        RunDirectory.open(tmp_path)


def test_open_rejects_another_schema_version(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    lines = (run.root / "events.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["config"]["schema_version"] = SCHEMA_VERSION + 1
    (run.root / "events.jsonl").write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")

    with pytest.raises(ValueError, match="schema_version"):
        RunDirectory.open(run.root)


def test_candidates_get_display_numbers_and_reload_with_the_same_id(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    baseline, child = _baseline(), _child(_baseline())

    assert run.save_candidate(baseline) == 0
    assert run.save_candidate(child, changed_keys=["x"]) == 1

    reopened = RunDirectory.open(run.root)
    assert reopened.number(child.id) == 1
    loaded = reopened.load_candidate(child.id)
    assert loaded == child
    assert loaded.values == child.values
    assert loaded.parent_id == baseline.id
    with pytest.raises(KeyError):
        reopened.number("missing")


def test_a_candidate_proposed_again_is_a_rescore_with_its_first_number(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    baseline = _baseline()
    run.save_candidate(baseline)
    run.events.append("iteration_end", iteration=1, record={}, exhausted=[])

    assert run.save_candidate(baseline) == 0

    rescore = run.events.read()[-1]
    assert rescore["rescore"] is True
    assert "values" not in rescore
    assert RunDirectory.open(run.root).load_candidate(baseline.id) == baseline


def test_a_killed_iterations_candidate_proposed_again_is_a_full_propose_with_its_old_number(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    baseline = _baseline()
    run.save_candidate(baseline)
    run.events.append("iteration_end", iteration=1, record={}, exhausted=[])
    killed = _child(baseline)
    run.save_candidate(killed)  # iteration 2, killed before its iteration_end

    reopened = RunDirectory.open(run.root)
    reopened.events.append("resume", spent=0, budget={}, code={})
    again = Candidate(values=killed.values, parent_id=baseline.id, origin="numbers", rationale="second try")
    assert reopened.save_candidate(again, changed_keys=["x"]) == 1

    event = reopened.events.read()[-1]
    assert event["rescore"] is False
    assert event["values"] == killed.values
    assert event["rationale"] == "second try"
    # From then on it is an ordinary candidate: proposing it again is a rescore.
    assert reopened.save_candidate(again) == 1
    assert reopened.events.read()[-1]["rescore"] is True
    assert RunDirectory.open(run.root).load_candidate(killed.id).rationale == "second try"


def test_a_candidate_whose_values_no_longer_hash_to_its_id_fails_to_load(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    baseline = _baseline()
    run.save_candidate(baseline)
    text = (run.root / "events.jsonl").read_text().replace('"x": 1', '"x": 5')
    (run.root / "events.jsonl").write_text(text)

    with pytest.raises(ValueError, match="hash"):
        RunDirectory.open(run.root).load_candidate(baseline.id)


def test_a_trial_round_trips_with_its_session_transcript(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    session = Session(session_id="s1", traces=[])
    record = _trial("c1", trajectory=session)

    run.save_trial(record)

    loaded = RunDirectory.open(run.root).load_trial("c1", 0, "a")
    assert loaded == record
    assert isinstance(loaded.trajectory, Session)
    assert run.load_trial("c1", 1, "a") is None


def test_an_identical_transcript_is_stored_once(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    run.save_trial(_trial("c1", "a", trajectory=["read", "answer"]))
    run.save_trial(_trial("c1", "b", trajectory=["read", "answer"]))
    run.save_trial(_trial("c1", "c", trajectory=None))

    assert len(list((run.root / "transcripts").iterdir())) == 1
    assert RunDirectory.open(run.root).load_trial("c1", 0, "b").trajectory == ["read", "answer"]
    assert run.load_trial("c1", 0, "c").trajectory is None


def test_a_later_trial_event_replaces_the_earlier_one_on_reload(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    first = _trial("c1", score=0.0)
    run.save_trial(first)
    replaced = AttemptRecord(
        status=first.status,
        error_type=None,
        error=None,
        usage=first.usage,
        judge_usage=_usage(3),
        pass_index=0,
        rollouts=1,
    )
    run.save_trial(_trial("c1", score=1.0, previous_attempts=[replaced]))

    loaded = RunDirectory.open(run.root).load_trial("c1", 0, "a")
    assert loaded.evaluations["Helpfulness"].score == 1.0
    assert loaded.attempts == 2


def test_a_trial_event_cut_off_mid_write_counts_as_not_run(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    run.save_trial(_trial("c1", "a"))
    run.save_trial(_trial("c1", "b"))
    path = run.root / "events.jsonl"
    path.write_bytes(path.read_bytes()[:-40])

    reopened = RunDirectory.open(run.root)
    assert reopened.load_trial("c1", 0, "a") is not None
    assert reopened.load_trial("c1", 0, "b") is None


def test_effects_diffs_and_the_result_are_written(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)

    run.save_effects({"effects": [], "reason": "no fit"})
    run.save_diff("best", "--- prompt\n", {"x": {"old": 1, "new": 2}})
    run.save_diff("front/1-abc", "", {})
    run.save_result({"run_id": run.run_id, "split": ("train",)})
    run.save_result({"run_id": run.run_id, "split": ("train", "test")})  # a resumed finished run rewrites it

    assert json.loads((run.root / "result.json").read_text()) == {"run_id": run.run_id, "split": ["train", "test"]}

    assert json.loads((run.root / "effects.json").read_text()) == {"effects": [], "reason": "no fit"}
    assert (run.root / "best.diff").read_text() == "--- prompt\n"
    assert json.loads((run.root / "best.values.json").read_text()) == {"x": {"old": 1, "new": 2}}
    assert json.loads((run.root / "front" / "1-abc.values.json").read_text()) == {}
    assert run.workspace("abc") == run.root / "candidates" / "abc" / "optimization_files"
