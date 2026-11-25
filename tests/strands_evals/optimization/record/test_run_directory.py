import errno
import json
import os
import re
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone

import pytest

from strands_evals.optimization.candidate import Candidate
from strands_evals.optimization.record import run_directory as run_directory_module
from strands_evals.optimization.record.run_directory import SCHEMA_VERSION, RunDirectory
from strands_evals.types.evaluation import EvaluationOutput
from strands_evals.types.optimization import AttemptRecord, TrialRecord, TrialStatus, Usage
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolCallContent,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    ToolResultContent,
    Trace,
    UserMessage,
)

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
    return Candidate(values={"x": 1, "prompt": "hi"}, parent_id=None, origin=("baseline",), rationale="")


def _child(parent, x=2):
    return Candidate(values={"x": x, "prompt": "hi"}, parent_id=parent.id, origin=("numbers",), rationale="try")


def test_create_writes_run_json_and_the_config_event(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG, name="quickstart")

    assert re.fullmatch(r"quickstart-\d{8}-\d{6}-[0-9a-f]{6}", run.run_id)
    saved = json.loads((run.root / "run.json").read_text())
    assert saved["schema_version"] == SCHEMA_VERSION
    assert set(saved["code"]) == {"version", "commit", "dirty", "numpy", "optuna"}
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
    again = Candidate(values=killed.values, parent_id=baseline.id, origin=("numbers",), rationale="second try")
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


def test_trial_status_reads_the_index_without_the_transcript(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    run.save_trial(_trial("c1", trajectory=["read", "answer"]))
    reopened = RunDirectory.open(run.root)
    for path in (run.root / "transcripts").iterdir():
        path.unlink()

    assert reopened.trial_status("c1", 0, "a") is TrialStatus.COMPLETED
    assert reopened.trial_status("c1", 1, "a") is None


def test_an_identical_transcript_is_stored_once(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    run.save_trial(_trial("c1", "a", trajectory=["read", "answer"]))
    run.save_trial(_trial("c1", "b", trajectory=["read", "answer"]))
    run.save_trial(_trial("c1", "c", trajectory=None))

    assert len(list((run.root / "transcripts").iterdir())) == 1
    assert RunDirectory.open(run.root).load_trial("c1", 0, "b").trajectory == ["read", "answer"]
    assert run.load_trial("c1", 0, "c").trajectory is None


def _conversation(turns, *, result="x" * 200):
    """One agent invocation of `turns` tool calls, as captured: each model call's span repeats the conversation so
    far, then its reply."""
    start = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

    def info(span_id, parent="agent"):
        return SpanInfo(
            trace_id="t", span_id=span_id, session_id="s", parent_span_id=parent, start_time=start, end_time=start
        )

    spans = [
        AgentInvocationSpan(
            span_info=info("agent", None),
            user_prompt="help me",
            agent_response="done",
            available_tools=[ToolConfig(name="search")],
        )
    ]
    history = [UserMessage(content=[TextContent(text="help me")])]
    for turn in range(turns):
        call = ToolCallContent(name="search", arguments={"query": f"q{turn}"}, tool_call_id=f"call{turn}")
        reply = AssistantMessage(content=[call])
        spans.append(InferenceSpan(span_info=info(f"chat{turn}"), messages=[*history, reply]))
        spans.append(
            ToolExecutionSpan(
                span_info=info(f"tool{turn}"),
                tool_call=ToolCall(name="search", arguments=call.arguments, tool_call_id=call.tool_call_id),
                tool_result=ToolResult(content=result, tool_call_id=call.tool_call_id),
            )
        )
        history += [reply, UserMessage(content=[ToolResultContent(content=result, tool_call_id=call.tool_call_id)])]
    answer = AssistantMessage(content=[TextContent(text="done")])
    spans.append(InferenceSpan(span_info=info("chat-last"), messages=[*history, answer]))
    return Session(session_id="s", traces=[Trace(trace_id="t", session_id="s", spans=spans)])


def _transcript_file(run):
    (path,) = (run.root / "transcripts").iterdir()
    return path


def test_a_session_transcript_stores_each_message_once(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    session = _conversation(3)
    run.save_trial(_trial("c1", trajectory=session))

    saved = json.loads(_transcript_file(run).read_text())
    assert len(saved["messages"]) == 8  # the prompt, 3 tool calls and their results, the answer
    chats = [s for s in saved["session"]["traces"][0]["spans"] if s["span_type"] == "inference"]
    assert [s["messages"] for s in chats] == [[[0, 2]], [[0, 4]], [[0, 6]], [[0, 8]]]

    loaded = RunDirectory.open(run.root).load_trial("c1", 0, "a").trajectory
    # What reading the plain dump gives: every span, field, and message, in order.
    assert loaded == Session.model_validate(json.loads(session.model_dump_json()))
    first, *_, last = [s for s in loaded.traces[0].spans if isinstance(s, InferenceSpan)]
    assert all(a is b for a, b in zip(first.messages, last.messages[:2], strict=True))  # one object per message
    # Saving the loaded session again (as a judge-only re-run does) writes the same file.
    run.save_trial(_trial("c1", "b", trajectory=loaded))
    assert run.transcript("c1", 0, "b") == run.transcript("c1", 0, "a")
    assert len(list((run.root / "transcripts").iterdir())) == 1


def test_a_session_transcript_grows_with_the_conversation_not_its_square(tmp_path):
    sizes = {}
    for turns in (40, 80):
        run = RunDirectory.create(tmp_path / str(turns), CONFIG)
        run.save_trial(_trial("c1", trajectory=_conversation(turns)))
        sizes[turns] = _transcript_file(run).stat().st_size
    assert sizes[80] / sizes[40] < 2.1  # every model call repeating the whole conversation would be about 4


def test_a_message_repeated_within_a_span_round_trips_in_place(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    start = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    again = UserMessage(content=[TextContent(text="again")])
    other = AssistantMessage(content=[ToolCallContent(name="search", arguments={}, tool_call_id="1")])
    span = InferenceSpan(
        span_info=SpanInfo(session_id="s", start_time=start, end_time=start), messages=[again, other, again]
    )
    session = Session(session_id="s", traces=[Trace(trace_id="t", session_id="s", spans=[span])])
    run.save_trial(_trial("c1", trajectory=session))

    saved = json.loads(_transcript_file(run).read_text())
    assert len(saved["messages"]) == 2
    assert saved["session"]["traces"][0]["spans"][0]["messages"] == [[0, 2], [0, 1]]
    assert run.load_trial("c1", 0, "a").trajectory == Session.model_validate(json.loads(session.model_dump_json()))


def test_a_record_reads_back_without_its_trajectory_and_the_transcript_on_its_own(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    record = _trial("c1", trajectory=_conversation(2))
    run.save_trial(record)
    run.save_trial(_trial("c1", "b"))
    reopened = RunDirectory.open(run.root)

    transcript = reopened.transcript("c1", 0, "a")
    assert transcript == {"hash": _transcript_file(run).stem, "format": "packed_session"}
    assert reopened.transcript("c1", 0, "b") is None
    with pytest.raises(KeyError):
        reopened.transcript("c1", 1, "a")
    assert reopened.load_transcript(transcript) == reopened.load_trial("c1", 0, "a").trajectory

    _transcript_file(run).unlink()
    saved = next(e["record"] for e in reopened.events.read() if e["event"] == "trial")
    stripped = reopened.record_from(saved, load_trajectory=False)  # reads no transcript
    assert stripped == record.model_copy(update={"trajectory": None})


def test_a_session_saved_whole_still_loads_and_an_unknown_format_raises(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    dump = _conversation(2).model_dump(mode="json")
    (run.root / "transcripts").mkdir()
    (run.root / "transcripts" / "0123456789abcdef.json").write_text(json.dumps(dump))

    assert run.load_transcript({"hash": "0123456789abcdef", "format": "session"}) == Session.model_validate(dump)
    with pytest.raises(ValueError, match="unknown format 'tree'"):
        run.load_transcript({"hash": "0123456789abcdef", "format": "tree"})


def test_a_transcript_reads_the_same_by_its_format_or_its_shape(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    run.save_trial(_trial("c1", "a", trajectory=_conversation(2)))
    run.save_trial(_trial("c1", "b", trajectory=["read", "x"]))
    dump = _conversation(1).model_dump(mode="json")
    (run.root / "transcripts" / "0123456789abcdef.json").write_text(json.dumps(dump))

    for transcript in (run.transcript("c1", 0, "a"), run.transcript("c1", 0, "b")):
        by_shape = run_directory_module.read_transcript(run.root, transcript["hash"])
        assert by_shape == run.load_transcript(transcript)
    assert run_directory_module.read_transcript(run.root, "0123456789abcdef") == Session.model_validate(dump)


@pytest.mark.parametrize("digest", ["../../x", "0123456789ABCDEF", "0123456789abcde", "0123456789abcdef0", ""])
def test_a_transcript_hash_not_in_the_saved_form_raises(tmp_path, digest):
    run = RunDirectory.create(tmp_path, CONFIG)
    with pytest.raises(ValueError, match="not a transcript hash"):
        run.load_transcript({"hash": digest, "format": "list"})
    with pytest.raises(ValueError, match="not a transcript hash"):
        run_directory_module.read_transcript(run.root, digest)


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


def test_open_fails_when_a_transcript_any_trial_event_refers_to_is_missing(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    run.save_trial(_trial("c1", "a", trajectory=["first"]))
    first = run.transcript("c1", 0, "a")
    run.save_trial(_trial("c1", "a", trajectory=["second"]))  # replaces it in the index, not in the log
    run.save_trial(_trial("c1", "b", trajectory=None))
    RunDirectory.open(run.root)  # every file present

    (run.root / "transcripts" / f"{first['hash']}.json").unlink()

    with pytest.raises(FileNotFoundError, match=rf"1 transcript file.*{first['hash']}\.json"):
        RunDirectory.open(run.root)


def test_a_trial_event_cut_off_mid_write_counts_as_not_run(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    run.save_trial(_trial("c1", "a"))
    run.save_trial(_trial("c1", "b"))
    path = run.root / "events.jsonl"
    path.write_bytes(path.read_bytes()[:-40])

    reopened = RunDirectory.open(run.root)
    assert reopened.load_trial("c1", 0, "a") is not None
    assert reopened.load_trial("c1", 0, "b") is None


def test_open_writes_nothing_even_to_a_log_cut_off_mid_write(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    run.save_trial(_trial("c1", "a"))
    run.save_trial(_trial("c1", "b"))
    path = run.root / "events.jsonl"
    path.write_bytes(path.read_bytes()[:-40])  # the writer may still be finishing that line
    cut = path.read_bytes()

    RunDirectory.open(run.root)

    assert path.read_bytes() == cut


def test_one_writer_at_a_time_holds_the_lock(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    run.lock()
    run.lock()  # already held: a no-op
    other = RunDirectory.open(run.root)  # reading a run being written is fine

    with pytest.raises(RuntimeError, match="is open for writing in another process"):
        other.lock()

    run.unlock()
    run.unlock()  # not held: a no-op
    other.lock()
    with pytest.raises(RuntimeError, match="is open for writing"):
        run.lock()
    other.unlock()


def test_lock_refuses_a_run_another_writer_changed_after_it_was_read(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    reader = RunDirectory.open(run.root)
    run.save_candidate(_baseline())  # another writer appends after `reader` read the log

    with pytest.raises(RuntimeError, match="events.jsonl changed after it was read"):
        reader.lock()

    run.lock()  # the refused lock() didn't keep the lock
    run.save_candidate(_child(_baseline()))  # this object's own appends never make its log look changed
    run.unlock()
    run.lock()
    run.unlock()


def _wait_until(condition, seconds):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.1)


def _locks(run):
    try:
        run.lock()
    except RuntimeError:
        return False
    return True


def test_the_lock_of_a_killed_process_is_released(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path\n"
            "from strands_evals.optimization.record.run_directory import RunDirectory\n"
            "run = RunDirectory.open(Path(sys.argv[1]))\n"
            "run.lock()\n"
            "Path(sys.argv[2]).touch()\n"
            "sys.stdin.read()",
            str(run.root),
            str(ready),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        _wait_until(lambda: ready.exists() or child.poll() is not None, seconds=60)
        assert child.poll() is None, f"the child exited with {child.returncode}"
        with pytest.raises(RuntimeError, match="is open for writing in another process"):
            run.lock()
    finally:
        child.kill()
        child.wait()
        child.stdin.close()

    _wait_until(lambda: _locks(run), seconds=10)  # the OS may release a dead process's lock a moment later
    run.unlock()


@pytest.mark.skipif(sys.platform == "win32", reason="the fallback is tested through fcntl.flock")
def test_a_filesystem_that_cannot_lock_runs_unlocked_with_a_warning(tmp_path, monkeypatch, caplog):
    def flock(fd, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(run_directory_module.fcntl, "flock", flock)
    run = RunDirectory.create(tmp_path, CONFIG)
    reader = RunDirectory.open(run.root)

    run.lock()
    run.save_candidate(_baseline())
    run.unlock()

    assert "the filesystem can't lock the run" in caplog.text
    with pytest.raises(RuntimeError, match="events.jsonl changed after it was read"):
        reader.lock()  # the stale check still holds


def test_effects_diffs_and_the_result_are_written(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)

    run.save_effects({"effects": [], "reason": "no fit"})
    run.save_diff("best", "--- prompt\n", {"x": {"old": 1, "new": 2}})
    run.save_diff("front/1-abc", "", {})
    run.save_result({"run_id": run.run_id, "split": ("train",)})
    run.save_result({"run_id": run.run_id, "split": ("train", "test")})  # a resumed finished run rewrites it

    # Compact: no indentation, no spaces after separators.
    assert (run.root / "result.json").read_text() == f'{{"run_id":"{run.run_id}","split":["train","test"]}}\n'

    assert json.loads((run.root / "effects.json").read_text()) == {"effects": [], "reason": "no fit"}
    assert (run.root / "best.diff").read_text() == "--- prompt\n"
    assert json.loads((run.root / "best.values.json").read_text()) == {"x": {"old": 1, "new": 2}}
    assert json.loads((run.root / "front" / "1-abc.values.json").read_text()) == {}
    assert run.workspace("abc") == run.root / "candidates" / "abc" / "optimization_files"


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX permission bits")
def test_the_files_written_whole_get_a_plain_writes_mode(tmp_path):
    umask = os.umask(0o022)
    try:
        run = RunDirectory.create(tmp_path, CONFIG)
        run.save_effects({"effects": []})
        run.save_diff("best", "", {})
        run.save_result({})
    finally:
        os.umask(umask)

    for name in ["run.json", "effects.json", "best.diff", "best.values.json", "result.json", "events.jsonl"]:
        assert stat.S_IMODE((run.root / name).stat().st_mode) == 0o644, name
