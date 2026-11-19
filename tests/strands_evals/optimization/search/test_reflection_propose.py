"""Tests for `Reflection.propose`: a scripted model drives the real agent, shell, and submit tool."""

import asyncio
import os
import signal
import subprocess
import tempfile
import time

import pytest
from strands.types.exceptions import EventLoopException

from strands_evals.optimization import Criteria, InfrastructureError, Maximize, Reflection, Space, TextValue
from strands_evals.optimization.candidate import Candidate
from strands_evals.optimization.search.proposer import ProposalRequest
from strands_evals.types.trace import TextContent, ToolResultContent, UserMessage
from tests.strands_evals.optimization.scripted_model import ScriptedModel, ToolUse
from tests.strands_evals.optimization.search.test_reflection_evidence import (
    REGISTRY,
    scores_of,
    session,
    summary,
    trial,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Reflection needs Linux or macOS")

SPACE = Space(
    {
        "prompt": TextValue(default="Be helpful.\n", size_cap=60),
        "tone": TextValue(default="plain", size_cap=40),
    }
)


def request(*, n=1, keys=("prompt", "tone"), trials=None, events=None):
    criteria = Criteria(Maximize("Score"), [], REGISTRY)
    trials = trials or [
        trial("order-123", score=0.2, output="I could not find the order you asked about, sorry."),
        trial("b", score=0.9),
    ]
    parent = Candidate.baseline(SPACE)
    log = events.append if events is not None else None
    return ProposalRequest(
        space=SPACE,
        keys=list(keys),
        parent=parent,
        trials=trials,
        scores=scores_of(trials, criteria),
        inputs={t.case_name: f"Where is my order {t.case_name}?" for t in trials},
        fit=None,
        candidates=[summary(parent, number=0, rank=1)],
        n=n,
        iteration=1,
        seed=0,
        redraw=0,
        criteria=criteria,
        log=(lambda event, **fields: log((event, fields))) if log else (lambda *args, **kwargs: None),
    )


def write(name, text):
    """A turn that writes `text` to `name` in the work folder (the shell's working directory)."""
    return ToolUse("shell", {"command": f"cat > {name} << 'EOF'\n{text}\nEOF"})


def submit(**files):
    return ToolUse("submit_optimized_params", {"file_path_dict": files})


async def propose(turns, *, warn=False, **kwargs):
    """Run one proposal with a scripted model; the warning is silenced unless `warn`."""
    reflection = Reflection(ScriptedModel(turns), **kwargs)
    reflection._warned = not warn
    events = []
    patches = await reflection.propose(request(events=events))
    return patches, events, reflection


@pytest.fixture(autouse=True)
def temp_root(tmp_path, monkeypatch):
    """Every temp folder in `tmp_path/tmp`, so a test can see none is left behind."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


async def test_a_rewrite_becomes_a_patch_and_its_folders_are_removed(temp_root):
    turns = [write("p.txt", "Be helpful. Ask for the order number."), submit(prompt="p.txt"), "Asked for the number."]
    (patch,), events, reflection = await propose(turns)
    assert patch.updates == {"prompt": "Be helpful. Ask for the order number.\n"}  # the parent's final newline kept
    assert patch.rationale == "Asked for the number."
    assert patch.usage.tokens == 45 and patch.usage.output_tokens == 15 and patch.usage.models == ["scripted-model"]
    assert patch.state["trials"] == 2 and patch.state["versions"] == 1
    assert patch.state["digest"]["failing_cases"] == 1 and len(patch.state["digest"]["sha256"]) == 64
    assert patch.state["revised"] == [] and patch.state["dropped"] == {} and patch.state["rejected"] == []
    assert events == []
    assert os.listdir(temp_root) == []


async def test_the_first_proposal_warns_that_the_shell_is_not_a_sandbox():
    with pytest.warns(UserWarning, match="not a sandbox") as caught:
        await propose(["Nothing to change."], warn=True)
    assert caught[0].filename.endswith("reflection.py")  # not asyncio's: the coroutine's caller is the event loop


async def test_nothing_submitted_or_nothing_changed_proposes_nothing():
    assert (await propose(["I see nothing to change."]))[0] == []
    unchanged = [write("p.txt", "Be helpful."), submit(prompt="p.txt"), "Same."]
    assert (await propose(unchanged))[0] == []


async def test_a_value_over_its_size_cap_is_sent_back_and_the_shorter_one_merged():
    long = "Be helpful and always ask for the order number first, politely and clearly."
    turns = [
        write("p.txt", long),
        write("t.txt", "warm"),
        submit(prompt="p.txt", tone="t.txt"),
        "Done.",
        write("p.txt", "Ask for the order number."),
        submit(prompt="p.txt"),  # only the shortened value: the tone is kept from the first submission
        "Shortened.",
    ]
    (patch,), _, _ = await propose(turns)
    assert patch.updates == {"prompt": "Ask for the order number.\n", "tone": "warm"}
    assert patch.state["revised"] == [{"prompt": len(long)}]  # over the cap: the parent's newline isn't restored
    assert patch.rationale == "Done."  # the first call's final text, not the shortening turn's


async def test_a_value_still_over_its_size_cap_is_dropped_and_the_parents_kept():
    turns = [write("p.txt", "x" * 80), write("t.txt", "warm"), submit(prompt="p.txt", tone="t.txt"), "Done."]
    (patch,), _, _ = await propose(turns, over_limit_retries=0)
    assert patch.updates == {"tone": "warm"}
    assert "over the size_cap of 60" in patch.state["dropped"]["prompt"]


async def test_a_refused_submission_is_sent_back_once_more():
    turns = [
        write("p.txt", "# Version 0 (the baseline)\nrank: 1\n\nBe brief."),
        submit(prompt="p.txt"),
        "Submitted.",
        write("p.txt", "Be brief."),
        submit(prompt="p.txt"),
        "Fixed.",
    ]
    (patch,), _, _ = await propose(turns)
    assert patch.updates == {"prompt": "Be brief.\n"}
    assert patch.state["rejected"][0]["name"] == "prompt"


async def test_every_submission_refused_proposes_nothing():
    header = "# Version 0 (the baseline)\nrank: 1\n\nBe brief."
    turns = [write("p.txt", header), submit(prompt="p.txt"), "Submitted."]
    assert (await propose(turns, over_limit_retries=1))[0] == []


async def test_a_failed_agent_call_is_retried_and_logged():
    turns = [ValueError("bad stream"), write("p.txt", "Be brief."), submit(prompt="p.txt"), "Done."]
    (patch,), events, _ = await propose(turns)
    assert patch.updates == {"prompt": "Be brief.\n"}
    assert events == [("proposer_retry", {"error_type": "ValueError", "error": "bad stream"})]


async def test_an_infrastructure_failure_on_every_attempt_raises():
    with pytest.raises(InfrastructureError):
        await propose([InfrastructureError("bedrock is down")])


async def test_any_failure_on_every_attempt_raises_after_three_retries():
    """A wrong model id or no access fails every attempt the same way: raised, never hidden as `[]`."""
    reflection = Reflection(ScriptedModel([ValueError("The provided model identifier is invalid.")]))
    reflection._warned = True
    events = []
    with pytest.raises(ValueError, match="model identifier"):
        await reflection.propose(request(events=events))
    assert [e for e, _ in events] == ["proposer_retry"] * 3 + ["proposer_usage"]


async def test_a_value_naming_a_shown_case_is_screened():
    turns = [write("p.txt", "If the order is order-123, apologize."), submit(prompt="p.txt"), "Done."]
    patches, events, _ = await propose(turns)
    assert patches == []
    assert events[0] == ("screened", {"key": "prompt", "reason": "adds the name of a shown case, 'order-123'"})
    assert [e for e, _ in events] == ["screened", "proposer_usage"]
    assert events[1][1]["usage"].tokens == 45  # the call's three model calls: no patch carries them


async def test_a_value_copying_a_shown_output_is_screened():
    turns = [write("p.txt", "Never say: I could not find the order you asked about."), submit(prompt="p.txt"), "Done."]
    patches, events, _ = await propose(turns)
    assert patches == []
    assert events[0][1]["reason"].startswith("copies 40 characters from a shown trial verbatim")


async def test_a_value_copying_a_shown_tool_result_is_screened():
    """The trace files show the agent every message, so a tool result is screened like the input and the output."""
    result = "the warehouse in Cologne ships on Mondays only"
    conversation = [
        UserMessage(content=[TextContent(text="Where is my order a?")]),
        UserMessage(
            content=[ToolResultContent(name="lookup", content=result, tool_call_id="1")],
            role="assistant",
        ),
    ]
    trials = [trial("a", score=0.2, output="sorry", trajectory=session(conversation))]
    turns = [write("p.txt", result), submit(prompt="p.txt"), "Done."]
    reflection = Reflection(ScriptedModel(turns))
    reflection._warned = True
    events = []
    patches = await reflection.propose(request(trials=trials, events=events))
    assert patches == []
    assert events[0][0] == "screened"
    assert events[0][1]["reason"].startswith("copies 40 characters from a shown trial verbatim")


async def test_a_key_outside_the_block_is_dropped():
    turns = [write("p.txt", "Be brief."), write("t.txt", "warm"), submit(prompt="p.txt", tone="t.txt"), "Done."]
    reflection = Reflection(ScriptedModel(turns))
    reflection._warned = True
    (patch,) = await reflection.propose(request(keys=("prompt",)))
    assert patch.updates == {"prompt": "Be brief.\n"}
    assert patch.state["dropped"] == {"tone": "not one of this block's keys"}


async def test_a_refused_shell_command_is_kept_with_its_proposal():
    turns = [
        ToolUse("shell", {"command": "cat /tmp/other-study/notes.txt"}),
        write("p.txt", "Be brief."),
        submit(prompt="p.txt"),
        "Done.",
    ]
    (patch,), _, reflection = await propose(turns)
    assert patch.state["refused"] == [
        {"command": "cat /tmp/other-study/notes.txt", "paths": ["/tmp/other-study/notes.txt"]}
    ]
    assert reflection.refused == patch.state["refused"]


async def test_an_empty_first_reply_gets_the_fallback_rationale():
    turns = [write("p.txt", "Be brief."), submit(prompt="p.txt"), ""]
    (patch,), _, _ = await propose(turns)
    assert patch.rationale == "rewritten by Reflection"


async def test_n_proposals_come_one_agent_each_in_order():
    turns = [
        write("p.txt", "First."),
        submit(prompt="p.txt"),
        "One.",
        write("p.txt", "Second."),
        submit(prompt="p.txt"),
        "Two.",
    ]
    reflection = Reflection(ScriptedModel(turns))
    reflection._warned = True
    patches = await reflection.propose(request(n=2))
    assert [p.updates["prompt"] for p in patches] == ["First.\n", "Second.\n"]


async def test_no_trial_to_learn_from_proposes_nothing():
    reflection = Reflection(ScriptedModel(["unused"]))
    reflection._warned = True
    assert await reflection.propose(request(trials=[trial("a", status="infrastructure_invalid")])) == []


async def test_a_send_back_after_a_long_first_turn_still_has_the_task_message():
    """The window (40 messages) is trimmed at the end of the first call; the task message is pinned through it."""

    class Recording(ScriptedModel):
        def __init__(self, turns):
            super().__init__(turns)
            self.first_messages = []

        async def stream(self, messages, *args, **kwargs):
            self.first_messages.append(messages[0])
            async for event in super().stream(messages, *args, **kwargs):
                yield event

    reads = [ToolUse("shell", {"command": "ls"}) for _ in range(25)]  # 50 messages
    turns = [*reads, write("p.txt", "x" * 80), submit(prompt="p.txt"), "Done."]
    turns += [write("p.txt", "Ask for the number."), submit(prompt="p.txt"), "Shortened."]
    model = Recording(turns)
    reflection = Reflection(model)
    reflection._warned = True
    (patch,) = await reflection.propose(request())
    assert patch.updates == {"prompt": "Ask for the number.\n"}
    task = model.first_messages[0]
    assert "Write a new version of the parameters" in task["content"][0]["text"]
    assert model.first_messages[-1] == task  # the send-back's call starts from the task message too


async def test_the_three_folders_share_one_folder_of_the_call():
    """`../traces` from the work folder is the agent's own; `..` alone (the shared folder) and above are refused."""
    turns = [
        ToolUse("shell", {"command": "ls ../traces ../versions | grep -c json > seen.txt"}),
        ToolUse("shell", {"command": "ls .."}),
        ToolUse("shell", {"command": "ls ../.."}),
        submit(tone="seen.txt"),
        "Done.",
    ]
    (patch,), _, reflection = await propose(turns)
    assert patch.updates == {"tone": "2"}  # the two trace files, read through ../traces
    assert [r["command"] for r in reflection.refused] == ["ls ..", "ls ../.."]


async def test_a_value_adopting_a_judges_advice_word_for_word_is_kept():
    """Reasons are advice to act on, not case content: copying one is no leak (an output copied still is)."""
    advice = "The agent should always confirm the order number before giving a refund."
    trials = [trial("order-123", score=0.2, reason=advice), trial("b", score=0.9)]
    turns = [write("p.txt", "Always confirm the order number before giving a refund."), submit(prompt="p.txt"), "Ok."]
    reflection = Reflection(ScriptedModel(turns))
    reflection._warned = True
    (patch,) = await reflection.propose(request(trials=trials))
    assert patch.updates == {"prompt": "Always confirm the order number before giving a refund.\n"}


# ---- the spend no patch carries: `proposer_usage` ------------------------------------------------------------------


def usage_events(events):
    return [fields["usage"] for event, fields in events if event == "proposer_usage"]


async def test_nothing_submitted_logs_the_calls_spend():
    patches, events, _ = await propose(["I see nothing to change."])
    assert patches == []
    (usage,) = usage_events(events)
    assert events == [("proposer_usage", {"usage": usage})]
    assert (usage.tokens, usage.model_calls, usage.models) == (15, 1, ["scripted-model"])


async def test_a_value_still_over_its_cap_after_the_send_backs_logs_the_total_spend():
    turns = [write("p.txt", "x" * 80), submit(prompt="p.txt"), "Done."]
    turns += [write("p.txt", "y" * 80), submit(prompt="p.txt"), "Again."]
    patches, events, _ = await propose(turns, over_limit_retries=1)
    assert patches == []
    (usage,) = usage_events(events)
    assert usage.tokens == 6 * 15  # the first call and its send-back


async def test_every_attempt_failing_logs_all_their_spend_then_raises():
    turns = [ToolUse("shell", {"command": "ls"}), ValueError("stream broke")] * 4
    reflection = Reflection(ScriptedModel(turns))
    reflection._warned = True
    events = []
    with pytest.raises(EventLoopException, match="stream broke"):  # Strands wraps an error after a tool call
        await reflection.propose(request(events=events))
    assert [e for e, _ in events] == ["proposer_retry"] * 3 + ["proposer_usage"]
    assert usage_events(events)[0].tokens == 4 * 15  # each attempt's one model call that answered


async def test_a_patch_carries_its_calls_spend_and_logs_none():
    turns = [ToolUse("shell", {"command": "ls"}), ValueError("stream broke")]
    turns += [write("p.txt", "Be brief."), submit(prompt="p.txt"), "Done."]
    (patch,), events, _ = await propose(turns)
    assert [e for e, _ in events] == ["proposer_retry"]
    assert patch.usage.tokens == 4 * 15  # the failed attempt's call and the successful attempt's three


async def test_nothing_to_learn_from_logs_no_spend():
    reflection = Reflection(ScriptedModel(["unused"]))
    reflection._warned = True
    events = []
    assert await reflection.propose(request(trials=[trial("a", status="infrastructure_invalid")], events=events)) == []
    assert events == []


async def test_a_cancelled_call_logs_nothing_and_the_cancellation_propagates():
    class Hanging(ScriptedModel):
        def __init__(self):
            super().__init__(["unused"])
            self.started = asyncio.Event()

        async def stream(self, *args, **kwargs):
            self.started.set()
            await asyncio.Event().wait()
            yield {}  # pragma: no cover

    model = Hanging()
    reflection = Reflection(model)
    reflection._warned = True
    events = []
    task = asyncio.create_task(reflection.propose(request(events=events)))
    await asyncio.wait_for(model.started.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == []


async def test_a_later_slot_raising_logs_the_spend_of_the_patches_never_returned():
    turns = [write("p.txt", "Be brief."), submit(prompt="p.txt"), "Done.", ValueError("model gone")]
    reflection = Reflection(ScriptedModel(turns))
    reflection._warned = True
    events = []
    with pytest.raises(ValueError, match="model gone"):
        await reflection.propose(request(n=2, events=events))
    assert [e for e, _ in events] == ["proposer_retry"] * 3 + ["proposer_usage"] * 2
    failed, first = usage_events(events)
    assert (failed.tokens, first.tokens) == (0, 45)  # the failing slot's attempts answered nothing; slot 0's patch


# ---- a proposal's turn and time limits: `max_turns` and `max_seconds` ----------------------------------------------


async def limited(model, *, n=1, **kwargs):
    """Run one `propose` call on `model` with the given limits; the warning is silenced."""
    reflection = Reflection(model, **kwargs)
    reflection._warned = True
    events = []
    patches = await reflection.propose(request(n=n, events=events))
    return patches, events


@pytest.fixture
def started(monkeypatch):
    """Every process the shell starts, in order."""
    processes = []
    popen = subprocess.Popen

    def recording(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", recording)
    return processes


def killed(processes, command):
    """Whether the process the shell started for `command` was killed (it would otherwise run on)."""
    process = next(p for p in processes if p.args[-1] == command)
    try:
        return process.wait(timeout=5) == -signal.SIGKILL
    except subprocess.TimeoutExpired:
        return False


async def test_a_looping_agent_stops_at_the_default_turn_limit_with_what_it_submitted(caplog):
    model = ScriptedModel([write("p.txt", "Be brief."), submit(prompt="p.txt"), submit(prompt="missing.txt")])
    (patch,), events = await limited(model)
    assert model.calls == 100
    assert patch.updates == {"prompt": "Be brief.\n"}
    assert patch.state["stopped_by"] == "max_turns" and patch.usage.model_calls == 100
    assert patch.rationale == "rewritten by Reflection"  # the agent never replied with text
    assert events == []
    assert "limit=<max_turns> | reflection agent stopped at its turn or time limit" in caplog.text


async def test_a_submission_on_the_last_allowed_turn_is_kept():
    model = ScriptedModel([write("p.txt", "Be brief."), submit(prompt="p.txt"), "Done."])
    (patch,), _ = await limited(model, max_turns=2)
    assert model.calls == 2
    assert patch.updates == {"prompt": "Be brief.\n"} and patch.state["stopped_by"] == "max_turns"


@pytest.mark.parametrize("limits", [{"max_turns": 3}, {"max_turns": None, "max_seconds": None}])
async def test_a_proposal_that_ends_within_its_limits_is_not_stopped(limits):
    model = ScriptedModel([write("p.txt", "Be brief."), submit(prompt="p.txt"), "Done."])
    (patch,), _ = await limited(model, **limits)
    assert patch.state["stopped_by"] is None and patch.rationale == "Done."


async def test_a_send_back_gets_only_the_turns_left():
    turns = [write("p.txt", "x" * 80), submit(prompt="p.txt"), "Done."]
    turns += [write("p.txt", "Ask for the number."), submit(prompt="p.txt"), "Shortened."]
    model = ScriptedModel(turns)
    (patch,), _ = await limited(model, max_turns=5)
    assert model.calls == 5  # the send-back's write and submit, not its reply
    assert patch.updates == {"prompt": "Ask for the number.\n"}
    assert patch.state["revised"] == [{"prompt": 80}] and patch.state["stopped_by"] == "max_turns"


async def test_no_send_back_when_no_turn_is_left():
    turns = [write("p.txt", "x" * 80), write("t.txt", "warm"), submit(prompt="p.txt", tone="t.txt"), "Done."]
    turns += [write("p.txt", "Short."), submit(prompt="p.txt"), "Shortened."]
    model = ScriptedModel(turns)
    (patch,), _ = await limited(model, max_turns=4)
    assert model.calls == 4
    assert patch.updates == {"tone": "warm"} and "over the size_cap" in patch.state["dropped"]["prompt"]
    assert patch.state["revised"] == [] and patch.state["stopped_by"] == "max_turns"


async def test_no_refusal_is_sent_back_when_no_turn_is_left():
    header = "# Version 0 (the baseline)\nrank: 1\n\nBe brief."
    turns = [
        write("p.txt", header),
        submit(prompt="p.txt"),
        "Submitted.",
        write("p.txt", "Be brief."),
        submit(prompt="p.txt"),
    ]
    model = ScriptedModel(turns)
    patches, events = await limited(model, max_turns=3)
    assert model.calls == 3 and patches == []
    assert [e for e, _ in events] == ["proposer_usage"]


async def test_a_slow_shell_command_is_killed_at_the_time_limit_and_what_was_submitted_kept(temp_root, started):
    turns = [write("p.txt", "Be brief."), submit(prompt="p.txt"), ToolUse("shell", {"command": "sleep 30"}), "Done."]
    begun = time.monotonic()
    (patch,), events = await limited(ScriptedModel(turns), max_seconds=3)
    assert time.monotonic() - begun < 10
    assert patch.updates == {"prompt": "Be brief.\n"} and patch.state["stopped_by"] == "max_seconds"
    assert patch.usage.model_calls == 3 and events == []
    assert killed(started, "sleep 30")
    assert os.listdir(temp_root) == []


async def test_the_time_limit_during_a_send_back_keeps_the_first_submission():
    turns = [write("p.txt", "x" * 80), write("t.txt", "warm"), submit(prompt="p.txt", tone="t.txt"), "Done."]
    turns += [ToolUse("shell", {"command": "sleep 30"}), "Shortened."]
    (patch,), _ = await limited(ScriptedModel(turns), max_seconds=3)
    assert patch.updates == {"tone": "warm"}
    assert patch.state["revised"] == [{"prompt": 80}] and patch.state["stopped_by"] == "max_seconds"


class NeverEnding(ScriptedModel):
    """A model whose stream never ends; `on_cancel`, when given, is raised in place of the cancellation."""

    def __init__(self, on_cancel=None):
        super().__init__(["unused"])
        self.on_cancel = on_cancel
        self.streaming = asyncio.Event()

    async def stream(self, *args, **kwargs):
        self.calls += 1
        self.streaming.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if self.on_cancel is None:
                raise
            raise self.on_cancel from None
        yield {}  # pragma: no cover


@pytest.mark.parametrize("on_cancel", [None, RuntimeError("connection closed")])
async def test_a_model_stream_that_never_ends_is_stopped_at_the_time_limit_not_failed(on_cancel):
    model = NeverEnding(on_cancel)
    patches, events = await limited(model, max_seconds=1)
    assert patches == [] and model.calls == 1  # never retried, however the model ended once cancelled
    (usage,) = usage_events(events)
    assert [e for e, _ in events] == ["proposer_usage"] and usage.tokens == 0


async def test_a_cancelled_call_propagates_even_when_the_model_raises_once_cancelled():
    model = NeverEnding(RuntimeError("connection closed"))
    reflection = Reflection(model)
    reflection._warned = True
    events = []
    task = asyncio.create_task(reflection.propose(request(events=events)))
    await asyncio.wait_for(model.streaming.wait(), timeout=10)
    task.cancel()
    await asyncio.wait({task}, timeout=10)
    cancelled = task.cancelled()  # the model's error once cancelled was not taken for a failed attempt and retried
    if not task.done():  # else the retried attempt never ends: stop it, so the test fails instead of hanging
        model.on_cancel = None
        task.cancel()
        await asyncio.wait({task}, timeout=10)
    assert cancelled and model.calls == 1 and events == []


async def test_a_failure_with_no_turn_left_raises_without_a_retry(caplog):
    model = ScriptedModel([ValueError("The provided model identifier is invalid.")])
    reflection = Reflection(model, max_turns=1)
    reflection._warned = True
    events = []
    with pytest.raises(ValueError, match="model identifier"):
        await reflection.propose(request(events=events))
    assert model.calls == 1 and [e for e, _ in events] == ["proposer_usage"]
    assert "limit=<max_turns> | reflection agent failed with no turn or time left to retry" in caplog.text


async def test_retries_share_the_proposals_turns():
    turns = [ToolUse("shell", {"command": "ls"}), ValueError("stream broke")]
    turns += [write("p.txt", "Be brief."), submit(prompt="p.txt"), "Done."]
    model = ScriptedModel(turns)
    (patch,), events = await limited(model, max_turns=4)
    assert model.calls == 4  # 2 in the failed attempt, then 2 in the retry: its reply is never asked for
    assert patch.updates == {"prompt": "Be brief.\n"} and patch.state["stopped_by"] == "max_turns"
    assert [e for e, _ in events] == ["proposer_retry"]


async def test_the_models_own_timeout_error_is_retried_not_the_time_limit():
    turns = [TimeoutError("read timed out"), write("p.txt", "Be brief."), submit(prompt="p.txt"), "Done."]
    (patch,), events = await limited(ScriptedModel(turns), max_seconds=600)
    assert patch.updates == {"prompt": "Be brief.\n"} and patch.state["stopped_by"] is None
    assert events == [("proposer_retry", {"error_type": "TimeoutError", "error": "read timed out"})]


async def test_each_of_n_proposals_has_its_own_limits():
    turns = [write("p.txt", "First."), submit(prompt="p.txt"), "One."]
    turns += [write("p.txt", "Second."), submit(prompt="p.txt"), "Two."]
    patches, _ = await limited(ScriptedModel(turns), n=2, max_turns=3)
    assert [(p.updates["prompt"], p.state["stopped_by"]) for p in patches] == [("First.\n", None), ("Second.\n", None)]


async def test_a_cancelled_call_kills_its_running_command_and_the_cancellation_propagates(temp_root, started):
    turns = [ToolUse("shell", {"command": "sleep 30"}), "Done."]
    reflection = Reflection(ScriptedModel(turns))
    reflection._warned = True
    events = []
    task = asyncio.create_task(reflection.propose(request(events=events)))
    for _ in range(1000):  # up to 10 s for the command to start
        if any(p.args[-1] == "sleep 30" for p in started):
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == [] and killed(started, "sleep 30")
    assert os.listdir(temp_root) == []
