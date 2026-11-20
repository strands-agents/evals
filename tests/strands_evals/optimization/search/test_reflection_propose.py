"""Tests for `Reflection.propose`: a scripted model drives the real agent, shell, and submit tool."""

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace

import pytest
from strands.types.exceptions import ContextWindowOverflowException, EventLoopException

from strands_evals.optimization import Criteria, InfrastructureError, Maximize, Reflection, Space, TextValue
from strands_evals.optimization.candidate import Candidate
from strands_evals.optimization.search.proposer import ProposalRequest
from strands_evals.optimization.search.reflection import _ShortenToolResultsOnOverflow
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


class Recording(ScriptedModel):
    """Records each request, reports its size as a provider does (utilization is read from the reported usage),
    and rejects one over its 4,000-token window, as a provider does, without using up a scripted turn.

    `believed` is the window Strands is told (the real one by default)."""

    window = 4000  # tokens: a few large reads fill it

    def __init__(self, turns, *, believed=None):
        super().__init__(turns)
        self.config["context_window_limit"] = believed or self.window
        self.seen, self.tools, self.rejected, self.metadata = [], set(), 0, []

    async def stream(self, messages, tool_specs=None, *args, **kwargs):
        self.metadata += [m["metadata"] for m in messages if "metadata" in m]
        request = [json.dumps(m["content"]) for m in messages]
        tokens = sum(map(len, request)) // 4
        if tokens > self.window:
            self.rejected += 1
            raise ContextWindowOverflowException(f"{tokens} tokens")
        self.tools.add(tuple(spec["name"] for spec in tool_specs))
        self.seen.append(request)
        self.usage = {**self.usage, "inputTokens": tokens}
        async for event in super().stream(messages, tool_specs, *args, **kwargs):
            yield event


async def test_a_full_context_elides_old_messages_but_keeps_the_task_message():
    """Past 85% of the context window the oldest messages are elided; the task message stays first through every
    call, the send-back's included, and the tool outputs that remain are whole."""

    reads = [ToolUse("shell", {"command": "printf 'y%.0s' $(seq 1500)"}) for _ in range(8)]
    turns = [*reads, write("p.txt", "x" * 80), submit(prompt="p.txt"), "Done."]
    turns += [write("p.txt", "Ask for the number."), submit(prompt="p.txt"), "Shortened."]
    model = Recording(turns)
    reflection = Reflection(model)
    reflection._warned = True
    (patch,) = await reflection.propose(request())
    assert patch.updates == {"prompt": "Ask for the number.\n"}
    task = model.seen[0][0]
    assert "Write a new version of the parameters" in task
    assert all(messages[0] == task for messages in model.seen)  # the send-back's call included
    elided = next(messages for messages in model.seen if "elided" in "".join(messages[1:2]))
    assert len(elided) == 5  # the task + the last 4: Reflection's strategy, not the SDK's emergency fallback
    assert model.tools == {("shell", "submit_optimized_params")}  # no retrieval tool
    assert all("y" * 1500 in m for messages in model.seen for m in messages if "yyyy" in m)  # outputs not cut


async def test_a_send_back_is_pinned_so_elision_keeps_it():
    """A send-back stays in every request until the agent has answered it, though older messages are elided."""
    reads = [ToolUse("shell", {"command": "printf 'y%.0s' $(seq 1500)"}) for _ in range(8)]
    turns = [write("p.txt", "x" * 80), submit(prompt="p.txt"), "Done."]
    turns += [*reads, write("p.txt", "Ask for the number."), submit(prompt="p.txt"), "Shortened."]
    model = Recording(turns)
    reflection = Reflection(model)
    reflection._warned = True
    (patch,) = await reflection.propose(request())
    assert patch.updates == {"prompt": "Ask for the number.\n"}
    start = next(i for i, messages in enumerate(model.seen) if any("its limit 60" in m for m in messages))
    after = model.seen[start:]
    assert any("elided" in m for messages in after for m in messages)  # elided during the send-back's turn
    assert all(any("its limit 60" in m for m in messages) for messages in after)
    assert model.metadata == []  # the pin never reaches the model


async def test_tool_results_too_large_for_the_window_are_cut_and_the_call_goes_on():
    """A turn whose tool results alone overflow the window: the provider rejects it once, the results are cut to
    their first and last 200 characters, and the same call goes on (no retry with a fresh agent)."""
    big = ToolUse(
        "shell", {"command": "printf 'a%.0s' $(seq 200); printf 'y%.0s' $(seq 20000); printf 'z%.0s' $(seq 200)"}
    )
    turns = [big, write("p.txt", "Ask for the number."), submit(prompt="p.txt"), "Done."]
    model = Recording(turns)
    reflection = Reflection(model)
    reflection._warned = True
    events = []
    (patch,) = await reflection.propose(request(events=events))
    assert patch.updates == {"prompt": "Ask for the number.\n"}
    assert model.rejected == 1 and events == []
    cut = next(m for messages in model.seen for m in messages if "[cut:" in m)
    assert "a" * 200 in cut and "z" * 150 in cut and "y" * 200 not in cut  # the tail ends with the exit code
    assert "characters removed because the context window was full" in cut


async def test_a_rejected_request_is_elided_first_then_a_kept_result_is_cut():
    """Strands believes the window is larger than it is, so nothing is elided until the provider rejects a request;
    then the elision runs first and the cut lands on a result the elided request keeps."""
    reads = [ToolUse("shell", {"command": f"printf '{c}%.0s' $(seq 2000)"}) for c in "abcdef"]
    turns = [*reads, write("p.txt", "Ask for the number."), submit(prompt="p.txt"), "Done."]
    model = Recording(turns, believed=200_000)
    reflection = Reflection(model)
    reflection._warned = True
    events = []
    (patch,) = await reflection.propose(request(events=events))
    assert patch.updates == {"prompt": "Ask for the number.\n"}
    assert model.rejected == 1 and events == []
    after = next(messages for messages in model.seen if "elided" in "".join(messages[1:2]))
    assert len(after) == 5  # the task + the last 4: two reads (call, result), the older result cut
    assert "[cut:" in after[2] and "d" * 200 in after[2] and "e" * 2000 in after[4]


async def test_an_unknown_window_is_taken_as_200000_tokens_and_a_rejected_request_still_recovers():
    """With no `context_window_limit`, Strands assumes 200,000 tokens and warns; the provider's rejection is
    handled as for a window Strands believes too large."""
    reads = [ToolUse("shell", {"command": f"printf '{c}%.0s' $(seq 2000)"}) for c in "abcdef"]
    turns = [*reads, write("p.txt", "Ask for the number."), submit(prompt="p.txt"), "Done."]
    model = Recording(turns)
    del model.config["context_window_limit"]
    reflection = Reflection(model)
    reflection._warned = True
    with pytest.warns(UserWarning, match="context_window_limit is not set"):
        (patch,) = await reflection.propose(request())
    assert patch.updates == {"prompt": "Ask for the number.\n"}
    assert model.rejected == 1
    assert any("[cut:" in m for messages in model.seen for m in messages)


async def test_a_request_no_strategy_can_shrink_fails_every_attempt_and_raises():
    """The task message alone is over the window: no strategy acts, so the SDK gives up on the first rejection,
    and Reflection raises after its last attempt."""
    model = Recording(["unused"])
    model.window = 50
    reflection = Reflection(model)
    reflection._warned = True
    events = []
    with pytest.raises(ContextWindowOverflowException):
        await reflection.propose(request(events=events))
    assert model.rejected == 4 and model.seen == []  # one rejection per attempt, no retry within one
    assert [e for e, _ in events].count("proposer_retry") == 3


async def test_every_large_tool_result_of_the_oldest_message_is_cut_at_once():
    shorten = _ShortenToolResultsOnOverflow()
    small = "s" * 800
    message = {
        "role": "user",
        "content": [
            {"toolResult": {"toolUseId": "t1", "content": [{"text": "o" * 1000}]}},
            {"toolResult": {"toolUseId": "t2", "content": [{"text": small}, {"text": "p" * 3000}]}},
        ],
    }
    assert await shorten.apply(SimpleNamespace(overflow=True, messages=[message]))
    texts = [item["text"] for block in message["content"] for item in block["toolResult"]["content"]]
    assert "[cut: 600 characters removed" in texts[0] and texts[1] == small
    assert "[cut: 2600 characters removed" in texts[2]
    assert not await shorten.apply(SimpleNamespace(overflow=True, messages=[message]))


async def test_tool_results_are_cut_only_on_overflow_oldest_message_first():
    shorten = _ShortenToolResultsOnOverflow()

    def result(text):
        return {"role": "user", "content": [{"toolResult": {"toolUseId": "t", "content": [{"text": text}]}}]}

    small = "s" * 800  # cutting it wouldn't shorten it enough to be worth a retry
    messages = [
        {"role": "user", "content": [{"text": "task " * 200}]},
        result("o" * 1000),
        result(small),
        result("n" * 1000),
    ]
    assert not await shorten.apply(SimpleNamespace(overflow=False, messages=messages))
    assert await shorten.apply(SimpleNamespace(overflow=True, messages=messages))
    texts = [m["content"][0].get("toolResult", {}).get("content", [{}])[0].get("text") for m in messages]
    assert "[cut: 600 characters removed" in texts[1] and texts[3] == "n" * 1000  # the oldest, alone
    assert await shorten.apply(SimpleNamespace(overflow=True, messages=messages))  # then the next large one
    assert messages[3]["content"][0]["toolResult"]["content"][0]["text"].endswith("n" * 200)
    assert not await shorten.apply(SimpleNamespace(overflow=True, messages=messages))  # a cut result isn't cut again
    assert messages[2]["content"][0]["toolResult"]["content"][0]["text"] == small
    assert messages[0]["content"][0]["text"] == "task " * 200


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
