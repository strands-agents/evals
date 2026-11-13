"""Tests for `Reflection.propose`: a scripted model drives the real agent, shell, and submit tool."""

import os
import tempfile

import pytest

from strands_evals.optimization import Criteria, InfrastructureError, Maximize, Reflection, Space, TextValue
from strands_evals.optimization.candidate import Candidate
from strands_evals.optimization.search.proposer import ProposalRequest
from tests.strands_evals.optimization.scripted_model import ScriptedModel, ToolUse
from tests.strands_evals.optimization.search.test_reflection_evidence import REGISTRY, scores_of, summary, trial

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
    assert patch.state["revised"] == [{"prompt": len(long) + 1}]
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


async def test_another_failure_on_every_attempt_proposes_nothing():
    patches, events, _ = await propose([ValueError("bad stream")])
    assert patches == []
    assert [e for e, _ in events] == ["proposer_retry"] * 3


async def test_a_value_naming_a_shown_case_is_screened():
    turns = [write("p.txt", "If the order is order-123, apologize."), submit(prompt="p.txt"), "Done."]
    patches, events, _ = await propose(turns)
    assert patches == []
    assert events == [("screened", {"key": "prompt", "reason": "adds the name of a shown case, 'order-123'"})]


async def test_a_value_copying_a_shown_output_is_screened():
    turns = [write("p.txt", "Never say: I could not find the order you asked about."), submit(prompt="p.txt"), "Done."]
    patches, events, _ = await propose(turns)
    assert patches == []
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
