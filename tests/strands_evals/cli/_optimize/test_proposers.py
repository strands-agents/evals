import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from strands_evals.cli._optimize import proposers
from strands_evals.cli._optimize.proposers import (
    FileProposer,
    ProposalContext,
    RandomProposer,
    ReflectionProposer,
    load_proposal_file,
    parse_patches,
)
from strands_evals.cli._optimize.space import CliSpace
from strands_evals.optimization.declaration.space import ParamInfo
from strands_evals.types.optimization import Patch

TEXT = ParamInfo(key="prompt", name="prompt", kind="text", default="base", size_cap=100)
SPACE = CliSpace(
    params=(
        TEXT,
        ParamInfo(key="turns", name="turns", kind="int", default=3, min_value=1, max_value=9),
        ParamInfo(key="rate", name="rate", kind="float", default=0.5, min_value=0.01, max_value=1.0, log=True),
        ParamInfo(key="model", name="model", kind="one_of", default="a", options=("a", "b", "c")),
    ),
    root=None,  # type: ignore[arg-type]
)


def context(keys=("prompt",), count=2, iteration=1, consumed=0, feedback=None):
    return ProposalContext(
        space=SPACE,
        keys=keys,
        parent=SPACE.baseline(),
        parent_number=0,
        count=count,
        iteration=iteration,
        consumed=consumed,
        feedback=feedback or {},
    )


def test_load_proposal_file_reads_plain_text_for_a_lone_text_key(tmp_path):
    path = tmp_path / "candidate.txt"
    path.write_text("better\n")
    space = CliSpace(params=(TEXT,), root=tmp_path)

    assert load_proposal_file(path, space) == (Patch(updates={"prompt": "better\n"}, rationale="proposal file item 1"),)


def test_load_proposal_file_reads_patches_and_turns_integers_into_floats(tmp_path):
    path = tmp_path / "patches.json"
    path.write_text(json.dumps([{"updates": {"rate": 1, "model": "b"}, "rationale": "faster"}]))

    (patch,) = load_proposal_file(path, SPACE)

    assert patch.updates == {"rate": 1.0, "model": "b"} and isinstance(patch.updates["rate"], float)
    assert patch.rationale == "faster"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[]", "nonempty list"),
        ("{", "not valid JSON"),
        ('["text"]', "exactly one text key"),
        ('[{"updates": {}}]', "nonempty 'updates'"),
    ],
)
def test_load_proposal_file_rejects_bad_files(tmp_path, content, message):
    path = tmp_path / "patches.json"
    path.write_text(content)
    space = CliSpace(params=(TEXT, replace(TEXT, key="other", name="other")), root=tmp_path)

    with pytest.raises(ValueError, match=message):
        load_proposal_file(path, space)


def test_file_proposer_continues_after_the_consumed_patches():
    patches = tuple(Patch(updates={"prompt": f"v{index}"}, rationale="") for index in range(3))
    proposer = FileProposer(patches=patches, keys=("prompt",), source="patches.json")

    assert proposer.propose(context(count=2)).patches == list(patches[:2])
    assert proposer.propose(context(count=2, consumed=2)).patches == [patches[2]]
    assert proposer.propose(context(count=2, consumed=3)).patches == []


def test_random_proposer_is_seeded_per_turn_and_stays_inside_the_bounds():
    proposer = RandomProposer(keys=("turns", "rate", "model"), seed=5)

    first = proposer.propose(context(keys=proposer.keys, count=20)).patches
    again = proposer.propose(context(keys=proposer.keys, count=20)).patches
    later = proposer.propose(context(keys=proposer.keys, count=20, iteration=2)).patches

    assert first == again and first != later
    for patch in first:
        SPACE.validate({**SPACE.baseline(), **patch.updates})
        assert isinstance(patch.updates["turns"], int) and isinstance(patch.updates["rate"], float)


def test_parse_patches_strips_code_fences_and_keeps_at_most_count():
    reply = (
        '```json\n{"patches": [{"updates": {"prompt": "a"}, "rationale": " r "}, {"updates": {"prompt": "b"}}]}\n```'
    )

    assert parse_patches(reply, 1) == [Patch(updates={"prompt": "a"}, rationale="r")]


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("nope", "not JSON"),
        ('{"items": []}', "'patches' list"),
        ('{"patches": [{"updates": {"prompt": 1}}]}', "text values"),
    ],
)
def test_parse_patches_rejects_other_shapes(reply, message):
    with pytest.raises(ValueError, match=message):
        parse_patches(reply, 2)


def test_reflection_proposer_sends_the_contrast_and_reports_its_usage(monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __call__(self, prompt):
            calls.append(json.loads(prompt))
            reply = '{"patches": [{"updates": {"prompt": "sharper"}, "rationale": "fix misses"}]}'
            return _Result(reply, {"totalTokens": 30, "outputTokens": 10})

    monkeypatch.setattr(proposers, "Agent", FakeAgent)
    proposer = ReflectionProposer(model="model-x", keys=("prompt",))

    batch = proposer.propose(context(count=1, feedback={"cases": [{"case": "a"}], "other_versions": []}))

    assert calls[0]["model"] == "model-x" and calls[0]["callback_handler"] is None
    assert calls[1]["editable_keys"][0]["key"] == "prompt" and calls[1]["cases"] == [{"case": "a"}]
    assert batch.patches == [Patch(updates={"prompt": "sharper"}, rationale="fix misses")]
    assert (batch.usage.tokens, batch.usage.output_tokens, batch.usage.input_tokens) == (30, 10, 20)
    assert batch.usage.models == ["model-x"]


class _Result:
    def __init__(self, text, usage):
        self._text = text
        self.metrics = SimpleNamespace(accumulated_usage=usage, cycle_count=1)

    def __str__(self):
        return self._text
