"""Tests for Reflection's prompts: what varies in the task message, and the fixed texts' placeholders."""

import pytest

from strands_evals.optimization.prompt_templates.reflection import get_template
from strands_evals.optimization.prompt_templates.reflection.reflection_v0 import TextVariable, VersionRow

T = get_template()


def _row(**fields):
    defaults = dict(
        rank=1,
        number=3,
        built_from=0,
        is_current=False,
        is_baseline=False,
        changed=["prompt"],
        cells=["0.8"],
        meets_limits=True,
        conversations=8,
    )
    return VersionRow(**{**defaults, **fields})


def _message(**fields):
    defaults = dict(
        versions=[_row()],
        versions_folder="/v",
        goal_metrics=["Helpfulness"],
        limits=[],
        traces_folder="/t",
        digest_file=None,
        label_metrics=["Helpfulness"],
        label_ranked=False,
        judged_by_version=[],
        variables=[TextVariable(name="prompt", value="Be brief.", edit_instructions=None, size_cap=None)],
        work_folder="/w",
    )
    return T.task_message(**{**defaults, **fields})


def test_system_prompt_names_the_work_folder():
    prompt = T.system_prompt("/w/reflection_work_1")
    assert "write only in your working folder, `/w/reflection_work_1`: it is your shell's working directory" in prompt
    assert "{" not in prompt


def test_submit_instructions_name_the_submit_tool():
    assert T.SUBMIT_INSTRUCTIONS.startswith("\n\n## Submitting the Optimized Parameters\n")
    assert 'submit_optimized_params(file_path_dict={"param_name": "<working folder>/<file>", ...})' in (
        T.SUBMIT_INSTRUCTIONS
    )


def test_versions_table_without_limits():
    message = _message(
        versions=[
            _row(rank=1, number=3, built_from=0, is_current=True, changed=["prompt", "other values"], cells=["0.81"]),
            _row(rank=None, number=0, built_from=None, is_baseline=True, changed=[], cells=["-"], conversations=0),
        ],
    )
    assert "2 versions, best first for this study's goal. Each version's results" in message
    assert (
        "| Rank | Version | Built from | Changed | Helpfulness | Conversations |\n"
        "|---|---|---|---|---|---|\n"
        "| 1 | 3 ← | 0 | prompt, other values | 0.81 | 8 |\n"
        "| - | 0 (baseline) | - | - | - | 0 |\n"
        "\n\n## The current version's conversations\n"
    ) in message


@pytest.mark.parametrize(("meets", "cell"), [(True, "yes"), (False, "no"), (None, "-")])
def test_versions_table_with_limits(meets, cell):
    message = _message(limits=["tokens <= 4000", "Correctness >= 0.8"], versions=[_row(meets_limits=meets)])
    assert "best first for this study's goal (its limits: tokens <= 4000; Correctness >= 0.8)." in message
    assert "| Helpfulness | Meets the limits | Conversations |\n|---|---|---|---|---|---|---|\n" in message
    assert f"| 0.8 | {cell} | 8 |\n" in message


def test_digest_paragraph_and_labels_step():
    with_digest = _message(digest_file="/t/DIGEST.md")
    assert "**Trajectory Folder**: /t\n\n**Digest**: `/t/DIGEST.md` lists every conversation" in with_digest
    assert "**Step 3: Categorize the traces by their label**\n```bash\ncat /t/DIGEST.md\n```\n\n**Step 4" in with_digest

    without = _message()
    assert "**Trajectory Folder**: /t\n\nEach trace's `reward` labels" in without
    assert "for f in /t/*.json; do\n" in without
    assert 'grep -oE \'"reward":\\s*[0-9.-]+\' "$f" | head -1\n' in without
    assert "Digest" not in without


@pytest.mark.parametrize(
    ("ranked", "judged", "sentence"),
    [
        (False, [], "its Helpfulness.\n\n## The parameters"),
        (True, [], "its Helpfulness, ranked among the current version's conversations.\n\n"),
        (False, ["tokens"], "A conversation's tokens is not in its label: compare the versions for that.\n\n"),
        (False, ["tokens", "seconds"], "tokens and seconds are not in its label: compare the versions for those."),
    ],
)
def test_reward_sentence(ranked, judged, sentence):
    assert sentence in _message(label_ranked=ranked, judged_by_version=judged)


def test_parameter_sections():
    message = _message(
        variables=[
            TextVariable(
                name="prompt", value="Line one.\nLine two.", edit_instructions="the system prompt", size_cap=200
            ),
            TextVariable(name="tools.search", value="Searches.", edit_instructions=None, size_cap=None),
        ]
    )
    assert (
        "## The parameters (2)\n"
        '\n<parameter name="prompt">\nLine one.\nLine two.\n</parameter>\n\n'
        "- **What it is**: the system prompt\n- **Its limit**: at most 200 characters (it is 19 now)\n"
        '\n<parameter name="tools.search">\nSearches.\n</parameter>\n\n'
        "- **What it is**: text the agent reads\n- **Its limit**: none (it is 9 characters now)\n"
        "\n\n---\n\n## Analysis Steps"
    ) in message


def test_folders_are_filled_in_and_braces_kept():
    message = _message()
    assert "ls /v | head -40\ndiff -r /v/rank-001-version-* /v/rank-002-version-*\n" in message
    assert "cat > /w/optimized_<name>.txt << 'PARAM_EOF'\n" in message
    assert 'submit_optimized_params(file_path_dict={"<name>": "/w/optimized_<name>.txt", ...})' in message
    assert message.endswith("no multi-step task workflow, no unsupported guarantees.")


def test_a_value_is_shown_as_is():
    """Braces and format fields in a value are text, never formatted."""
    message = _message(
        variables=[TextVariable(name="p", value="Use {name} and {{x}}.", edit_instructions=None, size_cap=None)]
    )
    assert '<parameter name="p">\nUse {name} and {{x}}.\n</parameter>' in message


def test_versions_folder_layout_and_copying_a_text():
    message = _message()
    assert "one folder per version, named by its rank, with its results in `version.md`" in message
    assert "to start from a version's text, copy its `<name>.txt`" in message


def test_refused_message():
    assert T.REFUSED.format(refusal="Refused, nothing was submitted:\na: why") == (
        "Refused, nothing was submitted:\na: why\n\nYour turn ended with nothing submitted. Fix the file(s) in your "
        "working folder and submit again with submit_optimized_params."
    )


def test_over_limit_message():
    assert T.over_limit_message({"prompt": (2500, 2000)}) == (
        "`prompt` is over its limit: `prompt` is 2,500 characters, its limit 2,000 (500 over). A value over its limit "
        "is discarded and the current value kept. Shorten it to fit, keeping what matters most (check each file with "
        "`wc -c`), write it to its file in your working folder again, and submit with submit_optimized_params."
    )
    two = T.over_limit_message({"a": (12345, 2000), "b": (900, 800)})
    assert two.startswith("`a`, `b` are over their limit: `a` is 12,345 characters, its limit 2,000 (10,345 over); ")
    assert "Shorten them to fit" in two and "write them to their file" in two
