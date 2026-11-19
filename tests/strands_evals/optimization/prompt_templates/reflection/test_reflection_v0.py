"""Tests for Reflection's prompts: what varies in the task message, the fixed texts' placeholders, and the whole
texts against golden files."""

from pathlib import Path

import pytest

from strands_evals.optimization.prompt_templates.reflection import get_template, reflection_v0
from strands_evals.optimization.prompt_templates.reflection.reflection_v0 import TextVariable, VersionRow

T = get_template()

# The texts `reflection_v0` renders, each followed by a newline. A fix to `reflection_v0` updates its file here in the
# same change, so the review shows the new text; a new version gets its own folder.
_GOLDEN = Path(__file__).parent / "golden" / "reflection_v0"


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


def _assert_matches_golden(name, text, tmp_path):
    """`text` plus a newline is exactly `golden/reflection_v0/<name>`; on a mismatch, the rendered text is written to
    `tmp_path` to compare with or copy over the golden file."""
    rendered = text + "\n"
    golden = (_GOLDEN / name).read_text(encoding="utf-8")
    if rendered != golden:
        (tmp_path / name).write_text(rendered, encoding="utf-8", newline="\n")
    assert rendered == golden, f"the rendered text differs from {_GOLDEN / name}; it was written to {tmp_path / name}"


def test_system_prompt_matches_golden(tmp_path):
    """The default system prompt as Reflection sends it: the prompt, then the submit instructions."""
    text = reflection_v0.system_prompt("/reflection/work") + reflection_v0.SUBMIT_INSTRUCTIONS
    _assert_matches_golden("system_prompt.txt", text, tmp_path)


def test_task_message_with_limits_matches_golden(tmp_path):
    """The message Reflection would build for `Maximize("Correctness")` with `AtMost("tokens", 4000)`,
    `AtLeast("Correctness", 0.4)` and `success_threshold=None`: two limits, labels ranked among the current version's
    conversations, tokens judged by version, and the digest. The rows are the current version, the baseline, a
    version over a limit, and an ineligible one (unranked, limits unknown, measured means alone)."""
    message = reflection_v0.task_message(
        versions=[
            _row(
                rank=1,
                number=2,
                built_from=0,
                is_current=True,
                changed=["prompt"],
                cells=["0.84 (measured 0.9)", "3,120 (measured 3,050)"],
                meets_limits=True,
                conversations=12,
            ),
            _row(
                rank=2,
                number=0,
                built_from=None,
                is_baseline=True,
                changed=[],
                cells=["0.62 (measured 0.6)", "3,400 (measured 3,380)"],
                meets_limits=True,
                conversations=12,
            ),
            _row(
                rank=3,
                number=1,
                built_from=0,
                changed=["prompt", "tools.search", "other values"],
                cells=["0.71 (measured 0.75)", "4,410 (measured 4,380)"],
                meets_limits=False,
                conversations=11,
            ),
            _row(
                rank=None,
                number=3,
                built_from=2,
                changed=["tools.search"],
                cells=["1", "2,980"],
                meets_limits=None,
                conversations=1,
            ),
        ],
        versions_folder="/reflection/versions",
        goal_metrics=["Correctness", "tokens"],
        limits=["tokens <= 4000", "Correctness >= 0.4"],
        traces_folder="/reflection/traces",
        digest_file="/reflection/traces/DIGEST.md",
        label_metrics=["Correctness"],
        label_ranked=True,
        judged_by_version=["tokens"],
        variables=[
            TextVariable(
                name="prompt",
                value='You are a support agent.\nCite each article as "<title> (id {id})" & quote it exactly.',
                edit_instructions="the agent's system prompt",
                size_cap=2000,
            ),
            TextVariable(name="tools.search", value="Searches the help center.", edit_instructions=None, size_cap=None),
        ],
        work_folder="/reflection/work",
    )
    _assert_matches_golden("task_message_with_limits.txt", message, tmp_path)


def test_task_message_without_limits_matches_golden(tmp_path):
    """No limits, labels from two bounded metrics (not ranked), two metrics judged by version, and no digest, so the
    labels come from the trace files: each the other way from the case with limits. Reflection always writes a
    digest; this case pins the template's text without one."""
    message = reflection_v0.task_message(
        versions=[
            _row(
                rank=1,
                number=1,
                built_from=0,
                is_current=True,
                changed=["prompt"],
                cells=["0.8 (measured 0.83)", "0.74 (measured 0.7)", "2,410 (measured 2,380)", "8.2 (measured 8.4)"],
                meets_limits=None,
                conversations=9,
            ),
            _row(
                rank=2,
                number=0,
                built_from=None,
                is_baseline=True,
                changed=[],
                cells=["0.66 (measured 0.67)", "0.7 (measured 0.72)", "2,950 (measured 3,010)", "9.6 (measured 9.1)"],
                meets_limits=None,
                conversations=9,
            ),
        ],
        versions_folder="/reflection/versions",
        goal_metrics=["Correctness", "Helpfulness", "tokens", "seconds"],
        limits=[],
        traces_folder="/reflection/traces",
        digest_file=None,
        label_metrics=["Correctness", "Helpfulness"],
        label_ranked=False,
        judged_by_version=["tokens", "seconds"],
        variables=[TextVariable(name="prompt", value="Be brief.", edit_instructions=None, size_cap=1500)],
        work_folder="/reflection/work",
    )
    _assert_matches_golden("task_message_without_limits.txt", message, tmp_path)
