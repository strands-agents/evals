"""Tests for `Reflection`'s tools: the confined shell, the submit tool, and the tool output cap."""

import os
from types import SimpleNamespace

import pytest

from strands_evals.optimization.search import reflection
from strands_evals.optimization.search.reflection import (
    Reflection,
    _cap_tool_output,
    _confined_shell,
    _outside,
    _submit_tool,
)


@pytest.fixture
def folders(tmp_path):
    """A work folder and a traces folder, the agent's two roots, plus a folder outside both."""
    work, traces, other = tmp_path / "work", tmp_path / "traces", tmp_path / "other"
    for folder in (work, traces, other):
        folder.mkdir()
    return SimpleNamespace(work=str(work), traces=str(traces), other=str(other), roots=[str(traces), str(work)])


def test_outside_lists_private_paths_outside_the_roots(folders):
    secret = os.path.join(folders.other, "secret.txt")
    command = f"cat {folders.traces}/a.json {secret} && ls /tmp/shared"
    assert _outside(command, folders.roots, folders.work) == sorted([secret, "/tmp/shared"])


def test_outside_lists_parent_references_and_allows_relative_paths(folders):
    assert _outside("cat ../traces/a.json", folders.roots, folders.work) == [".."]
    assert _outside("cat notes.txt && grep -r x .", folders.roots, folders.work) == []


def test_outside_accepts_a_roots_real_path(tmp_path):
    """A root reached through a symlink (macOS: `/var` -> `/private/var`) is inside by either path, as `pwd` prints
    the real one."""
    real = tmp_path / "real"
    (real / "work").mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(real)
    work = str(linked / "work")
    for root in (work, str(real / "work")):
        assert _outside(f"cat {real}/work/a.txt {linked}/work/b.txt", [root], work) == []


def test_outside_is_not_a_sandbox(folders):
    """Paths outside the private directories are not checked, and a built path is not seen."""
    assert _outside("cat /etc/hosts", folders.roots, folders.work) == []
    assert _outside('cat "$(printf /tm)p/x"', folders.roots, folders.work) == []


def test_shell_runs_in_the_work_folder(folders):
    shell = _confined_shell(folders.work, folders.roots, [])
    out = shell(command='pwd; echo "$HOME" "$TMPDIR"; echo oops >&2; exit 3')
    lines = out.splitlines()
    assert os.path.realpath(lines[0]) == os.path.realpath(folders.work)
    assert lines[1] == f"{folders.work} {folders.work}"
    assert "[stderr]\noops" in out
    assert out.endswith("[exit code 3]")


def test_shell_keeps_the_process_environment(folders, monkeypatch):
    monkeypatch.setenv("REFLECTION_TEST_VAR", "kept")
    shell = _confined_shell(folders.work, folders.roots, [])
    assert shell(command="echo $REFLECTION_TEST_VAR").startswith("kept\n")


def test_shell_refuses_and_records_a_command_outside_its_folders(folders):
    refused = []
    shell = _confined_shell(folders.work, folders.roots, refused)
    target = os.path.join(folders.other, "written.txt")
    out = shell(command=f"echo x > {target}")
    assert out.startswith(f"Refused: {target} is outside your folders")
    assert not os.path.exists(target)
    assert refused == [{"command": f"echo x > {target}", "paths": [target]}]


def test_shell_reads_its_roots(folders):
    with open(os.path.join(folders.traces, "a.json"), "w") as f:
        f.write("trace")
    shell = _confined_shell(folders.work, folders.roots, [])
    assert shell(command=f"cat {folders.traces}/a.json").startswith("trace")


def test_shell_stops_a_command_at_its_timeout(folders):
    shell = _confined_shell(folders.work, folders.roots, [])
    assert shell(command="sleep 5", timeout=1) == "Stopped after 1 seconds."


def test_submit_reads_files_in_the_work_folder(folders):
    with open(os.path.join(folders.work, "prompt.txt"), "w") as f:
        f.write("  New prompt.\n")
    submitted = {}
    submit = _submit_tool(folders.work, submitted, {})
    out = submit(file_path_dict={"system_prompt": "prompt.txt"})
    assert out == "Parameters submitted successfully: system_prompt (11 chars)"
    assert submitted == {"system_prompt": "New prompt."}


@pytest.mark.parametrize(
    ("parent", "written", "expected"),
    [
        ("Old line.\n", "New line.", "New line.\n"),  # a file's final newline survives
        ("Old.\n\n", "  New.  \n", "New.\n\n"),  # the parent's trailing whitespace, whatever it is
        ("Old.", "New.\n", "New."),  # no trailing whitespace in the parent: none added
        ("Same text.\n", "Same text.\n", "Same text.\n"),  # an unchanged rewrite equals the parent's value
    ],
)
def test_submit_takes_the_parents_trailing_whitespace(folders, parent, written, expected):
    with open(os.path.join(folders.work, "value.txt"), "w") as f:
        f.write(written)
    submitted = {}
    _submit_tool(folders.work, submitted, {"prompt": parent})(file_path_dict={"prompt": "value.txt"})
    assert submitted == {"prompt": expected}


@pytest.mark.parametrize("where", ["absolute", "parent", "symlink"])
def test_submit_rejects_a_file_outside_the_work_folder(folders, where):
    outside_file = os.path.join(folders.other, "prompt.txt")
    with open(outside_file, "w") as f:
        f.write("outside")
    path = {"absolute": outside_file, "parent": "../other/prompt.txt", "symlink": "link.txt"}[where]
    if where == "symlink":
        os.symlink(outside_file, os.path.join(folders.work, "link.txt"))
    submitted = {}
    out = _submit_tool(folders.work, submitted, {})(file_path_dict={"system_prompt": path})
    assert out.startswith(f"Errors:\nsystem_prompt: {path} is not in your working folder")
    assert submitted == {}


def test_submit_is_all_or_nothing(folders):
    with open(os.path.join(folders.work, "a.txt"), "w") as f:
        f.write("a")
    submitted = {"earlier": "kept"}
    out = _submit_tool(folders.work, submitted, {})(file_path_dict={"a": "a.txt", "b": "missing.txt"})
    assert out.startswith("Errors:\nb: file not found:")
    assert submitted == {"earlier": "kept"}


def test_cap_tool_output_cuts_long_text(monkeypatch):
    monkeypatch.setattr(reflection, "_MAX_TOOL_OUTPUT", 10)
    event = SimpleNamespace(result={"content": [{"text": "x" * 25}, {"text": "short"}, {"json": {"a": 1}}]})
    _cap_tool_output(event)
    long, short, other = event.result["content"]
    assert long["text"].startswith("x" * 10 + "\n\n[TRUNCATED: Output exceeded 10 characters. Original length: 25")
    assert short == {"text": "short"}
    assert other == {"json": {"a": 1}}


def test_cap_tool_output_limit_is_100000():
    assert reflection._MAX_TOOL_OUTPUT == 100_000


def test_over_limit_retries_in_config():
    assert Reflection().config()["over_limit_retries"] == 2
    assert Reflection(over_limit_retries=0).config()["over_limit_retries"] == 0


@pytest.mark.parametrize("retries", [1.5, True, "2"])
def test_over_limit_retries_must_be_an_integer(retries):
    with pytest.raises(TypeError, match="over_limit_retries"):
        Reflection(over_limit_retries=retries)


def test_over_limit_retries_must_be_at_least_0():
    with pytest.raises(ValueError, match="over_limit_retries"):
        Reflection(over_limit_retries=-1)


@pytest.mark.parametrize("threshold", [True, "0.5"])
def test_success_threshold_must_be_a_number_or_none(threshold):
    with pytest.raises(TypeError, match="success_threshold"):
        Reflection(success_threshold=threshold)


@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan"), float("inf")])
def test_success_threshold_must_be_in_0_1(threshold):
    with pytest.raises(ValueError, match="success_threshold"):
        Reflection(success_threshold=threshold)


def test_observe_keeps_nothing():
    proposer = Reflection()
    assert proposer.observe(SimpleNamespace()) is None
    assert proposer.refused == []
