"""Tests for `Reflection`'s tools (the confined shell, the submit tool, the output cap) and its case-name screen."""

import os
import time
from types import SimpleNamespace

import pytest

from strands_evals.optimization.search import reflection
from strands_evals.optimization.search.reflection import (
    Reflection,
    _cap_tool_output,
    _confined_shell,
    _leak,
    _outside,
    _submit_tool,
    _text_files,
)

posix_only = pytest.mark.skipif(os.name == "nt", reason="Reflection needs Linux or macOS")


@pytest.fixture
def folders(tmp_path):
    """A work folder and a traces folder, the agent's two roots, plus a folder outside both."""
    work, traces, other = tmp_path / "work", tmp_path / "traces", tmp_path / "other"
    for folder in (work, traces, other):
        folder.mkdir()
    return SimpleNamespace(work=str(work), traces=str(traces), other=str(other), roots=[str(traces), str(work)])


async def test_propose_raises_on_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    with pytest.raises(RuntimeError, match="needs Linux or macOS.*WSL"):
        await Reflection().propose(SimpleNamespace(n=1))


@posix_only
def test_outside_lists_private_paths_outside_the_roots(folders):
    secret = os.path.join(folders.other, "secret.txt")
    command = f"cat {folders.traces}/a.json {secret} && ls /tmp/shared"
    assert _outside(command, folders.roots, folders.work) == sorted([secret, "/tmp/shared"])


@posix_only
def test_outside_lists_parent_references_and_allows_relative_paths(folders):
    assert _outside("cat ../other/a.json && ls ..", folders.roots, folders.work) == ["..", "../other/a.json"]
    assert _outside("cat ../traces/a.json", folders.roots, folders.work) == []  # lands in a root
    assert _outside("cat notes.txt && grep -r x .", folders.roots, folders.work) == []


@posix_only
def test_outside_lets_the_folder_holding_the_roots_be_named_but_never_reached_by_dots(folders, tmp_path):
    """`home` may be named by its absolute path; a `..` path must land in a root, so `..` can't climb from it."""
    home = str(tmp_path)
    roots, work = folders.roots, folders.work
    assert _outside(f"cd {home} && cat traces/a.json && ls {home}/work", roots, work, home) == []
    assert _outside("cat ../traces/a.json; ls ../work", roots, work, home) == []
    assert _outside("ls .. && cd ../.. && cat ../../x", roots, work, home) == ["..", "../..", "../../x"]
    assert _outside(f"ls {home}/..", roots, work, home) == [f"{home}/.."]


@posix_only
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


@posix_only
def test_outside_is_not_a_sandbox(folders):
    """Paths outside the private directories are not checked, and a built path is not seen."""
    assert _outside("cat /etc/hosts", folders.roots, folders.work) == []
    assert _outside('cat "$(printf /tm)p/x"', folders.roots, folders.work) == []


@posix_only
def test_shell_runs_in_the_work_folder(folders):
    shell = _confined_shell(folders.work, folders.roots, [])
    out = shell(command='pwd; echo "$HOME" "$TMPDIR"; echo oops >&2; exit 3')
    lines = out.splitlines()
    assert os.path.realpath(lines[0]) == os.path.realpath(folders.work)
    assert lines[1] == f"{folders.work} {folders.work}"
    assert "[stderr]\noops" in out
    assert out.endswith("[exit code 3]")


@posix_only
def test_shell_keeps_the_process_environment(folders, monkeypatch):
    monkeypatch.setenv("REFLECTION_TEST_VAR", "kept")
    shell = _confined_shell(folders.work, folders.roots, [])
    assert shell(command="echo $REFLECTION_TEST_VAR").startswith("kept\n")


@posix_only
def test_shell_refuses_and_records_a_command_outside_its_folders(folders):
    refused = []
    shell = _confined_shell(folders.work, folders.roots, refused)
    target = os.path.join(folders.other, "written.txt")
    out = shell(command=f"echo x > {target}")
    assert out.startswith(f"Refused: {target} is outside your folders")
    assert not os.path.exists(target)
    assert refused == [{"command": f"echo x > {target}", "paths": [target]}]
    assert "shared folder" not in out  # none given


@posix_only
def test_a_refusal_names_the_shared_folder_when_there_is_one(folders, tmp_path):
    shell = _confined_shell(folders.work, folders.roots, [], str(tmp_path))
    assert shell(command="ls ..").endswith(
        f"Their shared folder, {tmp_path}, may be named by its absolute path, never reached by `..`."
    )


@posix_only
def test_shell_timeout_is_kept_between_1_and_3600_seconds(folders):
    shell = _confined_shell(folders.work, folders.roots, [])
    assert shell(command="sleep 0.2; echo done", timeout=0) == "done\n\n[exit code 0]"
    assert shell(command="sleep 5", timeout=-3).endswith(
        "[stopped after 1 seconds: the command and what it started were killed]"
    )


@posix_only
def test_shell_reads_its_roots(folders):
    with open(os.path.join(folders.traces, "a.json"), "w") as f:
        f.write("trace")
    shell = _confined_shell(folders.work, folders.roots, [])
    assert shell(command=f"cat {folders.traces}/a.json").startswith("trace")


@posix_only
def test_shell_stops_a_command_at_its_timeout(folders):
    shell = _confined_shell(folders.work, folders.roots, [])
    assert shell(command="sleep 5", timeout=1) == (
        "\n[stopped after 1 seconds: the command and what it started were killed]"
    )


@posix_only
def test_shell_keeps_what_a_timed_out_command_printed(folders):
    shell = _confined_shell(folders.work, folders.roots, [])
    assert shell(command="echo hello; echo err >&2; sleep 5", timeout=1) == (
        "hello\n\n[stderr]\nerr\n\n[stopped after 1 seconds: the command and what it started were killed]"
    )


@posix_only
def test_shell_kills_what_a_timed_out_command_started(folders):
    """A child of `bash` (not exec'd: `cd . &&`) dies with it, so its file is never written."""
    shell = _confined_shell(folders.work, folders.roots, [])
    assert "[stopped after 1 seconds" in shell(command="cd . && (sleep 2; touch late.txt)", timeout=1)
    time.sleep(2.5)
    assert not os.path.exists(os.path.join(folders.work, "late.txt"))


@posix_only
def test_shell_returns_when_bash_exits_and_kills_a_background_process(folders):
    shell = _confined_shell(folders.work, folders.roots, [])
    started = time.monotonic()
    out = shell(command="(sleep 2; touch late.txt) & echo started", timeout=30)
    assert time.monotonic() - started < 1.5
    assert out == "started\n\n[exit code 0]"
    time.sleep(2.5)
    assert not os.path.exists(os.path.join(folders.work, "late.txt"))


@posix_only
def test_shell_replaces_output_that_is_not_utf8(folders):
    shell = _confined_shell(folders.work, folders.roots, [])
    assert shell(command=r"printf 'ok \xe2\x82'; printf '\xff' >&2") == "ok �\n[stderr]\n�\n[exit code 0]"


def test_submit_reads_files_in_the_work_folder(folders):
    with open(os.path.join(folders.work, "prompt.txt"), "w") as f:
        f.write("  New prompt.\n")
    submitted = {}
    submit = _submit_tool(folders.work, submitted, {}, [])
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
        ("\n  Indented.\n", "Indented, again.", "\n  Indented, again.\n"),  # leading whitespace too
        ("\n  Indented.\n", "\n  Indented.\n", "\n  Indented.\n"),  # unchanged, whatever its edges
        ("\n", "", "\n"),  # a parent of whitespace alone: an empty rewrite is unchanged
        ("\n", "Text.", "Text.\n"),
        ("Old.\n", "", "\n"),  # an empty value is kept (validation decides)
    ],
)
def test_submit_takes_the_parents_trailing_whitespace(folders, parent, written, expected):
    with open(os.path.join(folders.work, "value.txt"), "w") as f:
        f.write(written)
    submitted = {}
    _submit_tool(folders.work, submitted, {"prompt": parent}, [])(file_path_dict={"prompt": "value.txt"})
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
    out = _submit_tool(folders.work, submitted, {}, [])(file_path_dict={"system_prompt": path})
    assert out.startswith(f"Errors:\nsystem_prompt: {path} is not in your working folder")
    assert submitted == {}


def test_submit_is_all_or_nothing(folders):
    with open(os.path.join(folders.work, "a.txt"), "w") as f:
        f.write("a")
    submitted = {"earlier": "kept"}
    out = _submit_tool(folders.work, submitted, {}, [])(file_path_dict={"a": "a.txt", "b": "missing.txt"})
    assert out.startswith("Errors:\nb: file not found:")
    assert submitted == {"earlier": "kept"}


@pytest.mark.parametrize(
    "header",
    ["# Version 26\nrank: 1\n", "# Version 0 (the baseline)\r\nbuilt from: nothing (the start)\n"],
)
def test_submit_refuses_a_value_with_a_versions_header(folders, header):
    with open(os.path.join(folders.work, "a.txt"), "w", newline="") as f:  # the header's own line endings, as written
        f.write(f"{header}conversations: 8\n\nBe helpful.")
    with open(os.path.join(folders.work, "b.txt"), "w") as f:
        f.write("Searches.")
    submitted, rejected = {"earlier": "kept"}, []
    out = _submit_tool(folders.work, submitted, {}, rejected)(file_path_dict={"a": "a.txt", "b": "b.txt"})
    first = header.splitlines()[0]
    reason = (
        f"it holds a version's header from the versions folder (`{first}` and the results below it), which is not "
        "part of the text. Write the text alone to the file (a version's text is alone in its `.txt` file, never in "
        "version.md) and submit it again"
    )
    assert out == f"Refused, nothing was submitted:\na: {reason}"
    assert submitted == {"earlier": "kept"}
    assert rejected == [{"a": reason}]


@pytest.mark.parametrize("text", ["# Version 2 of the prompt\nBe brief.", "Intro.\n# Version 3\nmore", "rank: 1"])
def test_submit_takes_a_value_that_only_looks_like_a_header(folders, text):
    with open(os.path.join(folders.work, "a.txt"), "w") as f:
        f.write(text)
    submitted, rejected = {}, []
    _submit_tool(folders.work, submitted, {}, rejected)(file_path_dict={"a": "a.txt"})
    assert submitted == {"a": text}
    assert rejected == []


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


@pytest.mark.parametrize("name", ["order-123", "case_7", "RefundFlow", "getUser", "HTTPServer"])
def test_leak_screens_an_identifier_like_case_name(name):
    assert _leak(f"Handle {name} first.", "", [name], []) == f"adds the name of a shown case, {name!r}"


@pytest.mark.parametrize("name", ["refund", "Refund", "API", "FAQ", "OK"])
def test_leak_skips_plain_words_and_acronyms(name):
    assert _leak(f"Handle {name} first.", "", [name], []) is None


@posix_only
@pytest.mark.parametrize(
    "command",
    ["cd ..;ls", "ls ..|head", "echo $(ls ..)", "cd ..;cd ..;ls", "ls ~/../..", "cat a/../../x", "cd .. &&ls"],
)
def test_outside_sees_dots_before_shell_punctuation_and_under_home(folders, tmp_path, command):
    assert _outside(command, folders.roots, folders.work, str(tmp_path)) != []


@posix_only
def test_outside_sees_a_quoted_path_followed_by_dots_and_a_leading_double_slash(folders, tmp_path):
    home = str(tmp_path)
    for command in [f"ls '{home}'/..", f'cat "{home}"/../other/x', "cat '../traces'/../../x", "cat //tmp/x"]:
        assert _outside(command, folders.roots, folders.work, home) != [], command
    assert _outside("cat '../traces/a.json'", folders.roots, folders.work, home) == []


@posix_only
@pytest.mark.parametrize("command", ["cat 'a /'../../x", "PATH=a:'/tmp'", "sh -c 'cat /tmp/x'"])
def test_outside_reads_a_command_both_as_written_and_without_quotes(folders, tmp_path, command):
    """Removing quotes splits `'a /'..` at the space; as written, `'/tmp` is still seen after the quote."""
    assert _outside(command, folders.roots, folders.work, str(tmp_path)) != []


@posix_only
def test_outside_sees_a_path_joined_to_a_short_option(folders):
    assert _outside("tar -C/tmp -xf a.tar", folders.roots, folders.work) == ["/tmp"]
    assert _outside("cc -I/usr/include x.c && ls -la", folders.roots, folders.work) == []
    assert _outside("grep -rn x --include=*.json .", folders.roots, folders.work) == []


@posix_only
def test_outside_refuses_dots_in_a_command_that_names_the_shared_folder(folders, tmp_path):
    """After `cd <shared folder>`, `../traces` starts from above it, not from the work folder."""
    home = str(tmp_path)
    assert _outside(f"cd {home} && cat ../traces/a.json", folders.roots, folders.work, home) == ["../traces/a.json"]
    assert _outside(f"cd {folders.traces} && cat ../traces/a.json", folders.roots, folders.work, home) == []


@posix_only
def test_outside_allows_home_and_an_assignment_into_a_root(folders):
    assert _outside("cat ~/notes.txt; T=../traces/a.json; cat $T", folders.roots, folders.work) == []


@posix_only
def test_shell_gives_a_command_no_standard_input(folders):
    """With the optimizer's stdin an open pipe, a bare `cat` still ends at once instead of reading it or waiting for
    its timeout."""
    read, write = os.pipe()
    saved = os.dup(0)
    os.dup2(read, 0)
    try:
        os.write(write, b"the optimizer's input\n")
        shell = _confined_shell(folders.work, folders.roots, [])
        started = time.monotonic()
        assert shell(command="cat; echo done", timeout=5) == "done\n\n[exit code 0]"
        assert time.monotonic() - started < 2
    finally:
        os.dup2(saved, 0)
        for fd in (saved, read, write):
            os.close(fd)


@posix_only
def test_a_running_command_is_in_running_until_it_ends_and_can_be_killed(folders):
    """The caller kills what `running` holds when its call ends early (cancelled)."""
    import threading

    running = set()
    shell = _confined_shell(folders.work, folders.roots, [], running=running)
    out = []
    thread = threading.Thread(target=lambda: out.append(shell(command="sleep 30", timeout=60)))
    thread.start()
    while not running:
        time.sleep(0.01)
    started = time.monotonic()
    for process in list(running):
        reflection._kill_group(process)
    thread.join(5)
    assert time.monotonic() - started < 2 and out == ["\n[exit code -9]"] and running == set()


def test_submit_merges_a_second_submission_in_the_same_turn(folders):
    """A submission replaces only the values it names; the turn's earlier ones are kept."""
    for name, text in {"p.txt": "New prompt.", "t1.txt": "warm", "t2.txt": "warmer"}.items():
        with open(os.path.join(folders.work, name), "w") as f:
            f.write(text)
    submitted = {}
    submit = _submit_tool(folders.work, submitted, {}, [])
    submit(file_path_dict={"prompt": "p.txt", "tone": "t1.txt"})
    submit(file_path_dict={"tone": "t2.txt"})
    assert submitted == {"prompt": "New prompt.", "tone": "warmer"}


def test_submit_restores_the_parents_whitespace_only_when_it_fits(folders):
    with open(os.path.join(folders.work, "p.txt"), "w") as f:
        f.write("x" * 10)
    submitted = {}
    _submit_tool(folders.work, submitted, {"p": "old\n"}, [], {"p": 10})(file_path_dict={"p": "p.txt"})
    assert submitted == {"p": "x" * 10}  # "x" * 10 + "\n" would be 11, over the cap of 10
    _submit_tool(folders.work, submitted, {"p": "old\n"}, [], {"p": 11})(file_path_dict={"p": "p.txt"})
    assert submitted == {"p": "x" * 10 + "\n"}


@pytest.mark.parametrize("name", ["order-1042", "case_7", "RefundFlow", "q12a"])
def test_leak_screens_a_long_identifier_like_name(name):
    assert _leak(f"Handle {name} first.", "", [name], []) is not None


@pytest.mark.parametrize("name", ["3", "v2", "a_1", "1234"])
def test_leak_skips_short_or_letterless_names(name):
    assert _leak(f"Answer in at most {name} sentences.", "", [name], []) is None


def test_text_files_never_differ_only_by_case():
    files = _text_files(["Prompt", "prompt", "VERSION.md"])
    assert len({f.casefold() for f in files.values()}) == 3
    assert files["prompt"] == "prompt-2.txt"
