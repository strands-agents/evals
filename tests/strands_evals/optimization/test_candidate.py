import json
from pathlib import Path

import pytest

from strands_evals.optimization import (
    Candidate,
    FileText,
    Float,
    Int,
    JsonField,
    OneOf,
    Space,
    TextValue,
    choice,
    component,
)
from strands_evals.optimization.candidate import changed_keys, changed_values, content_id, diff


class Window:
    def __init__(self, size: int = 40):
        self.size = size


class Summary:
    def __init__(self, ratio: float = 0.3):
        self.ratio = ratio


def make_space():
    return Space(
        {
            "manager": choice(component(Window, size=Int(1, 100)), component(Summary, ratio=Float(0.1, 0.9))),
            "temperature": Float(0.0, 1.0, default=0.7),
            "effort": OneOf("low", "high"),
            "prompt": TextValue(default="You help.\n"),
        }
    )


# content_id


def test_identical_content_gives_an_identical_id_whatever_the_key_order():
    assert content_id({"a": 1, "b": "x"}) == content_id({"b": "x", "a": 1})
    assert len(content_id({"a": 1})) == 12


def test_types_count_in_the_id():
    assert len({content_id({"a": 1}), content_id({"a": 1.0}), content_id({"a": True})}) == 3


def test_content_id_rejects_nan():
    with pytest.raises(ValueError):
        content_id({"a": float("nan")})


def test_values_reloaded_from_json_give_the_same_id():
    space = make_space()
    candidate = Candidate.baseline(space).child(
        space, {"temperature": 0.1, "effort": "high"}, origin=["numbers"], rationale=""
    )
    reloaded = json.loads(json.dumps(dict(candidate.values)))
    assert Candidate(reloaded, candidate.parent_id, candidate.origin, candidate.rationale).id == candidate.id
    space.validate(reloaded)  # typed values round-trip: ints stay ints, floats stay floats


# Candidate


def test_values_are_a_read_only_copy():
    values = {"temperature": 0.5}
    candidate = Candidate(values, None, ("baseline",), "")
    values["temperature"] = 0.9
    assert candidate.values["temperature"] == 0.5
    with pytest.raises(TypeError):
        candidate.values["temperature"] = 0.1  # type: ignore[index]


def test_equality_and_hash_follow_the_id():
    a = Candidate({"x": 1}, None, ("baseline",), "")
    b = Candidate({"x": 1}, "parent", ("numbers",), "another reason")
    assert a == b
    assert len({a, b}) == 1
    assert a != Candidate({"x": 2}, None, ("baseline",), "")


def test_the_id_cannot_go_stale():
    candidate = Candidate({"x": 1}, None, ("baseline",), "")
    with pytest.raises(AttributeError):
        candidate.values = {"x": 2}  # type: ignore[misc]


def test_baseline_takes_every_default():
    baseline = Candidate.baseline(make_space())
    assert dict(baseline.values) == {
        "manager": "Window",
        "manager.Window.size": 40,
        "temperature": 0.7,
        "effort": "low",
        "prompt": "You help.\n",
    }
    assert baseline.parent_id is None
    assert baseline.origin == ("baseline",)


def test_child_applies_updates_and_keeps_the_rest():
    space = make_space()
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"temperature": 0.2}, origin=["numbers"], rationale="cooler")
    assert child.values["temperature"] == 0.2
    assert child.values["manager.Window.size"] == 40
    assert (child.parent_id, child.origin, child.rationale) == (baseline.id, ("numbers",), "cooler")
    assert child.inherited == {} and child.defaulted == ()


def test_a_changed_choice_drops_old_keys_and_defaults_new_ones():
    space = make_space()
    child = Candidate.baseline(space).child(space, {"manager": "Summary"}, origin=["structure"], rationale="")
    assert "manager.Window.size" not in child.values
    assert child.values["manager.Summary.ratio"] == 0.3
    assert child.defaulted == ("manager.Summary.ratio",)


def test_a_changed_choice_inherits_from_fill():
    space = make_space()
    sources = {"manager.Summary.ratio": (0.6, "abc123")}
    child = Candidate.baseline(space).child(
        space, {"manager": "Summary"}, origin=["structure"], rationale="", fill=sources.get
    )
    assert child.values["manager.Summary.ratio"] == 0.6
    assert child.inherited == {"manager.Summary.ratio": "abc123"}
    assert child.defaulted == ()


def test_child_rejects_invalid_values():
    space = make_space()
    with pytest.raises(ValueError, match="outside"):
        Candidate.baseline(space).child(space, {"temperature": 3.0}, origin=["numbers"], rationale="")
    with pytest.raises(ValueError, match="not keys of this space"):
        Candidate.baseline(space).child(space, {"temprature": 0.3}, origin=["numbers"], rationale="")


def test_child_validates_the_values_it_stores(tmp_path):
    (tmp_path / "agent.json").write_text('{"tags": [1, 2]}\n', newline="")
    space = Space({"tags": JsonField(tmp_path / "agent.json", "/tags", size_cap=12)})
    # 9 characters as proposed, 22 once written as the file holds it
    with pytest.raises(ValueError, match="over the size_cap of 12"):
        Candidate.baseline(space).child(space, {"tags": "[1,2,3,4]"}, origin=["text"], rationale="")


def test_child_rejects_an_update_to_an_inactive_key():
    space = make_space()
    with pytest.raises(ValueError, match="inactive"):
        Candidate.baseline(space).child(space, {"manager.Summary.ratio": 0.5}, origin=["numbers"], rationale="")


# changed_keys


def test_changed_keys_include_keys_on_one_side_only():
    space = make_space()
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"manager": "Summary", "prompt": "New.\n"}, origin=["structure"], rationale="")
    assert changed_keys(baseline, child) == ["manager", "manager.Summary.ratio", "manager.Window.size", "prompt"]
    assert changed_keys(baseline, baseline) == []


def test_changed_keys_tell_types_apart():
    assert changed_keys(
        Candidate({"x": 1}, None, ("baseline",), ""), Candidate({"x": 1.0}, None, ("baseline",), "")
    ) == ["x"]


# changed_values


def test_changed_values_has_one_side_for_a_key_active_on_one_side():
    space = make_space()
    baseline = Candidate.baseline(space)
    child = baseline.child(
        space, {"manager": "Summary", "temperature": 0.1, "prompt": "New.\n"}, origin=["structure"], rationale=""
    )
    assert changed_values(space, baseline, child) == {
        "manager": {"old": "Window", "new": "Summary"},
        "manager.Window.size": {"old": 40},
        "manager.Summary.ratio": {"new": 0.3},
        "temperature": {"old": 0.7, "new": 0.1},
    }
    assert json.loads(json.dumps(changed_values(space, baseline, child))) == changed_values(space, baseline, child)


def test_changed_values_is_empty_when_only_text_changed():
    space = make_space()
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"prompt": "New.\n"}, origin=["text"], rationale="")
    assert changed_values(space, baseline, child) == {}


def test_changed_values_keeps_a_none_value():
    space = Space({"pin": OneOf(None, 1)})
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"pin": 1}, origin=["numbers"], rationale="")
    assert changed_values(space, baseline, child) == {"pin": {"old": None, "new": 1}}


# diff


@pytest.fixture
def project(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system.md").write_text("You help.\nBe kind.\n", newline="")
    (tmp_path / "agent.json").write_text(
        json.dumps({"agent": {"prompt": "Be brief.", "turns": 4}}, indent=2) + "\n", newline=""
    )
    return tmp_path


def file_space(project):
    return Space(
        {
            "system": FileText(project / "prompts" / "system.md"),
            "agent_prompt": JsonField(project / "agent.json", "/agent/prompt"),
            "tone": TextValue(default="calm"),
            "temperature": Float(0.0, 1.0, default=0.7),
        }
    )


def test_diff_is_empty_when_no_text_changed(project):
    space = file_space(project)
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"temperature": 0.1}, origin=["numbers"], rationale="")
    assert diff(space, baseline, child, root=project) == ""


def test_diff_puts_file_values_on_their_real_paths(project):
    space = file_space(project)
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"system": "You help.\nBe direct.\n"}, origin=["text"], rationale="")
    assert diff(space, baseline, child, root=project) == (
        "--- a/prompts/system.md\n+++ b/prompts/system.md\n@@ -1,2 +1,2 @@\n You help.\n-Be kind.\n+Be direct.\n"
    )


def test_diff_shows_a_json_field_on_its_whole_file(project):
    space = file_space(project)
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"agent_prompt": "Be thorough."}, origin=["text"], rationale="")
    text = diff(space, baseline, child, root=project)
    assert text.startswith("--- a/agent.json\n+++ b/agent.json\n")
    assert '-    "prompt": "Be brief.",\n+    "prompt": "Be thorough.",\n' in text
    assert '"turns"' in text  # context from the rest of the file


def test_diff_heads_non_file_text_with_its_key_and_never_shows_numbers(project):
    space = file_space(project)
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"tone": "warm", "temperature": 0.1}, origin=["text"], rationale="")
    assert diff(space, baseline, child, root=project) == (
        "--- tone\n+++ tone\n@@ -1 +1 @@\n-calm\n\\ No newline at end of file\n+warm\n\\ No newline at end of file\n"
    )


@pytest.mark.parametrize("separator", ["\f", "\v", "\r", "\x85", "\u2028"])
def test_diff_splits_lines_only_at_newlines(separator):
    space = Space({"notes": TextValue(default=f"one{separator}still one\nend\n")})
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"notes": f"one{separator}still one\nlast\n"}, origin=["text"], rationale="")
    assert diff(space, baseline, child, root=Path.cwd()) == (
        f"--- notes\n+++ notes\n@@ -1,2 +1,2 @@\n one{separator}still one\n-end\n+last\n"
    )


def test_diff_files_first_then_non_file_text(project):
    space = file_space(project)
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"tone": "warm", "system": "Hi.\n"}, origin=["text"], rationale="")
    text = diff(space, baseline, child, root=project)
    assert text.index("--- a/prompts/system.md") < text.index("--- tone")


def test_diff_uses_the_absolute_path_outside_root(project, tmp_path_factory):
    space = file_space(project)
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"system": "Hi.\n"}, origin=["text"], rationale="")
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    path = (project / "prompts" / "system.md").resolve().as_posix()
    assert diff(space, baseline, child, root=elsewhere).startswith(f"--- {path}\n+++ {path}\n")


def test_diff_never_reads_the_files_again(project):
    space = file_space(project)
    baseline = Candidate.baseline(space)
    child = baseline.child(space, {"system": "Hi.\n"}, origin=["text"], rationale="")
    (project / "prompts" / "system.md").write_text("edited after declaration\n", newline="")
    assert "-Be kind.\n" in diff(space, baseline, child, root=project)


def test_a_reformatted_json_value_is_the_same_candidate(tmp_path):
    (tmp_path / "agent.json").write_text('{"temperature": 0.5}\n', newline="")
    space = Space({"temperature": JsonField(tmp_path / "agent.json", "/temperature")})
    baseline = Candidate.baseline(space)
    same = baseline.child(space, {"temperature": "0.50"}, origin=["text"], rationale="")
    assert same.id == baseline.id
    assert diff(space, baseline, same, root=tmp_path) == ""
