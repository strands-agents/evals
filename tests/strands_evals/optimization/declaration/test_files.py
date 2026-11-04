import gc
import json
import os

import pytest

from strands_evals.optimization import FileText, JsonField, Space, TextValue, tunable
from strands_evals.optimization.declaration.tunable import _KEPT_UNTIL_EXIT


@pytest.fixture
def project(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "system.md").write_text("You help.\n", newline="")
    (prompts / "examples.md").write_text("Example A\n", newline="")
    config = tmp_path / "config"
    config.mkdir()
    (config / "agent.json").write_text(
        json.dumps({"agent": {"system_prompt": "Be brief.", "limits": {"turns": 4}, "tags": ["a", "b"]}}, indent=2)
        + "\n",
        newline="",
    )
    return tmp_path


# FileText


def test_file_text_baseline_is_the_file_content_byte_for_byte(project):
    (project / "prompts" / "crlf.md").write_bytes(b"line one\r\nline two")
    marker = FileText(project / "prompts" / "crlf.md")
    assert marker.default == "line one\r\nline two"


def test_file_text_relative_paths_resolve_from_the_working_directory(project, monkeypatch):
    monkeypatch.chdir(project)
    assert FileText("prompts/system.md").path == (project / "prompts" / "system.md").resolve()


def test_file_text_missing_file_fails_at_declaration(project):
    with pytest.raises(FileNotFoundError, match="is not a file"):
        FileText(project / "prompts" / "missing.md")


def test_file_text_directory_fails_at_declaration(project):
    with pytest.raises(FileNotFoundError):
        FileText(project / "prompts")


def test_file_text_non_utf8_fails_at_declaration(project):
    (project / "prompts" / "binary.md").write_bytes(b"\xff\xfe")
    with pytest.raises(TypeError, match="not UTF-8"):
        FileText(project / "prompts" / "binary.md")


@pytest.mark.parametrize("kwargs", [{"size_cap": 0}, {"as_path": "yes"}])
def test_file_text_bad_options_fail_at_declaration(project, kwargs):
    with pytest.raises(TypeError):
        FileText(project / "prompts" / "system.md", **kwargs)


def test_file_text_is_a_text_key_with_its_source(project):
    space = Space({"system": FileText(project / "prompts" / "system.md", as_path=True)})
    info = space.param("system")
    assert info.kind == "text"
    assert info.default == "You help.\n"
    assert info.source.path == (project / "prompts" / "system.md").resolve()
    assert info.source.pointer is None
    assert info.source.as_path is True
    assert info.size_cap == 2000


def test_size_cap_rejects_a_long_file_value_at_validation(project):
    space = Space({"system": FileText(project / "prompts" / "system.md", size_cap=20)})
    with pytest.raises(ValueError, match="over the size_cap of 20"):
        space.validate({"system": "x" * 21})


def test_a_file_over_its_size_cap_fails_at_declaration(project):
    with pytest.raises(TypeError, match="invalid default"):
        Space({"system": FileText(project / "prompts" / "system.md", size_cap=3)})


# JsonField


def test_json_field_string_baseline_is_the_string(project):
    marker = JsonField(project / "config" / "agent.json", "/agent/system_prompt")
    assert marker.default == "Be brief."


def test_json_field_non_string_baseline_is_its_json_text(project):
    marker = JsonField(project / "config" / "agent.json", "/agent/limits")
    assert json.loads(marker.default) == {"turns": 4}


def test_json_field_array_index(project):
    assert JsonField(project / "config" / "agent.json", "/agent/tags/1").default == "b"


def test_json_field_pointer_escapes(project):
    (project / "config" / "odd.json").write_text(json.dumps({"a/b": {"c~d": "x"}}), newline="")
    assert JsonField(project / "config" / "odd.json", "/a~1b/c~0d").default == "x"


@pytest.mark.parametrize("pointer", ["/agent/missing", "/agent/tags/2", "/agent/tags/01", "agent", "", 3])
def test_json_field_bad_pointer_fails_at_declaration(project, pointer):
    with pytest.raises(TypeError):
        JsonField(project / "config" / "agent.json", pointer)


def test_json_field_invalid_json_file_fails_at_declaration(project):
    (project / "config" / "broken.json").write_text("{not json", newline="")
    with pytest.raises(TypeError, match="not valid JSON"):
        JsonField(project / "config" / "broken.json", "/a")


def test_json_field_non_string_value_must_be_json(project):
    space = Space({"limits": JsonField(project / "config" / "agent.json", "/agent/limits")})
    space.validate({"limits": '{"turns": 8}'})
    with pytest.raises(ValueError, match="limits: not valid JSON"):
        space.validate({"limits": "{turns: 8}"})
    with pytest.raises(ValueError, match="not valid JSON"):
        space.validate({"limits": "NaN"})


def test_equal_json_values_give_one_candidate(project):
    space = Space({"limits": JsonField(project / "config" / "agent.json", "/agent/limits")})
    baseline = space.complete({})
    assert space.complete({"limits": '{"turns":   4}'}) == baseline
    assert space.complete({"limits": '{"turns": 4.0}'}) != baseline  # 4.0 is a float, as in Python


def test_canonical_json_keeps_key_order_and_leaves_bad_json_to_validate(project):
    space = Space({"limits": JsonField(project / "config" / "agent.json", "/agent/limits")})
    assert space.complete({"limits": '{"b": 1, "a": 2}'})["limits"] == '{\n  "b": 1,\n  "a": 2\n}'
    assert space.complete({"limits": "not json"})["limits"] == "not json"
    with pytest.raises(ValueError, match="not valid JSON"):
        space.validate(space.complete({"limits": "not json"}))


def test_a_string_json_field_is_never_rewritten(project):
    space = Space({"prompt": JsonField(project / "config" / "agent.json", "/agent/system_prompt")})
    assert space.complete({"prompt": " 5 "})["prompt"] == " 5 "


def test_json_field_string_value_takes_any_text(project):
    space = Space({"prompt": JsonField(project / "config" / "agent.json", "/agent/system_prompt")})
    space.validate({"prompt": "{not json, and that's fine"})


def test_json_field_source_names_the_pointer(project):
    space = Space({"prompt": JsonField(project / "config" / "agent.json", "/agent/system_prompt")})
    assert space.param("prompt").source.pointer == "/agent/system_prompt"
    assert space.param("prompt").source.as_path is False


# Overlapping declarations


def test_two_json_fields_on_one_file_are_allowed(project):
    path = project / "config" / "agent.json"
    Space({"prompt": JsonField(path, "/agent/system_prompt"), "limits": JsonField(path, "/agent/limits")})


@pytest.mark.parametrize("pointers", [("/agent/limits", "/agent/limits"), ("/agent", "/agent/limits")])
def test_overlapping_json_fields_fail_at_declaration(project, pointers):
    path = project / "config" / "agent.json"
    with pytest.raises(TypeError, match="one is inside the other"):
        Space({"one": JsonField(path, pointers[0]), "two": JsonField(path, pointers[1])})


def test_a_file_text_file_declared_twice_fails_at_declaration(project):
    path = project / "prompts" / "system.md"
    with pytest.raises(TypeError, match="already declared"):
        Space({"one": FileText(path), "two": FileText(path)})


def test_a_file_text_file_used_by_a_json_field_fails_at_declaration(project):
    path = project / "config" / "agent.json"
    with pytest.raises(TypeError, match="already declared"):
        Space({"whole": FileText(path), "prompt": JsonField(path, "/agent/system_prompt")})


# Resolve


def test_resolve_passes_the_text_without_as_path(project):
    space = Space({"system": FileText(project / "prompts" / "system.md")})
    assert space.resolve("system", {"system": "New."}, None) == "New."


def test_resolve_as_path_without_a_workspace_gives_the_original_path(project):
    space = Space({"system": FileText(project / "prompts" / "system.md", as_path=True)})
    assert space.resolve("system", {"system": "New."}, None) == (project / "prompts" / "system.md").resolve()


def test_as_path_resolves_to_the_copy(project, tmp_path_factory):
    space = Space({"system": FileText(project / "prompts" / "system.md", as_path=True)})
    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    values = {"system": "New prompt.\n"}
    space.create_candidate_copy(values, workspace)
    resolved = space.resolve("system", values, workspace)
    assert resolved.is_relative_to(workspace)
    assert resolved.read_text() == "New prompt.\n"
    assert resolved.with_name("examples.md").read_text() == "Example A\n"  # siblings come along


# create_candidate_copy


def make_space(project):
    return Space(
        {
            "system": FileText(project / "prompts" / "system.md"),
            "prompt": JsonField(project / "config" / "agent.json", "/agent/system_prompt"),
            "limits": JsonField(project / "config" / "agent.json", "/agent/limits"),
        }
    )


def copied(space, workspace, path):
    folders = space._copy_folders
    from strands_evals.optimization.declaration.files import _location_in_copy

    return workspace / _location_in_copy(folders, path.resolve())


def test_copy_writes_changed_values_and_never_the_originals(project, tmp_path_factory):
    space = make_space(project)
    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    originals = {p: p.read_bytes() for p in project.rglob("*") if p.is_file()}
    values = {"system": "Changed.\n", "prompt": "Be thorough.", "limits": '{"turns": 9}'}

    space.create_candidate_copy(values, workspace)

    assert copied(space, workspace, project / "prompts" / "system.md").read_text() == "Changed.\n"
    document = json.loads(copied(space, workspace, project / "config" / "agent.json").read_text())
    assert document == {"agent": {"system_prompt": "Be thorough.", "limits": {"turns": 9}, "tags": ["a", "b"]}}
    assert {p: p.read_bytes() for p in originals} == originals


def test_copy_keeps_unchanged_files_byte_for_byte(project, tmp_path_factory):
    (project / "config" / "agent.json").write_text(
        '{"agent": {"system_prompt": "Be brief.", "limits": {"turns": 4}}}', newline=""
    )
    space = make_space(project)
    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    space.create_candidate_copy(space.complete({}), workspace)
    copy = copied(space, workspace, project / "config" / "agent.json")
    assert copy.read_bytes() == (project / "config" / "agent.json").read_bytes()


def test_copy_is_built_once(project, tmp_path_factory):
    space = make_space(project)
    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    values = {**space.complete({}), "system": "First.\n"}
    assert space.create_candidate_copy(values, workspace) == workspace
    space.create_candidate_copy({**values, "system": "Second.\n"}, workspace)
    assert copied(space, workspace, project / "prompts" / "system.md").read_text() == "First.\n"
    assert [p.name for p in workspace.parent.iterdir()] == ["optimization_files"]  # no temp folder left behind


def test_copy_leaves_no_partial_directory_on_error(project, tmp_path_factory):
    space = make_space(project)
    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    with pytest.raises(ValueError):
        space.create_candidate_copy({**space.complete({}), "limits": "not json"}, workspace)
    assert not workspace.exists()
    assert list(workspace.parent.iterdir()) == []


def test_folders_with_the_same_name_do_not_collide(tmp_path):
    for side in ("a", "b"):
        (tmp_path / side / "prompts").mkdir(parents=True)
        (tmp_path / side / "prompts" / "p.md").write_text(side, newline="")
    space = Space(
        {"a": FileText(tmp_path / "a" / "prompts" / "p.md"), "b": FileText(tmp_path / "b" / "prompts" / "p.md")}
    )
    workspace = tmp_path / "run" / "optimization_files"
    space.create_candidate_copy({"a": "A2", "b": "B2"}, workspace)
    assert copied(space, workspace, tmp_path / "a" / "prompts" / "p.md").read_text() == "A2"
    assert copied(space, workspace, tmp_path / "b" / "prompts" / "p.md").read_text() == "B2"
    assert len(list(workspace.iterdir())) == 2


def test_a_folder_inside_another_is_copied_once(project, tmp_path_factory):
    (project / "prompts" / "nested").mkdir()
    (project / "prompts" / "nested" / "inner.md").write_text("inner", newline="")
    space = Space(
        {
            "system": FileText(project / "prompts" / "system.md"),
            "inner": FileText(project / "prompts" / "nested" / "inner.md"),
        }
    )
    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    space.create_candidate_copy({"system": "S", "inner": "I"}, workspace)
    (top,) = workspace.iterdir()
    assert top.name.startswith("prompts-")
    assert (top / "nested" / "inner.md").read_text() == "I"


def test_a_run_directory_inside_the_source_folder_is_not_copied(tmp_path):
    (tmp_path / "prompt.md").write_text("You help.", newline="")
    space = Space({"system": FileText(tmp_path / "prompt.md")})
    workspace = tmp_path / ".strands-evals" / "runs" / "r1" / "candidates" / "abc" / "optimization_files"
    space.create_candidate_copy({"system": "New."}, workspace)
    (top,) = workspace.iterdir()
    assert sorted(p.name for p in top.iterdir()) == ["prompt.md"]


def test_a_symlinked_file_is_replaced_not_written_through(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "target.md").write_text("original", newline="")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "system.md").write_text("You help.", newline="")
    try:
        (prompts / "link.md").symlink_to(real / "target.md")
    except OSError:  # Windows without the symlink privilege
        pytest.skip("symlinks unavailable")
    space = Space({"system": FileText(prompts / "system.md")})
    workspace = tmp_path / "run" / "optimization_files"
    space.create_candidate_copy({"system": "New."}, workspace)
    # The marker's own file is written in the copy; a symlink sibling stays a symlink.
    (top,) = workspace.iterdir()
    assert (top / "link.md").is_symlink()
    assert (real / "target.md").read_text() == "original"


def test_an_existing_empty_directory_gets_the_copy(project, tmp_path_factory):
    space = make_space(project)
    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    workspace.mkdir()
    space.create_candidate_copy({**space.complete({}), "system": "New.\n"}, workspace)
    assert copied(space, workspace, project / "prompts" / "system.md").read_text() == "New.\n"


def test_an_existing_directory_that_is_not_a_copy_fails(project, tmp_path_factory):
    space = make_space(project)
    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("mine", newline="")
    with pytest.raises(FileExistsError, match="not a candidate copy"):
        space.create_candidate_copy(space.complete({}), workspace)
    assert [p.name for p in workspace.iterdir()] == ["notes.txt"]


def test_the_copy_holds_the_values_even_after_the_source_files_changed(project, tmp_path_factory):
    space = make_space(project)
    baseline = space.complete({})
    (project / "prompts" / "system.md").write_text("Edited after declaration.\n", newline="")
    (project / "config" / "agent.json").write_text(
        json.dumps({"agent": {"system_prompt": "Edited.", "limits": {"turns": 99}}}), newline=""
    )
    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    space.create_candidate_copy(baseline, workspace)
    assert copied(space, workspace, project / "prompts" / "system.md").read_text() == "You help.\n"
    document = json.loads(copied(space, workspace, project / "config" / "agent.json").read_text())
    assert document["agent"] == {"system_prompt": "Be brief.", "limits": {"turns": 4}}


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX permission bits")
def test_a_written_file_keeps_its_permissions(tmp_path):
    script = tmp_path / "scripts" / "run.sh"
    script.parent.mkdir()
    script.write_text("echo hi\n", newline="")
    script.chmod(0o755)
    space = Space({"script": FileText(script, as_path=True)})
    workspace = tmp_path / "run" / "optimization_files"
    space.create_candidate_copy({"script": "echo bye\n"}, workspace)
    copy = copied(space, workspace, script)
    assert copy.read_text() == "echo bye\n"
    assert copy.stat().st_mode & 0o777 == 0o755


# bind


def test_bind_passes_the_candidate_file_as_a_path(project):
    @tunable(system=FileText(project / "prompts" / "system.md", as_path=True))
    def make(system):
        return system

    bound = make.bind({"system": "Bound.\n"})  # the temp copy lives as long as the bound factory
    path = bound()
    assert path.read_text() == "Bound.\n"
    assert path != (project / "prompts" / "system.md").resolve()
    assert (project / "prompts" / "system.md").read_text() == "You help.\n"


class _Agent:
    def __init__(self, path):
        self.path = path


def test_bind_outside_a_run_uses_one_temp_copy_deleted_when_nothing_uses_it(project):
    @tunable(system=FileText(project / "prompts" / "system.md", as_path=True))
    def make(system):
        return _Agent(system)

    bound = make.bind({"system": "Bound.\n"})
    first, second = bound(), bound()
    assert first.path == second.path
    root = bound._temp.root
    del bound, first
    gc.collect()
    assert root.exists()  # `second` still uses it
    del second
    gc.collect()
    assert not root.exists()


def test_an_agent_keeps_the_temp_copy_after_its_factory_is_gone(project):
    @tunable(system=FileText(project / "prompts" / "system.md", as_path=True))
    def make(system):
        return _Agent(system)

    agent = make.bind({"system": "Bound.\n"})()  # the bound factory is dropped at once
    gc.collect()
    assert agent.path.read_text() == "Bound.\n"
    root = agent.path.parents[2]
    del agent
    gc.collect()
    assert not root.exists()


def test_a_built_object_without_weak_references_keeps_the_temp_copy_until_exit(project):
    @tunable(system=FileText(project / "prompts" / "system.md", as_path=True))
    def make(system):
        return {"path": system}

    built = make.bind({"system": "Bound.\n"})()
    gc.collect()
    assert built["path"].read_text() == "Bound.\n"
    assert built["path"].is_relative_to(_KEPT_UNTIL_EXIT[-1].root)


def test_bind_with_a_workspace_writes_it_once(project, tmp_path_factory):
    @tunable(system=FileText(project / "prompts" / "system.md", as_path=True))
    def make(system):
        return system

    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    path = make.bind({"system": "Bound.\n"}, workspace=workspace)()
    assert path.is_relative_to(workspace)
    assert make.bind({"system": "Bound.\n"}, workspace=workspace)() == path


def test_bind_copies_nothing_when_no_value_is_a_path(project, tmp_path_factory):
    @tunable(system=FileText(project / "prompts" / "system.md"), tone=TextValue(default="calm"))
    def make(system, tone):
        return system

    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    assert make.bind({"system": "Text."}, workspace=workspace)() == "Text."
    assert not workspace.exists()


def test_the_plain_call_passes_the_original_path(project):
    @tunable(system=FileText(project / "prompts" / "system.md", as_path=True))
    def make(system):
        return system

    assert make() == (project / "prompts" / "system.md").resolve()
