import asyncio
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from strands_evals.optimization import FileText, JsonField, SkillLibrary, Space, TextValue, tunable
from strands_evals.optimization.declaration.files import _location_in_copy
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


def test_a_missing_relative_file_names_where_it_looked(project, monkeypatch):
    monkeypatch.chdir(project)
    with pytest.raises(FileNotFoundError, match="looked for .*nope.md, from the working directory"):
        FileText("prompts/nope.md")


def test_a_lone_surrogate_fails_validation(project):
    space = Space({"system": FileText(project / "prompts" / "system.md")})
    with pytest.raises(ValueError, match="lone surrogate"):
        space.validate({"system": "bad \ud800 text"})


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


@pytest.mark.parametrize("value", ['"7"', "[7]", '{"turns": "7"}', "true", "null"])
def test_a_non_string_json_field_takes_only_its_baselines_kind(project, value):
    space = Space({"turns": JsonField(project / "config" / "agent.json", "/agent/limits/turns")})
    with pytest.raises(ValueError, match="expected a JSON number like the file's 4"):
        space.validate({"turns": value})


def test_a_numeric_json_field_takes_an_int_or_a_float(project):
    space = Space({"turns": JsonField(project / "config" / "agent.json", "/agent/limits/turns")})
    space.validate({"turns": "4.5"})
    space.validate({"turns": "5"})


def test_a_null_json_field_takes_any_kind(tmp_path):
    (tmp_path / "agent.json").write_text('{"stop": null}', newline="")
    space = Space({"stop": JsonField(tmp_path / "agent.json", "/stop")})
    for value in ('["END"]', '"END"', "3", "null"):
        space.validate({"stop": value})


def test_an_out_of_range_json_number_fails_validation(project):
    space = Space({"turns": JsonField(project / "config" / "agent.json", "/agent/limits/turns")})
    with pytest.raises(ValueError, match="turns: not valid JSON"):
        space.validate({"turns": "1e400"})


def test_size_cap_is_checked_on_the_json_text_as_stored(project, tmp_path):
    (tmp_path / "agent.json").write_text('{"tags": [1, 2]}\n', newline="")
    space = Space({"tags": JsonField(tmp_path / "agent.json", "/tags", size_cap=12)})
    # 9 characters as proposed, 22 once written with the 2-space indent
    with pytest.raises(ValueError, match="22 characters, over the size_cap of 12"):
        space.validate({"tags": "[1,2,3,4]"})


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


@pytest.mark.parametrize("edited", [{"agent": "flat"}, {"other": 1}])
def test_writing_a_json_field_the_file_lost_since_declaration_names_it(project, edited):
    path = project / "config" / "agent.json"
    field = JsonField(path, "/agent/limits/turns")
    path.write_text(json.dumps(edited), newline="")
    with pytest.raises(TypeError, match="has no field at '/agent/limits/turns'"):
        field.write("5", path)


def test_a_json_file_deleted_or_broken_since_declaration_is_named_in_the_copy_error(project, tmp_path_factory):
    path = project / "config" / "agent.json"
    space = Space({"turns": JsonField(path, "/agent/limits/turns")})
    path.write_text("{oops", newline="")
    with pytest.raises(TypeError, match="agent.json is no longer valid JSON"):
        space.create_candidate_copy({"turns": "5"}, tmp_path_factory.mktemp("run") / "broken")
    path.unlink()
    with pytest.raises(TypeError, match="agent.json no longer exists"):
        space.create_candidate_copy({"turns": "5"}, tmp_path_factory.mktemp("run") / "deleted")


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


def folders(workspace):
    """The copied folders, without the marker file a finished copy holds."""
    return [p for p in workspace.iterdir() if p.is_dir()]


def copied(space, workspace, path):
    return workspace / _location_in_copy(space._copy_folders, path.resolve())


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
    target = copied(space, workspace, project / "prompts" / "system.md")
    target.write_text("touched\n", newline="")  # a rebuild would overwrite this
    space.create_candidate_copy(values, workspace)
    assert target.read_text() == "touched\n"
    assert [p.name for p in workspace.parent.iterdir()] == ["optimization_files"]  # no temp folder left behind


def test_a_workspace_holding_another_candidates_copy_is_refused(project, tmp_path_factory):
    space = make_space(project)
    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    values = {**space.complete({}), "system": "First.\n"}
    space.create_candidate_copy(values, workspace)
    with pytest.raises(FileExistsError, match="another candidate's copy"):
        space.create_candidate_copy({**values, "system": "Second.\n"}, workspace)
    assert copied(space, workspace, project / "prompts" / "system.md").read_text() == "First.\n"


def test_concurrent_copies_into_one_empty_folder_all_succeed(project, tmp_path_factory):
    space = make_space(project)
    values = {**space.complete({}), "system": "Same.\n"}
    for trial in range(10):
        workspace = tmp_path_factory.mktemp(f"run{trial}") / "optimization_files"
        workspace.mkdir()
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(space.create_candidate_copy, [values] * 8, [workspace] * 8))
        assert results == [workspace] * 8
        assert copied(space, workspace, project / "prompts" / "system.md").read_text() == "Same.\n"
        assert [p.name for p in workspace.parent.iterdir()] == ["optimization_files"]


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX permission bits")
def test_a_read_only_source_folder_can_still_be_copied_and_written(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "system.md").write_text("You help.\n", newline="")
    space = Space({"system": FileText(prompts / "system.md", as_path=True)})
    prompts.chmod(0o555)
    try:
        workspace = tmp_path / "run" / "optimization_files"
        space.create_candidate_copy({"system": "New.\n"}, workspace)
        assert copied(space, workspace, prompts / "system.md").read_text() == "New.\n"
    finally:
        prompts.chmod(0o755)


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
    assert len(folders(workspace)) == 2


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
    (top,) = folders(workspace)
    assert [p.name for p in top.iterdir()] == ["prompts"]  # the copy keeps the folder's name
    assert (top / "prompts" / "nested" / "inner.md").read_text() == "I"


def test_a_run_directory_inside_the_source_folder_is_not_copied(tmp_path):
    (tmp_path / "prompt.md").write_text("You help.", newline="")
    space = Space({"system": FileText(tmp_path / "prompt.md")})
    workspace = tmp_path / ".strands-evals" / "runs" / "r1" / "candidates" / "abc" / "optimization_files"
    space.create_candidate_copy({"system": "New."}, workspace)
    folder = copied(space, workspace, tmp_path / "prompt.md").parent
    assert sorted(p.name for p in folder.iterdir()) == ["prompt.md"]


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
    assert (workspace / _location_in_copy(space._copy_folders, prompts.resolve() / "link.md")).is_symlink()
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
    root = agent.path.parents[3]  # <temp>/optimization_files/<hash>/prompts/system.md
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


def test_an_async_factorys_agent_keeps_the_temp_copy(project):
    class Agent:
        def __init__(self, path):
            self.path = path

    @tunable(system=FileText(project / "prompts" / "system.md", as_path=True))
    async def make(system=None):
        return Agent(system)

    async def build():
        return await make.bind({"system": "New.\n"})()

    agent = asyncio.run(build())
    gc.collect()
    assert agent.path.read_text() == "New.\n"


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


# SkillLibrary


def skill_text(name, description, body="Do the thing.\n"):
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"


@pytest.fixture
def library(tmp_path):
    skills = tmp_path / "skills"
    (skills / "triage" / "references").mkdir(parents=True)
    (skills / "triage" / "SKILL.md").write_text(skill_text("triage", "Sort incoming tickets."), newline="")
    (skills / "triage" / "references" / "levels.md").write_text("P0 to P3\n", newline="")
    (skills / "billing").mkdir()
    (skills / "billing" / "skill.md").write_text(skill_text("billing", "Answer billing questions."), newline="")
    (skills / "notes").mkdir()  # no SKILL.md: not a skill
    (skills / "README.md").write_text("Our skills.\n", newline="")
    return skills


def test_skill_library_members_are_each_skills_skill_md(library):
    space = Space({"skills": SkillLibrary(library)})
    infos = space.params()
    assert [info.key for info in infos] == ["skills/billing/skill.md", "skills/triage/SKILL.md"]
    triage = space.param("skills/triage/SKILL.md")
    assert triage.kind == "text"
    assert triage.name == "skills"
    assert triage.default == skill_text("triage", "Sort incoming tickets.")
    assert triage.source.path == library.resolve() / "triage" / "SKILL.md"
    assert triage.source.as_path is True
    assert triage.size_cap == 2000


def test_a_library_that_is_itself_a_skill_has_one_member(library):
    space = Space({"triage": SkillLibrary(library / "triage")})
    assert [info.key for info in space.params()] == ["triage/SKILL.md"]


def test_only_tunes_the_named_skills_with_the_same_keys(library):
    space = Space({"skills": SkillLibrary(library, only=["triage"])})
    assert [info.key for info in space.params()] == ["skills/triage/SKILL.md"]


def test_only_copies_every_skill_and_a_strict_runtime_loads_them(library):
    skills_module = pytest.importorskip("strands.vended_plugins.skills")

    @tunable(skills=SkillLibrary(library, only=("triage",)))
    def make(skills):
        return _Agent(skills)

    originals = {p: p.read_bytes() for p in library.rglob("*") if p.is_file()}
    edited = skill_text("triage", "Sort tickets by customer impact.", "Ask for the account id first.\n")
    agent = make.bind({"skills/triage/SKILL.md": edited})()
    assert agent.path.name == "skills"
    loaded = {skill.name: skill for skill in skills_module.Skill.from_directory(agent.path, strict=True)}
    assert set(loaded) == {"billing", "triage"}
    assert loaded["triage"].instructions == "Ask for the account id first."
    assert (agent.path / "billing" / "skill.md").read_bytes() == originals[library / "billing" / "skill.md"]
    assert {p: p.read_bytes() for p in originals} == originals
    assert make().path == library.resolve()  # the plain call: the original library


def test_only_never_reads_the_skills_it_does_not_tune(library):
    (library / "billing" / "skill.md").write_text("No frontmatter.\n", newline="")
    space = Space({"skills": SkillLibrary(library, only=("triage",))})
    assert [info.key for info in space.params()] == ["skills/triage/SKILL.md"]


@pytest.mark.parametrize(
    ("only", "problem"),
    [
        ("triage", "must be a list or tuple of skill names, got 'triage'"),
        ([1], "must be a list or tuple of skill names, got \\[1\\]"),
        ({"triage"}, "must be a list or tuple of skill names"),
        ((name for name in ["triage"]), "must be a list or tuple of skill names"),
        ((), "names no skill"),
        (("triage", "faq"), "no skill named 'faq' in .* \\(it has billing, triage\\)"),
        (("notes",), "no skill named 'notes'"),
    ],
)
def test_only_must_name_skills_of_the_library(library, only, problem):
    with pytest.raises(TypeError, match=problem):
        SkillLibrary(library, only=only)


def test_only_on_a_single_skill_fails(library):
    with pytest.raises(TypeError, match="is a single skill"):
        SkillLibrary(library / "triage", only=("triage",))


def test_a_single_skill_is_copied_alone_with_its_directory_name(library):
    @tunable(triage=SkillLibrary(library / "triage"))
    def make(triage):
        return _Agent(triage)

    edited = skill_text("triage", "Sort incoming tickets.", "Check the level first.\n")
    agent = make.bind({"triage/SKILL.md": edited})()  # the temp copy lives as long as the agent
    copy = agent.path
    assert copy.name == "triage"  # the runtime checks the directory name against the skill's name
    assert [p.name for p in copy.parent.iterdir()] == ["triage"]  # nothing around it is copied
    assert (copy / "SKILL.md").read_text() == edited
    assert (copy / "references" / "levels.md").read_text() == "P0 to P3\n"
    assert make().path == library.resolve() / "triage"


def test_a_run_directory_inside_a_single_skill_is_left_out_of_its_copy(library):
    skill = library / "triage"
    space = Space({"triage": SkillLibrary(skill)})
    workspace = skill / ".strands-evals" / "runs" / "r1" / "candidates" / "abc" / "optimization_files"
    values = {"triage/SKILL.md": skill_text("triage", "Sort.", "New.\n")}
    space.create_candidate_copy(values, workspace)
    copy = space.resolve("triage", values, workspace)
    assert copy.name == "triage"
    assert sorted(p.name for p in copy.iterdir()) == ["SKILL.md", "references"]
    assert (copy / "SKILL.md").read_text() == values["triage/SKILL.md"]


def test_skill_library_size_cap_and_instructions_apply_to_every_member(library):
    space = Space({"skills": SkillLibrary(library, size_cap=500, edit_instructions="Keep it short.")})
    assert {(info.size_cap, info.edit_instructions) for info in space.params()} == {(500, "Keep it short.")}


@pytest.mark.parametrize("where", ["missing", "README.md"])
def test_skill_library_path_must_be_a_directory(library, where):
    with pytest.raises(FileNotFoundError, match="is not a directory"):
        SkillLibrary(library / where)


def test_a_library_with_no_skill_fails_at_declaration(library):
    with pytest.raises(TypeError, match="holds no skill"):
        SkillLibrary(library / "notes")


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_a_symlinked_skill_directory_fails_at_declaration(library, tmp_path):
    outside = tmp_path / "shared"
    outside.mkdir()
    (outside / "SKILL.md").write_text(skill_text("shared", "Shared."), newline="")
    (library / "shared").symlink_to(outside)
    with pytest.raises(TypeError, match="is a symlink"):
        SkillLibrary(library)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("Just instructions.\n", "must start with --- frontmatter"),
        ("---\nname: triage\ndescription: x\n", "missing closing --- delimiter"),
        ("---\nname: triage\n---\nBody\n", "must have a 'description' field"),
        ("---\nname: ''\ndescription: x\n---\nBody\n", "must have a 'name' field"),
        ("---\nname: [triage\ndescription: x\n---\nBody\n", "malformed YAML frontmatter"),
        ("---\n- triage\n---\nBody\n", "must have a 'name' field"),
    ],
)
def test_a_skill_md_without_well_formed_frontmatter_fails_at_declaration(library, text, problem):
    (library / "triage" / "SKILL.md").write_text(text, newline="")
    with pytest.raises(TypeError, match=rf"triage[/\\]SKILL.md: .*{problem}"):
        SkillLibrary(library)


def test_a_skill_md_with_an_unquoted_colon_loads_as_the_runtime_loads_it(library):
    text = skill_text("triage", "Use this skill when: a ticket arrives")
    (library / "triage" / "SKILL.md").write_text(text, newline="")
    space = Space({"skills": SkillLibrary(library)})
    space.validate(space.complete({"skills/triage/SKILL.md": skill_text("triage", "Use when: tickets", "New.\n")}))


def test_a_skill_md_that_is_not_utf8_fails_at_declaration(library):
    (library / "triage" / "SKILL.md").write_bytes(b"\xff\xfe")
    with pytest.raises(TypeError, match="not UTF-8"):
        SkillLibrary(library)


def test_a_proposed_skill_md_may_change_anything_but_its_name(library):
    space = Space({"skills": SkillLibrary(library)})
    values = space.complete({"skills/triage/SKILL.md": skill_text("triage", "Sort and route tickets.", "New steps.\n")})
    space.validate(values)
    with pytest.raises(ValueError, match="frontmatter name must stay 'triage', got 'router'"):
        space.validate({**values, "skills/triage/SKILL.md": skill_text("router", "Route tickets.")})
    with pytest.raises(ValueError, match="skills/triage/SKILL.md: SKILL.md must start with --- frontmatter"):
        space.validate({**values, "skills/triage/SKILL.md": "No frontmatter.\n"})


def test_the_baseline_skill_md_is_parsed_once(library, monkeypatch):
    from strands_evals.optimization.declaration import files

    parsed = []
    parse = files._skill_name
    monkeypatch.setattr(files, "_skill_name", lambda text: parsed.append(text) or parse(text))
    space = Space({"skills": SkillLibrary(library, only=("triage",))})
    space.validate(space.complete({}))
    assert len(parsed) == 1


def test_a_skill_md_over_its_size_cap_fails_validation(library):
    space = Space({"skills": SkillLibrary(library, size_cap=200)})
    long = skill_text("triage", "Sort.", "x" * 200)
    with pytest.raises(ValueError, match="over the size_cap of 200"):
        space.validate(space.complete({"skills/triage/SKILL.md": long}))


def test_a_skill_library_resolves_to_the_original_directory_without_a_workspace(library):
    space = Space({"skills": SkillLibrary(library)})
    assert space.resolve("skills", space.complete({}), None) == library.resolve()


def test_bind_passes_the_copy_of_the_library_with_the_candidates_text(library):
    @tunable(skills=SkillLibrary(library))
    def make(skills):
        return _Agent(skills)

    originals = {p: p.read_bytes() for p in library.rglob("*") if p.is_file()}
    edited = skill_text("triage", "Sort incoming tickets.", "Check the level first.\n")
    bound = make.bind({"skills/triage/SKILL.md": edited})
    copy = bound().path
    assert copy != library.resolve()
    assert (copy / "triage" / "SKILL.md").read_text() == edited
    assert (copy / "billing" / "skill.md").read_bytes() == originals[library / "billing" / "skill.md"]
    assert (copy / "triage" / "references" / "levels.md").read_text() == "P0 to P3\n"
    assert (copy / "README.md").read_text() == "Our skills.\n"
    assert {p: p.read_bytes() for p in originals} == originals


def test_the_plain_call_passes_the_original_library(library):
    @tunable(skills=SkillLibrary(library))
    def make(skills):
        return skills

    assert make() == library.resolve()


def test_a_skills_runtime_sees_the_candidates_edits(library):
    skills_module = pytest.importorskip("strands.vended_plugins.skills")

    @tunable(skills=SkillLibrary(library))
    def make(skills):
        return _Agent(skills)

    edited = skill_text("triage", "Sort tickets by customer impact.", "Ask for the account id first.\n")
    agent = make.bind({"skills/triage/SKILL.md": edited})()
    loaded = {skill.name: skill for skill in skills_module.Skill.from_directory(agent.path)}
    assert set(loaded) == {"billing", "triage"}
    assert loaded["triage"].description == "Sort tickets by customer impact."
    assert loaded["triage"].instructions == "Ask for the account id first."
    assert loaded["billing"].description == "Answer billing questions."


def test_a_file_text_on_a_reference_inside_the_library_shares_its_copy(library, tmp_path_factory):
    space = Space(
        {
            "skills": SkillLibrary(library),
            "levels": FileText(library / "triage" / "references" / "levels.md", as_path=True),
        }
    )
    assert space._copy_folders == (library.resolve(),)
    workspace = tmp_path_factory.mktemp("run") / "optimization_files"
    values = space.complete({"levels": "P0 to P4\n"})
    space.create_candidate_copy(values, workspace)
    skills_copy = space.resolve("skills", values, workspace)
    assert space.resolve("levels", values, workspace) == skills_copy / "triage" / "references" / "levels.md"
    assert (skills_copy / "triage" / "references" / "levels.md").read_text() == "P0 to P4\n"


def test_a_skill_md_declared_again_fails_at_declaration(library):
    with pytest.raises(TypeError, match="already declared"):
        Space({"skills": SkillLibrary(library), "triage": FileText(library / "triage" / "SKILL.md")})
    with pytest.raises(TypeError, match="already declared"):
        Space({"skills": SkillLibrary(library), "triage": SkillLibrary(library / "triage")})


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
@pytest.mark.parametrize("linked_to", ["another skill's", "an undeclared file"])
def test_a_skill_md_that_is_a_symlink_fails_at_declaration(library, tmp_path, linked_to):
    if linked_to == "an undeclared file":
        target = tmp_path / "shared.md"
        target.write_text(skill_text("router", "Route tickets."), newline="")
    else:
        target = library / "triage" / "SKILL.md"
    (library / "router").mkdir()
    (library / "router" / "SKILL.md").symlink_to(target)
    with pytest.raises(TypeError, match=r"router[/\\]SKILL.md is a symlink"):
        SkillLibrary(library)
    with pytest.raises(TypeError, match="is a symlink"):
        SkillLibrary(library / "router")
    assert [info.key for info in Space({"s": SkillLibrary(library, only=("triage",))}).params()] == [
        "s/triage/SKILL.md"
    ]  # an untuned skill is never read


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions, not as root")
def test_an_unreadable_folder_is_skipped_as_the_runtime_skips_it(library):
    locked = library / "locked"
    locked.mkdir()
    (locked / "SKILL.md").write_text(skill_text("locked", "Locked."), newline="")
    locked.chmod(0o000)
    try:
        space = Space({"skills": SkillLibrary(library)})
    finally:
        locked.chmod(0o755)
    assert [info.key for info in space.params()] == ["skills/billing/skill.md", "skills/triage/SKILL.md"]


def test_a_skill_edit_diffs_on_its_real_file(library):
    from strands_evals.optimization.candidate import Candidate, diff

    space = Space({"skills": SkillLibrary(library)})
    base = Candidate.baseline(space)
    edited = skill_text("triage", "Sort incoming tickets.", "Check the level first.\n")
    other = base.child(space, {"skills/triage/SKILL.md": edited}, origin="text", rationale="")
    text = diff(space, base, other, root=library.parent)
    assert text.startswith("--- a/skills/triage/SKILL.md\n+++ b/skills/triage/SKILL.md\n")
    assert "-Do the thing.\n+Check the level first.\n" in text
