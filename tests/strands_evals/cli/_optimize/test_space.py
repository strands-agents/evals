import json

import pytest

from strands_evals.cli._optimize.space import content_id, default_size_cap, instructions_space, load_space


def write_space(tmp_path, data):
    path = tmp_path / "space.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def root(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "instructions.md").write_text("Be helpful.\n")
    return project


def test_load_space_flattens_supported_markers_in_declaration_order(root, tmp_path):
    space = load_space(
        write_space(
            tmp_path,
            {
                "instructions": {"marker": "FileText", "path": "instructions.md", "edit_instructions": "Short."},
                "tone": {"marker": "TextValue", "default": "warm", "size_cap": 50},
                "turns": {"marker": "Int", "min_value": 1, "max_value": 9, "default": 3},
                "temperature": {"marker": "Float", "min_value": 0.1, "max_value": 1.0, "default": 0.5, "log": True},
                "model": {"marker": "OneOf", "options": ["a", "b"]},
            },
        ),
        root,
    )

    assert [p.key for p in space.params] == ["instructions", "tone", "turns", "temperature", "model"]
    assert space.text_keys == ("instructions", "tone")
    assert space.value_keys == ("turns", "temperature", "model")
    assert space.baseline() == {
        "instructions": "Be helpful.\n",
        "tone": "warm",
        "turns": 3,
        "temperature": 0.5,
        "model": "a",
    }
    instructions = space.param("instructions")
    assert instructions.size_cap == default_size_cap("Be helpful.\n") == 2000
    assert instructions.source is not None and instructions.source.path == root / "instructions.md"
    assert space.param("temperature").log is True


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"marker": "JsonField", "path": "a.json"}, "planned but not supported"),
        ({"marker": "Mystery"}, "unknown marker"),
        ({"marker": "TextValue", "default": "x", "colour": 1}, "unknown TextValue fields: colour"),
        ({"marker": "FileText", "path": "../outside.md"}, "inside the project"),
        ({"marker": "FileText", "path": "empty.md"}, "is empty"),
        ({"marker": "TextValue", "default": "  "}, "nonempty string"),
        ({"marker": "TextValue", "default": "long text", "size_cap": 3}, "size_cap"),
        ({"marker": "Int", "min_value": 1, "max_value": 5, "default": 9}, "default between"),
        ({"marker": "Int", "min_value": 1, "max_value": 5, "default": True}, "int markers need"),
        ({"marker": "Float", "min_value": 0, "max_value": 1.0, "default": 0.5}, "float markers need"),
        ({"marker": "Float", "min_value": 0.0, "max_value": 1.0, "default": 0.5, "log": True}, "min_value > 0"),
        ({"marker": "OneOf", "options": ["a"]}, "at least two"),
        ({"marker": "OneOf", "options": ["a", "b"], "default": "c"}, "not one of the options"),
        ("FileText", "must be a JSON object"),
    ],
)
def test_load_space_rejects_invalid_markers(root, tmp_path, spec, message):
    (root / "empty.md").write_text(" \n")
    (tmp_path / "outside.md").write_text("x")

    with pytest.raises(ValueError, match=message):
        load_space(write_space(tmp_path, {"key": spec}), root)


@pytest.mark.parametrize("name", ["a.b", "a/b", ""])
def test_load_space_rejects_names_with_separators(root, tmp_path, name):
    with pytest.raises(ValueError, match="names must be nonempty"):
        load_space(write_space(tmp_path, {name: {"marker": "TextValue", "default": "x"}}), root)


def test_load_space_rejects_a_file_that_is_not_a_marker_object(root, tmp_path):
    with pytest.raises(ValueError, match="nonempty JSON object"):
        load_space(write_space(tmp_path, []), root)


def test_validate_lists_every_problem_with_strict_types(root, tmp_path):
    space = load_space(
        write_space(
            tmp_path,
            {
                "tone": {"marker": "TextValue", "default": "warm", "size_cap": 10},
                "turns": {"marker": "Int", "min_value": 1, "max_value": 9, "default": 3},
                "model": {"marker": "OneOf", "options": [1, 2]},
            },
        ),
        root,
    )

    with pytest.raises(ValueError) as error:
        space.validate({"tone": "x" * 11, "turns": True, "model": 1.0, "extra": 1})

    message = str(error.value)
    assert "extra: unknown key" in message
    assert "tone: text over its size cap (11 > 10)" in message
    assert "turns: must be a int" in message
    assert "model: not one of the options" in message
    space.validate({"tone": "cool", "turns": 9, "model": 2})


def test_diff_puts_file_texts_on_their_paths_and_other_texts_by_key(root, tmp_path):
    space = load_space(
        write_space(
            tmp_path,
            {
                "tone": {"marker": "TextValue", "default": "warm\n"},
                "instructions": {"marker": "FileText", "path": "instructions.md"},
                "turns": {"marker": "Int", "min_value": 1, "max_value": 9, "default": 3},
            },
        ),
        root,
    )
    base = space.baseline()
    other = {**base, "tone": "cool\n", "instructions": "Be brief.\n", "turns": 5}

    diff = space.diff(base, other)

    assert diff.index("--- a/instructions.md") < diff.index("--- tone")
    assert "+Be brief.\n" in diff and "+cool\n" in diff
    assert space.changed_values(base, other) == {"turns": {"old": 3, "new": 5}}
    assert space.diff(base, base) == ""


def test_instructions_space_declares_one_file_text_key(root):
    space = instructions_space(root / "instructions.md", root)

    assert space.baseline() == {"instructions": "Be helpful.\n"}
    assert space.config()[0]["source"] == "instructions.md"


def test_content_id_ignores_key_order_and_tracks_values():
    assert content_id({"a": 1, "b": "x"}) == content_id({"b": "x", "a": 1})
    assert content_id({"a": 1}) != content_id({"a": 1.0})
    assert len(content_id({"a": 1})) == 12
