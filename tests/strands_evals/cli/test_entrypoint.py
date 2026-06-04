"""Tests for ``strands_evals.cli._entrypoint``."""

from __future__ import annotations

import sys
import textwrap

import pytest

from strands_evals.cli._entrypoint import (
    EntryPointError,
    classify_agent,
    classify_task,
    import_attr,
)


def test_import_attr_resolves_module_function():
    func = import_attr("tests.strands_evals.cli.fixtures.tasks:answer")
    assert callable(func)
    assert func.__name__ == "answer"


def test_import_attr_invalid_format_no_colon():
    with pytest.raises(EntryPointError, match="exactly one ':'"):
        import_attr("not_a_valid_spec")


def test_import_attr_invalid_format_too_many_colons():
    with pytest.raises(EntryPointError, match="exactly one ':'"):
        import_attr("a:b:c")


def test_import_attr_empty_module():
    with pytest.raises(EntryPointError, match="both module and attribute"):
        import_attr(":foo")


def test_import_attr_empty_attr():
    with pytest.raises(EntryPointError, match="both module and attribute"):
        import_attr("os:")


def test_import_attr_module_not_found():
    with pytest.raises(EntryPointError, match="cannot import module"):
        import_attr("definitely_not_a_real_module_xyz:foo")


def test_import_attr_dotted_picks_up_cwd(tmp_path, monkeypatch):
    """A bare module name resolves when its file lives in cwd, no PYTHONPATH=."""
    (tmp_path / "cwd_local_agent.py").write_text(
        textwrap.dedent(
            """
            def factory():
                return "from-cwd"
            """
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "cwd_local_agent", raising=False)
    func = import_attr("cwd_local_agent:factory")
    assert callable(func)
    assert func() == "from-cwd"


def test_import_attr_dotted_does_not_leak_cwd_into_sys_path(tmp_path, monkeypatch):
    """The cwd injection used during import must not leak into sys.path."""
    (tmp_path / "leak_check_agent.py").write_text("value = 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "leak_check_agent", raising=False)
    snapshot = list(sys.path)
    import_attr("leak_check_agent:value")
    assert sys.path == snapshot


def test_import_attr_path_relative(tmp_path, monkeypatch):
    """A `./agent.py:attr` style spec loads the file directly."""
    (tmp_path / "rel_agent.py").write_text(
        textwrap.dedent(
            """
            def factory():
                return "from-relative"
            """
        )
    )
    monkeypatch.chdir(tmp_path)
    func = import_attr("./rel_agent.py:factory")
    assert func() == "from-relative"


def test_import_attr_path_relative_no_extension(tmp_path, monkeypatch):
    """A path-style spec without a `.py` suffix still resolves the file."""
    (tmp_path / "noext_agent.py").write_text("attr = 'ok'\n")
    monkeypatch.chdir(tmp_path)
    assert import_attr("./noext_agent:attr") == "ok"


def test_import_attr_path_parent_directory(tmp_path, monkeypatch):
    """A `../sibling/agent:attr` style spec resolves across directories.

    Mirrors how the example folders address each other:
    ``--agent ../01_simple_agent/agent:simple_agent``.
    """
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    (sibling / "sibling_agent.py").write_text("attr = 'sibling-ok'\n")

    invoker = tmp_path / "invoker"
    invoker.mkdir()
    monkeypatch.chdir(invoker)

    assert import_attr("../sibling/sibling_agent:attr") == "sibling-ok"


def test_import_attr_path_absolute(tmp_path):
    """An absolute path also works."""
    f = tmp_path / "abs_agent.py"
    f.write_text("attr = 42\n")
    spec = f"{f}:attr"
    assert import_attr(spec) == 42


def test_import_attr_path_file_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(EntryPointError, match="file not found"):
        import_attr("./nope.py:attr")


def test_import_attr_path_attribute_missing(tmp_path, monkeypatch):
    (tmp_path / "missing_attr_agent.py").write_text("a = 1\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(EntryPointError, match="has no attribute 'b'"):
        import_attr("./missing_attr_agent.py:b")


def test_import_attr_path_with_syntax_error(tmp_path, monkeypatch):
    (tmp_path / "broken_agent.py").write_text("def oops(:\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(EntryPointError, match="failed to import"):
        import_attr("./broken_agent.py:anything")


def test_import_attr_attribute_missing():
    with pytest.raises(EntryPointError, match="has no attribute 'nope'"):
        import_attr("tests.strands_evals.cli.fixtures.tasks:nope")


def test_classify_agent_zero_arg_callable():
    from tests.strands_evals.cli.fixtures.agents import build_agent

    resolved = classify_agent(build_agent, "fixtures.agents:build_agent")
    assert resolved.kind == "callable_no_arg"
    assert resolved.obj is build_agent


def test_classify_agent_one_arg_callable():
    from tests.strands_evals.cli.fixtures.agents import build_agent_for_case

    resolved = classify_agent(build_agent_for_case, "fixtures.agents:build_agent_for_case")
    assert resolved.kind == "callable_one_arg"


def test_classify_agent_rejects_two_arg_callable():
    from tests.strands_evals.cli.fixtures.agents import two_arg_factory

    with pytest.raises(EntryPointError, match="2 positional arguments"):
        classify_agent(two_arg_factory, "fixtures.agents:two_arg_factory")


def test_classify_agent_rejects_non_callable():
    with pytest.raises(EntryPointError, match="is neither"):
        classify_agent(42, "x:y")


def test_classify_agent_strands_instance_when_available():
    """If real strands.Agent is importable, an instance is classified as agent_instance."""
    pytest.importorskip("strands")
    from unittest.mock import MagicMock

    from strands import Agent

    fake = MagicMock(spec=Agent)
    resolved = classify_agent(fake, "x:y")
    assert resolved.kind == "agent_instance"


def test_classify_agent_strands_subclass_when_available():
    pytest.importorskip("strands")
    from strands import Agent

    class MyAgent(Agent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    resolved = classify_agent(MyAgent, "x:y")
    assert resolved.kind == "agent_class"


def test_classify_task_callable():
    def my_task(case):
        return {"output": "x"}

    resolved = classify_task(my_task, "x:y")
    assert resolved.kind == "task"


def test_classify_task_rejects_non_callable():
    with pytest.raises(EntryPointError, match="not callable"):
        classify_task(42, "x:y")
