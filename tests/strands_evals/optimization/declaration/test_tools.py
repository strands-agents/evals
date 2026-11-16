import pytest
from strands import tool
from strands.tools.tools import PythonAgentTool
from strands.types.tools import AgentTool

from strands_evals.optimization import tool_descriptions


@tool
def lookup(query: str) -> str:
    """Look up a record by query."""
    return query


@tool
def search(query: str) -> str:
    """Search the index."""
    return query


class SpecOnlyTool(AgentTool):
    """Like MCPAgentTool: a read-only tool_spec."""

    @property
    def tool_name(self) -> str:
        return "spec_only"

    @property
    def tool_spec(self):
        return {"name": "spec_only", "description": "d", "inputSchema": {"json": {}}}

    @property
    def tool_type(self) -> str:
        return "python"

    async def stream(self, tool_use, invocation_state, **kwargs):
        yield {}


def test_tools_are_keyed_by_name_with_their_description_as_baseline():
    declared = tool_descriptions(lookup, search, size_cap=100, edit_instructions="short")
    assert list(declared.tools) == ["lookup", "search"]
    assert (declared.size_cap, declared.edit_instructions) == (100, "short")


def test_build_returns_copies_and_never_modifies_the_users_tool():
    declared = tool_descriptions(lookup)
    first = declared.build({"lookup": "Find one record."})[0]
    second = declared.build({"lookup": "Find one record."})[0]
    assert first.tool_spec["description"] == "Find one record."
    assert first is not lookup and first is not second
    assert first.tool_name == "lookup"
    assert first.tool_spec["inputSchema"] == lookup.tool_spec["inputSchema"]
    assert lookup.tool_spec["description"] == "Look up a record by query."


def test_build_shares_no_part_of_the_spec_between_builds():
    declared = tool_descriptions(lookup)
    first, second = (declared.build({"lookup": "Find."})[0] for _ in range(2))
    assert first.tool_spec["inputSchema"] is not second.tool_spec["inputSchema"]
    assert first.tool_spec["inputSchema"] is not lookup.tool_spec["inputSchema"]


def test_python_agent_tools_are_supported():
    spec = {"name": "add", "description": "Add two numbers.", "inputSchema": {"json": {}}}
    adder = PythonAgentTool("add", spec, lambda tool_use, **kwargs: None)
    built = tool_descriptions(adder).build({"add": "Sum."})[0]
    assert built.tool_spec["description"] == "Sum."
    assert adder.tool_spec["description"] == "Add two numbers."


def test_a_tool_without_a_tool_spec_setter_fails_at_declaration():
    with pytest.raises(TypeError, match="no tool_spec setter"):
        tool_descriptions(SpecOnlyTool())


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: tool_descriptions(), "at least one tool"),
        (lambda: tool_descriptions(lookup, lookup), "two tools named"),
        (lambda: tool_descriptions("lookup"), "takes tools"),  # type: ignore[arg-type]
        (lambda: tool_descriptions(lookup, size_cap=0), "size_cap must be a positive integer"),
        (lambda: tool_descriptions(lookup, edit_instructions=1), "edit_instructions must be a str"),  # type: ignore[arg-type]
    ],
)
def test_bad_declarations_fail(make, match):
    with pytest.raises(TypeError, match=match):
        make()


def test_a_tool_without_a_text_description_fails_at_declaration():
    spec = {"name": "lookup", "description": None, "inputSchema": {"json": {}}}
    with pytest.raises(TypeError, match="no text description"):
        tool_descriptions(PythonAgentTool("lookup", spec, lambda tool_use, **kwargs: None))  # type: ignore[arg-type]


def test_a_spec_the_setter_rejects_fails_at_declaration_not_on_every_build():
    spec = {"name": "other", "description": "Look up.", "inputSchema": {"json": {}}}
    with pytest.raises(TypeError, match="setter rejects its own spec"):
        tool_descriptions(PythonAgentTool("lookup", spec, lambda tool_use, **kwargs: None))  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["db.lookup", "db/lookup"])
def test_a_tool_name_that_cant_be_a_key_segment_fails(name):
    spec = {"name": name, "description": "Look up.", "inputSchema": {"json": {}}}
    with pytest.raises(TypeError, match="key segments"):
        tool_descriptions(PythonAgentTool(name, spec, lambda tool_use, **kwargs: None))
