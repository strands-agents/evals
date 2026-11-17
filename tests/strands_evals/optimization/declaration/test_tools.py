import pytest
from mcp.types import Tool as MCPTool
from strands import tool
from strands.tools.mcp import MCPAgentTool
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
    """A tool with a read-only tool_spec."""

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


class SpecIgnoringTool(SpecOnlyTool):
    """A tool_spec setter that keeps the spec it has."""

    @property
    def tool_spec(self):
        return {"name": "spec_only", "description": "d", "inputSchema": {"json": {}}}

    @tool_spec.setter
    def tool_spec(self, value):
        pass


class WriteThroughTool(SpecOnlyTool):
    """A tool_spec setter that writes into the dict a shallow copy shares."""

    def __init__(self):
        super().__init__()
        self._spec = {"name": "spec_only", "description": "d", "inputSchema": {"json": {}}}

    @property
    def tool_spec(self):
        return self._spec

    @tool_spec.setter
    def tool_spec(self, value):
        self._spec.update(value)


class FixedSpecMCPTool(MCPAgentTool):
    """An MCP tool whose spec doesn't come from its `mcp_tool`."""

    @property
    def tool_spec(self):
        return {"name": "search_docs", "description": "Search the docs.", "inputSchema": {"json": {}}}


def mcp_agent_tool(
    description: str | None = "Search the docs.", mcp_client: object | None = None, name_override: str | None = None
) -> MCPAgentTool:
    """An MCP tool as the SDK's MCP client lists it, with a stand-in client (no server is called)."""
    definition = MCPTool(
        name="search_docs",
        description=description,
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
        outputSchema={"type": "object", "properties": {"pages": {"type": "array"}}},
        annotations={"readOnlyHint": True},
    )
    return MCPAgentTool(definition, mcp_client or object(), name_override=name_override)  # type: ignore[arg-type]


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


def test_mcp_agent_tools_are_supported():
    client = object()
    docs = mcp_agent_tool(mcp_client=client, name_override="docs_search")
    original, fields = docs.mcp_tool, docs.mcp_tool.model_dump()
    declared = tool_descriptions(lookup, docs)
    assert list(declared.tools) == ["lookup", "docs_search"]
    built_lookup, built = declared.build({"lookup": "Find one record.", "docs_search": "Find pages."})
    assert built_lookup.tool_spec["description"] == "Find one record."
    assert isinstance(built, MCPAgentTool) and built is not docs
    assert built.tool_spec == {**docs.tool_spec, "description": "Find pages."}
    assert (built.tool_name, built.mcp_tool.name) == ("docs_search", "search_docs")
    assert built.mcp_client is client
    assert docs.mcp_tool is original and docs.mcp_tool.model_dump() == fields
    assert docs.tool_spec["description"] == "Search the docs."


def test_mcp_builds_share_no_part_of_the_spec():
    docs = mcp_agent_tool()
    declared = tool_descriptions(docs)
    first, second = (declared.build({"search_docs": "Find."})[0] for _ in range(2))
    first.tool_spec["inputSchema"]["json"]["properties"]["query"]["type"] = "integer"
    first.tool_spec["outputSchema"]["json"]["properties"]["pages"]["type"] = "object"
    for other in (second, docs):
        assert other.tool_spec["inputSchema"]["json"]["properties"]["query"]["type"] == "string"
        assert other.tool_spec["outputSchema"]["json"]["properties"]["pages"]["type"] == "array"


def test_an_mcp_tool_without_a_description_builds_its_baseline_unchanged():
    docs = mcp_agent_tool(description=None)
    baseline = docs.tool_spec["description"]  # the SDK's text for a tool without one
    assert tool_descriptions(docs).build({"search_docs": baseline})[0].tool_spec == docs.tool_spec


@pytest.mark.parametrize(
    "make",
    [SpecIgnoringTool, lambda: FixedSpecMCPTool(mcp_agent_tool().mcp_tool, object())],  # type: ignore[arg-type]
    ids=["setter", "mcp"],
)
def test_a_tool_whose_copy_ignores_a_new_description_fails_at_declaration(make):
    with pytest.raises(TypeError, match="doesn't show the description set on it"):
        tool_descriptions(make())


def test_a_tool_whose_copy_changes_the_tool_itself_fails_at_declaration():
    with pytest.raises(TypeError, match="changed the tool itself"):
        tool_descriptions(WriteThroughTool())


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
    with pytest.raises(TypeError, match="a copy with a new description can't be built"):
        tool_descriptions(PythonAgentTool("lookup", spec, lambda tool_use, **kwargs: None))  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["db.lookup", "db/lookup"])
def test_a_tool_name_that_cant_be_a_key_segment_fails(name):
    spec = {"name": name, "description": "Look up.", "inputSchema": {"json": {}}}
    with pytest.raises(TypeError, match="key segments"):
        tool_descriptions(PythonAgentTool(name, spec, lambda tool_use, **kwargs: None))
