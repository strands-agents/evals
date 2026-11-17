"""Tool description text as tunable values, applied to copies of the tools."""

from __future__ import annotations

import copy
import inspect
from typing import Any

from strands.tools.mcp import MCPAgentTool
from strands.types.tools import AgentTool

from .markers import _check_positive_int, _check_str, _Marker


class ToolDescriptions(_Marker):
    """One text param per tool (key `name.<tool_name>`), returned by `tool_descriptions()`.

    Attributes:
        tools: Tool name -> the user's tool, in declaration order; never modified.
        size_cap: Max characters per description; None means `max(800, 3 * len(baseline))` per tool.
        edit_instructions: Shown to the proposer next to every description.
    """

    def __init__(self, tools: dict[str, AgentTool], size_cap: int | None, edit_instructions: str | None):
        self.tools = tools
        self.size_cap = size_cap
        self.edit_instructions = edit_instructions

    def build(self, descriptions: dict[str, str]) -> list[AgentTool]:
        """Fresh copies of the tools with `descriptions` (tool name -> text) applied, in declaration order."""
        return [_with_description(tool, descriptions[name]) for name, tool in self.tools.items()]


def tool_descriptions(
    *tools: AgentTool, size_cap: int | None = None, edit_instructions: str | None = None
) -> ToolDescriptions:
    """Declare each tool's top-level description as editable text.

    The baseline is the tool's current `tool_spec["description"]`. Every build gets
    `copy.copy(tool)` with a deep copy of its spec set through the public `tool_spec` setter, so
    the user's tool and its spec are never modified. An `MCPAgentTool` has no setter and computes
    its spec from its `mcp_tool` on every read, so its copy gets a deep copy of `mcp_tool` with the
    new description instead (`model_copy`). Every MCP copy shares the tool's `mcp_client`, which
    must stay started while the optimizer runs, and an empty description makes an MCP tool fall
    back to the SDK's default text. Other tools without a `tool_spec` setter (a bare `AgentTool`)
    raise `TypeError` at declaration. Without `size_cap`, each tool's cap is
    `max(800, 3 * len(baseline))`: a lower floor than the text markers' 2000, because every model
    call carries all the descriptions, but never below a long baseline.

    Args:
        *tools: The tools whose descriptions may change.
        size_cap: Max characters per description; None means `max(800, 3 * len(baseline))`.
        edit_instructions: Shown to the proposer next to every description.

    Raises:
        TypeError: No tools; a tool has no `tool_spec` setter and isn't an `MCPAgentTool`, has no
            text description, or can't be copied with a new description that the copy's
            `tool_spec` shows and the tool's doesn't; its name contains `.` or `/` (it becomes a key
            segment); two tools share a name; or `edit_instructions` isn't a str.
    """
    if not tools:
        raise TypeError("tool_descriptions() needs at least one tool")
    _check_positive_int("size_cap", size_cap)
    _check_str("edit_instructions", edit_instructions)
    by_name: dict[str, AgentTool] = {}
    for tool in tools:
        if not isinstance(tool, AgentTool):
            raise TypeError(f"tool_descriptions() takes tools, got {type(tool).__name__}")
        if not _has_spec_setter(tool) and not isinstance(tool, MCPAgentTool):
            raise TypeError(
                f"tool {tool.tool_name!r} ({type(tool).__name__}) has no tool_spec setter, so its description "
                "can't be changed; use a @tool function, a PythonAgentTool or an MCPAgentTool"
            )
        name = tool.tool_name
        if "." in name or "/" in name:
            raise TypeError(f"tool name {name!r} contains '.' or '/'; tool names become key segments")
        if name in by_name:
            raise TypeError(f"tool_descriptions() got two tools named {name!r}")
        baseline = tool.tool_spec.get("description")
        if not isinstance(baseline, str):
            raise TypeError(f"tool {name!r} has no text description in its tool_spec")
        # Probed once on a copy with a different text, so a spec the setter rejects (a name other
        # than the tool's, no inputSchema) fails here instead of on every build, and so does a tool
        # whose copy doesn't show the description set on it or whose own spec changes with it.
        probe_text = f"{baseline} (probe)"
        try:
            shown = _with_description(tool, probe_text).tool_spec.get("description")
        except Exception as error:
            raise TypeError(f"tool {name!r}: a copy with a new description can't be built: {error}") from error
        if shown != probe_text:
            raise TypeError(
                f"tool {name!r} ({type(tool).__name__}): a copy's tool_spec doesn't show the description set on it"
            )
        if tool.tool_spec.get("description") != baseline:
            raise TypeError(
                f"tool {name!r} ({type(tool).__name__}): setting a description on a copy changed the tool itself"
            )
        by_name[name] = tool
    return ToolDescriptions(by_name, size_cap, edit_instructions)


def _has_spec_setter(tool: AgentTool) -> bool:
    spec_property = inspect.getattr_static(type(tool), "tool_spec", None)
    return isinstance(spec_property, property) and spec_property.fset is not None


def _with_description(tool: AgentTool, description: str) -> AgentTool:
    """A copy of `tool` whose `tool_spec` shows `description`.

    The tool is copied shallowly (it may hold clients or locks) and what its spec is made from deeply,
    so no nested part of the spec (e.g. `inputSchema`) is shared with `tool` or another build.
    """
    tool_copy: Any = copy.copy(tool)  # AgentTool types tool_spec read-only; the setter is checked at declaration
    if isinstance(tool, MCPAgentTool) and not _has_spec_setter(tool):
        # Its spec is computed from `mcp_tool` on every read.
        tool_copy.mcp_tool = tool.mcp_tool.model_copy(update={"description": description}, deep=True)
    else:
        tool_copy.tool_spec = {**copy.deepcopy(tool.tool_spec), "description": description}
    return tool_copy
