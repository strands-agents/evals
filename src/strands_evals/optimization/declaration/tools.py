"""Tool description text as tunable values, applied to copies of the tools."""

from __future__ import annotations

import copy
import inspect
from typing import Any

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
        built = []
        for name, tool in self.tools.items():
            # A shallow copy of the tool (it may hold clients or locks), a deep copy of its spec (so no
            # nested part, e.g. `inputSchema`, is shared between builds).
            tool_copy: Any = copy.copy(tool)  # the setter exists (checked at declaration); AgentTool types it read-only
            tool_copy.tool_spec = {**copy.deepcopy(tool.tool_spec), "description": descriptions[name]}
            built.append(tool_copy)
        return built


def tool_descriptions(
    *tools: AgentTool, size_cap: int | None = None, edit_instructions: str | None = None
) -> ToolDescriptions:
    """Declare each tool's top-level description as editable text.

    The baseline is the tool's current `tool_spec["description"]`. Every build gets
    `copy.copy(tool)` with a deep copy of its spec set through the public `tool_spec` setter, so
    the user's tool and its spec are never modified. Tools without a `tool_spec` setter
    (`MCPAgentTool`, bare `AgentTool`) raise `TypeError` at declaration. Without `size_cap`, each
    tool's cap is `max(800, 3 * len(baseline))`: a lower floor than the text markers' 2000, because
    every model call carries all the descriptions, but never below a long baseline.

    Args:
        *tools: The tools whose descriptions may change.
        size_cap: Max characters per description; None means `max(800, 3 * len(baseline))`.
        edit_instructions: Shown to the proposer next to every description.

    Raises:
        TypeError: No tools, a tool has no `tool_spec` setter, no text description, or a spec its
            setter rejects; its name contains `.` or `/` (it becomes a key segment); two tools share
            a name; or `edit_instructions` isn't a str.
    """
    if not tools:
        raise TypeError("tool_descriptions() needs at least one tool")
    _check_positive_int("size_cap", size_cap)
    _check_str("edit_instructions", edit_instructions)
    by_name: dict[str, AgentTool] = {}
    for tool in tools:
        if not isinstance(tool, AgentTool):
            raise TypeError(f"tool_descriptions() takes tools, got {type(tool).__name__}")
        spec_property = inspect.getattr_static(type(tool), "tool_spec", None)
        if not isinstance(spec_property, property) or spec_property.fset is None:
            raise TypeError(
                f"tool {tool.tool_name!r} ({type(tool).__name__}) has no tool_spec setter, so its description "
                "can't be changed; use a @tool function or a PythonAgentTool"
            )
        name = tool.tool_name
        if "." in name or "/" in name:
            raise TypeError(f"tool name {name!r} contains '.' or '/'; tool names become key segments")
        if name in by_name:
            raise TypeError(f"tool_descriptions() got two tools named {name!r}")
        if not isinstance(tool.tool_spec.get("description"), str):
            raise TypeError(f"tool {name!r} has no text description in its tool_spec")
        try:
            # Probed once on a copy, so a spec the setter rejects (a name other than the tool's, no
            # inputSchema) fails here instead of on every build.
            probe: Any = copy.copy(tool)
            probe.tool_spec = copy.deepcopy(tool.tool_spec)
        except Exception as error:
            raise TypeError(f"tool {name!r}: its tool_spec setter rejects its own spec: {error}") from error
        by_name[name] = tool
    return ToolDescriptions(by_name, size_cap, edit_instructions)
