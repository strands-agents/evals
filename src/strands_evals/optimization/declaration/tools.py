"""Tool description text as tunable values, applied to copies of the tools."""

from __future__ import annotations

import copy
import inspect
from typing import Any

from strands.types.tools import AgentTool

from .markers import _check_size_cap, _Marker


class ToolDescriptions(_Marker):
    """One text param per tool (key `name.<tool_name>`), returned by `tool_descriptions()`.

    Attributes:
        tools: Tool name -> the user's tool, in declaration order; never modified.
        size_cap: Max characters per description.
        edit_instructions: Shown to the proposer next to every description.
    """

    def __init__(self, tools: dict[str, AgentTool], size_cap: int, edit_instructions: str | None):
        self.tools = tools
        self.size_cap = size_cap
        self.edit_instructions = edit_instructions

    def build(self, descriptions: dict[str, str]) -> list[AgentTool]:
        """Fresh copies of the tools with `descriptions` (tool name -> text) applied, in declaration order."""
        built = []
        for name, tool in self.tools.items():
            tool_copy: Any = copy.copy(tool)  # the setter exists (checked at declaration); AgentTool types it read-only
            tool_copy.tool_spec = {**tool.tool_spec, "description": descriptions[name]}
            built.append(tool_copy)
        return built


def tool_descriptions(*tools: AgentTool, size_cap: int = 800, edit_instructions: str | None = None) -> ToolDescriptions:
    """Declare each tool's top-level description as editable text.

    The baseline is the tool's current `tool_spec["description"]`. Every build gets
    `copy.copy(tool)` with the public `tool_spec` setter, so the user's tool is never modified.
    Tools without a `tool_spec` setter (`MCPAgentTool`, bare `AgentTool`) raise `TypeError` at
    declaration. `size_cap` defaults to 800 rather than the text markers' derived cap, because tool
    descriptions are short and every model call carries all of them.

    Args:
        *tools: The tools whose descriptions may change.
        size_cap: Max characters per description.
        edit_instructions: Shown to the proposer next to every description.

    Raises:
        TypeError: No tools, a tool has no `tool_spec` setter or no text description, its name
            contains `.` or `/` (it becomes a key segment), or two tools share a name.
    """
    if not tools:
        raise TypeError("tool_descriptions() needs at least one tool")
    _check_size_cap(size_cap)
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
        by_name[name] = tool
    return ToolDescriptions(by_name, size_cap, edit_instructions)
