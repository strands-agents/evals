"""Tool description text as tunable values, applied to copies of the tools."""

from __future__ import annotations

from strands.types.tools import AgentTool


class ToolDescriptions:
    """One text param per tool (key `name.<tool_name>`), returned by `tool_descriptions()`."""


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
        TypeError: A tool has no `tool_spec` setter, or its name contains `.` or `/` (it becomes a
            key segment).
    """
    raise NotImplementedError
