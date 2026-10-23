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
    declaration.
    """
    raise NotImplementedError
