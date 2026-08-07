"""Anthropic / Claude Code block shapes.

The Anthropic content-block shape: `{"type": "tool_use", "id", "name", "input"}` and
`{"type": "tool_result", "tool_use_id", "content"}`. Claude Code streams these inside its own
event envelope, which `_common._iter_indexed_blocks` unwraps before they reach here.

Claude Code delivers a skill body as a separate injected user message rather than in the tool
result, so its calls arrive with no body attached. Recovering it is the extractor's job
(`_claude_body_after`), because it means matching a later message to this call rather than reading
one block.
"""

from __future__ import annotations

from typing import Any

from ._common import CallMatch, ResultMatch


def _anthropic_call(block: dict[str, Any]) -> CallMatch | None:
    """Anthropic-style content block: `{"type": "tool_use", "name", "input"}`."""
    if block.get("type") == "tool_use":
        return block.get("id") or block.get("tool_use_id"), block.get("name"), block.get("input")
    return None


def _anthropic_result(block: dict[str, Any]) -> ResultMatch | None:
    """Anthropic-style content block: `{"type": "tool_result", "tool_use_id"}`."""
    if block.get("type") == "tool_result":
        return block, block.get("tool_use_id") or block.get("id")
    return None
