"""Bedrock / Strands native block shapes, and the `strands_evals` typed messages.

Strands' in-memory message shape is the Bedrock Converse one: an assistant message carries
`{"toolUse": {"toolUseId", "name", "input"}}` and the following user message carries
`{"toolResult": {"toolUseId", "content"}}`. `TraceExtractor` parses those into `ToolCallContent` /
`ToolResultContent`, which serialize to a flatter `content_type`-tagged dict, so both live here.
"""

from __future__ import annotations

from typing import Any

from ._common import CallMatch, ResultMatch


def _bedrock_call(block: dict[str, Any]) -> CallMatch | None:
    """Bedrock / Strands native: `{"toolUse": {"toolUseId", "name", "input"}}`."""
    raw = block.get("toolUse")
    if isinstance(raw, dict):
        return raw.get("toolUseId"), raw.get("name"), raw.get("input")
    return None


def _bedrock_result(block: dict[str, Any]) -> ResultMatch | None:
    """Bedrock / Strands native: `{"toolResult": {"toolUseId", "content"}}`."""
    if isinstance(block.get("toolResult"), dict):
        return block["toolResult"], block["toolResult"].get("toolUseId")
    return None


def _typed_call(block: dict[str, Any]) -> CallMatch | None:
    """A strands_evals `ToolCallContent`, dumped to a dict."""
    if block.get("content_type") == "tool_use":
        return block.get("tool_call_id"), block.get("name"), block.get("arguments")
    return None


def _typed_result(block: dict[str, Any]) -> ResultMatch | None:
    """A strands_evals `ToolResultContent`, dumped to a dict."""
    if block.get("content_type") == "tool_result":
        return block, block.get("tool_call_id")
    return None
