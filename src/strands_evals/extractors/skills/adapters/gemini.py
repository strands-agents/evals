"""Gemini CLI and Google ADK block shapes.

Both are built on the Google GenAI content parts, `{"functionCall": {"id", "name", "args"}}` and
`{"functionResponse": {"id", "response"}}`, so one pair of recognizers serves them. Gemini CLI's
stream events use a flatter `{"tool_name", "parameters"}` instead, and ADK event items a bare
`{"name", "args"}`; both are here rather than in a separate module because they are the same two
products emitting the same call in a different envelope.

The `{"name", "args"}` recognizer is the loosest in the registry and is tried last for that reason:
any block with a string `name` and a dict `args` matches it.
"""

from __future__ import annotations

from typing import Any

from ._common import CallMatch, ResultMatch


def _gemini_call(block: dict[str, Any]) -> CallMatch | None:
    """Gemini CLI / Google ADK content parts: `{"functionCall": {"id", "name", "args"}}`."""
    raw = block.get("functionCall") or block.get("function_call")
    if isinstance(raw, dict):
        return raw.get("id"), raw.get("name"), raw.get("args")
    return None


def _gemini_result(block: dict[str, Any]) -> ResultMatch | None:
    """Gemini CLI / Google ADK content parts: payload nests under `response`."""
    raw = block.get("functionResponse") or block.get("function_response")
    if isinstance(raw, dict):
        return raw, raw.get("id")
    return None


def _named_tool_call(block: dict[str, Any]) -> CallMatch | None:
    """Gemini CLI stream events: `{"tool_name", "parameters"}`."""
    if block.get("tool_name"):
        return block.get("id"), block.get("tool_name"), block.get("parameters")
    return None


def _args_call(block: dict[str, Any]) -> CallMatch | None:
    """A bare `{"name", "args"}` call, as Google ADK and Codex event items emit."""
    if isinstance(block.get("name"), str) and isinstance(block.get("args"), dict):
        return block.get("id"), block.get("name"), block.get("args")
    return None
