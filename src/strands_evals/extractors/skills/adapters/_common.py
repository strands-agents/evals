"""What every adapter returns, and the block-flattening they all feed from.

A recognizer is a function from one content block to either the shape it knows or None. Calls
report `(call_id, name, arguments)` and results report `(raw_payload, call_id)`, both unvalidated:
`registry._tool_call` drops a call whose name or arguments are not the expected type, and
`registry._tool_result` interprets the payload. Keeping recognizers this thin is what lets a single
harness module be read on its own.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .._normalize import _as_dict

# A recognized call, as its harness module returns it: (call_id, name, arguments).
CallMatch = tuple[Any, Any, Any]
# A recognized result: (raw payload, call_id). The payload is handed to `_normalize` to read.
ResultMatch = tuple[Any, Any]


class ToolCallBlock(NamedTuple):
    """A tool call recovered from a trajectory, harness-independent."""

    call_id: str | None
    name: str
    arguments: dict[str, Any]


class ToolResultBlock(NamedTuple):
    """A tool result recovered from a trajectory, harness-independent."""

    call_id: str | None
    refused: bool
    body: str | None
    error: str | None


def _looks_like_block(value: dict[str, Any]) -> bool:
    return (
        "toolUse" in value
        or "toolResult" in value
        or value.get("type")
        in {"tool_use", "tool_result", "text", "command_execution", "function_call", "function_call_output"}
        or "tool_name" in value
        or ("name" in value and "args" in value)
        or str(value.get("kind", "")).startswith("InvokeSkill")
    )


def _iter_indexed_blocks(messages: list[Any]) -> list[tuple[int, str | None, dict[str, Any]]]:
    """Flatten raw, Claude stream, and Codex event wrappers into content blocks."""
    blocks: list[tuple[int, str | None, dict[str, Any]]] = []
    for index, item in enumerate(messages):
        outer = _as_dict(item)
        if outer is None:
            continue

        if outer.get("type") == "item.completed" and (codex_item := _as_dict(outer.get("item"))):
            blocks.append((index, None, codex_item))
            continue
        if outer.get("type") in {
            "tool_response",
            "function_response",
            # Responses API items, as a Codex session rollout records them: the item *is* the
            # block, with no message envelope around it.
            "function_call",
            "function_call_output",
        } or str(outer.get("kind", "")).startswith("InvokeSkill"):
            blocks.append((index, None, outer))
            continue

        message = _as_dict(outer.get("message")) or outer
        role = message.get("role") or outer.get("role")
        # Google GenAI puts the blocks in `parts`, not `content`: a `types.Content` dumps to
        # `{"role", "parts": [...]}`, which is what Gemini and Google ADK trajectories are made of.
        content = message.get("content", message.get("parts"))
        if isinstance(content, list):
            blocks.extend(
                (index, str(role) if role else None, block)
                for content_item in content
                if (block := _as_dict(content_item)) is not None
            )
        elif (block := _as_dict(content)) is not None:
            blocks.append((index, str(role) if role else None, block))
        elif isinstance(content, str):
            blocks.append((index, str(role) if role else None, {"type": "text", "text": content}))
        elif _looks_like_block(message):
            blocks.append((index, str(role) if role else None, message))
    return blocks
