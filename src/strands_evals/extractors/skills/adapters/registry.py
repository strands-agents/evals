"""The order the harness recognizers are tried in, and the common blocks they produce.

Each harness module knows one wire format and nothing about the others. This module is the only
place that knows they compete: a block is offered to each recognizer in turn and the first usable
match wins. A recognizer that matches the block's shape but cannot read a name and arguments out of
it does not end the search, because the same block can be a truncated call in one shape and a whole
one in another; see `_tool_call`.

**The order is behavior, not style.** Shapes overlap, so a block can satisfy more than one
recognizer and the first one reached decides how it is read:

- `strands._bedrock_*` before `claude._anthropic_*`: a harness can wrap a `toolUse` and also tag the
  block `type: "tool_use"`, and the wrapper carries the identifier the flat block lacks.
- `gemini._args_call` last among the calls: `{"name", "args"}` is the loosest shape here, and any
  block with a string name and a dict of arguments matches it, including blocks a more specific
  recognizer would have read correctly. Order alone is not enough to protect it once the search can
  continue past a match, so it is also withheld from any block that declares a harness
  (`_UNTAGGED_CALL_ADAPTERS`).
- `openhands._openhands_call` after the rest: its blocks are `kind`-tagged and cannot collide, so
  its position is free, but keeping it last leaves the loose recognizers' relative order intact.

Adding a harness means adding a module and one entry here. Put it above `_args_call` unless its
blocks are tagged in a way nothing else matches.
"""

from __future__ import annotations

from typing import Any

from .._normalize import _body_from_result, _load_refused, _refusal_message
from . import claude, codex, gemini, openhands, strands
from ._common import ToolCallBlock, ToolResultBlock

_CALL_ADAPTERS = (
    strands._bedrock_call,
    gemini._gemini_call,
    strands._typed_call,
    claude._anthropic_call,
    codex._function_call,
    gemini._named_tool_call,
    gemini._args_call,
    openhands._openhands_call,
)


_RESULT_ADAPTERS = (
    strands._bedrock_result,
    gemini._gemini_result,
    strands._typed_result,
    claude._anthropic_result,
    codex._function_call_output,
    codex._event_result,
    openhands._openhands_result,
)

# The recognizers that match on shape alone, with no tag naming the harness they belong to. They
# are correct for the harnesses that emit a bare call, and wrong for anything else, so they are the
# ones `_tool_call` withholds from a block that already declares a harness.
_UNTAGGED_CALL_ADAPTERS = frozenset({gemini._args_call})

# The keys and `type` values that name the harness a block came from. A block carrying one is that
# harness's block, well-formed or not.
_HARNESS_TAGS = ("toolUse", "functionCall", "function_call", "content_type", "tool_name", "kind")
_HARNESS_TYPES = frozenset({"tool_use", "function_call"})


def _declares_a_harness(block: dict[str, Any]) -> bool:
    """Whether this block identifies which harness's call shape it is."""
    return any(key in block for key in _HARNESS_TAGS) or block.get("type") in _HARNESS_TYPES


def _tool_call(block: dict[str, Any]) -> ToolCallBlock | None:
    """The tool call this block carries, or None if it is not one.

    A recognizer that matches but reports an unusable name or arguments does not end the search.
    Shapes overlap, so the same block can be a malformed instance of one harness's call and a
    well-formed instance of another's: the dual-tagged block in this module's docstring carries
    both a `toolUse` wrapper and the flat `type: "tool_use"` fields, and a truncated wrapper there
    should not hide the flat fields that did survive.

    Falling through would not be safe on its own. Every recognizer but `_args_call` is gated on a
    tag that names its harness, so a second reading is a second reading of the same call.
    `_args_call` is gated on nothing but the presence of a string `name` and a dict `args`, and
    until now only its position at the end of the registry kept it away from blocks a specific
    recognizer had already claimed. Reached after a malformed match it would read those two keys off
    a block that is not its shape: `{"toolUse": {...}, "name": "other", "args": {...}}` is one
    malformed Bedrock call, not a valid bare one, and reporting `other` would attribute a skill the
    agent never asked for. So a block that declares a harness is offered only to the recognizers
    that read tagged shapes.
    """
    tagged_only = _declares_a_harness(block)
    for adapter in _CALL_ADAPTERS:
        if tagged_only and adapter in _UNTAGGED_CALL_ADAPTERS:
            continue
        matched = adapter(block)
        if matched is None:
            continue
        call_id, name, arguments = matched
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        return ToolCallBlock(str(call_id) if call_id is not None else None, name, arguments)
    return None


def _tool_result(block: dict[str, Any]) -> ToolResultBlock | None:
    """The tool result this block carries, or None if it is not one."""
    for adapter in _RESULT_ADAPTERS:
        matched = adapter(block)
        if matched is not None:
            raw_result, result_id = matched
            refused = _load_refused(raw_result)
            return ToolResultBlock(
                call_id=str(result_id) if result_id is not None else None,
                refused=refused,
                body=_body_from_result(raw_result),
                error=_refusal_message(raw_result) if refused else None,
            )
    return None
