"""Codex and OpenAI Agents event shapes.

Both emit events rather than messages, and Codex emits them in two different envelopes depending
on how it was run:

- `codex exec --json` streams `{"type": "item.completed", "item": {"type": "command_execution", ...}}`,
  unwrapped by `_common._iter_indexed_blocks`.
- The interactive CLI writes a session rollout of OpenAI Responses API items, so the same shell
  call arrives as `{"type": "function_call", "name": "exec_command", "arguments": "<json>"}` with
  its output in a matching `function_call_output`.

Either way a Codex skill load is a shell read of a `SKILL.md` path rather than a reserved skill
tool, so recognizing the load is the extractor's job. OpenAI Agents exposes a real `load_skill`
tool, whose call rides the `{"name", "args"}` recognizer in `gemini`.

What is left here are the call and result envelopes.
"""

from __future__ import annotations

import json
from typing import Any

from .._patterns import _CODEX_EXEC_OUTPUT, _CODEX_EXIT_CODE
from ._common import CallMatch, ResultMatch


def _function_call(block: dict[str, Any]) -> CallMatch | None:
    """A Responses API `function_call` item, as a Codex session rollout records one.

    `arguments` is a JSON string rather than an object. A call whose arguments do not decode to a
    dict is reported with an empty one instead of being dropped, so the call still counts as an
    attempt.
    """
    if block.get("type") != "function_call":
        return None
    raw = block.get("arguments")
    arguments: Any = raw
    if isinstance(raw, str):
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            arguments = {}
    return block.get("call_id") or block.get("id"), block.get("name"), arguments if isinstance(arguments, dict) else {}


def _function_call_output(block: dict[str, Any]) -> ResultMatch | None:
    """A Responses API `function_call_output` item, with Codex's shell preamble stripped.

    Codex prefixes the command's output with its own header lines (chunk id, wall time, exit code,
    token count). The exit code is lifted out so a failed read is recognized as one, and the rest
    of the preamble is dropped: it is the harness talking about the command, not the command's
    output, and a body that starts with "Wall time" parses as neither frontmatter nor instructions.
    """
    if block.get("type") != "function_call_output":
        return None
    payload: dict[str, Any] = {"tool_call_id": block.get("call_id") or block.get("id")}
    output = block.get("output")
    if isinstance(output, str):
        if match := _CODEX_EXEC_OUTPUT.match(output):
            payload["output"] = match.group("output")
            if code := _CODEX_EXIT_CODE.search(match.group("preamble")):
                payload["exit_code"] = int(code.group("code"))
        else:
            payload["output"] = output
    else:
        payload["output"] = output
    return payload, payload["tool_call_id"]


def _event_result(block: dict[str, Any]) -> ResultMatch | None:
    """A `tool_response` / `function_response` event, as OpenAI Agents and Codex emit."""
    if block.get("type") in {"tool_response", "function_response"}:
        return block, block.get("tool_use_id") or block.get("id")
    return None
