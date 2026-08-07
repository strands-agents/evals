"""Per-harness block shapes.

Each harness writes a tool call and a tool result its own way. These modules know only those
shapes: given one content block, they say whether it is a call or a result and pull out the
identifier, the name and the arguments or payload. Deciding what a call *means* (a skill load, a
file read, a refusal) is the extractor's job, not theirs.

One module per harness (`strands`, `claude`, `gemini`, `codex`, `openhands`), `registry` for the
order they are tried in and the `ToolCallBlock` / `ToolResultBlock` they produce, and `_common` for
what they share. A harness whose skill load is not a tool call at all (Codex reads `SKILL.md` with a
shell command) contributes only its result envelope here; recognizing the load is policy and lives
in `extractor`.
"""

from ._common import ToolCallBlock, ToolResultBlock

__all__ = [
    "ToolCallBlock",
    "ToolResultBlock",
]
