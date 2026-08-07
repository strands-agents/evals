"""OpenHands action and observation shapes.

OpenHands models a skill load as a first-class event pair rather than a tool call:
`{"kind": "InvokeSkillAction", "name": "<skill>"}` and `{"kind": "InvokeSkillObservation"}`. The
skill name is the action's own `name`, not an argument, so the recognizer reports a synthetic
`invoke_skill` call carrying it as one. That keeps the extractor's skill-name lookup uniform across
harnesses instead of special-casing this one.
"""

from __future__ import annotations

from typing import Any

from ._common import CallMatch, ResultMatch


def _openhands_call(block: dict[str, Any]) -> CallMatch | None:
    """OpenHands: `{"kind": "InvokeSkillAction", "name"}`, with the skill name as the action name."""
    if block.get("kind") == "InvokeSkillAction":
        return block.get("tool_call_id") or block.get("id"), "invoke_skill", {"name": block.get("name")}
    return None


def _openhands_result(block: dict[str, Any]) -> ResultMatch | None:
    """OpenHands: `{"kind": "InvokeSkillObservation"}`."""
    if block.get("kind") == "InvokeSkillObservation":
        return block, block.get("tool_call_id") or block.get("id")
    return None
