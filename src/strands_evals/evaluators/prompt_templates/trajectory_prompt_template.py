"""Rendering a trajectory into judge-prompt text."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from ...types.trace import Session

# Cap on serialized trajectory size in judge prompts, ~150k tokens at 4 chars/token.
MAX_TRAJECTORY_CHARS = 600_000


def serialize_trajectory(trajectory: Session | list[Any] | None, max_chars: int = MAX_TRAJECTORY_CHARS) -> str:
    """Serialize a trajectory into stable JSON, for use in judge prompts.

    The middle of a long run is dropped: a real trajectory can reach millions of tokens (one read
    of a large artifact is enough), which overflows any judge context window. The head and tail are
    kept because skills are loaded early and the outcome lands late.

    Args:
        trajectory: A `Session` or a raw message list, or None.
        max_chars: Size ceiling for the returned text. Pass 0 to keep the whole trajectory.

    Returns:
        str: The serialized trajectory, "(no trajectory)" when None, with an inline note naming the
        character count where the middle was omitted.
    """
    if trajectory is None:
        return "(no trajectory)"
    if isinstance(trajectory, Session):
        value: Any = trajectory.model_dump(mode="json")
    else:
        value = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in trajectory]
    text = json.dumps(value, indent=2, default=str)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    # The note says what a judge should infer from the gap, not just that there is one. The
    # instruction-following rubric offers "skipped" for a step with no visible evidence, so a long
    # but correct run would otherwise be marked down for the evidence that fell in the middle.
    return (
        f"{text[:keep]}\n\n"
        f"... [{len(text) - 2 * keep} characters omitted; evidence for a step may lie in this gap, "
        f"so do not treat a step as skipped solely because nothing here shows it] ...\n\n"
        f"{text[-keep:]}"
    )
