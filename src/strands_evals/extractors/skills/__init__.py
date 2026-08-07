"""Skill trajectory parsing helpers.

`parse_available_skills` recovers the skills exposed to the agent (name plus
description) from the harness-injected `<available_skills>` block.
`extract_skill_load_events` recovers every load attempt in trajectory order, and
`extract_selected_skills` folds those into one row per skill, each with its
`SKILL.md` body when the trajectory carried it. All accept a `Session` or a raw
message list.

Skills are not first-class in the trace schema the way tools are: there is no
`AgentInvocationSpan.available_skills` field, and a skill invocation surfaces as
an ordinary tool call. A load is detected either by a reserved skill-tool name and
its skill-name argument, or by a read of a known `SKILL.md` path.

The work is split three ways: `models` holds what the extractors return, `adapters`
holds the per-harness block shapes, and `extractor` holds the harness-independent
decisions about which calls are skill loads. `_patterns` and `_normalize` are the
literals and primitives the other two share.
"""

from .extractor import extract_selected_skills, extract_skill_load_events, parse_available_skills
from .models import AvailableSkill, InvokedSkill, SkillLoadEvent

__all__ = [
    "AvailableSkill",
    "InvokedSkill",
    "SkillLoadEvent",
    "extract_selected_skills",
    "extract_skill_load_events",
    "parse_available_skills",
]
