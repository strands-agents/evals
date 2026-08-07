"""Reading text, statuses and skill names out of whatever shape a harness used.

These are the primitives the adapters and the extraction policy share: flattening content
wrappers, deciding whether a result failed, and naming a skill from its body or its path.
"""

from __future__ import annotations

import html
import json
from typing import Any

import yaml
from pydantic import BaseModel
from strands import Skill

from ._patterns import (
    _ACKNOWLEDGEMENT,
    _AVAILABLE_BLOCK,
    _AVAILABLE_MARKDOWN,
    _AVAILABLE_MARKDOWN_ENTRY,
    _DESC_TAG,
    _FAILED_STATUSES,
    _FILE_LOCATOR,
    _HARNESS_TOOLS,
    _LOAD_ERROR,
    _NAME_TAG,
    _SKILL_ENTRY,
    _SKILL_PATH,
    _SUCCESS_EXIT_CODES,
)
from .models import AvailableSkill


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Normalize raw dictionaries and strands_evals Pydantic trace objects."""
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return None


def _content_text(content: Any) -> str:
    """Flatten common text/content wrappers into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_content_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    item = _as_dict(content)
    if item is not None:
        for key in (
            "instructions",
            "llmContent",
            "content",
            "output",
            "aggregated_output",
            "text",
            # Google ADK nests its tool payload under `response`, with plain tool output
            # under `result`; both must be traversed to reach the skill body or catalog.
            "response",
            "result",
        ):
            if key in item:
                text = _content_text(item[key])
                if text:
                    return text
    return ""


def _result_failed(result: Any) -> bool:
    value = _as_dict(result)
    if value is None:
        return False
    status = str(value.get("status", "")).casefold()
    exit_code = value.get("exit_code")
    failed = (
        status in _FAILED_STATUSES
        or value.get("is_error") is True
        or value.get("error") not in (None, "", False)
        or exit_code not in _SUCCESS_EXIT_CODES
    )
    if failed:
        return True
    return any(
        _result_failed(value[key])
        for key in ("response", "result")
        if key in value and _as_dict(value[key]) is not None
    )


def _load_refused(result: Any) -> bool:
    """Whether the load was refused, including refusals the harness marks successful.

    A plugin can report a lookup failure in the payload rather than in the status. The Strands
    AgentSkills plugin returns "Skill 'x' not found. ..." as a plain string from an `@tool`
    function, and `@tool` reports a plain string return as `status="success"`, so the only signal
    that no skill was loaded is the text. Recognizing it here is what separates a refused load
    from a load whose body simply was not captured: both have `body=None`, but only the refusal
    means the agent never received instructions.
    """
    return _result_failed(result) or bool(_LOAD_ERROR.fullmatch(_content_text(result).strip()))


def _refusal_message(result: Any) -> str | None:
    """What the harness said when it refused, or None when it said nothing usable.

    Reported alongside the refusal because which refusal it was decides what to fix, and the
    text is the only place that distinction survives: "Skill 'pdf-procesing' not found" is a
    misspelled name in the agent's call, while "Available skills: (none)" is a harness that
    mounted nothing. A structured failure can carry no message at all, hence the None.

    The harness's own `error` field is preferred over the result text, including when it is
    nested a level down the way `_result_failed` finds it, since that is where a harness that
    keeps its diagnostics apart from tool output puts them.
    """
    error = _error_field(result)
    if error is not None:
        return error
    return _content_text(result).strip() or None


def _error_field(result: Any) -> str | None:
    """The harness's `error` message, at the top level or nested under `response`/`result`."""
    value = _as_dict(result)
    if value is None:
        return None
    error = value.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    for key in ("response", "result"):
        if key in value:
            nested = _error_field(value[key])
            if nested is not None:
                return nested
    return None


def _body_from_result(result: Any) -> str | None:
    """Return actual skill instructions, excluding errors and load acknowledgements."""
    if _result_failed(result):
        return None
    text = _content_text(result).strip()
    if not text or _ACKNOWLEDGEMENT.fullmatch(text) or _LOAD_ERROR.fullmatch(text):
        return None

    # Some tool integrations JSON-encode their structured result.
    if text.startswith(("{", "[")):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if decoded != result:
                return _body_from_result(decoded)
    return text


def _last_path_segment(path: str) -> str:
    """The final component of a directory path, with separators normalized."""
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _skill_name_from_path(path: str) -> str:
    """The directory a `SKILL.md` sits in, which is the skill's name by convention.

    Returns "" when the path has no usable parent directory. A bare `SKILL.md` or `./SKILL.md`
    names no skill, and answering `SKILL.md` or `.` would put a name that is wrong on its face in
    front of the judge. Callers treat the empty result as "not a skill read".
    """
    normalized = path.replace("\\", "/").rstrip("/")
    parts = normalized.split("/")
    if len(parts) < 2:
        return ""
    parent = parts[-2]
    return "" if parent in {"", ".", ".."} else parent


def _canonical_skill_key(name: str) -> str:
    """Fold the naming variants that refer to one skill.

    An agent may read the same `SKILL.md` more than once, and a partial read (a paged `sed`
    window that misses the frontmatter) falls back to the directory name while a full read
    reports the frontmatter name. A directory named `pdf_processing` holding a skill whose
    frontmatter says `pdf-processing` is the same skill, so the two separators fold together.
    `.` is left alone: it is legal in a skill name, so folding it would merge `data.clean`
    and `data-clean`, which are two different skills.
    """
    return name.casefold().replace("_", "-")


def _skill_name_from_body(body: str, path: str) -> str:
    """Prefer the runtime-visible frontmatter name over a directory alias.

    `Skill.from_content` parses the `SKILL.md` YAML frontmatter. Bodies are
    read from arbitrary on-disk files, so malformed frontmatter is expected;
    `yaml` raises `YAMLError` (not a `ValueError`) on bad structure. Both
    fall back to the directory-derived name rather than aborting extraction.

    Returns "" when neither source yields a name, which callers treat as "not a skill read". A
    file literally named `SKILL.md` with no frontmatter and no parent directory carries nothing
    that identifies a skill, and any name invented for it would be wrong.
    """
    try:
        return Skill.from_content(body).name
    except (ValueError, yaml.YAMLError):
        return _skill_name_from_path(path)


def _skill_path_from_text(value: str) -> str | None:
    match = _SKILL_PATH.search(value)
    if not match:
        return None
    return next(group for group in match.groups() if group is not None)


def _skill_name_from_args(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Read the skill name from a reserved tool's input arguments."""
    key = _HARNESS_TOOLS.get(tool_name)
    if key is None and tool_name.casefold().endswith("_load_skill"):
        key = "skill_name"  # Google ADK permits a tool_name_prefix.
    if key is None:
        return None
    value = arguments.get(key)
    # A missing or empty name is a malformed call rather than a selection, and it yields no event:
    # there is no skill to attribute the attempt to. Harnesses do refuse these for real (the Strands
    # plugin answers "Error: skill_name is required.", Google ADK answers with an INVALID_ARGUMENTS
    # error code), but reporting the attempt would mean inventing a skill named "" and counting it
    # against the agent's selection accuracy, which is worse than not seeing the fumbled call.
    return str(value) if value else None


def _parse_available_block(text: str) -> list[AvailableSkill]:
    """Parse an XML or Markdown available-skills section from prompt text.

    Skills missing a name are skipped. A missing description yields an empty
    description rather than dropping the skill.
    """
    block_match = _AVAILABLE_BLOCK.search(text or "")
    if block_match:
        out: list[AvailableSkill] = []
        for entry in _SKILL_ENTRY.finditer(block_match.group(1)):
            body = entry.group(1)
            name_m = _NAME_TAG.search(body)
            if not name_m:
                continue
            desc_m = _DESC_TAG.search(body)
            out.append(
                AvailableSkill(
                    name=html.unescape(name_m.group(1).strip()),
                    description=html.unescape(desc_m.group(1).strip()) if desc_m else "",
                )
            )
        return out

    markdown_match = _AVAILABLE_MARKDOWN.search(text or "")
    if not markdown_match:
        return []
    return [
        AvailableSkill(
            name=match.group("name").strip(),
            description=_FILE_LOCATOR.sub("", match.group("description")).strip(),
        )
        for match in _AVAILABLE_MARKDOWN_ENTRY.finditer(markdown_match.group("body"))
    ]
