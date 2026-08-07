"""What the trajectory says the agent was offered and what it loaded.

This is the harness-independent half: given blocks the adapters have normalized, decide which
calls are skill loads, pair each with its result, and recover the body. That produces a
`SkillLoadEvent` per attempt, which `extract_selected_skills` then folds into one `InvokedSkill`
per skill. The public entry points are `parse_available_skills`, `extract_skill_load_events` and
`extract_selected_skills`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ...types.trace import (
    AgentInvocationSpan,
    Session,
    ToolExecutionSpan,
)
from ._normalize import (
    _as_dict,
    _body_from_result,
    _canonical_skill_key,
    _content_text,
    _last_path_segment,
    _load_refused,
    _parse_available_block,
    _refusal_message,
    _skill_name_from_args,
    _skill_name_from_body,
    _skill_path_from_text,
)
from ._patterns import (
    _CLAUDE_BASE_DIR,
    _DISCOVERY_TOOL_NAMES,
    _READ_TOOL_NAMES,
    _READ_VERBS,
    _SED_IN_PLACE,
    _SHELL_PREFIX,
    _SHELL_SEPARATOR,
    _SHELL_TOOL_NAMES,
    _SHELL_WRAPPER,
)
from .adapters._common import ToolResultBlock, _iter_indexed_blocks
from .adapters.registry import _tool_call, _tool_result
from .models import AvailableSkill, InvokedSkill, SkillLoadEvent

logger = logging.getLogger(__name__)


# ---- Recognizing a skill read ------------------------------------------------


def _shell_read_skill_path(command: str) -> str | None:
    """Return the `SKILL.md` path a shell command reads, or None if it does not read one.

    The verb and the path have to belong to the same command, and the path has to be an operand
    of that verb rather than anywhere in the line. Searching the whole command independently
    turns writes and unrelated work into phantom skill loads that the judge then scores:

        cat draft.md > /skills/new/SKILL.md   # creates a skill, does not load one
        sed -i 's/a/b/' /skills/pdf/SKILL.md  # edits it
        cat data.csv; ls -l /skills/pdf/SKILL.md
    """
    if wrapper := _SHELL_WRAPPER.match(command):
        command = wrapper.group("inner")

    for segment in _SHELL_SEPARATOR.split(command):
        # A redirection target is a write, not a read, whichever verb precedes it.
        head = segment.split(">")[0]
        while (stripped := _SHELL_PREFIX.sub("", head, count=1)) != head:
            head = stripped
        parts = head.split()
        if not parts:
            continue
        verb = parts[0].rsplit("/", 1)[-1].casefold()
        if verb not in _READ_VERBS:
            continue
        if verb == "sed" and _SED_IN_PLACE.search(head):
            continue  # `sed -i` rewrites the file
        operands = head[len(parts[0]) :]
        if path := _skill_path_from_text(operands):
            return path
    return None


def _skill_read_path(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Return a SKILL.md path only for recognizable file-read operations."""
    lowered = tool_name.casefold()
    if lowered in _READ_TOOL_NAMES or any(lowered.endswith(f".{name}") for name in _READ_TOOL_NAMES):
        for key in ("path", "file_path", "filename"):
            value = arguments.get(key)
            if isinstance(value, str) and (path := _skill_path_from_text(value)):
                return path

    if lowered in _SHELL_TOOL_NAMES:
        command = arguments.get("command") or arguments.get("cmd")
        if isinstance(command, str):
            return _shell_read_skill_path(command)
    return None


def _summarize_events(events: list[SkillLoadEvent]) -> list[InvokedSkill]:
    """Fold load attempts into one row per skill, in first-attempt order.

    Args:
        events: The attempts as they appeared in the trajectory.

    Returns:
        list: One `InvokedSkill` per skill, carrying the fullest body recovered for it.
    """
    out: list[InvokedSkill] = []
    index_by_key: dict[str, int] = {}
    for event in events:
        # An attempt whose outcome the trajectory never carried is still a selection the agent
        # made, so it is reported. It reads as loaded-without-a-body here, since "the agent asked
        # for this skill" is all a per-skill summary can say about it.
        summary = InvokedSkill(
            name=event.name,
            body=event.body,
            status="failed" if event.status == "failed" else "loaded",
            error=event.error,
        )
        key = _canonical_skill_key(event.name)
        index = index_by_key.get(key)
        if index is None:
            index_by_key[key] = len(out)
            out.append(summary)
            continue
        # One success anywhere in the run means the agent got the skill, so a retry after a
        # refusal is reported as loaded. Only an all-refused skill stays failed.
        if out[index].status == "failed" and summary.status == "loaded":
            out[index] = summary
            continue
        if summary.status == "failed":
            continue
        # Prefer the fullest body, and with it the name that body declares. A later read only
        # wins when it contains what was already recovered, which is what a re-read of the
        # same file looks like: a paged window is contained in the whole file. Unrelated
        # output that happened to be attributed to this skill is not, so it cannot displace
        # a real body just by being longer.
        kept = out[index].body or ""
        candidate = summary.body or ""
        if len(candidate) > len(kept) and kept in candidate:
            out[index] = summary
    return out


# ---- Session path -----------------------------------------------------------


def _available_from_session(session: Session) -> list[AvailableSkill]:
    """Recover available skills from the first AgentInvocationSpan.system_prompt that has the block."""
    for trace in session.traces:
        for span in trace.spans:
            if isinstance(span, AgentInvocationSpan) and span.system_prompt:
                skills = _parse_available_block(span.system_prompt)
                if skills:
                    return skills
    return []


def _agent_id_of(span: ToolExecutionSpan) -> str | None:
    """Which agent ran the span, when the trace records it.

    Read from `metadata` rather than a schema field, since the trace types carry no agent identity
    of their own. A multi-agent mapper that records one is honoured; the rest report None, which is
    truthful about a single-agent run and about a mapper that dropped the attribution.
    """
    metadata = span.metadata or {}
    for key in ("agent_id", "agent_name", "agent"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _events_from_session(session: Session) -> list[SkillLoadEvent]:
    """Recover load attempts from ToolExecutionSpans with a reserved skill-tool name.

    The skill body is taken from the tool result content when present. (Some
    harnesses put the body elsewhere, e.g. Claude Code's following message; those
    are handled by their raw-list adapters and are follow-ups for the Session path.)

    Args:
        session: The trace to read.

    Returns:
        list: One event per attempt, `position` counting tool spans across the whole session so it
        orders attempts made in different traces.
    """
    out: list[SkillLoadEvent] = []
    position = 0
    for trace in session.traces:
        for span in trace.spans:
            if not isinstance(span, ToolExecutionSpan):
                continue
            position += 1
            call_id = span.tool_result.tool_call_id or span.tool_call.tool_call_id
            agent_id = _agent_id_of(span)
            failed = bool(span.tool_result.error)
            skill_name = _skill_name_from_args(span.tool_call.name, span.tool_call.arguments)
            if skill_name is not None:
                if failed or _load_refused(span.tool_result.content):
                    out.append(
                        SkillLoadEvent(
                            name=skill_name,
                            status="failed",
                            error=span.tool_result.error or _refusal_message(span.tool_result.content),
                            call_id=call_id,
                            position=position,
                            agent_id=agent_id,
                        )
                    )
                else:
                    out.append(
                        SkillLoadEvent(
                            name=skill_name,
                            status="loaded",
                            body=_body_from_result(span.tool_result.content),
                            call_id=call_id,
                            position=position,
                            agent_id=agent_id,
                        )
                    )
                continue

            if failed:
                continue
            read_path = _skill_read_path(span.tool_call.name, span.tool_call.arguments)
            body = _body_from_result(span.tool_result.content)
            name = _skill_name_from_body(body, read_path) if read_path and body else ""
            if name:
                out.append(
                    SkillLoadEvent(
                        name=name,
                        status="loaded",
                        body=body,
                        call_id=call_id,
                        position=position,
                        agent_id=agent_id,
                    )
                )
    return out


# ---- Raw message-list path --------------------------------------------------
#
# Strands' native in-memory message shape: assistant messages carry
# {"toolUse": {"name", "toolUseId", "input": {...}}} blocks, and the following
# user message carries {"toolResult": {"toolUseId", "content": [...]}}. We also
# accept already-parsed strands_evals message objects (UserMessage/AssistantMessage).


def _structured_available_skills(message: Any) -> list[AvailableSkill]:
    """Recover structured catalogs, including discovery-tool response wrappers."""
    pending = [message]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))

        if isinstance(candidate, str) and candidate.lstrip().startswith(("{", "[")):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(candidate, list):
            pending.extend(candidate)
            continue

        value = _as_dict(candidate)
        if value is None:
            continue
        skills = value.get("skills")
        if isinstance(skills, list):
            out: list[AvailableSkill] = []
            for skill in skills:
                if isinstance(skill, str):
                    out.append(AvailableSkill(skill, ""))
                elif isinstance(skill, dict) and skill.get("name"):
                    out.append(AvailableSkill(str(skill["name"]), str(skill.get("description", ""))))
            if out:
                return out
        pending.extend(
            value[key]
            for key in (
                "response",
                "result",
                "output",
                "content",
                "toolResult",
                "functionResponse",
                "function_response",
                "toolResponse",
            )
            if key in value
        )
    return []


def _text_candidates(value: Any) -> list[str]:
    """Collect prompt/result text recursively without stringifying opaque objects."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _text_candidates(item)]
    item = _as_dict(value)
    if item is None:
        return []
    texts: list[str] = []
    for key in (
        "system_prompt",
        # Result wrappers, so a discovery tool's catalog is reachable: harnesses nest the
        # payload one level down (Bedrock `toolResult`, Gemini/ADK `functionResponse`).
        "toolResult",
        "functionResponse",
        "function_response",
        "toolResponse",
        "content",
        "text",
        "output",
        "aggregated_output",
        "llmContent",
        "instructions",
        "message",
        "response",
        "result",
    ):
        if key in item:
            texts.extend(_text_candidates(item[key]))
    return texts


def _discovery_tool_name(block: dict[str, Any]) -> str | None:
    for candidate in (
        block.get("name"),
        block.get("tool_name"),
        block.get("functionResponse"),
        block.get("function_response"),
        block.get("toolResponse"),
    ):
        if isinstance(candidate, str):
            return candidate
        value = _as_dict(candidate)
        if value is not None and isinstance(value.get("name"), str):
            return value["name"]
    return None


def _result_id(block: dict[str, Any]) -> str | None:
    candidates: list[Any] = [
        block.get("tool_call_id"),
        block.get("tool_use_id"),
        block.get("id"),
    ]
    for key in ("toolResult", "functionResponse", "function_response", "toolResponse"):
        value = _as_dict(block.get(key))
        if value is not None:
            candidates.extend(
                (
                    value.get("toolUseId"),
                    value.get("tool_call_id"),
                    value.get("tool_use_id"),
                    value.get("id"),
                )
            )
    return next((str(candidate) for candidate in candidates if candidate is not None), None)


def _is_discovery_tool_name(name: str) -> bool:
    lowered = name.casefold()
    return lowered in _DISCOVERY_TOOL_NAMES or any(
        lowered.endswith(f"_{discovery_name}") for discovery_name in _DISCOVERY_TOOL_NAMES
    )


def _available_from_list(messages: list[Any]) -> list[AvailableSkill]:
    """Parse trusted system catalogs and skill-discovery tool results."""
    indexed_blocks = _iter_indexed_blocks(messages)
    discovery_ids = {
        call.call_id
        for _, _, block in indexed_blocks
        if (call := _tool_call(block)) is not None and _is_discovery_tool_name(call.name) and call.call_id is not None
    }

    for msg in messages:
        outer = _as_dict(msg)
        if outer is None:
            continue
        message = _as_dict(outer.get("message")) or outer
        role = str(message.get("role") or outer.get("role") or "").casefold()
        is_system = role in {"system", "developer"} or str(outer.get("type", "")).casefold() == "system"
        if is_system:
            structured = _structured_available_skills(msg)
            if structured:
                return structured
            for text in _text_candidates(msg):
                skills = _parse_available_block(text)
                if skills:
                    return skills
        elif "system_prompt" in outer:
            for text in _text_candidates(outer["system_prompt"]):
                skills = _parse_available_block(text)
                if skills:
                    return skills

    for _, _, block in indexed_blocks:
        tool_name = _discovery_tool_name(block)
        is_discovery_result = (isinstance(tool_name, str) and _is_discovery_tool_name(tool_name)) or _result_id(
            block
        ) in discovery_ids
        if not is_discovery_result:
            continue
        structured = _structured_available_skills(block)
        if structured:
            return structured
        for text in _text_candidates(block):
            skills = _parse_available_block(text)
            if skills:
                return skills
    return []


def _claude_body_after(
    indexed_blocks: list[tuple[int, str | None, dict[str, Any]]],
    call_index: int,
    skill_name: str,
) -> str | None:
    """Find Claude Code's injected skill body after its launch acknowledgement.

    The body is matched to the call by the skill directory named on the `Base directory` line,
    not by position: Claude Code can launch several skills in one assistant turn, and then the
    injected bodies arrive in an order the call order does not fix. Taking the first block after
    the call gives every skill in the turn the first skill's instructions, which the adherence
    judge would then score against the wrong steps.
    """
    candidates: list[tuple[str, str]] = []  # (base directory, full injected text)
    for index, role, block in indexed_blocks:
        if index <= call_index or role not in (None, "user"):
            continue
        text = _content_text(block.get("text") if block.get("type") == "text" else block)
        if match := _CLAUDE_BASE_DIR.match(text.lstrip()):
            candidates.append((_last_path_segment(match.group("path")), text))

    wanted = _canonical_skill_key(skill_name)
    for directory, text in candidates:
        if _canonical_skill_key(directory) == wanted:
            return text
    # No directory matched. With one candidate that is still this call's body, since the directory
    # can be an alias for the name in the frontmatter. With several it is unknowable which belongs
    # to this call, and guessing would attribute another skill's instructions to it.
    return candidates[0][1] if len(candidates) == 1 else None


def _events_from_list(messages: list[Any]) -> list[SkillLoadEvent]:
    """Recover load attempts from raw or typed message lists.

    Matches assistant `toolUse` blocks with a reserved skill-tool name, then
    pairs each with the `toolResult` block (by toolUseId) that carries the body.
    Typed `ToolCallContent` / `ToolResultContent` blocks use the equivalent
    `content_type` and `tool_call_id` fields.

    Args:
        messages: The raw message list.

    Returns:
        list: One event per attempt, `position` being the index in `messages` the attempt was
        found at. An attempt whose result the list never carries is reported as "attempted".
    """
    indexed_blocks = _iter_indexed_blocks(messages)
    results_by_id: dict[str, ToolResultBlock] = {}
    unkeyed_results: list[tuple[int, ToolResultBlock]] = []
    for result_index, _, block in indexed_blocks:
        parsed_result = _tool_result(block)
        if parsed_result is None:
            continue
        if parsed_result.call_id is not None:
            results_by_id[parsed_result.call_id] = parsed_result
        else:
            unkeyed_results.append((result_index, parsed_result))

    # Every tool call's position, so an unkeyed result is only paired with the call it follows
    # directly. Without that bound the next unclaimed result wins, and an unrelated tool's output
    # in between is attributed to the skill: the adherence judge then scores the agent against a
    # weather report instead of the skill's steps.
    call_indices = sorted({index for index, _, block in indexed_blocks if _tool_call(block) is not None})

    out: list[SkillLoadEvent] = []
    used_unkeyed_results: set[int] = set()
    for message_index, _, block in indexed_blocks:
        if block.get("type") == "command_execution":
            command = block.get("command")
            body = _body_from_result(block)
            path = _shell_read_skill_path(command) if isinstance(command, str) else None
            name = _skill_name_from_body(body, path) if path and body else ""
            if name:
                out.append(
                    SkillLoadEvent(
                        name=name,
                        status="loaded",
                        body=body,
                        position=message_index,
                    )
                )
            continue

        call = _tool_call(block)
        if call is None:
            continue
        matched_result = results_by_id.get(call.call_id) if call.call_id is not None else None
        if matched_result is None and call.call_id is None:
            next_call_index = next((index for index in call_indices if index > message_index), None)
            unkeyed_match = next(
                (
                    (index, result)
                    for index, result in unkeyed_results
                    if message_index < index and index not in used_unkeyed_results
                    if next_call_index is None or index < next_call_index
                ),
                None,
            )
            if unkeyed_match is not None:
                used_unkeyed_results.add(unkeyed_match[0])
                matched_result = unkeyed_match[1]

        skill_name = _skill_name_from_args(call.name, call.arguments)
        if skill_name is not None:
            if matched_result is not None and matched_result.refused:
                out.append(
                    SkillLoadEvent(
                        name=skill_name,
                        status="failed",
                        error=matched_result.error,
                        call_id=call.call_id,
                        position=message_index,
                    )
                )
                continue
            body = matched_result.body if matched_result is not None else None
            if call.name == "Skill" and body is None:
                body = _claude_body_after(indexed_blocks, message_index, skill_name)
            out.append(
                SkillLoadEvent(
                    name=skill_name,
                    # No result and no injected body means the trajectory stops before the
                    # outcome, a different run from one that loaded and whose body went uncaptured.
                    status="attempted" if matched_result is None and body is None else "loaded",
                    body=body,
                    call_id=call.call_id,
                    position=message_index,
                )
            )
            continue

        read_path = _skill_read_path(call.name, call.arguments)
        if read_path is None or matched_result is None or matched_result.refused or not matched_result.body:
            continue
        # An unnamed read is not a skill load: see `_skill_name_from_body`.
        name = _skill_name_from_body(matched_result.body, read_path)
        if name:
            out.append(
                SkillLoadEvent(
                    name=name,
                    status="loaded",
                    body=matched_result.body,
                    call_id=call.call_id,
                    position=message_index,
                )
            )
    return out


# ---- Public API -------------------------------------------------------------


def parse_available_skills(trajectory: Session | list[Any] | str | None) -> list[AvailableSkill]:
    """Return the skills exposed to the agent (name + description).

    Args:
        trajectory: A `Session`, a raw message list, or a bare prompt string (e.g. a harness's
            system prompt, which is where the block is injected but which some session mappers
            store separately from the message list).

    Returns:
        list: The advertised skills, or [] when no `<available_skills>` block is found.
    """
    if isinstance(trajectory, Session):
        return _available_from_session(trajectory)
    if isinstance(trajectory, str):
        return _parse_available_block(trajectory)
    if isinstance(trajectory, list):
        return _available_from_list(trajectory)
    if trajectory is not None:
        logger.debug("type=<%s> | unsupported trajectory type for available skills", type(trajectory).__name__)
    return []


def extract_skill_load_events(trajectory: Session | list[Any] | None) -> list[SkillLoadEvent]:
    """Return every skill load attempt, in trajectory order.

    The harness-independent form, before any folding: repeated loads of one skill are separate
    events, and a refusal followed by a successful retry is two. Use this where the individual
    attempts matter (how often a skill was reloaded, which agent loaded it, whether an attempt's
    outcome was ever recorded); `extract_selected_skills` gives the per-skill summary.

    Args:
        trajectory: A `Session` or a raw message list.

    Returns:
        list: One `SkillLoadEvent` per attempt, or [] when the trajectory carries none.
    """
    if isinstance(trajectory, Session):
        return _events_from_session(trajectory)
    if isinstance(trajectory, list):
        return _events_from_list(trajectory)
    if trajectory is not None:
        logger.debug("type=<%s> | unsupported trajectory type for skill load events", type(trajectory).__name__)
    return []


def extract_selected_skills(trajectory: Session | list[Any] | None) -> list[InvokedSkill]:
    """Return the skills the agent selected, one row per skill, in first-attempt order.

    Args:
        trajectory: A `Session` or a raw message list.

    Returns:
        list: One `InvokedSkill` per skill, carrying the `SKILL.md` body when the trajectory
        surfaced it (else `None`) and `status="failed"` with the harness's message when every
        attempt was refused.
    """
    return _summarize_events(extract_skill_load_events(trajectory))
