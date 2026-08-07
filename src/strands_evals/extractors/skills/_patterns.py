"""The literal text and shapes the harnesses emit.

Everything here is a fact about some harness's output format, keyed by the design-doc B.1
table and verified against real runs. Kept apart from the extraction policy so that adding a
harness is a change to this module and its adapter, not to the traversal logic.
"""

from __future__ import annotations

import re

# Reserved skill-tool name -> the input-argument key that holds the skill name.
_HARNESS_TOOLS: dict[str, str] = {
    "skills": "skill_name",  # Strands AgentSkills plugin
    "Skill": "skill",  # Claude Code / Claude Agent SDK
    "load_skill": "skill_name",  # OpenAI Agents SDK, Google ADK
    "activate_skill": "name",  # Gemini CLI
    "invoke_skill": "name",  # OpenHands
}

# Claude Code injects the skill body as a user message headed by the skill's directory.
_CLAUDE_BASE_DIR = re.compile(r"Base directory for this skill:\s*(?P<path>\S+)", re.IGNORECASE)

# The available-skills block the harness injects into the system prompt.
_AVAILABLE_BLOCK = re.compile(r"<available_skills>(.*?)</available_skills>", re.DOTALL | re.IGNORECASE)
_SKILL_ENTRY = re.compile(r"<skill>(.*?)</skill>", re.DOTALL | re.IGNORECASE)
_NAME_TAG = re.compile(r"<name>(.*?)</name>", re.DOTALL | re.IGNORECASE)
_DESC_TAG = re.compile(r"<description>(.*?)</description>", re.DOTALL | re.IGNORECASE)
_AVAILABLE_MARKDOWN = re.compile(
    r"^### Available skills\s*$\n(?P<body>.*?)(?=^###\s|\Z)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_AVAILABLE_MARKDOWN_ENTRY = re.compile(
    r"^\s*-\s+(?P<name>[^:\n]+):\s*(?P<description>.*?)\s*$",
    re.MULTILINE,
)
_FILE_LOCATOR = re.compile(r"\s+\(file:\s*.+?\)\s*$", re.IGNORECASE)
_SKILL_PATH = re.compile(
    r'"([^"\n]*SKILL\.md)"|\'([^\'\n]*SKILL\.md)\'|([^\s"\'=;|<>]*SKILL\.md)',
    re.IGNORECASE,
)
_READ_VERBS = {"cat", "sed", "head", "tail", "bat", "less", "type", "get-content"}
# `bash -lc "..."`, `/bin/sh -c '...'`: the real command is inside the quotes.
_SHELL_WRAPPER = re.compile(
    r"^\s*(?:\S*/)?(?:ba|da|k|z)?sh\s+(?:-\w+\s+)*(?P<quote>['\"])(?P<inner>.*)(?P=quote)\s*$",
    re.DOTALL,
)
# Command separators. Splitting inside a quoted argument is harmless here: the pieces are only
# used to locate a read verb and a path, and a quoted separator does not produce either.
_SHELL_SEPARATOR = re.compile(r"\|\||&&|[;|\n]")
# Leading noise before the verb: `sudo`, `env`, `FOO=bar`, `command`, `time`.
_SHELL_PREFIX = re.compile(r"^(?:sudo|env|command|time|nohup|\S+=\S*)\s+", re.IGNORECASE)
_SED_IN_PLACE = re.compile(r"(?:^|\s)(?:-i\S*|--in-place\b)")
_READ_TOOL_NAMES = {
    "read",
    "read_file",
    "file_read",
    "filesystem_read",
    "read_text_file",
}
_SHELL_TOOL_NAMES = {"bash", "shell", "terminal", "command", "execute_command", "exec_command", "run_command"}
# Codex wraps shell output in a fixed preamble before the output itself, e.g.
#     Chunk ID: 92cb1e
#     Wall time: 0.0631 seconds
#     Process exited with code 0
#     Original token count: 63
#     Output:
#     ---
#     name: pdf-processing
# Leaving the preamble attached would make the skill body start with the wall time, so the
# frontmatter would not parse and the judge would score the envelope as instructions.
_CODEX_EXEC_OUTPUT = re.compile(
    r"\A(?P<preamble>(?:[A-Za-z][A-Za-z ]*:.*\n|Process exited with code\s+-?\d+\n)*?)Output:\n(?P<output>.*)\Z",
    re.DOTALL,
)
_CODEX_EXIT_CODE = re.compile(r"^Process exited with code\s+(?P<code>-?\d+)\s*$", re.MULTILINE)
_DISCOVERY_TOOL_NAMES = {"list_skills", "search_skills"}
_FAILED_STATUSES = {"error", "errored", "fail", "failed", "failure", "cancelled", "canceled"}
# Exit codes that still leave usable output on stdout. 141 is SIGPIPE, which is what a paged read
# reports: `cat SKILL.md | head -20` exits 141 once head closes the pipe, having printed the part
# of the file the agent actually saw. Treating that as a failed load discards a real skill body.
_SUCCESS_EXIT_CODES = {None, 0, "0", 141, "141"}
_ACKNOWLEDGEMENT = re.compile(
    # A load acknowledgement is not the skill body. Harnesses word this either way round
    # ("Launching skill: x", or Gemini CLI's "Skill activated. Resources loaded from ..."),
    # and mistaking one for the body would have the judge score a status line as instructions.
    # The optional group is the skill name some harnesses interpose, e.g. the Strands
    # AgentSkills plugin's "Skill 'x' activated (no instructions available).".
    # Anchored to a single line, and deliberately not with `$`/`.*`: `[^\n]*\Z` is what keeps a
    # body whose first line is an acknowledgement. `\s*` matches newlines, so a trailing `\s*.*$`
    # reaches past the first line and discards a short real body along with its status line.
    r"^(?:(?:Launching|Loading|Activating|Loaded|Activated)\s+skill"
    r"|skill\s+(?:'[^']*'|\"[^\"]*\"|[\w.-]+)?[ \t]*(?:activated|loaded|launched))"
    r"\b(?:[ \t]*[:.]?[^\n]*)?\Z",
    re.IGNORECASE,
)
_LOAD_ERROR = re.compile(
    # A refused load is not the skill body either, and it does not arrive marked as an error.
    # The Strands AgentSkills plugin returns these as plain strings from an `@tool` function,
    # and `@tool` reports a plain string return as `status="success"`, so `_result_failed` sees
    # nothing wrong and the judge would be handed "Skill 'x' not found. Available skills: ..."
    # as the instructions the agent was supposed to follow.
    # Anchored to a single line, like `_ACKNOWLEDGEMENT` and for the same reason: a refusal-shaped
    # first line must not carry away the real body that follows it. Reporting that body's skill as
    # a failed load is the worse half of the bug, since the harness did activate it.
    r"^(?:"
    r"error\s*:\s*skill_name\s+is\s+required"
    r"|skill\s+(?:'[^']*'|\"[^\"]*\"|[\w.-]+)\s+(?:was\s+|is\s+)?not\s+found"
    r"|(?:unknown|unrecognized)\s+skill\b"
    r"|no\s+such\s+skill\b"
    r")[ \t]*[:.]?[^\n]*\Z",
    re.IGNORECASE,
)
