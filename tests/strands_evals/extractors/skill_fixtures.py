"""Golden fixtures for the skill parsing helpers.

Cross-harness raw-message-list shapes plus near-miss cases. Session-object
fixtures are built in the test module (they need real trace types); these are
the harness-native raw shapes the helpers must also handle.
"""

PDF_DESCRIPTION = "Use this skill when the task requires reading or extracting text from PDF files."

# The available-skills block a harness injects into the system prompt.
AVAILABLE_BLOCK = f"""You are a helpful agent.

<available_skills>
<skill>
<name>pdf-processing</name>
<description>{PDF_DESCRIPTION}</description>
<location>/skills/pdf-processing/SKILL.md</location>
</skill>
<skill>
<name>spreadsheet-analysis</name>
<description>Analyze, edit, or generate spreadsheets.</description>
<location>/skills/spreadsheet-analysis/SKILL.md</location>
</skill>
</available_skills>
"""

SKILL_BODY = "# PDF Processing Skill\n\n1. Identify the PDF file path.\n2. Extract the text.\n3. Summarize it."


def _tool_use(tool_use_id, name, inp):
    return {"role": "assistant", "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": inp}}]}


def _tool_result(tool_use_id, text):
    return {"role": "user", "content": [{"toolResult": {"toolUseId": tool_use_id, "content": [{"text": text}]}}]}


# Strands native in-memory message shape: reserved `skills` tool, arg `skill_name`,
# body returned in the following user message's toolResult.
STRANDS_MESSAGES = [
    {"role": "system", "content": AVAILABLE_BLOCK},
    {"role": "user", "content": [{"text": "Extract text from report.pdf"}]},
    {
        "role": "assistant",
        "content": [
            {"text": "I'll use the pdf-processing skill."},
            {"toolUse": {"toolUseId": "tu-1", "name": "skills", "input": {"skill_name": "pdf-processing"}}},
        ],
    },
    _tool_result("tu-1", SKILL_BODY),
    {"role": "assistant", "content": [{"text": "Done."}]},
]

# Claude Code shape: reserved `Skill` tool, arg `skill`. (Body arrives in a
# following user message rather than the launch acknowledgement.)
CLAUDE_CODE_MESSAGES = [
    {"role": "system", "content": AVAILABLE_BLOCK},
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "cc-1",
                    "name": "Skill",
                    "input": {"skill": "pdf-processing", "args": "report.pdf"},
                }
            ],
        },
    },
    {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "cc-1",
                    "content": "Launching skill: pdf-processing",
                }
            ],
        },
    },
    {
        "role": "user",
        "content": (f"Base directory for this skill: /skills/pdf-processing\n\n{SKILL_BODY}"),
    },
]

# Gemini CLI shape: reserved `activate_skill` tool, arg `name`.
GEMINI_MESSAGES = [
    {"role": "system", "content": AVAILABLE_BLOCK},
    _tool_use("g-1", "activate_skill", {"name": "pdf-processing"}),
    _tool_result("g-1", f"<activated_skill><instructions>{SKILL_BODY}</instructions></activated_skill>"),
]

CODEX_AVAILABLE_PROMPT = """# Skills
### Available skills
- pdf-processing: Use this skill for PDFs. (file: /skills/pdf-processing/SKILL.md)
### How to use skills
- Read the source before applying a skill.
"""

CODEX_MESSAGES = [
    {"role": "system", "content": CODEX_AVAILABLE_PROMPT},
    {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "/bin/bash -lc \"sed -n '1,220p' /skills/pdf-processing/SKILL.md\"",
            "status": "completed",
            "exit_code": 0,
            "aggregated_output": SKILL_BODY,
        },
    },
]

OPENAI_AGENTS_MESSAGES = [
    {"role": "system", "content": CODEX_AVAILABLE_PROMPT},
    _tool_use("oa-1", "load_skill", {"skill_name": "pdf-processing"}),
    _tool_result(
        "oa-1",
        '{"status": "loaded", "skill_name": "pdf-processing", "path": ".agents/pdf-processing"}',
    ),
    _tool_use("oa-2", "read_file", {"path": ".agents/pdf-processing/SKILL.md"}),
    _tool_result("oa-2", SKILL_BODY),
]

FAILED_LOAD_MESSAGES = [
    _tool_use("fail-1", "skills", {"skill_name": "pdf-processing"}),
    {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "fail-1",
                    "status": "error",
                    "content": [{"text": "skill not found"}],
                }
            }
        ],
    },
]

BODY_MISSING_MESSAGES = [
    _tool_use("missing-1", "load_skill", {"skill_name": "pdf-processing"}),
    _tool_result(
        "missing-1",
        '{"status": "loaded", "skill_name": "pdf-processing", "path": ".agents/pdf-processing"}',
    ),
]

DUPLICATE_LOAD_MESSAGES = [
    _tool_use("dup-1", "skills", {"skill_name": "pdf-processing"}),
    _tool_result("dup-1", SKILL_BODY),
    _tool_use("dup-2", "skills", {"skill_name": "pdf-processing"}),
    _tool_result("dup-2", SKILL_BODY),
]

GEMINI_STREAM_MESSAGES = [
    {"tool_name": "activate_skill", "parameters": {"name": "pdf-processing"}},
    {
        "type": "tool_result",
        "status": "success",
        "llmContent": f"<activated_skill><instructions>{SKILL_BODY}</instructions></activated_skill>",
    },
]

GOOGLE_ADK_MESSAGES = [
    {"name": "myapp_load_skill", "args": {"skill_name": "pdf-processing"}, "id": "adk-1"},
    {
        "type": "tool_response",
        "id": "adk-1",
        "response": {"skill_name": "pdf-processing", "instructions": SKILL_BODY},
    },
]

OPENHANDS_MESSAGES = [
    {"kind": "InvokeSkillAction", "name": "pdf-processing"},
    {
        "kind": "InvokeSkillObservation",
        "skill_name": "pdf-processing",
        "is_error": False,
        "content": [{"type": "text", "text": SKILL_BODY}],
    },
]

# Near-miss: a skill NAME mentioned in prose / a non-skill tool call. Neither is an invocation.
NEAR_MISS_MESSAGES = [
    {"role": "system", "content": AVAILABLE_BLOCK},
    {
        "role": "assistant",
        "content": [{"text": "I could use the pdf-processing skill, but let me just read the file."}],
    },
    _tool_use("nm-1", "file_read", {"path": "/skills/pdf-processing/SKILL.md"}),
]

# No available block, no invocation.
EMPTY_MESSAGES = [
    {"role": "user", "content": [{"text": "hello"}]},
    {"role": "assistant", "content": [{"text": "hi"}]},
]

# Multiple invocations in one run (two distinct skills).
MULTI_INVOKE_MESSAGES = [
    {"role": "system", "content": AVAILABLE_BLOCK},
    _tool_use("m-1", "skills", {"skill_name": "pdf-processing"}),
    _tool_result("m-1", SKILL_BODY),
    _tool_use("m-2", "skills", {"skill_name": "spreadsheet-analysis"}),
    _tool_result("m-2", "# Spreadsheet Analysis\n1. Inspect."),
]
