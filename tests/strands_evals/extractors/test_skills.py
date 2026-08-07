"""Unit tests for the skill parsing helpers (parse_available_skills, extract_selected_skills)."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from strands_evals.extractors import (
    AvailableSkill,
    InvokedSkill,
    extract_selected_skills,
    extract_skill_load_events,
    parse_available_skills,
)
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    Session,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolCallContent,
    ToolExecutionSpan,
    ToolResult,
    ToolResultContent,
    Trace,
    UserMessage,
)

from . import skill_fixtures as fx


def _loaded(name: str, body: str | None) -> InvokedSkill:
    """A skill the harness loaded successfully, for comparing whole extraction results."""
    return InvokedSkill(name, body)


def _failed(name: str, error: str | None = None) -> InvokedSkill:
    """A skill the agent asked for and the harness refused, with what the harness said."""
    return InvokedSkill(name, None, status="failed", error=error)


# ---- raw message-list path --------------------------------------------------


def test_available_skills_from_strands_list():
    skills = parse_available_skills(fx.STRANDS_MESSAGES)
    assert skills == [
        AvailableSkill("pdf-processing", fx.PDF_DESCRIPTION),
        AvailableSkill("spreadsheet-analysis", "Analyze, edit, or generate spreadsheets."),
    ]


def test_available_skills_unescapes_xml_entities():
    prompt = (
        "<available_skills><skill><name>research&amp;review</name>"
        "<description>Compare A &lt; B &amp; report.</description></skill></available_skills>"
    )

    assert parse_available_skills(prompt) == [
        AvailableSkill("research&review", "Compare A < B & report."),
    ]


def test_selected_skills_from_strands_list_with_body():
    invoked = extract_selected_skills(fx.STRANDS_MESSAGES)
    assert len(invoked) == 1
    assert invoked[0].name == "pdf-processing"
    assert invoked[0].body is not None and "PDF Processing Skill" in invoked[0].body


@pytest.mark.parametrize(
    "messages,expected_name,expect_body_substr",
    [
        (fx.STRANDS_MESSAGES, "pdf-processing", "PDF Processing Skill"),
        (fx.CLAUDE_CODE_MESSAGES, "pdf-processing", "PDF Processing Skill"),
        (fx.CODEX_MESSAGES, "pdf-processing", "PDF Processing Skill"),
        (fx.OPENAI_AGENTS_MESSAGES, "pdf-processing", "PDF Processing Skill"),
        (fx.GEMINI_MESSAGES, "pdf-processing", "<instructions>"),
        (fx.GEMINI_STREAM_MESSAGES, "pdf-processing", "<instructions>"),
        (fx.GOOGLE_ADK_MESSAGES, "pdf-processing", "PDF Processing Skill"),
        (fx.OPENHANDS_MESSAGES, "pdf-processing", "PDF Processing Skill"),
    ],
)
def test_selected_skills_cross_harness(messages, expected_name, expect_body_substr):
    invoked = extract_selected_skills(messages)
    assert len(invoked) == 1
    assert invoked[0].name == expected_name
    assert invoked[0].body is not None and expect_body_substr in invoked[0].body


def test_near_miss_is_not_an_invocation():
    # A skill name mentioned in prose is not an invocation, and here the file_read
    # of the SKILL.md path is rejected specifically because its result carried NO
    # skill body. A SKILL.md read whose result DOES return a body is a valid
    # filesystem-skill load (Codex / OpenAI Agents); see
    # test_session_skill_file_read_with_body. Do not remove the file-read branch.
    assert extract_selected_skills(fx.NEAR_MISS_MESSAGES) == []
    # The available block is still recoverable from the same messages.
    assert [s.name for s in parse_available_skills(fx.NEAR_MISS_MESSAGES)] == [
        "pdf-processing",
        "spreadsheet-analysis",
    ]


def test_empty_messages():
    assert parse_available_skills(fx.EMPTY_MESSAGES) == []
    assert extract_selected_skills(fx.EMPTY_MESSAGES) == []


def test_multiple_invocations_in_order():
    invoked = extract_selected_skills(fx.MULTI_INVOKE_MESSAGES)
    assert [i.name for i in invoked] == ["pdf-processing", "spreadsheet-analysis"]
    assert all(i.body for i in invoked)


def test_available_skills_from_markdown_section():
    assert parse_available_skills(fx.CODEX_MESSAGES) == [AvailableSkill("pdf-processing", "Use this skill for PDFs.")]


def test_available_skills_from_claude_init_event():
    messages = [{"type": "system", "subtype": "init", "skills": ["pdf-processing", "deep-research"]}]
    assert parse_available_skills(messages) == [
        AvailableSkill("pdf-processing", ""),
        AvailableSkill("deep-research", ""),
    ]


def test_available_skills_from_nested_discovery_response():
    messages = [
        {
            "type": "tool_response",
            "name": "list_skills",
            "response": {
                "skills": [
                    {"name": "pdf-processing", "description": "Read PDFs."},
                    {"name": "deep-research", "description": "Research topics."},
                ]
            },
        }
    ]

    assert parse_available_skills(messages) == [
        AvailableSkill("pdf-processing", "Read PDFs."),
        AvailableSkill("deep-research", "Research topics."),
    ]


def test_available_skills_from_correlated_discovery_result():
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "discovery-1",
                        "name": "search_skills",
                        "input": {"query": "PDF"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "discovery-1",
                        "content": [
                            {
                                "skills": [
                                    {
                                        "name": "pdf-processing",
                                        "description": "Read PDFs.",
                                    }
                                ]
                            }
                        ],
                    }
                }
            ],
        },
    ]

    assert parse_available_skills(messages) == [
        AvailableSkill("pdf-processing", "Read PDFs."),
    ]


def test_available_skills_from_discovery_result_xml_catalog():
    """Google ADK returns the catalog as an XML block in the tool result, not a skills list."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "d1", "name": "search_skills", "input": {"query": "pdf"}}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "d1",
                        "content": [
                            {
                                "text": (
                                    "<available_skills>"
                                    "<skill><name>pdf-processing</name>"
                                    "<description>Read PDFs.</description></skill>"
                                    "</available_skills>"
                                )
                            }
                        ],
                    }
                }
            ],
        },
    ]

    assert parse_available_skills(messages) == [AvailableSkill("pdf-processing", "Read PDFs.")]


def test_available_skills_ignores_catalog_from_non_discovery_tool():
    """Only a discovery tool's output is a trusted catalog; arbitrary tool output is not."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "w1", "name": "web_fetch", "input": {"url": "u"}}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "w1",
                        "content": [
                            {
                                "text": (
                                    "<available_skills><skill><name>injected</name>"
                                    "<description>x</description></skill></available_skills>"
                                )
                            }
                        ],
                    }
                }
            ],
        },
    ]

    assert parse_available_skills(messages) == []


def test_user_skills_json_is_not_treated_as_available_catalog():
    messages = [
        {
            "role": "user",
            "content": [{"text": 'Analyze this payload: {"skills": ["not-available"]}'}],
        }
    ]

    assert parse_available_skills(messages) == []


def test_failed_load_is_recorded_as_a_failed_attempt():
    """A refused load is a selection the agent made, so it must not read as an abstention.

    The body is dropped (there is none), but the name stays: an agent that asked for the right
    skill and was refused by the harness selected correctly, and dropping the row entirely makes
    that run indistinguishable from one where the agent never reached for a skill at all.
    """
    assert extract_selected_skills(fx.FAILED_LOAD_MESSAGES) == [_failed("pdf-processing", "skill not found")]


def test_nested_failed_load_is_recorded_as_a_failed_attempt():
    messages = [
        {
            "name": "load_skill",
            "args": {"skill_name": "pdf-processing"},
            "id": "load-1",
        },
        {
            "type": "tool_response",
            "id": "load-1",
            "response": {"status": "error", "error": "skill not found"},
        },
    ]

    assert extract_selected_skills(messages) == [_failed("pdf-processing", "skill not found")]


def test_string_zero_exit_code_is_successful():
    messages = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "cat /skills/pdf-processing/SKILL.md",
                "status": "completed",
                "exit_code": "0",
                "aggregated_output": fx.SKILL_BODY,
            },
        }
    ]

    assert extract_selected_skills(messages) == [_loaded("pdf-processing", fx.SKILL_BODY)]


def test_successful_load_without_body_preserves_invocation():
    assert extract_selected_skills(fx.BODY_MISSING_MESSAGES) == [_loaded("pdf-processing", None)]


def test_duplicate_loads_are_coalesced():
    invoked = extract_selected_skills(fx.DUPLICATE_LOAD_MESSAGES)
    assert invoked == [_loaded("pdf-processing", fx.SKILL_BODY)]


def test_selected_skills_from_typed_messages():
    messages = [
        AssistantMessage(
            content=[
                TextContent(text="Loading the PDF skill"),
                ToolCallContent(
                    name="skills",
                    arguments={"skill_name": "pdf-processing"},
                    tool_call_id="typed-1",
                ),
            ]
        ),
        UserMessage(
            content=[
                ToolResultContent(
                    content=fx.SKILL_BODY,
                    tool_call_id="typed-1",
                )
            ]
        ),
    ]

    invoked = extract_selected_skills(messages)

    assert invoked == [_loaded("pdf-processing", fx.SKILL_BODY)]


def test_unsupported_trajectory_type_returns_empty():
    assert parse_available_skills(None) == []
    assert extract_selected_skills(None) == []
    assert parse_available_skills("not a trajectory") == []


def test_parse_available_from_bare_system_prompt_string():
    # Some session mappers store the system prompt separately from the message list;
    # parse_available_skills accepts the bare prompt string too.
    skills = parse_available_skills(fx.AVAILABLE_BLOCK)
    assert [s.name for s in skills] == ["pdf-processing", "spreadsheet-analysis"]


# ---- Session path -----------------------------------------------------------


def _span_info() -> SpanInfo:
    return SpanInfo(session_id="s", start_time=datetime(2026, 7, 14), end_time=datetime(2026, 7, 14))


def _session(spans) -> Session:
    return Session(session_id="s", traces=[Trace(trace_id="t", session_id="s", spans=spans)])


def test_session_available_from_system_prompt():
    agent_span = AgentInvocationSpan(
        span_info=_span_info(),
        user_prompt="do pdf",
        agent_response="done",
        available_tools=[],
        system_prompt=fx.AVAILABLE_BLOCK,
    )
    skills = parse_available_skills(_session([agent_span]))
    assert [s.name for s in skills] == ["pdf-processing", "spreadsheet-analysis"]


def test_session_selected_from_tool_execution_span():
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(name="skills", arguments={"skill_name": "pdf-processing"}, tool_call_id="tu-1"),
        tool_result=ToolResult(content=fx.SKILL_BODY),
    )
    invoked = extract_selected_skills(_session([tool_span]))
    assert len(invoked) == 1
    assert invoked[0].name == "pdf-processing"
    assert "PDF Processing Skill" in invoked[0].body


def test_session_non_skill_tool_ignored():
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}, tool_call_id="c-1"),
        tool_result=ToolResult(content="4"),
    )
    assert extract_selected_skills(_session([tool_span])) == []


def test_session_failed_skill_load_is_recorded_as_a_failed_attempt():
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(name="skills", arguments={"skill_name": "pdf-processing"}, tool_call_id="tu-1"),
        tool_result=ToolResult(content="skill not found", error="error"),
    )
    assert extract_selected_skills(_session([tool_span])) == [_failed("pdf-processing", "error")]


def test_session_failed_read_of_a_skill_file_is_not_an_invocation():
    """A read of a `SKILL.md` that errored recovered no name and no body, so there is nothing
    to report: unlike a reserved skill tool, the path read carries no declared skill name."""
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(name="read_file", arguments={"path": "/skills/pdf/SKILL.md"}, tool_call_id="r-1"),
        tool_result=ToolResult(content="No such file", error="error"),
    )
    assert extract_selected_skills(_session([tool_span])) == []


def test_session_skill_file_read_with_body():
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(
            name="read_file",
            arguments={"path": "/skills/pdf-processing/SKILL.md"},
            tool_call_id="read-1",
        ),
        tool_result=ToolResult(content=fx.SKILL_BODY),
    )
    assert extract_selected_skills(_session([tool_span])) == [_loaded("pdf-processing", fx.SKILL_BODY)]


def test_session_skill_file_read_uses_frontmatter_name():
    body = "---\nname: canonical-skill\ndescription: Test skill.\n---\n# Steps\n1. Test."
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(
            name="read_file",
            arguments={"path": "/skills/directory-alias/SKILL.md"},
            tool_call_id="read-1",
        ),
        tool_result=ToolResult(content=body),
    )

    assert extract_selected_skills(_session([tool_span])) == [_loaded("canonical-skill", body)]


def test_opaque_load_and_alias_path_read_are_coalesced():
    body = "---\nname: canonical-skill\ndescription: Test skill.\n---\n# Steps\n1. Test."
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "load-1",
                        "name": "load_skill",
                        "input": {"skill_name": "canonical-skill"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "load-1", "content": [{"text": "Loaded skill"}]}}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "read-1",
                        "name": "read_file",
                        "input": {"path": "/skills/directory-alias/SKILL.md"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "read-1", "content": [{"text": body}]}}],
        },
    ]

    assert extract_selected_skills(messages) == [_loaded("canonical-skill", body)]


def test_command_execution_skill_read_uses_frontmatter_name():
    body = "---\nname: canonical-skill\ndescription: Test skill.\n---\n# Steps\n1. Test."
    messages = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "cat /skills/directory-alias/SKILL.md",
                "status": "completed",
                "exit_code": 0,
                "aggregated_output": body,
            },
        }
    ]

    assert extract_selected_skills(messages) == [_loaded("canonical-skill", body)]


def _shell_command(command: str, output: str = "col1,col2\n1,2", exit_code: int = 0):
    return [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": command,
                "status": "completed",
                "exit_code": exit_code,
                "aggregated_output": output,
            },
        }
    ]


@pytest.mark.parametrize(
    "command",
    [
        "cat draft.md > /skills/my-new-skill/SKILL.md",  # writes a skill, does not load one
        "echo '# Steps' >> /skills/my-new-skill/SKILL.md",
        "sed -i 's/a/b/' /skills/pdf-processing/SKILL.md",  # edits in place
        "sed --in-place 's/a/b/' /skills/pdf-processing/SKILL.md",
        "echo '# Steps' | tee /skills/my-new-skill/SKILL.md",
        "cat data.csv; ls -l /skills/pdf-processing/SKILL.md",  # verb and path, different commands
        "grep -n Extract /skills/pdf-processing/SKILL.md",  # not a read of the whole file
    ],
)
def test_shell_command_that_does_not_read_a_skill_is_not_an_invocation(command):
    """The read verb has to own the path, not merely appear somewhere in the same line.

    Searching for a verb and a path independently makes writes and unrelated work look like
    skill loads, and the phantom body is whatever the command happened to print.
    """
    assert extract_selected_skills(_shell_command(command)) == []


@pytest.mark.parametrize(
    "command",
    [
        "cat /skills/pdf-processing/SKILL.md",
        "sed -n '1,220p' /skills/pdf-processing/SKILL.md",
        "/bin/bash -lc \"sed -n '1,220p' /skills/pdf-processing/SKILL.md\"",  # harness wrapper
        "sudo cat /skills/pdf-processing/SKILL.md",
        "cat /skills/pdf-processing/SKILL.md | head -20",  # paged
        "cd /tmp && cat /skills/pdf-processing/SKILL.md",  # read in a later segment
        "cat draft.md > /tmp/out.md; cat /skills/pdf-processing/SKILL.md",  # write then read
    ],
)
def test_shell_read_of_a_skill_is_an_invocation(command):
    body = "# PDF Processing\n1. Identify the path.\n2. Extract."

    assert extract_selected_skills(_shell_command(command, output=body)) == [_loaded("pdf-processing", body)]


@pytest.mark.parametrize(
    "command,expected",
    [
        ("cat data.csv > /skills/my-new-skill/SKILL.md", []),
        ("cat /skills/pdf-processing/SKILL.md", [_loaded("pdf-processing", "# PDF Processing\n1. Extract.")]),
    ],
)
def test_shell_tool_read_uses_the_same_rule_as_command_execution(command, expected):
    """A `bash` tool call and a Codex `command_execution` event are the same shell command."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "b1", "name": "bash", "input": {"command": command}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "b1", "content": [{"text": "# PDF Processing\n1. Extract."}]}}],
        },
    ]

    assert extract_selected_skills(messages) == expected


def test_sigpipe_exit_code_does_not_discard_a_read_body():
    """`cat SKILL.md | head -20` exits 141 once head closes the pipe, having printed the body."""
    body = "# PDF Processing\n1. Identify the path."

    invoked = extract_selected_skills(
        _shell_command("cat /skills/pdf-processing/SKILL.md | head -20", output=body, exit_code=141)
    )

    assert invoked == [_loaded("pdf-processing", body)]


def test_failing_read_is_still_discarded():
    """Only SIGPIPE is tolerated; a read that actually failed carries no body."""
    assert extract_selected_skills(_shell_command("cat /skills/pdf/SKILL.md", output="No such file", exit_code=1)) == []


def test_malformed_frontmatter_body_falls_back_to_path_name():
    # A SKILL.md whose frontmatter is not parseable YAML must not abort the whole
    # extraction; the name degrades to the directory alias and the body is kept.
    body = "---\nname: [unclosed\ndescription: broken\n---\n# Steps\n1. Test."
    messages = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "cat /skills/directory-alias/SKILL.md",
                "status": "completed",
                "exit_code": 0,
                "aggregated_output": body,
            },
        }
    ]

    assert extract_selected_skills(messages) == [_loaded("directory-alias", body)]


def test_unkeyed_results_are_not_reused_across_skill_calls():
    messages = [
        {"tool_name": "activate_skill", "parameters": {"name": "first"}},
        {"type": "tool_result", "status": "success", "llmContent": "first body"},
        {"tool_name": "activate_skill", "parameters": {"name": "second"}},
        {"type": "tool_result", "status": "success", "llmContent": "second body"},
    ]

    assert extract_selected_skills(messages) == [
        _loaded("first", "first body"),
        _loaded("second", "second body"),
    ]


def test_session_available_absent_when_no_system_prompt():
    # Mapped sessions may drop the system prompt; then the block is not recoverable
    # from the Session (would fall back to a raw message list in practice).
    agent_span = AgentInvocationSpan(
        span_info=_span_info(),
        user_prompt="do pdf",
        agent_response="done",
        available_tools=[],
        system_prompt=None,
    )
    assert parse_available_skills(_session([agent_span])) == []


def test_google_adk_function_call_shape():
    """Google ADK emits Gemini content parts and nests its payload under response/result."""
    messages = [
        {
            "role": "model",
            "content": [{"functionCall": {"id": "c1", "name": "list_skills", "args": {}}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "functionResponse": {
                        "id": "c1",
                        "name": "list_skills",
                        "response": {
                            "result": (
                                "<available_skills><skill><name>pdf-processing</name>"
                                "<description>Read PDFs.</description></skill></available_skills>"
                            )
                        },
                    }
                }
            ],
        },
        {
            "role": "model",
            "content": [{"functionCall": {"id": "c2", "name": "load_skill", "args": {"skill_name": "pdf-processing"}}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "functionResponse": {
                        "id": "c2",
                        "name": "load_skill",
                        "response": {"skill_name": "pdf-processing", "instructions": "## Phase 1\nRun pdfinfo."},
                    }
                }
            ],
        },
    ]

    assert parse_available_skills(messages) == [AvailableSkill("pdf-processing", "Read PDFs.")]
    invoked = extract_selected_skills(messages)
    assert [s.name for s in invoked] == ["pdf-processing"]
    assert invoked[0].body == "## Phase 1\nRun pdfinfo."


def test_load_acknowledgement_is_not_treated_as_a_body():
    """Gemini CLI's displayed output is a status line; scoring it as instructions would be wrong."""
    messages = [
        {"tool_name": "activate_skill", "parameters": {"name": "pdf-processing"}, "id": "g1"},
        {
            "type": "tool_result",
            "id": "g1",
            "output": "Skill activated. Resources loaded from pdf-processing/",
        },
    ]

    invoked = extract_selected_skills(messages)

    assert [s.name for s in invoked] == ["pdf-processing"]
    assert invoked[0].body is None


# The three strings the Strands AgentSkills plugin returns instead of a skill body. They arrive
# marked successful, because `@tool` reports any plain string return as `status="success"`.
_AGENT_SKILLS_NON_BODIES = [
    "Skill 'pdf-processing' not found. Available skills: spreadsheet-analysis, docx-editing",
    "Error: skill_name is required. Available skills: pdf-processing, spreadsheet-analysis",
    "Skill 'pdf-processing' activated (no instructions available).",
]


@pytest.mark.parametrize("result_text", _AGENT_SKILLS_NON_BODIES)
def test_agent_skills_status_string_is_not_a_body(result_text):
    """A refused or empty load carries no instructions, so the judge must not be handed one."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "skills", "input": {"skill_name": "pdf-processing"}}}],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": result_text}]}}]},
    ]

    invoked = extract_selected_skills(messages)

    assert [s.name for s in invoked] == ["pdf-processing"]
    assert invoked[0].body is None


@pytest.mark.parametrize("result_text", _AGENT_SKILLS_NON_BODIES)
def test_agent_skills_status_string_is_not_a_body_on_the_session_path(result_text):
    """Same on the Session path: the plugin's string lands in `content` with `error` unset."""
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(name="skills", arguments={"skill_name": "pdf-processing"}, tool_call_id="tu-1"),
        tool_result=ToolResult(content=result_text),
    )

    invoked = extract_selected_skills(_session([tool_span]))

    assert [s.name for s in invoked] == ["pdf-processing"]
    assert invoked[0].body is None


# The two plugin strings above that mean no skill was loaded. The third
# ("activated (no instructions available)") is a real load of a skill with no body, so it stays
# `loaded`: the agent did receive the skill, there was just nothing prescriptive in it.
_AGENT_SKILLS_REFUSALS = _AGENT_SKILLS_NON_BODIES[:2]


@pytest.mark.parametrize("result_text", _AGENT_SKILLS_REFUSALS)
def test_agent_skills_refusal_is_recorded_as_a_failed_load(result_text):
    """A mistyped lookup key is a refused load, not a load whose body went uncaptured.

    `@tool` marks the plugin's plain-string return `status="success"`, so the text is the only
    signal. Without reading it, requesting `pdf-procesing` for a registered `pdf-processing`
    reports as a successful invocation and the adherence judge blames the agent for not
    following instructions it never received.
    """
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "skills", "input": {"skill_name": "pdf-procesing"}}}],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": result_text}]}}]},
    ]

    invoked = extract_selected_skills(messages)

    assert invoked == [_failed("pdf-procesing", result_text)]


@pytest.mark.parametrize("result_text", _AGENT_SKILLS_REFUSALS)
def test_agent_skills_refusal_is_recorded_as_a_failed_load_on_the_session_path(result_text):
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(name="skills", arguments={"skill_name": "pdf-procesing"}, tool_call_id="tu-1"),
        tool_result=ToolResult(content=result_text),
    )

    invoked = extract_selected_skills(_session([tool_span]))

    assert invoked == [_failed("pdf-procesing", result_text)]


def test_refusal_message_distinguishes_a_bad_name_from_an_empty_mount():
    """Both runs fail the same way; only the harness's message says what to fix.

    A misspelled skill name is the agent's mistake, an empty catalog is the harness's. Without
    the message both read as "the load failed" and whoever reads the result cannot tell which.
    """

    def refused(text: str) -> list[dict]:
        return [
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "t1", "name": "skills", "input": {"skill_name": "pdf-procesing"}}}
                ],
            },
            {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": text}]}}]},
        ]

    typo = "Skill 'pdf-procesing' not found. Available skills: pdf-processing"
    empty = "Skill 'pdf-procesing' not found. Available skills: (none)"

    assert extract_selected_skills(refused(typo))[0].error == typo
    assert extract_selected_skills(refused(empty))[0].error == empty


def test_a_loaded_skill_carries_no_refusal_message():
    assert extract_selected_skills(fx.STRANDS_MESSAGES)[0].error is None


def test_a_skill_that_activated_with_no_instructions_still_counts_as_loaded():
    """An empty skill is not a refusal: the agent got what it asked for, body and all."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "skills", "input": {"skill_name": "pdf-processing"}}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "content": [{"text": "Skill 'pdf-processing' activated (no instructions available)."}],
                    }
                }
            ],
        },
    ]

    assert extract_selected_skills(messages) == [_loaded("pdf-processing", None)]


def test_body_mentioning_a_missing_file_is_kept():
    """The load-error filter matches a whole status line, not the words wherever they appear."""
    body = "# PDF Processing\n\n1. If the skill file is not found, stop.\n2. Extract the text."
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "skills", "input": {"skill_name": "pdf-processing"}}}],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": body}]}}]},
    ]

    assert extract_selected_skills(messages)[0].body == body


def test_skill_read_twice_keeps_the_fullest_body():
    """Repeated reads of one skill collapse, keeping the read that carried the whole file."""
    body = "---\nname: chart-builder\ndescription: Charts.\n---\n\n## Phase 1\nBuild the chart.\n"
    messages = [
        {
            "type": "command_execution",
            "command": "sed -n '1,3p' /skills/chart_builder/SKILL.md",
            "aggregated_output": "---\nname: chart-builder\ndescription: Charts.\n",
        },
        {
            "type": "command_execution",
            "command": "cat /skills/chart-builder/SKILL.md",
            "aggregated_output": body,
        },
    ]

    invoked = extract_selected_skills(messages)

    assert len(invoked) == 1
    assert invoked[0].name == "chart-builder"
    assert "## Phase 1" in (invoked[0].body or "")


def test_body_prefixed_by_an_acknowledgement_is_kept():
    """A status line ahead of the instructions must not discard the instructions with it."""
    body = "# PDF Processing\n\n1. Identify the PDF path.\n2. Extract the text."
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "skills", "input": {"skill_name": "pdf-processing"}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": f"Skill activated.\n\n{body}"}]}}],
        },
    ]

    invoked = extract_selected_skills(messages)

    assert [s.name for s in invoked] == ["pdf-processing"]
    assert "1. Identify the PDF path." in (invoked[0].body or "")


def test_skill_names_differing_only_by_a_dot_stay_separate():
    """`.` is legal in a skill name, so `data.clean` and `data-clean` are two skills."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "a", "name": "skills", "input": {"skill_name": "data.clean"}}}],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "a", "content": [{"text": "# Dotted\n1. a"}]}}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "b", "name": "skills", "input": {"skill_name": "data-clean"}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "b", "content": [{"text": "# Hyphenated\n1. b\n2. c"}]}}],
        },
    ]

    assert [s.name for s in extract_selected_skills(messages)] == ["data.clean", "data-clean"]


def test_a_longer_read_of_the_same_skill_only_wins_if_it_contains_what_was_kept():
    """The containment rule, on the shape that actually exercises it: two real read verbs.

    A paged read recovers part of a skill, and a later read of the same skill returns unrelated
    but longer output. Length alone would let the second displace the first, so the judge would be
    handed stray stdout as the skill's instructions. The kept body has to be a subset of the
    challenger for it to win.
    """
    partial = "---\nname: pdf-processing\n---\n# Real\n1. first step"
    messages = [
        {
            "type": "command_execution",
            "command": "sed -n '1,5p' /skills/pdf-processing/SKILL.md",
            "exit_code": 0,
            "aggregated_output": partial,
        },
        {
            # A read verb, the same skill path, and longer output that does NOT contain `partial`.
            "type": "command_execution",
            "command": "cat /skills/pdf-processing/SKILL.md",
            "exit_code": 0,
            "aggregated_output": "unrelated stdout that happens to be much longer " * 4,
        },
    ]

    invoked = extract_selected_skills(messages)

    assert [s.name for s in invoked] == ["pdf-processing"]
    assert invoked[0].body == partial


def test_longer_unrelated_output_does_not_displace_a_recovered_body():
    """Only a superset of what was already recovered wins, so stray stdout cannot take over."""
    real_body = "---\nname: pdf-processing\n---\n# Real\n1. step"
    messages = [
        {
            "type": "command_execution",
            "command": "cat /skills/pdf-processing/SKILL.md",
            "exit_code": 0,
            "aggregated_output": real_body,
        },
        {
            "type": "command_execution",
            "command": "cat report.csv; ls -l /skills/pdf-processing/SKILL.md",
            "exit_code": 0,
            "aggregated_output": "col1,col2\n" + "x,y\n" * 40,
        },
    ]

    invoked = extract_selected_skills(messages)

    assert len(invoked) == 1
    assert invoked[0].body == real_body


def _claude_launch(call_id: str, skill: str) -> list[dict]:
    """A Claude Code `Skill` call and the launch acknowledgement that carries no body."""
    return [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": call_id, "name": "Skill", "input": {"skill": skill}}],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": call_id, "content": f"Launching skill: {skill}"}],
            },
        },
    ]


def _claude_injected_body(skill: str, body: str) -> dict:
    return {"role": "user", "content": f"Base directory for this skill: /skills/{skill}\n\n{body}"}


def test_parallel_claude_skill_calls_each_get_their_own_body():
    """Claude Code can launch several skills in one turn; each must get its own instructions.

    Taking the first injected body after the call index gives every skill the first skill's
    steps, and the adherence judge then scores the agent against instructions it never received.
    """
    messages = [
        *_claude_launch("cc-1", "pdf-processing"),
        *_claude_launch("cc-2", "spreadsheet-analysis"),
        _claude_injected_body("spreadsheet-analysis", "# Spreadsheet\n1. Open the sheet."),
        _claude_injected_body("pdf-processing", "# PDF\n1. Extract the text."),
    ]

    invoked = extract_selected_skills(messages)

    assert [s.name for s in invoked] == ["pdf-processing", "spreadsheet-analysis"]
    assert "1. Extract the text." in (invoked[0].body or "")
    assert "1. Open the sheet." in (invoked[1].body or "")


def test_single_claude_body_is_used_even_when_the_directory_is_an_alias():
    """One launch and one injected body pair up, since the directory can differ from the name."""
    messages = [
        *_claude_launch("cc-1", "pdf-processing"),
        _claude_injected_body("directory-alias", "# PDF\n1. Extract the text."),
    ]

    invoked = extract_selected_skills(messages)

    assert [s.name for s in invoked] == ["pdf-processing"]
    assert "1. Extract the text." in (invoked[0].body or "")


def _load_attempt(call_id: str, skill: str, result: dict) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": call_id, "name": "load_skill", "input": {"skill_name": skill}}}],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": call_id, **result}}]},
    ]


_REFUSED = {"status": "error", "content": [{"text": "skill not found"}]}
_LOADED_OPAQUE = {"content": [{"text": '{"status": "loaded", "path": ".agents/pdf-processing"}'}]}


def test_retry_after_a_refused_load_is_reported_as_loaded():
    """One success anywhere in the run means the agent got the skill.

    The retry here returns no body, so a merge that only compares body length would leave the
    first attempt's refusal in place and report a skill the agent did receive as failed.
    """
    messages = [
        *_load_attempt("1", "pdf-processing", _REFUSED),
        *_load_attempt("2", "pdf-processing", _LOADED_OPAQUE),
    ]

    assert extract_selected_skills(messages) == [_loaded("pdf-processing", None)]


def test_skill_refused_on_every_attempt_stays_failed():
    messages = [
        *_load_attempt("1", "pdf-processing", _REFUSED),
        *_load_attempt("2", "pdf-processing", _REFUSED),
    ]

    assert extract_selected_skills(messages) == [_failed("pdf-processing", "skill not found")]


def test_a_later_refusal_does_not_discard_a_recovered_body():
    """The agent already had the instructions; a failed re-load does not take them away."""
    body = "# PDF\n1. Extract the text."
    messages = [
        *_load_attempt("1", "pdf-processing", {"content": [{"text": body}]}),
        *_load_attempt("2", "pdf-processing", _REFUSED),
    ]

    assert extract_selected_skills(messages) == [_loaded("pdf-processing", body)]


def test_unkeyed_result_of_an_unrelated_tool_is_not_taken_as_the_skill_body():
    """An unkeyed result only pairs with the call it follows, not the next unclaimed one.

    Here the skill call's own result is missing and a later tool's is not; pairing across the
    intervening call attributes that tool's output to the skill as its instructions.
    """
    messages = [
        {"tool_name": "activate_skill", "parameters": {"name": "pdf-processing"}},
        {"tool_name": "get_weather", "parameters": {"city": "Paris"}},
        {"type": "tool_result", "status": "success", "llmContent": "Weather in Paris: 21C"},
    ]

    assert extract_selected_skills(messages) == [_loaded("pdf-processing", None)]


def test_a_malformed_match_does_not_hide_a_second_reading_of_the_same_call():
    """Shapes overlap, so one block can be a broken call in one shape and a whole one in another.

    A harness that wraps a `toolUse` and also tags the block `type: "tool_use"` carries the call
    twice. If the wrapper is truncated, stopping at the first recognizer that matched throws away
    the flat fields that did survive, and a real skill load reads as no load at all.
    """
    messages = [
        {
            "toolUse": {"toolUseId": "t1", "name": "skills"},  # no `input`: unusable
            "type": "tool_use",
            "id": "t1",
            "name": "skills",
            "input": {"skill_name": "pdf-processing"},  # the same call, intact
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": fx.SKILL_BODY}],
        },
    ]

    assert extract_selected_skills(messages) == [_loaded("pdf-processing", fx.SKILL_BODY)]


@pytest.mark.parametrize(
    "block",
    [
        pytest.param({"toolUse": {"toolUseId": "t1", "name": "skills"}}, id="bedrock-wrapper"),
        pytest.param({"tool_name": "skills", "parameters": None}, id="gemini-stream"),
        pytest.param({"content_type": "tool_use", "name": "skills"}, id="typed-message"),
    ],
)
def test_a_sibling_name_and_args_pair_is_not_read_as_the_broken_call(block):
    """Recovering a second reading must not become inventing a different call.

    `{"name", "args"}` is the loosest shape recognized, and a block that declares a harness is not
    it: these are one malformed call, not a valid bare one. The sibling pair here names a real skill
    tool and a skill the agent never asked for, so reading it would not merely lose the broken call
    but report the wrong skill as loaded. Reporting no call is the honest answer, so a block that
    declares a harness is offered only to the recognizers that read tagged shapes.
    """
    messages = [{**block, "name": "skills", "args": {"skill_name": "never-requested"}}]

    assert extract_selected_skills(messages) == []


# ---- The load-event layer ----------------------------------------------------
#
# `extract_selected_skills` reports one row per skill, which is what the judges want. These
# assert the distinctions that folding necessarily loses, and that they survive one layer down.


def test_events_keep_repeated_loads_that_the_summary_folds():
    events = extract_skill_load_events(fx.DUPLICATE_LOAD_MESSAGES)
    summary = extract_selected_skills(fx.DUPLICATE_LOAD_MESSAGES)

    assert [e.name for e in events] == ["pdf-processing", "pdf-processing"]
    assert len(summary) == 1
    # Distinct calls, so an evaluator counting reloads has something to count.
    assert len({e.call_id for e in events}) == 2
    assert [e.position for e in events] == sorted(e.position for e in events)


def test_a_call_whose_outcome_never_appears_is_attempted_not_loaded():
    """A trajectory that stops before the result is not a load that succeeded silently.

    The summary can only say the agent asked for the skill, so it reports it as invoked with no
    body, which is the same row a successful load with an uncaptured body produces. The event
    keeps the two apart.
    """
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "no-result", "name": "skills", "input": {"skill_name": "xlsx"}}}],
        }
    ]

    (event,) = extract_skill_load_events(messages)
    assert (event.name, event.status, event.body) == ("xlsx", "attempted", None)

    (loaded,) = extract_skill_load_events(fx.BODY_MISSING_MESSAGES)
    assert loaded.status == "loaded"
    assert loaded.body is None


def test_a_refusal_and_its_retry_are_two_events_but_one_row():
    body = "# PDF\n\n1. Read it."
    messages = [
        *_load_attempt("1", "pdf-processing", _REFUSED),
        *_load_attempt("2", "pdf-processing", {"content": [{"text": body}]}),
    ]

    assert [(e.status, e.error) for e in extract_skill_load_events(messages)] == [
        ("failed", "skill not found"),
        ("loaded", None),
    ]
    assert extract_selected_skills(messages) == [_loaded("pdf-processing", body)]


def test_events_from_a_session_carry_position_and_the_call_id():
    spans = [
        ToolExecutionSpan(
            span_info=_span_info(),
            tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}, tool_call_id="c-1"),
            tool_result=ToolResult(content="4"),
        ),
        ToolExecutionSpan(
            span_info=_span_info(),
            tool_call=ToolCall(name="skills", arguments={"skill_name": "pdf-processing"}, tool_call_id="tu-1"),
            tool_result=ToolResult(content=fx.SKILL_BODY, tool_call_id="tu-1"),
        ),
    ]

    (event,) = extract_skill_load_events(_session(spans))
    assert (event.name, event.status, event.call_id) == ("pdf-processing", "loaded", "tu-1")
    # Position counts tool spans, so it locates the load among them rather than among skill loads.
    assert event.position == 2


def test_a_load_is_attributed_to_the_agent_that_made_it_when_the_trace_says_so():
    """Which sub-agent loaded a skill, for traces whose mapper records it.

    The trace types carry no agent identity, so this reads `metadata`. A trace without one
    reports None rather than guessing.
    """
    spans = [
        ToolExecutionSpan(
            span_info=_span_info(),
            metadata={"agent_name": "researcher"},
            tool_call=ToolCall(name="skills", arguments={"skill_name": "pdf-processing"}, tool_call_id="a"),
            tool_result=ToolResult(content=fx.SKILL_BODY),
        ),
        ToolExecutionSpan(
            span_info=_span_info(),
            tool_call=ToolCall(name="skills", arguments={"skill_name": "spreadsheet-analysis"}, tool_call_id="b"),
            tool_result=ToolResult(content="# Spreadsheets\n\n1. Open it."),
        ),
    ]

    assert [(e.name, e.agent_id) for e in extract_skill_load_events(_session(spans))] == [
        ("pdf-processing", "researcher"),
        ("spreadsheet-analysis", None),
    ]


def test_no_trajectory_yields_no_events():
    assert extract_skill_load_events(None) == []
    assert extract_skill_load_events([]) == []


# ---- Captured harness fixtures -----------------------------------------------
#
# The shapes above are hand-written from each harness's documented format. These read what the
# harnesses actually emitted, so a wire format that drifts from the hand-written version fails here
# rather than in a user's run. See `fixtures/capture_skill_fixtures.py` for how each was captured
# and which SDK version produced it.


def _captured(name: str):
    """Load a captured fixture by filename stem."""
    path = Path(__file__).resolve().parent / "fixtures" / f"{name}.json"
    return json.loads(path.read_text())


def _system_prompt_text(prompt) -> str:
    """The system prompt as text, whether the harness recorded a string or content blocks."""
    if isinstance(prompt, list):
        return "\n".join(block.get("text", "") for block in prompt)
    return prompt or ""


@pytest.mark.parametrize(
    "case, expected",
    [
        ("loaded", [("pdf-processing", "loaded")]),
        # The typo case: the plugin returns a plain string, which `@tool` marks status="success",
        # so only the text says the load failed.
        ("typo_refusal", [("pdf-procesing", "failed")]),
        ("retry_after_refusal", [("pdf-procesing", "failed"), ("pdf-processing", "loaded")]),
        ("two_skills", [("pdf-processing", "loaded"), ("spreadsheet-analysis", "loaded")]),
        ("repeated_load", [("pdf-processing", "loaded"), ("pdf-processing", "loaded")]),
    ],
)
def test_the_real_agent_skills_plugin_is_read_as_captured(case, expected):
    run = _captured("strands_agent_skills")[case]
    events = extract_skill_load_events(run["messages"])
    assert [(event.name, event.status) for event in events] == expected


def test_the_real_agent_skills_refusal_text_is_carried_not_mistaken_for_a_body():
    """jjbuck's case, against the plugin's own output rather than a transcription of it."""
    run = _captured("strands_agent_skills")["typo_refusal"]

    (event,) = extract_skill_load_events(run["messages"])
    assert event.body is None
    assert event.error == ("Skill 'pdf-procesing' not found. Available skills: pdf-processing, spreadsheet-analysis")
    assert extract_selected_skills(run["messages"]) == [_failed("pdf-procesing", event.error)]


def test_the_real_agent_skills_catalog_injection_parses():
    run = _captured("strands_agent_skills")["loaded"]
    skills = parse_available_skills(_system_prompt_text(run["system_prompt"]))
    assert [skill.name for skill in skills] == ["pdf-processing", "spreadsheet-analysis"]
    assert skills[0].description.startswith("Use this skill when the task requires")


def test_a_repeated_load_is_two_captured_events_and_one_row():
    run = _captured("strands_agent_skills")["repeated_load"]
    assert len(extract_skill_load_events(run["messages"])) == 2
    assert len(extract_selected_skills(run["messages"])) == 1


def test_the_real_claude_code_skill_tool_is_read_as_captured():
    """Claude Code acknowledges the launch and injects the body as a separate user message."""
    messages = _captured("claude_code_skill_tool")["messages"]

    (event,) = extract_skill_load_events(messages)
    assert (event.name, event.status) == ("pdf-processing", "loaded")
    # The acknowledgement ("Launching skill: pdf-processing") is not the body.
    assert event.body is not None
    assert "Identify the PDF file path." in event.body


def test_the_real_codex_exec_stream_is_read_as_captured():
    """`codex exec --json`: a skill load is a shell read, wrapped in an `item.completed` event."""
    events = _captured("codex_exec_json")["events"]

    (event,) = extract_skill_load_events(events)
    assert (event.name, event.status) == ("pdf-processing", "loaded")
    assert event.body is not None
    assert "Identify the PDF file path." in event.body


def test_the_real_codex_session_rollout_is_read_as_captured():
    """The same run recorded as Responses API items, with Codex's output preamble stripped."""
    items = _captured("codex_session_rollout")["items"]

    (event,) = extract_skill_load_events(items)
    assert (event.name, event.status) == ("pdf-processing", "loaded")
    assert event.body is not None
    # The preamble Codex prints ahead of the output is not part of the skill body.
    assert event.body.startswith("---\nname: pdf-processing")
    assert "Wall time" not in event.body


@pytest.mark.parametrize(
    "case, expected",
    [
        ("loaded", [("pdf-processing", "loaded")]),
        ("not_found", [("pdf-procesing", "failed")]),
    ],
)
def test_the_real_google_adk_load_skill_is_read_as_captured(case, expected):
    contents = _captured("google_adk_load_skill")[case]["contents"]
    events = extract_skill_load_events(contents)
    assert [(event.name, event.status) for event in events] == expected


def test_the_real_google_adk_refusal_carries_its_own_message():
    contents = _captured("google_adk_load_skill")["not_found"]["contents"]
    (event,) = extract_skill_load_events(contents)
    assert event.body is None
    assert event.error == "Skill 'pdf-procesing' not found."


@pytest.mark.parametrize(
    "fixture, case",
    [
        ("strands_agent_skills", "empty_name_refusal"),
        ("google_adk_load_skill", "missing_arg"),
    ],
)
def test_a_call_with_no_skill_name_yields_no_event(fixture, case):
    """Both harnesses refuse these for real, and neither refusal names a skill to report.

    Deliberate: an event would have to invent a skill named "", which would then be scored as a
    wrong selection. See `_skill_name_from_args`.
    """
    run = _captured(fixture)[case]
    messages = run.get("messages") or run["contents"]
    assert extract_skill_load_events(messages) == []
    assert extract_selected_skills(messages) == []
