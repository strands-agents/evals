"""Regenerate the captured skill fixtures from the harnesses themselves.

Every JSON file next to this script came out of a real harness rather than out of a guess at its
wire format, and this is the script that got it. Run it to refresh a fixture after an SDK upgrade,
or to check that the recorded shape is still the shape the harness emits:

    python tests/strands_evals/extractors/fixtures/capture_skill_fixtures.py --list
    python tests/strands_evals/extractors/fixtures/capture_skill_fixtures.py strands adk

It is not part of the test run. The fixtures are checked in, and the tests read those, so the suite
needs neither these SDKs nor network access. What the script buys is a way to prove the fixture is
still faithful and to regenerate it when a harness changes.

Provenance, per fixture:

- `strands_agent_skills.json` -- captured here. Runs a real `strands.Agent` with the real
  `AgentSkills` plugin (strands-agents 1.44.0) against a scripted model, so the tool spec, the
  system-prompt injection, the result envelope and the refusal strings are all the SDK's own. The
  model is scripted only to decide which calls happen, which is what makes the run deterministic
  and offline.
- `google_adk_load_skill.json` -- captured here. Drives the real `LoadSkillTool` from
  `google.adk.tools.skill_toolset` (google-adk 2.4.0) for a load, a misspelled name and a missing
  argument.
- `claude_code_skill_tool.json` -- transcribed from a real Claude Code run
  (`claude -p "Use the pdf-processing skill on report.pdf" --model haiku`, CLI transcript at
  `~/.claude/projects/<slug>/<session>.jsonl`), trimmed to the skill-load window. Not captured here
  because it needs the CLI and a model call.
- `codex_exec_json.json` -- transcribed from a real `codex exec --json` run (codex-cli 0.144.4),
  trimmed to the skill-load window.
- `codex_session_rollout.json` -- the same run's session rollout
  (`~/.codex/sessions/<date>/rollout-*.jsonl`), which records Responses API items rather than the
  `item.completed` events the `--json` stream emits. Both shapes are real and the extractor reads
  both, so both are fixtures.

Gemini CLI, OpenAI Agents and OpenHands have no captured fixture. Their shapes in
`skill_fixtures.py` are hand-written from the documented format, and are marked as such there.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parent
SKILL_MD = """---
name: pdf-processing
description: Use this skill when the task requires reading or extracting text from PDF files.
---

# PDF Processing Skill

1. Identify the PDF file path.
2. Extract the text.
3. Summarize it.
"""
SPREADSHEET_MD = """---
name: spreadsheet-analysis
description: Analyze, edit, or generate spreadsheets.
---

# Spreadsheet Analysis
1. Inspect.
"""


# ---- Strands ----------------------------------------------------------------


class _ScriptedModel:
    """A `Model` that replays a fixed list of turns, so the capture is deterministic and offline."""

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self._turns = list(turns)
        self.system_prompts: list[Any] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> Any:
        return {}

    def structured_output(self, output_model: Any, prompt: Any, system_prompt: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError

    @property
    def stateful(self) -> bool:
        return False

    @property
    def context_window_limit(self) -> int | None:
        return None

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, **kwargs: Any) -> Any:
        self.system_prompts.append(kwargs.get("system_prompt_content") or system_prompt)
        blocks = self._turns.pop(0) if self._turns else [{"text": "Done."}]
        yield {"messageStart": {"role": "assistant"}}
        for index, block in enumerate(blocks):
            if "text" in block:
                yield {"contentBlockStart": {"start": {}, "contentBlockIndex": index}}
                yield {"contentBlockDelta": {"delta": {"text": block["text"]}, "contentBlockIndex": index}}
            else:
                use = block["toolUse"]
                yield {
                    "contentBlockStart": {
                        "start": {"toolUse": {"toolUseId": use["toolUseId"], "name": use["name"]}},
                        "contentBlockIndex": index,
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "delta": {"toolUse": {"input": json.dumps(use["input"])}},
                        "contentBlockIndex": index,
                    }
                }
            yield {"contentBlockStop": {"contentBlockIndex": index}}
        yield {"messageStop": {"stopReason": "tool_use" if any("toolUse" in b for b in blocks) else "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}, "metrics": {}}}


def _skill_call(tool_use_id: str, skill_name: str) -> dict[str, Any]:
    return {"toolUse": {"toolUseId": tool_use_id, "name": "skills", "input": {"skill_name": skill_name}}}


async def capture_strands(skills_dir: Path) -> dict[str, Any]:
    """Run the real AgentSkills plugin through the loads, refusals and repeats worth recording."""
    from strands import Agent
    from strands.vended_plugins.skills import AgentSkills

    async def run(turns: list[list[dict[str, Any]]], prompt: str) -> dict[str, Any]:
        model = _ScriptedModel(turns)
        agent = Agent(
            model=model,  # type: ignore[arg-type]
            plugins=[AgentSkills(skills=str(skills_dir))],
            system_prompt="You are a helpful agent.",
        )
        await agent.invoke_async(prompt)
        prompts = [prompt_content for prompt_content in model.system_prompts if prompt_content]
        return {
            "system_prompt": prompts[0] if prompts else None,
            # Only the wire fields: the SDK also attaches per-message usage and metrics, which say
            # nothing about the shape a skill load takes.
            "messages": [
                {key: value for key, value in m.items() if key in ("role", "content")} for m in agent.messages
            ],
        }

    pdf = "Extract text from report.pdf"
    return {
        "loaded": await run(
            [
                [{"text": "I'll use the pdf-processing skill."}, _skill_call("tu-1", "pdf-processing")],
                [{"text": "Extracted and summarized."}],
            ],
            pdf,
        ),
        # A one-letter typo. The plugin returns an ordinary string, which `@tool` marks
        # status="success", so the refusal is only visible in the text.
        "typo_refusal": await run(
            [
                [{"text": "Loading the skill."}, _skill_call("tu-1", "pdf-procesing")],
                [{"text": "Sorry, I could not load it."}],
            ],
            pdf,
        ),
        "empty_name_refusal": await run([[_skill_call("tu-1", "")], [{"text": "Sorry."}]], pdf),
        "retry_after_refusal": await run(
            [
                [_skill_call("tu-1", "pdf-procesing")],
                [{"text": "Retrying with the right name."}, _skill_call("tu-2", "pdf-processing")],
                [{"text": "Done."}],
            ],
            pdf,
        ),
        "two_skills": await run(
            [
                [_skill_call("tu-1", "pdf-processing")],
                [_skill_call("tu-2", "spreadsheet-analysis")],
                [{"text": "Done."}],
            ],
            "Extract the PDF then build a sheet",
        ),
        "repeated_load": await run(
            [[_skill_call("tu-1", "pdf-processing")], [_skill_call("tu-2", "pdf-processing")], [{"text": "Done."}]],
            pdf,
        ),
    }


# ---- Google ADK -------------------------------------------------------------


async def capture_adk() -> dict[str, Any]:
    """Drive the real `LoadSkillTool` and wrap each payload in the Content shape ADK records."""
    from unittest.mock import MagicMock

    from google.adk.skills.models import Frontmatter, Skill
    from google.adk.tools.skill_toolset import SkillToolset

    skill = Skill(
        frontmatter=Frontmatter(
            name="pdf-processing",
            description="Use this skill when the task requires reading or extracting text from PDF files.",
        ),
        instructions=SKILL_MD.split("---", 2)[2].strip(),
    )
    tools = {tool.name: tool for tool in await SkillToolset(skills=[skill]).get_tools()}
    load = tools["load_skill"]

    def context() -> Any:
        ctx = MagicMock()
        ctx.state = {}
        ctx.agent_name = "pdf-agent"
        ctx.invocation_id = "inv-1"
        return ctx

    async def call(**args: Any) -> dict[str, Any]:
        response = await load.run_async(args=args, tool_context=context())
        name = args.get("skill_name", "")
        return {
            "contents": [
                {
                    "role": "model",
                    "parts": [{"function_call": {"id": "adk-1", "args": {"skill_name": name}, "name": "load_skill"}}],
                },
                {
                    "role": "user",
                    "parts": [{"function_response": {"id": "adk-1", "name": "load_skill", "response": response}}],
                },
            ]
        }

    return {
        "loaded": await call(skill_name="pdf-processing"),
        "not_found": await call(skill_name="pdf-procesing"),
        "missing_arg": await call(),
    }


# ---- Entry point ------------------------------------------------------------

CAPTURES = {
    "strands": ("strands_agent_skills.json", "strands-agents, real AgentSkills plugin"),
    "adk": ("google_adk_load_skill.json", "google-adk, real LoadSkillTool"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "which", nargs="*", default=[], choices=[*sorted(CAPTURES), []], help="which fixtures to regenerate"
    )
    parser.add_argument("--list", action="store_true", help="list what this script can capture and exit")
    args = parser.parse_args()

    if args.list or not args.which:
        for key, (filename, description) in sorted(CAPTURES.items()):
            print(f"{key:10} {filename:32} {description}")  # noqa: T201
        return

    for key in args.which:
        filename, _ = CAPTURES[key]
        if key == "strands":
            skills_dir = Path("/tmp/agent_skills_capture/skills")
            for name, body in (("pdf-processing", SKILL_MD), ("spreadsheet-analysis", SPREADSHEET_MD)):
                (skills_dir / name).mkdir(parents=True, exist_ok=True)
                (skills_dir / name / "SKILL.md").write_text(body)
            captured = asyncio.run(capture_strands(skills_dir))
        else:
            captured = asyncio.run(capture_adk())
        (FIXTURES / filename).write_text(json.dumps(captured, indent=2) + "\n")
        print(f"wrote {filename}")  # noqa: T201


if __name__ == "__main__":
    main()
