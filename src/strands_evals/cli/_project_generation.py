"""Generate review-only project drafts without writing a runnable dataset."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from typing import Any

from ..experiment import Experiment
from ..generators import ExperimentGenerator
from ._project import ProjectError, ProjectRunner

_DRAFT_TASK = """Generate distinct evaluation drafts grounded in the project's actual
instructions and tool schemas. Each case input must be exactly one realistic user
prompt as a plain string, not a transcript or JSON. Each expected_output must be a
nonempty plain-language assertion describing observable correct behavior, not a
sample assistant answer, a rubric, or JSON. Do not invent tools or capabilities.
Include varied normal, boundary, and error scenarios where supported by the
project. These are unreviewed suggestions for human review, not approved tests.
"""


def _to_drafts(experiment: Experiment[str, str], num_cases: int) -> dict[str, Any]:
    """Validate generated cases and convert them to the review-only wire format."""
    if not isinstance(experiment.cases, list) or len(experiment.cases) != num_cases:
        raise ProjectError(f"Generator must return exactly {num_cases} draft case(s)")
    drafts: list[dict[str, Any]] = []
    prompts: set[str] = set()
    for index, case in enumerate(experiment.cases, start=1):
        prompt = getattr(case, "input", None)
        assertion = getattr(case, "expected_output", None)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ProjectError(f"Generated case {index} must contain a nonempty string user prompt")
        if not isinstance(assertion, str) or not assertion.strip():
            raise ProjectError(f"Generated case {index} must contain a nonempty expected_output assertion")
        prompt, assertion = prompt.strip(), assertion.strip()
        if prompt in prompts:
            raise ProjectError("Generated drafts must have distinct user prompts")
        prompts.add(prompt)
        drafts.append(
            {
                "id": f"draft-{index:03d}",
                "assertion": assertion,
                "userTurns": [{"role": "user", "content": prompt}],
                "metadata": {"source": "strands-evals-generator", "reviewed": False},
            }
        )
    return {"schemaVersion": 1, "drafts": drafts}


def generate_project_drafts(
    project: str,
    runner: str | None,
    num_cases: int,
    model: str,
    task_description: str = "",
) -> dict[str, Any]:
    """Inspect a project and explicitly invoke the generator for review-only drafts.

    No files are created. Project and model output is redirected to stderr so the
    caller can emit exactly one JSON document on stdout.
    """
    if not 1 <= num_cases <= 50:
        raise ProjectError("--num-cases must be between 1 and 50 for --project")
    if not model or not model.strip():
        raise ProjectError("--project requires an explicit --model")
    with contextlib.redirect_stdout(sys.stderr):
        context = ProjectRunner(project, runner=runner).get_generation_context()
        instructions, tools = context.get("instructions"), context.get("tools")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ProjectError("Project generation requires nonempty instructions")
        if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
            raise ProjectError("Project generation requires a list of tool schemas")
        context_json = json.dumps({"instructions": instructions, "tools": tools}, ensure_ascii=False, allow_nan=False)
        generator: ExperimentGenerator[str, str] = ExperimentGenerator(input_type=str, output_type=str, model=model)
        experiment = asyncio.run(
            generator.from_context_async(
                context=context_json,
                task_description=_DRAFT_TASK
                + (f"\nAdditional guidance: {task_description}" if task_description else ""),
                num_cases=num_cases,
            )
        )
        return _to_drafts(experiment, num_cases)
