"""Source-project execution shared by run and optimize.

External runners receive one JSONL request on stdin and must write one JSON object
on stdout: ``{output: string, trajectory?: array}``. Requests contain ``project``,
``case_id``, ``user_turns`` and optional ``overrides``. Each case gets a fresh process.
Project sources are trusted executable code, not sandboxed configuration.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, cast
from urllib.parse import urlparse
from uuid import uuid4

from ..case import Case
from ..evaluators.deterministic import Contains
from ..evaluators.evaluator import Evaluator
from ..evaluators.output_evaluator import OutputEvaluator
from ..experiment import Experiment
from ..types.evaluation_report import EvaluationReport


class ProjectError(ValueError):
    """Invalid project inputs or a failed project execution."""


def resolve_project(project: str | Path) -> Path:
    """Resolve an explicit source file or an unambiguous conventional entry point."""
    path = Path(project).expanduser().resolve()
    if path.is_dir():
        candidates = [path / name for name in ("agent.py", "main.py", "agent.ts", "index.ts", "agent.js", "index.js")]
        candidates = [candidate for candidate in candidates if candidate.is_file()]
        if len(candidates) != 1:
            raise ProjectError(f"Project directory must have one entry point; pass an explicit source file: {path}")
        path = candidates[0]
    if not path.is_file() or path.suffix not in {".py", ".ts", ".js", ".mts", ".mjs"}:
        raise ProjectError(f"Project must be an existing Python/TypeScript/JavaScript source file: {path}")
    return path


def load_project_cases(directory: str | Path) -> list[Case]:
    """Load strict CLI capture-v1 JSON cases, without replaying captured assistants."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ProjectError(f"Cases must be an existing directory: {root}")
    files = sorted(root.iterdir())
    if not files:
        raise ProjectError(f"Cases directory is empty: {root}")
    cases: list[Case] = []
    ids: set[str] = set()
    for path in files:
        if not path.is_file() or path.suffix != ".json":
            raise ProjectError(f"Cases directory may contain only JSON files: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProjectError(f"Invalid case {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ProjectError(f"Case must be a JSON object: {path}")
        case_id, assertion, turns = data.get("id"), data.get("assertion"), data.get("userTurns")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ProjectError(f"Case id must be a unique nonempty string: {path}")
        if not isinstance(assertion, str) or not assertion.strip():
            raise ProjectError(f"Case assertion must be a nonempty string: {path}")
        if (
            not isinstance(turns, list)
            or not turns
            or any(
                not isinstance(turn, dict) or not isinstance(turn.get("content"), str) or not turn["content"].strip()
                for turn in turns
            )
        ):
            raise ProjectError(f"Case userTurns must contain nonempty text content: {path}")
        trajectory = data.get("trajectory", {"turns": [], "spans": []})
        if not isinstance(trajectory, dict) or any(
            not isinstance(trajectory.get(key), list) for key in ("turns", "spans")
        ):
            raise ProjectError(f"Case trajectory must contain turns and spans arrays: {path}")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ProjectError(f"Case metadata must be an object: {path}")
        ids.add(case_id)
        cases.append(
            Case(
                name=case_id,
                input=[turn["content"] for turn in turns],
                expected_output=assertion,
                expected_assertion=assertion,
                metadata=metadata,
            )
        )
    return cases


class AssertionContains(Contains):
    """Contains evaluator whose expected substring comes from each capture assertion."""

    def __init__(self) -> None:
        super().__init__(value="", name="Contains")

    def evaluate(self, evaluation_case: Any) -> Any:
        return Contains(value=evaluation_case.expected_output).evaluate(evaluation_case)


def _collector_env(collector: str | None) -> dict[str, str]:
    if collector is None:
        return {}
    return {
        "OTEL_EXPORTER_OTLP_ENDPOINT": collector,
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
            collector.rstrip("/")
            if collector.rstrip("/").endswith("/v1/traces")
            else collector.rstrip("/") + "/v1/traces"
        ),
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    }


@contextlib.contextmanager
def _project_environment(project: Path, collector: str | None) -> Iterator[None]:
    """Temporarily expose project imports, cwd and optional OTLP endpoint."""
    cwd, path = Path.cwd(), sys.path[:]
    updates = _collector_env(collector)
    previous = {key: os.environ.get(key) for key in updates}
    modules = sys.modules.copy()
    try:
        os.chdir(project.parent)
        sys.path.insert(0, str(project.parent))
        # Permit relative imports when the entrypoint lives in a Python package.
        root = project.parent
        while (root / "__init__.py").is_file():
            root = root.parent
        sys.path.insert(0, str(root))
        os.environ.update(updates)
        yield
    finally:
        os.chdir(cwd)
        sys.path[:] = path
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name, module in list(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if name not in modules and filename and Path(filename).resolve().is_relative_to(project.parent):
                sys.modules.pop(name, None)


class ProjectRunner:
    """Run a fresh source-defined agent for every captured case."""

    def __init__(self, project: str | Path, runner: str | None = None, collector_endpoint: str | None = None):
        self.project = resolve_project(project)
        self.collector_endpoint = collector_endpoint
        if collector_endpoint:
            parsed = urlparse(collector_endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ProjectError("Collector endpoint must be an absolute HTTP(S) URL")
        self.runner = shutil.which(runner) if runner else None
        if runner and not self.runner:
            raise ProjectError(f"Runner executable not found or not executable: {runner}")
        if self.project.suffix != ".py" and not self.runner:
            raise ProjectError("Non-Python projects require --runner EXECUTABLE")

    def _load_agent(self, overrides: dict[str, Any] | None = None) -> Any:
        captured: list[tuple[Any, tuple, dict]] = []
        try:
            harness: Any = importlib.import_module("strands_harness")
        except ModuleNotFoundError as exc:
            if exc.name != "strands_harness":
                raise
            harness = None
        original = getattr(harness, "create_harness", None)

        def capture(*args: Any, **kwargs: Any) -> Any:
            assert original is not None
            agent = original(*args, **kwargs)
            captured.append((agent, args, kwargs))
            return agent

        if original:
            harness.create_harness = capture
        try:
            package_parts: list[str] = []
            root = self.project.parent
            while (root / "__init__.py").is_file():
                package_parts.insert(0, root.name)
                root = root.parent
            name = ".".join([*package_parts, f"_evals_{uuid4().hex}"])
            spec = importlib.util.spec_from_file_location(name, self.project)
            if spec is None or spec.loader is None:
                raise ProjectError(f"Cannot load project: {self.project}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        finally:
            if original:
                harness.create_harness = original
        agent = getattr(module, "agent", None)
        self._captured_instructions = next(
            (opts.get("instructions") for obj, _, opts in captured if obj is agent), None
        )
        factory = getattr(module, "create_agent", None)
        if callable(factory):
            agent = factory(**(overrides or {}))
        elif overrides:
            construction = next(((args, opts) for obj, args, opts in captured if obj is agent), None)
            if construction:
                args, opts = construction
                assert original is not None
                agent = original(*args, **{**opts, **overrides})
            elif agent is not None and isinstance(getattr(agent, "system_prompt", None), str):
                agent.system_prompt = overrides["instructions"]
            else:
                raise ProjectError("Project does not support runtime instructions overrides")
        if not callable(agent):
            raise ProjectError('Project must export callable "agent" or "create_agent"')
        return agent

    def get_generation_context(self) -> dict[str, Any]:
        """Inspect authored instructions and tool schemas without invoking the agent."""
        if self.runner:
            response = self._external({"project": str(self.project), "action": "inspect"})
            instructions = response.get("instructions")
            tools = response.get("tools", [])
        else:
            with _project_environment(self.project, self.collector_endpoint), contextlib.redirect_stdout(sys.stderr):
                agent = self._load_agent()
                instructions = self._captured_instructions or getattr(agent, "system_prompt", None)
                registry = getattr(agent, "tool_registry", None)
                tools = registry.get_all_tool_specs() if registry is not None else []
        if not isinstance(instructions, str) or not instructions.strip():
            raise ProjectError("Project must expose nonempty instructions for case generation")
        if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
            raise ProjectError("Project tool schemas must be a list of objects")
        return {"instructions": instructions, "tools": tools}

    def get_instructions(self) -> str:
        """Extract the actual runtime Python system prompt, never a guessed source literal."""
        if self.runner:
            response = self._external({"project": str(self.project), "action": "inspect"})
            prompt = response.get("instructions")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ProjectError("Runner inspect must return nonempty string instructions")
            return prompt
        with _project_environment(self.project, self.collector_endpoint), contextlib.redirect_stdout(sys.stderr):
            agent = self._load_agent()
            prompt = self._captured_instructions or getattr(agent, "system_prompt", None)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ProjectError("Project must expose a nonempty string system_prompt for optimization")
        return prompt

    def run(self, case: Case, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Replay only user turns; return the final output and newly generated trajectory."""
        if overrides is not None and (
            set(overrides) != {"instructions"} or not isinstance(overrides["instructions"], str)
        ):
            raise ProjectError("Only a string instructions runtime override is supported")
        if self.runner:
            request = {"project": str(self.project), "case_id": case.name, "user_turns": case.input}
            if overrides is not None:
                request["overrides"] = overrides
            response = self._external(request)
            if not isinstance(response.get("output"), str):
                raise ProjectError("Runner response must contain a string output")
            if "trajectory" in response and not isinstance(response["trajectory"], list):
                raise ProjectError("Runner trajectory must be an array")
            return response
        with _project_environment(self.project, self.collector_endpoint), contextlib.redirect_stdout(sys.stderr):
            agent = self._load_agent(overrides)
            trajectory = []
            output = ""
            for turn in case.input:
                response = agent(turn)
                if inspect.isawaitable(response):
                    response = asyncio.run(response)
                output = str(response)
                trajectory.extend([{"role": "user", "content": turn}, {"role": "assistant", "content": output}])
            return {"output": output, "trajectory": trajectory}

    def _external(self, request: dict[str, Any]) -> dict[str, Any]:
        env = {**os.environ, **_collector_env(self.collector_endpoint)}
        result = subprocess.run(
            [cast(str, self.runner)],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            cwd=self.project.parent,
            env=env,
            check=False,
            timeout=300,
        )
        if result.returncode:
            raise ProjectError(f"Runner failed ({result.returncode}): {result.stderr.strip()}")
        try:
            response = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise ProjectError("Runner returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise ProjectError("Runner response must be an object")
        return response


def evaluate_project(
    runner: ProjectRunner,
    cases: list[Case],
    evaluators: list[Evaluator] | None = None,
    overrides: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Execute before grading so source failures cannot become successful score rows."""
    if not cases:
        raise ProjectError("At least one case is required")
    outputs = {case.name: runner.run(case, overrides) for case in cases}
    experiment = Experiment(
        cases=cases,
        evaluators=evaluators
        or [
            OutputEvaluator(
                rubric="Judge whether the actual output satisfies the assertion in expected_output. "
                "Use the complete user conversation as context. Do not require verbatim wording."
            )
        ],
    )
    return asyncio.run(experiment.run_evaluations_async(lambda case: outputs[case.name], max_workers=1))
