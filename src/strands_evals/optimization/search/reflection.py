"""Reflection: an agent contrasts the parent's train conversations and every measured version, then rewrites texts."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import subprocess
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from strands import tool
from strands.hooks.events import AfterToolCallEvent
from strands.models.model import Model
from strands.types.tools import AgentTool

from ...types.optimization import Observation, Patch
from ..declaration.space import ParamKind
from .proposer import ProposalRequest, Proposer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-5-5"  # run at temperature 1.0

# A tool result longer than this is cut, with a note telling the agent to read less at once.
_MAX_TOOL_OUTPUT = 100_000

# Directories where other work on the machine keeps its files: a command that names a path in one of them, outside
# the agent's own folders, is refused.
_PRIVATE_DIRS = ("/tmp", "/private", "/home", "/Users", "/root", "/var", "/mnt", "/opt", "/srv", "/data", "/run/user")
_PATH = re.compile(r"(?<![\w.:/~-])(/[^\s'\"|;&<>()`*?\[\]{},]*|\.\.(?=/|$|[\s'\"]))")


class Reflection(Proposer):
    """An agentic, contrastive proposer: complete new text values for its block.

    An agent reads the parent's train conversations (built from `request.trials`: a traces folder with
    `DIGEST.md`, `expected_output` left out) and every measured version (`request.candidates`: a versions
    folder plus a ranked table, re-ranked by the latest refit on every request), then rewrites the block's
    texts in full. The versions table is its only history; it keeps no record of its own attempts. Its
    tools are a shell confined by a path check to its own folders (not a sandbox: it runs with this
    process's environment and network) and a submit tool; tool output is cut at 100,000 characters.

    Which of the parent's train trials went well: each trial gets a reward in [0, 1] from the goal's
    bounded metrics (an `agent_failure` gets 0), and a trial succeeds when reward >= `success_threshold`;
    a case is failing when any of its trials isn't. `success_threshold=None` ranks trials by objective
    instead (a median split), as does a goal with no bounded metric.

    Each `propose` call (n=1) runs one agent (`concurrent = True`). The model call is retried 3 times,
    each logged as `proposer_retry`; if the last error is infrastructure it raises, otherwise unparseable
    output or every proposal dropped returns `[]`. Guardrails run in code after the agent returns,
    including the leakage screen (drop a value that adds a shown case's identifier-like name, or a
    verbatim run of >= 40 characters from a shown trial's input, output, or feedback, that the parent's
    value didn't contain), logged as `screened`. A submitted value over its `size_cap` is sent back, in the
    same conversation, to be shortened, up to `over_limit_retries` times; one still over is dropped and the
    parent's value kept.
    """

    kinds: ClassVar[frozenset[ParamKind]] = frozenset({"text"})
    concurrent: ClassVar[bool] = True

    def __init__(
        self,
        model: Model | str | None = None,
        *,
        system_prompt: str | None = None,
        success_threshold: float | None = 0.5,
        over_limit_retries: int = 2,
    ):
        """Configure the proposer agent.

        Args:
            model: The proposer's model; None means Claude Sonnet 5.5 on Bedrock
                (`global.anthropic.claude-sonnet-5-5`) at temperature 1.0.
            system_prompt: Replaces the default prompt.
            success_threshold: Reward a trial needs to count as a success; None splits by rank.
            over_limit_retries: How many times a submission with a value over its `size_cap` is sent back to be
                shortened; 0 drops it at once.

        Raises:
            TypeError: `model` is not a `Model`, a model id, or None, `system_prompt` is not a string,
                `success_threshold` is not a number or None, or `over_limit_retries` is not an integer.
            ValueError: `success_threshold` is outside [0, 1], or `over_limit_retries` is below 0.
        """
        if model is not None and not isinstance(model, (Model, str)):
            raise TypeError(f"Reflection(model=...) takes a Model, a model id, or None, not {model!r}")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError(f"Reflection(system_prompt=...) must be a string, not {system_prompt!r}")
        if success_threshold is not None:
            if not isinstance(success_threshold, (int, float)) or isinstance(success_threshold, bool):
                raise TypeError(
                    f"Reflection(success_threshold=...) must be a number or None, not {success_threshold!r}"
                )
            if not math.isfinite(success_threshold) or not 0 <= success_threshold <= 1:
                raise ValueError(f"Reflection(success_threshold=...) must be in [0, 1], not {success_threshold!r}")
        if not isinstance(over_limit_retries, int) or isinstance(over_limit_retries, bool):
            raise TypeError(f"Reflection(over_limit_retries=...) must be an integer, not {over_limit_retries!r}")
        if over_limit_retries < 0:
            raise ValueError(f"Reflection(over_limit_retries=...) must be 0 or more, not {over_limit_retries!r}")
        self.model = model
        self.system_prompt = system_prompt
        self.success_threshold = success_threshold
        self.over_limit_retries = over_limit_retries
        # Each command the shell refused: {"command": its first 500 characters, "paths": the paths it named}.
        self.refused: list[dict[str, Any]] = []
        self._warned = False

    async def propose(self, request: ProposalRequest) -> list[Patch]:
        """Warns once per instance, on the first call, that the shell is not a sandbox."""
        if not self._warned:
            self._warned = True
            warnings.warn(
                "Reflection runs model-written shell commands on this machine, with this process's environment "
                "(credentials included) and network. Its path check guards a cooperative model's mistakes and is "
                "not a sandbox, and trace text can carry prompt injection. Run the optimizer in a container or VM "
                "without credentials it doesn't need.",
                UserWarning,
                stacklevel=2,
            )
        raise NotImplementedError

    def observe(self, observation: Observation) -> None:
        """Nothing to keep: the versions table in each request is the history."""

    def config(self) -> dict[str, Any]:
        """Model id, the sha256 of the system prompt (None for the default prompt), `success_threshold`, and
        `over_limit_retries`."""
        if isinstance(self.model, str):
            model_id: str | None = self.model
        elif self.model is None:
            model_id = _DEFAULT_MODEL_ID
        else:
            model_id = self.model.get_config().get("model_id")
        prompt = None if self.system_prompt is None else hashlib.sha256(self.system_prompt.encode()).hexdigest()
        return {
            "model": model_id,
            "system_prompt_sha256": prompt,
            "success_threshold": self.success_threshold,
            "over_limit_retries": self.over_limit_retries,
        }


def _outside(command: str, roots: Sequence[str], cwd: str) -> list[str]:
    """The paths `command` names that lie outside `roots`, in a directory where other work keeps its files
    (`_PRIVATE_DIRS`), or above `cwd` (`..`), sorted.

    A path is inside a root when it is, as written or with symlinks resolved, so a root's real path counts too: on
    macOS a temp folder is `/var/folders/...` while `pwd` in it prints `/private/var/folders/...`.

    A guard against a cooperative model's mistakes, such as reading or writing a fixed name in a shared `/tmp`; not a
    sandbox: a program can still build a path, and paths elsewhere (`/etc`) are not checked.
    """
    prefixes = {r.rstrip(os.sep) for root in roots for r in (os.path.normpath(root), os.path.realpath(root))}
    found = []
    for token in _PATH.findall(command):
        path = os.path.normpath(os.path.join(cwd, token))
        if any(p == r or p.startswith(r + os.sep) for p in (path, os.path.realpath(path)) for r in prefixes):
            continue
        if token.startswith("..") or any(path == d or path.startswith(d + "/") for d in _PRIVATE_DIRS):
            found.append(token)
    return sorted(set(found))


def _confined_shell(work: str, roots: Sequence[str], refused: list[dict[str, Any]]) -> AgentTool:
    """A shell tool that runs each command with `bash -c` in `work` (also `HOME` and `TMPDIR`), with this process's
    environment otherwise, and refuses a command naming a path outside `roots` (`_outside`), appending it to
    `refused`."""

    @tool
    def shell(command: str, timeout: int = 300) -> str:
        """Run a bash command in your working folder and return its output (stdout, stderr and the exit code).
        Read only your trajectory folder and the other folders you were given, and write only in your working
        folder (also HOME and TMPDIR): other work on this machine is not yours to read.

        Args:
            command: The bash command to run.
            timeout: Seconds before the command is stopped.
        """
        named = _outside(command, roots, work)
        if named:
            refused.append({"command": command[:500], "paths": named})
            return (
                f"Refused: {', '.join(named)} is outside your folders ({', '.join(roots)}). Read only those "
                f"folders and write only in your working folder, {work}."
            )
        env = {**os.environ, "HOME": work, "TMPDIR": work}
        try:
            done = subprocess.run(
                ["bash", "-c", command], cwd=work, env=env, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return f"Stopped after {timeout} seconds."
        err = f"\n[stderr]\n{done.stderr}" if done.stderr else ""
        return f"{done.stdout}{err}\n[exit code {done.returncode}]"

    return shell


def _submit_tool(work: str, submitted: dict[str, str], parent: Mapping[str, Any]) -> AgentTool:
    """A tool that reads each submitted value from a file in `work` (real path checked) into `submitted`; all or
    nothing: one bad file and nothing is submitted.

    Each value is stripped, then given the trailing whitespace of the parent's value under that key, so a file's
    final newline survives: an unchanged rewrite equals the parent's value, and the diff doesn't show a changed last
    line.
    """

    @tool
    def submit_optimized_params(file_path_dict: dict[str, str]) -> str:
        """Submit optimized parameters by providing a mapping of param names to file paths.

        Each file should contain ONLY the parameter value as plain text.

        Args:
            file_path_dict: Dict mapping parameter names to file paths.
                e.g., {"system_prompt": "<working folder>/system_prompt.txt"}
        """
        values = {}
        errors = []
        for key, path in file_path_dict.items():
            full = os.path.realpath(os.path.join(work, path))
            if not full.startswith(os.path.realpath(work) + os.sep):
                errors.append(f"{key}: {path} is not in your working folder; write it in {work} and submit that")
                continue
            try:
                with open(full) as f:
                    value = f.read().strip()
                held = parent.get(key)
                values[key] = value + held[len(held.rstrip()) :] if isinstance(held, str) else value
            except FileNotFoundError:
                errors.append(f"{key}: file not found: {full}")
            except Exception as e:
                errors.append(f"{key}: error reading {full}: {e}")
        if errors:
            return "Errors:\n" + "\n".join(errors)
        submitted.clear()
        submitted.update(values)
        summary = ", ".join(f"{k} ({len(v)} chars)" for k, v in values.items())
        return f"Parameters submitted successfully: {summary}"

    return submit_optimized_params


def _cap_tool_output(event: AfterToolCallEvent) -> None:
    """Cut each text item of a tool result to `_MAX_TOOL_OUTPUT` characters, telling the agent to read less."""
    result = event.result
    if not result or "content" not in result:
        return
    for item in result["content"]:
        if isinstance(item, dict) and "text" in item and len(item["text"]) > _MAX_TOOL_OUTPUT:
            text = item["text"]
            item["text"] = (
                f"{text[:_MAX_TOOL_OUTPUT]}\n\n"
                f"[TRUNCATED: Output exceeded {_MAX_TOOL_OUTPUT} characters. Original length: {len(text)} "
                f"characters. Use targeted commands (grep, head, jq with selectors) to extract specific content.]"
            )
            logger.warning("Truncated tool output from %d to %d characters", len(text), _MAX_TOOL_OUTPUT)
