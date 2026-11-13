"""Reflection: an agent contrasts the parent's train conversations and every measured version, then rewrites texts."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import signal
import subprocess
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from botocore.config import Config as BotocoreConfig
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.hooks.events import AfterToolCallEvent
from strands.models.bedrock import BedrockModel
from strands.models.model import CacheConfig, Model
from strands.types.tools import AgentTool

from ...types.optimization import CandidateSummary, Observation, Patch, TrialRecord, TrialStatus, Usage, Value
from ...types.trace import AgentInvocationSpan, InferenceSpan, Session, ToolCallContent, ToolResultContent
from ..declaration.space import ParamKind
from ..execution.status import classify
from ..prompt_templates.reflection.reflection_v0 import (
    REFUSED,
    SUBMIT_INSTRUCTIONS,
    TextVariable,
    VersionRow,
    over_limit_message,
    system_prompt,
    task_message,
)
from ..scoring.objectives import AtLeast, AtMost, Maximize, Pareto, WeightedSum
from .proposer import ProposalRequest, Proposer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-5-5"  # run at temperature 1.0

# A tool result longer than this is cut, with a note telling the agent to read less at once.
_MAX_TOOL_OUTPUT = 100_000

# Directories where other work on the machine keeps its files: a command that names a path in one of them, outside
# the agent's own folders, is refused.
_PRIVATE_DIRS = ("/tmp", "/private", "/home", "/Users", "/root", "/var", "/mnt", "/opt", "/srv", "/data", "/run/user")
_PATH = re.compile(r"(?<![\w.:/~-])(/[^\s'\"|;&<>()`*?\[\]{},]*|\.\.(?=/|$|[\s'\"])(?:/[^\s'\"|;&<>()`*?\[\]{},]*)?)")

# A version's results and other values, in its folder of the versions folder; its texts are files beside it.
_VERSION_FILE = "version.md"
# A version's header as the versions folder writes it, a line `# Version N` and the line below it: a submitted text
# carrying it was copied from `version.md`, and is refused.
_VERSION_HEADER = re.compile(r"^# Version \d+(?: \(the baseline\))?[ \t]*\r?\n(?:rank|built from): ", re.M)

# The digest's length limit, about 5k tokens: the agent reads it once instead of opening every trace to learn how
# each one went.
_DIGEST_CHARS = 20_000
_DIGEST_FILE = "DIGEST.md"

_ATTEMPTS = 4  # the agent's first try, then 3 retries
_RATIONALE_CHARS = 2_000
_FALLBACK_RATIONALE = "rewritten by Reflection"  # when the agent's first call ends with no text
_VERBATIM_CHARS = 40  # a run this long copied from a shown trial is a leak


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
            system_prompt: Replaces the default prompt; each literal `{work_folder}` in it becomes the agent's
                working folder (the only folder it may write in). The submit instructions are appended either way.
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
        """Up to `request.n` patches, one agent each, one after another (the optimizer calls a concurrent proposer
        with n=1, several at once). Warns once per instance, on the first call, that the shell is not a sandbox.

        Raises:
            Exception: The agent failed `_ATTEMPTS` times and the last error is infrastructure (`classify`).
        """
        if not self._warned:
            self._warned = True
            warnings.warn(
                "Reflection runs model-written shell commands on this machine, with this process's environment "
                "(credentials included) and network. Its path check guards a cooperative model's mistakes and is "
                "not a sandbox, and trace text can carry prompt injection. Run the optimizer in a container or VM "
                "without credentials it doesn't need.",
                UserWarning,
                stacklevel=1,  # here: `propose` is a coroutine, so the frame above it is the event loop's
            )
        patches = []
        for _ in range(request.n):
            patch = await self._propose_one(request)
            if patch is not None:
                patches.append(patch)
        return patches

    async def _propose_one(self, request: ProposalRequest) -> Patch | None:
        """One agent over fresh folders: the evidence, the conversation (with its send-backs), then the guardrails.
        None when there is nothing to learn from, nothing usable was submitted, or every value was dropped.

        The call's three folders (`traces`, `versions`, `work`) are made inside one temp folder of its own, which the
        shell lets the agent name: a model that `cd`s to the folder holding them, or reads `../versions/...` from
        `work`, is not refused."""
        keys = list(request.keys)
        parent = dict(request.parent.values)
        with tempfile.TemporaryDirectory(prefix="reflection_") as root:
            traces, versions, work = (os.path.join(root, name) for name in ("traces", "versions", "work"))
            for folder in (traces, versions, work):
                os.mkdir(folder)
            evidence = _build_evidence(request, self.success_threshold, traces, versions)
            if evidence is None:
                return None
            prompt = (
                system_prompt(work) if self.system_prompt is None else self.system_prompt.replace("{work_folder}", work)
            )
            message = task_message(
                versions=evidence.versions,
                versions_folder=versions,
                goal_metrics=evidence.goal_metrics,
                limits=evidence.limits,
                traces_folder=traces,
                digest_file=evidence.digest_file,
                label_metrics=evidence.label_metrics,
                label_ranked=evidence.label_ranked,
                judged_by_version=evidence.judged_by_version,
                variables=[
                    TextVariable(
                        name=k,
                        value=str(parent.get(k, "")),
                        edit_instructions=request.space.param(k).edit_instructions,
                        size_cap=request.space.param(k).size_cap,
                    )
                    for k in keys
                ],
                work_folder=work,
            )
            caps = {k: request.space.param(k).size_cap for k in keys}
            usage = Usage.zero()
            refused: list[dict[str, Any]] = []
            rejected: list[dict[str, str]] = []
            for attempt in range(_ATTEMPTS):
                submitted: dict[str, str] = {}
                agent = Agent(
                    model=self._model(),
                    system_prompt=prompt + SUBMIT_INSTRUCTIONS,
                    tools=[
                        _confined_shell(work, [traces, versions, work], refused, root),
                        _submit_tool(work, submitted, parent, rejected),
                    ],
                    # Strands' 40-message window, trimmed at the end of each call and on overflow, with the task
                    # message pinned: a send-back after a long first turn would otherwise go without it.
                    conversation_manager=SlidingWindowConversationManager(pin_first=1),
                    callback_handler=None,
                )
                agent.add_hook(_cap_tool_output, AfterToolCallEvent)
                started = time.monotonic()
                try:
                    params, rationale, revised = await self._converse(agent, message, submitted, caps, rejected)
                    break
                except Exception as error:  # noqa: BLE001 - any failure of the agent is retried, then judged
                    if attempt == _ATTEMPTS - 1:
                        self.refused.extend(refused)
                        if classify(error) is TrialStatus.INFRASTRUCTURE_INVALID:
                            raise
                        logger.warning("Reflection gave up after %d attempts: %s", _ATTEMPTS, error)
                        return None
                    request.log("proposer_retry", error_type=type(error).__name__, error=str(error)[:2000])
                finally:
                    usage = usage + _agent_usage(agent, time.monotonic() - started, self._model_id())
            self.refused.extend(refused)
            updates, dropped = _guard(request, params or {}, keys, parent)
        if not updates:
            return None
        state = {
            "trials": evidence.trials,
            "versions": len(evidence.versions),
            "revised": revised,
            "dropped": dropped,
            "rejected": [{"name": k, "reason": r[:300]} for reasons in rejected for k, r in reasons.items()][:10],
            "refused": refused[:10],
            "digest": {
                "sha256": hashlib.sha256(evidence.digest_text.encode()).hexdigest(),
                "chars": len(evidence.digest_text),
                "failing_cases": evidence.failing_cases,
            },
        }
        return Patch(updates=dict[str, Value](updates), rationale=rationale, state=state, usage=usage)

    async def _converse(
        self,
        agent: Agent,
        message: str,
        submitted: dict[str, str],
        caps: Mapping[str, int | None],
        rejected: list[dict[str, str]],
    ) -> tuple[dict[str, str] | None, str, list[dict[str, int]]]:
        """Send the task message, then send back, in the same conversation and up to `over_limit_retries` times, a
        submission with values over their `size_cap` (`over_limit_message`) or a turn that ended with nothing
        accepted on a refused submission (`REFUSED`); each new submission replaces the values it gives.

        Returns:
            What was submitted (None for nothing), the rationale (the first call's final text, capped at
            `_RATIONALE_CHARS`; `_FALLBACK_RATIONALE` when it has none), and the lengths sent back to be shortened.
        """
        before = len(rejected)
        result = await agent.invoke_async(message)
        rationale = (str(result).strip() or _FALLBACK_RATIONALE)[:_RATIONALE_CHARS]
        params: dict[str, str] | None = dict(submitted) or None
        submitted.clear()
        revised: list[dict[str, int]] = []
        for _ in range(self.over_limit_retries):
            over = {
                k: (len(v), cap) for k, v in (params or {}).items() if (cap := caps.get(k)) is not None and len(v) > cap
            }
            if params and over:
                revised.append({k: n for k, (n, _) in over.items()})
                text = over_limit_message(over)
            elif not params and len(rejected) > before:
                refusal = "Refused, nothing was submitted:\n" + "\n".join(f"{k}: {r}" for k, r in rejected[-1].items())
                text = REFUSED.format(refusal=refusal)
            else:
                break
            before = len(rejected)
            await agent.invoke_async(text)
            if submitted:
                params = {**(params or {}), **submitted}
                submitted.clear()
        return params, rationale, revised

    def _model(self) -> Model:
        """The proposer's model: the given `Model`, else Bedrock at temperature 1.0, streaming, with prompt caching
        (the agent rereads its whole conversation every turn; a cache point covers everything before it, the tool
        definitions included) and adaptive retries."""
        if isinstance(self.model, Model):
            return self.model
        return BedrockModel(
            model_id=self.model if isinstance(self.model, str) else _DEFAULT_MODEL_ID,
            temperature=1.0,
            streaming=True,
            cache_config=CacheConfig(strategy="auto"),
            boto_client_config=BotocoreConfig(
                retries={"max_attempts": 3, "mode": "adaptive"}, connect_timeout=256, read_timeout=256
            ),
        )

    def _model_id(self) -> str | None:
        return self.config()["model"]

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


def _outside(command: str, roots: Sequence[str], cwd: str, home: str | None = None) -> list[str]:
    """The paths `command` names that lie outside `roots`, in a directory where other work keeps its files
    (`_PRIVATE_DIRS`), or that start with `..` (resolved from `cwd`), sorted.

    A path is inside a root when it is, as written or with symlinks resolved, so a root's real path counts too: on
    macOS a temp folder is `/var/folders/...` while `pwd` in it prints `/private/var/folders/...`.

    `home`, the folder holding the roots, may be named by an absolute path (`cd` to it, then use short names). A path
    starting with `..` must land inside a root, never just in `home`: the text doesn't say where earlier `cd`s left
    the shell, so a `..` allowed to reach `home` could be repeated to climb above it.

    A guard against a cooperative model's mistakes, such as reading or writing a fixed name in a shared `/tmp`; not a
    sandbox: a program can still build a path, and paths elsewhere (`/etc`) are not checked.
    """

    def real(folders: Sequence[str]) -> set[str]:
        return {r.rstrip(os.sep) for f in folders for r in (os.path.normpath(f), os.path.realpath(f))}

    inside, named = real(roots), real(roots) | real([home] if home else [])
    found = []
    for token in _PATH.findall(command):
        path = os.path.normpath(os.path.join(cwd, token))
        allowed = inside if token.startswith("..") else named
        if any(p == r or p.startswith(r + os.sep) for p in (path, os.path.realpath(path)) for r in allowed):
            continue
        if token.startswith("..") or any(path == d or path.startswith(d + "/") for d in _PRIVATE_DIRS):
            found.append(token)
    return sorted(set(found))


def _confined_shell(
    work: str, roots: Sequence[str], refused: list[dict[str, Any]], home: str | None = None
) -> AgentTool:
    """A shell tool that runs each command with `bash -c` in `work` (also `HOME` and `TMPDIR`), with this process's
    environment otherwise, and refuses a command naming a path outside `roots` and `home` (`_outside`), appending it
    to `refused`.

    Each command runs in its own process group, killed when `bash` exits or its timeout passes, so a process it left
    in the background neither holds the call open until the timeout nor outlives it. Output that isn't valid UTF-8
    (a `head -c` cutting a character) is decoded with replacement characters."""

    @tool
    def shell(command: str, timeout: int = 300) -> str:
        """Run a bash command in your working folder and return its output (stdout, stderr and the exit code).
        Read only your trajectory folder and the other folders you were given, and write only in your working
        folder (also HOME and TMPDIR): other work on this machine is not yours to read.

        Args:
            command: The bash command to run.
            timeout: Seconds before the command is stopped.
        """
        named = _outside(command, roots, work, home)
        if named:
            refused.append({"command": command[:500], "paths": named})
            return (
                f"Refused: {', '.join(named)} is outside your folders ({', '.join(roots)}). Read only those "
                f"folders and write only in your working folder, {work}."
            )
        env = {**os.environ, "HOME": work, "TMPDIR": work}
        # Files, not pipes: a background process holding a pipe open would keep the call waiting until it exits.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            process = subprocess.Popen(
                ["bash", "-c", command], cwd=work, env=env, stdout=out, stderr=err, start_new_session=True
            )
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return f"Stopped after {timeout} seconds."
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:  # the group already exited
                    pass
                process.wait()
            out.seek(0)
            err.seek(0)
            stdout = out.read().decode("utf-8", errors="replace")
            stderr = err.read().decode("utf-8", errors="replace")
        tail = f"\n[stderr]\n{stderr}" if stderr else ""
        return f"{stdout}{tail}\n[exit code {code}]"

    return shell


def _submit_tool(
    work: str, submitted: dict[str, str], parent: Mapping[str, Any], rejected: list[dict[str, str]]
) -> AgentTool:
    """A tool that reads each submitted value from a file in `work` (real path checked) into `submitted`; all or
    nothing: one bad file, or one value carrying a version's header (`_VERSION_HEADER`), and nothing is submitted.
    Each refused submission is appended to `rejected` (value name -> why), and the agent is told why.

    Each value is stripped (a shell's `echo` and heredocs add a newline the agent didn't mean), then given the leading
    and trailing whitespace of the parent's value under that key, so a file's final newline survives: an unchanged
    rewrite equals the parent's value, and the diff doesn't show a changed first or last line.
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
                with open(full, encoding="utf-8") as f:
                    value = f.read().strip()
                held = parent.get(key)
                if isinstance(held, str) and held.strip():
                    value = held[: len(held) - len(held.lstrip())] + value + held[len(held.rstrip()) :]
                elif isinstance(held, str):
                    value += held  # a parent of whitespace alone: kept after the new text, so "" stays unchanged
                values[key] = value
            except FileNotFoundError:
                errors.append(f"{key}: file not found: {full}")
            except Exception as e:
                errors.append(f"{key}: error reading {full}: {e}")
        if errors:
            return "Errors:\n" + "\n".join(errors)
        reasons = {}
        for key, value in values.items():
            found = _VERSION_HEADER.search(value)
            if found is not None:
                first = found.group(0).splitlines()[0].strip()
                reasons[key] = (
                    f"it holds a version's header from the versions folder (`{first}` and the results below it), "
                    f"which is not part of the text. Write the text alone to the file (a version's text is alone in "
                    f"its `.txt` file, never in {_VERSION_FILE}) and submit it again"
                )
        if reasons:
            rejected.append(reasons)
            return "Refused, nothing was submitted:\n" + "\n".join(f"{k}: {r}" for k, r in reasons.items())
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


@dataclass(frozen=True)
class _Evidence:
    """What the evidence folders hold, as the task message needs it.

    Attributes:
        versions: The versions table's rows, in `request.candidates` order.
        goal_metrics: The table's metric columns: every goal metric.
        limits: The constraints, as written.
        digest_file: The digest's path in the traces folder.
        digest_text: The digest's text.
        label_metrics: The metrics a trial's reward comes from.
        label_ranked: Whether the reward is a rank among the parent's trials.
        judged_by_version: Goal metrics left out of the reward, judged on whole versions.
        trials: How many trace files were written.
        failing_cases: How many cases the digest lists as not succeeding.
    """

    versions: list[VersionRow]
    goal_metrics: list[str]
    limits: list[str]
    digest_file: str
    digest_text: str
    label_metrics: list[str]
    label_ranked: bool
    judged_by_version: list[str]
    trials: int
    failing_cases: int


def _build_evidence(
    request: ProposalRequest, success_threshold: float | None, traces_folder: str, versions_folder: str
) -> _Evidence | None:
    """Write the parent's train trials (one file each, plus `DIGEST.md`) into `traces_folder` and every measured
    version into `versions_folder`; None when the parent has no trial to learn from (none completed or failed).

    Train data only: each trace holds the case's input from `request.inputs`, never its expected output.
    """
    rewards, label_metrics, label_ranked = _rewards(request, success_threshold)
    if not rewards:
        return None
    rows = _write_traces(traces_folder, request, rewards)
    digest_file = os.path.join(traces_folder, _DIGEST_FILE)
    threshold = 0.5 if success_threshold is None else success_threshold
    text = _digest_text(rows, threshold, _DIGEST_CHARS)
    with open(digest_file, "w", encoding="utf-8") as f:
        f.write(text)
    goal = list(request.criteria.goal_metrics)
    return _Evidence(
        versions=_write_versions(versions_folder, request, goal),
        goal_metrics=goal,
        limits=[str(c) for c in request.criteria.constraints],
        digest_file=digest_file,
        digest_text=text,
        label_metrics=label_metrics,
        label_ranked=label_ranked,
        judged_by_version=[m for m in goal if m not in label_metrics],
        trials=len(rewards),
        failing_cases=len({r["case"] for r in rows if r["reward"] < threshold}),
    )


def _rewards(
    request: ProposalRequest, success_threshold: float | None
) -> tuple[list[tuple[TrialRecord, float, dict[str, Any]]], list[str], bool]:
    """One reward in [0, 1] per parent train trial that completed or failed (0 when the agent failed), each with
    what its trace lists; and the metrics the rewards come from, and whether they are ranks.

    With a `success_threshold`: from the goal's bounded metrics (each scaled by its range and oriented so higher is
    better: the objectives', then the constraints'), combined with the objectives' weights when they are exactly the
    objectives' metrics, else their mean. A metric with no range, such as tokens, would rank a trial only against the
    parent's other trials, so it is judged on whole versions instead. With no bounded goal metric, each objective's
    mid-rank among the parent's trials, combined with the objectives' weights. With `success_threshold=None`: each
    trial's mid-rank by its objective among the parent's trials.
    """
    criteria = request.criteria
    registry = criteria.registry
    objective = criteria.objective
    objectives = list(objective.objectives) if isinstance(objective, (WeightedSum, Pareto)) else [objective]

    def bounded(metric: str) -> bool:
        low, high = registry.range(metric)
        return low is not None and high is not None and high > low

    goal: dict[str, bool] = {}  # bounded goal metric -> whether higher is better
    for o in objectives:
        if bounded(o.metric):
            goal.setdefault(o.metric, isinstance(o, Maximize))
    for c in criteria.constraints:
        if bounded(c.metric):
            goal.setdefault(c.metric, isinstance(c, AtLeast))
    trials = [t for t in request.trials if t.status in (TrialStatus.COMPLETED, TrialStatus.AGENT_FAILURE)]
    runs = request.scores.metrics_by_run

    def value(trial: TrialRecord, metric: str) -> float | None:
        values = runs.get(trial.case_name, {}).get(metric)
        return values[trial.run_index] if values is not None and trial.run_index < len(values) else None

    def unit(trial: TrialRecord, metric: str, maximize: bool) -> float | None:
        v = value(trial, metric)
        if v is None:
            return None
        low, high = registry.range(metric)
        assert low is not None and high is not None
        s = (v - low) / (high - low)
        return s if maximize else 1.0 - s

    def mid_ranks(values: list[float | None], maximize: bool) -> list[float | None]:
        known = [v for v in values if v is not None]
        out: list[float | None] = []
        for v in values:
            if v is None:
                out.append(None)
            elif len(known) < 2 or len(set(known)) == 1:
                out.append(0.5)
            else:
                better = sum(1 for w in known if (w < v if maximize else w > v))
                equal = sum(1 for w in known if w == v) - 1
                out.append((better + 0.5 * equal) / (len(known) - 1))
        return out

    if success_threshold is None:
        label, ranked = [o.metric for o in objectives], True
        by_run = request.scores.objective_by_run
        values = [
            by_run.get(t.case_name, [])[t.run_index] if t.run_index < len(by_run.get(t.case_name, [])) else None
            for t in trials
        ]
        scores = [0.0 if r is None else r for r in mid_ranks(values, True)]
    else:
        label, ranked = (list(goal), False) if goal else ([o.metric for o in objectives], True)
        columns = []
        for o in objectives:
            maximize = isinstance(o, Maximize)
            if bounded(o.metric):
                columns.append([unit(t, o.metric, maximize) for t in trials])
            else:
                columns.append(mid_ranks([value(t, o.metric) for t in trials], maximize))
        combined = all(bounded(o.metric) for o in objectives) and set(goal) == {o.metric for o in objectives}
        weights = [o.weight if len(objectives) > 1 else 1.0 for o in objectives]
        scores = []
        for i, t in enumerate(trials):
            if combined or not goal:
                parts = [0.0 if (x := col[i]) is None else x for col in columns]
                scores.append(sum(w * s for w, s in zip(weights, parts, strict=True)) / sum(weights))
            else:
                units = [unit(t, m, maximize) for m, maximize in goal.items()]
                scores.append(sum(u if u is not None else 0.0 for u in units) / len(goal))
    out = []
    for t, score in zip(trials, scores, strict=True):
        broken = {}
        for c in criteria.constraints:
            v = value(t, c.metric)
            excess = None if v is None else (v - c.bound if isinstance(c, AtMost) else c.bound - v)
            if excess is not None and excess > 0:
                broken[str(c)] = excess
        failed = t.status is TrialStatus.AGENT_FAILURE
        metrics = {
            m: v
            for m, vs in runs.get(t.case_name, {}).items()
            if (not failed or registry.is_usage(m)) and (v := value(t, m)) is not None
        }
        metadata = {
            "status": t.status.value,
            "error": t.error,
            "metrics": metrics,
            "reasons": {} if failed else {name: e.reason for name, e in t.evaluations.items()},
            "objectives": {o.metric: value(t, o.metric) for o in objectives},
            "constraints_broken": broken,
        }
        out.append((t, 0.0 if failed else score, metadata))
    return out, label, ranked


def _write_traces(
    folder: str, request: ProposalRequest, rewards: Sequence[tuple[TrialRecord, float, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """One JSON file per trial (`rollout_0000.json`, ...): its reward, how it ended (`output`), its messages once
    (`response.messages`), and its task (case name, run, and input: never the expected output). Returns each trial's
    row for the digest."""
    rows = []
    for i, (trial, reward, metadata) in enumerate(rewards):
        messages = _messages(trial.trajectory)
        name = f"rollout_{i:04d}"
        data = {
            "reward": reward,
            "output": trial.output,
            "response": {"messages": messages, "eval_result": {"reward": reward}},
            "eval_result": {"reward": reward},
            "task_id": name,
            "task": {
                "name": trial.case_name,
                "repetition": trial.run_index,
                "input": request.inputs.get(trial.case_name),
            },
            "metrics": {"reward": reward, "metadata": metadata},
        }
        with open(os.path.join(folder, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        calls = sum(
            1
            for m in messages
            if isinstance(m, dict)
            for b in (m.get("content") or [])
            if isinstance(b, dict) and "toolUse" in b
        )
        rows.append(
            {
                "file": f"{name}.json",
                "case": trial.case_name,
                "repetition": trial.run_index,
                "reward": float(reward),
                "status": metadata["status"],
                "error": metadata["error"],
                "reasons": {k: v for k, v in metadata["reasons"].items() if v},
                "objectives": {k: v for k, v in metadata["objectives"].items() if v is not None},
                "ended": trial.output,
                "messages": len(messages),
                "calls": calls,
            }
        )
    return rows


def _messages(trajectory: Any) -> list[Any]:
    """The trial's conversation as Strands messages (`{"role", "content": [{"text"} | {"toolUse"} | {"toolResult"}]}`):
    for each agent in the trial, in the order they started, the messages of its last model call (what it sent, then
    what it answered). A trajectory that isn't a `Session` (a rollout's own list) is kept as is.

    Messages are read by `role`, never by class: a reloaded assistant message can come back as a `UserMessage` with
    role "assistant".
    """
    if not isinstance(trajectory, Session):
        return list(trajectory) if isinstance(trajectory, list) else []
    out: list[Any] = []
    for trace in trajectory.traces:
        agents = sorted(
            (s for s in trace.spans if isinstance(s, AgentInvocationSpan)), key=lambda s: s.span_info.start_time
        )
        last: dict[int, InferenceSpan] = {}  # agent's index -> its last model call
        for span in trace.spans:
            if not isinstance(span, InferenceSpan):
                continue
            inside = [
                i
                for i, a in enumerate(agents)
                if a.span_info.start_time <= span.span_info.start_time
                and span.span_info.end_time <= a.span_info.end_time
            ]
            owner = max(inside, key=lambda i: agents[i].span_info.start_time) if inside else -1  # the innermost
            held = last.get(owner)
            if held is None or span.span_info.end_time >= held.span_info.end_time:
                last[owner] = span
        for owner in sorted(last, key=lambda i: (i >= 0, agents[i].span_info.start_time if i >= 0 else None)):
            for message in last[owner].messages:
                content: list[dict[str, Any]] = []
                for item in message.content:
                    if isinstance(item, ToolCallContent):
                        content.append(
                            {"toolUse": {"toolUseId": item.tool_call_id, "name": item.name, "input": item.arguments}}
                        )
                    elif isinstance(item, ToolResultContent):
                        content.append(
                            {
                                "toolResult": {
                                    "toolUseId": item.tool_call_id,
                                    "status": "error" if item.error else "success",
                                    "content": [{"text": item.content}],
                                }
                            }
                        )
                    else:
                        content.append({"text": item.text})
                role = getattr(message.role, "value", message.role)
                out.append({"role": role, "content": content})
    return out


def _short(text: Any, limit: int) -> str:
    """`text` on one line, cut to `limit` characters."""
    flat = " ".join(str(text).split()) if text is not None else ""
    return flat if len(flat) <= limit else flat[: max(0, limit - 3)] + "..."


def _digest_number(x: Any) -> str:
    if isinstance(x, (int, float)) and abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.3g}" if isinstance(x, float) else str(x)


def _trial_line(r: dict[str, Any], cap: int) -> str:
    parts = [f"{r['file']}: reward {r['reward']:.3g}"]
    if r["status"] and r["status"] != "completed":
        parts[0] += f" ({r['status']}{': ' + _short(r['error'], cap) if r['error'] else ''})"
    if r["ended"] is not None:
        parts.append(f"ended: {_short(r['ended'], cap)}")
    if r["reasons"]:
        parts.append("reasons: " + "; ".join(f"{k}: {_short(v, cap)}" for k, v in sorted(r["reasons"].items())))
    others = {k: v for k, v in r["objectives"].items() if not isinstance(v, float) or not 0 <= v <= 1}
    if others:
        parts.append(", ".join(f"{k} {_digest_number(v)}" for k, v in others.items()))
    parts.append(f"{r['messages']} messages, {r['calls']} tool calls")
    return "- " + "; ".join(parts)


def _digest_text(rows: Sequence[dict[str, Any]], threshold: float, limit: int) -> str:
    """The digest of every trace: a header, then every case with a conversation whose reward is below `threshold`
    (each such conversation: its file, reward, how it ended, the evaluators' reasons, its size), then the cases where
    every conversation succeeded. Its fields are cut until it fits in `limit` characters; every case that did not
    succeed is always named (when even their names do not fit, the digest says how many lines are left out)."""
    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    failing = sorted(
        (c for c, rs in by_case.items() if any(r["reward"] < threshold for r in rs)),
        key=lambda c: (-sum(r["reward"] < threshold for r in by_case[c]), c),
    )
    passing = sorted(c for c in by_case if c not in failing)
    bad = sum(r["reward"] < threshold for r in rows)
    head = [
        f"# Digest: the current version's {len(rows)} conversations, {len(by_case)} cases",
        "",
        f"{bad} conversations did not succeed (reward below {threshold:g}), in {len(failing)} cases; "
        f"every case where every conversation succeeded: {len(passing)}. Each conversation is a file in this folder: "
        "its messages in `response.messages`, its `reward`, `output` (how it ended) and `metrics`.",
        "",
    ]

    def build(cap: int, detailed: bool) -> str:
        out = list(head)
        out.append(f"## Every case with a conversation that did not succeed ({len(failing)})")
        for c in failing:
            rs = by_case[c]
            ok = sum(r["reward"] >= threshold for r in rs)
            if detailed:
                out.append(f"### {c}: {ok} of {len(rs)} succeeded")
                out += [_trial_line(r, cap) for r in sorted(rs, key=lambda r: (r["reward"], r["file"]))]
            else:
                worst = min(rs, key=lambda r: r["reward"])
                ended = f"; ended: {_short(worst['ended'], cap)}" if worst["ended"] is not None else ""
                files = ", ".join(r["file"] for r in rs if r["reward"] < threshold)
                out.append(f"- {c}: {ok} of {len(rs)} succeeded ({files}){ended}")
        out += ["", f"## Cases where every conversation succeeded ({len(passing)})"]
        out.append(", ".join(f"{c} ({len(by_case[c])}/{len(by_case[c])})" for c in passing) or "(none)")
        return "\n".join(out) + "\n"

    for detailed in (True, False):
        cap = 300
        while cap >= 40:
            text = build(cap, detailed)
            if len(text) <= limit:
                return text
            cap //= 2
    text = build(40, False)
    lines, kept = text.splitlines(), list[str]()
    for line in lines:  # still over: keep what fits, and say what was left out
        if sum(len(x) + 1 for x in kept) + len(line) + 1 > limit - 120:
            break
        kept.append(line)
    left = len(lines) - len(kept)
    note = f"({left} more lines left out at the digest's limit of {limit:,} characters: see the files)"
    return "\n".join(kept) + f"\n\n{note}\n"


def _write_versions(folder: str, request: ProposalRequest, goal_metrics: Sequence[str]) -> list[VersionRow]:
    """One folder per measured version, in `request.candidates` order, named by its rank (`rank-001-version-26`,
    `unranked-version-5`): `version.md` holds its results (each goal metric as `_shown`, and whether it meets
    the limits as the ranking assessed it) and other values, and each of the block's texts is a file of its own
    (`_text_files`) holding the text alone, so a copy of a version's text is the text and nothing else. Returns the
    versions table's rows."""
    keys = list(request.keys)
    by_id = {c.candidate_id: c for c in request.candidates}
    rows = []
    for c in request.candidates:
        parent = by_id.get(c.parent_id) if c.parent_id is not None else None
        changed = [k for k in keys if parent is not None and c.values.get(k) != parent.values.get(k)]
        if parent is not None and any(c.values.get(k) != parent.values.get(k) for k in set(c.values) - set(keys)):
            changed.append("other values")
        row = VersionRow(
            rank=c.rank,
            number=c.number,
            built_from=None if parent is None else parent.number,
            is_current=c.candidate_id == request.parent.id,
            is_baseline=c.parent_id is None,
            changed=changed,
            cells=[_shown(c, m) for m in goal_metrics],
            meets_limits=c.is_feasible if request.criteria.constraints else None,
            conversations=c.trial_count - c.failures,
        )
        rows.append(row)
        rank = f"rank-{c.rank:03d}" if c.rank is not None else "unranked"
        path = os.path.join(folder, f"{rank}-version-{c.number}")
        os.makedirs(path)
        files = _text_files(keys)
        lines = [f"# Version {c.number}" + (" (the baseline)" if row.is_baseline else "")]
        lines.append(f"rank: {c.rank if c.rank is not None else 'not ranked (a result is missing)'}")
        lines.append(
            f"built from: version {row.built_from}" if row.built_from is not None else "built from: nothing (the start)"
        )
        lines += [f"{m}: {cell}" for m, cell in zip(goal_metrics, row.cells, strict=True)]
        if row.meets_limits is not None:
            lines.append(f"meets the limits: {'yes' if row.meets_limits else 'no'}")
        lines.append(f"conversations: {row.conversations}")
        others = {k: v for k, v in c.values.items() if k not in keys}
        if others:
            lines += ["", "## Other values", json.dumps(others, indent=1, default=str)]
        lines += ["", "## Its texts, each in its own file in this folder (the text alone)"]
        for k in keys:
            text = c.values.get(k)
            value = "" if text is None else str(text)
            lines.append(f"- {k}: `{files[k]}` ({len(value):,} characters)")
            with open(os.path.join(path, files[k]), "w", encoding="utf-8") as f:
                f.write(value)
        with open(os.path.join(path, _VERSION_FILE), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return rows


def _cell(x: float | None) -> str:
    """A metric's mean as the versions table shows it."""
    if x is None:
        return "-"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def _shown(c: CandidateSummary, metric: str) -> str:
    """A metric as the versions table shows it: the fit's estimate the ranking used, then the measured mean over the
    version's train cases (`0.84 (measured 0.9)`); the measured mean alone when there is no estimate (no usable fit, a
    flagged metric, an ineligible version)."""
    measured = c.measured[metric].mean if metric in c.measured else None
    if metric not in c.estimates:
        return _cell(measured)
    estimate = _cell(c.estimates[metric].mean)
    return estimate if measured is None else f"{estimate} (measured {_cell(measured)})"


def _text_files(names: Sequence[str]) -> dict[str, str]:
    """Each text's file name in a version's folder: its name, with anything but letters, digits, `.`, `_` and `-`
    made `_` and leading dots dropped, then `.txt` (numbered when two names would share one, or `version.md`)."""
    out: dict[str, str] = {}
    for name in names:
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".") or "text"
        file, i = f"{stem}.txt", 2
        while file in out.values() or file == _VERSION_FILE:
            file, i = f"{stem}-{i}.txt", i + 1
        out[name] = file
    return out


def _guard(
    request: ProposalRequest, params: Mapping[str, str], keys: Sequence[str], parent: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    """The submitted values that become the patch, and why each other one was dropped (the parent's value kept).

    In order: a key outside the block is dropped; a value equal to the parent's is left out (nothing to change); a
    value `space.validate` refuses (over its `size_cap`, a `SKILL.md` that changes its `name`, invalid JSON) is
    dropped; a value that leaks a shown trial (`_leak`) is dropped and logged as `screened`.
    """
    shown = [t for t in request.trials if t.status in (TrialStatus.COMPLETED, TrialStatus.AGENT_FAILURE)]
    names = sorted({t.case_name for t in shown})
    sources = [str(request.inputs.get(name, "")) for name in names]
    for t in shown:
        sources.append("" if t.output is None else str(t.output))
        sources += [e.reason for e in t.evaluations.values() if e.reason]
    updates: dict[str, str] = {}
    dropped: dict[str, str] = {}
    for key, value in params.items():
        if key not in keys:
            dropped[key] = "not one of this block's keys"
            continue
        if value == parent.get(key):
            continue
        try:
            request.space.validate({**parent, key: value})
        except ValueError as error:
            dropped[key] = str(error)
            continue
        leak = _leak(value, parent.get(key), names, sources)
        if leak is not None:
            dropped[key] = leak
            request.log("screened", key=key, reason=leak)
            continue
        updates[key] = value
    return updates, dropped


def _leak(value: str, held: Any, names: Sequence[str], sources: Sequence[str]) -> str | None:
    """Why `value` leaks a shown trial, or None: it adds an identifier-like case name (one with a digit, `_`, `-`, or
    `/`, or CamelCase such as "RefundFlow"; plain words such as "refund" and acronyms such as "API" are skipped) as a
    whole word, or a run of `_VERBATIM_CHARS` characters (whitespace-normalized, case-insensitive) from a shown
    trial's input, output, or evaluator reasons, that the parent's value `held` didn't already contain."""
    held = held if isinstance(held, str) else ""
    for name in names:
        camel = re.fullmatch(r"(?=.*[a-z])[A-Za-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+", name)
        if not re.search(r"[\d_/-]", name) and not camel:
            continue
        word = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
        if word.search(value) and not word.search(held):
            return f"adds the name of a shown case, {name!r}"

    def normal(text: str) -> str:
        return " ".join(text.split()).lower()

    n = _VERBATIM_CHARS
    new, old = normal(value), normal(held)
    runs = {new[i : i + n] for i in range(len(new) - n + 1)} - {old[i : i + n] for i in range(len(old) - n + 1)}
    if runs:
        for source in sources:
            text = normal(source)
            for i in range(len(text) - n + 1):
                if text[i : i + n] in runs:
                    return f"copies {n} characters from a shown trial verbatim: {text[i : i + n]!r}"
    return None


def _agent_usage(agent: Agent, seconds: float, model_id: str | None) -> Usage:
    """The proposer agent's spend, from `event_loop_metrics` (its own spans are dropped at export): `tokens`
    includes cache, `input_tokens` is `tokens - output_tokens`, cache tokens None when never reported, model calls
    counted as event-loop cycles."""
    metrics = agent.event_loop_metrics
    used: Mapping[str, Any] = metrics.accumulated_usage
    total, output = int(used.get("totalTokens", 0)), int(used.get("outputTokens", 0))
    read, write = used.get("cacheReadInputTokens"), used.get("cacheWriteInputTokens")
    return Usage(
        tokens=total,
        input_tokens=total - output,
        output_tokens=output,
        cache_read_tokens=None if read is None else int(read),
        cache_write_tokens=None if write is None else int(write),
        model_calls=int(metrics.cycle_count),
        seconds=seconds,
        models=[model_id] if model_id else [],
    )
