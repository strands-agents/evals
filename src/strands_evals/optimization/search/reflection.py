"""Reflection: an agent contrasts the parent's train conversations and every measured version, then rewrites texts."""

from __future__ import annotations

import hashlib
import math
import warnings
from typing import Any, ClassVar

from strands.models.model import Model

from ...types.optimization import Observation, Patch
from ..declaration.space import ParamKind
from .proposer import ProposalRequest, Proposer

_DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-5-5"  # run at temperature 1.0


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
    value didn't contain), logged as `screened`.
    """

    kinds: ClassVar[frozenset[ParamKind]] = frozenset({"text"})
    concurrent: ClassVar[bool] = True

    def __init__(
        self,
        model: Model | str | None = None,
        *,
        system_prompt: str | None = None,
        success_threshold: float | None = 0.5,
    ):
        """Configure the proposer agent.

        Args:
            model: The proposer's model; None means Claude Sonnet 5.5 on Bedrock
                (`global.anthropic.claude-sonnet-5-5`) at temperature 1.0.
            system_prompt: Replaces the default prompt.
            success_threshold: Reward a trial needs to count as a success; None splits by rank.

        Raises:
            TypeError: `model` is not a `Model`, a model id, or None, or `system_prompt` is not a string.
            ValueError: `success_threshold` is outside [0, 1].
        """
        if model is not None and not isinstance(model, (Model, str)):
            raise TypeError(f"Reflection(model=...) takes a Model, a model id, or None, not {model!r}")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError(f"Reflection(system_prompt=...) must be a string, not {system_prompt!r}")
        if success_threshold is not None and (
            not isinstance(success_threshold, (int, float))
            or isinstance(success_threshold, bool)
            or not math.isfinite(success_threshold)
            or not 0 <= success_threshold <= 1
        ):
            raise ValueError(f"Reflection(success_threshold=...) must be in [0, 1] or None, not {success_threshold!r}")
        self.model = model
        self.system_prompt = system_prompt
        self.success_threshold = success_threshold
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
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """Model id, the sha256 of the system prompt (None for the default prompt), and `success_threshold`."""
        if isinstance(self.model, str):
            model_id: str | None = self.model
        elif self.model is None:
            model_id = _DEFAULT_MODEL_ID
        else:
            model_id = self.model.get_config().get("model_id")
        prompt = None if self.system_prompt is None else hashlib.sha256(self.system_prompt.encode()).hexdigest()
        return {"model": model_id, "system_prompt_sha256": prompt, "success_threshold": self.success_threshold}
