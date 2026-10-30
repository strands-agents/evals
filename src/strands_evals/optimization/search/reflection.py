"""Reflection: an agent contrasts the parent's train conversations and every measured version, then rewrites texts."""

from __future__ import annotations

from typing import Any, ClassVar

from strands.models.model import Model

from ...types.optimization import Observation, Patch
from ..declaration.space import ParamKind
from .proposer import ProposalRequest, Proposer


class Reflection(Proposer):
    """An agentic, contrastive proposer: complete new text values for its block.

    An agent reads the parent's train conversations (built from `request.trials`: a traces folder with
    `DIGEST.md`, `expected_output` left out) and every measured version (`request.candidates`: a versions
    folder plus a ranked table, re-ranked by the latest refit on every request), then rewrites the block's
    texts in full. The versions table is its only history; it keeps no record of its own attempts. It
    runs in an empty environment, and its tool results are capped.

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
            model: The proposer's model; None means the repo's default Bedrock model.
            system_prompt: Replaces the default prompt.
            success_threshold: Reward a trial needs to count as a success; None splits by rank.
        """
        raise NotImplementedError

    async def propose(self, request: ProposalRequest) -> list[Patch]:
        raise NotImplementedError

    def observe(self, observation: Observation) -> None:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """Model id, the sha256 of the system prompt, and `success_threshold`."""
        raise NotImplementedError
