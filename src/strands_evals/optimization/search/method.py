"""The `Method` base class: proposes values, never decides what is accepted."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ...types.optimization import Observation, Patch, ProposalRequest
from ..declaration.space import ParamKind


class Method(ABC):
    """Proposes patches on a parent and learns from how each one did.

    `kinds` are the variable types it can propose. `uses_records=True` means the parent is scored
    on the train split before `propose` if it has no train records.
    """

    kinds: ClassVar[frozenset[ParamKind]]
    uses_records: ClassVar[bool] = False

    @abstractmethod
    async def propose(self, request: ProposalRequest) -> list[Patch]:
        """Return up to `request.n` patches; `[]` means nothing useful this time (a redraw cause).

        Raising means the method itself failed and ends the run with `stop_reason="proposer_error"`.
        """

    @abstractmethod
    def observe(self, observation: Observation) -> None:
        """Called exactly once per patch this method returned."""

    def config(self) -> dict[str, Any]:
        """JSON-serializable constructor arguments, stored in `run.json` and compared on resume."""
        raise NotImplementedError
