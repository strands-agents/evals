"""Thompson sampling over choices from the fit's sampled effects (opt-in)."""

from __future__ import annotations

from typing import Any, ClassVar

from ...types.optimization import Observation, Patch
from ..declaration.space import ParamKind
from .proposer import ProposalRequest, Proposer


class Thompson(Proposer):
    """Proposes the unmeasured combination of its block's choices that a posterior draw likes best.

    One sampled set of effects per goal metric (`sample_effects(1, rng)` and
    `value_under`), combined into the objective by `Criteria.objective_value`, scores every combination of
    the block's choices (sampled down to 512 if more), each at a template: the best-ranked measured
    candidate with those choices, else the parent with them. It proposes the best unmeasured combination
    that meets the drawn constraints, else the one with the least miss. With no fit or any metric flagged,
    an unmeasured option at random. Under a `Pareto` objective each draw weighs the objectives with fresh
    Dirichlet(1, ..., 1) weights from the same generator, each term oriented and divided by its metric
    model's `unit`, so successive draws aim at different parts of the front. The numpy `Generator` is seeded from
    `f"{seed}:{iteration}:{redraw}:thompson"` through sha256, so resume replays it. Never a default block.
    """

    kinds: ClassVar[frozenset[ParamKind]] = frozenset({"one_of", "choice"})

    def __init__(self, *, seed: int | None = None):
        """Configure the sampler.

        Args:
            seed: None uses the run seed.
        """
        raise NotImplementedError

    async def propose(self, request: ProposalRequest) -> list[Patch]:
        raise NotImplementedError

    def observe(self, observation: Observation) -> None:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """`seed`."""
        raise NotImplementedError
