"""TPE over numbers and choices, one study per context (optional extra: optimization)."""

from __future__ import annotations

from typing import Any, ClassVar

from ...types.optimization import Evidence, Observation, Patch
from ..declaration.space import ParamKind
from .proposer import ProposalRequest, Proposer


class TPE(Proposer):
    """Optuna TPE with one study per context (the values outside its block); studies maximize.

    `TPESampler(multivariate=True, group=True, constant_liar=True, constraints_func=...)`: a
    step's n proposals are asked before any is told. Each candidate is valued as it was told, once, when
    observed (its own block) or shared as evidence (other blocks): `objective` as the value, `violations`
    as Optuna constraint values, `objective=None` as `TrialState.FAIL`; invalid patches are left out, and a
    later refit doesn't re-value it. `observe` and `evidence` append to TPE's own list (replayed on
    resume); the study for a context is rebuilt from that list before each ask, and the sampler is
    reseeded from `f"{seed}:{iteration}:{redraw}:{context id}"` through sha256. `seed=None` uses the run
    seed. Under a `Pareto` objective each study is multi-objective, one direction per objective, told
    `term_values` instead of `objective`.
    """

    kinds: ClassVar[frozenset[ParamKind]] = frozenset({"int", "float", "one_of", "choice"})

    def __init__(self, *, seed: int | None = None, n_startup_trials: int = 4):
        """Configure the sampler; studies are rebuilt per context before each ask.

        Args:
            seed: None uses the run seed.
            n_startup_trials: Random trials before TPE models the results; the default structure block
                uses 2.

        Raises:
            TypeError: `seed` is not an int or None.
            ValueError: `n_startup_trials` is not an int >= 0.
        """
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError(f"TPE(seed=...) must be an int or None, not {seed!r}")
        if not isinstance(n_startup_trials, int) or isinstance(n_startup_trials, bool) or n_startup_trials < 0:
            raise ValueError(f"TPE(n_startup_trials=...) must be an int >= 0, not {n_startup_trials!r}")
        self.seed = seed
        self.n_startup_trials = n_startup_trials

    async def propose(self, request: ProposalRequest) -> list[Patch]:
        raise NotImplementedError

    def observe(self, observation: Observation) -> None:
        raise NotImplementedError

    def evidence(self, evidence: Evidence) -> None:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """`seed` and `n_startup_trials`."""
        return {"seed": self.seed, "n_startup_trials": self.n_startup_trials}
