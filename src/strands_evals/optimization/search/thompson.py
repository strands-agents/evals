"""Thompson sampling over choices from the fit's sampled effects (opt-in)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, ClassVar

import numpy as np

from ...types.optimization import Observation, Patch, Value, Values
from ..declaration.space import ParamKind
from ..score_model.protocols import MetricModel
from ..scoring.objectives import AtMost, Criteria, Maximize, Minimize, Pareto, WeightedSum
from .proposer import ProposalRequest, Proposer


@dataclass(frozen=True)
class _Pending:
    """A patch proposed and not yet observed, with the values it held outside its block."""

    held: tuple[tuple[str, str, str], ...]
    combination: tuple[tuple[str, str, str], ...]
    patch: Patch


class Thompson(Proposer):
    """Proposes the combination of its block's choices that a posterior draw likes best, unmeasured first.

    Each proposal draws one set of effects per goal metric (`sample_effects(1, rng)` and `value_under`) and
    scores every combination of the block's choices (sampled down to `MAX_COMBINATIONS` if more), each at a
    template: the best-ranked eligible candidate with those choices, else the parent with them (keys a
    combination makes active filled from the best-ranked eligible candidate that has them, else their
    defaults, as `Candidate.child` does). It proposes the best fresh combination (not measured with the
    parent's other values, nor proposed and not yet observed, so a step's proposals differ) that meets the
    drawn constraints, else the one that misses them least (each miss over its metric model's `unit`).
    With no fit or any metric flagged, a fresh combination at random. With
    none fresh, every combination is a choice: the draw's best measured one is a duplicate, which the
    optimizer redraws, and a `Nested` opener opens its round on, so Thompson keeps choosing which structure
    to refine. `[]` only when no block key is active under the parent's choices.

    `random_weights=True` (default): with two or more objectives (`WeightedSum` or `Pareto`), each draw
    weighs them with fresh Dirichlet(1, ..., 1) weights, each term oriented and divided by its metric
    model's `unit`, so successive draws aim at different trade-offs. `random_weights=False`: each draw is
    combined by `Criteria.objective_value`, the declared weights (under `Pareto` the weighted gain the
    ranking uses), so Thompson aims only at what the final pick values. With one objective the flag
    changes nothing.

    The numpy `Generator` is seeded from `f"{seed}:{iteration}:{redraw}:thompson:{keys}:{pending}"` through
    sha256, where `keys` are the block's keys (so two Thompson blocks in one step draw differently) and
    `pending` counts proposals not yet observed, so resume replays it. Never a default block.
    """

    kinds: ClassVar[frozenset[ParamKind]] = frozenset({"one_of", "choice"})
    MAX_COMBINATIONS: ClassVar[int] = 512

    def __init__(self, *, seed: int | None = None, random_weights: bool = True):
        """Configure the sampler.

        Args:
            seed: None uses the run seed.
            random_weights: With two or more objectives, weigh them at random for each draw (True) or by
                their declared weights (False).

        Raises:
            TypeError: `seed` is not an int or None, or `random_weights` is not a bool.
        """
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError(f"Thompson(seed=...) must be an int or None, not {seed!r}")
        if not isinstance(random_weights, bool):
            raise TypeError(f"Thompson(random_weights=...) must be a bool, not {random_weights!r}")
        self.seed = seed
        self.random_weights = random_weights
        self._pending: list[_Pending] = []

    async def propose(self, request: ProposalRequest) -> list[Patch]:
        """Up to `request.n` patches, one posterior draw each."""
        space, parent = request.space, request.parent.values
        owned = set(request.keys)
        keys = [info.key for info in space.params() if info.key in owned]  # a choice before its sub-keys
        held = {k: v for k, v in parent.items() if k not in owned}
        seed = request.seed if self.seed is None else self.seed
        key = f"{seed}:{request.iteration}:{request.redraw}:thompson:{keys}:{len(self._pending)}"
        rng = np.random.default_rng(int.from_bytes(sha256(key.encode()).digest()[:8], "big"))

        # Every combination, reduced to the keys active under it (the rest held), each once.
        combinations: dict[tuple[tuple[str, str, str], ...], dict[str, Value]] = {}
        for options in itertools.product(*(space.param(k).options or () for k in keys)):
            combination: dict[str, Value] = {}
            for k, v in zip(keys, options, strict=True):
                if space.is_active(k, {**held, **combination}):
                    combination[k] = v
            if combination:
                combinations.setdefault(_key(combination, keys), combination)
        if not combinations:
            return []
        ids = list(combinations)
        if len(ids) > self.MAX_COMBINATIONS:
            picks = rng.choice(len(ids), size=self.MAX_COMBINATIONS, replace=False)
            ids = [ids[i] for i in sorted(picks)]

        def agrees(values: Values, ref: Mapping[str, Value], *, partial: bool) -> bool:
            """`values` hold `ref`'s values, typed (`1` is not `True`); `partial`: only the keys both have."""
            shared = [k for k in ref if k in values] if partial else list(ref)
            return _key(values, shared) == _key(ref, shared)

        eligible = [c for c in request.candidates if c.is_eligible]  # in ranking order
        measured = {_key(c.values, keys) for c in eligible if agrees(c.values, held, partial=True)}
        held_key = _key(held, sorted(held))
        proposed = {p.combination for p in self._pending if p.held == held_key}
        fit = request.fit if request.fit is not None and not request.fit.flagged else None

        def inherit(k: str) -> tuple[Value, str] | None:
            found = next((c for c in eligible if k in c.values), None)
            return None if found is None else (found.values[k], found.candidate_id)

        templates: list[Values] = []
        if fit is not None:
            for i in ids:
                made = next((c for c in eligible if agrees(c.values, combinations[i], partial=False)), None)
                templates.append(made.values if made else space.complete({**held, **combinations[i]}, fill=inherit))

        objective = request.criteria.objective
        objectives = objective.objectives if isinstance(objective, (WeightedSum, Pareto)) else (objective,)
        patches: list[Patch] = []
        for _ in range(request.n):
            fresh = [j for j, i in enumerate(ids) if i not in measured and i not in proposed]
            choices = fresh or list(range(len(ids)))
            if fit is None:
                j = choices[int(rng.integers(len(choices)))]
                rationale = "Thompson: a combination at random (no usable fit)"
            else:
                effects = {m: model.sample_effects(1, rng)[0] for m, model in fit.metrics.items()}
                weights = (
                    rng.dirichlet(np.ones(len(objectives))) if self.random_weights and len(objectives) > 1 else None
                )
                scores, misses = [], []
                for template in templates:
                    drawn = {m: float(fit.metrics[m].value_under(e[None, :], template)[0]) for m, e in effects.items()}
                    scores.append(_score(drawn, weights, objectives, fit.metrics, request.criteria))
                    miss = 0.0
                    for c in request.criteria.constraints:
                        over = drawn[c.metric] - c.bound if isinstance(c, AtMost) else c.bound - drawn[c.metric]
                        miss += max(over, 0.0) / fit.metrics[c.metric].unit
                    misses.append(miss)
                feasible = [j for j in choices if misses[j] == 0.0]
                pool = feasible or [min(choices, key=lambda j: misses[j])]
                j = max(pool, key=lambda j: scores[j])
                rationale = "Thompson: best under a posterior draw"
            proposed.add(ids[j])
            patch = Patch(updates=dict(combinations[ids[j]]), rationale=rationale)
            self._pending.append(_Pending(held_key, ids[j], patch))
            patches.append(patch)
        return patches

    def observe(self, observation: Observation) -> None:
        """Forget the patch as pending; the fit is what learns from it."""
        for i, pending in enumerate(self._pending):
            if pending.patch == observation.patch:
                del self._pending[i]
                return

    def config(self) -> dict[str, Any]:
        """`seed` and `random_weights`."""
        return {"seed": self.seed, "random_weights": self.random_weights}


def _key(values: Values, keys: Sequence[str]) -> tuple[tuple[str, str, str], ...]:
    """The block's active keys and values in `values`, typed, so `1` and `True` stay apart."""
    return tuple((k, type(values[k]).__name__, repr(values[k])) for k in keys if k in values)


def _score(
    drawn: Mapping[str, float],
    weights: Any,
    objectives: Sequence[Maximize | Minimize],
    metrics: Mapping[str, MetricModel],
    criteria: Criteria,
) -> float:
    """One template's objective under a draw: Dirichlet-weighted terms, each over its unit, else the declared one."""
    if weights is None:
        value = criteria.objective_value(drawn)
        return -math.inf if value is None else value
    return float(
        sum(
            w * (1.0 if isinstance(o, Maximize) else -1.0) * drawn[o.metric] / metrics[o.metric].unit
            for w, o in zip(weights, objectives, strict=True)
        )
    )
