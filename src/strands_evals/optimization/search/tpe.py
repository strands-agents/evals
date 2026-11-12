"""TPE over numbers and choices, one study per context."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, ClassVar, Literal

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial

from ...types.optimization import Evidence, Observation, Patch, Value, Values
from ..candidate import content_id
from ..declaration.space import ParamInfo, ParamKind, Space
from ..scoring.objectives import Maximize
from .proposer import ProposalRequest, Proposer


@dataclass(frozen=True)
class _Told:
    """One trial as TPE was told it: its values, the snapshot it was told, and the candidates whose
    current assessment stands for it (the best of them, by the latest ranking)."""

    values: Values
    objective: float | None
    term_values: list[float] | None
    violations: list[float] | None
    members: tuple[str, ...]


@dataclass(frozen=True)
class _Pending:
    """A patch proposed and not yet observed: a running trial in its context's study (constant liar)."""

    context: str
    params: dict[str, Any]
    patch: Patch


class TPE(Proposer):
    """Optuna TPE with one study per context (the choices outside its block); studies maximize.

    `TPESampler(multivariate=True, group=True, constant_liar=True)`: proposals not yet observed are running
    trials, so a step's n proposals keep apart before any is told, whether they come from one `propose` call
    or several. `observe` (its own block) and `evidence` (other blocks) append each candidate told to TPE's
    own list (replayed on resume); invalid patches are left out. Before each ask the study for a context is
    rebuilt from that list, each trial valued at its candidate's current assessment in `request.candidates`
    (`ranking_objective`, `violations` as the trial's constraint values), so TPE's good and bad trials are
    those of the latest fit, as the parent and the final pick are. A `Nested` opener's trial is valued at the
    best of its round (`Observation.round_candidate_ids`) by the latest ranking; a duplicate at the existing
    candidate's. A trial told with `objective=None`, or whose candidate is no longer eligible, is
    `TrialState.FAIL`; one whose candidates the request doesn't list keeps the values it was told. The
    sampler is reseeded from `f"{seed}:{iteration}:{redraw}:{context id}:{keys}:{pending}"` through sha256,
    where `keys` are the block's keys (so two TPE blocks in one step draw differently) and `pending` counts
    the context's proposals not yet observed (so two asks in one step draw differently).
    `seed=None` uses the run seed. Under a `Pareto` objective each study is multi-objective, one direction
    per objective, told `term_values` instead of `objective`.

    The context: a block holding a choice with two or more options has one; any other block one per
    combination of the choices outside it, so numbers under a choice's alternative are learned per
    structure. Keys are suggested in declaration order, a choice before its sub-keys, and a key inactive
    under the choices so far is not suggested. Choices and `OneOf` options are categories, told by position,
    so options Python compares equal (`1` and `True`) stay apart.
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
        self._told: list[_Told] = []
        # each told trial's context, computed once (a block's space and keys never change)
        self._contexts: list[str] = []
        self._pending: list[_Pending] = []

    async def propose(self, request: ProposalRequest) -> list[Patch]:
        """Ask `request.n` trials of the parent's context's study, each a patch on the block's active keys.

        Returns `[]` when no block key is active under the parent's choices.
        """
        space = request.space
        owned = set(request.keys)
        keys = [info.key for info in space.params() if info.key in owned]  # a choice before its sub-keys
        context = _context_of(space, keys, request.parent.values)
        seed = request.seed if self.seed is None else self.seed
        patches: list[Patch] = []
        for _ in range(request.n):
            pending = [p for p in self._pending if p.context == context]
            key = f"{seed}:{request.iteration}:{request.redraw}:{context}:{keys}:{len(pending)}"
            # 4 bytes: Optuna seeds numpy's RandomState, which takes < 2**32
            study = self._study_for(
                context, space, keys, request, int.from_bytes(sha256(key.encode()).digest()[:4], "big")
            )
            trial = study.ask()
            current: dict[str, Value] = dict(request.parent.values)
            updates: dict[str, Value] = {}
            params: dict[str, Any] = {}
            for k in keys:
                if not space.is_active(k, current):
                    continue
                info = space.param(k)
                if info.kind == "int":
                    params[k] = trial.suggest_int(k, int(info.min_value), int(info.max_value), log=info.log)  # type: ignore[arg-type]
                    updates[k] = params[k]
                elif info.kind == "float":
                    params[k] = trial.suggest_float(k, info.min_value, info.max_value, log=info.log)  # type: ignore[arg-type]
                    updates[k] = float(params[k])
                else:
                    assert info.options is not None
                    params[k] = trial.suggest_categorical(k, tuple(range(len(info.options))))
                    updates[k] = info.options[params[k]]
                current[k] = updates[k]
            if not updates:
                return []
            told = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)  # what the sampler counts
            phase = "random startup" if told < self.n_startup_trials else f"modeled on {told} told trials"
            patch = Patch(updates=updates, rationale=f"TPE: {phase}", state={"context": context})
            self._pending.append(_Pending(context, params, patch))
            patches.append(patch)
        return patches

    def observe(self, observation: Observation) -> None:
        """Close the patch's running trial and keep the candidate, re-valued at each ask; invalid ones are left out."""
        for i, pending in enumerate(self._pending):
            if pending.patch == observation.patch:
                del self._pending[i]
                break
        if observation.outcome == "invalid" or observation.values is None:
            return
        if observation.round_candidate_ids is not None:
            members = tuple(observation.round_candidate_ids)
        else:
            members = () if observation.candidate_id is None else (observation.candidate_id,)
        self._told.append(
            _Told(observation.values, observation.objective, observation.term_values, observation.violations, members)
        )

    def evidence(self, evidence: Evidence) -> None:
        """Keep another block's candidate, re-valued at each ask, so it informs the study of its context.

        A candidate never measured (`objective=None`: ineligible or unscored) is no evidence and is left out.
        """
        if evidence.objective is None:
            return
        self._told.append(
            _Told(
                evidence.values, evidence.objective, evidence.term_values, evidence.violations, (evidence.candidate_id,)
            )
        )

    def config(self) -> dict[str, Any]:
        """`seed` and `n_startup_trials`."""
        return {"seed": self.seed, "n_startup_trials": self.n_startup_trials}

    def _study_for(self, context: str, space: Space, keys: Sequence[str], request: ProposalRequest, seed: int) -> Any:
        """A fresh study for `context`: every told trial in it, valued now, then its pending proposals as running
        trials."""
        criteria = request.criteria
        listed = {c.candidate_id: c for c in request.candidates}  # in ranking order
        verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # "A new study created in memory" on every ask
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)  # multivariate, group
                sampler = optuna.samplers.TPESampler(
                    seed=seed,
                    n_startup_trials=self.n_startup_trials,
                    multivariate=True,
                    group=True,
                    constant_liar=True,
                )
            if criteria.is_pareto:
                directions: list[Literal["maximize", "minimize"]] = [
                    "maximize" if isinstance(o, Maximize) else "minimize"
                    for o in criteria.objective.objectives  # type: ignore[union-attr]
                ]
                study = optuna.create_study(directions=directions, sampler=sampler)
            else:
                study = optuna.create_study(direction="maximize", sampler=sampler)
        finally:
            optuna.logging.set_verbosity(verbosity)
        self._contexts.extend(_context_of(space, keys, t.values) for t in self._told[len(self._contexts) :])
        for told, told_context in zip(self._told, self._contexts, strict=True):
            if told_context != context:
                continue
            params = {k: _internal(space.param(k), told.values[k]) for k in keys if k in told.values}
            if not params:
                continue
            distributions = {k: _distribution(space.param(k)) for k in params}
            objective, term_values, violations = told.objective, told.term_values, told.violations
            members = [m for m in listed if m in told.members]
            if objective is not None and members:
                now = next((listed[m] for m in members if listed[m].is_eligible), None)
                objective = None if now is None else now.ranking_objective
                term_values = None if now is None else now.term_values
                violations = None if now is None else now.violations
            if criteria.is_pareto:
                values = term_values
            else:
                values = None if objective is None else [objective]
            if values is None:
                study.add_trial(create_trial(state=TrialState.FAIL, params=params, distributions=distributions))
                continue
            constraints = {str(i): v for i, v in enumerate(violations or [])} or None
            study.add_trial(
                create_trial(params=params, distributions=distributions, values=values, constraints=constraints)
            )
        for pending in self._pending:
            if pending.context == context:
                distributions = {k: _distribution(space.param(k)) for k in pending.params}
                study.add_trial(
                    create_trial(state=TrialState.RUNNING, params=pending.params, distributions=distributions)
                )
        return study


def _context_of(space: Space, keys: Sequence[str], values: Values) -> str:
    """The id of the landscape `values` belong to for a block of `keys`.

    A block holding a choice with two or more options has one landscape; any other block one per
    combination of the choices outside it.
    """
    owned = set(keys)
    infos = [space.param(k) for k in keys]
    if any(i.kind == "choice" and len(i.options or ()) >= 2 for i in infos):
        return content_id({})
    choices: Mapping[str, Value] = {
        info.key: values[info.key]
        for info in space.params()
        if info.kind == "choice" and info.key not in owned and info.key in values
    }
    return content_id(choices)


def _distribution(info: ParamInfo) -> Any:
    """The Optuna distribution of one key; a choice or `OneOf` is a category per option position."""
    if info.kind == "int":
        return IntDistribution(int(info.min_value), int(info.max_value), log=info.log)  # type: ignore[arg-type]
    if info.kind == "float":
        return FloatDistribution(info.min_value, info.max_value, log=info.log)  # type: ignore[arg-type]
    assert info.options is not None
    return CategoricalDistribution(tuple(range(len(info.options))))


def _internal(info: ParamInfo, value: Value) -> Any:
    """`value` as the trial holds it: a number as is, an option by its position (type and value must match).

    Raises:
        ValueError: `value` is none of the key's options.
    """
    if info.kind in ("int", "float"):
        return value
    assert info.options is not None
    position = next((i for i, o in enumerate(info.options) if type(o) is type(value) and o == value), None)
    if position is None:
        raise ValueError(f"TPE was told {value!r} for {info.key!r}, which is none of its options {info.options!r}")
    return position
