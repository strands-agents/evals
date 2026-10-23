"""The optimizer loop: the only module that sees every layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..experiment import Experiment
from ..types.optimization import IterationRecord, Value
from .acceptance.budget import Budget
from .acceptance.gate import Gate
from .declaration.tunable import TunableFactory
from .execution.rollout import EnvironmentFn, RolloutFn, TaskFn
from .result import OptimizationResult
from .scoring.objectives import Constraint, Objective
from .search.blocks import Search
from .selection.base import Selection

if TYPE_CHECKING:
    from .declaration.space import Marker


class Optimizer:
    """Searches for changes to a declared agent and keeps only those the holdout gate accepts.

    Pass either `factory=` (a `TunableFactory`) or `build_task=` with `space=`. `environment=`
    and `rollout_fn=` work only with `factory=`.
    """

    def __init__(
        self,
        experiment: Experiment,
        factory: TunableFactory | None = None,
        *,
        rollout_fn: RolloutFn | None = None,
        build_task: Callable[[Mapping[str, Value], Callable[[str], Any]], TaskFn] | None = None,
        space: Mapping[str, Marker] | None = None,
        objectives: Objective | None = None,
        constraints: Sequence[Constraint] = (),
        search: Search | None = None,
        selection: Selection | None = None,
        gate: Gate | None = None,
        budget: Budget | None = None,
        environment: EnvironmentFn | None = None,
        proposals_per_iteration: int = 3,
        max_workers: int = 4,
        retries: int = 2,
        seed: int = 0,
        base_dir: str | Path | None = None,
        on_iteration: Callable[[IterationRecord], None] | None = None,
    ):
        raise NotImplementedError

    async def run(self) -> OptimizationResult:
        raise NotImplementedError

    def run_sync(self) -> OptimizationResult:
        raise NotImplementedError

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        *,
        experiment: Experiment,
        factory: TunableFactory | None = None,
        **kwargs: Any,
    ) -> Optimizer:
        """Reopen a run directory; every saved trial is reused as is.

        If you changed cases, evaluators, judge settings, or agent code, start a new run instead:
        resuming would mix results from the old and new setups. Resuming with a larger `Budget` is
        how to extend a run.
        """
        raise NotImplementedError
