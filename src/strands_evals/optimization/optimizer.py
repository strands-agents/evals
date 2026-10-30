"""The optimizer loop and the end phase: the only module that sees every layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from ..experiment import Experiment
from ..types.optimization import IterationRecord
from .declaration.space import Marker
from .declaration.tunable import BuildContext, TunableFactory
from .end_phase import FinalPick
from .execution.rollout import EnvironmentFn, RolloutFn, TaskFn
from .result import OptimizationResult
from .score_model.protocols import ScoreModel
from .scoring.objectives import Constraint, Objective
from .search.blocks import Search
from .selection.base import Selection
from .setup.budget import Budget
from .setup.splits import Split


class Optimizer:
    """Searches for changes to a declared agent on train, then picks what ships once, on validation.

    Each iteration the schedule plans a step, selection picks a parent, proposers propose, every candidate is
    scored on train, and the score model is refitted. There is no champion during the run: after the search
    stops, the top finalists and the baseline are measured on validation, a fresh fit picks the champion
    (only a candidate meeting every constraint ships, else the baseline), and the champion is compared with
    the baseline on test, reported only.

    Pass either `factory` (a `TunableFactory`, positionally as in `Optimizer(experiment, create_agent)` or
    by keyword) or `build_task=` with `space=`. `environment=` and `rollout_fn=` work only with `factory=`.
    `objective=None` maximizes the mean of every evaluator's score, as a plain eval run reports it.

    `build_task(ctx)` is called once per candidate and returns the `Experiment` task, `task(case)`.
    Build the agent inside `task`, never in `build_task` itself: an agent shared across cases and
    runs carries conversation state from one trial into the next, so every trial must build its
    own. The factory path enforces this; `build_task` can't.
    """

    def __init__(
        self,
        experiment: Experiment,
        factory: TunableFactory | None = None,
        *,
        rollout_fn: RolloutFn | None = None,
        build_task: Callable[[BuildContext], TaskFn] | None = None,
        space: Mapping[str, Marker] | None = None,
        objective: Objective | None = None,
        constraints: Sequence[Constraint] = (),
        search: Search | None = None,
        selection: Selection | None = None,
        score_model: ScoreModel | None = None,
        final_pick: FinalPick | None = None,
        split: Split | None = None,
        budget: Budget | None = None,
        runs_per_candidate: int = 1,
        crash_score: Literal["worst"] | float | None = "worst",
        feasibility: float = 0.9,
        environment: EnvironmentFn | None = None,
        proposals_per_iteration: int = 4,
        max_workers: int = 4,
        rerun_passes: int = 2,
        seed: int = 0,
        base_dir: str | Path | None = None,
        name: str | None = None,
        on_iteration: Callable[[IterationRecord], None] | None = None,
    ):
        """`base_dir=None` puts runs under `.strands-evals/optimization/runs/`, relative to the working directory.

        Args:
            experiment: Supplies the cases and evaluators; it is never run itself.
            factory: The `@tunable` agent factory. Pass this or `build_task`.
            rollout_fn: Runs one case with a fresh agent; defaults to building the agent and invoking it
                on `case.input`.
            build_task: Returns the task for one candidate, from a `BuildContext`; needs `space`.
            space: Declared name -> marker, for `build_task`.
            objective: What to maximize or minimize.
            constraints: Bounds every shipped candidate must meet; during the search they only rank.
            search: Which proposer changes which variables; None means `Search()`.
            selection: How the parent is chosen; None means `TopRanked()`.
            score_model: The statistical model of scores (not an LLM); None means `BayesBenchModel()`.
            final_pick: How many finalists go to validation and how the pick values them; None means
                `FinalPick(finalists=3)`.
            split: Train / validation / test; None means `Split(train=0.4, validation=0.3, test=0.3)`.
            budget: Rollout and iteration caps; None means `Budget()`.
            runs_per_candidate: Times every candidate is scored on each case.
            crash_score: What an `agent_failure` run's evaluator metrics count as: "worst" (each metric's
                worst end), a number, or None (missing data).
            feasibility: The P(met) threshold a constraint needs to count as met (not a fraction of cases);
                held by `Criteria`.
            environment: Per-attempt resources passed to the factory by name.
            proposals_per_iteration: Patches each proposing block returns per iteration.
            max_workers: Parallel trials, shared by the candidates scored at once.
            rerun_passes: Max infrastructure re-run passes per candidate and phase.
            seed: Seeds the split (unless `Split(seed=)` is set), selection, TPE, and Thompson.
            base_dir: Where run directories go.
            name: The run folder's prefix (lowercase letters, digits, `-`, `_`); for reading only.
            on_iteration: Called with each finished iteration.

        Raises:
            ImportError: The `optimization` extra (numpy, scipy, optuna) is missing, naming the install
                command.
            TypeError: Conflicting arguments, e.g. `factory` and `build_task` together.
            ValueError: No evaluators and no objective, fewer than one train or validation case, a bad
                `name`, or a budget below the baseline plus the end phase's reserve. (Split shares not
                summing to 1 are rejected by `Split` itself.)
        """
        raise NotImplementedError

    def run(self) -> OptimizationResult:
        """Run the search and the end phase; the sync form of `run_async`.

        Raises:
            RuntimeError: As `run_async`.
        """
        raise NotImplementedError

    async def run_async(self) -> OptimizationResult:
        """Score the baseline, run the search until it stops, then the end phase.

        Returns:
            The champion, the validation and test reports, the effects, and the run's costs.

        Raises:
            RuntimeError: The baseline isn't eligible after its re-run passes (some train case has no
                counted run), or a goal metric has no value on any train case. A `stop` event with
                reason `"setup"` is written first, and the message names the unmeasured cases or the
                metric, the trial status counts, and the first error.
        """
        raise NotImplementedError

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        *,
        experiment: Experiment,
        factory: TunableFactory | None = None,
        rollout_fn: RolloutFn | None = None,
        build_task: Callable[[BuildContext], TaskFn] | None = None,
        space: Mapping[str, Marker] | None = None,
        objective: Objective | None = None,
        constraints: Sequence[Constraint] = (),
        search: Search | None = None,
        selection: Selection | None = None,
        score_model: ScoreModel | None = None,
        environment: EnvironmentFn | None = None,
        budget: Budget | None = None,
        max_workers: int | None = None,
        on_iteration: Callable[[IterationRecord], None] | None = None,
    ) -> Optimizer:
        """Reopen a killed run; every saved trial is reused as is.

        Pass again what `run.json` can't store: the experiment, the agent (`factory=`, or `build_task=`
        with `space=`), `rollout_fn`, `environment`, the objective, constraints, search, selection, and
        score model. Of those, the cases, the space, the objective, constraints, search, selection, and score
        model are compared with `run.json` (`ValueError` naming the field); the agent code, `rollout_fn`,
        `environment`, and the evaluators can't be compared and are the caller's to keep. Everything else
        is read from `run.json`, never passed again: the split (with the seed it used), the final pick
        (`FinalPick.from_config`), `runs_per_candidate`, `crash_score`, `feasibility`, seed,
        `proposals_per_iteration`, and `rerun_passes`. Only `budget`, `max_workers`, and `on_iteration` may differ
        (`None` keeps the saved value).

        Resume continues a run that was killed; it never extends a finished one. Every stop runs the end
        phase, so a finished run only finishes or returns its end phase. To search more, start a new run.
        A killed search may get a larger `Budget`, which only moves where it stops. If you changed cases,
        evaluators, judge settings, or agent code, start a new run instead: resuming would mix results from
        the old and new setups.

        Args:
            run_dir: The run directory to reopen.
            experiment: As in `__init__`; its cases must match the run.
            factory: As in `__init__`.
            rollout_fn: As in `__init__`.
            build_task: As in `__init__`.
            space: As in `__init__`; must match the run.
            objective: As in `__init__`; must match the run.
            constraints: As in `__init__`; must match the run.
            search: As in `__init__`; must match the run.
            selection: As in `__init__`; must match the run.
            score_model: As in `__init__`; must match the run.
            environment: As in `__init__`.
            budget: A new budget for a killed search; None keeps the saved one.
            max_workers: None keeps the saved value.
            on_iteration: Called only for iterations run after the resume.

        Raises:
            ValueError: The configuration differs from `run.json`, naming the field.
        """
        raise NotImplementedError
