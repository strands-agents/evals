"""The optimizer loop and the end phase: the only module that sees every layer."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import statistics
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .._async import run_async as _run_in_thread
from ..evaluators.evaluator import Evaluator
from ..experiment import Experiment
from ..types.optimization import (
    CandidateScores,
    CandidateSummary,
    Evidence,
    IterationRecord,
    Observation,
    ObservationOutcome,
    Patch,
    ProposalOutcome,
    SplitName,
    StopReason,
    TestReport,
    TrialRecord,
    TrialStatus,
    Usage,
    ValidationReport,
    Value,
    Values,
)
from .candidate import Candidate, changed_keys
from .declaration.space import Marker, Space
from .declaration.tunable import BuildContext, TunableFactory
from .end_phase import (
    PARETO_MIN_FINALISTS,
    TEST_LEVEL,
    FinalPick,
    choose_finalists,
    paired_interval,
    paired_ratio,
    pick,
)
from .execution.rollout import EnvironmentFn, RolloutFn, TaskFn, default_rollout, make_task
from .execution.traces import TraceCapture
from .execution.trials import TrialRunner
from .record.resume import OpenRound, ResumeState, load_state
from .record.run_directory import _NAME, RunDirectory, code_version
from .result import CandidateView, OptimizationResult
from .score_model.bayesbench import BayesBenchModel
from .score_model.effects import effects_json
from .score_model.protocols import Fit, FitData, ScoreModel
from .score_model.ranking import Assessment, Mean, assess, label_candidate, pareto_front, rank, round_to_sd
from .scoring.metrics import MetricRegistry
from .scoring.objectives import Constraint, Criteria, Objective, Pareto, WeightedSum, measured_metrics, score
from .search.blocks import Block, ScheduleContext, Search, Step
from .search.proposer import ProposalRequest, Proposer
from .selection.base import Selection, SelectionContext
from .selection.top_ranked import TopRanked
from .setup.budget import Budget, BudgetTracker
from .setup.splits import Split

_DEFAULT_BASE_DIR = Path(".strands-evals/optimization/runs")


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
        agent_failure_score: Literal["worst"] | float | None = "worst",
        feasibility_threshold: float = 0.9,
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
                `FinalPick(n_finalists=3)`.
            split: Train / validation / test; None means `Split(train=0.4, validation=0.3, test=0.3)`.
            budget: Rollout and iteration caps; None means `Budget()`.
            runs_per_candidate: Times every candidate is scored on each case.
            agent_failure_score: What an `agent_failure` run's evaluator metrics count as: "worst" (each metric's
                worst end), a number, or None (missing data).
            feasibility_threshold: The P(met) threshold a constraint needs to count as met (not a fraction of cases);
                held by `Criteria`.
            environment: Per-attempt resources passed to the factory by name.
            proposals_per_iteration: Candidates per step, its full width (a step's candidates merge one
                patch of each of its blocks). The schedule may give a step fewer (`Step.n_candidates`).
            max_workers: Parallel trials, shared by the candidates scored at once.
            rerun_passes: Max infrastructure re-run passes per candidate and phase.
            seed: Seeds the split (unless `Split(seed=)` is set), selection, TPE, and Thompson.
            base_dir: Where run directories go.
            name: The run folder's prefix (lowercase letters, digits, `-`, `_`); for reading only.
            on_iteration: Called with each finished iteration.

        Raises:
            TypeError: Conflicting arguments, e.g. `factory` and `build_task` together.
            ValueError: No evaluators and no objective, fewer than one train or validation case, a bad
                `name`, or a budget below the baseline plus the end phase's reserve. (Split shares not
                summing to 1 are rejected by `Split` itself.)
        """
        if (factory is None) == (build_task is None):
            raise TypeError(
                "pass either factory (a @tunable agent factory) or build_task= with space=, not both or neither"
            )
        if factory is not None:
            if not isinstance(factory, TunableFactory):
                raise TypeError(f"factory must be a TunableFactory (made by @tunable), got {type(factory).__name__}")
            if space is not None:
                raise TypeError("space= goes with build_task=; a factory declares its space with @tunable")
            self._space = factory.space
        else:
            if space is None:
                raise TypeError("build_task= needs space=, the declared name -> marker mapping")
            for argument, given in (("rollout_fn", rollout_fn), ("environment", environment)):
                if given is not None:
                    raise TypeError(f"{argument}= works only with factory=; build_task returns the whole task")
            self._space = Space(space)
        for argument, number, least in (
            ("runs_per_candidate", runs_per_candidate, 1),
            ("proposals_per_iteration", proposals_per_iteration, 1),
            ("max_workers", max_workers, 1),
            ("rerun_passes", rerun_passes, 0),
            ("seed", seed, None),
        ):
            if not isinstance(number, int) or isinstance(number, bool):
                raise TypeError(f"{argument} must be an int, got {number!r}")
            if least is not None and number < least:
                raise ValueError(f"{argument} must be >= {least}, got {number}")
        if name is not None and not _NAME.fullmatch(name):
            raise ValueError(f"name={name!r}: use only lowercase letters, digits, '-' and '_'")
        for argument, setting, kind in (
            ("search", search, Search),
            ("selection", selection, Selection),
            ("score_model", score_model, ScoreModel),
            ("final_pick", final_pick, FinalPick),
            ("split", split, Split),
            ("budget", budget, Budget),
        ):
            if setting is not None and not isinstance(setting, kind):
                raise TypeError(f"{argument} must be a {kind.__name__}, got {type(setting).__name__}")

        cases = experiment.cases
        unnamed = [index for index, case in enumerate(cases) if case.name is None]
        if unnamed:
            raise ValueError(f"every case needs a name (Case(name=...)) to pair its runs; cases {unnamed} have none")
        names = [case.name for case in cases if case.name is not None]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"case names must be unique; repeated: {repeated}")
        # Experiment replaces an empty evaluator list with a placeholder Evaluator() that always raises.
        evaluators = [e for e in experiment.evaluators if type(e) is not Evaluator]
        registry = MetricRegistry(
            {e.get_name(): tuple(getattr(e, "score_range", None) or (0.0, 1.0)) for e in evaluators}
        )
        self._criteria = Criteria(
            objective,
            constraints,
            registry,
            agent_failure_score=agent_failure_score,
            feasibility_threshold=feasibility_threshold,
        )
        if agent_failure_score == "worst" and all(registry.is_usage(m) for m in self._criteria.goal_metrics):
            warnings.warn(
                'every goal is a usage metric and agent_failure_score="worst" scores only evaluators, so a run that '
                "crashes early counts its real, small usage and looks cheap; add an evaluator term to the objective "
                "or an AtLeast constraint on one",
                UserWarning,
                stacklevel=2,
            )

        self._split = split if split is not None else Split()
        self._case_split = self._split.assign(cases, run_seed=seed)
        self._search = search if search is not None else Search()
        self._blocks, self._schedule = self._search.resolve(self._space)
        self._selection = selection if selection is not None else TopRanked()
        self._score_model = score_model if score_model is not None else BayesBenchModel()
        self._final_pick = final_pick if final_pick is not None else FinalPick()

        budget = budget if budget is not None else Budget()
        train, validation, test = (
            len(self._case_split.train),
            len(self._case_split.validation),
            len(self._case_split.test),
        )
        finalists = self._final_pick.n_finalists
        if self._criteria.is_pareto:
            finalists = max(finalists, PARETO_MIN_FINALISTS)
        validation_reserve = (finalists + 1) * validation * runs_per_candidate
        # Champion and baseline; under Pareto every front member.
        test_reserve = ((finalists + 1) if self._criteria.is_pareto else 2) * test * runs_per_candidate
        needed = train * runs_per_candidate + validation_reserve + test_reserve
        if budget.max_rollouts < needed:
            raise ValueError(
                f"Budget(max_rollouts={budget.max_rollouts}) can't cover the baseline ({train} train cases x "
                f"{runs_per_candidate} runs) plus the end phase's reserve ({validation_reserve} validation, "
                f"{test_reserve} test): give max_rollouts >= {needed}"
            )
        iteration = proposals_per_iteration * train * runs_per_candidate
        if budget.max_rollouts < needed + iteration:
            warnings.warn(
                f"Budget(max_rollouts={budget.max_rollouts}) leaves less than one iteration ({iteration} rollouts) "
                f"after the baseline and the end phase's reserve; give max_rollouts >= {needed + iteration} to search",
                UserWarning,
                stacklevel=2,
            )
        self._budget = BudgetTracker(budget, validation_reserve=validation_reserve, test_reserve=test_reserve)

        self._experiment = experiment
        self._cases = cases
        self._evaluators = evaluators
        self._factory = factory
        self._build_task = build_task
        self._rollout_fn = rollout_fn
        self._environment = environment
        self._runs_per_candidate = runs_per_candidate
        self._proposals_per_iteration = proposals_per_iteration
        self._max_workers = max_workers
        self._rerun_passes = rerun_passes
        self._seed = seed
        self._name = name
        self._on_iteration = on_iteration
        self._workdir = Path.cwd().resolve()
        # Resolved now, so a later change of working directory doesn't move the runs.
        self._base_dir = (self._workdir / (base_dir if base_dir is not None else _DEFAULT_BASE_DIR)).resolve()
        self._run: RunDirectory | None = None  # created by run_async; set by resume
        self._resumed: ResumeState | None = None  # set by resume: what the event log already holds
        # Candidate id -> (case name, run index) -> its trial record, as scored this process.
        self._records: dict[str, dict[tuple[str, int], TrialRecord]] = {}
        self._rollouts: dict[tuple[str, SplitName], int] = {}  # a _score call's spend, until its event

    def _config(self) -> dict[str, Any]:
        """`run.json`'s contents but `schema_version` and `code` (added by `RunDirectory.create`).

        Built the same way at `run_async` and at `resume`, since `load_state` compares the fields
        `Optimizer.resume` is given again (cases, space, objective, constraints, search, selection, score
        model) with the saved ones. Text in the space (a text default, `edit_instructions`) is stored as
        sha256 + length, never the text.
        """

        def text(value: str | None) -> dict[str, Any] | None:
            if value is None:
                return None
            return {"sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(), "length": len(value)}

        def qualified(obj: Any) -> str | None:
            if obj is None:
                return None
            return f"{getattr(obj, '__module__', None)}:{getattr(obj, '__qualname__', type(obj).__qualname__)}"

        space = [
            {
                "key": info.key,
                "name": info.name,
                "kind": info.kind,
                "default": text(info.default) if info.kind == "text" else info.default,  # type: ignore[arg-type]
                "min_value": info.min_value,
                "max_value": info.max_value,
                "log": info.log,
                "options": None if info.options is None else list(info.options),
                "size_cap": info.size_cap,
                "edit_instructions": text(info.edit_instructions),
                "active_when": None if info.active_when is None else list(info.active_when),
                "source": None
                if info.source is None
                else {"path": str(info.source.path), "pointer": info.source.pointer, "as_path": info.source.as_path},
            }
            for info in self._space.params()
        ]

        criteria = self._criteria
        registry = criteria.registry
        objective = criteria.objective
        directions = objective.objectives if isinstance(objective, (WeightedSum, Pareto)) else (objective,)
        metrics: dict[str, dict[str, Any]] = {}
        for metric in criteria.goal_metrics:
            low, high = registry.range(metric)
            metrics[metric] = {
                "kind": "usage" if registry.is_usage(metric) else "evaluator",
                "roles": [],
                "range": [low, high],
            }
        for direction in directions:
            entry = metrics[direction.metric]
            entry["roles"].append("objective")
            if len(directions) > 1:
                low, high = registry.range(direction.metric)
                entry["weight"] = direction.weight
                # None means the range width; an unbounded metric without scale= was rejected by Criteria.
                entry["scale"] = direction.scale if direction.scale is not None else high - low  # type: ignore[operator]
        for constraint in criteria.constraints:
            if "constraint" not in metrics[constraint.metric]["roles"]:
                metrics[constraint.metric]["roles"].append("constraint")

        split = self._split
        shares = isinstance(split.train, (int, float))
        budget = self._budget.budget
        return {
            "name": self._name,
            "cases": sorted(case.name for case in self._cases if case.name is not None),
            "split": {
                "train": split.train if shares else None,
                "validation": split.validation if shares else None,
                "test": split.test if shares else None,
                "seed": split.seed if split.seed is not None else self._seed,
                "cases": {
                    "train": list(self._case_split.train),
                    "validation": list(self._case_split.validation),
                    "test": list(self._case_split.test),
                },
            },
            "runs_per_candidate": self._runs_per_candidate,
            "agent_failure_score": criteria.agent_failure_score,
            "feasibility_threshold": criteria.feasibility_threshold,
            "seed": self._seed,
            "space": space,
            "objective": str(objective),
            "constraints": [str(c) for c in criteria.constraints],
            "metrics": metrics,
            "search": {
                "blocks": [
                    {
                        "name": block.name,
                        "proposer": type(block.proposer).__name__,
                        "config": block.proposer.config(),
                        "keys": list(keys),
                    }
                    for block, keys in self._blocks
                ],
                "schedule": {"name": self._schedule.name, "config": self._schedule.config()},
                "redraws": self._search.redraws,
            },
            "selection": {"name": self._selection.name, "config": self._selection.config()},
            "score_model": {"name": self._score_model.name, "config": self._score_model.config()},
            "final_pick": self._final_pick.config(),
            "proposals_per_iteration": self._proposals_per_iteration,
            "rerun_passes": self._rerun_passes,
            "budget": {"max_rollouts": budget.max_rollouts, "max_iterations": budget.max_iterations},
            "max_workers": self._max_workers,
            "agent": {
                "factory": qualified(getattr(self._factory, "__wrapped__", None)),
                "build_task": qualified(self._build_task),
                "rollout_fn": qualified(self._rollout_fn),
            },
            "workdir": str(self._workdir),
        }

    def run(self) -> OptimizationResult:
        """Run the search and the end phase; the sync form of `run_async`.

        Raises:
            RuntimeError: As `run_async`.
        """
        return _run_in_thread(self.run_async)

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
        await self._start()
        resumed = self._resumed
        if resumed is not None and resumed.stop_reason is not None:
            reason: StopReason = resumed.stop_reason  # the search is over: only the end phase is left
        else:
            reason = await self._run_search()
            self._stop(reason)
        return await self._end_phase(reason)

    async def _start(self) -> None:
        """Set up the run, score the baseline, and write the setup's fit (None: one eligible candidate).

        After `resume`, the state the event log holds is restored instead; with no closed iteration (killed
        before the first ended) only the setup ran, so it runs again on cached trials.
        """
        self._setup()
        self._propose_lock = asyncio.Lock()
        if self._resumed is not None and self._resumed.candidates:
            self._restore(self._resumed)
            return
        scores = await self._score_baseline()
        self._init_state(self._baseline)
        self._candidates[self._baseline.id] = self._baseline
        self._train[self._baseline.id] = scores
        self._refit(scored=True)

    def _init_state(self, baseline: Candidate) -> None:
        """The search's state before anything is scored."""
        self._candidates: dict[str, Candidate] = {}  # scored candidates, in scoring order (baseline first)
        self._train: dict[str, CandidateScores] = {}
        self._iterations: list[IterationRecord] = []
        self._history: list[str] = []
        self._fit: Fit | None = None  # the fit in use: None before one exists, or after a refit raised
        self._last_fit: Fit | None = None  # the last successful fit, the next refit's `previous`
        self._fit_errors: set[str] = set()
        self._fit_error: str | None = None  # the last refit's error while no refit has succeeded since
        self._saved: dict[str, Candidate] = {baseline.id: baseline}  # every saved candidate, unscored ones too
        self._open_rounds: list[OpenRound] = []
        self._search_cost = Usage.zero()
        self._iteration = 0

    def _restore(self, state: ResumeState) -> None:
        """Rebuild the search from the closed iterations: candidates and their scores from the saved trials,
        the fits replayed in order, and each block's `observe` / `evidence` history told to its proposer again."""
        assert self._run is not None
        run = self._run
        self._baseline = run.load_candidate(state.candidates[0])
        self._init_state(self._baseline)
        for candidate_id in state.candidates:
            candidate = run.load_candidate(candidate_id)
            self._saved[candidate_id] = candidate
            # As the closed iterations left them, not as a killed iteration may have replaced them.
            records = {
                (case, index): run.record_from(saved_trial)
                for split in ("train", "validation", "test")
                for case in getattr(self._case_split, split)
                for index in range(self._runs_per_candidate)
                if (saved_trial := state.trials.get((candidate_id, index, case))) is not None
            }
            if not records:
                continue  # saved but never scored (the budget couldn't cover it)
            self._records[candidate_id] = records
            train = [r for r in records.values() if r.split == "train"]
            if train:
                self._candidates[candidate_id] = candidate
                scores = score(candidate_id, train, self._criteria, self._case_split).train
                assert scores is not None
                self._train[candidate_id] = scores
        for replay in state.fits:  # each on exactly its listed candidates, with the previous replayed fit
            if replay.error is not None:
                self._fit, self._fit_error = None, replay.error
                self._fit_errors.add(replay.error)
            elif replay.is_fitted:
                fit = self._score_model.fit(self._fit_data(replay.ids, "train", self._train), previous=self._last_fit)
                if fit is not None:
                    self._fit = self._last_fit = fit
                    self._fit_error = None
        for block, told in state.feedback.items():
            proposer = self._proposer(block)
            for item in told:
                if isinstance(item, Observation):
                    proposer.observe(item)
                else:
                    proposer.evidence(item)
        self._iterations = list(state.iterations)
        self._history = list(state.history)
        self._open_rounds = list(state.open_rounds)
        self._search_cost = state.search_cost
        self._iteration = state.iteration

    async def _run_search(self) -> StopReason:
        """Run iterations until a stop: budget, iteration cap, the schedule's None, or a proposer raising."""
        assert self._run is not None
        per_candidate = len(self._case_split.train) * self._runs_per_candidate
        blocks = [block for block, _ in self._blocks]
        keys = {_block_name(block): list(block_keys) for block, block_keys in self._blocks}
        while True:
            budget = self._budget.budget
            room = None  # candidate slots the iteration cap leaves: max_iterations steps' worth
            if budget.max_iterations is not None:
                room = budget.max_iterations * self._proposals_per_iteration - _slots_used(self._iterations)
                if room <= 0:
                    return "iterations"
            if self._budget.allowance(phase="train") < per_candidate:
                return "budget"
            self._iteration += 1
            ranking, assessments = self._rank()
            step = self._schedule.next(
                ScheduleContext(
                    past=list(self._iterations),
                    blocks=blocks,
                    keys=keys,
                    space=self._space,
                    values={c: candidate.values for c, candidate in self._candidates.items()},
                    eligible=frozenset(ranking),
                    ranking=ranking,
                    run_ranking=ranking,
                    proposals_per_iteration=self._proposals_per_iteration,
                    redraws=self._search.redraws,
                )
            )
            self._close_rounds(step, ranking, assessments)
            if step is None:
                return "exhausted"
            n = min(step.n_candidates, self._budget.allowance(phase="train") // per_candidate)
            if room is not None:
                n = min(n, room)  # a last step narrower than the rest
            if step.parents is not None and not step.parents:
                raise ValueError(f"{type(self._schedule).__name__}.next returned a step with no possible parents")
            parent_id = self._select(list(step.parents) if step.parents is not None else ranking, assessments)
            try:
                record = await self._step(step, n, self._candidates[parent_id], ranking, assessments)
            except _ProposerFailed:
                return "proposer_error"
            self._iterations.append(record)
            if self._on_iteration is not None:
                self._on_iteration(record)

    def _stop(self, reason: StopReason) -> None:
        """Close every open round (its opener told the round's best), then write the `stop` event."""
        assert self._run is not None
        ranking, assessments = self._rank()
        self._close_rounds(None, ranking, assessments)
        cap = self._budget.budget.max_iterations
        messages = {
            "budget": "the rollout budget left after the end phase's reserve can't score another candidate",
            "iterations": f"used Budget(max_iterations={cap})'s candidate slots ({cap} steps' worth)",
            "exhausted": "the schedule has nothing left to try",
            "proposer_error": "a proposer raised; see the proposer_error event",
        }
        self._run.events.append("stop", iteration=self._iteration, reason=reason, message=messages.get(reason, ""))

    async def _end_phase(self, stop_reason: StopReason) -> OptimizationResult:
        """Finalists from the search ranking (E1), measured with the baseline on validation (E2), the champion
        picked on a fresh validation fit (E3), the test report (E4); then `effects.json`, the diffs, and
        `result.json`.

        After `resume`, each of E1, E3 (with E2), and E4 that the event log already holds is reused as saved, so a
        finished run only rewrites its files.
        """
        assert self._run is not None
        run, criteria, baseline = self._run, self._criteria, self._baseline.id
        resumed = self._resumed
        if resumed is not None and resumed.finalists is not None:
            finalists = resumed.finalists
        else:
            ranking, assessments = self._rank()
            n, front = self._final_pick.n_finalists, None
            if criteria.is_pareto:
                n, front = max(n, PARETO_MIN_FINALISTS), pareto_front(assessments, ranking)
            finalists = choose_finalists(ranking, baseline, n, front=front)
            run.events.append("finalists", iteration=self._iteration, candidate_ids=finalists)
        if resumed is not None and resumed.validation is not None:
            report = resumed.validation
        else:
            report = await self._pick(finalists)
        if resumed is not None and resumed.is_test_done:
            test, test_reason, front_test = resumed.test, resumed.test_reason, resumed.front_test
        else:
            test, test_reason, front_test = await self._test(report)
            run.events.append("test", iteration=self._iteration, report=test, reason=test_reason, front=front_test)
        return self._finish(stop_reason, report, test, test_reason, front_test)

    async def _pick(self, finalists: Sequence[str]) -> ValidationReport:
        """E2 and E3: the finalists and the baseline on validation, then the pick on a fresh validation fit."""
        assert self._run is not None
        criteria, baseline = self._criteria, self._baseline.id
        ids = [baseline, *finalists]
        validation = dict(zip(ids, await self._score_split(ids, "validation"), strict=True))
        # A fresh fit on the validation runs only, never mixed with the search fit (no previous fit).
        fit: Fit | None = None
        problem: str | None = "fewer than 2 eligible candidates"
        if sum(validation[c].is_eligible for c in ids) >= 2:
            try:
                fit = self._score_model.fit(self._fit_data(ids, "validation", validation))
            except Exception as error:  # noqa: BLE001 - a failed validation fit picks by measured means
                problem = f"the model raised: {type(error).__name__}: {error}"
            else:
                problem = None if fit is not None else problem
        report = pick(
            baseline_id=baseline,
            finalist_ids=finalists,
            validation=validation,
            fit=fit,
            values={c: self._candidates[c].values for c in ids},
            trial_counts={c: self._trial_count(c, "validation") for c in ids},
            criteria=criteria,
            rule=self._final_pick.rule,
            problem=problem,
        )
        self._run.events.append("picked", iteration=self._iteration, report=report)
        return report

    def _finish(
        self,
        stop_reason: StopReason,
        report: ValidationReport,
        test: TestReport | None,
        test_reason: str | None,
        front_test: dict[str, TestReport],
    ) -> OptimizationResult:
        """`effects.json`, the diffs, and `result.json`, from the end phase's reports; returns the result."""
        assert self._run is not None
        run, criteria, baseline = self._run, self._criteria, self._baseline.id
        effects = effects_json(self._fit, criteria, error=self._fit_error)
        run.save_effects(effects)
        views = {
            c: CandidateView.from_candidate(
                self._saved[c], self._baseline, self._space, number=run.number(c), root=self._workdir
            )
            for c in sorted(self._saved, key=run.number)
        }
        champion = views[report.champion_id]
        run.save_diff("best", champion.diff(), champion.changed_values())
        for c in report.front if criteria.is_pareto else []:
            run.save_diff(f"front/{views[c].number}-{c}", views[c].diff(), views[c].changed_values())
        agent_cost: dict[SplitName, Usage] = {split: Usage.zero() for split in ("train", "validation", "test")}
        judge_cost = retry_cost = Usage.zero()
        for records in self._records.values():
            for record in records.values():
                agent_cost[record.split] = agent_cost[record.split] + record.usage
                for usage in record.judge_usage.values():
                    judge_cost = judge_cost + usage
                for attempt in record.previous_attempts:
                    retry_cost = retry_cost + attempt.usage + attempt.judge_usage
        result = OptimizationResult(
            run_id=run.root.name,
            run_dir=run.root,
            baseline_id=baseline,
            candidates=views,
            validation=report,
            test=test,
            test_reason=test_reason,
            front_test=front_test,
            effects=effects,
            iterations=list(self._iterations),
            rollouts_spent=self._budget.spent,
            stop_reason=stop_reason,
            agent_cost=agent_cost,
            search_cost=self._search_cost,
            judge_cost=judge_cost,
            retry_cost=retry_cost,
        )
        run.save_result(result.to_dict())
        return result

    async def _score_split(self, ids: Sequence[str], split: SplitName) -> list[CandidateScores]:
        """Score already-measured candidates on an end-phase split at once (the reserves cover the first passes)."""
        runner = self._runner(self._max_workers)
        cases = list(getattr(self._case_split, split))
        return await self._score_together(
            [(self._candidates[c], False, runner.pending(c, cases, runs=self._runs_per_candidate)) for c in ids],
            split,
        )

    async def _test(self, report: ValidationReport) -> tuple[TestReport | None, str | None, dict[str, TestReport]]:
        """E4: the champion and the baseline on test (and under `Pareto` every validated front member), report only.

        Returns:
            The champion's report (None when skipped), why it or some front members were skipped, and under
            `Pareto` each tested front member's report.
        """
        cases = list(self._case_split.test)
        if not cases:
            return None, "no test cases", {}
        baseline, champion = self._baseline.id, report.champion_id
        front = report.front if self._criteria.is_pareto else []
        runner = self._runner(self._max_workers)
        pending = {
            c: runner.pending(c, cases, runs=self._runs_per_candidate)
            for c in dict.fromkeys([champion, baseline, *front])
        }
        chosen = list(dict.fromkeys([champion, baseline]))
        total = sum(pending[c] for c in chosen)
        if not self._budget.can_afford(total, phase="test"):
            allowance = self._budget.allowance(phase="test")
            return None, f"the test report needs {total} rollouts, {allowance} are left", {}
        skipped = []
        for c in pending:
            if c in chosen:
                continue
            if self._budget.can_afford(total + pending[c], phase="test"):
                chosen.append(c)
                total += pending[c]
            else:
                skipped.append(c)
        scores = dict(
            zip(
                chosen,
                await self._score_together([(self._candidates[c], False, pending[c]) for c in chosen], "test"),
                strict=True,
            )
        )
        reports = {c: self._test_report(c, scores) for c in dict.fromkeys([champion, *front]) if c in scores}
        reason = None
        if skipped:
            numbers = ", ".join(str(self._run.number(c)) for c in skipped)  # type: ignore[union-attr]
            reason = f"the budget couldn't test front members {numbers}"
        return reports[champion], reason, {c: reports[c] for c in front if c in reports}

    def _test_report(self, candidate_id: str, scores: Mapping[str, CandidateScores]) -> TestReport:
        """`candidate_id` against the baseline per test case (paired t-interval at `TEST_LEVEL`); the baseline
        against itself when it is the candidate."""
        baseline = self._baseline.id
        registry = self._criteria.registry
        mine, theirs = scores[candidate_id], scores[baseline]
        difference, ratio = {}, {}
        for metric in self._criteria.goal_metrics:
            a = {case: m[metric] for case, m in mine.metrics_by_case.items() if metric in m}
            b = {case: m[metric] for case, m in theirs.metrics_by_case.items() if metric in m}
            compared = paired_interval(a, b, level=TEST_LEVEL)
            if compared is None:
                continue
            difference[metric] = compared
            low, _ = registry.range(metric)
            ratio[metric] = paired_ratio(a, b, level=TEST_LEVEL) if low is not None and low >= 0 else None
        ids = list(dict.fromkeys([candidate_id, baseline]))
        measured = {
            c: {r.case_name for r in self._records[c].values() if r.split == "test" and self._counted(r)} for c in ids
        }
        return TestReport(
            champion_id=candidate_id,
            baseline_id=baseline,
            objective=paired_interval(mine.objective_by_case, theirs.objective_by_case, level=TEST_LEVEL),
            difference=difference,
            ratio=ratio,
            missing={c: len(set(self._case_split.test) - measured[c]) for c in ids},
            means={c: measured_metrics(scores[c], registry) for c in ids},
        )

    def _rank(self) -> tuple[list[str], dict[str, Assessment]]:
        """Every eligible candidate in ranking order (the fit's estimates, else measured means), and each one's
        assessment."""
        eligible = [c for c in self._candidates if self._train[c].is_eligible]
        assessments = assess(
            eligible,
            fit=self._fit,
            criteria=self._criteria,
            values={c: self._candidates[c].values for c in eligible},
            measured=self._train,
            rule=Mean(),
        )
        return rank(assessments, eligible, is_pareto=self._criteria.is_pareto), assessments

    def _refit(self, *, scored: bool) -> None:
        """Refit on every eligible candidate and write the `fit` event; with nothing new scored, or when the
        model returns None, keep the fit in use.

        A refit that raises is caught: the error goes in the event (a warning once per distinct error) and
        measured means are used until a later refit succeeds, which still gets the last successful fit as
        `previous`.
        """
        assert self._run is not None
        eligible = [c for c in self._candidates if self._train[c].is_eligible]
        event = {"iteration": self._iteration, "candidates": eligible, "flagged": [], "diagnostics": {}}
        if not scored:
            self._run.events.append("fit", **event, is_fitted=False, error=None)
            return
        data = self._fit_data(eligible, "train", self._train)
        try:
            fit = self._score_model.fit(data, previous=self._last_fit)
        except Exception as error:  # noqa: BLE001 - a failed refit never stops the run
            text = f"{type(error).__name__}: {error}"
            if text not in self._fit_errors:
                self._fit_errors.add(text)
                warnings.warn(
                    f"the score model's refit raised ({text}); measured means are used until a refit succeeds",
                    UserWarning,
                    stacklevel=2,
                )
            self._fit, self._fit_error = None, text
            self._run.events.append("fit", **event, is_fitted=False, error=text)
            return
        if fit is None:  # the model declined to fit: the fit in use stays (as a step that scored nothing)
            self._run.events.append("fit", **event, is_fitted=False, error=None)
            return
        self._fit = self._last_fit = fit
        self._fit_error = None
        event.update(flagged=list(fit.flagged), diagnostics=dict(fit.diagnostics))
        self._run.events.append("fit", **event, is_fitted=True, error=None)

    def _fit_data(self, ids: Sequence[str], split: SplitName, scores: Mapping[str, CandidateScores]) -> FitData:
        """What the score model fits: `ids` (the baseline first) with their scores on `split`."""
        return FitData(
            space=self._space,
            text_blocks={
                _block_name(block): [k for k in keys if self._space.param(k).kind == "text"]
                for block, keys in self._blocks
                if any(self._space.param(k).kind == "text" for k in keys)
            },
            candidates={c: self._candidates[c].values for c in ids},
            cases=list(getattr(self._case_split, split)),
            scores={c: scores[c] for c in ids},
            ranges={m: self._criteria.registry.range(m) for m in self._criteria.goal_metrics},
        )

    def _select(self, pool: list[str], assessments: Mapping[str, Assessment]) -> str:
        """The parent `Selection` chooses among `pool` (the step's possible parents, in ranking order).

        Raises:
            ValueError: The strategy named a candidate outside `pool`.
        """
        assert self._run is not None
        chosen = self._selection.choose(
            SelectionContext(
                ranking=pool,
                assessments={c: assessments[c] for c in pool},
                scores={c: s.objective_by_case for c, s in self._train.items()},
                values={c: candidate.values for c, candidate in self._candidates.items()},
                parents={c: candidate.parent_id for c, candidate in self._candidates.items()},
                trial_counts={c: self._trial_count(c) for c in self._candidates},
                fit=self._fit if self._fit is not None and not self._fit.flagged else None,
                iteration=self._iteration,
                history=list(self._history),
                rng=random.Random(f"{self._seed}:{self._iteration}"),
            )
        )
        if chosen not in pool:
            raise ValueError(
                f"{type(self._selection).__name__}.choose returned {chosen!r}, which is not among the step's "
                f"possible parents {pool}"
            )
        self._history.append(chosen)
        self._run.events.append("select", iteration=self._iteration, parent_id=chosen, strategy=self._selection.name)
        return chosen

    def _counted(self, record: TrialRecord) -> bool:
        """Whether a trial counts: completed, or an `agent_failure` scored by `agent_failure_score`."""
        return record.status is TrialStatus.COMPLETED or (
            record.status is TrialStatus.AGENT_FAILURE and self._criteria.agent_failure_score is not None
        )

    def _trial_count(self, candidate_id: str, split: SplitName = "train") -> int:
        """Trials on `split` that count (`_counted`)."""
        return sum(1 for r in self._records.get(candidate_id, {}).values() if r.split == split and self._counted(r))

    async def _step(
        self,
        step: Step,
        n: int,
        parent: Candidate,
        ranking: list[str],
        assessments: Mapping[str, Assessment],
    ) -> IterationRecord:
        """Propose `n` candidates (redrawing a slot that is invalid, a duplicate, or empty), score the new
        ones on train at once, refit, label, and tell the proposers; write `iteration_end`.

        Raises:
            _ProposerFailed: A proposer raised (its `proposer_error` event is written).
        """
        assert self._run is not None
        iteration = self._iteration
        summaries = self._summaries(ranking, assessments)
        deferred = set(step.observe_at_round_end)
        outcomes: list[ProposalOutcome] = []
        accepted: list[_Draw] = []  # in proposal order: new candidates, re-scored ones, and same-step twins
        opened: _Draw | None = None  # an opener repeating a measured candidate: the round opens on it
        need = n
        for redraw in range(self._search.redraws + 1):
            if need <= 0 or opened is not None:
                break
            for draw in await self._draw(step, parent, need, redraw, ranking, assessments, summaries):
                candidate = draw.candidate
                if candidate is None:
                    for block, patch in draw.patches.items():
                        self._run.events.append(
                            "invalid",
                            iteration=iteration,
                            block=block,
                            redraw=redraw,
                            updates=patch.updates,
                            reason=draw.reason,
                        )
                        self._tell(block, patch, None, None, "invalid", draw.reason or "", None)
                    outcomes.append(_outcome(draw, None, None, parent.id, "invalid", None))
                    continue
                existing = self._train.get(candidate.id)
                if existing is not None and existing.is_eligible:
                    a = assessments.get(candidate.id)
                    if step.role == "opener" and deferred & set(draw.patches):
                        for block, patch in draw.patches.items():
                            self._run.events.append(
                                "duplicate",
                                iteration=iteration,
                                block=block,
                                redraw=redraw,
                                candidate_id=candidate.id,
                                patch=patch if block in deferred else None,
                                round=list(step.round or ()),
                            )
                            if block in deferred:
                                self._open_rounds.append(OpenRound(step.round or (), candidate.id, block, patch))
                        opened = draw
                    else:
                        for block, patch in draw.patches.items():
                            self._run.events.append(
                                "duplicate", iteration=iteration, block=block, redraw=redraw, candidate_id=candidate.id
                            )
                            self._tell(
                                block,
                                patch,
                                candidate.id,
                                candidate.values,
                                "duplicate",
                                f"repeats candidate {self._run.number(candidate.id)}",
                                a,
                            )
                    outcomes.append(
                        _outcome(draw, candidate.id, self._run.number(candidate.id), parent.id, "duplicate", a)
                    )
                    continue
                accepted.append(draw)
                need -= 1
        # Save, then score every new candidate of the step at once (a same-step twin is scored once).
        runner = self._runner(self._max_workers)
        to_score: list[tuple[Candidate, bool, int]] = []  # (candidate, rescore, pending trials)
        unscored: set[str] = set()
        first_of: dict[str, _Draw] = {}
        budgeted = 0
        for draw in accepted:
            candidate = draw.candidate
            assert candidate is not None
            if candidate.id in first_of:
                continue
            first_of[candidate.id] = draw
            rescore = candidate.id in self._train
            opener_patch = next((p for b, p in draw.patches.items() if b in deferred), None)
            self._run.save_candidate(
                candidate,
                changed_keys=changed_keys(parent, candidate),
                round=None if step.round is None else list(step.round),
                role=step.role,
                patch=opener_patch,
            )
            self._saved.setdefault(candidate.id, candidate)
            pending = runner.pending(candidate.id, list(self._case_split.train), runs=self._runs_per_candidate)
            if not self._budget.can_afford(budgeted + pending, phase="train"):
                remaining = self._budget.remaining
                self._run.events.append(
                    "unscored",
                    iteration=iteration,
                    candidate_id=candidate.id,
                    pending=pending,
                    remaining=remaining,
                    reserve=self._budget.validation_reserve + self._budget.test_reserve,
                )
                unscored.add(candidate.id)
                continue
            budgeted += pending
            to_score.append((candidate, rescore, pending))
            if opener_patch is not None:
                for block in deferred & set(draw.patches):
                    self._open_rounds.append(OpenRound(step.round or (), candidate.id, block, draw.patches[block]))
        spent_before = self._budget.spent
        results = await self._score_together(to_score, "train")
        for (candidate, _, _), scores in zip(to_score, results, strict=True):
            self._candidates.setdefault(candidate.id, candidate)
            self._train[candidate.id] = scores
        self._refit(scored=bool(to_score))
        ranking, assessments = self._rank()

        # Label each scored candidate, then tell each patch's proposer (an opener's waits for its round's end).
        labels = {}
        for candidate, _, _ in to_score:
            labels[candidate.id] = label_candidate(
                self._train[candidate.id], assessments.get(candidate.id), criteria=self._criteria
            )
            self._run.events.append("label", iteration=iteration, candidate_id=candidate.id, label=labels[candidate.id])
        listening = {r.block for r in self._open_rounds}  # told only their round's best, at its end
        told: set[str] = set()
        for draw in accepted:
            candidate = draw.candidate
            assert candidate is not None
            twin = candidate.id in told
            a = assessments.get(candidate.id)
            if candidate.id in unscored:
                outcome: ObservationOutcome = "unscored"
                reason = "the budget couldn't cover it"
            elif twin:
                outcome, reason = "duplicate", "repeats a candidate of the same step"
            else:
                outcome, reason = labels[candidate.id].outcome, labels[candidate.id].reason
            for block, patch in draw.patches.items():
                if block in deferred and candidate.id not in unscored and not twin:
                    continue  # told the round's best at the round's end
                self._tell(block, patch, candidate.id, candidate.values, outcome, reason, a)
            if not twin and candidate.id not in unscored:
                for other, _ in self._blocks:
                    name = _block_name(other)
                    if name in draw.patches or name in deferred or name in listening:
                        continue
                    self._share(name, candidate, outcome, a)
            told.add(candidate.id)
            outcomes.append(_outcome(draw, candidate.id, self._run.number(candidate.id), parent.id, outcome, a))
        saved = {o.candidate_id for o in outcomes if o.outcome not in ("invalid", "duplicate")}
        produced = {block for o in outcomes if o.candidate_id in saved for block in o.origin}
        if opened is not None:
            produced |= set(opened.patches)
        record = IterationRecord(
            iteration=iteration,
            parent_id=parent.id,
            round=step.round,
            role=step.role,
            blocks=[_block_name(b) for b in step.blocks],
            n_candidates=n,
            proposals=outcomes,
            rollouts_spent=self._budget.spent - spent_before,
        )
        # The best so far, the baseline included when it ranks there (the finalists would exclude it).
        finalists = self._final_pick.n_finalists
        if self._criteria.is_pareto:
            best_after = pareto_front(assessments, ranking)[: max(finalists, PARETO_MIN_FINALISTS)]
        else:
            best_after = ranking[:finalists]
        self._run.events.append(
            "iteration_end",
            iteration=iteration,
            record=record,
            exhausted=[_block_name(b) for b in step.blocks if _block_name(b) not in produced],
            best_after=best_after,
        )
        return record

    async def _draw(
        self,
        step: Step,
        parent: Candidate,
        need: int,
        redraw: int,
        ranking: list[str],
        assessments: Mapping[str, Assessment],
        summaries: list[CandidateSummary],
    ) -> list[_Draw]:
        """Up to `need` proposals on `parent`, each merging one patch of every block of the step.

        One block: its proposer asked once for `need` patches (a concurrent one: `need` calls at once). Several
        blocks: each slot built in turn, structural blocks first (each later block asked for 1 patch on the
        slot's values so far, so it sees the structure chosen for the slot), a block with no key active under
        those values skipped; the slots drawn at once when any of the step's proposers is concurrent.
        """
        fill = self._fill(ranking)
        if len(step.blocks) == 1:
            block = step.blocks[0]
            patches = await self._ask(block, parent, parent, need, redraw, summaries)
            return [self._merge(parent, {_block_name(block): p}, fill) for p in patches[:need]]
        order = sorted(step.blocks, key=lambda b: not self._is_structural(b))
        keys = dict((_block_name(b), k) for b, k in self._blocks)

        async def slot() -> _Draw | None:
            current = parent
            patches: dict[str, Patch] = {}
            for block in order:
                if not any(self._space.is_active(k, current.values) for k in keys[_block_name(block)]):
                    continue
                got = await self._ask(block, parent, current, 1, redraw, summaries)
                if not got:
                    continue
                patches[_block_name(block)] = got[0]
                if self._is_structural(block):
                    draw = self._merge(parent, patches, fill)
                    if draw.candidate is None:
                        return draw
                    current = draw.candidate
            return self._merge(parent, patches, fill) if patches else None

        if any(block.proposer.concurrent for block in step.blocks):
            drawn = await _gather_or_cancel([slot() for _ in range(need)])
        else:
            drawn = [await slot() for _ in range(need)]
        return [d for d in drawn if d is not None]

    async def _ask(
        self,
        block: Block,
        parent: Candidate,
        request_parent: Candidate,
        n: int,
        redraw: int,
        summaries: list[CandidateSummary],
    ) -> list[Patch]:
        """`block`'s patches for one request; a non-concurrent proposer proposes under the step's lock.

        `parent` is the step's parent (its train trials and scores go in the request); `request_parent` is
        what the proposer changes: the parent, or, inside a merged slot, the slot's values so far.

        Raises:
            _ProposerFailed: The proposer raised, returned more than `n` patches, or a patch whose state isn't
                JSON.
        """
        assert self._run is not None
        name = _block_name(block)
        iteration = self._iteration
        run = self._run

        def log(event: str, **fields: Any) -> None:
            if event not in ("screened", "proposer_retry", "proposer_usage"):
                raise ValueError(
                    f"a proposer may log only 'screened', 'proposer_retry', and 'proposer_usage' events, not {event!r}"
                )
            if event == "proposer_usage":  # spend on no returned patch: counted here, once
                self._search_cost = self._search_cost + fields["usage"]
            run.events.append(event, iteration=iteration, block=name, redraw=redraw, **fields)  # type: ignore[arg-type]

        train = list(self._case_split.train)
        cases = {case.name: case for case in self._cases}

        def request(count: int) -> ProposalRequest:
            return ProposalRequest(
                space=self._space,
                keys=dict((_block_name(b), k) for b, k in self._blocks)[name],
                parent=request_parent,
                trials=[r for r in self._records[parent.id].values() if r.split == "train"],
                scores=self._train[parent.id],
                inputs={c: cases[c].input for c in train},
                # A fit with a flagged metric isn't used for decisions, so proposers don't get it either.
                fit=self._fit if self._fit is not None and not self._fit.flagged else None,
                candidates=summaries,
                n=count,
                iteration=iteration,
                seed=self._seed,
                redraw=redraw,
                criteria=self._criteria,
                log=log,
            )

        try:
            if block.proposer.concurrent:
                batches = [
                    list(b) for b in await _gather_or_cancel([block.proposer.propose(request(1)) for _ in range(n)])
                ]
                asked = 1
            else:
                async with self._propose_lock:
                    batches = [list(await block.proposer.propose(request(n)))]
                asked = n
            # Every patch must be observed once (proposers count the ones still pending), so none is dropped.
            for batch in batches:
                if len(batch) > asked:
                    raise ValueError(
                        f"{type(block.proposer).__name__} (block {name!r}) returned {len(batch)} patches for a "
                        f"request of {asked}"
                    )
            patches = [p for batch in batches for p in batch]
            for patch in patches:
                try:
                    json.dumps(patch.state)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"{type(block.proposer).__name__} (block {name!r}) returned a Patch whose state is not "
                        f"JSON-serializable: {error}"
                    ) from error
        except Exception as error:
            run.events.append(
                "proposer_error",
                iteration=iteration,
                block=name,
                redraw=redraw,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise _ProposerFailed(name) from error
        for patch in patches:
            if patch.usage is not None:
                self._search_cost = self._search_cost + patch.usage
        return patches

    def _merge(
        self, parent: Candidate, patches: Mapping[str, Patch], fill: Callable[[str], tuple[Value, str] | None]
    ) -> _Draw:
        """One proposal from each block's patch: the candidate, or the reason it is invalid."""
        keys = dict((_block_name(b), set(k)) for b, k in self._blocks)
        updates: dict[str, Value] = {}
        for block, patch in patches.items():
            outside = sorted(set(patch.updates) - keys[block])
            if outside:
                return _Draw(dict(patches), None, f"block {block!r} proposed keys outside it: {outside}")
            updates.update(patch.updates)
        try:
            candidate = parent.child(
                self._space,
                updates,
                origin=list(patches),
                rationale="\n".join(p.rationale for p in patches.values() if p.rationale),
                fill=fill,
            )
        except ValueError as error:
            return _Draw(dict(patches), None, str(error))
        return _Draw(dict(patches), candidate, None)

    def _fill(self, ranking: list[str]) -> Callable[[str], tuple[Value, str] | None]:
        """A key the parent lacks from the best-ranked eligible candidate that has it (`Candidate.child`)."""

        def fill(key: str) -> tuple[Value, str] | None:
            for candidate_id in ranking:
                values = self._candidates[candidate_id].values
                if key in values:
                    return values[key], candidate_id
            return None

        return fill

    def _is_structural(self, block: Block) -> bool:
        """Whether the block holds a `choice()` selector with 2+ options."""
        keys = dict((_block_name(b), k) for b, k in self._blocks)[_block_name(block)]
        return any((info := self._space.param(k)).kind == "choice" and len(info.options or ()) >= 2 for k in keys)

    def _summaries(self, ranking: list[str], assessments: Mapping[str, Assessment]) -> list[CandidateSummary]:
        """Every scored candidate as a proposer sees it: eligible in ranking order, then the ineligible."""
        assert self._run is not None
        position = {c: i + 1 for i, c in enumerate(ranking)}
        out = []
        for c in [*ranking, *(c for c in self._candidates if c not in position)]:
            candidate, scores, a = self._candidates[c], self._train[c], assessments.get(c)
            out.append(
                CandidateSummary(
                    candidate_id=c,
                    parent_id=candidate.parent_id,
                    origin=list(candidate.origin),
                    values=dict(candidate.values),
                    number=self._run.number(c),
                    measured_objective=scores.objective,
                    measured=measured_metrics(scores, self._criteria.registry),
                    trial_count=self._trial_count(c),
                    failures=0
                    if self._criteria.agent_failure_score is None
                    else sum(
                        1
                        for r in self._records.get(c, {}).values()
                        if r.split == "train" and r.status is TrialStatus.AGENT_FAILURE
                    ),
                    ranking_objective=None if a is None else a.objective,
                    term_values=None if a is None or a.objective is None else self._term_values(c, a),
                    rank=position.get(c),
                    estimates={} if a is None else dict(a.estimates),
                    violations=None if a is None else list(a.violations),
                    is_feasible=None if a is None else a.is_feasible,
                    is_eligible=scores.is_eligible,
                    changed_keys=[]
                    if candidate.parent_id is None
                    else changed_keys(self._candidates[candidate.parent_id], candidate),
                )
            )
        return out

    def _term_values(self, candidate_id: str, a: Assessment) -> list[float] | None:
        """Each objective term's raw value (the resolved estimate, else the measured mean), in objective order."""
        objective = self._criteria.objective
        metrics = [
            d.metric for d in (objective.objectives if isinstance(objective, (WeightedSum, Pareto)) else (objective,))
        ]
        if a.is_from_fit:
            return [round_to_sd(a.estimates[m].mean, a.estimates[m].sd) for m in metrics]
        by_case = self._train[candidate_id].metrics_by_case.values()
        means = [[case[m] for case in by_case if m in case] for m in metrics]
        return None if any(not v for v in means) else [statistics.fmean(v) for v in means]

    def _tell(
        self,
        block: str,
        patch: Patch,
        candidate_id: str | None,
        values: Values | None,
        outcome: ObservationOutcome,
        reason: str,
        assessment: Assessment | None,
        *,
        source_id: str | None = None,
        round_candidate_ids: Sequence[str] | None = None,
    ) -> None:
        """Observe one patch: write the `observe` event, then call its block's proposer."""
        assert self._run is not None
        measured = assessment is not None and outcome != "ineligible"
        observation = Observation(
            patch=patch,
            candidate_id=candidate_id,
            values=None if values is None else dict(values),
            source_id=source_id,
            round_candidate_ids=None if round_candidate_ids is None else list(round_candidate_ids),
            objective=assessment.objective if measured and assessment is not None else None,
            term_values=self._term_values(source_id or candidate_id, assessment)  # type: ignore[arg-type]
            if measured and assessment is not None and assessment.objective is not None
            else None,
            outcome=outcome,
            reason=reason,
            violations=list(assessment.violations) if measured and assessment is not None else None,
        )
        self._run.events.append(
            "observe", iteration=self._iteration, block=block, candidate_id=candidate_id, observation=observation
        )
        self._proposer(block).observe(observation)

    def _share(self, block: str, candidate: Candidate, outcome: ObservationOutcome, a: Assessment | None) -> None:
        """Tell another block's proposer about a scored candidate (`Proposer.evidence`)."""
        assert self._run is not None
        evidence = Evidence(
            candidate_id=candidate.id,
            values=dict(candidate.values),
            origin=list(candidate.origin),
            objective=None if a is None else a.objective,
            term_values=None if a is None or a.objective is None else self._term_values(candidate.id, a),
            violations=None if a is None else list(a.violations),
            outcome=outcome,
        )
        self._run.events.append("evidence", iteration=self._iteration, block=block, evidence=evidence)
        self._proposer(block).evidence(evidence)

    def _proposer(self, block: str) -> Proposer:
        return next(b.proposer for b, _ in self._blocks if _block_name(b) == block)

    def _close_rounds(self, step: Step | None, ranking: list[str], assessments: Mapping[str, Assessment]) -> None:
        """Tell each round's opener, once its round is over, the round's best (innermost rounds first).

        A round is over when the next step lies outside it (or there is none). Its best is the first of the
        ranking among its candidates; an opener that isn't eligible is told `ineligible`.
        """
        assert self._run is not None
        for open_round in sorted(self._open_rounds, key=lambda r: -len(r.round)):
            path = open_round.round
            if step is not None and step.round is not None and tuple(step.round[: len(path)]) == path:
                continue
            self._open_rounds.remove(open_round)
            opener = open_round.opener_id
            assert opener is not None
            # The opener, then what the round's steps saved, in proposal order (saved in the observe event).
            members = list(
                dict.fromkeys(
                    [opener]
                    + [
                        p.candidate_id
                        for record in self._iterations
                        if record.round is not None and tuple(record.round[: len(path)]) == path
                        for p in record.proposals
                        if p.candidate_id is not None and p.outcome not in ("invalid", "duplicate", "unscored")
                    ]
                )
            )
            if opener not in assessments:
                self._tell(
                    open_round.block,
                    open_round.patch,
                    opener,
                    self._candidates[opener].values,
                    "ineligible",
                    "the round's opener has a train case unmeasured",
                    None,
                    round_candidate_ids=members,
                )
                continue
            best = next(c for c in ranking if c in members)
            a = assessments[best]
            self._tell(
                open_round.block,
                open_round.patch,
                opener,
                self._candidates[opener].values,
                "feasible" if a.is_feasible else "infeasible",
                f"the round's best: candidate {self._run.number(best)}",
                a,
                source_id=None if best == opener else best,
                round_candidate_ids=members,
            )

    def _runner(self, max_workers: int) -> TrialRunner:
        assert self._run is not None
        return TrialRunner(
            self._cases,
            self._evaluators,
            self._traces,
            self._run,
            split_of=self._split_of,
            max_workers=max_workers,
            rerun_passes=self._rerun_passes,
        )

    def _setup(self) -> None:
        """Install trace capture and create the run directory (kept when `resume` already opened one)."""
        # Before any Experiment is built, so trials' spans are recorded.
        self._traces = TraceCapture()
        if self._run is None:
            self._run = RunDirectory.create(self._base_dir, self._config(), name=self._name)
        self._split_of: dict[str, SplitName] = {
            name: role
            for role in ("train", "validation", "test")
            for name in getattr(self._case_split, role)  # type: ignore[misc]
        }
        need = self._proposals_per_iteration * len(self._case_split.train) * self._runs_per_candidate
        if self._proposals_per_iteration > 1 and need > self._max_workers:
            warnings.warn(
                f"a step scores {self._proposals_per_iteration} candidates x {len(self._case_split.train)} train "
                f"cases x {self._runs_per_candidate} runs = {need} trials, but max_workers is {self._max_workers}, "
                f"so steps run in batches and take longer; give max_workers={need} to run them at once",
                UserWarning,
                stacklevel=3,
            )

    def _task_for(self, candidate: Candidate) -> TaskFn:
        """The candidate's task, not yet instrumented; file-backed values copied into its workspace first."""
        assert self._run is not None
        workspace = self._run.workspace(candidate.id)
        if self._factory is not None:
            bound = self._factory.bind(candidate.values, workspace=workspace)
            return make_task(bound, self._rollout_fn or default_rollout, self._environment)
        assert self._build_task is not None
        values = candidate.values
        if self._space._copies_files:
            self._space.create_candidate_copy(values, workspace)
        else:
            workspace = None  # type: ignore[assignment]
        return self._build_task(
            BuildContext(values, lambda name: self._space.resolve(name, values, workspace), workspace)
        )

    async def _score(
        self,
        candidate: Candidate,
        split: SplitName,
        *,
        max_workers: int,
        allowance: int,
        rejudge: bool = False,
    ) -> CandidateScores:
        """Score one candidate on one split's cases and count the rollouts; `_score_event` writes its event.

        Every trial is saved as it finishes (`TrialRunner` writes the `trial` events); the caller has checked
        `can_afford(pending(...))`, so the first pass fits and `allowance` caps the re-run passes. Candidates
        scored at once write their `score` events afterwards, in proposal order.
        """
        assert self._run is not None
        case_names = list(getattr(self._case_split, split))
        records, rollouts = await self._runner(max_workers).run(
            candidate.id,
            self._task_for(candidate),
            case_names,
            runs=self._runs_per_candidate,
            allowance=allowance,
            rejudge=rejudge,
        )
        self._budget.spend(rollouts)
        scores = getattr(score(candidate.id, records, self._criteria, self._case_split), split)
        assert scores is not None  # every requested trial has a record on this split
        self._records.setdefault(candidate.id, {}).update({(r.case_name, r.run_index): r for r in records})
        self._rollouts[(candidate.id, split)] = rollouts
        return scores

    async def _score_together(
        self, entries: Sequence[tuple[Candidate, bool, int]], split: SplitName
    ) -> list[CandidateScores]:
        """Score `(candidate, rejudge, pending trials)` entries on `split` at once and write their `score` events
        in entry order.

        Each call gets `max_workers // count` (at least 1) of the concurrency, and its pending trials plus an even
        share of what the phase's allowance leaves for re-runs (earlier entries get the remainder), so the run is
        the same whatever finishes first. The caller has checked the first passes fit.
        """
        if not entries:
            return []
        count = len(entries)
        spare = max(0, self._budget.allowance(phase=split) - sum(pending for _, _, pending in entries))
        workers = max(1, self._max_workers // count)
        results = await _gather_or_cancel(
            [
                self._score(
                    candidate,
                    split,
                    max_workers=workers,
                    allowance=pending + spare // count + (1 if i < spare % count else 0),
                    rejudge=rejudge,
                )
                for i, (candidate, rejudge, pending) in enumerate(entries)
            ]
        )
        for (candidate, _, _), scores in zip(entries, results, strict=True):
            self._score_event(candidate, split, scores)
        return list(results)

    def _score_event(self, candidate: Candidate, split: SplitName, scores: CandidateScores) -> None:
        """The `score` event of a `_score` call."""
        assert self._run is not None
        case_names = list(getattr(self._case_split, split))
        records = [r for r in self._records[candidate.id].values() if r.split == split]
        self._run.events.append(
            "score",
            candidate_id=candidate.id,
            split=split,
            case_names=case_names,
            runs=self._runs_per_candidate,
            rollouts=self._rollouts.pop((candidate.id, split)),
            objective_by_run=scores.objective_by_run,
            statuses=dict(Counter(r.status.value for r in records)),
            unavailable=scores.unavailable,
        )

    async def _score_baseline(self) -> CandidateScores:
        """Save and score the baseline on train; stop the run at setup when it can't be a parent.

        Raises:
            RuntimeError: The baseline isn't eligible, or a goal metric has no value on any train case.
        """
        assert self._run is not None
        baseline = self._baseline = Candidate.baseline(self._space)
        self._run.save_candidate(baseline, changed_keys=[], round=None, role=None)
        scores = await self._score(
            baseline, "train", max_workers=self._max_workers, allowance=self._budget.allowance(phase="train")
        )
        self._score_event(baseline, "train", scores)
        records = list(self._records[baseline.id].values())
        measured = {metric for metrics in scores.metrics_by_case.values() for metric in metrics}
        never = [m for m in self._criteria.goal_metrics if m not in measured]
        counted = [r for r in records if r.status in (TrialStatus.COMPLETED, TrialStatus.AGENT_FAILURE)]
        if counted and never:
            problem = f"goal metric(s) {never} have no value on any train case, so they can't be modeled"
        elif not scores.is_eligible:
            unmeasured = sorted(c for c, runs in scores.objective_by_run.items() if all(v is None for v in runs))
            unmeasured = unmeasured or sorted(
                c
                for c, metrics in scores.metrics_by_case.items()
                if any(m not in metrics for m in self._criteria.goal_metrics)
            )
            problem = f"{len(unmeasured)}/{len(scores.objective_by_run)} train cases unmeasured for the baseline" + (
                f" ({', '.join(unmeasured)})" if unmeasured else ""
            )
        else:
            return scores
        statuses = Counter(r.status.value for r in records)
        message = f"{problem}: trials {', '.join(f'{n} {s}' for s, n in sorted(statuses.items()))}"
        first = next((r for r in records if r.error is not None or r.evaluator_errors), None)
        if first is not None:
            if first.error is not None:
                message += f"; first error ({first.case_name}): {first.error_type}: {first.error}"
            else:
                evaluator, error_type = next(iter(first.evaluator_errors.items()))
                message += f"; first error ({first.case_name}): evaluator {evaluator} raised {error_type}"
        self._run.events.append("stop", reason="setup", message=message)
        raise RuntimeError(message)

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
        (`FinalPick.from_config`), `runs_per_candidate`, `agent_failure_score`, `feasibility_threshold`, seed,
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
            ValueError: The configuration differs from `run.json`, naming the field; or `budget` can't cover
                what is already spent plus the validation reserve (when the pick hasn't run yet).
            RuntimeError: The run stopped at setup (its baseline couldn't be a parent); start a new run.
        """
        run = RunDirectory.open(Path(run_dir))
        saved = run.config
        # The case set first: the split is rebuilt from the saved case lists, which need every case.
        cases = {case.name: case for case in experiment.cases}
        if sorted(name for name in cases if name is not None) != saved["cases"]:
            given = sorted(name for name in cases if name is not None)
            raise ValueError(f"cases changed: {saved['cases']!r} -> {given!r}; start a new run to change them")
        lists = {role: [cases[name] for name in names] for role, names in saved["split"]["cases"].items()}
        optimizer = cls(
            experiment,
            factory,
            rollout_fn=rollout_fn,
            build_task=build_task,
            space=space,
            objective=objective,
            constraints=constraints,
            search=search,
            selection=selection,
            score_model=score_model,
            final_pick=FinalPick.from_config(saved["final_pick"]),
            split=Split(train=lists["train"], validation=lists["validation"], test=lists["test"] or None),
            budget=Budget(**saved["budget"]),
            runs_per_candidate=saved["runs_per_candidate"],
            agent_failure_score=saved["agent_failure_score"],
            feasibility_threshold=saved["feasibility_threshold"],
            environment=environment,
            proposals_per_iteration=saved["proposals_per_iteration"],
            max_workers=saved["max_workers"] if max_workers is None else max_workers,
            rerun_passes=saved["rerun_passes"],
            seed=saved["seed"],
            base_dir=run.root.parent,
            name=saved["name"],
            on_iteration=on_iteration,
        )
        optimizer._workdir = Path(saved["workdir"])  # diff paths stay relative to where the run started
        state = load_state(run, config=optimizer._config())
        if state.stop_reason == "setup":
            raise RuntimeError(f"run {run.run_id} stopped at setup, so there is nothing to resume; start a new run")
        tracker = optimizer._budget
        budget = Budget(**saved["budget"]) if budget is None else budget
        # What the pick still needs: the whole reserve before the finalists exist, else their unrun validation trials.
        if state.validation is not None:
            left = 0
        elif state.finalists is None:
            left = tracker.validation_reserve
        else:
            ids = [state.candidates[0], *state.finalists]
            left = sum(
                1
                for c in ids
                for case in optimizer._case_split.validation
                for index in range(optimizer._runs_per_candidate)
                if run.trial_status(c, index, case) in (None, TrialStatus.INFRASTRUCTURE_INVALID)
            )
        needed = state.spent + left
        if budget.max_rollouts < needed:
            raise ValueError(
                f"Budget(max_rollouts={budget.max_rollouts}) can't cover the {state.spent} rollouts already spent "
                f"plus the {left} the validation pick still needs: give max_rollouts >= {needed}"
            )
        optimizer._budget = BudgetTracker(
            budget, state.spent, validation_reserve=tracker.validation_reserve, test_reserve=tracker.test_reserve
        )
        code = code_version()
        changed = [p for p in ("numpy", "optuna") if saved.get("code", {}).get(p) != code[p]]
        if changed:
            versions = ", ".join(f"{p} {saved.get('code', {}).get(p)} -> {code[p]}" for p in changed)
            warnings.warn(
                f"resuming run {run.run_id} under different library versions ({versions}): a release can change "
                "TPE's, Thompson's, or the fit's draws for a seed, so the resumed part may propose differently",
                UserWarning,
                stacklevel=2,
            )
        run.events.append(
            "resume",
            iteration=state.iteration,
            spent=state.spent,
            budget={"max_rollouts": budget.max_rollouts, "max_iterations": budget.max_iterations},
            code=code,
        )
        optimizer._run, optimizer._resumed = run, state
        return optimizer


def _slots_used(records: Sequence[IterationRecord]) -> int:
    """The candidate slots a run's steps used against `Budget.max_iterations`, charged as a `Nested` round charges
    its own: a step that saved a candidate, the `n_candidates` it was given; a round's opener, its 1 unless it repeated
    a measured candidate (that costs only its own round); a step that saved nothing, none."""
    used = 0
    for record in records:
        if record.role == "opener":
            used += record.n_candidates if record.proposals and record.proposals[-1].outcome != "duplicate" else 0
        elif any(p.outcome not in ("invalid", "duplicate") for p in record.proposals):
            used += record.n_candidates
    return used


async def _gather_or_cancel(awaitables: Sequence[Any]) -> list[Any]:
    """`asyncio.gather`, but the first exception cancels the others and waits for them before it propagates, so
    nothing they would still do (model calls, events) happens after the caller has stopped."""
    tasks = [asyncio.ensure_future(a) for a in awaitables]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class _ProposerFailed(Exception):
    """A proposer raised; its `proposer_error` event is written and the run stops with `proposer_error`."""


@dataclass(frozen=True)
class _Draw:
    """One proposal of a step: each block's patch, and the candidate they make, or why they make none."""

    patches: dict[str, Patch]
    candidate: Candidate | None
    reason: str | None


def _block_name(block: Block) -> str:
    """A resolved block's name (`Search.resolve` names every block)."""
    assert block.name is not None
    return block.name


def _outcome(
    draw: _Draw,
    candidate_id: str | None,
    number: int | None,
    parent_id: str,
    outcome: ObservationOutcome,
    assessment: Assessment | None,
) -> ProposalOutcome:
    """The iteration record's entry for one proposal."""
    updates = [key for patch in draw.patches.values() for key in patch.updates]
    return ProposalOutcome(
        candidate_id=candidate_id,
        number=number,
        parent_id=parent_id,
        origin=list(draw.patches),
        changed_keys=updates if draw.candidate is None else [k for k in draw.candidate.values if k in updates],
        rationale="\n".join(p.rationale for p in draw.patches.values() if p.rationale),
        outcome=outcome,
        objective=None if assessment is None or outcome in ("ineligible", "unscored") else assessment.objective,
    )
