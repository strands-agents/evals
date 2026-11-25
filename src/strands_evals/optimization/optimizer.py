"""The optimizer loop and the end phase: the only module that sees every layer."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import random
import threading
import warnings
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .._async import run_async as _run_in_thread
from ..evaluators.evaluator import Evaluator
from ..experiment import Experiment
from ..types.optimization import (
    SPLIT_NAMES,
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
from .declaration.tunable import BuildContext, TunableFactory, _FactoryArgumentError
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
from .execution.status import classify
from .execution.traces import TraceCapture
from .execution.trials import TrialRunner
from .record.events import closed_events
from .record.resume import OpenRound, ResumeState, load_state
from .record.run_directory import _NAME, RunDirectory, code_version
from .result import (
    CandidateView,
    Grain,
    OptimizationResult,
    _case_rows,
    _iteration_rows,
    _proposal_rows,
    _reports,
    _run_cost,
)
from .score_model.bayesbench import BayesBenchModel
from .score_model.effects import effects_json
from .score_model.protocols import Fit, FitData, ScoreModel
from .score_model.ranking import Assessment, Mean, assess, label_candidate, pareto_front, rank
from .scoring.metrics import USAGE_METRICS, MetricRegistry
from .scoring.objectives import Constraint, Criteria, Maximize, Objective, Pareto, WeightedSum, measured_metrics, score
from .search.blocks import Block, ScheduleContext, Search, Step
from .search.proposer import ProposalRequest, Proposer
from .selection.base import Selection, SelectionContext
from .selection.top_ranked import TopRanked
from .setup.budget import Budget, BudgetTracker, end_phase_reserve
from .setup.splits import Split

_DEFAULT_BASE_DIR = Path("optimization/runs")
_BINDING = threading.Lock()  # one optimizer at a time checks and binds proposers, across threads


class Optimizer:
    """Searches for changes to a declared agent on train, then picks what ships once, on validation.

    Each iteration the schedule plans a step, selection picks a parent, proposers propose, every candidate is
    scored on train, and the score model is refitted. There is no champion during the run: after the search
    stops, the finalists (the top of the ranking, those meeting every constraint with margin first) and the
    baseline are measured on validation, a fresh fit picks the champion (only a candidate meeting every
    constraint ships, else the baseline), and the champion is compared with the baseline on test, reported
    only.

    An optimizer runs once, with proposers of its own: to continue a stopped run, `Optimizer.resume` it with
    a new `Search` built the same way.

    Pass either `factory` (a `TunableFactory`, positionally as in `Optimizer(experiment, create_agent)` or
    by keyword) or `build_task=` with `space=`. `environment=` and `rollout_fn=` work only with `factory=`.
    `objective=None` maximizes the mean of every evaluator's score, as a plain eval run reports it.

    `build_task(ctx)` is called once per candidate and returns the `Experiment` task, `task(case)`.
    Build the agent inside `task`, never in `build_task` itself: an agent shared across cases and
    runs carries conversation state from one trial into the next, so every trial must build its
    own. The factory path enforces this; `build_task` can't.

    A model that makes its own `boto3.Session` (`BedrockModel()` does) reads the AWS config files and
    resolves credentials again on every attempt. With many trials at once that can run the process out of
    file descriptors or overload the credential provider, so build one `boto3.Session` up front and pass
    it to every model (`BedrockModel(boto_session=session)`). A session is not thread-safe: when the agent
    is built in a worker thread (a sync `rollout_fn` or task), build its models under a lock. Each model
    still opens its own connections, so a high `max_workers` may also need a higher open-file limit
    (`ulimit -n`). The credential and "Too many open files" errors this causes are
    `infrastructure_invalid` and re-run, not scored.
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
        on_iteration: Callable[[IterationRecord], Awaitable[None] | None] | None = None,
    ):
        """`base_dir=None` puts runs under `optimization/runs/`, relative to the working directory.

        Args:
            experiment: Supplies the cases and evaluators; it is never run itself.
            factory: The `@tunable` agent factory. Pass this or `build_task`.
            rollout_fn: Runs one case with a fresh agent; defaults to building the agent and invoking it
                on `case.input`.
            build_task: Returns the task for one candidate, from a `BuildContext`; needs `space`.
            space: Declared name -> marker, for `build_task`.
            objective: What to maximize or minimize.
            constraints: Bounds every shipped candidate must meet; during the search they only rank.
            search: Which proposer changes which variables; None means `Search()`. Its proposers serve this
                optimizer only (see `Proposer`).
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
            rerun_passes: Max re-run passes per scoring call: trials still `infrastructure_invalid` are re-run, and
                evaluators that failed are re-judged on the saved output (0 rollouts). The end phase's reserves
                include room for these re-runs (`end_phase_reserve`). A trial whose agent or evaluator failed on
                infrastructure is re-run once a jittered 2.5-5 s has passed since it failed, doubling each pass up
                to 30-60 s; the pass's other trials start at once.
                Retries inside a model call belong to the model client, which the factory configures. For Bedrock,
                botocore's `standard` mode retries connection errors, timeouts, 500/502/503/504 responses, and
                throttling with jittered backoff, but not an error inside a streamed response.
                `AWS_RETRY_MODE=standard` and `AWS_MAX_ATTEMPTS=10` set it for every client that doesn't set
                `retries` itself, judges on the default model included. In code, give `BedrockModel`
                `boto_client_config=botocore.config.Config(retries={"mode": "standard", "total_max_attempts": 10},
                read_timeout=120)`; it sets the 120 s read timeout only when given no `boto_client_config`. More
                attempts also mean longer throttling backoff and more retried timeouts.
            seed: Seeds the split (unless `Split(seed=)` is set), selection, TPE, and Thompson.
            base_dir: Where run directories go; a relative path is relative to the working directory.
            name: The run folder's prefix (lowercase letters, digits, `-`, `_`); for reading only.
            on_iteration: Called with each finished iteration, after its `iteration_end` event; an `async def`
                callback is awaited, on the optimizer's own event loop under `run()` (use `run_async()` for
                a callback that awaits something bound to the caller's loop). An error it raises stops the run
                there and reaches the caller, and `Optimizer.resume` continues from the next iteration (pass
                it a new `Search` built the same way: this one's proposers are bound to this optimizer and may
                hold the run's history). Calling `run_async()` again on this optimizer raises.

        Raises:
            TypeError: Conflicting arguments, e.g. `factory` and `build_task` together.
            ValueError: No evaluators and no objective, case or evaluator names not unique, fewer than one
                train or validation case, a bad `name`, a budget below the baseline plus the end phase's
                reserve, or a proposer in `search` that another optimizer has run with. (Split shares not
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
        evaluator_names = [e.get_name() for e in evaluators]
        repeated = sorted({n for n in evaluator_names if evaluator_names.count(n) > 1})
        if repeated:
            raise ValueError(
                f"evaluator names must be unique; repeated: {repeated}; pass name=... to tell instances of one "
                "evaluator class apart"
            )
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
        self._token = object()  # set on each proposer this optimizer binds (`Proposer._bound_to`)
        self._bind_proposers(claim=False)
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
        validation_reserve = end_phase_reserve(finalists + 1, validation * runs_per_candidate, rerun_passes)
        # Champion and baseline; under Pareto every front member.
        test_arms = (finalists + 1) if self._criteria.is_pareto else 2
        test_reserve = end_phase_reserve(test_arms, test * runs_per_candidate, rerun_passes)
        needed = train * runs_per_candidate + validation_reserve + test_reserve
        if budget.max_rollouts < needed:
            raise ValueError(
                f"Budget(max_rollouts={budget.max_rollouts}) can't cover the baseline ({train} train cases x "
                f"{runs_per_candidate} runs) plus the end phase's reserve ({validation_reserve} validation, "
                f"{test_reserve} test, re-run headroom included): give max_rollouts >= {needed}"
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
        self._started = False  # set by run_async: an Optimizer runs once
        # Candidate id -> (case name, run index) -> its trial record, as scored this process, without its trajectory
        # (`_with_trajectories` reads it back from the transcript).
        self._records: dict[str, dict[tuple[str, int], TrialRecord]] = {}
        # The same keys -> the record's transcript (`{hash, format}`), None when it has no trajectory.
        self._transcripts: dict[str, dict[tuple[str, int], dict[str, str] | None]] = {}
        self._rollouts: dict[tuple[str, SplitName], int] = {}  # a _score call's spend, until its event

    @property
    def run_dir(self) -> Path | None:
        """The run's folder: None until `run()` / `run_async()` creates it, before the baseline is scored; after
        `resume`, the reopened folder."""
        return self._run.root if self._run is not None else None

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

        A Ctrl-C cancels the run, waits for the cancellation to finish, then raises `KeyboardInterrupt`; a second
        Ctrl-C stops the wait, and the cancellation finishes in the background. Trials and proposer calls in flight
        are cancelled, and a cancelled trial isn't saved. Work in a thread can't be stopped. The wait covers such
        work on asyncio's default executor (e.g. a model call that streams in a thread, as Bedrock's do) and a sync
        `environment`'s exit, which may wait for a sync rollout to free a trial thread. A sync rollout itself keeps
        running in the background until its call returns, and the process can't exit before then. The interrupt
        writes no `stop` event, so the run directory is left as a killed run leaves it, and `Optimizer.resume`
        continues it (with a new `Search` built the same way); a new run's directory is the most recently created
        folder under `base_dir`.

        Raises:
            RuntimeError: As `run_async`.
            ProposerError: As `run_async`.
            KeyboardInterrupt: A Ctrl-C, once the run is cancelled.
            ValueError: As `run_async`.
        """
        return _run_in_thread(self.run_async)

    async def run_async(self) -> OptimizationResult:
        """Score the baseline, run the search until it stops, then the end phase.

        The run directory is locked (`RunDirectory.lock`) until this returns or raises: a resumed run before its
        first write, a new run as soon as it is created.

        Returns:
            The champion, the validation and test reports, the effects, and the run's costs.

        Raises:
            RuntimeError: The baseline isn't eligible after its re-run passes (some train case has no
                counted run), no train trial of it completed (every one an `agent_failure`), or a goal
                metric has no value on any train case. A `stop` event with
                reason `"setup"` is written first, and the message names the unmeasured cases or the
                metric, the trial status counts, and the first error. Also: this optimizer has already started a
                run (an Optimizer runs once; continue its run with `Optimizer.resume` and a new `Search`); or,
                before writing to a resumed run, another process or Optimizer is writing it, or wrote to it after
                `resume` read it.
            ProposerError: A proposer raised. The search stops before the end phase, with the
                `proposer_error` event and no `stop` event, so `Optimizer.resume` continues it as it would a
                killed run.
            ValueError: Another optimizer holds one of the search's proposers; checked before the run directory is
                created or a `resume` event written.
        """
        try:
            await self._start()
            resumed = self._resumed
            if resumed is not None and resumed.stop_reason is not None:
                reason: StopReason = resumed.stop_reason  # the search is over: only the end phase is left
            else:
                reason = await self._run_search()
                self._stop(reason)
            return await self._end_phase(reason)
        finally:
            if self._run is not None:
                self._run.unlock()

    async def _start(self) -> None:
        """Set up the run, score the baseline, write the setup's fit (None: one eligible candidate), and tell
        every block's proposer the baseline as evidence.

        After `resume`, the state the event log holds is restored instead; with no closed iteration (killed
        before the first ended) only the setup ran, so it runs again on cached trials. The proposers are bound
        to this optimizer first, before the run directory is created or a `resume` event written, so of two
        optimizers started at once on one `Search` the second raises before spending anything. If the setup
        raises (e.g. the baseline isn't eligible) they are freed again, since nothing has been told to them.

        Raises:
            RuntimeError: This optimizer has already started a run.
            ValueError: Another optimizer holds one of the proposers.
        """
        if self._started:
            assert self._run is not None
            raise RuntimeError(
                f"this Optimizer has already started a run ({self._run.root}), and a second run would write into "
                "the same event log with proposers that hold the first one's history; to continue it, use "
                "Optimizer.resume(run_dir, ...) with a new Search built the same way, or build a new Optimizer to "
                "start over"
            )
        config = self._config()  # before binding: a proposer's `config()` may read its own attributes
        self._bind_proposers(claim=True)
        try:
            self._setup(config)
            self._started = True  # once the run directory exists: a setup that failed before it may be retried
            self._propose_lock = asyncio.Lock()
            if self._resumed is not None:
                assert self._run is not None
                budget = self._budget.budget
                self._run.events.append(
                    "resume",
                    iteration=self._resumed.iteration,
                    spent=self._resumed.spent,
                    budget={"max_rollouts": budget.max_rollouts, "max_iterations": budget.max_iterations},
                    code=code_version(),
                )
            if not (self._resumed is not None and self._resumed.candidates):
                scores = await self._score_baseline()
        except BaseException:
            # Nothing has been told to the proposers yet, so another optimizer may use them; the attribute is
            # removed, so the proposer's attributes are as they were.
            with _BINDING:
                for block, _ in self._blocks:
                    if vars(block.proposer).get("_bound_to") is self._token:
                        object.__delattr__(block.proposer, "_bound_to")
            raise
        if self._resumed is not None and self._resumed.candidates:
            self._restore(self._resumed)
            return
        self._init_state(self._baseline)
        self._candidates[self._baseline.id] = self._baseline
        self._train[self._baseline.id] = scores
        self._refit(scored=True)
        assert self._run is not None
        _, assessments = self._rank()
        assessment = assessments.get(self._baseline.id)
        label = label_candidate(scores, assessment, criteria=self._criteria)
        self._run.events.append("label", iteration=0, candidate_id=self._baseline.id, label=label)
        for block, _ in self._blocks:
            self._share(_block_name(block), self._baseline, label.outcome, assessment)

    def _bind_proposers(self, *, claim: bool) -> None:
        """Check that no other optimizer has run with this one's proposers; with `claim`, then bind them all to
        this one.

        A proposer may keep what it is told, so an instance serves one optimizer. Every block is checked before
        any is bound, so a failed check binds none.

        Raises:
            ValueError: A block's proposer is bound to another optimizer.
        """
        with _BINDING:
            for block, _ in self._blocks:
                # Read from the instance, so a mock's or a subclass's own `_bound_to` attribute isn't taken for one.
                bound_to = vars(block.proposer).get("_bound_to")
                if bound_to is not None and bound_to is not self._token:
                    kind = type(block.proposer).__name__
                    raise ValueError(
                        f"{kind} (block {_block_name(block)!r}) is bound to another Optimizer, which ran or is "
                        f"running with it, so it may hold that run's history; give each Optimizer its own "
                        f"proposers, e.g. a new Search built the same way, with a new {kind} made with the same "
                        f"arguments in this block (Optimizer.resume replays the run's history into them)"
                    )
            if claim:
                for block, _ in self._blocks:
                    object.__setattr__(block.proposer, "_bound_to", self._token)  # a frozen dataclass's too

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
        self._build_errors: set[str] = set()  # infrastructure errors the build check has warned about
        self._saved: dict[str, Candidate] = {baseline.id: baseline}  # every saved candidate, by id
        self._open_rounds: list[OpenRound] = []
        self._iteration = 0

    def _restore(self, state: ResumeState) -> None:
        """Rebuild the search from the closed iterations: candidates and their scores from the saved trials,
        the fits replayed in order (a replayed refit that raises is caught as a live one), and each block's
        `observe` / `evidence` history told to its proposer again."""
        assert self._run is not None
        run = self._run
        self._baseline = run.load_candidate(state.candidates[0])
        self._init_state(self._baseline)
        for candidate_id in state.candidates:
            candidate = run.load_candidate(candidate_id)
            self._saved[candidate_id] = candidate
            # As the closed iterations left them, not as a killed iteration may have replaced them.
            saved = {
                (case, index): saved_trial
                for split in SPLIT_NAMES
                for case in getattr(self._case_split, split)
                for index in range(self._runs_per_candidate)
                if (saved_trial := state.trials.get((candidate_id, index, case))) is not None
            }
            if not saved:
                continue  # defensive: every candidate of a closed iteration has trials
            records = {key: run.record_from(s, load_trajectory=False) for key, s in saved.items()}
            self._records[candidate_id] = records
            self._transcripts[candidate_id] = {key: s["transcript"] for key, s in saved.items()}
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
                try:
                    fit = self._score_model.fit(
                        self._fit_data(replay.ids, "train", self._train), previous=self._last_fit
                    )
                except Exception as error:  # noqa: BLE001 - as a live refit: measured means until one succeeds
                    text = f"{type(error).__name__}: {error}"
                    if text not in self._fit_errors:
                        self._fit_errors.add(text)
                        warnings.warn(
                            f"replaying a refit on resume raised ({text}), though it succeeded in the run; measured"
                            " means are used until a refit succeeds, so the resumed run may differ from the original",
                            UserWarning,
                            stacklevel=2,
                        )
                    self._fit, self._fit_error = None, text
                    continue
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
        self._iteration = state.iteration

    async def _run_search(self) -> StopReason:
        """Run iterations until a stop: budget, iteration cap, or the schedule's None.

        Raises:
            ProposerError: A proposer raised (from its error): the search stops where it is, with no `stop` event.
        """
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
            except _ProposerFailed as failed:
                block, error = failed.args[0], failed.__cause__
                message = (
                    f"the proposer of block {block!r} raised {type(error).__name__}: {error}. The search stopped in "
                    f"iteration {self._iteration}, before the end phase, so no validation or test rollout was spent; "
                    f"once the cause is fixed, continue it with Optimizer.resume({str(self._run.root)!r}, ...), "
                    "passing a newly built search (resume tells its proposers the run's history)"
                )
                steps = _slots_used(self._iterations) // self._proposals_per_iteration
                if steps:  # a cap the closed iterations have used: resume stops the search at once
                    message += (
                        f"; to end it with what it found instead, also pass budget=Budget(max_rollouts="
                        f"{budget.max_rollouts}, max_iterations={steps})"
                    )
                raise ProposerError(message, block, self._run.root) from error
            self._iterations.append(record)
            if self._on_iteration is not None:
                outcome = self._on_iteration(record)
                if inspect.isawaitable(outcome):
                    await outcome

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
        }
        self._run.events.append(
            "stop",
            iteration=self._iteration,
            reason=reason,
            message=messages.get(reason, ""),
            max_rollouts=self._budget.budget.max_rollouts,
        )

    async def _end_phase(self, stop_reason: StopReason) -> OptimizationResult:
        """Finalists from the search ranking (E1), measured with the baseline on validation (E2), the champion
        picked on a fresh validation fit (E3), the test report (E4); then `effects.json`, the diffs, and
        `result.json`.

        After `resume`, each of E1, E3 (with E2), and E4 that the event log already holds is reused as saved, so a
        finished run only rewrites its files. A test report that measured nothing for a reason a resume can
        change (infrastructure failures, or a budget that can't afford it) writes no `test` event, so `resume`
        runs it again.
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
            finalists = choose_finalists(
                ranking,
                baseline,
                n,
                assessments=assessments,
                train=self._train,
                criteria=criteria,
                n_validation_cases=len(self._case_split.validation),
                front=front,
            )
            run.events.append("finalists", iteration=self._iteration, candidate_ids=finalists)
        if resumed is not None and resumed.validation is not None:
            report = resumed.validation
        else:
            report = await self._pick(finalists)
        if resumed is not None and resumed.is_test_done:
            test, test_reason, front_test = resumed.test, resumed.test_reason, resumed.front_test
            test_failed_ids = resumed.test_failed_ids
        else:
            test, test_reason, front_test, test_failed_ids, final = await self._test(report)
            if final:
                run.events.append(
                    "test",
                    iteration=self._iteration,
                    report=test,
                    reason=test_reason,
                    front=front_test,
                    failed=test_failed_ids,
                )
        return self._finish(stop_reason, finalists, report, test, test_reason, front_test, test_failed_ids)

    async def _pick(self, finalists: Sequence[str]) -> ValidationReport:
        """E2 and E3: the finalists and the baseline on validation, then the pick on a fresh validation fit."""
        assert self._run is not None
        criteria, baseline = self._criteria, self._baseline.id
        ids = [baseline, *finalists]
        validation = dict(zip(ids, await self._score_split(ids, "validation"), strict=True))
        completed = {c: self._completed_count(c, "validation") for c in ids}
        # A fresh fit on the validation runs only, never mixed with the search fit (no previous fit). A candidate
        # with no completed validation trial isn't assessed, so it stays out of the fit too: its failure values
        # would move the intercept, the noise, and the case effects the others' estimates use (and a baseline's
        # would set an unbounded metric's unit).
        fitted = [c for c in ids if completed[c] > 0]
        fit: Fit | None = None
        problem: str | None = "fewer than 2 eligible candidates" + (
            "" if len(fitted) == len(ids) else " with a completed validation trial"
        )
        if sum(validation[c].is_eligible for c in fitted) >= 2:
            try:
                fit = self._score_model.fit(self._fit_data(fitted, "validation", validation))
            except Exception as error:  # noqa: BLE001 - a failed validation fit picks by measured means
                problem = f"the model raised: {type(error).__name__}: {error}"
            else:
                problem = None if fit is not None else "the model returned no fit"
                if fit is not None and fit.flagged:
                    # As `assess` decides it: every goal metric flagged (or a flagged name that isn't one) ->
                    # measured means for all of them; else only the flagged ones, the rest by the fit.
                    problem = f"flagged: {', '.join(fit.flagged)}"
                    if set(fit.flagged) < set(criteria.goal_metrics):
                        problem += " (by adjusted means; the other metrics by the fit)"
        report = pick(
            baseline_id=baseline,
            finalist_ids=finalists,
            validation=validation,
            fit=fit,
            values={c: self._candidates[c].values for c in ids},
            trial_counts={c: self._trial_count(c, "validation") for c in ids},
            completed=completed,
            criteria=criteria,
            rule=self._final_pick.rule,
            problem=problem,
        )
        self._run.events.append("picked", iteration=self._iteration, report=report)
        return report

    def _finish(
        self,
        stop_reason: StopReason,
        finalists: Sequence[str],
        report: ValidationReport,
        test: TestReport | None,
        test_reason: str | None,
        front_test: dict[str, TestReport],
        test_failed_ids: Sequence[str],
    ) -> OptimizationResult:
        """`effects.json`, the diffs, and `result.json`, from the end phase's reports; returns the result, as
        `result.json` reloads it (so the returned result and a reloaded one are equal).

        The rows are built from the closed events, so a resumed run's rows cover the whole run and a killed
        iteration that resume re-ran shows once.
        """
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
        # Every saved trial, as `spent` counts them, and every event: after a resume, a killed iteration's too.
        cost = _run_cost(run._trials.values(), run.events.read(), blocks=[_block_name(b) for b, _ in self._blocks])
        result = OptimizationResult(
            run_id=run.root.name,
            run_dir=run.root,
            baseline_id=baseline,
            candidates=views,
            validation=report,
            test=test,
            test_reason=test_reason,
            front_test=front_test,
            test_failed_ids=list(test_failed_ids),
            effects=effects,
            objective=str(criteria.objective),
            constraints=[str(c) for c in criteria.constraints],
            feasibility_threshold=criteria.feasibility_threshold,
            proposers={_block_name(block): type(block.proposer).__name__ for block, _ in self._blocks},
            iterations=list(self._iterations),
            rollouts_spent=self._budget.spent,
            # The budget the search stopped under: resuming a finished run with a larger one runs nothing more.
            max_rollouts=self._budget.budget.max_rollouts
            if self._resumed is None or self._resumed.stop_max_rollouts is None
            else self._resumed.stop_max_rollouts,
            stop_reason=stop_reason,
            cost=cost,
            reports=_reports(
                self._records,
                self._transcripts,
                cases=self._cases,
                evaluators=self._evaluators,
                split=self._case_split,
            ),
            rows=self._rows(views, finalists, report),
        )
        data = result.to_dict()
        run.save_result(data)
        return OptimizationResult.from_dict(data)

    def _rows(
        self, views: Mapping[str, CandidateView], finalists: Sequence[str], report: ValidationReport
    ) -> dict[Grain, list[dict[str, Any]]]:
        """Every grain's result rows, from the closed events and the trial records."""
        assert self._run is not None
        events = closed_events(self._run.events.read())
        ranking, assessments = self._rank()
        criteria = self._criteria
        return {
            "proposal": _proposal_rows(
                events,
                iterations=self._iterations,
                candidates=views,
                baseline_id=self._baseline.id,
                final={c: (assessments[c].objective, i + 1) for i, c in enumerate(ranking)},
                trial_counts={c: self._trial_count(c) for c in self._candidates},
                failures={c: s.failures for c, s in self._train.items()},
                metrics=[*criteria.registry.evaluators, *USAGE_METRICS],
                goal_metrics=list(criteria.goal_metrics),
                constraints=[str(c) for c in criteria.constraints],
                validation=report,
                finalists=finalists,
                is_pareto=criteria.is_pareto,
            ),
            "iteration": _iteration_rows(events, iterations=self._iterations, candidates=views),
            "case": _case_rows(
                events,
                self._records,
                self._transcripts,
                candidates=views,
                split=self._case_split,
                evaluators=[e.get_name() for e in self._evaluators],
            ),
        }

    async def _score_split(self, ids: Sequence[str], split: SplitName) -> list[CandidateScores]:
        """Score already-measured candidates on an end-phase split at once (the reserves cover the first passes
        and each arm's re-run headroom)."""
        runner = self._runner()
        cases = list(getattr(self._case_split, split))
        return await self._score_together(
            [(self._candidates[c], False, runner.pending(c, cases, runs=self._runs_per_candidate)) for c in ids],
            split,
        )

    async def _test(
        self, report: ValidationReport
    ) -> tuple[TestReport | None, str | None, dict[str, TestReport], list[str], bool]:
        """E4: the champion and the baseline on test (and under `Pareto` every validated front member), report only.

        Returns:
            The champion's report (None when skipped: no test cases, the budget, or no completed test trial
            of the champion or the baseline), why it or some front members were skipped, under `Pareto`
            each tested front member's report, the tested candidates with no completed test trial, and whether
            the outcome is final. It isn't when the report was skipped for the budget, or when each of the
            champion and the baseline with no completed test trial still has one left `infrastructure_invalid`:
            a resume can then still produce the report, and the reason names the `max_rollouts` that affords
            every arm's remaining trials.
        """
        cases = list(self._case_split.test)
        if not cases:
            return None, "no test cases", {}, [], True
        baseline, champion = self._baseline.id, report.champion_id
        front = report.front if self._criteria.is_pareto else []
        runner = self._runner()
        pending = {
            c: runner.pending(c, cases, runs=self._runs_per_candidate)
            for c in dict.fromkeys([champion, baseline, *front])
        }
        chosen = list(dict.fromkeys([champion, baseline]))
        total = sum(pending[c] for c in chosen)
        if not self._budget.can_afford(total, phase="test"):
            allowance = self._budget.allowance(phase="test")
            needed = self._budget.spent + sum(pending.values())
            reason = (
                f"the test report needs {total} rollouts, {allowance} are left; resume with max_rollouts >= "
                f"{needed} to run it" + (" with every front member" if len(pending) > len(chosen) else "")
            )
            return None, reason, {}, [], False
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
        assert self._run is not None
        # An arm with no completed test trial measured nothing: its scores would be the failure value alone.
        failed = [c for c in chosen if self._completed_count(c, "test") == 0]
        if champion in failed or baseline in failed:
            arm = "the baseline" if baseline in failed else "the champion"
            # Trials left `infrastructure_invalid` are what a resume runs again (`TrialRunner.pending`).
            left = {c: runner.pending(c, cases, runs=self._runs_per_candidate) for c in pending}
            if all(left[c] for c in (champion, baseline) if c in failed):
                n = left[baseline if baseline in failed else champion]
                needed = self._budget.spent + sum(left.values())
                reason = (
                    f"every test trial of {arm} failed, {n} on infrastructure; resume with max_rollouts >= "
                    f"{needed} to run them again"
                )
                return None, reason, {}, failed, False
            return None, f"every test trial of {arm} failed, so there is nothing to compare", {}, failed, True
        reports = {
            c: self._test_report(c, scores)
            for c in dict.fromkeys([champion, *front])
            if c in scores and c not in failed
        }
        reasons = []
        if skipped:
            reasons.append(
                f"the budget couldn't test front members {', '.join(str(self._run.number(c)) for c in skipped)}"
            )
        if failed:
            numbers = ", ".join(str(self._run.number(c)) for c in failed)
            reasons.append(f"every test trial of front members {numbers} failed")
        front_test = {c: reports[c] for c in front if c in reports}
        return reports[champion], "; ".join(reasons) or None, front_test, failed, True

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
        """Every eligible candidate in ranking order (the fit's estimates, else measured means, metric by metric),
        and each one's assessment."""
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
        """What the score model fits: `ids` (the baseline first, when among them) with their scores on `split`."""
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
                fit=self._fit,
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

    def _completed_count(self, candidate_id: str, split: SplitName) -> int:
        """Trials on `split` that completed (an `agent_failure` scored by `agent_failure_score` isn't one)."""
        return sum(
            1
            for r in self._records.get(candidate_id, {}).values()
            if r.split == split and r.status is TrialStatus.COMPLETED
        )

    def _with_trajectories(
        self,
        records: Mapping[tuple[str, int], TrialRecord],
        transcripts: Mapping[tuple[str, int], dict[str, str] | None],
    ) -> dict[tuple[str, int], TrialRecord]:
        """`records` (a candidate's, from `_records`), each with its trajectory read from its transcript."""
        assert self._run is not None
        return {
            key: record
            if (transcript := transcripts[key]) is None
            else record.model_copy(update={"trajectory": self._run.load_transcript(transcript)})
            for key, record in records.items()
        }

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

        The step's proposals are written to a `draw` event before any is saved or scored. A resumed run
        replays the killed iteration's (`_recorded_draws`) instead of asking its proposers again, so the
        candidates scored before the kill come back with their trials.

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
        recorded = self._recorded_draws(step, parent, n)
        # Read back once for the whole step, only if a proposer reads them, from the records as the step starts.
        held = {k: r for k, r in self._records[parent.id].items() if r.split == "train"}
        transcripts = dict(self._transcripts[parent.id])
        parent_trials = _ParentTrials(lambda: list(self._with_trajectories(held, transcripts).values()))
        drawn: list[list[_Draw]] = []  # per redraw, asked or replayed: the step's `draw` event
        for redraw in range(self._search.redraws + 1):
            if need <= 0 or opened is not None:
                break
            if recorded is not None and redraw < len(recorded):  # what the killed attempt drew, not asked again
                fill = self._fill(ranking)
                drawn.append([self._merge(parent, patches, fill) for patches in recorded[redraw]])
            else:
                drawn.append(
                    await self._draw(step, parent, need, redraw, ranking, assessments, summaries, parent_trials)
                )
            for draw in drawn[-1]:
                if need <= 0:
                    # A replayed batch can hold more than the step still needs: the step is narrower on resume,
                    # or a draw the killed attempt rejected is accepted now. Live batches never do.
                    break
                candidate = draw.candidate
                if (
                    candidate is not None
                    and candidate.id not in self._train
                    and all(d.candidate is None or d.candidate.id != candidate.id for d in accepted)
                ):
                    error = await self._build_error(candidate)
                    if error is not None:
                        draw = _Draw(draw.patches, None, error)
                        candidate = None
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
                # An ineligible candidate is scored again only if a re-score can change it: a train trial still
                # to run (missing or `infrastructure_invalid`) or with a failed evaluator to judge again. One
                # with neither is a duplicate too, else re-proposing it would save a candidate for 0 rollouts.
                trials = [r for r in self._records.get(candidate.id, {}).values() if r.split == "train"]
                settled = len(trials) == len(self._case_split.train) * self._runs_per_candidate and not any(
                    r.status is TrialStatus.INFRASTRUCTURE_INVALID or r.evaluator_errors for r in trials
                )
                if existing is not None and (existing.is_eligible or settled):
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
        # Recorded before anything is saved or scored, so a resume after a kill replays these proposals.
        self._run.events.append(
            "draw",
            iteration=iteration,
            parent_id=parent.id,
            blocks=[_block_name(b) for b in step.blocks],
            n_candidates=n,
            draws=[[d.patches for d in batch] for batch in drawn],
        )
        # Save, then score every new candidate of the step at once (a same-step twin is scored once).
        runner = self._runner()
        to_score: list[tuple[Candidate, bool, int]] = []  # (candidate, rescore, pending trials)
        first_of: dict[str, _Draw] = {}
        budgeted = 0
        for draw in accepted:
            candidate = draw.candidate
            assert candidate is not None
            if candidate.id in first_of:
                continue
            first_of[candidate.id] = draw
            rescore = candidate.id in self._train
            deferred_patches = {b: p for b, p in draw.patches.items() if b in deferred}
            self._run.save_candidate(
                candidate,
                changed_keys=changed_keys(parent.values, candidate.values),
                round=None if step.round is None else list(step.round),
                role=step.role,
                deferred_patches=deferred_patches or None,
            )
            self._saved.setdefault(candidate.id, candidate)
            # The step's n was cut to what the search allowance covers and `pending` is at most one full pass,
            # so every candidate of the step is affordable.
            pending = runner.pending(candidate.id, list(self._case_split.train), runs=self._runs_per_candidate)
            budgeted += pending
            to_score.append((candidate, rescore, pending))
            for block, patch in deferred_patches.items():
                self._open_rounds.append(OpenRound(step.round or (), candidate.id, block, patch))
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
        # A re-score that ran no agent (only failed evaluators judged again) and left the candidate ineligible
        # changed nothing, so it saves nothing: else an evaluator that always fails on this output would be
        # re-judged on every repeat and the schedule's stop on a step that saved nothing would never fire.
        judged_only = {c.id for c, rescore, pending in to_score if rescore and pending == 0}
        told: set[str] = set()
        for draw in accepted:
            candidate = draw.candidate
            assert candidate is not None
            twin = candidate.id in told
            a = assessments.get(candidate.id)
            outcome: ObservationOutcome
            if twin:
                outcome, reason = "duplicate", "repeats a candidate of the same step"
            elif candidate.id in judged_only and labels[candidate.id].outcome == "ineligible":
                outcome = "duplicate"
                reason = f"repeats candidate {self._run.number(candidate.id)}, still unmeasured after its re-judge"
            else:
                outcome, reason = labels[candidate.id].outcome, labels[candidate.id].reason
            for block, patch in draw.patches.items():
                if block in deferred and not twin:
                    continue  # told the round's best at the round's end
                self._tell(block, patch, candidate.id, candidate.values, outcome, reason, a)
            if not twin:
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
            best_objective=assessments[best_after[0]].objective if best_after else None,
        )
        return record

    def _recorded_draws(self, step: Step, parent: Candidate, n: int) -> list[list[dict[str, Patch]]] | None:
        """The killed iteration's proposals per redraw, from its `draw` event, when resume runs that iteration
        again with the same parent and blocks and at most as many candidates; else None, and the proposers are
        asked. A wider step (a larger budget near the cap) asks again, as an uninterrupted run would have."""
        recorded = None if self._resumed is None else self._resumed.draw
        if (
            recorded is None
            or recorded.iteration != self._iteration
            or recorded.parent_id != parent.id
            or recorded.blocks != [_block_name(b) for b in step.blocks]
            or n > recorded.n_candidates
        ):
            return None
        return recorded.draws

    async def _draw(
        self,
        step: Step,
        parent: Candidate,
        need: int,
        redraw: int,
        ranking: list[str],
        assessments: Mapping[str, Assessment],
        summaries: list[CandidateSummary],
        trials: Sequence[TrialRecord],
    ) -> list[_Draw]:
        """Up to `need` proposals on `parent`, each merging one patch of every block of the step.

        One block: its proposer asked once for `need` patches (a concurrent one: `need` calls at once). Several
        blocks: each slot built in turn, structural blocks first (each later block asked for 1 patch on the
        slot's values so far, so it sees the structure chosen for the slot), a block with no key active under
        those values skipped; the slots drawn at once when any of the step's proposers is concurrent. Every
        request gets `trials`, the parent's train trials.
        """
        fill = self._fill(ranking)
        if len(step.blocks) == 1:
            block = step.blocks[0]
            patches = await self._ask(block, parent, parent, need, redraw, summaries, trials)
            return [self._merge(parent, {_block_name(block): p}, fill) for p in patches[:need]]
        order = sorted(step.blocks, key=lambda b: not self._is_structural(b))
        keys = dict((_block_name(b), k) for b, k in self._blocks)

        async def slot() -> _Draw | None:
            """One candidate of the step: each block's patch on the values the blocks before it chose.

            Blocks are asked in `order` (structural first) for 1 patch each, so a later block proposes under the
            structure this slot chose; a block with no key active under those values is skipped, as is one that
            returns nothing. A structural patch is merged at once, so `current` carries it to the next block. An
            invalid merge returns straight away, leaving the rest of the blocks unasked; no patch at all -> None.
            """
            current = parent
            patches: dict[str, Patch] = {}
            for block in order:
                if not any(self._space.is_active(k, current.values) for k in keys[_block_name(block)]):
                    continue
                got = await self._ask(block, parent, current, 1, redraw, summaries, trials)
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
        trials: Sequence[TrialRecord],
    ) -> list[Patch]:
        """`block`'s patches for one request; a non-concurrent proposer proposes under the step's lock.

        `parent` is the step's parent (its train scores go in the request, and `trials` are its train trials
        with their trajectories); `request_parent` is what the proposer changes: the parent, or, inside a
        merged slot, the slot's values so far.

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
            if event == "proposer_usage" and not isinstance(fields.get("usage"), Usage):
                raise TypeError(f"a 'proposer_usage' event needs usage= a Usage, not {fields.get('usage')!r}")
            run.events.append(event, iteration=iteration, block=name, redraw=redraw, **fields)  # type: ignore[arg-type]

        train = list(self._case_split.train)
        cases = {case.name: case for case in self._cases}

        def request(count: int) -> ProposalRequest:
            return ProposalRequest(
                space=self._space,
                keys=dict((_block_name(b), k) for b, k in self._blocks)[name],
                parent=request_parent,
                trials=trials,
                scores=self._train[parent.id],
                inputs={c: cases[c].input for c in train},
                fit=self._fit,
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
        # Logged now, not only in the observe events, so a killed iteration's proposals still count on resume.
        spent = [patch.usage for patch in patches if patch.usage is not None]
        if spent:
            usage = Usage.zero()
            for patch_usage in spent:
                usage = usage + patch_usage
            run.events.append("proposer_usage", iteration=iteration, block=name, redraw=redraw, usage=usage)
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
                    term_values=None if a is None or a.objective is None else self._term_values(a),
                    rank=position.get(c),
                    estimates={} if a is None else dict(a.estimates),
                    adjusted_means={} if a is None else dict(a.adjusted_means),
                    violations=None if a is None else list(a.violations),
                    is_feasible=None if a is None else a.is_feasible,
                    is_eligible=scores.is_eligible,
                    changed_keys=[]
                    if candidate.parent_id is None
                    else changed_keys(self._candidates[candidate.parent_id].values, candidate.values),
                )
            )
        return out

    def _term_values(self, a: Assessment) -> list[float] | None:
        """`a.terms` as raw values (each `Minimize` term negated back), in objective order; None when `a.terms` is."""
        if a.terms is None:
            return None
        objective = self._criteria.objective
        parts = objective.objectives if isinstance(objective, (WeightedSum, Pareto)) else (objective,)
        return [t if isinstance(d, Maximize) else -t for d, t in zip(parts, a.terms, strict=True)]

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
            term_values=self._term_values(assessment)
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
        """Tell a block's proposer about a scored candidate it didn't propose (`Proposer.evidence`)."""
        assert self._run is not None
        evidence = Evidence(
            candidate_id=candidate.id,
            values=dict(candidate.values),
            origin=list(candidate.origin),
            objective=None if a is None else a.objective,
            term_values=None if a is None or a.objective is None else self._term_values(a),
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
            # The opener, then what the round's steps saved, in proposal order (saved in the observe event).
            members = list(
                dict.fromkeys(
                    [opener]
                    + [
                        p.candidate_id
                        for record in self._iterations
                        if record.round is not None and tuple(record.round[: len(path)]) == path
                        for p in record.proposals
                        if p.candidate_id is not None and p.outcome not in ("invalid", "duplicate")
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

    def _runner(self) -> TrialRunner:
        assert self._run is not None
        return TrialRunner(
            self._cases,
            self._evaluators,
            self._traces,
            self._run,
            split_of=self._split_of,
            max_workers=self._max_workers,
            rerun_passes=self._rerun_passes,
        )

    def _setup(self, config: dict[str, Any] | None = None) -> None:
        """Install trace capture, create the run directory (kept when `resume` already opened one) with `config`
        (None: `_config()`) as its `run.json`, and lock it."""
        # Before any Experiment is built, so trials' spans are recorded.
        self._traces = TraceCapture()
        if self._run is None:
            self._run = RunDirectory.create(
                self._base_dir, config if config is not None else self._config(), name=self._name
            )
        self._run.lock()
        self._split_of: dict[str, SplitName] = {
            name: role for role in SPLIT_NAMES for name in getattr(self._case_split, role)
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

    async def _build_error(self, candidate: Candidate) -> str | None:
        """Why the candidate's agent can't be built, else None: the factory is called once, before any rollout.

        A candidate whose factory raises (a constructor rejecting a combination of values) is invalid, redrawn
        at no cost, instead of every trial being an `agent_failure` scored at the worst end. Built with no
        resources and no call kwargs, so a factory that needs a resource from `environment` or a kwarg only the
        rollout passes can't be built here and isn't checked (its trials run as usual). Neither is a factory that
        fails on infrastructure (`classify`: expired credentials, a dropped connection): that is no fault of the
        values, so the proposer isn't told they're invalid; the trials run, and are re-run, as in any outage (a
        warning once per distinct error). On the `build_task` path the agent is built inside the task, so nothing
        is checked.
        """
        if self._factory is None:
            return None
        try:
            # bind() copies path-backed files into a temp folder: in a worker thread, so other trials keep running.
            built = (await asyncio.to_thread(self._factory.bind, candidate.values))()
            if inspect.isawaitable(built):
                await built
        except _FactoryArgumentError:
            return None
        except Exception as error:  # noqa: BLE001 - any other failure to build makes the candidate invalid
            if classify(error) is TrialStatus.INFRASTRUCTURE_INVALID:
                text = f"{type(error).__name__}: {error}"
                if text not in self._build_errors:
                    self._build_errors.add(text)
                    warnings.warn(
                        f"the agent couldn't be built ({text}); its trials run, and are re-run, as "
                        "infrastructure failures",
                        UserWarning,
                        stacklevel=2,
                    )
                return None
            return f"the agent can't be built: {type(error).__name__}: {error}"
        return None

    async def _task_for(self, candidate: Candidate) -> TaskFn:
        """The candidate's task, not yet instrumented; file-backed values copied into its workspace first.

        When a value reaches the agent as a path (`as_path=True`, `SkillLibrary`), the copy is made at once, in a
        worker thread so concurrent trials keep running. Otherwise, on the `build_task` path, the first read of
        `BuildContext.workspace` makes it, in the reading thread, so a `build_task` that never reads it copies
        nothing; later reads don't wait.
        """
        assert self._run is not None
        workspace = self._run.workspace(candidate.id)
        values, space = candidate.values, self._space
        if space._copies_files:
            await asyncio.to_thread(space.create_candidate_copy, values, workspace)
        if self._factory is not None:
            bound = self._factory.bind(values, workspace=workspace)  # the copy exists: bind returns at once
            return make_task(bound, self._rollout_fn or default_rollout, self._environment)
        assert self._build_task is not None

        lock = threading.Lock()
        copied = False

        def copy() -> Path | None:
            nonlocal copied
            if not space._files:
                return None
            if not copied:
                with lock:  # one copy per candidate, however many trials read it first
                    if not copied:
                        space.create_candidate_copy(values, workspace)
                        copied = True
            return workspace

        return self._build_task(BuildContext(values, lambda name: space.resolve(name, values, workspace), copy))

    async def _score(
        self,
        candidate: Candidate,
        split: SplitName,
        *,
        allowance: int,
        rejudge: bool = False,
        limiter: asyncio.Semaphore | None = None,
    ) -> CandidateScores:
        """Score one candidate on one split's cases and count the rollouts; `_score_event` writes its event.

        Every trial is saved as it finishes (`TrialRunner` writes the `trial` events); the caller guarantees the
        first pass fits (an explicit `can_afford`, the step's `n` cut, or the end phase's reserves), and
        `allowance` caps the re-run passes. Candidates
        scored at once write their `score` events afterwards, in proposal order.
        """
        assert self._run is not None
        case_names = list(getattr(self._case_split, split))
        records, rollouts = await self._runner().run(
            candidate.id,
            await self._task_for(candidate),
            case_names,
            runs=self._runs_per_candidate,
            allowance=allowance,
            rejudge=rejudge,
            limiter=limiter,
        )
        self._budget.spend(rollouts)
        scores = getattr(score(candidate.id, records, self._criteria, self._case_split), split)
        assert scores is not None  # every requested trial has a record on this split
        held = self._records.setdefault(candidate.id, {})
        transcripts = self._transcripts.setdefault(candidate.id, {})
        for record in records:  # each was just saved or read from the run directory, so its index has it
            key = (record.case_name, record.run_index)
            held[key] = record.model_copy(update={"trajectory": None})
            transcripts[key] = self._run.transcript(candidate.id, record.run_index, record.case_name)
        self._rollouts[(candidate.id, split)] = rollouts
        return scores

    async def _score_together(
        self, entries: Sequence[tuple[Candidate, bool, int]], split: SplitName
    ) -> list[CandidateScores]:
        """Score `(candidate, rejudge, pending trials)` entries on `split` at once and write their `score` events
        in entry order.

        The calls share one `max_workers` semaphore, so that many trials run at once however many candidates
        the step has, and each gets its pending trials plus an even share of what the phase's allowance leaves
        for re-runs (earlier entries get the remainder), so the run is the same whatever finishes first. The
        caller guarantees the first passes fit.
        """
        if not entries:
            return []
        count = len(entries)
        spare = max(0, self._budget.allowance(phase=split) - sum(pending for _, _, pending in entries))
        limiter = asyncio.Semaphore(self._max_workers)
        results = await _gather_or_cancel(
            [
                self._score(
                    candidate,
                    split,
                    allowance=pending + spare // count + (1 if i < spare % count else 0),
                    rejudge=rejudge,
                    limiter=limiter,
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
            RuntimeError: The baseline isn't eligible, no train trial of it completed, or a goal metric has no
                value on any train case.
        """
        assert self._run is not None
        baseline = self._baseline = Candidate.baseline(self._space)
        self._run.save_candidate(baseline, changed_keys=[], round=None, role=None)
        scores = await self._score(baseline, "train", allowance=self._budget.allowance(phase="train"))
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
        elif not any(r.status is TrialStatus.COMPLETED for r in records):
            # Eligible only through agent_failure_score: its scores would be the failure value alone.
            problem = "every train trial of the baseline failed, so nothing was measured"
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
        on_iteration: Callable[[IterationRecord], Awaitable[None] | None] | None = None,
    ) -> Optimizer:
        """Reopen a killed run; every saved trial is reused as is.

        Pass again what `run.json` can't store: the experiment, the agent (`factory=`, or `build_task=`
        with `space=`), `rollout_fn`, `environment`, the objective, constraints, search, selection, and
        score model. Of those, the cases, the space, the objective, constraints, search, selection, and score
        model are compared with `run.json` (`ValueError` naming the field). The names of the factory (or
        `build_task`) and `rollout_fn` are compared too, with a warning when one changed (a module run as
        `__main__` matches any module name); their code, `environment`, and the evaluators can't be compared and
        are the caller's to keep. Everything else
        is read from `run.json`, never passed again: the split (with the seed it used), the final pick
        (`FinalPick.from_config`), `runs_per_candidate`, `agent_failure_score`, `feasibility_threshold`, seed,
        `proposals_per_iteration`, and `rerun_passes`. Only `budget`, `max_workers`, and `on_iteration` may differ
        (`None` keeps the saved value).

        Resume continues a run that was killed, or whose search a proposer stopped by raising (it stops as a
        killed run does, before the end phase); it never extends a finished one. Every `stop` event runs the
        end phase, so a finished run only finishes or returns its end phase. A test report that measured nothing
        for a reason a resume can change (trials left `infrastructure_invalid`, or a budget that couldn't
        afford it) isn't saved as done, so resuming that run runs the test again. To search more, start a new run.
        Pass a newly built search, not the one the run used in this process: resume tells its proposers the
        run's history, and the proposers that ran the search already hold it. A search a proposer stopped ends
        with what it found when resumed with a `max_iterations` its closed iterations have already used (the
        `ProposerError` message gives the `Budget`).
        A killed search may get a larger `Budget`, which only moves where it stops; a `budget` that would let a
        stopped search go further warns that it can't. If you changed cases,
        evaluators, judge settings, or agent code, start a new run instead: resuming would mix results from
        the old and new setups.

        The killed iteration runs again on the proposals its `draw` event recorded, without asking the
        proposers, so the candidates it scored come back with their trials even from a proposer that answers
        differently each time (`Reflection`). Killed before that event, it asks them again and pays for those
        calls twice; none of its candidates was scored yet. It also asks again when the step changed: another
        parent or blocks, or more candidates (a larger budget). Then, with such a proposer, the candidates the
        killed attempt scored are in no result, though their trials count in spend.

        With seeded proposers (`TPE`, `Thompson`), the resumed run proposes what an uninterrupted one would,
        with two exceptions. Near the rollout cap, what the killed iteration spent counts from the start while
        its trials are reused at no cost, so the search can stop a step earlier than without the kill, or the
        step can be cut narrower and take only the first recorded proposals it needs; the trials of
        candidates left out still count in spend. And a refit that succeeded in the run but raises when
        replayed (e.g. under another numpy) is replaced by measured means until a refit succeeds, with a
        warning.

        `resume` only reads the run directory. `run()` locks it (`RunDirectory.lock`) before it writes, so it
        raises `RuntimeError` while another process or Optimizer is writing the run, or once one wrote to it
        after `resume` read it.

        Args:
            run_dir: The run directory to reopen.
            experiment: As in `__init__`; its cases must match the run.
            factory: As in `__init__`.
            rollout_fn: As in `__init__`.
            build_task: As in `__init__`.
            space: As in `__init__`; must match the run.
            objective: As in `__init__`; must match the run.
            constraints: As in `__init__`; must match the run.
            search: As in `__init__`; must match the run, with proposers no other optimizer has run with
                (after an in-process stop, build the `Search` again: the run's history is replayed into it).
            selection: As in `__init__`; must match the run.
            score_model: As in `__init__`; must match the run.
            environment: As in `__init__`.
            budget: A new budget for a killed or stopped search, or for a test report that wasn't saved as done
                (its reason names the `max_rollouts` it needs); None keeps the one in `run.json`, the first
                run's, so pass a larger budget again on every later resume.
            max_workers: None keeps the saved value.
            on_iteration: Called only for iterations run after the resume.

        Raises:
            ValueError: The configuration differs from `run.json`, naming the field; a proposer in `search`
                is bound to another optimizer; or `budget` can't cover what is already spent plus the
                validation reserve (when the pick hasn't run yet), unless the test report
                is saved as done.
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
        config = optimizer._config()
        state = load_state(run, config=config)
        if state.stop_reason == "setup":
            raise RuntimeError(f"run {run.run_id} stopped at setup, so there is nothing to resume; start a new run")
        tracker = optimizer._budget
        requested, budget = budget, Budget(**saved["budget"]) if budget is None else budget
        # What the pick still needs: the whole reserve (re-run headroom included) before the finalists exist, else
        # their unrun validation trials.
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
        if not state.is_test_done and budget.max_rollouts < needed:  # a finished run spends nothing more
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
        before, after = saved["agent"], config["agent"]
        renamed = [
            f"{key} {before.get(key)!r} -> {name!r}"
            for key, name in after.items()
            if not _same_name(before.get(key), name)
        ]
        if renamed:
            warnings.warn(
                f"resuming run {run.run_id} with a differently named agent ({'; '.join(renamed)}); only the names "
                "are compared, not the code, so if the agent changed, start a new run instead of mixing results "
                "from both",
                UserWarning,
                stacklevel=2,
            )
        if requested is not None and state.stop_reason in ("budget", "iterations"):
            # The search loop's own stop tests, on the train spend it stopped at, under the given budget.
            searched = sum(
                r["rollouts"] + sum(a["rollouts"] for a in r["previous_attempts"])
                for r in state.trials.values()
                if r["split"] == "train"
            )
            at_stop = BudgetTracker(
                requested, searched, validation_reserve=tracker.validation_reserve, test_reserve=tracker.test_reserve
            )
            per_candidate = len(optimizer._case_split.train) * optimizer._runs_per_candidate
            cap = requested.max_iterations
            if (cap is None or cap * optimizer._proposals_per_iteration > _slots_used(state.iterations)) and (
                at_stop.allowance(phase="train") >= per_candidate
            ):
                warnings.warn(
                    f"budget= would let run {run.run_id} search further, but its search already stopped "
                    f"({state.stop_reason}) and resume never extends a stopped search: it "
                    + ("returns the finished result" if state.is_test_done else "only finishes the end phase")
                    + "; start a new run to search more",
                    UserWarning,
                    stacklevel=2,
                )
        optimizer._run, optimizer._resumed = run, state
        return optimizer


def _same_name(saved: str | None, given: str | None) -> bool:
    """Whether two `module:qualname` names from `run.json`'s `agent` name the same callable. A module run as
    `__main__` matches any module name: the script that started a run may be imported by name to resume it."""
    if saved is None or given is None:
        return saved == given
    (saved_module, _, saved_name), (given_module, _, given_name) = saved.partition(":"), given.partition(":")
    return saved_name == given_name and (saved_module == given_module or "__main__" in (saved_module, given_module))


def _slots_used(records: Sequence[IterationRecord]) -> int:
    """The candidate slots a run's steps used against `Budget.max_iterations`, charged as a `Nested` round charges
    its own: a step that saved a candidate, the `n_candidates` it was given; a round's opener, its 1 when it proposed
    something other than a repeat of a measured candidate (a repeat costs only its own round); a step that saved
    nothing, none."""
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


class ProposerError(RuntimeError):
    """A proposer raised (the `__cause__`), so the search stopped before the end phase; `Optimizer.resume`
    continues it.

    Attributes:
        block: The name of the block whose proposer raised.
        run_dir: The run directory to resume.
    """

    def __init__(self, message: str, block: str, run_dir: Path) -> None:
        super().__init__(message, block, run_dir)  # every argument in `args`, so the error pickles
        self.block = block
        self.run_dir = run_dir

    def __str__(self) -> str:
        return str(self.args[0])


class _ProposerFailed(Exception):
    """A proposer raised (the `__cause__`; `args[0]` is the block's name); its `proposer_error` event is written,
    then `_run_search` raises `ProposerError`."""


class _ParentTrials(Sequence[TrialRecord]):
    """A step's parent's train trials, with their trajectories read from the run directory the first time
    anything reads them (TPE and Thompson never do), once for every request of the step."""

    def __init__(self, load: Callable[[], list[TrialRecord]]) -> None:
        self._load = load
        self._trials: list[TrialRecord] | None = None
        self._lock = threading.Lock()  # Reflection reads them from worker threads, several at once

    def _loaded(self) -> list[TrialRecord]:
        with self._lock:
            if self._trials is None:
                self._trials = self._load()
            return self._trials

    def __getitem__(self, index: Any) -> Any:
        return self._loaded()[index]

    def __len__(self) -> int:
        return len(self._loaded())

    def __iter__(self) -> Iterator[TrialRecord]:
        return iter(self._loaded())


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
        objective=None if assessment is None or outcome == "ineligible" else assessment.objective,
    )
