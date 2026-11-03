"""The interim loop: search on train, pick once on validation, report test only.

It mirrors the planned `Optimizer` with measured means in place of the score model, `TopRanked` selection, and
the `Joint` schedule. The event names and records follow `record/events.py` in the plan.
"""

import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from strands.types.exceptions import EventLoopException

from ...case import Case
from ...optimization.execution.status import InfrastructureError
from ...optimization.setup.budget import Budget
from ...types.evaluation import EvaluationOutput
from ...types.optimization import (
    AttemptRecord,
    CandidateLabel,
    CandidateScores,
    CaseSplit,
    IterationRecord,
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
)
from ...utils import is_throttling_error
from .proposers import ProposalBatch, ProposalContext, Proposer
from .record import RunDirectory
from .scoring import Criteria, add_usage, measured, paired, paired_ratio, score_split, zero_usage
from .space import CliSpace, content_id

NO_SCORE_MODEL = "no score model yet: ranked by measured means"
OUTPUT_PREVIEW_CHARS = 2000
REDRAWS = 3
PROPOSER_RETRIES = 3
PROPOSER_BACKOFF_SECONDS = 2.0
SCHEDULE = "joint"
SELECTION = "top_ranked"
SCORED = ("feasible", "infeasible", "ineligible")
SERVICE_ERROR_CODES = frozenset(
    {
        "internalserverexception",
        "serviceunavailableexception",
        "modelstreamerrorexception",
        "modelnotreadyexception",
        "throttlingexception",
    }
)
_TRANSPORT_ERROR_NAMES = {
    "botocore.exceptions": (
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "NoCredentialsError",
        "ReadTimeoutError",
        "ResponseStreamingError",
        "ConnectionClosedError",
    ),
    "urllib3.exceptions": ("TimeoutError", "ProtocolError", "NewConnectionError"),
}


def _transport_errors() -> tuple[type[BaseException], ...]:
    found: list[type[BaseException]] = []
    for module_name, names in _TRANSPORT_ERROR_NAMES.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        found += [error for name in names if isinstance(error := getattr(module, name, None), type)]
    return tuple(found)


_TRANSPORT_ERRORS = _transport_errors()


class SetupError(RuntimeError):
    """The baseline cannot be measured on train, so the run stops before the search."""


class ProposerFailed(RuntimeError):
    """A proposer failed with an infrastructure error after its retries; the search stops with `proposer_error`."""

    def __init__(self, message: str, usage: Usage | None = None):
        """Keep what the block spent before it failed, for `search_cost`."""
        super().__init__(message)
        self.usage = usage or zero_usage()


@dataclass(frozen=True)
class Settings:
    """What `run.json` stores for the loop: the split, the budget, and the end phase's size."""

    split: CaseSplit
    budget: Budget
    finalists: int
    runs_per_candidate: int
    proposals_per_iteration: int
    rerun_passes: int

    def reserves(self) -> tuple[int, int]:
        """The validation and test reserves: `(finalists + 1) x |validation| x runs` and `2 x |test| x runs`."""
        runs = self.runs_per_candidate
        return (self.finalists + 1) * len(self.split.validation) * runs, 2 * len(self.split.test) * runs

    def check_budget(self) -> None:
        """Raise `ValueError` when `max_rollouts` cannot cover the baseline plus the reserves."""
        validation, test = self.reserves()
        needed = len(self.split.train) * self.runs_per_candidate + validation + test
        if needed > self.budget.max_rollouts:
            raise ValueError(
                f"--max-rollouts {self.budget.max_rollouts} is below the {needed} rollouts that the baseline and "
                f"the end phase need (validation reserve {validation}, test reserve {test})"
            )


@dataclass(frozen=True)
class Candidate:
    """One scored point of the space; `number` is for persons only, `id` is the key."""

    id: str
    number: int
    values: dict[str, Value]
    parent_id: str | None
    origin: str
    rationale: str
    changed_keys: tuple[str, ...]


@dataclass(frozen=True)
class RolloutOutput:
    """What one rollout returned."""

    output: Any
    trajectory: list[Any] | None


@dataclass(frozen=True)
class EvaluatorRow:
    """One evaluator's result on one trial; `error` is set when the evaluator failed."""

    name: str
    output: EvaluationOutput
    details: list[EvaluationOutput]
    error: str | None


RolloutFn = Callable[[Mapping[str, Value], Case], RolloutOutput]
EvaluateFn = Callable[[Case, RolloutOutput], list[EvaluatorRow]]
ProgressFn = Callable[[str], None]
TrialKey = tuple[str, str, int]


def classify(error: BaseException) -> TrialStatus:
    """The planned rules: unwrap event loop errors, then infrastructure, throttling, transport, and service codes."""
    cause: BaseException = error
    while isinstance(cause, EventLoopException) and isinstance(cause.original_exception, BaseException):
        cause = cause.original_exception
    if isinstance(cause, InfrastructureError) or isinstance(cause, _TRANSPORT_ERRORS):
        return TrialStatus.INFRASTRUCTURE_INVALID
    if not isinstance(cause, EventLoopException) and is_throttling_error(cause):
        return TrialStatus.INFRASTRUCTURE_INVALID
    response = getattr(cause, "response", None)
    error_info = response.get("Error") if isinstance(response, dict) else None
    code = error_info.get("Code") if isinstance(error_info, dict) else None
    if isinstance(code, str) and code.lower() in SERVICE_ERROR_CODES:
        return TrialStatus.INFRASTRUCTURE_INVALID
    return TrialStatus.AGENT_FAILURE


class InterimOptimizer:
    """Run, resume, or return again one optimization run inside a `RunDirectory`."""

    def __init__(
        self,
        *,
        space: CliSpace,
        cases: Sequence[Case],
        criteria: Criteria,
        settings: Settings,
        proposers: Sequence[Proposer],
        rollout: RolloutFn,
        evaluate: EvaluateFn,
        run: RunDirectory,
        progress: ProgressFn,
    ):
        """Wire the loop; nothing runs until `run()`.

        Raises:
            ValueError: No proposer, or the cases lack unique names.
        """
        if not proposers:
            raise ValueError("the space needs at least one key with a proposer")
        if any(not case.name for case in cases) or len({case.name for case in cases}) != len(cases):
            raise ValueError("every case needs a unique name")
        self._space = space
        self._cases = {str(case.name): case for case in cases}
        self._criteria = criteria
        self._settings = settings
        self._proposers = tuple(proposers)
        self._rollout = rollout
        self._evaluate = evaluate
        self._run = run
        self._progress = progress
        self._candidates: dict[str, Candidate] = {}
        self._trials: dict[TrialKey, TrialRecord] = {}
        self._train: dict[str, CandidateScores] = {}
        self._iterations: list[IterationRecord] = []
        self._produced: dict[str, bool] = {}
        self._search_cost = zero_usage()
        self._spent = 0
        self._iteration = 0
        self._stop: StopReason | None = None
        self._unaffordable = False
        self._finalists: list[str] | None = None
        self._validation: ValidationReport | None = None
        self._test: TestReport | None = None
        self._test_reason: str | None = None
        self._test_done = False

    @property
    def spent(self) -> int:
        """Rollouts spent so far, re-runs included."""
        return self._spent

    @property
    def iteration(self) -> int:
        """The last finished iteration (0 = setup and the baseline)."""
        return self._iteration

    @property
    def finished(self) -> bool:
        """Whether the end phase wrote its test event."""
        return self._test_done

    def restore(self, events: Sequence[Mapping[str, Any]]) -> None:
        """Rebuild the state up to the last `iteration_end`; a killed iteration is discarded and runs again.

        Raises:
            SetupError: The run stopped at setup, so it cannot continue.
        """
        open_iteration: list[Mapping[str, Any]] = []
        for event in events:
            kind = event["event"]
            if kind == "propose" and not event.get("rescore"):
                if event["iteration"] == 0:
                    self._restore_candidate(event)
                else:
                    open_iteration.append(event)
            elif kind == "iteration_end":
                for proposed in open_iteration:
                    self._restore_candidate(proposed)
                open_iteration = []
                self._restore_iteration(event)
            elif kind == "resume":
                open_iteration = []
            else:
                self._restore_event(event)
        self._spent = sum(r.rollouts + sum(a.rollouts for a in r.previous_attempts) for r in self._trials.values())
        for candidate_id in self._candidates:
            self._train[candidate_id] = self._split_scores(candidate_id, "train")

    def run(self) -> dict[str, Any]:
        """Search until a stop, then run the end phase; a finished run only returns its end phase again.

        Raises:
            SetupError: The baseline cannot be measured on train.
        """
        if not self._test_done and self._stop is None:
            self._ensure_baseline()
            self._stop = self._search()
            self._log("stop", reason=self._stop, message=f"search stopped: {self._stop}")
            self._progress(f"search stopped: {self._stop}")
        return self._end_phase()

    def _restore_candidate(self, event: Mapping[str, Any]) -> None:
        self._candidates[event["candidate_id"]] = Candidate(
            id=event["candidate_id"],
            number=event["number"],
            values=dict(event["values"]),
            parent_id=event["parent_id"],
            origin=event["origin"],
            rationale=event["rationale"],
            changed_keys=tuple(event["changed_keys"]),
        )

    def _restore_iteration(self, event: Mapping[str, Any]) -> None:
        record = IterationRecord.model_validate(event["record"])
        self._iterations.append(record)
        self._iteration = record.iteration
        exhausted = set(event["exhausted"])
        self._produced.update({name: name not in exhausted for name in record.blocks})
        self._search_cost = add_usage(self._search_cost, Usage.model_validate(event["usage"]))

    def _restore_event(self, event: Mapping[str, Any]) -> None:
        kind = event["event"]
        if kind == "trial":
            record = self._load_trial(event["record"])
            self._trials[(record.candidate_id, record.case_name, record.run_index)] = record
        elif kind == "stop":
            if event["reason"] == "setup":
                raise SetupError(f"{event['message']}; start a new run")
            self._stop = event["reason"]
        else:
            self._restore_end_phase(event)

    def _restore_end_phase(self, event: Mapping[str, Any]) -> None:
        kind = event["event"]
        if kind == "finalists":
            self._finalists = list(event["candidate_ids"])
        elif kind == "selected":
            self._validation = ValidationReport.model_validate(event["report"])
        elif kind == "test":
            self._test = None if event["report"] is None else TestReport.model_validate(event["report"])
            self._test_reason = event["reason"]
            self._test_done = True

    def _ensure_baseline(self) -> None:
        if self._iteration > 0:
            return
        values = self._space.baseline()
        baseline = self._candidates.get(content_id(values))
        if baseline is None:
            baseline = Candidate(content_id(values), 0, values, None, "baseline", "", ())
            self._add(baseline)
        scores = self._score_all([baseline], "train")[baseline.id]
        records = self._records(baseline.id, self._settings.split.train)
        if scores is None or not scores.eligible or all(r.status is not TrialStatus.COMPLETED for r in records):
            message = self._setup_message(records)
            self._log("stop", reason="setup", message=message)
            raise SetupError(message)
        self._label(baseline, scores)
        self._progress(f"baseline: train objective {scores.objective:.4f}")

    def _setup_message(self, records: Sequence[TrialRecord]) -> str:
        first = next((r for r in records if r.error), None)
        error = f"; first error: {first.error_type}: {first.error}" if first else ""
        return f"the baseline cannot be measured on train (trial statuses: {self._statuses(records)}){error}"

    def _search(self) -> StopReason:
        while True:
            reason = self._should_stop()
            if reason is not None:
                return reason
            self._iteration += 1
            parent = self._select_parent()
            outcomes: list[ProposalOutcome] = []
            usage = zero_usage()
            failed = False
            for block in self._proposers:
                try:
                    produced, spent = self._turn(block, parent, outcomes)
                except ProposerFailed as error:
                    failed, spent = True, error.usage
                    produced = False
                self._produced[block.name] = produced
                usage = add_usage(usage, spent)
                if failed:
                    break
            self._search_cost = add_usage(self._search_cost, usage)
            self._end_iteration(parent, outcomes, usage)
            if failed:
                return "proposer_error"

    def _should_stop(self) -> StopReason | None:
        if self._unaffordable or not self._can_afford(self._cheapest_candidate(), "train"):
            return "budget"
        cap = self._settings.budget.max_iterations
        if cap is not None and self._iteration >= cap:
            return "iterations"
        if len(self._produced) == len(self._proposers) and not any(self._produced.values()):
            return "exhausted"
        return None

    def _cheapest_candidate(self) -> int:
        """The train rollouts of the cheapest next candidate: a new one, or one with trials saved before a kill."""
        names = self._settings.split.train
        orphans = {key[0] for key in self._trials if key[0] not in self._candidates}
        costs = [len(self._pending(candidate_id, names)) for candidate_id in orphans]
        return min([len(names) * self._settings.runs_per_candidate, *costs])

    def _select_parent(self) -> Candidate:
        parent = self._candidates[self._ranking()[0]]
        self._log("select", parent_id=parent.id, strategy=SELECTION)
        return parent

    def _ranking(self) -> list[str]:
        eligible = [cid for cid, s in self._train.items() if s.eligible and s.objective is not None]
        return sorted(eligible, key=lambda cid: (-(self._train[cid].objective or 0.0), self._candidates[cid].number))

    def _turn(self, block: Proposer, parent: Candidate, outcomes: list[ProposalOutcome]) -> tuple[bool, Usage]:
        """One block's turn; ask again, up to `REDRAWS` times, after a duplicate, an invalid patch, or no patches.

        A redraw asks only for replacements of the unusable patches. A reply with fewer usable patches than
        asked for is valid and ends the turn.
        """
        usage, produced, count = zero_usage(), False, self._settings.proposals_per_iteration
        for redraw in range(REDRAWS + 1):
            try:
                batch = self._propose(block, parent, redraw, count, consumed=self._consumed(block, outcomes))
            except ProposerFailed as error:
                error.usage = add_usage(usage, error.usage)
                raise
            usage = add_usage(usage, batch.usage)
            for problem in batch.problems:
                self._log("invalid", block=block.name, redraw=redraw, updates={}, reason=problem)
                self._progress(f"{block.name} proposer: {problem}")
            known = set(self._candidates)
            results = [self._handle(patch, parent, block, redraw) for patch in batch.patches]
            outcomes += results
            produced = produced or any(r.candidate_id not in known and r.outcome in SCORED for r in results)
            unusable = len(batch.problems) + sum(r.outcome in ("duplicate", "invalid") for r in results)
            if self._unaffordable or (batch.patches and not unusable):
                break
            count = min(unusable, count) if batch.patches else count
        return produced, usage

    def _consumed(self, block: Proposer, current: Sequence[ProposalOutcome]) -> int:
        done = [outcome for record in self._iterations for outcome in record.proposals]
        return sum(outcome.origin == block.name for outcome in [*done, *current])

    def _propose(self, block: Proposer, parent: Candidate, redraw: int, count: int, *, consumed: int) -> ProposalBatch:
        context = ProposalContext(
            space=self._space,
            keys=block.keys,
            parent=parent.values,
            parent_number=parent.number,
            count=count,
            iteration=self._iteration,
            redraw=redraw,
            consumed=consumed,
            feedback=self._feedback(parent) if block.kind == "Reflection" else {},
        )
        for attempt in range(PROPOSER_RETRIES + 1):
            try:
                return block.propose(context)
            except Exception as error:
                infrastructure = classify(error) is TrialStatus.INFRASTRUCTURE_INVALID
                fields = {
                    "block": block.name,
                    "redraw": redraw,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                if attempt < PROPOSER_RETRIES:
                    self._log("proposer_retry", **fields)
                    if infrastructure:
                        time.sleep(PROPOSER_BACKOFF_SECONDS * (attempt + 1))
                    continue
                if not infrastructure:
                    return ProposalBatch([], zero_usage(), (f"proposer error: {type(error).__name__}: {error}",))
                self._log("proposer_error", **fields)
                self._progress(f"proposer {block.name} failed: {type(error).__name__}: {error}")
                raise ProposerFailed(str(error)) from error
        raise AssertionError("unreachable")

    def _feedback(self, parent: Candidate) -> dict[str, Any]:
        cases = [
            {
                "case": name,
                "input": self._cases[name].input,
                "runs": [_run_feedback(record) for record in self._records(parent.id, [name])],
            }
            for name in self._settings.split.train
        ]
        others = [
            {
                "version": self._candidates[cid].number,
                "changed_keys": list(self._candidates[cid].changed_keys),
                "objective": scores.objective,
                "by_case": scores.objective_by_case,
            }
            for cid, scores in self._train.items()
            if cid != parent.id
        ]
        return {"cases": cases, "other_versions": others}

    def _handle(self, patch: Patch, parent: Candidate, block: Proposer, redraw: int) -> ProposalOutcome:
        values = {**parent.values, **patch.updates}
        outside = sorted(set(patch.updates) - set(block.keys))
        try:
            if outside:
                raise ValueError(f"keys outside the {block.name} block: {', '.join(outside)}")
            self._space.validate(values)
        except ValueError as error:
            self._log("invalid", block=block.name, redraw=redraw, updates=patch.updates, reason=str(error))
            return _outcome(None, None, parent, block, patch, "invalid", None)
        candidate_id = content_id(values)
        known = self._candidates.get(candidate_id)
        train = self._settings.split.train
        retryable = self._pending(candidate_id, train) or self._keys_with(
            candidate_id, train, TrialStatus.VERIFIER_INVALID
        )
        if known is not None and (self._train[candidate_id].eligible or not retryable):
            self._log("duplicate", block=block.name, redraw=redraw, candidate_id=candidate_id)
            return _outcome(candidate_id, known.number, parent, block, patch, "duplicate", self._objective(known.id))
        changed = tuple(key for key in values if values[key] != parent.values[key])
        candidate = known or Candidate(
            candidate_id, len(self._candidates), values, parent.id, block.name, patch.rationale, changed
        )
        pending = len(self._pending(candidate_id, self._settings.split.train))
        if not self._can_afford(pending, "train"):
            self._unaffordable = True
            validation, test = self._settings.reserves()
            remaining = self._remaining()
            self._log(
                "unscored", candidate_id=candidate_id, pending=pending, remaining=remaining, reserve=validation + test
            )
            return _outcome(candidate_id, None, parent, block, patch, "unscored", None)
        if known is None:
            self._add(candidate)
        else:
            self._log_rescore(known)
        scores = self._score_all([candidate], "train")[candidate_id]
        assert scores is not None
        label = self._label(candidate, scores)
        return _outcome(candidate_id, candidate.number, parent, block, patch, label.outcome, scores.objective)

    def _end_iteration(self, parent: Candidate, outcomes: list[ProposalOutcome], usage: Usage) -> None:
        record = IterationRecord(
            iteration=self._iteration,
            parent_id=parent.id,
            round=None,
            role="inner",
            blocks=[block.name for block in self._proposers],
            proposals=outcomes,
            rollouts_spent=self._spent,
        )
        self._iterations.append(record)
        best = self._ranking()[0]
        self._log(
            "iteration_end",
            record=record.model_dump(mode="json"),
            exhausted=[name for name, made in self._produced.items() if not made],
            best_after=best,
            usage=usage.model_dump(mode="json"),
        )
        self._progress(
            f"iteration {self._iteration}: {len(outcomes)} proposal(s), best train objective "
            f"{self._objective(best):.4f} (candidate {self._candidates[best].number}), {self._spent} rollouts spent"
        )

    def _end_phase(self) -> dict[str, Any]:
        baseline = next(c for c in self._candidates.values() if c.origin == "baseline")
        if self._finalists is None:
            self._finalists = [cid for cid in self._ranking() if cid != baseline.id][: self._settings.finalists]
            self._log("finalists", candidate_ids=self._finalists)
        if self._validation is None:
            contenders = [baseline.id, *self._finalists]
            scores = self._score_all([self._candidates[cid] for cid in contenders], "validation")
            self._validation = self._pick(baseline.id, contenders, scores)
            self._log("selected", report=self._validation.model_dump(mode="json"))
        if not self._test_done:
            self._test, self._test_reason = self._test_report(self._validation.champion_id, baseline.id)
            report = None if self._test is None else self._test.model_dump(mode="json")
            self._log("test", report=report, reason=self._test_reason, front={})
            self._test_done = True
        self._progress(self._validation.reason)
        champion = self._candidates[self._validation.champion_id]
        return self._result(baseline, champion, self._validation, self._test, self._test_reason)

    def _pick(
        self, baseline_id: str, contenders: list[str], validation: Mapping[str, CandidateScores | None]
    ) -> ValidationReport:
        objectives: dict[str, float | None] = {}
        for cid in contenders:
            scores = validation[cid]
            objectives[cid] = scores.objective if scores is not None and scores.eligible else None
        eligible = [cid for cid in contenders if objectives[cid] is not None]
        counts = {cid: self._counted(cid, self._settings.split.validation) for cid in contenders}
        ranked = sorted(eligible, key=lambda cid: (-(objectives[cid] or 0.0), -counts[cid], contenders.index(cid)))
        champion = ranked[0] if ranked else baseline_id
        return ValidationReport(
            champion_id=champion,
            front=[],
            objectives=objectives,
            estimates={},
            infeasible={cid: _why_not(validation[cid]) for cid in contenders if cid not in eligible},
            measured={cid: measured(scores) for cid, scores in validation.items() if scores is not None},
            problem=NO_SCORE_MODEL,
            reason=self._pick_reason(champion, baseline_id, objectives),
        )

    def _pick_reason(self, champion: str, baseline: str, objectives: Mapping[str, float | None]) -> str:
        base = objectives.get(baseline)
        base_text = "unmeasured" if base is None else f"{base:.4f}"
        if champion == baseline:
            return f"baseline ships: no finalist beat it on validation (baseline {base_text})"
        number = self._candidates[champion].number
        return f"candidate {number} highest on validation: {objectives[champion]:.4f} vs baseline {base_text}"

    def _test_report(self, champion_id: str, baseline_id: str) -> tuple[TestReport | None, str | None]:
        if not self._settings.split.test:
            return None, "no test cases"
        ids = list(dict.fromkeys([champion_id, baseline_id]))
        scores = self._score_all([self._candidates[cid] for cid in ids], "test")
        champion, base = scores[champion_id], scores[baseline_id]
        if champion is None or base is None:
            return None, "the budget cannot cover the test report"
        metrics = self._criteria.goal_metrics
        names = self._settings.split.test
        report = TestReport(
            champion_id=champion_id,
            baseline_id=baseline_id,
            objective=paired(champion.objective_by_case, base.objective_by_case),
            difference={m: paired(_by_case(champion, m), _by_case(base, m)) for m in metrics},
            ratio={
                m: paired_ratio(_by_case(champion, m), _by_case(base, m), can_be_negative=self._can_be_negative(m))
                for m in metrics
            },
            missing={cid: sum(name not in s.objective_by_case for name in names) for cid, s in _known(scores)},
            means={cid: measured(s) for cid, s in _known(scores)},
        )
        return report, None

    def _result(
        self,
        baseline: Candidate,
        champion: Candidate,
        report: ValidationReport,
        test: TestReport | None,
        test_reason: str | None,
    ) -> dict[str, Any]:
        diff = self._space.diff(baseline.values, champion.values)
        changed = self._space.changed_values(baseline.values, champion.values)
        effects: dict[str, Any] = {"rows": [], "reason": NO_SCORE_MODEL}
        self._run.save_effects(effects)
        diff_path, values_path = self._run.write_review("best", diff, changed)
        result = {
            "run_id": self._run.run_id,
            "run_dir": str(self._run.root),
            "improved": champion.id != baseline.id,
            "stop_reason": self._stop,
            "baseline": _view(baseline),
            "best": _view(champion),
            "validation": report.model_dump(mode="json"),
            "test": None if test is None else test.model_dump(mode="json"),
            "test_reason": test_reason,
            "front": [],
            "front_test": {},
            "effects": effects,
            "iterations": [record.model_dump(mode="json") for record in self._iterations],
            "rollouts_spent": self._spent,
            **self._costs(),
            "diff": diff,
            "diff_path": str(diff_path),
            "values_path": str(values_path),
            "changed_values": changed,
        }
        self._run.write_result(result)
        return result

    def _costs(self) -> dict[str, Any]:
        agent = {split: zero_usage() for split in ("train", "validation", "test")}
        judge, retry = zero_usage(), zero_usage()
        for record in self._trials.values():
            agent[record.split] = add_usage(agent[record.split], record.usage)
            judge = add_usage(judge, record.judge_usage)
            for attempt in record.previous_attempts:
                retry = add_usage(add_usage(retry, attempt.usage), attempt.judge_usage)
        return {
            "agent_cost": {split: usage.model_dump(mode="json") for split, usage in agent.items()},
            "search_cost": self._search_cost.model_dump(mode="json"),
            "judge_cost": judge.model_dump(mode="json"),
            "retry_cost": retry.model_dump(mode="json"),
        }

    def _score_all(self, candidates: Sequence[Candidate], split: SplitName) -> dict[str, CandidateScores | None]:
        """Score candidates on a split: every first attempt, a judge pass, then infrastructure re-run passes.

        A candidate is scored only when its first attempts fit the budget; re-runs use what the reserves leave.
        """
        names = self._names(split)
        chosen: list[Candidate] = []
        total = 0
        for candidate in candidates:
            pending = len(self._pending(candidate.id, names))
            if pending == 0 or self._can_afford(total + pending, split):
                chosen.append(candidate)
                total += pending
        for candidate in chosen:
            for key in self._pending(candidate.id, names):
                self._attempt(candidate, key, split)
        self._rejudge_all(chosen, names, split)
        for _ in range(self._settings.rerun_passes):
            failed = TrialStatus.INFRASTRUCTURE_INVALID
            retry = [(c, key) for c in chosen for key in self._keys_with(c.id, names, failed)]
            for candidate, key in retry:
                if not self._can_afford(1, split):
                    break
                self._attempt(candidate, key, split)
            self._rejudge_all(chosen, names, split)
        chosen_ids = {candidate.id for candidate in chosen}
        return {c.id: self._finish_score(c, split) if c.id in chosen_ids else None for c in candidates}

    def _rejudge_all(self, candidates: Sequence[Candidate], names: Sequence[str], split: SplitName) -> None:
        for candidate in candidates:
            for key in self._keys_with(candidate.id, names, TrialStatus.VERIFIER_INVALID):
                self._rejudge(key, split)

    def _finish_score(self, candidate: Candidate, split: SplitName) -> CandidateScores:
        names = self._names(split)
        scores = self._split_scores(candidate.id, split)
        if split == "train":
            self._train[candidate.id] = scores
        records = self._records(candidate.id, names)
        self._log(
            "score",
            candidate_id=candidate.id,
            split=split,
            case_names=list(names),
            runs=self._settings.runs_per_candidate,
            rollouts=sum(record.rollouts + sum(a.rollouts for a in record.previous_attempts) for record in records),
            objective_by_run=scores.objective_by_run,
            statuses=self._statuses(records),
            unavailable={case: {str(k): v for k, v in runs.items()} for case, runs in scores.unavailable.items()},
        )
        return scores

    def _attempt(self, candidate: Candidate, key: TrialKey, split: SplitName) -> None:
        case = self._cases[key[1]]
        previous = self._previous(key, judge_only=False)
        started = time.monotonic()
        try:
            result, status, error = self._rollout(candidate.values, case), TrialStatus.COMPLETED, None
        except Exception as exc:
            result, status, error = None, classify(exc), exc
        usage = _rollout_usage(time.monotonic() - started)
        self._spent += 1
        rows = self._judge(case, result) if result is not None else []
        self._save_trial(key, split, status, error, previous, result, rows, usage, rollouts=1)

    def _rejudge(self, key: TrialKey, split: SplitName) -> None:
        old = self._trials[key]
        trajectory = old.trajectory if isinstance(old.trajectory, list) else None
        result = RolloutOutput(output=old.output, trajectory=trajectory)
        rows = self._judge(self._cases[key[1]], result)
        previous = self._previous(key, judge_only=True)
        self._save_trial(key, split, TrialStatus.COMPLETED, None, previous, result, rows, old.usage, rollouts=0)

    def _previous(self, key: TrialKey, *, judge_only: bool) -> list[AttemptRecord]:
        old = self._trials.get(key)
        if old is None:
            return []
        replaced = AttemptRecord(
            status=old.status,
            error_type=old.error_type,
            error=old.error,
            usage=zero_usage() if judge_only else old.usage,
            judge_usage=old.judge_usage,
            pass_index=len(old.previous_attempts),
            rollouts=old.rollouts,
        )
        return [*old.previous_attempts, replaced]

    def _save_trial(
        self,
        key: TrialKey,
        split: SplitName,
        status: TrialStatus,
        error: BaseException | None,
        previous: list[AttemptRecord],
        result: RolloutOutput | None,
        rows: list[EvaluatorRow],
        usage: Usage,
        *,
        rollouts: int,
    ) -> None:
        evaluations = {row.name: row.output for row in rows if row.error is None}
        if status is TrialStatus.COMPLETED and rows and not evaluations:
            status = TrialStatus.VERIFIER_INVALID
        record = TrialRecord(
            candidate_id=key[0],
            case_name=key[1],
            run_index=key[2],
            split=split,
            status=status,
            error_type=type(error).__name__ if error else None,
            error=str(error) if error else None,
            attempts=len(previous) + 1,
            previous_attempts=previous,
            output=None if result is None else result.output,
            trajectory=None if result is None else result.trajectory,
            interactions=None,
            environment_state=None,
            evaluations=evaluations,
            details={row.name: row.details for row in rows if row.error is None},
            evaluator_errors={row.name: row.error for row in rows if row.error is not None},
            usage=usage,
            judge_usage=zero_usage(),
            rollouts=rollouts,
        )
        self._trials[key] = record
        data = record.model_dump(mode="json")
        data["transcript"] = self._run.save_transcript(data.pop("trajectory"))
        self._log("trial", record=data)

    def _judge(self, case: Case, result: RolloutOutput) -> list[EvaluatorRow]:
        try:
            return self._evaluate(case, result)
        except Exception as exc:
            failed = EvaluationOutput(score=0.0, test_pass=False, reason=f"Evaluator error: {exc}")
            return [
                EvaluatorRow(name, failed, [], f"Evaluator error: {type(exc).__name__}: {exc}")
                for name in self._criteria.registry.evaluators
            ]

    def _label(self, candidate: Candidate, scores: CandidateScores) -> CandidateLabel:
        unmeasured = [name for name in self._settings.split.train if name not in scores.objective_by_case]
        label = CandidateLabel(
            outcome="feasible" if scores.eligible else "ineligible",
            reason="every train case measured" if scores.eligible else f"unmeasured train cases: {unmeasured}",
            unmet=[],
            estimates={},
            p_met={},
            by_fit=False,
            measured=measured(scores),
        )
        self._log("label", candidate_id=candidate.id, label=label.model_dump(mode="json"))
        return label

    def _add(self, candidate: Candidate) -> None:
        self._candidates[candidate.id] = candidate
        self._log(
            "propose",
            candidate_id=candidate.id,
            number=candidate.number,
            parent_id=candidate.parent_id,
            origin=candidate.origin,
            values=candidate.values,
            changed_keys=list(candidate.changed_keys),
            rationale=candidate.rationale,
            inherited={},
            defaulted=[],
            round=None,
            role="inner",
        )

    def _log_rescore(self, candidate: Candidate) -> None:
        self._log(
            "propose",
            candidate_id=candidate.id,
            number=candidate.number,
            parent_id=candidate.parent_id,
            origin=candidate.origin,
            rescore=True,
            round=None,
            role="inner",
        )

    def _split_scores(self, candidate_id: str, split: SplitName) -> CandidateScores:
        names = self._names(split)
        return score_split(candidate_id, self._records(candidate_id, names), names, self._criteria)

    def _can_afford(self, rollouts: int, phase: SplitName) -> bool:
        validation, test = self._settings.reserves()
        reserve = {"train": validation + test, "validation": test, "test": 0}[phase]
        return self._spent + rollouts + reserve <= self._settings.budget.max_rollouts

    def _remaining(self) -> int:
        return self._settings.budget.max_rollouts - self._spent

    def _names(self, split: SplitName) -> tuple[str, ...]:
        return getattr(self._settings.split, split)  # type: ignore[no-any-return]

    def _keys(self, candidate_id: str, names: Sequence[str]) -> list[TrialKey]:
        return [(candidate_id, name, run) for name in names for run in range(self._settings.runs_per_candidate)]

    def _pending(self, candidate_id: str, names: Sequence[str]) -> list[TrialKey]:
        """Keys without a trial, or with an infrastructure failure, which runs again when scored again."""
        failed = TrialStatus.INFRASTRUCTURE_INVALID
        keys = self._keys(candidate_id, names)
        return [key for key in keys if key not in self._trials or self._trials[key].status is failed]

    def _keys_with(self, candidate_id: str, names: Sequence[str], status: TrialStatus) -> list[TrialKey]:
        keys = self._keys(candidate_id, names)
        return [key for key in keys if key in self._trials and self._trials[key].status is status]

    def _records(self, candidate_id: str, names: Sequence[str]) -> list[TrialRecord]:
        return [self._trials[key] for key in self._keys(candidate_id, names) if key in self._trials]

    def _counted(self, candidate_id: str, names: Sequence[str]) -> int:
        counted = (TrialStatus.COMPLETED, TrialStatus.AGENT_FAILURE)
        return sum(record.status in counted for record in self._records(candidate_id, names))

    def _objective(self, candidate_id: str) -> float | None:
        scores = self._train.get(candidate_id)
        return None if scores is None else scores.objective

    def _statuses(self, records: Sequence[TrialRecord]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            counts[record.status.value] = counts.get(record.status.value, 0) + 1
        return counts

    def _can_be_negative(self, metric: str) -> bool:
        low, _ = self._criteria.registry.range(metric)
        return low is None or low < 0

    def _load_trial(self, data: Mapping[str, Any]) -> TrialRecord:
        fields = dict(data)
        fields["trajectory"] = self._run.load_transcript(fields.pop("transcript", None))
        return TrialRecord.model_validate(fields)

    def _log(self, event: str, **fields: Any) -> None:
        self._run.events.append(event, self._iteration, **fields)


def _outcome(
    candidate_id: str | None,
    number: int | None,
    parent: Candidate,
    block: Proposer,
    patch: Patch,
    outcome: Any,
    objective: float | None,
) -> ProposalOutcome:
    return ProposalOutcome(
        candidate_id=candidate_id,
        number=number,
        parent_id=parent.id,
        origin=block.name,
        changed_keys=sorted(key for key, value in patch.updates.items() if parent.values.get(key) != value),
        rationale=patch.rationale,
        outcome=outcome,
        objective=objective,
    )


def _rollout_usage(seconds: float) -> Usage:
    # Project mode captures no chat spans yet, so only the rollout time is measured.
    return Usage(**{**zero_usage().model_dump(), "seconds": seconds})


def _run_feedback(record: TrialRecord) -> dict[str, Any]:
    output = record.output if isinstance(record.output, str) else str(record.output)
    return {
        "status": record.status.value,
        "output": output[:OUTPUT_PREVIEW_CHARS],
        "scores": {name: result.score for name, result in record.evaluations.items()},
        "reasons": {name: result.reason for name, result in record.evaluations.items()},
        "error": record.error,
    }


def _known(scores: Mapping[str, CandidateScores | None]) -> list[tuple[str, CandidateScores]]:
    return [(cid, value) for cid, value in scores.items() if value is not None]


def _by_case(scores: CandidateScores, metric: str) -> dict[str, float]:
    return {case: metrics[metric] for case, metrics in scores.metrics_by_case.items() if metric in metrics}


def _why_not(scores: CandidateScores | None) -> str:
    if scores is None:
        return "not scored on validation: the budget ran out"
    unmeasured = len(scores.objective_by_run) - len(scores.objective_by_case)
    return f"not eligible on validation: {unmeasured} case(s) unmeasured"


def _view(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "number": candidate.number,
        "origin": candidate.origin,
        "parent_id": candidate.parent_id,
        "values": candidate.values,
        "rationale": candidate.rationale,
    }
