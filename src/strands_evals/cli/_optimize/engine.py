"""The interim loop: search on train, pick once on validation, report test only.

It mirrors the planned `Optimizer` with measured means in place of the score model, `TopRanked` selection, and
an `Alternate` schedule over the proposers. The event names and records follow `record/events.py` in the plan.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
SCORED_OUTCOMES = ("feasible", "infeasible", "ineligible")
# Runner failures arrive as text, so transient provider and network errors are matched by message.
_INFRASTRUCTURE_MARKERS = (
    "throttl",
    "too many requests",
    "rate exceeded",
    "serviceunavailable",
    "service unavailable",
    "internalserverexception",
    "modelnotready",
    "econnreset",
    "etimedout",
    "eai_again",
    "connection reset",
    "connection refused",
)


class SetupError(RuntimeError):
    """The baseline cannot be measured on train, so the run stops before the search."""


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


def classify(error: BaseException) -> TrialStatus:
    """`infrastructure_invalid` for transient provider or network failures, else `agent_failure`."""
    cause = getattr(error, "original_exception", None) or error
    if isinstance(cause, InfrastructureError) or is_throttling_error(cause):
        return TrialStatus.INFRASTRUCTURE_INVALID
    text = str(cause).lower()
    if any(marker in text for marker in _INFRASTRUCTURE_MARKERS):
        return TrialStatus.INFRASTRUCTURE_INVALID
    return TrialStatus.AGENT_FAILURE


class InterimOptimizer:
    """Run, resume, or replay one optimization run inside a `RunDirectory`."""

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
        """Wire the loop; nothing runs until `run()`."""
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
        self._trials: dict[tuple[str, str, int], TrialRecord] = {}
        self._train: dict[str, CandidateScores] = {}
        self._iterations: list[IterationRecord] = []
        self._produced: dict[str, bool] = {}
        self._search_cost = zero_usage()
        self._spent = 0
        self._iteration = 0
        self._stop: StopReason | None = None
        self._finished = False
        self._unaffordable = False
        self._replay = False

    def restore(self, events: Sequence[Mapping[str, Any]]) -> None:
        """Rebuild the state up to the last `iteration_end`; a killed iteration is discarded and runs again.

        Raises:
            SetupError: The run stopped at setup, so it cannot continue.
        """
        completed = max((e["record"]["iteration"] for e in events if e["event"] == "iteration_end"), default=0)
        for event in events:
            self._restore_event(event, completed)
        self._iteration = completed
        self._spent = sum(r.rollouts + sum(a.rollouts for a in r.previous_attempts) for r in self._trials.values())
        for candidate_id in self._candidates:
            self._train[candidate_id] = self._split_scores(candidate_id, "train")

    def run(self) -> dict[str, Any]:
        """Search until a stop, then run the end phase; a finished run only returns its end phase again.

        Raises:
            SetupError: The baseline cannot be measured on train.
        """
        if self._finished:
            self._replay = True
        elif self._stop is None:
            if not self._candidates:
                self._score_baseline()
            self._stop = self._search()
            self._log("stop", reason=self._stop, message=f"search stopped: {self._stop}")
            self._progress(f"search stopped: {self._stop}")
        return self._end_phase()

    def _restore_event(self, event: Mapping[str, Any], completed: int) -> None:
        kind = event["event"]
        if kind == "trial":
            record = self._load_trial(event["record"])
            self._trials[(record.candidate_id, record.case_name, record.run_index)] = record
        elif kind == "propose" and event["iteration"] <= completed:
            self._candidates[event["candidate_id"]] = Candidate(
                id=event["candidate_id"],
                number=event["number"],
                values=dict(event["values"]),
                parent_id=event["parent_id"],
                origin=event["origin"],
                rationale=event["rationale"],
                changed_keys=tuple(event["changed_keys"]),
            )
        elif kind == "iteration_end" and event["record"]["iteration"] <= completed:
            self._iterations.append(IterationRecord.model_validate(event["record"]))
            self._produced[event["block"]] = not event["exhausted"]
            self._search_cost = add_usage(self._search_cost, Usage.model_validate(event["usage"]))
        elif kind == "stop" and event["reason"] == "setup":
            raise SetupError(f"{event['message']}; start a new run")
        elif kind == "stop":
            self._stop = event["reason"]
        elif kind == "test":
            self._finished = True

    def _score_baseline(self) -> None:
        values = self._space.baseline()
        baseline = Candidate(content_id(values), 0, values, None, "baseline", "", ())
        self._add(baseline)
        scores = self._score(baseline, "train")
        records = self._records(baseline.id, self._settings.split.train)
        if scores is None or not scores.eligible or all(r.status is not TrialStatus.COMPLETED for r in records):
            message = self._setup_message(records)
            self._log("stop", reason="setup", message=message)
            raise SetupError(message)
        self._label(baseline, scores)
        self._progress(f"baseline: train objective {scores.objective:.4f}")

    def _setup_message(self, records: Sequence[TrialRecord]) -> str:
        statuses = self._statuses(records)
        first = next((r for r in records if r.error), None)
        error = f"; first error: {first.error_type}: {first.error}" if first else ""
        return f"the baseline cannot be measured on train (trial statuses: {statuses}){error}"

    def _search(self) -> StopReason:
        while True:
            reason = self._should_stop()
            if reason is not None:
                return reason
            self._iteration += 1
            block = self._proposers[(self._iteration - 1) % len(self._proposers)]
            parent = self._select_parent()
            try:
                batch = self._propose(block, parent)
            except Exception:
                return "proposer_error"
            outcomes = [self._handle(patch, parent, block) for patch in batch.patches]
            self._search_cost = add_usage(self._search_cost, batch.usage)
            self._end_iteration(parent, block, outcomes, batch)

    def _should_stop(self) -> StopReason | None:
        per_candidate = len(self._settings.split.train) * self._settings.runs_per_candidate
        if self._unaffordable or not self._can_afford(per_candidate, "train"):
            return "budget"
        cap = self._settings.budget.max_iterations
        if cap is not None and self._iteration >= cap:
            return "iterations"
        if len(self._produced) == len(self._proposers) and not any(self._produced.values()):
            return "exhausted"
        return None

    def _select_parent(self) -> Candidate:
        parent = self._candidates[self._ranking()[0]]
        self._log("select", parent_id=parent.id, strategy="TopRanked")
        return parent

    def _ranking(self) -> list[str]:
        eligible = [cid for cid, s in self._train.items() if s.eligible and s.objective is not None]
        return sorted(eligible, key=lambda cid: (-(self._train[cid].objective or 0.0), self._candidates[cid].number))

    def _propose(self, block: Proposer, parent: Candidate) -> ProposalBatch:
        context = self._context(block, parent)
        for attempt in range(2):
            try:
                return block.propose(context)
            except Exception as error:
                event = "proposer_retry" if attempt == 0 else "proposer_error"
                self._log(event, block=block.name, redraw=attempt, error_type=type(error).__name__, error=str(error))
                if attempt == 1:
                    self._progress(f"proposer {block.name} failed: {type(error).__name__}: {error}")
                    raise
        raise AssertionError("unreachable")

    def _context(self, block: Proposer, parent: Candidate) -> ProposalContext:
        consumed = sum(1 for r in self._iterations for outcome in r.proposals if outcome.origin == block.name)
        return ProposalContext(
            space=self._space,
            keys=block.keys,
            parent=parent.values,
            parent_number=parent.number,
            count=self._settings.proposals_per_iteration,
            iteration=self._iteration,
            consumed=consumed,
            feedback=self._feedback(parent) if block.kind == "Reflection" else {},
        )

    def _feedback(self, parent: Candidate) -> dict[str, Any]:
        cases = []
        for name in self._settings.split.train:
            case = self._cases[name]
            runs = [_run_feedback(record) for record in self._records(parent.id, [name])]
            cases.append({"case": name, "input": case.input, "expected": case.expected_output, "runs": runs})
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

    def _handle(self, patch: Patch, parent: Candidate, block: Proposer) -> ProposalOutcome:
        values = {**parent.values, **patch.updates}
        outside = sorted(set(patch.updates) - set(block.keys))
        try:
            if outside:
                raise ValueError(f"keys outside the {block.name} block: {', '.join(outside)}")
            self._space.validate(values)
        except ValueError as error:
            self._log("invalid", block=block.name, redraw=0, updates=patch.updates, reason=str(error))
            return _outcome(None, None, parent, block, patch, "invalid", None)
        candidate_id = content_id(values)
        if candidate_id in self._candidates:
            self._log("duplicate", block=block.name, redraw=0, candidate_id=candidate_id)
            known = self._candidates[candidate_id]
            return _outcome(candidate_id, known.number, parent, block, patch, "duplicate", self._objective(known.id))
        changed = tuple(key for key in values if values[key] != parent.values[key])
        candidate = Candidate(
            candidate_id, len(self._candidates), values, parent.id, block.name, patch.rationale, changed
        )
        pending = len(self._missing(candidate_id, self._settings.split.train))
        if not self._can_afford(pending, "train"):
            self._unaffordable = True
            self._log("unscored", candidate_id=candidate_id, pending=pending, remaining=self._remaining())
            return _outcome(candidate_id, None, parent, block, patch, "unscored", None)
        self._add(candidate)
        scores = self._score(candidate, "train")
        assert scores is not None
        label = self._label(candidate, scores)
        return _outcome(candidate_id, candidate.number, parent, block, patch, label.outcome, scores.objective)

    def _end_iteration(
        self, parent: Candidate, block: Proposer, outcomes: list[ProposalOutcome], batch: ProposalBatch
    ) -> None:
        record = IterationRecord(
            iteration=self._iteration,
            parent_id=parent.id,
            round=None,
            role="inner",
            blocks=[block.name],
            proposals=outcomes,
            rollouts_spent=self._spent,
        )
        self._iterations.append(record)
        produced = any(outcome.outcome in SCORED_OUTCOMES for outcome in outcomes)
        self._produced[block.name] = produced
        best = self._ranking()[0]
        self._log(
            "iteration_end",
            record=record.model_dump(mode="json"),
            block=block.name,
            exhausted=not produced,
            best_after=best,
            usage=batch.usage.model_dump(mode="json"),
        )
        objective = self._objective(best)
        self._progress(
            f"iteration {self._iteration} ({block.name}): {len(outcomes)} proposal(s), "
            f"best train objective {objective:.4f} (candidate {self._candidates[best].number}), "
            f"{self._spent} rollouts spent"
        )

    def _end_phase(self) -> dict[str, Any]:
        baseline = next(c for c in self._candidates.values() if c.origin == "baseline")
        finalists = [cid for cid in self._ranking() if cid != baseline.id][: self._settings.finalists]
        self._log("finalists", candidate_ids=finalists)
        contenders = [baseline.id, *finalists]
        validation = {cid: self._score(self._candidates[cid], "validation") for cid in contenders}
        report = self._pick(baseline.id, contenders, validation)
        self._log("selected", report=report.model_dump(mode="json"))
        test, test_reason = self._test_report(report.champion_id, baseline.id)
        self._log("test", report=None if test is None else test.model_dump(mode="json"), reason=test_reason, front={})
        champion = self._candidates[report.champion_id]
        self._progress(report.reason)
        return self._result(baseline, champion, report, test, test_reason)

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
        names = self._settings.split.test
        if not names:
            return None, "no test cases"
        ids = list(dict.fromkeys([champion_id, baseline_id]))
        pending = sum(len(self._missing(cid, names)) for cid in ids)
        if not self._can_afford(pending, "test"):
            return None, "the budget cannot cover the test report"
        scores = {cid: self._score(self._candidates[cid], "test") for cid in ids}
        champion, base = scores[champion_id], scores[baseline_id]
        assert champion is not None and base is not None
        metrics = self._criteria.goal_metrics
        report = TestReport(
            champion_id=champion_id,
            baseline_id=baseline_id,
            objective=paired(champion.objective_by_case, base.objective_by_case),
            difference={m: paired(_by_case(champion, m), _by_case(base, m)) for m in metrics},
            ratio={
                m: paired_ratio(_by_case(champion, m), _by_case(base, m), can_be_negative=self._can_be_negative(m))
                for m in metrics
            },
            missing={cid: sum(name not in s.objective_by_case for name in names) for cid, s in scores.items() if s},
            means={cid: measured(s) for cid, s in scores.items() if s is not None},
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
            "effects": {"rows": [], "reason": NO_SCORE_MODEL},
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

    def _score(self, candidate: Candidate, split: SplitName) -> CandidateScores | None:
        names = self._names(split)
        pending = self._missing(candidate.id, names)
        if not self._can_afford(len(pending), split):
            return None
        for name, run_index in pending:
            self._trial(candidate, name, run_index, split)
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
            rollouts=len(pending),
            objective_by_run=scores.objective_by_run,
            statuses=self._statuses(records),
            unavailable={case: {str(k): v for k, v in runs.items()} for case, runs in scores.unavailable.items()},
        )
        return scores

    def _trial(self, candidate: Candidate, name: str, run_index: int, split: SplitName) -> TrialRecord:
        case = self._cases[name]
        previous: list[AttemptRecord] = []
        while True:
            started = time.monotonic()
            try:
                result, status, error = self._rollout(candidate.values, case), TrialStatus.COMPLETED, None
            except Exception as exc:
                result, status, error = None, classify(exc), exc
            usage = _rollout_usage(time.monotonic() - started)
            self._spent += 1
            retry = status is TrialStatus.INFRASTRUCTURE_INVALID and len(previous) < self._settings.rerun_passes
            if not retry or not self._can_afford(1, split):
                break
            previous.append(_attempt(status, error, usage, len(previous)))
        rows = self._judge(case, result) if result is not None else []
        evaluations = {row.name: row.output for row in rows if row.error is None}
        if status is TrialStatus.COMPLETED and rows and not evaluations:
            status = TrialStatus.VERIFIER_INVALID
        record = TrialRecord(
            candidate_id=candidate.id,
            case_name=name,
            run_index=run_index,
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
            rollouts=1,
        )
        self._trials[(candidate.id, name, run_index)] = record
        data = record.model_dump(mode="json")
        data["transcript"] = self._run.save_transcript(data.pop("trajectory"))
        self._log("trial", record=data)
        return record

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

    def _split_scores(self, candidate_id: str, split: SplitName) -> CandidateScores:
        names = self._names(split)
        records = [self._trials[key] for key in self._keys(candidate_id, names) if key in self._trials]
        return score_split(candidate_id, records, names, self._criteria)

    def _can_afford(self, rollouts: int, phase: SplitName) -> bool:
        if self._replay:
            return rollouts == 0
        validation, test = self._settings.reserves()
        reserve = {"train": validation + test, "validation": test, "test": 0}[phase]
        return self._spent + rollouts + reserve <= self._settings.budget.max_rollouts

    def _remaining(self) -> int:
        return self._settings.budget.max_rollouts - self._spent

    def _names(self, split: SplitName) -> tuple[str, ...]:
        return getattr(self._settings.split, split)  # type: ignore[no-any-return]

    def _keys(self, candidate_id: str, names: Sequence[str]) -> list[tuple[str, str, int]]:
        return [(candidate_id, name, run) for name in names for run in range(self._settings.runs_per_candidate)]

    def _missing(self, candidate_id: str, names: Sequence[str]) -> list[tuple[str, int]]:
        return [
            (name, run)
            for _, name, run in self._keys(candidate_id, names)
            if (candidate_id, name, run) not in self._trials
        ]

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
        if not self._replay:
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


def _attempt(status: TrialStatus, error: BaseException | None, usage: Usage, pass_index: int) -> AttemptRecord:
    return AttemptRecord(
        status=status,
        error_type=type(error).__name__ if error else None,
        error=str(error) if error else None,
        usage=usage,
        judge_usage=zero_usage(),
        pass_index=pass_index,
        rollouts=1,
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
