"""What a run returns."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..case import Case
from ..evaluators.evaluator import Evaluator
from ..types.evaluation import EvaluationData
from ..types.evaluation_report import EvaluationReport
from ..types.optimization import (
    CandidateLabel,
    CaseSplit,
    IterationRecord,
    PairedComparison,
    SplitName,
    StopReason,
    TestReport,
    TrialRecord,
    TrialStatus,
    Usage,
    ValidationReport,
    Value,
)
from .candidate import Candidate, changed_values, diff
from .declaration.files import _write_text_atomically
from .declaration.space import Space
from .end_phase import TEST_LEVEL

Grain = Literal["iteration", "proposal", "case"]

_SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")
_LABELED = ("feasible", "infeasible", "ineligible")
_USAGE_FIELDS = (
    "tokens",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "model_calls",
    "seconds",
    "models",
)


class CandidateView(BaseModel):
    """A candidate as the results show it.

    Built by the optimizer at the end with `from_candidate`, which computes the diff and the changed values once,
    so a reloaded result needs no `Space`.

    Attributes:
        id: The content id, a hash of the values; the key everywhere.
        number: The display number (0 = baseline, then proposal order); for people only.
        parent_id: The candidate it was proposed from; None for the baseline.
        origin: The blocks whose patches it merges, in proposal order; `["baseline"]` for the baseline.
        rationale: Why its proposers changed what they did.
        values: Its typed values.
        diff_text: `diff()`'s text.
        changed: `changed_values()`'s mapping.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    number: int
    parent_id: str | None
    origin: list[str]
    rationale: str
    values: dict[str, Value]
    diff_text: str
    changed: dict[str, dict[str, Value]]

    @classmethod
    def from_candidate(
        cls, candidate: Candidate, baseline: Candidate, space: Space, *, number: int, root: Path
    ) -> CandidateView:
        """The view of `candidate`, its diff against `baseline` with paths shown relative to `root`.

        Args:
            candidate: The candidate shown.
            baseline: The run's baseline, for diffs.
            space: Maps file-backed keys to their real paths.
            number: Its display number.
            root: The run's working directory (`run.json` `workdir`).
        """
        return cls(
            id=candidate.id,
            number=number,
            parent_id=candidate.parent_id,
            origin=list(candidate.origin),
            rationale=candidate.rationale,
            values=dict(candidate.values),
            diff_text=diff(space, baseline, candidate, root=root),
            changed=changed_values(space, baseline, candidate),
        )

    def diff(self) -> str:
        """Unified diff of every changed text against the baseline, for review.

        File-backed values on their real paths (relative to the run's working directory), then non-file
        text values headed by their key. Numbers and choices are in `changed_values()`.
        """
        return self.diff_text

    def changed_values(self) -> dict[str, dict[str, Value]]:
        """Changed numbers and choices against the baseline: `{key: {"old": ..., "new": ...}}`, one side only
        for a key active on one side."""
        return {key: dict(sides) for key, sides in self.changed.items()}

    def write_diff(self, path: str | Path) -> Path:
        """Write this candidate's changes as the run writes the champion's (`best.diff`), for review.

        `path` gets `diff()`; `path.with_suffix(".values.json")` gets the changed numbers and choices as
        `{key: {"old": ..., "new": ...}}`, with one side only for a key active on one side. Both are always
        written (empty or `{}` when nothing of that kind changed). Applying is the CLI's job.

        Args:
            path: Where the diff goes.

        Returns:
            The diff path.
        """
        path = Path(path)
        _write_text_atomically(path, self.diff_text)
        values = json.dumps(self.changed, indent=2, ensure_ascii=False, allow_nan=False)
        _write_text_atomically(path.with_suffix(".values.json"), values + "\n")
        return path


class OptimizationResult(BaseModel):
    """What a run returns, saved and reloaded whole (`to_file` / `from_file`; the run writes `<run>/result.json`).

    Attributes:
        run_id: The run folder's name.
        run_dir: The run folder; it may not exist on another machine, and nothing here reads it.
        baseline_id: The baseline's id.
        candidates: Every saved candidate by id, in display-number order (unscored ones too).
        validation: How the champion was picked on validation: every finalist's objective and estimates, and
            why any was infeasible.
        test: The champion against the baseline on test, report only: a negative `mean_diff` is possible
            and never changes what ships. When the baseline ships it is compared with itself. None without
            test cases or when it couldn't be afforded (`test_reason` says which).
        test_reason: Why `test` is None (the `test` event's reason), or under `Pareto` which front members
            the test report couldn't afford; None when everything ran.
        front_test: Under a `Pareto` objective, each tested front member against the baseline on test
            (`front_test[best.id]` is `test`); `{}` otherwise or when the test report was skipped.
        effects: `effects.json`: per metric and option from the search fit, or empty with the reason.
        iterations: Every search step, in order.
        rollouts_spent: Rollouts counted against the budget, the end phase's included.
        stop_reason: Why the search stopped: `budget` (what's left after the end-phase reserves can't score
            one more candidate), `iterations` (the cap), `exhausted` (the schedule had nothing left), or
            `proposer_error` (a proposer raised). The end phase runs after any of them. Never `setup`: a setup
            stop raises instead of returning.
        agent_cost: The agent's own usage per split (each trial's last attempt): train = the search,
            validation and test = the end phase.
        search_cost: Total proposer usage.
        judge_cost: Judge usage per trial's last attempt, as `agent_cost` counts the agent's.
        retry_cost: Agent and judge usage in earlier attempts (`TrialRecord.previous_attempts`).
        reports: Candidate id -> split -> the `EvaluationReport` `Experiment` would have returned for those
            trials (every run's rows, each tagged with `run_index`; failed trials as `Experiment`'s score-0
            rows), for every candidate and split it has trials on. Display only: nothing in the run reads it.
        rows: Every grain's rows (`to_rows`), computed when the result was built.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    run_dir: Path
    baseline_id: str
    candidates: dict[str, CandidateView]
    validation: ValidationReport
    test: TestReport | None
    test_reason: str | None
    front_test: dict[str, TestReport]
    effects: dict[str, Any]
    iterations: list[IterationRecord]
    rollouts_spent: int
    stop_reason: StopReason
    agent_cost: dict[SplitName, Usage]
    search_cost: Usage
    judge_cost: Usage
    retry_cost: Usage
    reports: dict[str, dict[SplitName, EvaluationReport]]
    rows: dict[Grain, list[dict[str, Any]]]

    @property
    def baseline(self) -> CandidateView:
        """The baseline's view."""
        return self.candidates[self.baseline_id]

    @property
    def best(self) -> CandidateView:
        """The champion, `validation.champion_id` (the baseline when nothing feasible beats it)."""
        return self.candidates[self.validation.champion_id]

    @property
    def front(self) -> list[CandidateView]:
        """Under a `Pareto` objective, the validated front (`validation.front`), champion first, each with
        `diff()` and values, so a reviewer can apply another trade-off; `[]` otherwise."""
        return [self.candidates[c] for c in self.validation.front]

    @property
    def improved(self) -> bool:
        """Whether something other than the baseline ships (`best.id != baseline_id`)."""
        return self.validation.champion_id != self.baseline_id

    def summary(self) -> str:
        """The run in a few lines: why the search stopped, what ships and why, every finalist on validation
        (the validation fit's estimate next to the measured mean), under `Pareto` the validated front, the
        test report, what changed, and the costs."""
        goals = list(dict.fromkeys(goal["metric"] for goal in self.effects.get("goals", [])))
        v = self.validation
        steps = len(self.iterations)
        lines = [
            f"Run {self.run_id}: the search stopped ({self.stop_reason}) after {steps} step"
            f"{'' if steps == 1 else 's'}, {self.rollouts_spent} rollouts.",
        ]
        if self.improved:
            best = self.best
            parent = "?" if best.parent_id is None else self._name(best.parent_id)
            lines.append(f"Ships {self._name(best.id)}, proposed from {parent} by {' + '.join(best.origin)}.")
        else:
            lines.append("Ships the baseline.")
        lines.append(v.reason)

        lines += ["", "Validation" + (f" (measured means: {v.problem})" if v.problem else "") + ":"]
        table = [["", "candidate", "objective", *goals]]
        for c, objective in v.objectives.items():
            mark = "*" if c == v.champion_id else ""
            note = f"  infeasible: {v.infeasible[c]}" if c in v.infeasible else ""
            table.append(
                [mark, self._name(c), _number(objective), *(self._metric_cell(c, m) for m in goals)]
                + ([note] if note else [])
            )
        lines += _table(table)

        if v.front:
            objective_metrics = [g["metric"] for g in self.effects.get("goals", []) if g["role"] == "objective"]
            lines += ["", "Validated front (* = champion; test: difference vs the baseline):"]
            table = [["", "candidate", *goals, *(f"test {m}" for m in objective_metrics)]]
            for c in v.front:
                report = self.front_test.get(c)
                table.append(
                    ["*" if c == v.champion_id else "", self._name(c)]
                    + [self._metric_cell(c, m) for m in goals]
                    + [
                        "-" if report is None or m not in report.difference else _comparison(report.difference[m])
                        for m in objective_metrics
                    ]
                )
            lines += _table(table)

        lines.append("")
        if self.test is None:
            lines.append(f"Test: skipped ({self.test_reason}).")
        else:
            t = self.test
            versus = "the baseline against itself" if t.champion_id == t.baseline_id else "the champion vs the baseline"
            lines.append(f"Test, {versus} (paired {round(TEST_LEVEL * 100)}% interval; reported only):")
            table = [["objective", "-" if t.objective is None else _comparison(t.objective), ""]]
            for metric, compared in t.difference.items():
                ratio = t.ratio.get(metric)
                table.append([metric, _comparison(compared), "" if ratio is None else f"ratio {_comparison(ratio)}"])
            lines += _table(table)
            if self.test_reason:
                lines.append(self.test_reason)

        if self.improved:
            best = self.best
            texts = sum(1 for line in best.diff_text.splitlines() if line.startswith("+++ "))
            lines += ["", f"Changes: {texts} text value{'' if texts == 1 else 's'} (diff())"]
            for key, sides in best.changed.items():
                old = repr(sides["old"]) if "old" in sides else "(inactive)"
                new = repr(sides["new"]) if "new" in sides else "(inactive)"
                lines.append(f"  {key}: {old} -> {new}")

        cost = self.agent_cost
        lines += [
            "",
            f"Agent tokens: {cost['train'].tokens} train (the search), {cost['validation'].tokens} validation, "
            f"{cost['test'].tokens} test. Proposers {self.search_cost.tokens}, judges {self.judge_cost.tokens}, "
            f"earlier attempts {self.retry_cost.tokens}.",
        ]
        if self.effects.get("reason"):
            lines.append(f"Effects: none ({self.effects['reason']}).")
        return "\n".join(lines)

    def _name(self, candidate_id: str) -> str:
        """How the summary names a candidate: "the baseline" or "candidate 3 (<id>)"."""
        if candidate_id == self.baseline_id:
            return "the baseline"
        return f"candidate {self.candidates[candidate_id].number} ({candidate_id})"

    def _metric_cell(self, candidate_id: str, metric: str) -> str:
        """A validation cell: the fit's estimate with its sd and the measured mean, or the measured mean alone."""
        estimate = self.validation.estimates.get(candidate_id, {}).get(metric)
        measured = self.validation.measured.get(candidate_id, {}).get(metric)
        if estimate is not None:
            text = f"{estimate.mean:.3g} ± {estimate.sd:.2g}"
            return text + (f" (measured {measured.mean:.3g})" if measured is not None else "")
        return "-" if measured is None else f"{measured.mean:.3g} (measured)"

    def to_rows(self, grain: Grain = "proposal") -> list[dict[str, Any]]:
        """The run as flat rows, built from the event log, so a resumed run's rows cover the whole run.

        Every row of one grain has the same keys; a value that doesn't apply is None. Per-metric columns
        are named `<prefix>.<metric>`, e.g. `train.HelpfulnessEvaluator`. `pandas.DataFrame(rows)` works.

        Args:
            grain: What one row is: `proposal` (every proposal, the baseline included, with its label at
                its step and its rank under the last search fit), `iteration` (one search step), or
                `case` (one trial: candidate x case x run, on every split it was scored on).

        Raises:
            ValueError: An unknown grain.
        """
        if grain not in self.rows:
            raise ValueError(f"unknown grain {grain!r}; use 'proposal', 'iteration', or 'case'")
        return [dict(row) for row in self.rows[grain]]

    def candidate(self, candidate_id: str) -> CandidateView:
        """Any saved candidate by id; unknown ids raise `KeyError`."""
        return self.candidates[candidate_id]

    def to_dict(self) -> dict[str, Any]:
        """The whole result as JSON values (`model_dump(mode="json")`)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizationResult:
        """The result `to_dict` gave."""
        return cls.model_validate(data)

    def to_file(self, path: str | Path) -> Path:
        """Write `to_dict()` as JSON to `path`; returns `path`."""
        path = Path(path)
        _write_text_atomically(path, json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        return path

    @classmethod
    def from_file(cls, path: str | Path) -> OptimizationResult:
        """The result `to_file` wrote."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def _comparison(compared: PairedComparison) -> str:
    """A paired difference or ratio as text: the mean, its interval, and the cases."""
    interval = "" if compared.ci_low is None else f" [{compared.ci_low:+.3g}, {compared.ci_high:+.3g}]"
    return f"{compared.mean_diff:+.3g}{interval} over {compared.n_cases} case{'' if compared.n_cases == 1 else 's'}"


def _table(rows: list[list[str]]) -> list[str]:
    """Rows of cells as left-aligned columns, indented two spaces."""
    widths = [max(len(row[i]) for row in rows if i < len(row)) for i in range(max(map(len, rows)))]
    return ["  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]


def _proposal_rows(
    events: Sequence[Mapping[str, Any]],
    *,
    iterations: Sequence[IterationRecord],
    candidates: Mapping[str, CandidateView],
    baseline_id: str,
    final: Mapping[str, tuple[float | None, int]],
    trial_counts: Mapping[str, int],
    failures: Mapping[str, int],
    metrics: Sequence[str],
    goal_metrics: Sequence[str],
    constraints: Sequence[str],
    validation: ValidationReport,
    finalists: Sequence[str],
    is_pareto: bool,
) -> list[dict[str, Any]]:
    """One row per proposal, the baseline first, then each iteration's proposals in proposal order (API.md
    section 10, "Result rows").

    Args:
        events: The closed events (`closed_events`), so a killed iteration that resume re-ran shows once.
        iterations: Every search step, in order.
        candidates: Every saved candidate's view.
        baseline_id: The baseline's id.
        final: Eligible candidate id -> its objective and rank under the last search fit.
        trial_counts: Candidate id -> its counted train trials.
        failures: Candidate id -> its `agent_failure` train trials.
        metrics: Every metric, for the measured columns.
        goal_metrics: The goal metrics, for the estimate columns.
        constraints: Each constraint as text, for the `p_met` columns.
        validation: The end phase's validation report.
        finalists: The end phase's finalists.
        is_pareto: Whether the objective is a `Pareto` (else `on_front` is None).
    """
    proposed: dict[tuple[int, str], Mapping[str, Any]] = {}
    labels: dict[tuple[int, str], CandidateLabel] = {}
    rollouts: Counter[tuple[int, str]] = Counter()
    unscored: dict[tuple[int, str], Mapping[str, Any]] = {}
    repeated: set[tuple[int, str]] = set()
    invalid: dict[int, list[Mapping[str, Any]]] = {}
    for event in events:
        name, iteration = event["event"], event["iteration"]
        if name == "propose":
            proposed[(iteration, event["candidate_id"])] = event
        elif name == "label":
            labels[(iteration, event["candidate_id"])] = CandidateLabel.model_validate(event["label"])
        elif name == "score" and event["split"] == "train":
            rollouts[(iteration, event["candidate_id"])] += event["rollouts"]
        elif name == "unscored":
            unscored[(iteration, event["candidate_id"])] = event
        elif name == "duplicate":
            repeated.add((iteration, event["candidate_id"]))
        elif name == "invalid":
            invalid.setdefault(iteration, []).append(event)

    def row(
        iteration: int,
        record: IterationRecord | None,
        candidate_id: str | None,
        outcome: str,
        reason: str | None,
        proposal: Mapping[str, Any],
    ) -> dict[str, Any]:
        label = labels.get((iteration, candidate_id)) if outcome in _LABELED and candidate_id is not None else None
        scored = label is not None
        view = None if candidate_id is None else candidates.get(candidate_id)
        out: dict[str, Any] = {
            "iteration": iteration,
            "round": None if record is None or record.round is None else list(record.round),
            "role": None if record is None else record.role,
            "candidate_id": candidate_id,
            "number": None if view is None else view.number,
            "parent_id": None if record is None else proposal.get("parent_id"),
            "block": None if record is None else "+".join(proposal["origin"]),
            "outcome": outcome,
            "reason": reason,
            "rescore": bool(proposal.get("rescore", False)),
            "changed_keys": list(proposal.get("changed_keys", [])),
            "inherited": dict(proposal.get("inherited", {})),
            "defaulted": list(proposal.get("defaulted", [])),
            "rationale": proposal.get("rationale") or None,
            "is_from_fit": label.is_from_fit if label is not None else None,
            "objective_at_step": proposal.get("objective") if scored else None,
            "unmet_constraints": list(label.unmet_constraints) if label is not None else None,
        }
        for c in constraints:
            out[f"p_met.{c}"] = label.p_met.get(c) if label is not None else None
        for m in goal_metrics:
            estimate = label.estimates.get(m) if label is not None else None
            out[f"estimate.{m}"] = None if estimate is None else estimate.mean
            out[f"estimate_sd.{m}"] = None if estimate is None else estimate.sd
        for m in metrics:
            measured = label.measured.get(m) if label is not None else None
            out[f"train.{m}"] = None if measured is None else measured.mean
            out[f"train_ci95.{m}"] = None if measured is None or measured.ci95 is None else list(measured.ci95)
            out[f"train_n.{m}"] = None if measured is None else measured.n
        objective, rank = final.get(candidate_id, (None, None)) if scored else (None, None)  # type: ignore[arg-type]
        out.update(
            objective=objective,
            rank=rank,
            trials=trial_counts.get(candidate_id) if scored else None,  # type: ignore[arg-type]
            failures=failures.get(candidate_id) if scored else None,  # type: ignore[arg-type]
            rollouts=rollouts[(iteration, candidate_id)] if scored else 0,  # type: ignore[index]
            finalist=candidate_id in finalists if scored else None,
            champion=candidate_id == validation.champion_id if scored else None,
            validation_objective=validation.objectives.get(candidate_id) if scored else None,  # type: ignore[arg-type]
            on_front=candidate_id in validation.front if scored and is_pareto else None,
            values=None if view is None else dict(view.values),
            diff=None if view is None else view.diff_text,
        )
        return out

    baseline = labels.get((0, baseline_id))
    rows = [
        row(
            0,
            None,
            baseline_id,
            baseline.outcome if baseline is not None else "feasible",
            None if baseline is None else baseline.reason,
            proposed.get((0, baseline_id), {}),
        )
    ]
    for record in iterations:
        queue = list(invalid.get(record.iteration, []))
        for p in record.proposals:
            key = (record.iteration, p.candidate_id or "")
            fields: dict[str, Any] = {
                "parent_id": p.parent_id,
                "origin": p.origin,
                "changed_keys": p.changed_keys,
                "rationale": p.rationale,
                "objective": p.objective,
            }
            reason: str | None
            if p.outcome == "invalid":
                taken, queue = queue[: max(1, len(p.origin))], queue[max(1, len(p.origin)) :]
                reason = taken[0]["reason"] if taken else None
            elif p.outcome == "duplicate":
                reason = f"repeats candidate {p.number}" if key in repeated else "repeats a candidate of the same step"
            elif p.outcome == "unscored":
                event = unscored.get(key, {})
                reason = (
                    f"the budget couldn't cover it: {event.get('pending')} rollouts needed, "
                    f"{event.get('remaining')} left, {event.get('reserve')} reserved for the end phase"
                )
                fields.update(
                    {k: v for k, v in proposed.get(key, {}).items() if k in ("rescore", "inherited", "defaulted")}
                )
            else:
                label = labels.get(key)
                reason = None if label is None else label.reason
                saved = proposed.get(key, {})
                fields.update(
                    {k: saved[k] for k in ("rescore", "changed_keys", "inherited", "defaulted") if k in saved}
                )
            rows.append(row(record.iteration, record, p.candidate_id, p.outcome, reason, fields))
    return rows


def _iteration_rows(
    events: Sequence[Mapping[str, Any]],
    *,
    iterations: Sequence[IterationRecord],
    candidates: Mapping[str, CandidateView],
) -> list[dict[str, Any]]:
    """One row per search step (API.md section 10, "Result rows").

    Args:
        events: The closed events (`closed_events`).
        iterations: Every search step, in order.
        candidates: Every saved candidate's view.
    """
    fits: dict[int, Mapping[str, Any]] = {}
    ends: dict[int, Mapping[str, Any]] = {}
    tokens: Counter[int] = Counter()
    calls: Counter[int] = Counter()
    opened: dict[str, int] = {}  # a round opener's patch -> the step that proposed it (it is observed later)
    baseline_rollouts = 0
    for event in events:
        name, iteration = event["event"], event["iteration"]
        if name == "fit":
            fits[iteration] = event
        elif name == "iteration_end":
            ends[iteration] = event
        elif name in ("propose", "duplicate") and event.get("patch") is not None:
            opened.setdefault(json.dumps(event["patch"], sort_keys=True), iteration)
        elif name == "score" and iteration == 0 and event["split"] == "train":
            baseline_rollouts += event["rollouts"]
    for event in events:
        usage = None
        if event["event"] == "observe":
            patch = event["observation"]["patch"]
            usage = patch.get("usage")
            iteration = opened.get(json.dumps(patch, sort_keys=True), event["iteration"])
        elif event["event"] == "proposer_usage":
            usage, iteration = event["usage"], event["iteration"]
        if usage is not None:
            tokens[iteration] += usage["tokens"]
            calls[iteration] += usage["model_calls"]
    rows = []
    total = baseline_rollouts
    for record in iterations:
        total += record.rollouts_spent
        outcomes: Counter[str] = Counter(p.outcome for p in record.proposals)
        fit = fits.get(record.iteration, {})
        end = ends.get(record.iteration, {})
        row: dict[str, Any] = {
            "iteration": record.iteration,
            "round": None if record.round is None else list(record.round),
            "role": record.role,
            "parent_id": record.parent_id,
            "parent_number": candidates[record.parent_id].number,
            "blocks": list(record.blocks),
            "proposals": len(record.proposals),
        }
        row.update({outcome: outcomes[outcome] for outcome in (*_LABELED, "invalid", "duplicate", "unscored")})
        row.update(
            rollouts=record.rollouts_spent,
            rollouts_total=total,
            is_fitted=bool(fit.get("is_fitted", False)),
            fit_error=fit.get("error"),
            flagged=list(fit.get("flagged", [])),
            best_after=list(end.get("best_after", [])),
            best_objective=end.get("best_objective"),
            exhausted=list(end.get("exhausted", [])),
            search_tokens=tokens[record.iteration],
            search_model_calls=calls[record.iteration],
        )
        rows.append(row)
    return rows


def _case_rows(
    events: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[tuple[str, int], TrialRecord]],
    *,
    candidates: Mapping[str, CandidateView],
    split: CaseSplit,
    evaluators: Sequence[str],
) -> list[dict[str, Any]]:
    """One row per trial (candidate x case x run), the last attempt per key (API.md section 10, "Result rows").

    Args:
        events: The closed events (`closed_events`), for each trial's objective and transcript.
        records: Candidate id -> (case name, run index) -> its last trial record.
        candidates: Every saved candidate's view, for the order (display number) and numbers.
        split: The run's split, for the case order.
        evaluators: Every evaluator's name, for the per-evaluator columns.
    """
    scores: dict[tuple[str, str], Mapping[str, Any]] = {}
    transcripts: dict[tuple[str, int, str], str | None] = {}
    for event in events:
        if event["event"] == "score":
            scores[(event["candidate_id"], event["split"])] = event
        elif event["event"] == "trial":
            saved = event["record"]
            transcript = saved.get("transcript")
            key = (saved["candidate_id"], saved["run_index"], saved["case_name"])
            transcripts[key] = None if transcript is None else transcript["hash"]
    rows = []
    for candidate_id, view in candidates.items():
        trials = records.get(candidate_id, {})
        for split_name in _SPLITS:
            score = scores.get((candidate_id, split_name), {})
            for case_name in getattr(split, split_name):
                runs = sorted(index for case, index in trials if case == case_name)
                for index in runs:
                    record = trials[(case_name, index)]
                    by_run = score.get("objective_by_run", {}).get(case_name, [])
                    unavailable = score.get("unavailable", {}).get(case_name, {})
                    row: dict[str, Any] = {
                        "candidate_id": candidate_id,
                        "number": view.number,
                        "split": split_name,
                        "case_name": case_name,
                        "run_index": index,
                        "status": record.status.value,
                        "error_type": record.error_type,
                        "error": record.error,
                        "attempts": record.attempts,
                        "objective": by_run[index] if index < len(by_run) else None,
                    }
                    for kind in ("score", "test_pass", "label", "reason"):
                        for name in evaluators:
                            evaluation = record.evaluations.get(name)
                            row[f"{kind}.{name}"] = None if evaluation is None else getattr(evaluation, kind)
                    for name in evaluators:
                        evaluation = record.evaluations.get(name)
                        row[f"applicable.{name}"] = (
                            None
                            if evaluation is None
                            else EvaluationReport.is_applicable(record.details.get(name) or [evaluation])
                        )
                    usage = record.usage
                    row.update(
                        evaluator_errors=dict(record.evaluator_errors),
                        unavailable=dict(unavailable.get(str(index), {})),  # JSON keys: the run index as text
                        **{field: getattr(usage, field) for field in _USAGE_FIELDS},
                        judge_tokens=sum(u.tokens for u in record.judge_usage.values()),
                        output=record.output,
                        transcript=transcripts.get((candidate_id, index, case_name)),
                    )
                    row["models"] = list(usage.models)  # a copy, not the record's list
                    rows.append(row)
    return rows


def _reports(
    records: Mapping[str, Mapping[tuple[str, int], TrialRecord]],
    *,
    cases: Sequence[Case],
    evaluators: Sequence[Evaluator],
    split: CaseSplit,
) -> dict[str, dict[SplitName, EvaluationReport]]:
    """Each candidate's `EvaluationReport` per split, as `Experiment.run_evaluations_async` flattens it.

    Rows in evaluator order, then the split's case order, then run index. A trial whose task raised
    (`agent_failure`, `infrastructure_invalid`) gets a score-0 row per evaluator with `Experiment`'s reason
    ("An error occurred: <error>") and the case itself as the row; an evaluator in `evaluator_errors` gets a
    score-0 row with "Evaluator error: <exception type>" (the record keeps only the type). Each row is tagged
    `evaluator`, `evaluator_type`, and `run_index`. So `overall_score` is what a plain eval run of the same
    trials reports.

    Args:
        records: Candidate id -> (case name, run index) -> its last trial record.
        cases: The experiment's cases.
        evaluators: The experiment's evaluators, in its order.
        split: The run's split, for the case order.
    """
    by_name = {case.name: case for case in cases}
    out: dict[str, dict[SplitName, EvaluationReport]] = {}
    for candidate_id, trials in records.items():
        for split_name in _SPLITS:
            ordered = [
                trials[(case_name, index)]
                for case_name in getattr(split, split_name)
                for index in sorted(i for c, i in trials if c == case_name)
            ]
            if not ordered:
                continue
            scores: list[float] = []
            passes: list[bool] = []
            rows: list[dict] = []
            reasons: list[str] = []
            detailed: list[list[Any]] = []
            for evaluator in evaluators:
                name, type_name = evaluator.get_name(), evaluator.get_type_name()
                for record in ordered:
                    case = by_name[record.case_name]
                    tag = {"evaluator": name, "evaluator_type": type_name, "run_index": record.run_index}
                    if record.status in (TrialStatus.AGENT_FAILURE, TrialStatus.INFRASTRUCTURE_INVALID):
                        rows.append({**case.model_dump(), **tag})
                        scores.append(0.0)
                        passes.append(False)
                        reasons.append(f"An error occurred: {record.error}")
                        detailed.append([])
                        continue
                    data = EvaluationData(
                        input=case.input,
                        actual_output=record.output,
                        name=case.name,
                        expected_output=case.expected_output,
                        expected_assertion=case.expected_assertion,
                        expected_trajectory=case.expected_trajectory,
                        actual_trajectory=record.trajectory,
                        metadata=case.metadata,
                        actual_interactions=record.interactions,
                        expected_interactions=case.expected_interactions,
                        actual_environment_state=record.environment_state,
                        expected_environment_state=case.expected_environment_state,
                    )
                    rows.append({**data.model_dump(), **tag})
                    evaluation = record.evaluations.get(name)
                    if evaluation is None:
                        error = record.evaluator_errors.get(name, "not run")
                        scores.append(0.0)
                        passes.append(False)
                        reasons.append(f"Evaluator error: {error}")
                        detailed.append([])
                    else:
                        scores.append(float(evaluation.score))
                        passes.append(evaluation.test_pass)
                        reasons.append(evaluation.reason or "")
                        detailed.append(list(record.details.get(name, [])))
            out.setdefault(candidate_id, {})[split_name] = EvaluationReport(
                overall_score=EvaluationReport.calculate_overall_score(scores, detailed),
                scores=scores,
                cases=rows,
                test_passes=passes,
                reasons=reasons,
                detailed_results=detailed,
                diagnoses=[None] * len(scores),
                recommendations=[None] * len(scores),
            )
    return out
