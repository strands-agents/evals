"""What a run returns."""

from __future__ import annotations

import json
import math
import textwrap
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from ..case import Case
from ..evaluators.evaluator import Evaluator
from ..types.evaluation import EvaluationData
from ..types.evaluation_report import EvaluationReport
from ..types.optimization import (
    LABEL_OUTCOMES,
    SPLIT_NAMES,
    CandidateLabel,
    CaseSplit,
    IterationRecord,
    Measured,
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
from .candidate import Candidate, changed_keys, changed_values, diff
from .declaration.files import _write_text_atomically
from .declaration.space import Space
from .end_phase import TEST_LEVEL
from .scoring.metrics import USAGE_METRICS

Grain = Literal["iteration", "proposal", "case"]


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
        candidates: Every candidate of the run's closed iterations by id, in display-number order (a killed
            iteration's candidate that the resumed run didn't propose again is left out).
        validation: How the champion was picked on validation: every finalist's objective and estimates, and
            why any was infeasible.
        test: The champion against the baseline on test, report only: a negative `mean_diff` is possible
            and never changes what ships. When the baseline ships it is compared with itself. None without
            test cases, when it couldn't be afforded, or when every test trial of the champion or the baseline
            failed (`test_reason` says which). When it couldn't be afforded, or each of the champion and the
            baseline with no completed test trial has one left `infrastructure_invalid`, the `test` event isn't
            written, so `Optimizer.resume` on the run runs the test again (with a `Budget` that affords it).
        test_reason: Why `test` is None, or under `Pareto` which front members the test report couldn't
            afford or measured nothing for; None when everything ran.
        front_test: Under a `Pareto` objective, each tested front member against the baseline on test
            (`front_test[best.id]` is `test` when the champion is on the validated front; members not afforded
            or with every test trial failed are absent); `{}` otherwise or when the test report was skipped.
        test_failed_ids: The tested candidates (the champion, the baseline, and under `Pareto` the front members
            tested) with no completed test trial, so the test measured nothing for them. The champion is in it
            when every one of its test trials failed, though it ships. `[]` when every tested candidate
            completed a test trial or the test didn't run, and in a result written before this field existed.
        effects: `effects.json`: per metric and option from the search fit, or empty with the reason.
        objective: The objective as `run.json` records it, e.g. "Maximize(WordCount)"; None in a result
            written before this field existed (as for the four fields after it and `max_rollouts`).
        constraints: Each constraint as `run.json` records it, e.g. "tokens <= 4000", in declaration order.
        feasibility_threshold: The P(met) a constraint needs to count as met (`Optimizer(feasibility_threshold=)`).
        proposers: Block name -> its proposer's class name, in the search's block order.
        iterations: Every search step, in order.
        rollouts_spent: Rollouts counted against the budget, the end phase's included.
        max_rollouts: The budget's `max_rollouts` the run ended under.
        stop_reason: Why the search stopped: `budget` (what's left after the end-phase reserves can't score
            one more candidate), `iterations` (the cap), or `exhausted` (the schedule had nothing left). The end
            phase runs after any of them. Never `setup`: a setup stop raises instead of returning, and so does a
            proposer that raises (`ProposerError`, before the end phase).
        agent_cost: The agent's own usage per split (each trial's last attempt): train = the search,
            validation and test = the end phase.
        search_cost: Total proposer usage.
        judge_cost: Judge usage per trial's last attempt, as `agent_cost` counts the agent's.
        retry_cost: Agent and judge usage in earlier attempts (`TrialRecord.previous_attempts`).
        benchmark_cost: Usage of the other agents in the rollouts (`TrialRecord.benchmark_usage`: a
            simulated user, an agent the environment built), in every attempt; never counted in
            `agent_cost` or `retry_cost`. Zero when the agents can't be told apart (the `build_task` path, a
            factory that returned something else).
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
    test_failed_ids: list[str] = Field(default_factory=list)
    effects: dict[str, Any]
    objective: str | None = None
    constraints: list[str] = Field(default_factory=list)
    feasibility_threshold: float | None = None
    proposers: dict[str, str] = Field(default_factory=dict)
    iterations: list[IterationRecord]
    rollouts_spent: int
    max_rollouts: int | None = None
    stop_reason: StopReason
    agent_cost: dict[SplitName, Usage]
    search_cost: Usage
    judge_cost: Usage
    retry_cost: Usage
    benchmark_cost: Usage = Field(default_factory=Usage.zero)
    # Left out of the repr: every trial's rows and evaluation data, hundreds of kilobytes on a real run.
    reports: dict[str, dict[SplitName, EvaluationReport]] = Field(repr=False)
    rows: dict[Grain, list[dict[str, Any]]] = Field(repr=False)

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
        """Whether something other than the baseline ships (`best.id != baseline_id`).

        The validation pick decides it; the test never changes it. Check `test` (None when skipped, with
        `test_reason`) and `test_failed_ids` (holding `best.id` when every test trial of the champion failed)
        before applying `best`.
        """
        return self.validation.champion_id != self.baseline_id

    def summary(self) -> str:
        """The run as a header of recorded values, then tables: validation (one row per candidate, or per
        candidate and goal when there are several goals), why each candidate that can't ship can't, the test
        report, under `Pareto` each front member against the baseline on test, the changed values when a number
        or choice changed, and the tokens; then notes on what the columns hold."""
        v, best = self.validation, self.best
        goals = self.effects["goals"]
        first = [("Objective", self.objective or "-")]
        if self.constraints:
            threshold = "-" if self.feasibility_threshold is None else f"{self.feasibility_threshold:g}"
            first += [("Constraints", "; ".join(self.constraints)), ("P(met) >=", threshold)]
        rollouts = f"{self.rollouts_spent:,}" + ("" if self.max_rollouts is None else f" / {self.max_rollouts:,}")
        pairs = [
            ("Stop reason", self.stop_reason, "Rollouts", rollouts),
            ("Iterations", str(len(self.iterations)), "Pick rule", "-" if v.rule is None else _rule_name(v.rule)),
            ("Champion", self._label(best.id), "Outcome", v.outcome),
        ]
        missing = self.test.missing if self.test is not None else {}
        notes = [
            ("Validation fit", v.problem),
            ("Effects", self.effects.get("reason")),
            ("Held-out check", self._flagged_note()),
            ("Test reason", self.test_reason),
            (
                "Test missing",
                "; ".join(f"{self._label(c)}: {n} case{'' if n == 1 else 's'}" for c, n in missing.items() if n)
                or None,
            ),
        ]
        last = [
            ("Parent", self._label(best.parent_id)),
            # Every block of the search with its proposer, whichever of them the champion came from.
            ("Proposers", ", ".join(f"{block} ({name})" for block, name in self.proposers.items()) or "-"),
            ("Changed", ", ".join(changed_keys(self.baseline.values, best.values)) or "-"),
        ]
        last += [(label, note.splitlines()[0]) for label, note in notes if note]  # an exception message can run long
        width = max(len(label) for label, *_ in [*first, *pairs, *last]) + 3

        def pair(label: str, value: str) -> str:
            return f"{f'{label}:':<{width}}{value}"

        # The right-hand pairs start at column 38, so short values keep one layout from run to run, or after the
        # longest left value.
        left = max(38, *(width + len(value) + 3 for _, value, _, _ in pairs))
        right = max(len(label) for _, _, label, _ in pairs) + 3
        paired = [pair(a, value).ljust(left) + f"{f'{b}:':<{right}}{other}" for a, value, b, other in pairs]
        header = [pair(*item) for item in first] + paired + [pair(*item) for item in last]

        tables = [self._validation_table(goals)]
        if v.infeasible:
            tables.append(self._infeasible_table())
        lone = sum(goal["role"] == "objective" for goal in goals) == 1
        if self.test is not None:
            tables.append(_test_table(self.test, lone=lone))
        if v.front:
            tables.append(self._front_table(goals))
        if self.improved and best.changed:
            tables.append(self._changes_table())
        tables.append(self._tokens_table())

        # The rules span the two-column header and the tables; a full-width value may run past them.
        rule = max(len(line) for line in [*paired, *(line for table in tables for line in table)])
        lines = [f"Optimization result: {self.run_id}", "=" * rule, *header]
        for table in tables:
            lines += ["=" * rule, table[0], "-" * rule, *table[1:]]
        return "\n".join([*lines, "=" * rule, *self._notes(tables, width=rule)])

    def _label(self, candidate_id: str | None) -> str:
        """How the summary names a candidate: "baseline", "candidate 3", or "-" for none."""
        if candidate_id is None:
            return "-"
        if candidate_id == self.baseline_id:
            return "baseline"
        return f"candidate {self.candidates[candidate_id].number}"

    def _flagged_note(self) -> str | None:
        """Which metrics the held-out check flagged in the search and how the search ranked from then on; None
        when none was flagged or there are no effects (the `Effects` note says why)."""
        metrics = self.effects.get("metrics", {})
        flagged = [m for m, d in metrics.items() if d.get("diagnostics", {}).get("flagged")]
        if not flagged or self.effects.get("reason"):
            return None
        names = ", ".join(flagged)
        if len(flagged) == len(metrics):
            return f"flagged {names}; from then on the search ranked by measured means, not the fit"
        them = "it" if len(flagged) == 1 else "them"
        return f"flagged {names}; from then on the search ranked {them} by measured means, the other metrics by the fit"

    def _validation_table(self, goals: Sequence[Mapping[str, Any]]) -> list[str]:
        """Every candidate on validation, in the pick's order (the baseline, then the finalists): one row each
        with one goal, else one row per candidate and goal (each objective metric, then each constraint with
        its P(met)). The objective column is left out when it would print its lone metric's value; the adjusted
        column is shown when a metric the fit didn't judge has an adjusted mean other than its measured one."""
        v = self.validation
        objectives = [goal for goal in goals if goal["role"] == "objective"]
        constraints = sorted((g for g in goals if g["role"] == "constraint"), key=lambda g: g["index"])
        # A result written before `constraints` existed labels each constraint by its metric.
        texts = self.constraints if len(self.constraints) == len(constraints) else [g["metric"] for g in constraints]
        # Each goal row: its label, its metric, and the constraint it checks.
        rows: list[tuple[str, str, str | None]] = [(g["metric"], g["metric"], None) for g in objectives]
        rows += [(text, g["metric"], text) for g, text in zip(constraints, texts, strict=True)]
        decimals = {
            metric: _decimals(
                [e.mean for c in v.objectives if (e := v.estimates.get(c, {}).get(metric)) is not None]
                + [a for c in v.objectives if (a := v.adjusted_means.get(c, {}).get(metric)) is not None]
                + [m.mean for c in v.objectives if (m := v.measured.get(c, {}).get(metric)) is not None]
            )
            for _, metric, _ in rows
        }
        show_adjusted = any(
            _number(adjusted, decimals[metric]) != _number(measured.mean, decimals[metric])
            for c in v.objectives
            for _, metric, _ in rows
            if (adjusted := v.adjusted_means.get(c, {}).get(metric)) is not None
            and (measured := v.measured.get(c, {}).get(metric)) is not None
        )
        objective_decimals = _decimals(list(v.objectives.values()))
        show_objective = len(objectives) != 1 or self._objective_differs(objectives[0], objective_decimals)
        statuses = {c: self._status(c) for c in v.objectives}
        show_status = bool(constraints) or any(status != "feasible" for status in statuses.values())
        head = ["Validation", "id", *(["objective"] if show_objective else []), *(["goal"] if len(rows) > 1 else [])]
        head += ["estimate", "sd", *(["adjusted"] if show_adjusted else []), "measured", "n"]
        head += ["P(met)"] if constraints else []
        head += ["status"] if show_status else []
        table = [head]
        for c, objective in v.objectives.items():
            for i, (label, metric, constraint) in enumerate(rows):
                estimate = v.estimates.get(c, {}).get(metric)
                measured = v.measured.get(c, {}).get(metric)
                row = [self._label(c) + (" *" if c == v.champion_id else ""), c] if i == 0 else ["", ""]
                if show_objective:
                    row.append(_number(objective, objective_decimals) if i == 0 else "")
                row += [label] if len(rows) > 1 else []
                places = decimals[metric]
                row += (
                    ["-", "-"] if estimate is None else [_number(estimate.mean, places), _number(estimate.sd, places)]
                )
                if show_adjusted:
                    row.append(_number(v.adjusted_means.get(c, {}).get(metric), places))
                row += ["-", "-"] if measured is None else [_number(measured.mean, places), str(measured.n)]
                if constraints:
                    p = None if constraint is None else v.p_met.get(c, {}).get(constraint)
                    row.append("" if constraint is None else self._p_met_cell(p))
                if show_status:
                    row.append(statuses[c] if i == 0 else "")
                table.append(row)
        numbers = {"objective", "estimate", "sd", "adjusted", "measured", "n", "P(met)"}
        return _columns(table, right=[i for i, name in enumerate(head) if name in numbers])

    def _p_met_cell(self, p: float | None) -> str:
        """P(met) at 3 decimals, cut instead of rounded when rounding would carry it up to the threshold, so a
        constraint short of it never prints as meeting it (0.8997 -> 0.899 under 0.9)."""
        threshold = self.feasibility_threshold
        if p is not None and threshold is not None and p < threshold <= round(p, 3):
            return f"{math.floor(p * 1000) / 1000:.3f}"
        return _number(p, 3)

    def _objective_differs(self, goal: Mapping[str, Any], decimals: int) -> bool:
        """Whether a lone objective prints other than its metric's value (negated when lower is better) on any
        validation row: under `LowerBound`, or when `round_to_sd` moved it at the printed precision. The value is
        the one the pick used: the fit's estimate, else the adjusted mean, else the measured mean."""
        v, sign = self.validation, 1 if goal["better"] == "higher" else -1
        for c, objective in v.objectives.items():
            if objective is None:  # ineligible, which the status column shows
                continue
            estimate = v.estimates.get(c, {}).get(goal["metric"])
            adjusted = v.adjusted_means.get(c, {}).get(goal["metric"])
            measured = v.measured.get(c, {}).get(goal["metric"])
            if estimate is not None:
                value: float | None = estimate.mean
            else:
                value = adjusted if adjusted is not None else None if measured is None else measured.mean
            if value is None or round(objective, decimals) != round(sign * value, decimals):
                return True
        return False

    def _status(self, candidate_id: str) -> str:
        """A candidate's label on validation: `ineligible` (no objective there), `infeasible`, or `feasible`."""
        if self.validation.objectives.get(candidate_id) is None:
            return "ineligible"
        return "infeasible" if candidate_id in self.validation.infeasible else "feasible"

    def _infeasible_table(self) -> list[str]:
        """Each candidate that can't ship, in the pick's order, with why (`ValidationReport.infeasible`, its first
        line): a constraint not met, not eligible on validation, or no validation trial completed."""
        v = self.validation
        table = [["Not feasible", "why"]]
        table += [[self._label(c), v.infeasible[c].splitlines()[0]] for c in v.objectives if c in v.infeasible]
        return _columns(table, right=())

    def _front_table(self, goals: Sequence[Mapping[str, Any]]) -> list[str]:
        """Under `Pareto`, each validated front member against the baseline on test: its objective, then each
        objective metric; "-" for a member the test report has no comparison for."""
        metrics = [goal["metric"] for goal in goals if goal["role"] == "objective"]
        members = [(c, _test_rows(self.front_test.get(c), metrics, objective=True)) for c in self.validation.front]
        decimals = {
            metric: _decimals([x for _, rows in members for row in rows if row.metric == metric for x in row.numbers()])
            for metric in ["objective", *metrics]
        }
        table = [["Front test vs baseline", "id", "metric", "baseline", "member", "diff", f"{_LEVEL}% CI", "n"]]
        for c, rows in members:
            label = self._label(c) + (" *" if c == self.validation.champion_id else "")
            for i, row in enumerate(rows):
                table.append(
                    [label if i == 0 else "", c if i == 0 else "", row.metric, *row.cells(decimals[row.metric])]
                )
        return _columns(table, right=range(3, 8))

    def _changes_table(self) -> list[str]:
        """Every changed number and choice (`CandidateView.changed`; text is in `diff()`) with its baseline and
        champion value: its repr (a long float to 6 significant digits), cut to 40 characters, and
        "inactive" for a side the key isn't active on. The exact values are in `best.values`."""
        table = [["Changed", "baseline", "champion"]]
        for key, sides in sorted(self.best.changed.items()):  # sorted, as the header's `Changed:`
            table.append([key, *(_value_cell(sides[side]) if side in sides else "inactive" for side in ("old", "new"))])
        return _columns(table, right=())

    def _tokens_table(self) -> list[str]:
        """Tokens per role and phase (train is the search), from the cost fields: the agent per split, the
        proposers, who run only in the search, and the judges, the retried attempts, and the rollouts' other
        agents, whose fields hold run totals only ("-" for each phase)."""
        agent = [self.agent_cost[split].tokens for split in SPLIT_NAMES]
        proposers = [self.search_cost.tokens, 0, 0]
        table = [["Tokens", "search", "validation", "test", "total"]]
        table.append(["agent", *(f"{n:,}" for n in agent), f"{sum(agent):,}"])
        table.append(["proposers", *(f"{n:,}" for n in proposers), f"{sum(proposers):,}"])
        totals = [("judges", self.judge_cost), ("retries", self.retry_cost), ("other agents", self.benchmark_cost)]
        for role, usage in totals:
            table.append([role, "-", "-", "-", f"{usage.tokens:,}"])
        return _columns(table, right=range(1, 5))

    def _notes(self, tables: Sequence[Sequence[str]], *, width: int) -> list[str]:
        """What the columns hold, for the tables shown, wrapped to `width`."""
        heads = [table[0] for table in tables]
        notes = [
            "estimate, sd: the validation fit's estimate of the mean and its posterior sd; measured: the mean over "
            "the cases measured, n of them (the runs, when one case has a value); P(met) is - for a constraint judged "
            "by its measured or adjusted mean; * the champion."
        ]
        if any("adjusted" in head.split() for head in heads):
            notes.append(
                "adjusted: the mean the pick used for a metric the fit didn't judge, with the cases N/A for it "
                "filled in from the other candidates."
            )
        if self.test is not None or self.front_test:
            notes.append(
                "Test: baseline, champion, and diff (champion - baseline, with its "
                f"{_LEVEL}% interval) are over the n cases both measured; reported only."
            )
        notes.append(
            "Tokens: search is the train split; retries are the agent's and the judges' tokens in attempts run "
            "again; other agents are the rollouts' agents the factory didn't build (a simulated user)."
        )
        lines = ["Notes:"]
        for i, note in enumerate(notes, 1):
            lines += textwrap.wrap(note, width=width, initial_indent=f"[{i}] ", subsequent_indent="    ")
        return lines

    def to_rows(self, grain: Grain = "proposal") -> list[dict[str, Any]]:
        """The run as flat rows, built from the event log, so a resumed run's rows cover the whole run.

        Every row of one grain has the same keys; a value that doesn't apply is None. Per-metric columns
        are named `<prefix>.<metric>`, e.g. `train.HelpfulnessEvaluator`. `pandas.DataFrame(rows)` works.
        Iteration rows cover the closed iterations only, so after a kill and resume their `rollouts_total`
        and `search_tokens` can sum to less than `rollouts_spent` and `search_cost` (which count the killed
        iteration's spend too).

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


_LEVEL = round(TEST_LEVEL * 100)


def _rule_name(config: Mapping[str, Any]) -> str:
    """A pick rule's config as its name: "Mean", "LowerBound(q=0.1)"."""
    settings = ", ".join(f"{key}={value}" for key, value in config.items() if key != "kind")
    return f"{config['kind']}({settings})" if settings else str(config["kind"])


def _decimals(values: Sequence[float | None]) -> int:
    """One precision for numbers of one metric, by the largest: 0 decimals from 1,000, 1 from 100, 2 from 10,
    3 from 0.1, and below that enough for 3 significant digits of the largest (0.000105 -> 6)."""
    largest = max((abs(v) for v in values if v is not None and math.isfinite(v)), default=0.0)
    if 0 < largest < 0.1:
        return 2 - math.floor(math.log10(largest))
    return 0 if largest >= 1000 else 1 if largest >= 100 else 2 if largest >= 10 else 3


def _value_cell(value: Value) -> str:
    """A changed value for the Changed table: its repr (a float's to 6 significant digits when longer than 10
    characters), cut to 40."""
    text = repr(value)
    if isinstance(value, float) and len(text) > 10:
        text = f"{value:.6g}"
    return text if len(text) <= 40 else text[:39] + "…"


def _number(value: float | None, decimals: int, *, signed: bool = False) -> str:
    """`value` with thousands separators and `decimals` places; "-" for None."""
    return "-" if value is None else f"{value:{'+' if signed else ''},.{decimals}f}"


class _Compared(NamedTuple):
    """One row of a test table: a metric or "objective", both test means, and the paired difference."""

    metric: str
    base: float | None
    other: float | None
    difference: PairedComparison | None

    def numbers(self) -> list[float | None]:
        """Every number the row prints, to choose its precision."""
        d = self.difference
        return [self.base, self.other, *(() if d is None else (d.mean_diff, d.ci_low, d.ci_high))]

    def cells(self, decimals: int) -> list[str]:
        """Both means, the signed difference, its interval ("-" with one case), and the cases."""
        d = self.difference
        cells = [_number(self.base, decimals), _number(self.other, decimals)]
        if d is None:
            return [*cells, "-", "-", "-"]
        low, high = (_number(x, decimals, signed=True) for x in (d.ci_low, d.ci_high))
        interval = "-" if d.ci_low is None or d.ci_high is None else f"[{low}, {high}]"
        return [*cells, _number(d.mean_diff, decimals, signed=True), interval, str(d.n_cases)]


def _test_rows(report: TestReport | None, metrics: Sequence[str], *, objective: bool) -> list[_Compared]:
    """A test report's rows: the objective first when `objective`, then each metric, each with both sides' means
    over the cases both measured and the paired difference; all None without a report."""
    rows = []
    if objective:
        compared = None if report is None else report.objective
        rows.append(_shared("objective", compared, None, None))
    for metric in metrics:
        if report is None:
            rows.append(_Compared(metric, None, None, None))
            continue
        base, other = (report.means.get(c, {}).get(metric) for c in (report.baseline_id, report.champion_id))
        rows.append(_shared(metric, report.difference.get(metric), base, other))
    return rows


def _shared(metric: str, compared: PairedComparison | None, base: Measured | None, other: Measured | None) -> _Compared:
    """A test row with each side's mean over the cases both measured (`PairedComparison`), so the means and the
    difference cover the same cases; each side's own test mean (`base`, `other`) for a comparison saved
    before those were recorded."""
    if compared is not None and compared.reference_mean is not None:
        return _Compared(metric, compared.reference_mean, compared.candidate_mean, compared)
    return _Compared(metric, None if base is None else base.mean, None if other is None else other.mean, compared)


def _test_table(report: TestReport, *, lone: bool) -> list[str]:
    """The test report: the objective (unless it is a lone metric's), then each compared metric, with both
    test means and the paired difference, each row at its own precision."""
    table = [["Test vs baseline", "baseline", "champion", "diff", f"{_LEVEL}% CI", "n"]]
    for row in _test_rows(report, list(report.difference), objective=not lone):
        table.append([row.metric, *row.cells(_decimals(row.numbers()))])
    return _columns(table, right=range(1, 6))


def _columns(rows: Sequence[Sequence[str]], *, right: Iterable[int]) -> list[str]:
    """Rows of cells as columns two spaces apart, the `right` ones right-aligned, the rest left-aligned."""
    right = set(right)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        "  ".join(cell.rjust(widths[i]) if i in right else cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in rows
    ]


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
        label = (
            labels.get((iteration, candidate_id)) if outcome in LABEL_OUTCOMES and candidate_id is not None else None
        )
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
            out[f"adjusted.{m}"] = label.adjusted_means.get(m) if label is not None else None
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
    baseline_rollouts = 0
    for event in events:
        name, iteration = event["event"], event["iteration"]
        if name == "fit":
            fits[iteration] = event
        elif name == "iteration_end":
            ends[iteration] = event
        elif name == "score" and iteration == 0 and event["split"] == "train":
            baseline_rollouts += event["rollouts"]
        elif name == "proposer_usage":  # each closed iteration's proposer spend, in the step that asked
            tokens[iteration] += event["usage"]["tokens"]
            calls[iteration] += event["usage"]["model_calls"]
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
        row.update({outcome: outcomes[outcome] for outcome in (*LABEL_OUTCOMES, "invalid", "duplicate")})
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
        for split_name in SPLIT_NAMES:
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
                        **{metric: getattr(usage, metric) for metric in USAGE_METRICS},
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
    ("An error occurred: <error>") and the case itself as the row, as does each evaluator without a score in
    any other trial with `error` set (`Experiment` failed on its case after the task returned, or in a
    re-judge, which keeps the first pass's scores); an evaluator in `evaluator_errors` gets a score-0 row
    with "Evaluator error: <exception type>" (the record keeps only the type). Each row is tagged
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
        for split_name in SPLIT_NAMES:
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
                    raised = record.status in (TrialStatus.AGENT_FAILURE, TrialStatus.INFRASTRUCTURE_INVALID)
                    if raised or (record.error is not None and name not in record.evaluations):
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
