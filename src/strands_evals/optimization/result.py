"""What a run returns."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..types.optimization import IterationRecord, SplitName, StopReason, TestReport, Usage, ValidationReport, Value
from .candidate import Candidate, changed_values, diff
from .declaration.files import _write_text_atomically
from .declaration.space import Space

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
        raise NotImplementedError

    def to_rows(self, grain: Grain = "proposal") -> list[dict[str, Any]]:
        """The run as flat rows, built from the event log, so a resumed run's rows cover the whole run.

        Every row of one grain has the same keys; a value that doesn't apply is None. Per-metric columns
        are named `<prefix>.<metric>`, e.g. `train.HelpfulnessEvaluator`.

        Args:
            grain: What one row is: `proposal` (every proposal, the baseline included, with its label at
                its step and its rank under the last search fit), `iteration` (one search step), or
                `case` (one trial: candidate x case x run, on every split it was scored on).
        """
        raise NotImplementedError

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
