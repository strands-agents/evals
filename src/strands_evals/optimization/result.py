"""What a run returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..types.evaluation_report import EvaluationReport
from ..types.optimization import IterationRecord, Usage, Value
from .candidates.candidate import Candidate
from .declaration.space import Space

Grain = Literal["iteration", "proposal", "case"]


@dataclass(frozen=True)
class CandidateView:
    """A candidate as the results show it; built by the optimizer."""

    _candidate: Candidate
    _baseline: Candidate
    _space: Space

    @property
    def id(self) -> str:
        raise NotImplementedError

    @property
    def values(self) -> dict[str, Value]:
        raise NotImplementedError

    def diff(self) -> str:
        """Unified diff of text and file-backed values against the baseline, on the real paths."""
        raise NotImplementedError

    def changed_values(self) -> dict[str, tuple[Value, Value]]:
        """Changed numbers and choices against the baseline."""
        raise NotImplementedError


@dataclass
class OptimizationResult:
    """What a run returns.

    Attributes:
        frontier: Candidate id -> the selection cases it leads.
        stop_reason: `budget` (what's left can't cover another candidate), `iterations` (the cap),
            `exhausted` (every block produced no candidates on its latest turn), or `proposer_error`
            (a method raised).
        search_cost: Total proposer usage.
        judge_cost: Total judge usage across trials.
    """

    run_id: str
    run_dir: Path
    baseline: CandidateView
    best: CandidateView
    frontier: dict[str, list[str]]
    iterations: list[IterationRecord]
    rollouts_spent: int
    stop_reason: Literal["budget", "iterations", "exhausted", "proposer_error"]
    search_cost: Usage
    judge_cost: Usage

    def summary(self) -> str:
        raise NotImplementedError

    def to_rows(self, grain: Grain = "proposal") -> list[dict[str, Any]]:
        raise NotImplementedError

    def to_json(self, grain: Grain = "proposal", path: str | Path | None = None) -> str:
        raise NotImplementedError

    def reports(self, candidate_id: str) -> list[EvaluationReport]:
        raise NotImplementedError
