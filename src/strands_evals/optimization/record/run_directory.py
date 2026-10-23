"""The run directory: config, lineage, candidates, trials, and reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...types.evaluation_report import EvaluationReport
from ...types.optimization import TrialRecord
from ..candidates.candidate import Candidate
from .lineage import Lineage


class RunDirectory:
    """`<base>/<run_id>/`; implements `execution.trials.TrialCache`. Original user files are never written."""

    run_id: str
    root: Path
    lineage: Lineage

    @classmethod
    def create(cls, base_dir: Path, config: Mapping[str, Any]) -> RunDirectory:
        """Write `config` as the first lineage event and as `run.json`."""
        raise NotImplementedError

    @classmethod
    def open(cls, path: Path) -> RunDirectory:
        raise NotImplementedError

    def workspace(self, candidate_id: str) -> Path:
        """`<root>/candidates/<id>/optimization_files/`."""
        raise NotImplementedError

    def save_candidate(self, candidate: Candidate) -> None:
        """Write `values.json` and `candidate.json` (parent_ids, origin, rationale) once."""
        raise NotImplementedError

    def load_candidate(self, candidate_id: str) -> Candidate:
        raise NotImplementedError

    def load(self, candidate_id: str, run_index: int, case: str) -> TrialRecord | None:
        raise NotImplementedError

    def save(self, record: TrialRecord) -> None:
        """`candidates/<id>/trials/<case>/run<i>.json`."""
        raise NotImplementedError

    def save_report(
        self, candidate_id: str, split: str, run_index: int, pass_index: int, report: EvaluationReport
    ) -> None:
        """`candidates/<id>/reports/<split>.run<i>.pass<k>.json`."""
        raise NotImplementedError

    def reports(self, candidate_id: str) -> list[EvaluationReport]:
        raise NotImplementedError
