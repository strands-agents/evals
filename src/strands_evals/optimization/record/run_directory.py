"""The run directory: `run.json`, the event log, transcripts, effects, and per-candidate file copies."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...types.evaluation_report import EvaluationReport
from ...types.optimization import TrialRecord
from ..candidate import Candidate
from .events import EventLog


class RunDirectory:
    """`<base>/<run_id>/`; implements `execution.trials.TrialCache`. Original user files are never written.

    `run_id` is `[<name>-]<YYYYMMDD-HHMMSS>-<6 hex>`, the start time in UTC. Layout: `run.json`,
    `events.jsonl` (the only source of truth on resume), `transcripts/<hash>.json` (each trajectory once,
    named by the first 16 hex of the sha256 of its canonical JSON), `effects.json`, and
    `candidates/<id>/optimization_files/` (only when a value reaches the agent as a path). No
    per-candidate values, trial, or report files.
    """

    run_id: str
    root: Path
    events: EventLog

    @classmethod
    def create(cls, base_dir: Path, config: Mapping[str, Any], *, name: str | None = None) -> RunDirectory:
        """Make `<base_dir>/<run_id>/` and write `config` as the first event and as `run.json`.

        Args:
            base_dir: The parent of all run directories.
            config: The run's config, as compared on resume.
            name: The run folder's prefix, `Optimizer(name=)`; a collision draws new hex.
        """
        raise NotImplementedError

    @classmethod
    def open(cls, path: Path) -> RunDirectory:
        """Read `events.jsonl` once and index candidates and the last trial event per (candidate, run, case)."""
        raise NotImplementedError

    def workspace(self, candidate_id: str) -> Path:
        """`<root>/candidates/<id>/optimization_files/`."""
        raise NotImplementedError

    def save_candidate(self, candidate: Candidate, **fields: Any) -> int:
        """Append the `propose` event and return the candidate's display number.

        0 for the baseline, then 1, 2, ... in proposal order; for people only, never a key. An id already
        indexed (an ineligible candidate proposed again) gets `rescore=true`, no values, the same number.

        Args:
            candidate: The candidate.
            **fields: The optimizer's fields: `changed_keys`, `round`, `role`, `patch`.
        """
        raise NotImplementedError

    def number(self, candidate_id: str) -> int:
        raise NotImplementedError

    def load_candidate(self, candidate_id: str) -> Candidate:
        """From the index (its first `propose` event)."""
        raise NotImplementedError

    def load_trial(self, candidate_id: str, run_index: int, case_name: str) -> TrialRecord | None:
        """The last trial event for the key, its transcript read back as trajectory."""
        raise NotImplementedError

    def save_trial(self, record: TrialRecord) -> None:
        """Write the transcript (if new), then append the trial event and update the index."""
        raise NotImplementedError

    def reports(self, candidate_id: str) -> list[EvaluationReport]:
        """Rebuilt from the candidate's trial events, one per (split, run), as `Experiment` would return them."""
        raise NotImplementedError

    def save_effects(self, effects: Mapping[str, Any]) -> None:
        """`<root>/effects.json`."""
        raise NotImplementedError
