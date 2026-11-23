"""The append-only `events.jsonl` event log."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

EventName = Literal[
    "config",
    "draw",
    "propose",
    "trial",
    "score",
    "fit",
    "select",
    "label",
    "invalid",
    "duplicate",
    "proposer_retry",
    "proposer_usage",
    "proposer_error",
    "resume",
    "stop",
    "observe",
    "evidence",
    "screened",
    "iteration_end",
    "finalists",
    "picked",
    "test",
]


class EventLog:
    """Everything that happened, one JSON event per line; the first line is the run's config.

    Every event has `event`, `iteration` (0 for setup and the baseline), and `time` (ISO 8601, for reading
    only; resume never uses it). Records nest whole under the key shown, via `model_dump`.

    - `config`: the `run.json` contents, including `schema_version`.
    - `draw`: `parent_id`, `blocks` (the step's), `n_candidates`, `draws` (per redraw, 0 first, each
      proposal's patches as block -> `Patch`, in the order the proposers returned them); written after a
      step's last redraw and before any candidate is saved or scored (none when a proposer raised).
    - `propose`: `candidate_id`, `number` (display only), `parent_id`, `origin`, `values` (the only copy of
      the candidate's typed values), `changed_keys`, `rationale`, `inherited`, `defaulted`, `round`, `role`;
      `rescore=true` for an ineligible candidate proposed again that a re-score can change: a train trial
      left to run or a failed evaluator to judge again (one with neither is a duplicate). Then no `values`,
      and the first number and parent are kept; `deferred_patches` (block -> `Patch`) for the blocks whose
      observe is deferred (a round's opener), else None.
    - `trial`: `record` (`TrialRecord`, its trajectory replaced by `transcript`: `{hash, format}` or None),
      one per trial as soon as it finishes. A later one for the same (candidate, run, case) replaces it.
      Trials of concurrently scored candidates interleave; every other event is written in a fixed order.
    - `score`: `candidate_id`, `split`, `case_names`, `runs`, `rollouts`, `objective_by_run`, `statuses`
      (status -> trial count), `unavailable`.
    - `fit`: `candidates` (ids fitted), `is_fitted`, `error` (when the score model raised), `flagged`,
      `diagnostics`; one per step, `is_fitted=false` also when the step scored nothing new.
    - `select`: `parent_id`, `strategy`.
    - `label`: `candidate_id`, `label` (`CandidateLabel`).
    - `invalid`: `block`, `redraw`, `updates`, `reason` (no candidate).
    - `duplicate`: `block`, `redraw`, `candidate_id` (the existing one); `patch` and `round` for a round's
      opener.
    - `proposer_retry`, `proposer_error`: `block`, `redraw`, `error_type`, `error`.
    - `proposer_usage`: `block`, `redraw`, `usage` (`Usage`): a `propose` call's spend, written by the proposer
      for spend no returned patch carries, and by the optimizer for the returned patches' `usage` (also in
      their `observe` events, which `search_cost` doesn't read).
    - `screened`: `block`, `key`, `reason`.
    - `resume`: `spent`, `budget`, `code` (`code_version()` of the resuming code: strands-evals version, git commit,
      dirty, numpy and optuna versions).
    - `stop`: `reason` (a `stop_reason`, or `"setup"` when the baseline can't be a parent: ineligible, no
      completed train trial, or a goal metric never measured), `message`, and for a search stop `max_rollouts`
      (the budget it stopped under).
    - `observe`: `block`, `candidate_id` (None for an invalid patch), `observation` (`Observation`).
    - `evidence`: `block` (the proposer told), `evidence` (`Evidence`).
    - `iteration_end`: `record` (`IterationRecord`), `exhausted`, `best_after`, `best_objective` (the first
      `best_after` candidate's objective, or None; snapshots).
    - `finalists`: `candidate_ids` (ranked). `picked`: `report` (`ValidationReport`). `test`: `report`
      (`TestReport` or None), `reason` (also naming the front members it couldn't afford), `front` (under a
      `Pareto` objective, each tested front member's `TestReport`, the champion's included; `{}` otherwise),
      `failed` (the tested candidates with no completed test trial).

    Resume reads: `spent` from the trial events (the last per key: its `rollouts` plus its
    `previous_attempts`'); candidates, numbers, parents, and rounds from `propose` and `iteration_end`;
    exhausted blocks and iterations from `iteration_end`; history from `select`; open rounds from openers
    with no observe yet; `stop_reason` from the last `stop`; proposer feedback from `observe` / `evidence`;
    `search_cost` from every `proposer_usage` event, a killed iteration's included; the fit sequence from
    `fit`; the end phase's progress from `finalists` / `picked` / `test`. Only search events of closed iterations
    (ended by `iteration_end` or by `stop`) are replayed; trial and `proposer_usage` events are always read, and
    so is the killed iteration's last `draw`, whose proposals it runs again on instead of asking the proposers.
    """

    def __init__(self, path: Path):
        """Open the log at `path` without writing to it; the first append cuts off a truncated last line, so
        it starts a new line.

        Raises:
            ValueError: A line other than the last isn't a JSON object (the log is corrupt, not truncated).
        """
        self.path = path
        self._lock = threading.Lock()
        self._iteration = 0
        # The events as parsed at open, reused by read() until the next append, so opening a run parses it once.
        self._events: list[dict[str, Any]] | None = None
        self._size = 0  # the log's size in bytes when this object last parsed or appended to it
        self._cut: int | None = None  # where a truncated last line starts, until the first append cuts it off
        if not path.exists():
            return
        events, end, self._size = self._parse()
        if end < self._size:
            logger.warning("path=<%s> | ignoring a truncated last line | the next append cuts it off", path)
            self._cut = end
        self._events = events
        if events:
            self._iteration = events[-1].get("iteration", 0)

    def append(self, event: EventName, **fields: Any) -> None:
        """Append one event as a JSON line, with one write and flush; the optimizer is the only writer.

        `iteration` defaults to the last event's (0 at the start), so a trial saved during a step carries
        the step's iteration. Pydantic records are written with `model_dump(mode="json")`.

        Args:
            event: The event name.
            **fields: JSON-serializable event fields.

        Raises:
            TypeError: A field isn't JSON-serializable.
            ValueError: A field holds NaN or infinity (not JSON), or is named `time` (the log's own field).
            RuntimeError: The log has a truncated last line to cut off, but changed since it was read (another
                writer may still be writing that line); nothing is written.
        """
        if "time" in fields:
            raise ValueError("an event field can't be named 'time': the log writes it")
        with self._lock:
            iteration = fields.pop("iteration", self._iteration)
            line = json.dumps(
                {
                    "event": event,
                    "iteration": iteration,
                    **fields,
                    "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                },
                default=_encode,
                ensure_ascii=False,
                allow_nan=False,
            )
            if self._cut is not None:
                with self.path.open("r+b") as file:
                    if os.fstat(file.fileno()).st_size != self._size:
                        raise RuntimeError(
                            f"{self.path} changed since it was read, so its truncated last line may be another "
                            "writer's line in progress; open the log again"
                        )
                    file.truncate(self._cut)
                self._cut = None
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
                file.flush()
                self._size = os.fstat(file.fileno()).st_size
            self._iteration = iteration
            self._events = None

    def read(self) -> list[dict[str, Any]]:
        """Every event, dropping a truncated last line; parsed again only after an append. Don't modify the list."""
        with self._lock:
            if self._events is None:
                self._events = self._parse()[0]
            return self._events

    def changed(self) -> bool:
        """Whether the log's size differs from when this object last parsed or appended to it: another writer
        wrote to it since."""
        size = self.path.stat().st_size if self.path.exists() else 0
        return size != self._size

    def _parse(self) -> tuple[list[dict[str, Any]], int, int]:
        """The events, the byte offset where the last complete one ends, and the bytes read."""
        if not self.path.exists():
            return [], 0, 0
        data = self.path.read_bytes()
        events: list[dict[str, Any]] = []
        end = 0
        lines = data.split(b"\n")
        # The piece after the last newline is empty for a complete log, else a line cut off mid-write.
        for number, raw in enumerate(lines[:-1], start=1):
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                event = None
            if not isinstance(event, dict):
                if number == len(lines) - 1 and not lines[-1]:
                    # The last line, complete but unreadable: a kill that left garbage behind.
                    break
                raise ValueError(f"{self.path}: line {number} is not a JSON event; the log is corrupt")
            events.append(event)
            end += len(raw) + 1
        return events, end, len(data)


_OUTSIDE_ITERATIONS = frozenset({"config", "trial", "resume", "stop", "finalists", "picked", "test"})


def closed_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The events resume reads, in log order: a search event only once its iteration closed.

    An iteration closes with its `iteration_end`, or with the `stop` event (every stop runs the end phase,
    so from there everything counts). A killed iteration's events are dropped at the next `resume` event
    or at the end of the log, since resume runs that iteration again. `config`, `trial`, `resume`, `stop`,
    and the end phase's events are always kept.
    """
    keep = [False] * len(events)
    pending: list[int] = []
    stopped = False
    for index, event in enumerate(events):
        name = event["event"]
        if stopped or name in _OUTSIDE_ITERATIONS:
            keep[index] = True
            if name == "stop":
                for kept in pending:
                    keep[kept] = True
                stopped = True
            if name in ("stop", "resume"):
                pending.clear()
            continue
        pending.append(index)
        if name == "iteration_end":
            for kept in pending:
                keep[kept] = True
            pending.clear()
    return [event for event, kept in zip(events, keep, strict=True) if kept]


def _encode(value: Any) -> Any:
    """`json.dumps` fallback: pydantic records whole, read-only mappings as dicts, sets and tuples as lists."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"an event field of type {type(value).__name__} is not JSON-serializable")
