"""Print where an optimization run is, from its events.jsonl: phase, baseline and best, spend, latest candidates."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PHASES = {
    "stop": "end phase: picking finalists",
    "finalists": "end phase: scoring finalists on validation",
    "picked": "end phase: comparing the champion with the baseline on test",
    "test": "end phase: writing the result",
}
SHOWN_CANDIDATES = 5
RATIONALE_WIDTH = 100


def _read_events(path: Path) -> list[dict[str, Any]]:
    """Every complete event; a line cut off mid-write by a running or killed run is skipped."""
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _measured(score: dict[str, Any]) -> float | None:
    """The mean objective over the score event's measured runs, or None when nothing was measured."""
    values = [value for runs in score["objective_by_run"].values() for value in runs if value is not None]
    return mean(values) if values else None


def _rollouts(events: list[dict[str, Any]]) -> int:
    """Rollouts spent, as resume counts them: each trial key's last record, plus its earlier attempts."""
    last: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for event in events:
        if event["event"] == "trial":
            record = event["record"]
            last[(record["candidate_id"], record["split"], record["case_name"], record["run_index"])] = record
    return sum(r["rollouts"] + sum(p.get("rollouts", 0) for p in r["previous_attempts"]) for r in last.values())


def _elapsed(events: list[dict[str, Any]], finished: bool) -> str:
    start = datetime.fromisoformat(events[0]["time"])
    end = datetime.fromisoformat(events[-1]["time"]) if finished else datetime.now(timezone.utc)
    minutes, seconds = divmod(int((end - start).total_seconds()), 60)
    return f"{minutes}m {seconds:02d}s"


def status(run_dir: Path) -> str:
    """The run's status as a few lines of plain text."""
    events = _read_events(run_dir / "events.jsonl")
    if not events:
        return f"run {run_dir.name}: starting, no events yet"
    config = events[0]["config"]
    proposed = {e["candidate_id"]: e for e in events if e["event"] == "propose" and not e.get("rescore")}
    train = {e["candidate_id"]: _measured(e) for e in events if e["event"] == "score" and e["split"] == "train"}
    baseline = next((cid for cid, e in proposed.items() if e["origin"] == ["baseline"]), None)
    ends = [e for e in events if e["event"] == "iteration_end"]
    finished = (run_dir / "result.json").is_file()
    stop = next((e for e in reversed(events) if e["event"] == "stop"), None)
    phase = f"search: iteration {len(ends) + 1}" if baseline in train else "measuring the baseline"
    phase = next((PHASES[e["event"]] for e in reversed(events) if e["event"] in PHASES), phase)
    stopped_at_setup = stop is not None and stop["reason"] == "setup"
    if stopped_at_setup:
        phase = "stopped at setup"
    if finished:
        phase = "finished"
    budget = config["budget"]
    iterations = budget["max_iterations"]
    planned = f" of {iterations * config['proposals_per_iteration']} planned" if iterations else ""
    candidates = [cid for cid in train if cid != baseline]
    lines = [
        f"run {run_dir.name} · {phase} · {_elapsed(events, finished or stopped_at_setup)}"
        + (f" · stopped: {stop['reason']}" if stop and not stopped_at_setup else ""),
        f"objective {config['objective']} · rollouts {_rollouts(events)} of {budget['max_rollouts']} · "
        f"candidates: {len(proposed) - (baseline is not None)} proposed, {len(candidates)} scored on train{planned}",
    ]
    base = train.get(baseline) if baseline is not None else None
    if base is not None:
        measured = {cid: value for cid in candidates if (value := train[cid]) is not None}
        top = max(measured, key=measured.__getitem__, default=None)
        best = (
            f" · best measured {measured[top]:.4g} ({measured[top] - base:+.4g}, candidate {proposed[top]['number']})"
            if top is not None
            else ""
        )
        lines.append(f"baseline {base:.4g} on train{best}")
    if stop and stopped_at_setup:
        lines.append(f"setup failed: {stop['message']}")
    for cid in reversed(candidates[-SHOWN_CANDIDATES:]):
        value = train[cid]
        shown = "unmeasured" if value is None else f"{value:.4g}"
        why = " ".join(proposed[cid]["rationale"].split())[:RATIONALE_WIDTH]
        lines.append(f"  {proposed[cid]['number']:>3}  {' + '.join(proposed[cid]['origin']):<16} {shown:>10}  {why}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="the run folder `strands-evals optimize run` printed")
    run_dir = parser.parse_args().run_dir
    if not (run_dir / "events.jsonl").is_file():
        sys.stderr.write(f"{run_dir} has no events.jsonl: not a run folder, or the run hasn't started\n")
        return 2
    sys.stdout.write(status(run_dir) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
