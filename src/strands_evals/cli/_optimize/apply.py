"""Write a run's candidate into the project files; texts without a file and numbers are reported, not written."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...types.optimization import Value
from .record import RunDirectory


def apply_candidate(run_dir: Path, candidate_id: str | None, *, dry_run: bool) -> dict[str, Any]:
    """Apply the run's pick, or `candidate_id`, to the files that its `FileText` keys came from.

    Each file must still hold the run's baseline text; a file that already holds the new text is left as is.

    Raises:
        ValueError: The run has no pick and no `candidate_id` is given, the candidate is unknown, a file
            changed since the run started, or a path is outside the project.
    """
    run = RunDirectory.open(run_dir)
    config = run.config()
    events = run.events.read()
    values = {e["candidate_id"]: e["values"] for e in events if e["event"] == "propose" and "values" in e}
    baseline = next(e["values"] for e in events if e["event"] == "propose" and e["origin"] == "baseline")
    picks = [e["report"]["champion_id"] for e in events if e["event"] == "selected"]
    target = candidate_id or (picks[-1] if picks else None)
    if target is None:
        raise ValueError(f"the run has no final pick yet: resume it, or pass --candidate ID: {run.root}")
    if target not in values:
        raise ValueError(f"unknown candidate {target!r} in {run.root}")
    writes, manual = _changes(config["space"], baseline, values[target], Path(config["workdir"]))
    if not dry_run:
        for path, text in writes:
            path.write_text(text, encoding="utf-8")
    return {
        "run_id": run.run_id,
        "candidate_id": target,
        "dry_run": dry_run,
        "files": [str(path) for path, _ in writes],
        "manual": manual,
    }


def _changes(
    space: Sequence[Mapping[str, Any]], baseline: Mapping[str, Value], target: Mapping[str, Value], root: Path
) -> tuple[list[tuple[Path, str]], dict[str, dict[str, Value]]]:
    writes: list[tuple[Path, str]] = []
    manual: dict[str, dict[str, Value]] = {}
    for param in space:
        key = param["key"]
        old, new = baseline.get(key), target.get(key)
        if old == new:
            continue
        if param["source"] is None:
            manual[key] = {"old": old, "new": new}
            continue
        path = (root / param["source"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"refusing to write outside the project: {path}")
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == new:
            continue
        if current != old:
            raise ValueError(f"{path} changed since the run started: review best.diff and edit the file by hand")
        writes.append((path, str(new)))
    return writes, manual
