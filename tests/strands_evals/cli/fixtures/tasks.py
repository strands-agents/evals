"""Test-only ``--task`` entry-point shapes for the CLI."""

from __future__ import annotations

from typing import Any

from strands_evals.case import Case


def answer(case: Case) -> dict[str, Any]:
    """Deterministic ``--task`` callable.

    Echoes the case input, optionally pulling an override from metadata.
    """
    metadata = case.metadata or {}
    return {"output": metadata.get("answer", f"answered: {case.input}")}


def answer_string(case: Case) -> str:
    """Returns a plain string (Experiment normalizes via the dict path)."""
    return f"answered: {case.input}"
