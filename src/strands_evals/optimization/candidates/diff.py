"""Differences between two candidates."""

from __future__ import annotations

from ...types.optimization import Value
from ..declaration.space import Space
from .candidate import Candidate


def diff(space: Space, base: Candidate, other: Candidate) -> str:
    """A valid unified diff of text and file-backed values, on the real file paths."""
    raise NotImplementedError


def changed_values(space: Space, base: Candidate, other: Candidate) -> dict[str, tuple[Value, Value]]:
    """Changed numbers and choices as `key -> (before, after)`."""
    raise NotImplementedError


def changed_keys(base: Candidate, other: Candidate) -> list[str]:
    """Every key whose value differs, including keys present in only one of them."""
    raise NotImplementedError
