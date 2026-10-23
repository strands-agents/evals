"""A candidate: one typed point of the space, its content id, and its lineage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ...types.optimization import Value
from ..declaration.space import Space


def content_id(values: Mapping[str, Value]) -> str:
    """First 12 hex characters of the sha256 of the canonical JSON (sorted keys) of `values`."""
    raise NotImplementedError


@dataclass(frozen=True)
class Candidate:
    """Typed flat values plus lineage; `origin` is "baseline", a method name, or "merge"."""

    values: dict[str, Value]
    id: str
    parent_ids: tuple[str, ...]
    origin: str
    rationale: str

    @classmethod
    def baseline(cls, space: Space) -> Candidate:
        """Every value at its default (`space.complete({})`)."""
        raise NotImplementedError

    def child(self, space: Space, updates: Mapping[str, Value], *, origin: str, rationale: str) -> Candidate:
        """Copy these values, apply `updates`, complete, and validate."""
        raise NotImplementedError

    @classmethod
    def merged(
        cls,
        space: Space,
        values: Mapping[str, Value],
        parents: tuple[str, str],
        rationale: str,
    ) -> Candidate:
        raise NotImplementedError
