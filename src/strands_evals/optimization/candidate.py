"""A candidate: one typed point of the space, its content id, its parent, and how it differs from another."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ..types.optimization import Value, Values
from .declaration.space import Space


def content_id(values: Values) -> str:
    """First 12 hex characters of the sha256 of the canonical JSON (sorted keys) of `values`."""
    raise NotImplementedError


@dataclass(frozen=True, eq=False)
class Candidate:
    """Typed flat values plus their parent.

    `values` is stored as a read-only copy (`MappingProxyType`) and `id` is computed from it in
    `__post_init__`, so the content id can't go stale. Equality and hashing use `id`.
    `origin` is "baseline" or the proposing block's name (`Block.name`). One parent per candidate
    (None for the baseline); there are no merges.

    Attributes:
        inherited: Key -> the candidate it was taken from: a key the parent lacked, made active by a
            changed `choice()`. Saved in the `propose` event.
        defaulted: Keys no measured candidate had yet, left at their defaults.
    """

    values: Values
    parent_id: str | None
    origin: str
    rationale: str
    inherited: Mapping[str, str] = field(default_factory=dict)
    defaulted: tuple[str, ...] = ()
    id: str = field(init=False)

    def __post_init__(self) -> None:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    @classmethod
    def baseline(cls, space: Space) -> Candidate:
        """Every value at its default (`space.complete({})`)."""
        raise NotImplementedError

    def child(
        self,
        space: Space,
        updates: Values,
        *,
        origin: str,
        rationale: str,
        fill: Callable[[str], tuple[Value, str] | None] | None = None,
    ) -> Candidate:
        """Copy these values, apply `updates`, fill keys the parent lacks, and validate.

        Proposed keys come from `updates`; every other key as the parent has it; a key the
        parent lacks (it exists only under a choice the parent didn't make) from `fill(key)`, which returns
        `(value, source id)` from the best-ranked eligible candidate that has it (recorded in `inherited`),
        else its default (recorded in `defaulted`). A tuple, because None is a valid value.

        Args:
            space: Completes and validates the new values.
            updates: The keys to change, from `Patch.updates`.
            origin: The proposing block's name.
            rationale: Why, from the patch.
            fill: Where a key the parent lacks comes from; None means its default.

        Raises:
            ValueError: The completed values fail `space.validate`.
        """
        raise NotImplementedError


def diff(space: Space, base: Candidate, other: Candidate) -> str:
    """A valid unified diff of text and file-backed values, on the real file paths.

    Args:
        space: Maps file-backed keys to their real paths.
        base: The candidate diffed against, usually the baseline.
        other: The candidate shown.
    """
    raise NotImplementedError


def changed_values(space: Space, base: Candidate, other: Candidate) -> dict[str, tuple[Value, Value]]:
    """Changed numbers and choices as `key -> (before, after)`.

    Args:
        space: Tells numbers and choices apart from text.
        base: The candidate compared against, usually the baseline.
        other: The candidate shown.

    Returns:
        Key -> (base value, other value) for every changed number or choice.
    """
    raise NotImplementedError


def changed_keys(base: Candidate, other: Candidate) -> list[str]:
    """Every key whose value differs, including keys present in only one of them.

    Args:
        base: The candidate compared against, e.g. the parent.
        other: The candidate shown.
    """
    raise NotImplementedError
