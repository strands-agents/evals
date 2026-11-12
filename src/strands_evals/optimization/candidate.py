"""A candidate: one typed point of the space, its content id, its parent, and how it differs from another."""

from __future__ import annotations

import difflib
import hashlib
import io
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..types.optimization import Value, Values
from .declaration.space import Space

# A key a candidate doesn't have; None is a valid value, so absence needs its own marker.
_ABSENT: Any = object()


def content_id(values: Values) -> str:
    """First 12 hex characters of the sha256 of the canonical JSON (sorted keys) of `values`.

    Types count: `1` and `1.0` give different ids, as `Space.validate` keeps them apart.

    Raises:
        ValueError: A value is NaN or infinite (not JSON).
    """
    canonical = json.dumps(dict(values), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, eq=False)
class Candidate:
    """Typed flat values plus their parent.

    `values` is stored as a read-only copy (`MappingProxyType`) and `id` is computed from it in
    `__post_init__`, so the content id can't go stale. Equality and hashing use `id`.
    `origin` is `("baseline",)` or the names of the blocks whose patches it merges, in proposal order
    (`Block.name`; one under `Alternate`, every proposing block of a `Joint` step). One parent per candidate
    (None for the baseline); there are no merges.

    Attributes:
        inherited: Key -> the candidate it was taken from: a key the parent lacked, made active by a
            changed `choice()`. Saved in the `propose` event.
        defaulted: Keys no measured candidate had yet, left at their defaults.
    """

    values: Values
    parent_id: str | None
    origin: tuple[str, ...]
    rationale: str
    inherited: Mapping[str, str] = field(default_factory=dict)
    defaulted: tuple[str, ...] = ()
    id: str = field(init=False)

    def __post_init__(self) -> None:
        # Frozen: set through object.__setattr__, once, so the id always matches the values.
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "inherited", MappingProxyType(dict(self.inherited)))
        object.__setattr__(self, "defaulted", tuple(self.defaulted))
        if isinstance(self.origin, str):
            raise TypeError(f"origin is a sequence of block names, not the string {self.origin!r}")
        object.__setattr__(self, "origin", tuple(self.origin))
        object.__setattr__(self, "id", content_id(self.values))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Candidate):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def baseline(cls, space: Space) -> Candidate:
        """Every value at its default (`space.complete({})`)."""
        return cls(values=space.complete({}), parent_id=None, origin=("baseline",), rationale="")

    def child(
        self,
        space: Space,
        updates: Values,
        *,
        origin: Sequence[str],
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
            origin: The names of the blocks whose patches `updates` merges, in proposal order.
            rationale: Why, from the patch.
            fill: Where a key the parent lacks comes from; None means its default.

        Raises:
            ValueError: The completed values fail `space.validate`, including an update to a key
                inactive under the resulting choices (it would be dropped, so the patch would
                change nothing there).
        """
        inherited: dict[str, str] = {}
        defaulted: list[str] = []

        def record(key: str) -> tuple[Value, str] | None:
            found = fill(key) if fill is not None else None
            if found is None:
                defaulted.append(key)
            else:
                inherited[key] = found[1]
            return found

        completed = space.complete({**self.values, **updates}, fill=record)
        # Validate what is stored; an update complete() dropped as inactive is added back, so it's reported.
        space.validate({**updates, **completed})
        return Candidate(
            values=completed,
            parent_id=self.id,
            origin=tuple(origin),
            rationale=rationale,
            inherited=inherited,
            defaulted=tuple(defaulted),
        )


def diff(space: Space, base: Candidate, other: Candidate, *, root: Path) -> str:
    """A valid unified diff of every changed text value, for review.

    File-backed values are hunks on their real files, headed `--- a/<path>` / `+++ b/<path>` with the
    path relative to `root` (absolute for a file outside it); a `JsonField`'s hunk is on its whole
    file. Each changed non-file text value follows as a hunk headed `--- <key>` / `+++ <key>`. Not
    promised to apply with `git apply`: the CLI applies a candidate from its values. Numbers and
    choices never appear (see `changed_values`).

    Args:
        space: Maps file-backed keys to their real paths.
        base: The candidate diffed against, usually the baseline.
        other: The candidate shown.
        root: The run's working directory (`run.json` `workdir`); paths are shown relative to it.

    Returns:
        The diff, or "" when no text changed.
    """
    before_files, after_files = space._render_files(base.values), space._render_files(other.values)
    root = root.resolve()
    hunks: list[str] = []
    for path in dict.fromkeys([*before_files, *after_files]):
        before, after = before_files.get(path, ""), after_files.get(path, "")
        if path.is_relative_to(root):
            relative = path.relative_to(root).as_posix()
            hunks.extend(_unified(before, after, f"a/{relative}", f"b/{relative}"))
        else:
            hunks.extend(_unified(before, after, path.as_posix(), path.as_posix()))
    for info in space.params():
        if info.kind != "text" or info.source is not None:
            continue
        before_text, after_text = str(base.values.get(info.key, "")), str(other.values.get(info.key, ""))
        hunks.extend(_unified(before_text, after_text, info.key, info.key))
    return "".join(hunks)


def changed_values(space: Space, base: Candidate, other: Candidate) -> dict[str, dict[str, Value]]:
    """Changed numbers and choices, in `best.values.json`'s format.

    Each changed key maps to `{"old": base value, "new": other value}`; a key active on one side only
    (a sub-key of a switched `choice()`) has only that side's entry, since None is a valid value.
    Text never appears (see `diff`). Keys in declaration order.

    Args:
        space: Tells numbers and choices apart from text.
        base: The candidate compared against, usually the baseline.
        other: The candidate shown.

    Returns:
        Key -> `{"old": ..., "new": ...}` for every changed number or choice; {} when none changed.
    """
    changed: dict[str, dict[str, Value]] = {}
    for info in space.params():
        if info.kind == "text":
            continue
        before, after = base.values.get(info.key, _ABSENT), other.values.get(info.key, _ABSENT)
        if _same(before, after):
            continue
        entry: dict[str, Value] = {}
        if before is not _ABSENT:
            entry["old"] = before
        if after is not _ABSENT:
            entry["new"] = after
        changed[info.key] = entry
    return changed


def changed_keys(base: Candidate, other: Candidate) -> list[str]:
    """Every key whose value differs, including keys present in only one of them.

    Args:
        base: The candidate compared against, e.g. the parent.
        other: The candidate shown.
    """
    keys = sorted(set(base.values) | set(other.values))
    return [key for key in keys if not _same(base.values.get(key, _ABSENT), other.values.get(key, _ABSENT))]


def _same(a: Any, b: Any) -> bool:
    """Equal and of the same type, so `1`, `1.0`, and `True` differ, as their content ids do."""
    return type(a) is type(b) and a == b


def _unified(before: str, after: str, from_label: str, to_label: str) -> list[str]:
    """One file's unified diff lines, each ending in a newline; a last line without one is marked as git does."""
    if before == after:
        return []
    # Split at "\n" only, as git does; `str.splitlines` also splits at "\f", a lone "\r", U+2028, ...
    lines = difflib.unified_diff(
        io.StringIO(before, newline="\n").readlines(),
        io.StringIO(after, newline="\n").readlines(),
        fromfile=from_label,
        tofile=to_label,
    )
    out = []
    for line in lines:
        out.append(line if line.endswith("\n") else f"{line}\n\\ No newline at end of file\n")
    return out
