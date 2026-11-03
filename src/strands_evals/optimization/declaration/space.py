"""The declared space: flat typed keys, which are active, validation, and resolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from typing_extensions import TypeAlias

from ...types.optimization import Value, Values
from .files import FileText, JsonField, SkillLibrary
from .markers import Choice, Component, Float, Int, OneOf, TextValue
from .tools import ToolDescriptions

Marker: TypeAlias = (
    Int | Float | OneOf | TextValue | Component | Choice | FileText | JsonField | SkillLibrary | ToolDescriptions
)

ParamKind = Literal["int", "float", "one_of", "choice", "text"]


@dataclass(frozen=True)
class FileSource:
    """Where a file-backed text value lives."""

    path: Path
    pointer: str | None
    as_path: bool


@dataclass(frozen=True)
class ParamInfo:
    """One flat key of the space.

    `active_when` is `(choice key, label)` when the key exists only under one alternative of a
    `choice()`, else None. E.g. `cm=choice(window=component(SlidingWindowConversationManager,
    window_size=Int(4, 40)), summary=component(...))` gives `cm` (always active) and
    `cm.window.window_size` with `active_when=("cm", "window")`. When a proposal switches `cm` to
    `summary`, `Space.complete` drops `cm.window.window_size` (kept, it would change the candidate
    id without changing the agent) and fills the keys `summary` brings in. Validation, TPE, the score
    model's design, and `Nested` read it too. A choice inside an alternative carries its own
    `active_when`, so conditions chain upward. Text keys can't sit under a `choice()`, so theirs is
    always None.

    `min_value` / `max_value` are the inclusive bounds of an `Int` / `Float`. Fields after `default`
    default to None (`log` to False), so each kind sets only its own.
    """

    key: str
    name: str
    kind: ParamKind
    default: Value
    min_value: float | None = None
    max_value: float | None = None
    log: bool = False
    options: tuple[Value, ...] | None = None
    size_cap: int | None = None
    edit_instructions: str | None = None
    active_when: tuple[str, str] | None = None
    source: FileSource | None = None


class Space:
    """Flattens markers into dotted keys and knows which keys are active under which values."""

    def __init__(self, markers: Mapping[str, Marker]):
        """Flatten the markers; components were already probed by `component()`.

        A bare component counts as a one-alternative choice.

        Raises:
            TypeError: A name contains `.` or `/`, or a default fails `validate` (e.g. a text default
                over its size cap).
        """
        raise NotImplementedError

    def params(self) -> list[ParamInfo]:
        """Every flat key, in declaration order, a choice before its sub-keys."""
        raise NotImplementedError

    def param(self, key: str) -> ParamInfo:
        """One flat key's info; an unknown key raises `KeyError` naming the known keys."""
        raise NotImplementedError

    def is_active(self, key: str, values: Values) -> bool:
        """Whether a key is active.

        Args:
            key: A flat key.
            values: The candidate's values; a missing choice counts as its default.

        Returns:
            Whether `key` is active under the alternatives chosen in `values`.
        """
        raise NotImplementedError

    def complete(
        self, values: Values, *, fill: Callable[[str], tuple[Value, str] | None] | None = None
    ) -> dict[str, Value]:
        """Drop keys inactive under the chosen alternatives and fill missing active keys.

        Unknown keys are kept, so `validate` reports them.

        Args:
            values: The candidate's values.
            fill: `fill(key) -> (value, source id)` for a missing active key, or None to use its
                default; a tuple because None is a valid value.
        """
        raise NotImplementedError

    def validate(self, values: Values) -> None:
        """Check bounds, size caps, formats, active keys, and that nothing is missing.

        Types are strict, so equal candidates hash alike: `Int` takes an int (not a bool), `Float` a
        float (not an int), and `OneOf` compares type and value.

        Raises:
            ValueError: Listing every problem.
        """
        raise NotImplementedError

    def resolve(self, name: str, values: Values, workspace: Path | None) -> Any:
        """What the factory receives for one declared name: a fresh component, a value, or a path.

        `values` must already pass `validate`; nothing is checked here. A component's fixed kwargs
        pass through as is, so every build shares those objects.

        Args:
            name: A declared name, not a flat key.
            values: The candidate's values.
            workspace: The candidate's copy, or None when no value reaches the agent as a path.
        """
        raise NotImplementedError

    def create_candidate_copy(self, values: Values, directory: Path) -> Path:
        """Copy each file-backed value's folder into `directory`, then write every active file-backed key.

        Each source folder goes to `directory/<folder name>-<short hash of its original path>/`; a folder
        inside another is copied once, as part of the outer one. Each active key is written by its
        marker's `write()`. Built in a temp folder and renamed into place, so an existing `directory`
        is always complete and a second call returns at once. Originals are never written.

        Args:
            values: The candidate's values.
            directory: The candidate's copy, `<run>/candidates/<id>/optimization_files/`.

        Returns:
            `directory`.
        """
        raise NotImplementedError
