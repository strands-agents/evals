"""Markers for plain values and components: what may change about an agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


class Int:
    """An integer argument with inclusive bounds; `default` must lie in `[ge, le]`."""

    def __init__(self, ge: int, le: int, *, default: int, log: bool = False):
        raise NotImplementedError


class Float:
    """A float argument with inclusive bounds; `default` must lie in `[ge, le]`."""

    def __init__(self, ge: float, le: float, *, default: float, log: bool = False):
        raise NotImplementedError


_FIRST: Any = object()


class OneOf:
    """A choice among plain values; the first option is the default unless one is given."""

    def __init__(self, *options: str | int | float | bool | None, default: Any = _FIRST):
        raise NotImplementedError


class TextValue:
    """A text knob; `size_cap` rejects prompt bloat (default: `_default_size_cap(default)`)."""

    def __init__(self, *, default: str, size_cap: int | None = None, edit_instructions: str | None = None):
        raise NotImplementedError


class Component:
    """One constructible alternative, returned by `component()`."""


class Choice:
    """A swap between components, returned by `choice()`; the first alternative is the baseline."""


def component(factory: Callable[..., Any], **kwargs: Any) -> Component:
    """Declare one constructible alternative; `Int`/`Float` kwargs are tunable, others pass through."""
    raise NotImplementedError


def choice(*alternatives: Component, labels: Sequence[str] | None = None) -> Choice:
    """Declare a swap between components.

    Labels default to class names. The same class twice without labels is a `TypeError` at
    declaration. Labels must be unique and contain no dots (they become key segments).
    """
    raise NotImplementedError


def _default_size_cap(baseline: str) -> int:
    """`max(2000, 3 * len(baseline))`; shared with `files.py`."""
    raise NotImplementedError
