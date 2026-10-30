"""Markers for plain values and components: what may change about an agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...types.optimization import Value

# `OneOf(default=)` not given: `None` can be an option, so `OneOf` needs its own sentinel.
_UNSET: Any = object()


class Int:
    """An integer argument with inclusive bounds; the default must lie in `[ge, le]`.

    Without `default=`, the wrapped function's parameter default is used. With neither (always the
    case under `build_task`'s `space=`, which has no function), `tunable` or `Space` raises `TypeError`.
    """

    def __init__(self, ge: int, le: int, *, default: int | None = None, log: bool = False):
        """Declare a tunable integer.

        Args:
            ge: Smallest allowed value.
            le: Largest allowed value.
            default: The baseline value; omit to use the wrapped parameter's default.
            log: Search on a log scale.

        Raises:
            TypeError: A default outside `[ge, le]`.
        """
        raise NotImplementedError


class Float:
    """A float argument with inclusive bounds; the default must lie in `[ge, le]`.

    Without `default=`, the wrapped function's parameter default is used; with neither, `tunable` or
    `Space` raises `TypeError`.
    """

    def __init__(self, ge: float, le: float, *, default: float | None = None, log: bool = False):
        """Declare a tunable float.

        Args:
            ge: Smallest allowed value.
            le: Largest allowed value.
            default: The baseline value; omit to use the wrapped parameter's default.
            log: Search on a log scale.

        Raises:
            TypeError: A default outside `[ge, le]`.
        """
        raise NotImplementedError


class OneOf:
    """A choice among plain values.

    The default is `default=` if given, else the wrapped function's parameter default, else the first
    option. A default that isn't one of the options raises `TypeError` at declaration.
    """

    def __init__(self, *options: str | int | float | bool | None, default: Value = _UNSET):
        """Declare a choice among plain values.

        Args:
            *options: The plain values to choose from.
            default: The baseline value; omit to use the wrapped parameter's default, or the first option.

        Raises:
            TypeError: The default is not one of the options.
        """
        raise NotImplementedError


class TextValue:
    """A text knob; `size_cap` rejects prompt bloat (default: `_default_size_cap(default)`).

    Without `default=`, the wrapped function's parameter default is used; with neither, `tunable`
    raises `TypeError`.
    """

    def __init__(
        self, *, default: str | None = None, size_cap: int | None = None, edit_instructions: str | None = None
    ):
        """Declare a tunable text.

        Args:
            default: The baseline text; omit to use the wrapped parameter's default.
            size_cap: Max characters; None means `max(2000, 3 * len(default))`.
            edit_instructions: Shown to the proposer next to this value.
        """
        raise NotImplementedError


class Component:
    """One constructible alternative, returned by `component()`.

    Attributes:
        factory: The class or callable to construct.
        fixed: Constructor kwargs passed through as is.
        tunable: Constructor kwargs declared as `Int` / `Float`.
        label: The alternative's label inside a `choice()`; the class name by default.
    """

    factory: Callable[..., Any]
    fixed: dict[str, Any]
    tunable: dict[str, Int | Float]
    label: str

    def build(self, values: dict[str, Any]) -> Any:
        """A fresh instance through the real constructor, with `values` for the tunable kwargs."""
        raise NotImplementedError


class Choice:
    """A swap between alternatives, returned by `choice()`.

    Attributes:
        alternatives: Label -> a `Component` or a fixed value, baseline first.
    """

    alternatives: dict[str, Component | Any]


def component(factory: Callable[..., Any], /, **kwargs: Any) -> Component:
    """Declare one constructible alternative; `Int`/`Float` kwargs are tunable, others pass through.

    `factory` is positional-only, so a constructor kwarg named `factory` can still be passed. An
    `Int` / `Float` kwarg without `default=` takes the constructor parameter's default. The
    constructor is probed once, here, with its defaults.

    Other kwargs pass through as is, so every build gets the same objects: build stateful ones (an
    `Agent`, a client) inside the factory, or they are shared across trials.

    Args:
        factory: The class or callable to construct.
        **kwargs: Constructor arguments; `Int` / `Float` markers among them are tunable.

    Raises:
        TypeError: The constructor rejects its own defaults (the exception is chained), an `Int` /
            `Float` kwarg has no default anywhere, or another marker is passed as a kwarg.
    """
    raise NotImplementedError


def choice(*alternatives: Component, **labeled: Component | Any) -> Choice:
    """Declare a swap between alternatives; the first is the baseline.

    Pass components positionally, labeled by class name, or by keyword, labeled by the keyword:
    `choice(window=component(...), summary=component(...))`. The keyword form is needed when the same
    class appears twice. A keyword alternative may also be a fixed value, passed through as is:
    `choice(on=component(ToolCallLimiter, max_calls=Int(1, 20)), off=None)`. Labels become key
    segments, so they contain no `.` or `/`.

    Args:
        *alternatives: From `component()`, labeled by class name.
        **labeled: From `component()` or a fixed value, labeled by keyword.

    Raises:
        TypeError: No alternatives, positional and keyword alternatives mixed, a positional fixed
            value, the same class twice positionally, or a label with `.` or `/`.
    """
    raise NotImplementedError


def _default_size_cap(baseline: str) -> int:
    """`max(2000, 3 * len(baseline))`; shared with `files.py`."""
    raise NotImplementedError
