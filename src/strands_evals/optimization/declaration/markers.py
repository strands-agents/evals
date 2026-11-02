"""Markers for plain values and components: what may change about an agent."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from typing import Any

from ...types.optimization import Value

# `OneOf(default=)` not given: `None` can be an option, so `OneOf` needs its own sentinel.
_UNSET: Any = object()


class _Marker:
    """Base of every marker, so `component()` and `choice()` can reject the ones they can't take."""


class _ValueMarker(_Marker):
    """Base of the markers for one plain value: `Int`, `Float`, `OneOf`, `TextValue`.

    Without `default=`, `tunable` fills in the wrapped function's parameter default, so `fn()` and the
    baseline build the same agent (`component()` likewise takes the constructor parameter's default
    for an `Int` / `Float` / `OneOf` kwarg). With neither, `tunable`, `component()`, or `Space` raises
    `TypeError`, which is always the case under `build_task`'s `space=` (no function). Only `OneOf`
    falls back further, to its first option.

    Attributes:
        default: The baseline value; None (`OneOf`: unset, since None can be an option) until filled in.
    """

    default: Any

    def _with_default(self, default: Any) -> _ValueMarker:
        """A copy with `default` filled in, re-running the declaration checks on it; raises `TypeError`."""
        raise NotImplementedError


class Int(_ValueMarker):
    """An integer argument with inclusive bounds; without `default=`, the parameter's (see `_ValueMarker`)."""

    def __init__(self, min_value: int, max_value: int, *, default: int | None = None, log: bool = False):
        """Declare a tunable integer.

        Args:
            min_value: Smallest allowed value.
            max_value: Largest allowed value.
            default: The baseline value; omit to use the wrapped parameter's default.
            log: Search on a log scale.

        Raises:
            TypeError: Bounds that aren't integers or are empty, `log=True` with `min_value <= 0`, or a
                default that isn't an integer in `[min_value, max_value]`.
        """
        _check_number("Int", min_value, max_value, default, log, valid=_is_int, expected="integers")
        self.min_value = min_value
        self.max_value = max_value
        self.default = default
        self.log = log

    def _with_default(self, default: Any) -> Int:
        return Int(self.min_value, self.max_value, default=default, log=self.log)


class Float(_ValueMarker):
    """A float argument with inclusive bounds; without `default=`, the parameter's (see `_ValueMarker`)."""

    def __init__(self, min_value: float, max_value: float, *, default: float | None = None, log: bool = False):
        """Declare a tunable float.

        Args:
            min_value: Smallest allowed value.
            max_value: Largest allowed value.
            default: The baseline value; omit to use the wrapped parameter's default.
            log: Search on a log scale.

        Raises:
            TypeError: Bounds that aren't finite numbers or are empty, `log=True` with `min_value <= 0`, or a
                default that isn't a finite number in `[min_value, max_value]`.
        """
        _check_number("Float", min_value, max_value, default, log, valid=_is_finite_number, expected="finite numbers")
        # Stored as floats, so an integer default (`temperature=1`) hashes like the 1.0 a search proposes.
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.default = None if default is None else float(default)
        self.log = log

    def _with_default(self, default: Any) -> Float:
        return Float(self.min_value, self.max_value, default=default, log=self.log)


class OneOf(_ValueMarker):
    """A choice among plain values; without `default=`, the parameter's, else the first option."""

    def __init__(self, *options: str | int | float | bool | None, default: Value = _UNSET):
        """Declare a choice among plain values.

        Args:
            *options: The plain values to choose from.
            default: The baseline value; omit to use the wrapped parameter's default, or the first option.

        Raises:
            TypeError: No options, an option that isn't a plain value (NaN and infinity included),
                duplicate options, or a default that isn't one of the options.
        """
        if not options:
            raise TypeError("OneOf needs at least one option")
        for option in options:
            if not isinstance(option, Value):
                raise TypeError(
                    f"OneOf options must be str, int, float, bool, or None, got {option!r}; "
                    "use choice(component(...)) for objects"
                )
            if isinstance(option, float) and not math.isfinite(option):
                # NaN never equals itself, so it could never be chosen or validated, and neither
                # NaN nor infinity is valid JSON in a candidate's values.
                raise TypeError(f"OneOf options must be finite, got {option!r}")
        if len({(type(option), option) for option in options}) != len(options):
            raise TypeError(f"OneOf options must be distinct, got {list(options)!r}")
        if default is not _UNSET and not _is_option(default, options):
            raise TypeError(f"OneOf default {default!r} is not one of {list(options)!r}")
        self.options = options
        self.default = default

    def _with_default(self, default: Any) -> OneOf:
        return OneOf(*self.options, default=default)


class TextValue(_ValueMarker):
    """A text knob; `size_cap` rejects prompt bloat. Without `default=`, the parameter's (see `_ValueMarker`)."""

    def __init__(
        self, *, default: str | None = None, size_cap: int | None = None, edit_instructions: str | None = None
    ):
        """Declare a tunable text.

        Args:
            default: The baseline text; omit to use the wrapped parameter's default.
            size_cap: Max characters; None means `max(2000, 3 * len(default))`.
            edit_instructions: Shown to the proposer next to this value.

        Raises:
            TypeError: A default that isn't a str, or a `size_cap` that isn't a positive integer.
        """
        if default is not None and not isinstance(default, str):
            raise TypeError(f"TextValue default must be a str, got {type(default).__name__}")
        _check_size_cap(size_cap)
        self.default = default
        self.size_cap = size_cap
        self.edit_instructions = edit_instructions

    def _with_default(self, default: Any) -> TextValue:
        return TextValue(default=default, size_cap=self.size_cap, edit_instructions=self.edit_instructions)


class Component(_Marker):
    """One constructible alternative, returned by `component()`.

    Create it with `component()`, which fills defaults and probes the constructor; constructing it
    directly skips those checks.

    Attributes:
        factory: The class or callable to construct.
        fixed: Constructor kwargs passed through as is.
        tunable: Constructor kwargs declared as `Int` / `Float` / `OneOf`, each with a default.
        label: The alternative's label inside a `choice()`; the class name by default.
    """

    def __init__(
        self,
        factory: Callable[..., Any],
        fixed: dict[str, Any],
        tunable: dict[str, Int | Float | OneOf],
        label: str | None = None,
    ):
        self.factory = factory
        self.fixed = fixed
        self.tunable = tunable
        self.label: str = label if label is not None else getattr(factory, "__name__", type(factory).__name__)

    def build(self, values: dict[str, Any]) -> Any:
        """A fresh instance through the real constructor, with `values` for the tunable kwargs."""
        return self.factory(**self.fixed, **values)

    def _labeled(self, label: str) -> Component:
        """A copy carrying `label`; a copy, because one component may sit in several choices."""
        return Component(self.factory, self.fixed, self.tunable, label)


class Choice(_Marker):
    """A swap between alternatives, returned by `choice()`.

    Create it with `choice()`, which checks the labels; constructing it directly skips that.

    Attributes:
        alternatives: Label -> a `Component` or a fixed value, baseline first; each component's
            `label` is its key.
    """

    def __init__(self, alternatives: dict[str, Component | Any]):
        self.alternatives = alternatives


def component(factory: Callable[..., Any], /, **kwargs: Any) -> Component:
    """Declare one constructible alternative; `Int`/`Float`/`OneOf` kwargs are tunable, others pass through.

    `factory` is positional-only, so a constructor kwarg named `factory` can still be passed. An
    `Int` / `Float` / `OneOf` kwarg without `default=` takes the constructor parameter's default
    (a `OneOf` without one, its first option), e.g.
    `component(SlidingWindowConversationManager, window_size=Int(4, 40), pin_first=OneOf(None, 1))`
    (the constructor's defaults, 40 and None, are the baseline). A constructor default of None
    doesn't count for an `Int` / `Float`: pass `default=`.

    The constructor is probed once, here, with its defaults, so its side effects (a client it
    creates) run at declaration and the probed instance is discarded. A positional label comes from
    the factory's `__name__` (`"partial"` for a `functools.partial`, `"<lambda>"` for a lambda), so
    label such alternatives by keyword in `choice()`.

    Other kwargs pass through as is, so every build gets the same objects: build stateful ones (an
    `Agent`, a client) inside the factory, or they are shared across trials.

    Args:
        factory: The class or callable to construct.
        **kwargs: Constructor arguments; `Int` / `Float` / `OneOf` markers among them are tunable.

    Raises:
        TypeError: The constructor rejects its own defaults (the exception is chained), an `Int` /
            `Float` kwarg has no default anywhere (a constructor default of None counts as none), a
            default that doesn't fit its marker, or another
            marker (`TextValue`, a nested component or choice) is passed as a kwarg.
    """
    if not callable(factory):
        raise TypeError(f"component() needs a class or callable, got {factory!r}")
    result = Component(factory, {}, {})
    name = result.label
    parameters = _parameters(factory)
    for key, value in kwargs.items():
        if isinstance(value, (Int, Float, OneOf)):
            result.tunable[key] = _with_constructor_default(name, key, value, parameters)
        elif isinstance(value, _Marker):
            raise TypeError(
                f"component({name}): {key}= is a {type(value).__name__}; "
                "only Int, Float, and OneOf arguments are tunable"
            )
        else:
            result.fixed[key] = value
    try:
        result.build({key: marker.default for key, marker in result.tunable.items()})
    except Exception as error:
        raise TypeError(f"component({name}): the constructor rejects its own defaults: {error}") from error
    return result


def choice(*alternatives: Component, **labeled: Component | Any) -> Choice:
    """Declare a swap between alternatives; the first is the baseline.

    Pass components positionally, labeled by class name, or by keyword, labeled by the keyword:
    `choice(window=component(...), summary=component(...))`. The keyword form is needed when the same
    class appears twice. A keyword alternative may also be a fixed value, passed through as is:
    `choice(on=component(ToolCallLimiter, max_calls=Int(1, 20)), off=None)`. Labels become key
    segments, so they contain no `.` or `/`. Each component in the result is a copy carrying its
    label, so one component can sit in several choices.

    Args:
        *alternatives: From `component()`, labeled by class name.
        **labeled: From `component()` or a fixed value, labeled by keyword.

    Raises:
        TypeError: No alternatives, positional and keyword alternatives mixed, a positional fixed
            value, a marker other than a component, the same class twice positionally, or a label
            with `.` or `/`.
    """
    if alternatives and labeled:
        raise TypeError("choice() takes alternatives either positionally or by keyword, not both")
    if not alternatives and not labeled:
        raise TypeError("choice() needs at least one alternative")
    for alternative in alternatives:
        if not isinstance(alternative, Component):
            raise TypeError(
                f"choice() takes positional component(...) alternatives, got {type(alternative).__name__}; "
                "pass a fixed value by keyword, e.g. choice(on=component(...), off=None)"
            )
    for label, alternative in labeled.items():
        if isinstance(alternative, _Marker) and not isinstance(alternative, Component):
            raise TypeError(
                f"choice({label}=...) is a {type(alternative).__name__}; an alternative is a component(...) "
                "or a fixed value"
            )
    pairs = list(labeled.items()) if labeled else [(alternative.label, alternative) for alternative in alternatives]
    names = [label for label, _ in pairs]
    if len(set(names)) != len(names):
        raise TypeError(
            f"choice() has the same class twice ({names}); label them by keyword, e.g. choice(low=..., high=...)"
        )
    for label in names:
        if not label or "." in label or "/" in label:
            raise TypeError(f"choice() labels must be non-empty and contain no '.' or '/', got {label!r}")
    return Choice(
        {
            label: alternative._labeled(label) if isinstance(alternative, Component) else alternative
            for label, alternative in pairs
        }
    )


def _is_option(value: Any, options: Sequence[Any]) -> bool:
    """Whether `value` is one of `options`, comparing types too (so `True` is not the option `1`)."""
    return any(type(option) is type(value) and option == value for option in options)


def _default_size_cap(baseline: str) -> int:
    """`max(2000, 3 * len(baseline))`; shared with `files.py`."""
    return max(2000, 3 * len(baseline))


def _is_int(value: Any) -> bool:
    """Whether `value` is an int; `bool` is a subclass of int but doesn't count."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    """Whether `value` is a finite int or float, excluding `bool`; NaN would slip past every range check."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_number(
    kind: str, min_value: Any, max_value: Any, default: Any, log: bool, *, valid: Callable[[Any], bool], expected: str
) -> None:
    """The `Int` / `Float` declaration checks; `valid` says what a bound or default may be. Raises `TypeError`."""
    if not (valid(min_value) and valid(max_value)):
        raise TypeError(f"{kind} bounds must be {expected}, got min_value={min_value!r}, max_value={max_value!r}")
    if min_value > max_value:
        raise TypeError(f"{kind} bounds are empty: min_value={min_value!r} > max_value={max_value!r}")
    if log and min_value <= 0:
        raise TypeError(f"{kind}(log=True) needs min_value > 0, got min_value={min_value!r}")
    if default is not None and not (valid(default) and min_value <= default <= max_value):
        raise TypeError(f"{kind} default must lie in [{min_value!r}, {max_value!r}] ({expected} only), got {default!r}")


def _check_size_cap(size_cap: int | None) -> None:
    """Raise `TypeError` unless `size_cap` is None or a positive integer; shared with `tools.py`."""
    if size_cap is not None and (not _is_int(size_cap) or size_cap < 1):
        raise TypeError(f"size_cap must be a positive integer, got {size_cap!r}")


def _parameters(factory: Callable[..., Any]) -> dict[str, inspect.Parameter] | None:
    """The factory's parameters, or None when its signature can't be read (some builtins)."""
    try:
        return dict(inspect.signature(factory).parameters)
    except (TypeError, ValueError):
        return None


def _with_constructor_default(
    name: str, key: str, marker: Int | Float | OneOf, parameters: dict[str, inspect.Parameter] | None
) -> Int | Float | OneOf:
    """`marker` with a missing default taken from the constructor parameter `key`; raises `TypeError`.

    A `OneOf` with neither takes its first option, so every returned marker has a default.
    """
    if marker.default is not (_UNSET if isinstance(marker, OneOf) else None):
        return marker
    parameter = (parameters or {}).get(key)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        if isinstance(marker, OneOf):
            return marker._with_default(marker.options[0])
        raise TypeError(f"component({name}): {key}= has no default; pass default= or give the parameter one")
    if parameter.default is None and not isinstance(marker, OneOf):
        # As under tunable: None is never an Int / Float value, so the probe would build with a value
        # the space rejects. A OneOf may list None as an option, so it goes through the check below.
        raise TypeError(
            f"component({name}): the constructor's default for {key}= is None; pass default= to {type(marker).__name__}"
        )
    try:
        return marker._with_default(parameter.default)
    except TypeError as error:
        raise TypeError(f"component({name}): the default of {key}= doesn't fit its marker: {error}") from error
