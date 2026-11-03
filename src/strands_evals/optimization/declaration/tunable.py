"""The `tunable` mounts and the factory they return."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from ...types.optimization import Value, Values
from .markers import _fill_default, _ValueMarker
from .space import Marker, Space

logger = logging.getLogger(__name__)

_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


@dataclass(frozen=True)
class BuildContext:
    """What `Optimizer(build_task=...)` receives, once per candidate.

    A dataclass, not a record: it holds a callable and is never saved. New fields may be added
    later without breaking existing `build_task` functions.

    Attributes:
        values: The candidate's typed flat values.
        resolve: `resolve(name)` returns what the factory path would pass for that declared name:
            a fresh component, a value, or a path inside the candidate's copy.
        workspace: The candidate's copy of the file-backed folders
            (`<run>/candidates/<id>/optimization_files/`), or None when the space has no
            file-backed values. Lets a config-defined agent load the whole edited file, e.g. the
            `agent.json` a `JsonField` points into.
    """

    values: Values
    resolve: Callable[[str], Any]
    workspace: Path | None


class _FactoryArgumentError(TypeError):
    """A bad argument to a bound factory; a setup error, never a trial status.

    Raised for a resource or kwarg named like a declared marker, or a call kwarg the user's function
    doesn't accept. Never an `agent_failure`: under `agent_failure_score="worst"` that would score every
    trial at its worst end and the run would go on measuring nothing.
    """


class TunableFactory:
    """A user's agent factory with declared tunable parameters.

    Built by `tunable`; wraps the user's function (name, docstring, signature) without changing it.
    """

    space: Space

    def __init__(self, fn: Callable[..., Any], markers: Mapping[str, Marker]):
        """Check the declared names against `fn` and build the space; raises `TypeError`."""
        if isinstance(fn, TunableFactory):
            raise TypeError(f"{fn._label} is already tunable; declare every marker in one tunable(...)")
        if not callable(fn):
            raise TypeError(f"tunable() needs a function, got {fn!r}")
        self._fn = fn
        self._signature = inspect.signature(fn)
        # Not every callable has a __name__ (functools.partial); used in error messages and logs.
        self._label: str = getattr(fn, "__name__", repr(fn))
        self._names = tuple(markers)
        parameters = self._signature.parameters
        self._accepted = frozenset(name for name, p in parameters.items() if p.kind in _KEYWORD_KINDS)
        self._takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        self.space = Space(_fill_defaults(markers, self._label, parameters, self._takes_kwargs))
        self._baseline = self.space.complete({})
        # updated=(): copying fn.__dict__ could overwrite this factory's own attributes.
        functools.update_wrapper(self, fn, updated=())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Build the baseline (every value at its default); never copies files.

        Arguments passed here, by position or by keyword, win over the declared values, as in a
        plain call of the function.
        """
        given = self._signature.bind_partial(*args, **kwargs).arguments
        resolved = {
            name: self.space.resolve(name, self._baseline, None)
            for name in self._names
            if name not in given and name not in kwargs  # a name caught by **kwargs is only in kwargs
        }
        return self._fn(*args, **kwargs, **resolved)

    def validate(self, values: Values) -> None:
        """Same as `self.space.validate(values)`; raises `ValueError`."""
        self.space.validate(values)

    def bind(self, values: Values, *, workspace: Path | None = None) -> Callable[..., Any]:
        """Return a factory that builds fresh components with `values` bound on every call.

        Fills missing keys with defaults but drops nothing (an inactive key raises `ValueError`), and
        validates before any build. Copies files only if an active value reaches the agent as a path
        (`as_path=True`, `SkillLibrary`): into `workspace` (cached), or with `workspace=None` into one
        temp folder per `bind()` call, deleted when the bound factory is garbage-collected.

        The bound factory takes values for the function's non-tunable parameters, by two rules:
        resources from `environment()` (handed over through its private `_with_resources(resources)`)
        are passed only if the function declares that parameter by name; kwargs passed in the call go
        through as in a normal call, and one the function doesn't accept raises `_FactoryArgumentError`.
        Either kind named like a declared marker raises `_FactoryArgumentError`.

        Args:
            values: The candidate's values.
            workspace: Where to write the candidate's file copy; None outside a run.

        Returns:
            A factory that builds fresh components on every call.

        Raises:
            ValueError: The values fail validation.
        """
        bound_values = {**self.space.complete(values), **values}
        self.space.validate(bound_values)
        return _BoundFactory(self, bound_values, workspace, {})


class _BoundFactory:
    """What `TunableFactory.bind` returns: builds fresh components with the bound values on every call."""

    def __init__(
        self,
        factory: TunableFactory,
        values: dict[str, Value],
        workspace: Path | None,
        resources: Mapping[str, Any],
    ):
        self._factory = factory
        self._values = values
        self._workspace = workspace
        self._resources = resources

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Build with the bound values; `kwargs` go to the function's non-tunable parameters."""
        factory = self._factory
        if args:
            raise _FactoryArgumentError(
                f"{factory._label}: a bound factory takes keyword arguments only, got {len(args)} positional"
            )
        _check_not_markers(factory, kwargs, "arguments")
        if not factory._takes_kwargs:
            unknown = sorted(name for name in kwargs if name not in factory._accepted)
            if unknown:
                raise _FactoryArgumentError(f"{factory._label} takes no arguments named {unknown}")
        resources = {
            name: value
            for name, value in self._resources.items()
            if name in factory._accepted and name not in kwargs  # a call kwarg wins, as passed explicitly
        }
        resolved = {name: factory.space.resolve(name, self._values, self._workspace) for name in factory._names}
        return factory._fn(**resolved, **resources, **kwargs)

    def _with_resources(self, resources: Mapping[str, Any]) -> _BoundFactory:
        """The same bound factory, also passing each resource the function declares by name.

        `**kwargs` doesn't count as declaring one, so one environment can serve factories that take
        different resources; the rest are dropped.

        Raises:
            _FactoryArgumentError: A resource is named like a declared marker.
        """
        _check_not_markers(self._factory, resources, "resources")
        dropped = sorted(name for name in resources if name not in self._factory._accepted)
        if dropped:
            logger.debug("factory=<%s>, resources=<%s> | factory does not declare them", self._factory._label, dropped)
        return _BoundFactory(self._factory, self._values, self._workspace, {**self._resources, **resources})


@overload
def tunable(fn: Callable[..., Any], /, *, inject: Mapping[str, Marker]) -> TunableFactory: ...


@overload
def tunable(**markers: Marker) -> Callable[[Callable[..., Any]], TunableFactory]: ...


def tunable(
    fn: Callable[..., Any] | None = None, /, **kwargs: Any
) -> TunableFactory | Callable[[Callable[..., Any]], TunableFactory]:
    """Declare tunable parameters: `tunable(fn, inject={...})` or `@tunable(name=marker, ...)`.

    A marker naming no parameter of `fn` (unless `fn` takes `**kwargs`) raises `TypeError`. A marker
    without `default=` takes the parameter's own default, so `fn()` and the baseline build the same
    agent; a marker with neither raises `TypeError` (see `_ValueMarker`).

    Args:
        fn: The factory; omit when used as a decorator.
        **kwargs: `inject={name: marker}` when `fn` is given, otherwise the markers themselves.

    Returns:
        A `TunableFactory`, or a decorator returning one.

    Raises:
        TypeError: A marker names no parameter, has no default anywhere, or is of an unknown type;
            or `fn` is already a `TunableFactory` (declare every marker in one `tunable(...)`).
    """
    if fn is not None:
        if set(kwargs) != {"inject"} or not isinstance(kwargs["inject"], Mapping):
            raise TypeError("tunable(fn, ...) takes only inject={name: marker}")
        return TunableFactory(fn, kwargs["inject"])
    if isinstance(kwargs.get("inject"), Mapping):
        raise TypeError("inject= needs the function: tunable(fn, inject={name: marker})")

    def decorate(function: Callable[..., Any]) -> TunableFactory:
        return TunableFactory(function, kwargs)

    return decorate


def _check_not_markers(factory: TunableFactory, given: Mapping[str, Any], kind: str) -> None:
    """Raise `_FactoryArgumentError` if any name in `given` is a declared marker's."""
    conflicts = sorted(name for name in given if name in factory._names)
    if conflicts:
        raise _FactoryArgumentError(
            f"{kind} {conflicts} are named like declared markers of {factory._label}; tunable values come "
            "only from the candidate"
        )


def _fill_defaults(
    markers: Mapping[str, Marker], fn_name: str, parameters: Mapping[str, inspect.Parameter], takes_kwargs: bool
) -> dict[str, Marker]:
    """Check each name against the function's parameters; a marker without `default=` takes the parameter's."""
    filled: dict[str, Marker] = {}
    for name, marker in markers.items():
        parameter = parameters.get(name)
        if parameter is not None and parameter.kind not in _KEYWORD_KINDS:
            raise TypeError(f"{name}: {fn_name} takes {name!r} as {parameter.kind.description}, not by keyword")
        if parameter is None and not takes_kwargs:
            raise TypeError(f"{name}: {fn_name} has no parameter {name!r} and takes no **kwargs")
        if parameter is not None and parameter.default is not inspect.Parameter.empty:
            filled[name] = _with_parameter_default(name, marker, parameter.default)
        else:
            filled[name] = marker
    return filled


def _with_parameter_default(name: str, marker: Marker, default: Any) -> Marker:
    """`marker` with the parameter's `default` filled in, if the marker has none; raises `TypeError`."""
    if not isinstance(marker, _ValueMarker):
        return marker  # components, choices, tools, and files take no parameter default
    return _fill_default(marker, default, where=name, source="the parameter's default")
