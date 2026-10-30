"""The `tunable` mounts and the factory they return."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from ...types.optimization import Values
from .space import Marker, Space


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
    doesn't accept. Never an `agent_failure`: under `crash_score="worst"` that would score every
    trial at its worst end and the run would go on measuring nothing.
    """


class TunableFactory:
    """A user's agent factory with declared tunable parameters."""

    space: Space

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Build the baseline (every value at its default); never copies files."""
        raise NotImplementedError

    def validate(self, values: Values) -> None:
        """Same as `self.space.validate(values)`; raises `ValueError`."""
        raise NotImplementedError

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
        raise NotImplementedError


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
    agent; a marker with neither raises `TypeError`.

    Args:
        fn: The factory; omit when used as a decorator.
        **kwargs: `inject={name: marker}` when `fn` is given, otherwise the markers themselves.

    Returns:
        A `TunableFactory`, or a decorator returning one.

    Raises:
        TypeError: A marker names no parameter, has no default anywhere, or is of an unknown type;
            or `fn` is already a `TunableFactory` (declare every marker in one `tunable(...)`).
    """
    raise NotImplementedError
