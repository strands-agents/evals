"""The `tunable` mounts and the factory they return."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from ...types.optimization import Value
from .space import Space

if TYPE_CHECKING:
    from .space import Marker


class _ResourceConflictError(TypeError):
    """A resource passed to a bound factory is named like a declared marker; a setup error, not a trial status."""


class TunableFactory:
    """A user's agent factory with declared tunable parameters."""

    space: Space

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Build the baseline (every value at its default); never copies files."""
        raise NotImplementedError

    def validate(self, values: Mapping[str, Value]) -> None:
        """Same as `self.space.validate(values)`; raises `ValueError`."""
        raise NotImplementedError

    def bind(self, values: Mapping[str, Value], *, workspace: Path | None = None) -> Callable[..., Any]:
        """Return a factory that builds fresh components with `values` bound on every call.

        Copies files only if an active value reaches the agent as a path (`as_path=True`,
        `SkillLibrary`): into `workspace` (cached), or with `workspace=None` into one temp folder per
        `bind()` call, deleted when the bound factory is garbage-collected. The bound factory takes
        resource kwargs, each passed only if the user's function declares it; a kwarg named like a
        declared marker raises `_ResourceConflictError`.
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

    A marker naming no parameter of `fn` (unless `fn` takes `**kwargs`) raises `TypeError`.
    """
    raise NotImplementedError
