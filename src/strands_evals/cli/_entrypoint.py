"""Resolve `module.path:attr` strings into Python objects, then classify them.

Mirrors the entry-point convention used by `pytest --pyargs`,
`gunicorn`, and `inspect-ai eval`. The same resolver backs both
`run --agent` and `run --task`; classification distinguishes the
shape so the caller can synthesize the right wrapper.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

EntryPointKind = Literal[
    "agent_instance",
    "agent_class",
    "callable_no_arg",
    "callable_one_arg",
    "task",
]


@dataclass(frozen=True)
class ResolvedEntryPoint:
    """A resolved `module:attr` reference annotated with its callable shape."""

    kind: EntryPointKind
    obj: Any
    spec: str


class EntryPointError(ValueError):
    """Raised when a `module:attr` string cannot be parsed or resolved."""


def _looks_like_path(module_part: str) -> bool:
    """Return True when the module portion of a spec looks like a filesystem path.

    Triggers on any of: a path separator, a leading `.` segment (`./`, `../`),
    a `.py` suffix, or a leading `~`. Plain dotted module names like
    `pkg.module` and `agent` fall through to the importlib path.
    """
    if module_part.startswith(("./", "../", "~")):
        return True
    if "/" in module_part or "\\" in module_part:
        return True
    if module_part.endswith(".py"):
        return True
    return False


def _import_from_path(path_str: str, original_spec: str) -> Any:
    """Load a module from a filesystem path and register it in `sys.modules`.

    Args:
        path_str: The module portion of the spec, with or without `.py`.
        original_spec: The full `module:attr` string, used in error messages.

    Returns:
        The loaded module object.

    Raises:
        EntryPointError: If the file does not exist or cannot be loaded.
    """
    path = Path(path_str).expanduser()
    if path.suffix != ".py":
        path = path.with_suffix(".py")
    resolved = path.resolve()
    if not resolved.is_file():
        raise EntryPointError(f"cannot import module from '{path_str}': file not found ({resolved})")

    module_name = f"_strands_evals_cli_{resolved.stem}_{abs(hash(str(resolved)))}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec_obj = importlib.util.spec_from_file_location(module_name, resolved)
    if spec_obj is None or spec_obj.loader is None:
        raise EntryPointError(f"cannot import module from '{path_str}': importlib could not build a loader")

    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    try:
        spec_obj.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise EntryPointError(f"failed to import '{original_spec}' from {resolved}: {e}") from e
    return module


def import_attr(spec: str) -> Any:
    """Resolve `"module:attr"` into the named attribute.

    Two forms are accepted:

    - **Dotted module:** `"pkg.module:attr"`. Resolved via
      :func:`importlib.import_module`. The current working directory is added
      to ``sys.path`` so a sibling file like ``agent.py`` works as
      ``agent:attr`` without ``PYTHONPATH=.``.
    - **Path-like:** anything that contains a path separator, starts with
      ``./``/``../``/``~``, or ends in ``.py`` — e.g.
      ``"./agent.py:attr"``, ``"../sibling/agent:attr"``,
      ``"/abs/path/agent.py:attr"``. Loaded via
      :func:`importlib.util.spec_from_file_location`.

    Args:
        spec: A `module:attribute` string. Exactly one `:` separates the two
            halves.

    Returns:
        The attribute referenced by `spec`.

    Raises:
        EntryPointError: If `spec` lacks a single `:`, the module is not
            importable / the file is missing, or the attribute is absent.
    """
    if spec.count(":") != 1:
        raise EntryPointError(f"invalid entry point '{spec}': expected 'module:attr' (exactly one ':')")

    module_name, attr_name = spec.split(":", 1)
    if not module_name or not attr_name:
        raise EntryPointError(f"invalid entry point '{spec}': both module and attribute parts are required")

    if _looks_like_path(module_name):
        module = _import_from_path(module_name, spec)
    else:
        cwd = str(Path.cwd())
        added = False
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
            added = True
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise EntryPointError(f"cannot import module '{module_name}': {e}") from e
        finally:
            if added:
                try:
                    sys.path.remove(cwd)
                except ValueError:
                    pass

    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise EntryPointError(f"module '{module_name}' has no attribute '{attr_name}'") from e


def _is_strands_agent_instance(obj: Any) -> bool:
    """Return True if `obj` is a `strands.Agent` instance.

    Lazy-imports `strands` so help text and unrelated subcommands don't pay
    the cost of loading the SDK.
    """
    try:
        from strands import Agent
    except ImportError:
        return False
    return isinstance(obj, Agent)


def _is_strands_agent_class(obj: Any) -> bool:
    """Return True if `obj` is a subclass of `strands.Agent`."""
    if not inspect.isclass(obj):
        return False
    try:
        from strands import Agent
    except ImportError:
        return False
    return issubclass(obj, Agent)


def _callable_arity(func: Callable[..., Any]) -> int:
    """Count required positional parameters of `func`.

    Returns the minimum positional arg count: positional parameters with no
    default and no `*args`/`**kwargs` substitute.
    """
    sig = inspect.signature(func)
    required = 0
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is param.empty and param.kind in (
            param.POSITIONAL_ONLY,
            param.POSITIONAL_OR_KEYWORD,
        ):
            required += 1
    return required


def classify_agent(obj: Any, spec: str) -> ResolvedEntryPoint:
    """Classify a resolved object passed as `--agent`.

    Accepts:
      - `strands.Agent` instance → `agent_instance`
      - `strands.Agent` subclass → `agent_class`
      - zero-arg callable → `callable_no_arg`
      - one-arg callable (`Case`) → `callable_one_arg`

    Raises:
        EntryPointError: If the object is none of the above.
    """
    if _is_strands_agent_instance(obj):
        return ResolvedEntryPoint(kind="agent_instance", obj=obj, spec=spec)
    if _is_strands_agent_class(obj):
        return ResolvedEntryPoint(kind="agent_class", obj=obj, spec=spec)
    if callable(obj):
        arity = _callable_arity(obj)
        if arity == 0:
            return ResolvedEntryPoint(kind="callable_no_arg", obj=obj, spec=spec)
        if arity == 1:
            return ResolvedEntryPoint(kind="callable_one_arg", obj=obj, spec=spec)
        raise EntryPointError(
            f"--agent target '{spec}' is callable but requires {arity} positional "
            f"arguments; expected 0 (factory) or 1 (factory taking a Case)"
        )
    raise EntryPointError(
        f"--agent target '{spec}' is neither a strands.Agent instance/subclass nor a callable; got {type(obj).__name__}"
    )


def classify_task(obj: Any, spec: str) -> ResolvedEntryPoint:
    """Classify a resolved object passed as `--task`.

    The value must be a callable accepting a single `Case` argument.
    """
    if not callable(obj):
        raise EntryPointError(f"--task target '{spec}' is not callable (got {type(obj).__name__})")
    return ResolvedEntryPoint(kind="task", obj=obj, spec=spec)


def resolve_agent(spec: str) -> ResolvedEntryPoint:
    """Convenience: `import_attr` then `classify_agent`."""
    return classify_agent(import_attr(spec), spec)


def resolve_task(spec: str) -> ResolvedEntryPoint:
    """Convenience: `import_attr` then `classify_task`."""
    return classify_task(import_attr(spec), spec)
