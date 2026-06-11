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

from ..evaluators import (
    CoherenceEvaluator,
    ConcisenessEvaluator,
    CorrectnessEvaluator,
    Equals,
    FaithfulnessEvaluator,
    GoalSuccessRateEvaluator,
    HarmfulnessEvaluator,
    HelpfulnessEvaluator,
    InstructionFollowingEvaluator,
    RefusalEvaluator,
    ResponseRelevanceEvaluator,
    StereotypingEvaluator,
    ToolParameterAccuracyEvaluator,
    ToolSelectionAccuracyEvaluator,
)
from ..evaluators.evaluator import Evaluator

# Built-in evaluator shortnames for `--evaluator NAME`. Only evaluators that
# instantiate cleanly with no required arguments are listed here — anything
# requiring config (rubrics, target values, tool names) belongs in an
# experiment file. `equals` is included because `Equals()` falls back to the
# case's `expected_output` when no value is supplied, which is exactly what
# `--expected-output` provides in ad-hoc mode.
BUILTIN_EVALUATOR_SHORTNAMES: dict[str, type[Evaluator]] = {
    "coherence": CoherenceEvaluator,
    "conciseness": ConcisenessEvaluator,
    "correctness": CorrectnessEvaluator,
    "equals": Equals,
    "faithfulness": FaithfulnessEvaluator,
    "goal-success-rate": GoalSuccessRateEvaluator,
    "harmfulness": HarmfulnessEvaluator,
    "helpfulness": HelpfulnessEvaluator,
    "instruction-following": InstructionFollowingEvaluator,
    "refusal": RefusalEvaluator,
    "response-relevance": ResponseRelevanceEvaluator,
    "stereotyping": StereotypingEvaluator,
    "tool-parameter-accuracy": ToolParameterAccuracyEvaluator,
    "tool-selection-accuracy": ToolSelectionAccuracyEvaluator,
}

EntryPointKind = Literal[
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
        spec: A `module:attribute` string. The attribute is split off at the
            last `:` so Windows absolute paths (`C:\\path\\to\\agent.py:attr`),
            whose drive letter introduces a second colon, still resolve
            correctly.

    Returns:
        The attribute referenced by `spec`.

    Raises:
        EntryPointError: If `spec` lacks a `:`, either half is empty, the
            module is not importable / the file is missing, or the attribute
            is absent.
    """
    if ":" not in spec:
        raise EntryPointError(f"invalid entry point '{spec}': expected 'module:attr'")

    module_name, attr_name = spec.rsplit(":", 1)
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


_FACTORY_HINT = (
    "--agent must reference a factory callable, e.g. "
    "'def build_agent() -> Agent: return Agent(...)' "
    "or 'def build_agent(case: Case) -> Agent: ...'. "
    "Each call must return a fresh agent so concurrent runs and "
    "shared conversation state cannot leak between cases. The CLI's "
    "per-case freshness guarantee is enforced for strands.Agent only; "
    "non-Strands callables are accepted but cross-case isolation is the "
    "factory's responsibility."
)


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

    Accepts callables — the CLI invokes the resolved object once per case and
    uses the return value as the agent. Two shapes are explicitly rejected
    because they break the per-case freshness guarantee for `strands.Agent`:

    - **Prebuilt `strands.Agent` instance**: would share `self.messages` and
      other mutable state across cases. Under `--max-workers > 1` this races;
      under `--max-workers == 1` it leaks history from earlier cases.
    - **`strands.Agent` subclass**: works mechanically but forces users to
      subclass purely to satisfy the CLI's calling convention. A factory
      function is the idiomatic Strands shape.

    Accepts:
      - zero-arg callable → `callable_no_arg`
      - one-arg callable (`Case`) → `callable_one_arg`

    **Scope of the freshness guarantee.** The instance/subclass rejections only
    cover `strands.Agent` (the SDK type the CLI was designed around). A
    *non-Strands* callable instance — e.g. `class MyAgent: __call__(self, prompt): ...`
    exposed as a module-level instance — passes the `callable(obj)` check and
    classifies as `callable_one_arg`. The CLI will call it as `obj(case)` once
    per case (treating it as a factory) and then call its return value with
    `case.input`. If that misuse happens to "work" mechanically, the same
    instance is reused across cases and any per-call state it accumulates
    leaks between them. There is no general way to detect this without a
    catalog of every agent framework's base class, so the contract is:
    `--agent` accepts a *factory* that returns a fresh agent per call. Pass a
    function, lambda, `functools.partial`, or class object — not a prebuilt
    instance whose `__call__` carries state.

    Raises:
        EntryPointError: If the object is a prebuilt `strands.Agent` instance,
            a `strands.Agent` subclass, a callable with the wrong arity, or
            not callable at all. The message points users at the factory
            pattern.
    """
    if _is_strands_agent_instance(obj):
        raise EntryPointError(
            f"--agent target '{spec}' is a prebuilt strands.Agent instance, which is unsafe "
            f"because all cases would share its conversation state. {_FACTORY_HINT}"
        )
    if _is_strands_agent_class(obj):
        raise EntryPointError(
            f"--agent target '{spec}' is a strands.Agent subclass; --agent only accepts "
            f"factory callables. {_FACTORY_HINT}"
        )
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
    raise EntryPointError(f"--agent target '{spec}' is not callable (got {type(obj).__name__}). {_FACTORY_HINT}")


def classify_task(obj: Any, spec: str) -> ResolvedEntryPoint:
    """Classify a resolved object passed as `--task`.

    The value must be a callable accepting exactly one positional argument
    (the `Case`). Wrong-arity callables are rejected here so the failure
    surfaces at resolution time with a clear message, instead of crashing
    deep inside `Experiment.run_evaluations_async`.
    """
    if not callable(obj):
        raise EntryPointError(f"--task target '{spec}' is not callable (got {type(obj).__name__})")
    arity = _callable_arity(obj)
    if arity != 1:
        raise EntryPointError(
            f"--task target '{spec}' requires {arity} positional arguments; expected exactly 1 (the Case)"
        )
    return ResolvedEntryPoint(kind="task", obj=obj, spec=spec)


def resolve_agent(spec: str) -> ResolvedEntryPoint:
    """Convenience: `import_attr` then `classify_agent`."""
    return classify_agent(import_attr(spec), spec)


def resolve_task(spec: str) -> ResolvedEntryPoint:
    """Convenience: `import_attr` then `classify_task`."""
    return classify_task(import_attr(spec), spec)


def resolve_evaluator_spec(spec: str) -> Evaluator:
    """Resolve `--evaluator SPEC` into an instantiated `Evaluator`.

    Two forms are accepted:

    - **Built-in shortname** (e.g. `"helpfulness"`, `"equals"`): looked up in
      :data:`BUILTIN_EVALUATOR_SHORTNAMES` and instantiated with no arguments.
      `Contains` is a special case — it requires a `value=` and is not in the
      shortname registry; users who want substring matching pass
      `--expected-output TEXT` instead, which the caller turns into
      `Contains(value=TEXT)`.
    - **`MODULE:CLASS`** (contains `:`): resolved via :func:`import_attr` and
      validated as an :class:`Evaluator` subclass, then instantiated with no
      arguments. This is the escape hatch for custom evaluators in ad-hoc
      mode; richer construction belongs in an experiment file.

    Raises:
        EntryPointError: When the spec is unknown, the resolved object isn't
            an `Evaluator` subclass, or instantiation fails.
    """
    if ":" in spec:
        cls = import_attr(spec)
        if not isinstance(cls, type) or not issubclass(cls, Evaluator):
            raise EntryPointError(
                f"--evaluator '{spec}' must reference a subclass of "
                f"strands_evals.evaluators.Evaluator (got {type(cls).__name__})"
            )
    else:
        cls = BUILTIN_EVALUATOR_SHORTNAMES.get(spec)
        if cls is None:
            choices = ", ".join(sorted(BUILTIN_EVALUATOR_SHORTNAMES))
            raise EntryPointError(
                f"--evaluator '{spec}' is not a known built-in shortname. "
                f"Choose from: {choices}, or pass MODULE:CLASS for a custom evaluator."
            )

    try:
        return cls()
    except TypeError as e:
        raise EntryPointError(
            f"--evaluator '{spec}' could not be instantiated with no arguments: {e}. "
            f"Configure it in an experiment file instead."
        ) from e


def resolve_custom_evaluators(specs: list[str]) -> list[type[Evaluator]]:
    """Resolve `--custom-evaluator MODULE:CLASS` specs to Evaluator subclasses.

    Shared by `run` and `validate` so the validation rule cannot drift between
    the two commands. Raises :class:`EntryPointError` on a non-class value or
    a class that is not a subclass of `Evaluator`.
    """
    classes: list[type[Evaluator]] = []
    for spec in specs:
        obj = import_attr(spec)
        if not isinstance(obj, type) or not issubclass(obj, Evaluator):
            raise EntryPointError(
                f"--custom-evaluator '{spec}' must reference a subclass of "
                f"strands_evals.evaluators.Evaluator (got {type(obj).__name__})"
            )
        classes.append(obj)
    return classes
