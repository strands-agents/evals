"""What the built-in evaluators take, read off their classes for `strands-evals ui`."""

from __future__ import annotations

import functools
import inspect
import json
from typing import Any

from ..experiment import _builtin_evaluator_types


def check_evaluators(evaluators: Any) -> list[dict]:
    """Say whether each `evaluator_type` would survive `Experiment.from_dict`, without running it.

    Uses `inspect.signature` on the built-in registry, never construction. The notes are advice and block nothing.
    An unregistered name is only a warning: custom evaluators are passed to `from_dict` by the caller.
    """
    if not isinstance(evaluators, list):
        return []

    registry = _builtin_evaluator_types()
    notes: list[dict] = []
    for index, evaluator in enumerate(evaluators):
        if not isinstance(evaluator, dict):
            continue
        name = evaluator.get("evaluator_type")
        if not isinstance(name, str):
            continue

        def note(level: str, message: str, index: int = index, name: str = name) -> None:
            notes.append({"index": index, "evaluator_type": name, "level": level, "message": message})

        target = registry.get(name)
        if target is None:
            note("warning", "not a built-in evaluator; `run` and `validate` need it passed in as a custom evaluator")
            continue

        # `from_dict` renames this one before constructing, so the file is allowed to say either.
        settings = {("model" if key == "model_id" else key) for key in evaluator if key != "evaluator_type"}
        try:
            parameters = inspect.signature(target).parameters
        except (TypeError, ValueError):  # a builtin or C-implemented __init__
            continue

        accepts_anything = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        named = {
            key
            for key, parameter in parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        required = {key for key in named if parameters[key].default is inspect.Parameter.empty}

        for missing in sorted(required - settings):
            note("error", f"needs a '{missing}' setting; {name} cannot be constructed without it")
        if not accepts_anything:
            for extra in sorted(settings - named):
                note("error", f"'{extra}' is not a setting of {name}")

    return notes


@functools.cache
def evaluator_catalog() -> dict[str, dict]:
    """What the page says about each built-in evaluator: its summary and its settings, read off the class.

    Read from the library so that the page describes an evaluator added later without being edited.
    """
    catalog: dict[str, dict] = {}
    for name, target in _builtin_evaluator_types().items():
        try:
            parameters = list(inspect.signature(target).parameters.values())
        except (TypeError, ValueError):  # a builtin or C-implemented __init__
            parameters = []
        settings = [
            {
                "name": parameter.name,
                "required": parameter.default is inspect.Parameter.empty,
                "default": None if parameter.default is inspect.Parameter.empty else _shown(parameter.default),
            }
            for parameter in parameters
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        summary = (inspect.getdoc(target) or "").strip().split("\n", 1)[0]
        catalog[name] = {"summary": summary, "settings": settings}
    return catalog


def _shown(default: Any) -> str:
    """A setting's default as a user would write it in the file, cut short when it is a long prompt."""
    text = json.dumps(default) if isinstance(default, (str, int, float, bool, type(None))) else repr(default)
    return text if len(text) <= 48 else text[:45] + "..."
