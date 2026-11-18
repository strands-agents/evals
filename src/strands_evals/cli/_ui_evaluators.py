"""What the built-in evaluators take, read off their classes for `strands-evals ui`."""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Iterator
from typing import Any

from ..experiment import _builtin_evaluator_types

_SETTING_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


@functools.cache
def _constructor(name: str) -> tuple[list[inspect.Parameter], bool] | None:
    """The settings a built-in evaluator's constructor names, and whether it takes others too.

    None when the signature cannot be read.
    """
    try:
        parameters = inspect.signature(_builtin_evaluator_types()[name]).parameters.values()
    except (TypeError, ValueError):  # a builtin or C-implemented __init__
        return None
    takes_others = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    return [parameter for parameter in parameters if parameter.kind in _SETTING_KINDS], takes_others


def _problems(evaluator: dict) -> Iterator[tuple[str, str]]:
    """Each `(level, message)` there is to say about one evaluator of an experiment."""
    name = evaluator["evaluator_type"]
    if name not in _builtin_evaluator_types():
        yield "warning", "not a built-in evaluator; `run` and `validate` need it passed in as a custom evaluator"
        return
    constructor = _constructor(name)
    if constructor is None:
        return
    named, takes_others = constructor

    # `from_dict` renames this one before constructing, so the file is allowed to say either.
    given = {("model" if key == "model_id" else key) for key in evaluator if key != "evaluator_type"}
    required = {parameter.name for parameter in named if parameter.default is inspect.Parameter.empty}
    for missing in sorted(required - given):
        yield "error", f"needs a '{missing}' setting; {name} cannot be constructed without it"
    if not takes_others:
        for extra in sorted(given - {parameter.name for parameter in named}):
            yield "error", f"'{extra}' is not a setting of {name}"


def check_evaluators(evaluators: Any) -> list[dict]:
    """Say whether each `evaluator_type` would survive `Experiment.from_dict`, without constructing it.

    The notes are advice and block nothing. An unregistered name is only a warning: custom evaluators are passed
    to `from_dict` by the caller.
    """
    if not isinstance(evaluators, list):
        return []
    return [
        {"index": index, "evaluator_type": evaluator["evaluator_type"], "level": level, "message": message}
        for index, evaluator in enumerate(evaluators)
        if isinstance(evaluator, dict) and isinstance(evaluator.get("evaluator_type"), str)
        for level, message in _problems(evaluator)
    ]


@functools.cache
def evaluator_catalog() -> dict[str, dict]:
    """What the page says about each built-in evaluator: its summary and its settings, read off the class.

    Read from the library so that the page describes an evaluator added later without being edited.
    """
    catalog: dict[str, dict] = {}
    for name, target in _builtin_evaluator_types().items():
        named, _ = _constructor(name) or ([], False)
        settings = [
            {
                "name": parameter.name,
                "required": parameter.default is inspect.Parameter.empty,
                "default": None if parameter.default is inspect.Parameter.empty else _shown(parameter.default),
            }
            for parameter in named
        ]
        summary = (inspect.getdoc(target) or "").strip().split("\n", 1)[0]
        catalog[name] = {"summary": summary, "settings": settings}
    return catalog


def _shown(default: Any) -> str:
    """A setting's default as a user would write it in the file, cut short when it is a long prompt."""
    text = json.dumps(default) if isinstance(default, (str, int, float, bool, type(None))) else repr(default)
    return text if len(text) <= 48 else text[:45] + "..."
