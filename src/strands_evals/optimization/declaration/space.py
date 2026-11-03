"""The declared space: flat typed keys, which are active, validation, and resolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from typing_extensions import TypeAlias

from ...types.optimization import Value, Values
from .files import FileText, JsonField, SkillLibrary
from .markers import _UNSET, Choice, Component, Float, Int, OneOf, TextValue, _default_size_cap, _is_option, choice
from .tools import ToolDescriptions

Marker: TypeAlias = (
    Int | Float | OneOf | TextValue | Component | Choice | FileText | JsonField | SkillLibrary | ToolDescriptions
)

ParamKind = Literal["int", "float", "one_of", "choice", "text"]


@dataclass(frozen=True)
class FileSource:
    """Where a file-backed text value lives."""

    path: Path
    pointer: str | None
    as_path: bool


@dataclass(frozen=True)
class ParamInfo:
    """One flat key of the space.

    `active_when` is `(choice key, label)` when the key exists only under one alternative of a
    `choice()`, else None. E.g. `cm=choice(window=component(SlidingWindowConversationManager,
    window_size=Int(4, 40)), summary=component(...))` gives `cm` (always active) and
    `cm.window.window_size` with `active_when=("cm", "window")`. When a proposal switches `cm` to
    `summary`, `Space.complete` drops `cm.window.window_size` (kept, it would change the candidate
    id without changing the agent) and fills the keys `summary` brings in. Validation, TPE, the score
    model's design, and `Nested` read it too. A choice inside an alternative carries its own
    `active_when`, so conditions chain upward. Text keys can't sit under a `choice()`, so theirs is
    always None.

    `min_value` / `max_value` are the inclusive bounds of an `Int` / `Float`. Fields after `default`
    default to None (`log` to False), so each kind sets only its own.
    """

    key: str
    name: str
    kind: ParamKind
    default: Value
    min_value: float | None = None
    max_value: float | None = None
    log: bool = False
    options: tuple[Value, ...] | None = None
    size_cap: int | None = None
    edit_instructions: str | None = None
    active_when: tuple[str, str] | None = None
    source: FileSource | None = None


class Space:
    """Flattens markers into dotted keys and knows which keys are active under which values.

    `values` is a candidate's values: flat key -> typed value, e.g.
    `{"manager": "Window", "manager.Window.size": 40}`.
    """

    def __init__(self, markers: Mapping[str, Marker]):
        """Flatten the markers; components were already probed by `component()`.

        A bare component counts as a one-alternative choice. Every default must pass `validate`, so
        the baseline always does.

        Raises:
            TypeError: No markers, a name that is empty or contains `.` or `/`, an unknown marker
                type, a marker with no default, or a default that fails `validate` (e.g. a text
                default over its size cap).
        """
        if not markers:
            raise TypeError("nothing to optimize: declare at least one marker")
        self._markers: dict[str, Marker] = {}
        self._params: dict[str, ParamInfo] = {}
        for name, marker in markers.items():
            if not isinstance(name, str) or not name or "." in name or "/" in name:
                raise TypeError(f"declared name {name!r} must be a non-empty string without '.' or '/'")
            if isinstance(marker, Component):
                marker = choice(marker)
            if not isinstance(marker, Marker):
                raise TypeError(f"{name}: {type(marker).__name__} is not a marker")
            self._markers[name] = marker
            for info in _flatten(name, marker):
                problem = _check_value(info, info.default)
                if problem is not None:
                    raise TypeError(f"invalid default: {problem}")
                self._params[info.key] = info

    def params(self) -> list[ParamInfo]:
        """Every flat key, in declaration order, a choice before its sub-keys.

        Includes the keys under alternatives that aren't chosen by default.
        """
        return list(self._params.values())

    def param(self, key: str) -> ParamInfo:
        """One flat key's info; an unknown key raises `KeyError` naming the known keys."""
        if key not in self._params:
            raise KeyError(f"{key!r} is not a key of this space; keys: {sorted(self._params)}")
        return self._params[key]

    def is_active(self, key: str, values: Values) -> bool:
        """Whether a key is active.

        Conditions chain: a key under an alternative is active only if every enclosing choice
        selects it.

        Args:
            key: A flat key.
            values: The candidate's values; a missing choice counts as its default.

        Returns:
            Whether `key` is active under the alternatives chosen in `values`.
        """
        info = self.param(key)
        while info.active_when is not None:
            condition_key, label = info.active_when
            condition = self._params[condition_key]
            if values.get(condition_key, condition.default) != label:
                return False
            info = condition
        return True

    def complete(
        self, values: Values, *, fill: Callable[[str], tuple[Value, str] | None] | None = None
    ) -> dict[str, Value]:
        """Drop keys inactive under the chosen alternatives and fill missing active keys.

        Keys are taken in declaration order, so a filled choice decides which of its sub-keys are
        active. Unknown keys are kept, so `validate` reports them instead of a typo vanishing.

        Args:
            values: The candidate's values.
            fill: `fill(key) -> (value, source id)` for a missing active key, or None to use its
                default; a tuple because None is a valid value.
        """
        completed: dict[str, Value] = {}
        for key, info in self._params.items():
            # A choice comes before its sub-keys, so `completed` already holds every condition.
            if not self.is_active(key, completed):
                continue
            if key in values:
                completed[key] = values[key]
                continue
            filled = fill(key) if fill is not None else None
            completed[key] = info.default if filled is None else filled[0]
        completed.update({key: value for key, value in values.items() if key not in self._params})
        return completed

    def validate(self, values: Values) -> None:
        """Check bounds, size caps, formats, active keys, and that nothing is missing.

        Types are strict, so equal candidates hash alike: `Int` takes an int (not a bool), `Float` a
        float (not an int), and `OneOf` compares type and value.

        Raises:
            ValueError: Listing every problem.
        """
        problems: list[str] = []
        unknown = sorted(key for key in values if key not in self._params)
        if unknown:
            problems.append(f"not keys of this space: {unknown}")
        known = [key for key in values if key in self._params]
        inactive = sorted(key for key in known if not self.is_active(key, values))
        if inactive:
            problems.append(f"inactive under the chosen alternatives: {inactive}")
        missing = sorted(key for key in self._params if key not in values and self.is_active(key, values))
        if missing:
            problems.append(f"missing: {missing}")
        for key in known:
            problem = _check_value(self._params[key], values[key])
            if problem is not None:
                problems.append(problem)
        if problems:
            raise ValueError("invalid values: " + "; ".join(problems))

    def resolve(self, name: str, values: Values, workspace: Path | None) -> Any:
        """What the factory receives for one declared name: a fresh component, a value, or a path.

        - A choice: a new instance of the chosen component, e.g. `Summary(ratio=0.5)`, or the chosen
          fixed value as is.
        - `tool_descriptions`: new copies of the tools with the candidate's descriptions.
        - Any other marker: the value itself, e.g. `0.2` or the prompt text.

        `values` must already pass `validate`; nothing is checked here. A component's fixed kwargs
        pass through as is, so every build shares those objects.

        Args:
            name: A declared name (`"manager"`), not a flat key (`"manager.Summary.ratio"`).
            values: The candidate's values.
            workspace: The candidate's copy, or None when no value reaches the agent as a path.
        """
        if name not in self._markers:
            raise KeyError(f"{name!r} is not a declared name; declared: {list(self._markers)}")
        marker = self._markers[name]
        if isinstance(marker, Choice):
            label = cast(str, values[name])
            alternative = marker.alternatives[label]
            if not isinstance(alternative, Component):
                return alternative
            return alternative.build({arg: values[f"{name}.{label}.{arg}"] for arg in alternative.tunable})
        if isinstance(marker, ToolDescriptions):
            return marker.build({tool: cast(str, values[f"{name}.{tool}"]) for tool in marker.tools})
        return values[name]

    def create_candidate_copy(self, values: Values, directory: Path) -> Path:
        """Copy each file-backed value's folder into `directory`, then write every active file-backed key.

        Each source folder goes to `directory/<folder name>-<short hash of its original path>/`; a folder
        inside another is copied once, as part of the outer one. Each active key is written by its
        marker's `write()`. Built in a temp folder and renamed into place, so an existing `directory`
        is always complete and a second call returns at once. Originals are never written.

        Args:
            values: The candidate's values.
            directory: The candidate's copy, `<run>/candidates/<id>/optimization_files/`.

        Returns:
            `directory`.
        """
        raise NotImplementedError


# ---- Private helpers: markers -> ParamInfo, and the per-value check ---------


_NO_DEFAULT = "pass default=, or under tunable give the wrapped parameter a default"


def _flatten(name: str, marker: Marker) -> list[ParamInfo]:
    """The typed keys one declared marker expands into, a choice before its sub-keys."""
    if isinstance(marker, (Int, Float)):
        return [_number(name, name, marker, None)]
    if isinstance(marker, OneOf):
        return [_one_of(name, name, marker, None)]
    if isinstance(marker, TextValue):
        if marker.default is None:
            raise TypeError(f"{name}: TextValue has no default; {_NO_DEFAULT}")
        return [_text(name, name, marker.default, marker.size_cap, marker.edit_instructions)]
    if isinstance(marker, Choice):
        labels = tuple(marker.alternatives)
        infos = [ParamInfo(name, name, "choice", labels[0], options=labels)]
        for label, alternative in marker.alternatives.items():
            if isinstance(alternative, Component):
                for arg, tunable in alternative.tunable.items():
                    key, condition = f"{name}.{label}.{arg}", (name, label)
                    if isinstance(tunable, OneOf):
                        infos.append(_one_of(key, name, tunable, condition))
                    else:
                        infos.append(_number(key, name, tunable, condition))
        return infos
    if isinstance(marker, ToolDescriptions):
        return [
            _text(f"{name}.{tool_name}", name, tool.tool_spec["description"], marker.size_cap, marker.edit_instructions)
            for tool_name, tool in marker.tools.items()
        ]
    raise TypeError(f"{name}: {type(marker).__name__} can't be declared yet")


def _number(key: str, name: str, marker: Int | Float, active_when: tuple[str, str] | None) -> ParamInfo:
    if marker.default is None:
        raise TypeError(f"{key}: {type(marker).__name__} has no default; {_NO_DEFAULT}")
    kind: ParamKind = "int" if isinstance(marker, Int) else "float"
    return ParamInfo(
        key,
        name,
        kind,
        marker.default,
        min_value=marker.min_value,
        max_value=marker.max_value,
        log=marker.log,
        active_when=active_when,
    )


def _one_of(key: str, name: str, marker: OneOf, active_when: tuple[str, str] | None) -> ParamInfo:
    default = marker.options[0] if marker.default is _UNSET else marker.default
    return ParamInfo(key, name, "one_of", default, options=tuple(marker.options), active_when=active_when)


def _text(key: str, name: str, default: str, size_cap: int | None, edit_instructions: str | None) -> ParamInfo:
    cap = _default_size_cap(default) if size_cap is None else size_cap
    return ParamInfo(key, name, "text", default, size_cap=cap, edit_instructions=edit_instructions)


def _check_value(info: ParamInfo, value: Any) -> str | None:
    """A one-line problem with one value, or None; text values are described by length only."""
    key = info.key
    if info.kind in ("int", "float"):
        expected = int if info.kind == "int" else float
        if type(value) is not expected:
            return f"{key}: expected {expected.__name__}, got {value!r}"
        if not (
            info.min_value is not None and info.max_value is not None and info.min_value <= value <= info.max_value
        ):
            return f"{key}: {value!r} is outside [{info.min_value!r}, {info.max_value!r}]"
        return None
    if info.kind in ("one_of", "choice"):
        if not _is_option(value, info.options or ()):
            return f"{key}: {value!r} is not one of {list(info.options or ())!r}"
        return None
    if not isinstance(value, str):
        return f"{key}: expected str, got {type(value).__name__}"
    if info.size_cap is not None and len(value) > info.size_cap:
        return f"{key}: {len(value)} characters, over the size_cap of {info.size_cap}"
    return None
