"""The CLI space file: planned markers written as JSON, flattened into `ParamInfo` keys."""

import difflib
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...optimization.declaration.space import FileSource, ParamInfo, ParamKind
from ...types.optimization import Value

SUPPORTED_MARKERS = ("FileText", "TextValue", "Int", "Float", "OneOf")
PLANNED_MARKERS = ("JsonField", "SkillLibrary", "choice", "component", "ToolDescriptions")
MIN_SIZE_CAP = 2000
SIZE_CAP_FACTOR = 3
CONTENT_ID_LENGTH = 12

_FIELDS = {
    "FileText": {"path", "size_cap", "edit_instructions"},
    "TextValue": {"default", "size_cap", "edit_instructions"},
    "Int": {"min_value", "max_value", "default", "log"},
    "Float": {"min_value", "max_value", "default", "log"},
    "OneOf": {"options", "default"},
}
_NUMBER_KINDS: dict[str, ParamKind] = {"Int": "int", "Float": "float"}


def default_size_cap(baseline: str) -> int:
    """The planned text size cap: `max(2000, 3 * len(baseline))`."""
    return max(MIN_SIZE_CAP, SIZE_CAP_FACTOR * len(baseline))


def content_id(values: Mapping[str, Value]) -> str:
    """First 12 hex characters of the sha256 of the canonical JSON of `values`."""
    canonical = json.dumps(dict(values), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:CONTENT_ID_LENGTH]


@dataclass(frozen=True)
class CliSpace:
    """The declared keys of a project, in declaration order; `root` is the project root."""

    params: tuple[ParamInfo, ...]
    root: Path

    @property
    def text_keys(self) -> tuple[str, ...]:
        """Keys a text proposer may change."""
        return tuple(param.key for param in self.params if param.kind == "text")

    @property
    def value_keys(self) -> tuple[str, ...]:
        """Number and choice keys."""
        return tuple(param.key for param in self.params if param.kind != "text")

    def param(self, key: str) -> ParamInfo:
        """One key's info; an unknown key raises `KeyError` naming the known keys."""
        for param in self.params:
            if param.key == key:
                return param
        raise KeyError(f"unknown key {key!r}; known keys: {', '.join(p.key for p in self.params)}")

    def baseline(self) -> dict[str, Value]:
        """Every key at its default."""
        return {param.key: param.default for param in self.params}

    def validate(self, values: Mapping[str, Value]) -> None:
        """Check every key, raising one `ValueError` that lists every problem."""
        known = {param.key for param in self.params}
        problems = [f"{key}: unknown key" for key in values if key not in known]
        problems += [f"{key}: missing" for key in known if key not in values]
        problems += [
            f"{param.key}: {problem}"
            for param in self.params
            if param.key in values
            for problem in [_value_problem(param, values[param.key])]
            if problem
        ]
        if problems:
            raise ValueError("invalid candidate values: " + "; ".join(problems))

    def diff(self, base: Mapping[str, Value], other: Mapping[str, Value]) -> str:
        """Unified diff of every changed text: files on their project paths, then other texts by key."""
        hunks = [
            _unified(str(base[param.key]), str(other[param.key]), *self._labels(param))
            for param in sorted(self.params, key=lambda param: param.source is None)
            if param.kind == "text" and base.get(param.key) != other.get(param.key)
        ]
        return "".join(hunks)

    def changed_values(self, base: Mapping[str, Value], other: Mapping[str, Value]) -> dict[str, dict[str, Value]]:
        """Changed numbers and choices as `{key: {"old": ..., "new": ...}}`."""
        return {
            key: {"old": base.get(key), "new": other.get(key)}
            for key in self.value_keys
            if base.get(key) != other.get(key)
        }

    def config(self) -> list[dict[str, Any]]:
        """Every `ParamInfo` field, saved in `run.json` and compared on resume."""
        return [_param_config(param, self.root) for param in self.params]

    def _labels(self, param: ParamInfo) -> tuple[str, str]:
        if param.source is None:
            return param.key, param.key
        relative = param.source.path.relative_to(self.root).as_posix()
        return f"a/{relative}", f"b/{relative}"


def load_space(path: Path, root: Path) -> CliSpace:
    """Read a space file: `{name: {"marker": "FileText", ...}}`; file paths resolve against `root`.

    Raises:
        ValueError: The file is not a JSON object of markers, or a marker is unknown, planned only, or invalid.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read space file {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError(f"space file must be a nonempty JSON object of markers: {path}")
    return CliSpace(params=tuple(_param(str(name), spec, root) for name, spec in data.items()), root=root)


def instructions_space(path: Path, root: Path) -> CliSpace:
    """The `--instructions-file` shortcut: `{"instructions": FileText(path)}`."""
    return CliSpace(params=(_param("instructions", {"marker": "FileText", "path": str(path)}, root),), root=root)


def _param(name: str, spec: Any, root: Path) -> ParamInfo:
    if not name or "." in name or "/" in name:
        raise ValueError(f"key {name!r}: names must be nonempty and contain no '.' or '/'")
    if not isinstance(spec, dict):
        raise ValueError(f"key {name!r}: a marker must be a JSON object with a 'marker' field")
    marker = spec.get("marker")
    if marker in PLANNED_MARKERS:
        raise ValueError(
            f"key {name!r}: {marker} is planned but not supported by the CLI yet; use one of "
            + ", ".join(SUPPORTED_MARKERS)
        )
    if marker not in _FIELDS:
        raise ValueError(f"key {name!r}: unknown marker {marker!r}; use one of " + ", ".join(SUPPORTED_MARKERS))
    unknown = set(spec) - _FIELDS[marker] - {"marker"}
    if unknown:
        raise ValueError(f"key {name!r}: unknown {marker} fields: {', '.join(sorted(unknown))}")
    if marker == "FileText":
        return _file_text(name, spec, root)
    if marker == "TextValue":
        return _text_value(name, spec)
    if marker == "OneOf":
        return _one_of(name, spec)
    return _number(name, spec, _NUMBER_KINDS[marker])


def _file_text(name: str, spec: dict[str, Any], root: Path) -> ParamInfo:
    raw = spec.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"key {name!r}: FileText needs a 'path'")
    path = Path(raw).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"key {name!r}: FileText path must be an existing file inside the project: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"key {name!r}: FileText file is empty: {path}")
    return _text_param(name, text, spec, FileSource(path=path, pointer=None, as_path=False))


def _text_value(name: str, spec: dict[str, Any]) -> ParamInfo:
    default = spec.get("default")
    if not isinstance(default, str) or not default.strip():
        raise ValueError(f"key {name!r}: TextValue needs a nonempty string 'default'")
    return _text_param(name, default, spec, None)


def _text_param(name: str, text: str, spec: dict[str, Any], source: FileSource | None) -> ParamInfo:
    size_cap = spec.get("size_cap", default_size_cap(text))
    if isinstance(size_cap, bool) or not isinstance(size_cap, int) or size_cap < len(text):
        raise ValueError(f"key {name!r}: size_cap must be an integer >= the baseline length ({len(text)})")
    edit_instructions = spec.get("edit_instructions")
    if edit_instructions is not None and not isinstance(edit_instructions, str):
        raise ValueError(f"key {name!r}: edit_instructions must be a string")
    return ParamInfo(
        key=name,
        name=name,
        kind="text",
        default=text,
        size_cap=size_cap,
        edit_instructions=edit_instructions,
        source=source,
    )


def _one_of(name: str, spec: dict[str, Any]) -> ParamInfo:
    options = spec.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError(f"key {name!r}: OneOf needs at least two 'options'")
    if any(isinstance(option, (dict, list)) for option in options):
        raise ValueError(f"key {name!r}: OneOf options must be plain values")
    default = spec.get("default", options[0])
    if not any(_same(default, option) for option in options):
        raise ValueError(f"key {name!r}: OneOf default {default!r} is not one of the options")
    return ParamInfo(key=name, name=name, kind="one_of", default=default, options=tuple(options))


def _number(name: str, spec: dict[str, Any], kind: ParamKind) -> ParamInfo:
    if any(not _is_number(spec.get(field), kind) for field in ("min_value", "max_value", "default")):
        raise ValueError(f"key {name!r}: {kind} markers need {kind} 'min_value', 'max_value', and 'default'")
    low, high, default = spec["min_value"], spec["max_value"], spec["default"]
    log = spec.get("log", False)
    if not isinstance(log, bool) or (log and low <= 0):
        raise ValueError(f"key {name!r}: 'log' must be a boolean, and a log scale needs min_value > 0")
    if not low <= default <= high or low == high:
        raise ValueError(f"key {name!r}: needs min_value < max_value and a default between them")
    return ParamInfo(key=name, name=name, kind=kind, default=default, min_value=low, max_value=high, log=log)


def _is_number(value: Any, kind: str) -> bool:
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, float) and math.isfinite(value)


def _same(left: Value, right: Value) -> bool:
    return type(left) is type(right) and left == right


def _value_problem(param: ParamInfo, value: Value) -> str | None:
    if param.kind == "text":
        if not isinstance(value, str) or not value.strip():
            return "must be nonempty text"
        too_long = len(value) > (param.size_cap or 0)
        return f"text over its size cap ({len(value)} > {param.size_cap})" if too_long else None
    if param.kind == "one_of":
        return None if any(_same(value, option) for option in param.options or ()) else "not one of the options"
    if not isinstance(value, (int, float)) or not _is_number(value, param.kind):
        return f"must be a {param.kind}"
    low, high = param.min_value, param.max_value
    return None if low is not None and high is not None and low <= value <= high else "outside its bounds"


def _unified(before: str, after: str, old_label: str, new_label: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=old_label, tofile=new_label
    )
    return "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in lines)


def _param_config(param: ParamInfo, root: Path) -> dict[str, Any]:
    source = param.source
    return {
        "key": param.key,
        "kind": param.kind,
        "default": param.default,
        "min_value": param.min_value,
        "max_value": param.max_value,
        "log": param.log,
        "options": list(param.options) if param.options is not None else None,
        "size_cap": param.size_cap,
        "edit_instructions": param.edit_instructions,
        "source": None if source is None else source.path.relative_to(root).as_posix(),
    }
