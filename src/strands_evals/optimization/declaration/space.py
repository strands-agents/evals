"""The declared space: flat typed keys, which are active, validation, and resolution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from typing_extensions import TypeAlias

from ...types.optimization import Value, Values
from .files import (
    FileText,
    JsonField,
    SkillLibrary,
    _folder_in_copy,
    _ignore_containing,
    _location_in_copy,
    _outer_folders,
)
from .markers import _UNSET, Choice, Component, Float, Int, OneOf, TextValue, _default_size_cap, _is_option, choice
from .tools import ToolDescriptions

Marker: TypeAlias = (
    Int | Float | OneOf | TextValue | Component | Choice | FileText | JsonField | SkillLibrary | ToolDescriptions
)

ParamKind = Literal["int", "float", "one_of", "choice", "text", "skills"]


@dataclass(frozen=True)
class FileSource:
    """Where a file-backed value lives: a `FileText`'s or `JsonField`'s file, or a `SkillLibrary`'s folder."""

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
    `summary`, `Space.complete` drops `cm.window.window_size` (if it were kept, it would change the
    candidate id without changing the agent) and fills the keys `summary` brings in. Validation,
    TPE, the score model's design, and `Nested` read it too. Conditions are one level deep, since
    `component()` takes no nested markers. Text and skills keys can't sit under a `choice()`, so theirs
    is always None.

    `min_value` / `max_value` are the inclusive bounds of an `Int` / `Float`. Fields after `default`
    default to None (`log` to False), so each kind sets only its own. A `"skills"` key (a `SkillLibrary`)
    holds a library version, `"skills/<hash>"`; its `size_cap` and `edit_instructions` apply to each
    `SKILL.md` in it (`size_cap` None: each one's default cap).
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
                type, a marker with no default, a default that fails `validate` (e.g. a text
                default over its size cap), or two file markers on the same text (a `FileText`'s
                file or a tuned skill's `SKILL.md` declared again, two `JsonField`s on one field or
                one inside the other, or a `SkillLibrary` inside another).
        """
        if not markers:
            raise TypeError("nothing to optimize: declare at least one marker")
        self._markers: dict[str, Marker] = {}
        self._params: dict[str, ParamInfo] = {}
        self._files: dict[str, FileText | JsonField | SkillLibrary] = {}  # file-backed key -> its marker
        self._owned: dict[str, dict[str, str]] = {}  # filled below, once every file marker is known
        for name, marker in markers.items():
            if not isinstance(name, str) or not name or "." in name or "/" in name:
                raise TypeError(f"declared name {name!r} must be a non-empty string without '.' or '/'")
            if isinstance(marker, Component):
                marker = choice(marker)
            if not isinstance(marker, Marker):
                raise TypeError(f"{name}: {type(marker).__name__} is not a marker")
            self._markers[name] = marker
            files = {name: marker} if isinstance(marker, (FileText, JsonField, SkillLibrary)) else {}
            for info in _flatten(name, marker):
                if info.key in files:
                    _check_no_overlap(info.key, files[info.key], self._files)
                    self._files[info.key] = files[info.key]
                problem = self._problem(info, info.default)
                if problem is not None:
                    raise TypeError(f"invalid default: {problem}")
                self._params[info.key] = info
        # Folders a candidate's copy holds (a SkillLibrary's whole directory); fixed by the declaration,
        # so every copy has one layout.
        self._copy_folders = _outer_folders(
            m.path if isinstance(m, SkillLibrary) else m.path.parent for m in self._files.values()
        )
        self._copies_files = any(
            isinstance(m, SkillLibrary) or (isinstance(m, FileText) and m.as_path) for m in self._files.values()
        )
        # A library's key -> each of its files another file marker declares (path in the library -> that key);
        # every version keeps those as declared, so the key's value alone says what the copy holds.
        self._owned = {
            key: {
                other.path.relative_to(library.path).as_posix(): other_key
                for other_key, other in self._files.items()
                if not isinstance(other, SkillLibrary) and other.path.is_relative_to(library.path)
            }
            for key, library in self._files.items()
            if isinstance(library, SkillLibrary)
        }

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

    def marker(self, name: str) -> Marker:
        """The marker declared under `name`.

        A proposer reaches a `SkillLibrary` this way, e.g. `request.space.marker("skills").revise(...)`.

        Args:
            name: A declared name, not a flat key.

        Returns:
            The marker; a bare `component()` comes back as the one-option `choice()` it counts as.

        Raises:
            KeyError: `name` is not a declared name.
        """
        if name not in self._markers:
            raise KeyError(f"{name!r} is not a declared name; declared: {list(self._markers)}")
        return self._markers[name]

    def is_active(self, key: str, values: Values) -> bool:
        """Whether a key is active.

        A key under an alternative is active only if its choice selects that alternative.

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
        active. Unknown keys are kept, so `validate` reports them instead of a typo vanishing. A
        `Float` value of -0.0 becomes 0.0, and a non-string `JsonField` value is rewritten as its JSON
        text is written to the file ("0.50" becomes "0.5"), so one agent gets one candidate id.

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
                value = values[key]
                if key in self._files:
                    value = self._files[key]._canonical(value)
            else:
                filled = fill(key) if fill is not None else None
                value = info.default if filled is None else filled[0]
            # `+ 0.0` turns a Float's -0.0 into 0.0, whatever its source, so both get the same candidate id.
            completed[key] = value + 0.0 if info.kind == "float" and type(value) is float else value
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
            problem = self._problem(self._params[key], values[key])
            if problem is not None:
                problems.append(problem)
        if problems:
            raise ValueError("invalid values: " + "; ".join(problems))

    def resolve(self, name: str, values: Values, workspace: Path | None) -> Any:
        """What the factory receives for one declared name: a fresh component, a value, or a path.

        - A choice: a new instance of the chosen component, e.g. `Summary(ratio=0.5)`, or the chosen
          fixed value as is.
        - `tool_descriptions`: new copies of the tools with the candidate's descriptions.
        - `FileText(as_path=True)`: the path of the file inside `workspace`, or the original path
          when `workspace` is None (the plain call).
        - `SkillLibrary`: the path of the directory inside `workspace`, which holds the candidate's
          version, or the original directory when `workspace` is None.
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
        if isinstance(marker, SkillLibrary) or (isinstance(marker, FileText) and marker.as_path):
            return marker.path if workspace is None else workspace / _location_in_copy(self._copy_folders, marker.path)
        return values[name]

    def create_candidate_copy(self, values: Values, directory: Path) -> Path:
        """Copy each file-backed value's folder into `directory`, then write every active file-backed key.

        Each source folder goes to `directory/<short hash of its original path>/<folder name>/`, so it
        keeps its name; a folder inside another is copied once, as part of the outer one. A
        `SkillLibrary`'s folder is never copied: its place holds the candidate's version, written from
        memory. Then each other active key is written by its marker's `write()`, so the copy holds the
        candidate's values even if a source file changed
        since declaration; a file that already holds its value keeps the original bytes. Built in a
        temp folder and renamed into place, with a marker file holding the sha256 of its file-backed values,
        so a second call with the same values returns at once and one with other values raises. An
        empty `directory` is replaced; concurrent calls for the same values all succeed. Originals are
        never written: files are replaced rather than written through, and a folder containing
        `directory` (e.g. the run directory under the project) is left out of the copy. A symlink
        leads where the original's leads, into the copy when that is copied too.

        Args:
            values: The candidate's values; must already pass `validate`.
            directory: The candidate's copy, `<run>/candidates/<id>/optimization_files/`.

        Returns:
            `directory`.

        Raises:
            FileExistsError: `directory` holds another candidate's copy, or files that aren't a copy.
        """
        key = self._copy_key(values)
        if self._has_copy(directory, key):
            return directory
        directory.parent.mkdir(parents=True, exist_ok=True)
        building = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
        try:
            destinations = (building.resolve(), directory.resolve())
            libraries = {k: m for k, m in self._files.items() if isinstance(m, SkillLibrary)}
            library_folders = tuple(library.path for library in libraries.values())
            for folder in self._copy_folders:
                if folder in library_folders:
                    continue
                shutil.copytree(
                    folder,
                    building / _folder_in_copy(folder),
                    symlinks=True,
                    ignore=_ignore_containing(destinations, library_folders),
                )
            _make_folders_writable(building)  # a read-only source folder would block the writes below
            for folder in self._copy_folders:
                if folder not in library_folders:
                    _relink(building, folder, self._copy_folders)
            for name, library in libraries.items():  # first: a file marker inside a library writes over its file
                if self.is_active(name, values):
                    library._write(str(values[name]), building / _location_in_copy(self._copy_folders, library.path))
            for info in self._params.values():
                if info.source is None or info.key in libraries or not self.is_active(info.key, values):
                    continue
                target = building / _location_in_copy(self._copy_folders, info.source.path)
                marker = self._files[info.key]
                assert not isinstance(marker, SkillLibrary)
                marker.write(str(values[info.key]), target)
            (building / _COPY_MARKER).write_bytes(key.encode("ascii"))  # last: only a finished copy has it
        except BaseException:
            shutil.rmtree(building, ignore_errors=True)
            raise
        try:
            os.rename(building, directory)
        except OSError:
            shutil.rmtree(building, ignore_errors=True)
            # Another call finished first: fine if it holds these values, else this raises.
            if not self._has_copy(directory, key):
                raise
        return directory

    def _copy_key(self, values: Values) -> str:
        """sha256 of the canonical JSON of the active file-backed values: what a copy holds."""
        held = {k: values[k] for k in self._files if k in values and self.is_active(k, values)}
        canonical = json.dumps(held, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _has_copy(self, directory: Path, key: str) -> bool:
        """True when `directory` already holds the copy for `key`; False when the caller should build it.

        Removes an empty `directory`. A copy only arrives by rename with its marker inside, so a folder
        that isn't empty either holds a copy or holds files that aren't one. Raises `FileExistsError`
        when it holds another candidate's copy or files that aren't a copy.
        """
        held = _read_marker(directory)
        if held is None:
            try:
                directory.rmdir()
                return False
            except FileNotFoundError:
                return False  # missing, or another call removed the empty folder
            except OSError:
                if not directory.is_dir():
                    raise FileExistsError(f"{directory} exists and is not a folder") from None
                held = _read_marker(directory)  # not empty: another call's copy may have just landed
                if held is None:
                    raise FileExistsError(
                        f"{directory} exists and is not a candidate copy; pass an empty or new folder"
                    ) from None
        if held != key:
            raise FileExistsError(f"{directory} holds another candidate's copy; use one folder per candidate")
        return True

    def _render_files(self, values: Values) -> dict[Path, str]:
        """Each file-backed file's whole text under `values`, in memory: every UTF-8 file of each library
        version, then the other keys in declaration order.

        Every key on a file is rendered, changed or not, so two candidates' texts of one JSON file are
        formatted alike and differ only where their values do. A library's binary files are left out.
        """
        texts: dict[Path, str] = {}
        for key, library in self._files.items():
            if isinstance(library, SkillLibrary) and self.is_active(key, values):
                for path, data in library.files(str(values[key])).items():
                    try:
                        texts[library.path / path] = data.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
        for info in self._params.values():
            marker = self._files.get(info.key)
            if marker is None or isinstance(marker, SkillLibrary) or not self.is_active(info.key, values):
                continue
            assert info.source is not None
            text = texts.get(info.source.path, marker._original())
            texts[info.source.path] = marker._render(str(values[info.key]), text)
        return texts

    def _save_versions(self, values: Values, artifacts: Path) -> None:
        """Write each library version in `values` once into `artifacts/skills/<hash>/`, where `_load_versions`
        reads it back."""
        for key, library in self._files.items():
            if isinstance(library, SkillLibrary) and key in values:
                library._save(str(values[key]), artifacts)

    def _load_versions(self, values: Values, artifacts: Path) -> None:
        """Store each library version in `values` that this process lacks from `artifacts/skills/<hash>/`.

        Raises:
            ValueError: A value isn't a version's value, its folder is missing, doesn't hold the version it is
                named for (it changed since it was written), or holds one that fails its library's checks.
        """
        for key, library in self._files.items():
            if not isinstance(library, SkillLibrary) or key not in values:
                continue
            value = str(values[key])
            if value in library._versions:
                continue
            try:
                library._load(value, artifacts)
            except ValueError as error:
                raise ValueError(f"{key}: {error}") from error

    def _problem(self, info: ParamInfo, value: Any) -> str | None:
        """A one-line problem with one value, including a file marker's format check, or None.

        A file-backed value is checked as `complete` stores it, so a value that passes here also
        passes once completed (e.g. compact JSON that the stored 2-space indent puts over `size_cap`).
        """
        if info.key in self._files:
            value = self._files[info.key]._canonical(value)
        problem = _check_value(info, value)
        if problem is None and info.key in self._files:
            format_problem = self._files[info.key]._format_problem(value)
            if format_problem is not None:
                problem = f"{info.key}: {format_problem}"
        library = self._files.get(info.key)
        if problem is None and isinstance(library, SkillLibrary):
            files, declared = library._version(value).files, library._version(library.default).files
            for path, owner in self._owned.get(info.key, {}).items():
                if files.get(path) != declared.get(path):
                    return f"{info.key}: {path} is declared by {owner}, so every version keeps it as declared"
        return problem


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
        infos = []
        for tool_name, tool in marker.tools.items():
            description = tool.tool_spec["description"]
            cap = max(800, 3 * len(description)) if marker.size_cap is None else marker.size_cap
            infos.append(_text(f"{name}.{tool_name}", name, description, cap, marker.edit_instructions))
        return infos
    if isinstance(marker, FileText):
        source = FileSource(marker.path, None, marker.as_path)
        return [_text(name, name, marker.default, marker.size_cap, marker.edit_instructions, source)]
    if isinstance(marker, JsonField):
        source = FileSource(marker.path, marker.pointer, False)
        return [_text(name, name, marker.default, marker.size_cap, marker.edit_instructions, source)]
    if isinstance(marker, SkillLibrary):
        source = FileSource(marker.path, None, True)
        return [
            ParamInfo(
                name,
                name,
                "skills",
                marker.default,
                size_cap=marker.size_cap,
                edit_instructions=marker.edit_instructions,
                source=source,
            )
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


def _text(
    key: str,
    name: str,
    default: str,
    size_cap: int | None,
    edit_instructions: str | None,
    source: FileSource | None = None,
) -> ParamInfo:
    cap = _default_size_cap(default) if size_cap is None else size_cap
    return ParamInfo(key, name, "text", default, size_cap=cap, edit_instructions=edit_instructions, source=source)


def _check_no_overlap(
    key: str,
    marker: FileText | JsonField | SkillLibrary,
    declared: Mapping[str, FileText | JsonField | SkillLibrary],
) -> None:
    """Raise `TypeError` if `marker` edits text another declared file marker edits too.

    A `SkillLibrary` overlaps another one it is inside or holds, and a file marker on one of its tuned
    `SKILL.md` files (Reflection edits those); a file marker on another of its files is allowed.
    """
    # Every marker's path is already resolved (FileText, JsonField and SkillLibrary resolve theirs), but a
    # case-insensitive file system keeps the casing it was given, so paths that differ are compared as files too.
    for other_key, other in declared.items():
        if isinstance(marker, SkillLibrary) or isinstance(other, SkillLibrary):
            _check_library_overlap(key, marker, other_key, other)
            continue
        if other.path != marker.path and not os.path.samefile(other.path, marker.path):
            continue
        if isinstance(marker, JsonField) and isinstance(other, JsonField):
            shorter, longer = sorted((marker._tokens, other._tokens), key=len)
            if longer[: len(shorter)] != shorter:
                continue
            raise TypeError(
                f"{key}: {marker.pointer!r} and {other_key}'s {other.pointer!r} are the same field of {marker.path} "
                "or one is inside the other"
            )
        raise TypeError(f"{key}: {marker.path} is already declared by {other_key}; a whole file's text can't be shared")


def _check_library_overlap(
    key: str,
    marker: FileText | JsonField | SkillLibrary,
    other_key: str,
    other: FileText | JsonField | SkillLibrary,
) -> None:
    """`_check_no_overlap` for a pair where at least one is a `SkillLibrary`."""
    if isinstance(marker, SkillLibrary) and isinstance(other, SkillLibrary):
        if marker.path.is_relative_to(other.path) or other.path.is_relative_to(marker.path):
            raise TypeError(f"{key}: {marker.path} is already declared by {other_key}, or holds it")
        return
    library, file = (marker, other) if isinstance(marker, SkillLibrary) else (other, marker)
    assert isinstance(library, SkillLibrary) and not isinstance(file, SkillLibrary)
    for skill_file in library._texts(library.default):
        path = library.path / skill_file
        if file.path == path or (path.exists() and os.path.samefile(file.path, path)):
            raise TypeError(f"{key}: {path} is already declared by {other_key}; a whole file's text can't be shared")


def _check_value(info: ParamInfo, value: Any) -> str | None:
    """A one-line problem with one value, or None; text values are described by length only."""
    key = info.key
    if info.kind in ("int", "float"):
        expected = int if info.kind == "int" else float
        if type(value) is not expected:
            return f"{key}: expected {expected.__name__}, got {value!r}"
        if not cast(float, info.min_value) <= value <= cast(float, info.max_value):  # numbers always have bounds
            return f"{key}: {value!r} is outside [{info.min_value!r}, {info.max_value!r}]"
        return None
    if info.kind in ("one_of", "choice"):
        if not _is_option(value, info.options or ()):
            return f"{key}: {value!r} is not one of {list(info.options or ())!r}"
        return None
    if info.kind == "skills":
        return None if isinstance(value, str) else f"{key}: expected a library version (str), got {value!r}"
    if not isinstance(value, str):
        return f"{key}: expected str, got {type(value).__name__}"
    if info.size_cap is not None and len(value) > info.size_cap:
        return f"{key}: {len(value)} characters, over the size_cap of {info.size_cap}"
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate: no candidate id or file could hold it
        return f"{key}: not valid Unicode text (contains a lone surrogate)"
    return None


# ---- Private helpers: the candidate copy -------------------------------------

# Written last into a finished copy: the sha256 of the file-backed values it holds.
_COPY_MARKER = ".candidate-copy"


def _read_marker(directory: Path) -> str | None:
    """The values key a finished copy at `directory` holds, or None when there is no finished copy."""
    try:
        return (directory / _COPY_MARKER).read_bytes().decode("ascii")
    except (FileNotFoundError, NotADirectoryError):
        return None


def _relink(building: Path, folder: Path, folders: tuple[Path, ...]) -> None:
    """Point each symlink in `folder`'s copy where the original link leads.

    `copytree(symlinks=True)` copies a link's text, so a relative link out of the folder would lead
    nowhere in the copy, and the candidate would lose a skill or file the baseline reads through it.
    A link leading into a copied folder points at the copy (relative, so the copy can be renamed into
    place), else at the original's real path.
    """
    copy = building / _folder_in_copy(folder)
    for here, dirs, files in os.walk(copy):  # never descends into a symlinked folder
        for name in dirs + files:
            link = Path(here, name)
            if not link.is_symlink():
                continue
            real = Path(os.path.realpath(folder / link.relative_to(copy)))
            try:
                target = os.path.relpath(building / _location_in_copy(folders, real), link.parent)
            except KeyError:  # outside every copied folder
                target = str(real)
            if os.readlink(link) != target:
                link.unlink()
                os.symlink(target, link, target_is_directory=real.is_dir())


def _make_folders_writable(root: Path) -> None:
    """Give the owner write access to every folder under `root`, which `copytree` copied with its mode."""
    for folder, _, _ in os.walk(root):
        mode = os.stat(folder).st_mode
        if not mode & stat.S_IWUSR:
            os.chmod(folder, mode | stat.S_IWUSR)
