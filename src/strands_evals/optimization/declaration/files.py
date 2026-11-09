"""File-backed markers; `Space.create_candidate_copy` calls their `write()`."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from strands import Skill

from .markers import _check_positive_int, _check_str, _Marker


class FileText(_Marker):
    """Text whose source of truth is a file; the baseline is the file's content.

    With `as_path=True` the factory receives the path of the candidate's copy instead of the text.
    The file is read once, at declaration, as UTF-8 and kept byte for byte (line endings included).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        size_cap: int | None = None,
        as_path: bool = False,
        edit_instructions: str | None = None,
    ):
        """Declare a file's text as tunable.

        Args:
            path: The file whose text is the baseline; a relative path is taken from the current
                working directory.
            size_cap: Max characters; None means `max(2000, 3 * len(baseline))`.
            as_path: Pass the path of the candidate's copy instead of the text.
            edit_instructions: Shown to the proposer next to this value.

        Raises:
            FileNotFoundError: `path` is not a file.
            TypeError: `size_cap` isn't a positive integer, `edit_instructions` isn't a str,
                `as_path` isn't a bool, or the file isn't UTF-8 text.
        """
        _check_positive_int("size_cap", size_cap)
        _check_str("edit_instructions", edit_instructions)
        if not isinstance(as_path, bool):
            raise TypeError(f"FileText as_path must be a bool, got {as_path!r}")
        self.path = _existing_file(path, "FileText")
        self.default = _read_text(self.path, "FileText")
        self.size_cap = size_cap
        self.as_path = as_path
        self.edit_instructions = edit_instructions

    def write(self, value: str, target: Path) -> None:
        """Replace the text of `target`, this value's file inside the copy.

        A `target` that already holds `value` is left untouched, so its bytes stay the original's.

        Args:
            value: The candidate's text.
            target: This value's file inside the copy.
        """
        if target.is_file() and target.read_bytes() == value.encode("utf-8"):
            return
        _replace_file(target, value)

    def _original(self) -> str:
        """The whole file's text at declaration."""
        return self.default

    def _render(self, value: str, text: str) -> str:
        """The whole file's text with `value` applied to `text`."""
        return value

    def _canonical(self, value: Any) -> Any:
        """The value as is: a file's text has no other form."""
        return value

    def _format_problem(self, value: str) -> str | None:
        """A one-line format problem with `value`, or None."""
        return None


class JsonField(_Marker):
    """One field of a JSON file, addressed by JSON pointer (RFC 6901).

    A string field's value is its text. Any other field's value is its JSON text (2-space indent),
    and a proposed value must parse as JSON; `Space.complete` rewrites it in that form, so "0.50" and
    "0.5" are one candidate. Writing a changed field rewrites the whole file as JSON
    with a 2-space indent.
    """

    def __init__(
        self,
        path: str | Path,
        pointer: str,
        *,
        size_cap: int | None = None,
        edit_instructions: str | None = None,
    ):
        """Declare one JSON field as tunable.

        Args:
            path: The JSON file; a relative path is taken from the current working directory.
            pointer: JSON pointer to the field, e.g. "/agent/system_prompt".
            size_cap: Max characters; None means `max(2000, 3 * len(baseline))`.
            edit_instructions: Shown to the proposer next to this value.

        Raises:
            FileNotFoundError: `path` is not a file.
            TypeError: `size_cap` isn't a positive integer, `edit_instructions` isn't a str, the
                file isn't UTF-8 JSON, or `pointer` isn't a JSON pointer to an existing field.
        """
        _check_positive_int("size_cap", size_cap)
        _check_str("edit_instructions", edit_instructions)
        self.path = _existing_file(path, "JsonField")
        self.pointer = pointer
        self._tokens = _parse_pointer(pointer)
        self._text = _read_text(self.path, "JsonField")
        try:
            document = _loads(self._text)
        except ValueError as error:
            raise TypeError(f"JsonField: {self.path} is not valid JSON: {error}") from error
        field = _get(document, self._tokens, pointer, self.path)
        self._is_text = isinstance(field, str)
        self._kind = _json_kind(field)
        self.default: str = field if isinstance(field, str) else _dumps(field)
        self.size_cap = size_cap
        self.edit_instructions = edit_instructions

    def write(self, value: str, target: Path) -> None:
        """Set the field at the pointer; a non-string baseline means `value` is parsed as JSON.

        A `target` whose field already equals `value` is left untouched, so its bytes (formatting
        included) stay the original's.

        Args:
            value: The field's new value as text.
            target: The JSON file inside the copy.

        Raises:
            TypeError: The file changed since declaration: it is gone, is no longer valid JSON, or no
                longer has a field at the pointer. The message names the original file, not the copy.
        """
        if not target.is_file():
            raise TypeError(f"JsonField: {self.path} no longer exists (changed since declaration)")
        try:
            text = target.read_bytes().decode("utf-8")
            _loads(text)
        except ValueError as error:  # UnicodeDecodeError included
            raise TypeError(
                f"JsonField: {self.path} is no longer valid JSON (changed since declaration): {error}"
            ) from error
        if self._holds(value, text):
            return
        _replace_file(target, self._render(value, text))

    def _holds(self, value: str, text: str) -> bool:
        """Whether the field at the pointer in `text` already equals `value` (same JSON, same type)."""
        try:
            field = _get(_loads(text), self._tokens, self.pointer, self.path)
        except (TypeError, ValueError):
            return False
        if self._is_text:
            return isinstance(field, str) and field == value
        return not isinstance(field, str) and _dumps(field) == _dumps(_loads(value))

    def _original(self) -> str:
        """The whole file's text at declaration."""
        return self._text

    def _render(self, value: str, text: str) -> str:
        """The whole file's text with the field at the pointer set to `value`."""
        document = _loads(text)
        # Only to raise when the file lost the field since declaration.
        _get(document, self._tokens, self.pointer, self.path)
        _set(document, self._tokens, value if self._is_text else _loads(value))
        return _dumps(document) + "\n"

    def _canonical(self, value: Any) -> Any:
        """A non-string field's value as `_dumps` writes it, so equal JSON gives one candidate id.

        E.g. "0.50", " 0.5" and "5e-1" all become "0.5". Object key order is kept, as written to the
        file. A string field's text, and a value that isn't valid JSON (left for `_format_problem`), are
        returned as is.
        """
        if self._is_text or not isinstance(value, str):
            return value
        try:
            return _dumps(_loads(value))
        except ValueError:
            return value

    def _format_problem(self, value: str) -> str | None:
        """A one-line format problem with `value`, or None.

        A non-string field must get JSON of its baseline's kind (a number for a number, int or
        float); a `null` baseline takes any kind.
        """
        if self._is_text:
            return None
        try:
            parsed = _loads(value)
            _dumps(parsed)  # e.g. 1e400 parses as infinity, which no JSON file can hold
        except ValueError as error:
            return f"not valid JSON ({error})"
        kind = _json_kind(parsed)
        if self._kind != "null" and kind != self._kind:
            return f"expected a JSON {self._kind} like the file's {self.default}, got {value} ({kind})"
        return None


class SkillLibrary(_Marker):
    """A directory of skills; each skill's `SKILL.md` text is editable.

    The first argument is the path of the folder the agent loads its skills from, the same path
    passed to `AgentSkills`; the declared name (`skills=` below) is the factory's parameter:

        skills/                    # SkillLibrary("skills")
          triage/SKILL.md          # key skills/triage/SKILL.md
          billing/SKILL.md         # key skills/billing/SKILL.md

        @tunable(skills=SkillLibrary("skills"))
        def create_agent(skills: Path) -> Agent:
            return Agent(plugins=[AgentSkills(skills=skills)])  # the candidate's copy of skills/

    The skills are found as the Strands skills runtime finds them: the directory itself when it holds
    a `SKILL.md` (or `skill.md`), else each subdirectory that does. Each `SKILL.md` is one text value,
    keyed `<declared name>/<path relative to the library>`, e.g. `skills/triage/SKILL.md`. Other files
    (references, scripts) are copied with the library but not edited; a `FileText` or `JsonField` may
    declare one of them.

    To tune only some skills of a library, name them: `SkillLibrary("skills", only=("triage",))`
    tunes `skills/triage/SKILL.md` and copies the other skills as they are, never edited, so the
    factory loads the library from the same path as without `only`.

    A `SKILL.md` must open with a YAML frontmatter block (between `---` lines) holding a non-empty
    `name` and `description`, parsed as the Strands skills runtime parses it, and a proposed text
    must keep the baseline's `name`, so the skill keeps its identity. The factory receives the path of
    this directory in the candidate's copy (the original directory in the plain call); the copy keeps
    the directory's name, which the runtime checks against a skill's `name`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        only: list[str] | tuple[str, ...] | None = None,
        size_cap: int | None = None,
        edit_instructions: str | None = None,
    ):
        """Declare every skill's `SKILL.md` text as tunable, or only the named skills'.

        Args:
            path: The directory of skills, or one skill's directory; a relative path is taken from the
                current working directory.
            only: The skills to tune, a list or tuple of directory names, e.g. `("triage", "billing")`;
                None tunes every skill. The others are copied but not edited.
            size_cap: Max characters per `SKILL.md`; None means `max(2000, 3 * len(baseline))` for each.
            edit_instructions: Shown to the proposer next to every `SKILL.md`.

        Raises:
            FileNotFoundError: `path` is not a directory.
            TypeError: `size_cap` isn't a positive integer, `edit_instructions` isn't a str, `only`
                isn't a non-empty list or tuple of names of skills in the directory (or `path` is a
                single skill), the directory or a folder in it can't be read (every candidate's copy
                would fail on it), the directory holds no skill, a tuned skill's directory or `SKILL.md`
                is a symlink (its copy would write through to, or keep following, the original), or a
                tuned `SKILL.md` isn't UTF-8 or lacks the frontmatter above.
        """
        _check_positive_int("size_cap", size_cap)
        _check_str("edit_instructions", edit_instructions)
        if not isinstance(path, (str, Path)):
            raise TypeError(f"SkillLibrary path must be a str or Path, got {path!r}")
        if only is not None and not (isinstance(only, (list, tuple)) and all(isinstance(name, str) for name in only)):
            raise TypeError(f"SkillLibrary only must be a list or tuple of skill names, got {only!r}")
        if only is not None and not only:
            raise TypeError("SkillLibrary only names no skill; pass None to tune every skill")
        selected = None if only is None else set(only)
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            where = (
                ""
                if Path(path).is_absolute()
                else f" (looked for {self.path}, from the working directory {Path.cwd()})"
            )
            raise FileNotFoundError(f"SkillLibrary: {path} is not a directory{where}")
        self.size_cap = size_cap
        self.edit_instructions = edit_instructions
        root_file = _skill_file_in(self.path)
        if root_file is not None:
            if selected is not None:
                raise TypeError(f"SkillLibrary: {self.path} is a single skill; only= picks skills of a directory")
            skill_files = [root_file]
        else:
            found = {
                child.name: (child, skill_file)
                for child in sorted(self.path.iterdir())
                if child.is_dir() and (skill_file := _skill_file_in(child)) is not None
            }
            if not found:
                raise TypeError(f"SkillLibrary: {self.path} holds no skill (no SKILL.md in it or in a subdirectory)")
            unknown = sorted((selected or set()) - found.keys())
            if unknown:
                raise TypeError(
                    f"SkillLibrary: no skill named {', '.join(map(repr, unknown))} in {self.path} "
                    f"(it has {', '.join(found)})"
                )
            skill_files = []
            for name, (child, skill_file) in found.items():
                if selected is not None and name not in selected:
                    continue
                if child.is_symlink():
                    raise TypeError(
                        f"SkillLibrary: {child} is a symlink; its copy would write through to the original, "
                        "so replace it with a directory"
                    )
                skill_files.append(skill_file)
        # Path relative to the library (posix) -> the member, in the runtime's order.
        self._members = {
            file.relative_to(self.path).as_posix(): _SkillFile(file, self.path, size_cap, edit_instructions)
            for file in skill_files
        }


class _SkillFile(_Marker):
    """One `SKILL.md` of a `SkillLibrary`: its text, written into the candidate's copy."""

    def __init__(self, path: Path, copied: Path, size_cap: int | None, edit_instructions: str | None):
        if path.is_symlink():  # the copy would keep the link, so the baseline would follow the original
            raise TypeError(f"SkillLibrary: {path} is a symlink; replace it with the file it links to")
        self.path = path
        self.copied = copied  # the library's directory, which the candidate's copy holds
        self.default = _read_text(path, "SkillLibrary")
        self.size_cap = size_cap
        self.edit_instructions = edit_instructions
        try:
            self._name = _skill_name(self.default)
        except ValueError as error:
            raise TypeError(f"SkillLibrary: {path}: {error}") from error

    def write(self, value: str, target: Path) -> None:
        """Replace the text of `target`; one that already holds `value` keeps the original's bytes."""
        if target.is_file() and target.read_bytes() == value.encode("utf-8"):
            return
        _replace_file(target, value)

    def _original(self) -> str:
        """The whole file's text at declaration."""
        return self.default

    def _render(self, value: str, text: str) -> str:
        """The whole file's text with `value` applied to `text`."""
        return value

    def _canonical(self, value: Any) -> Any:
        """The value as is: a file's text has no other form."""
        return value

    def _format_problem(self, value: str) -> str | None:
        """A one-line problem with the frontmatter of `value`, or None; the `name` must stay the baseline's."""
        if value == self.default:  # parsed in __init__
            return None
        try:
            name = _skill_name(value)
        except ValueError as error:
            return str(error)
        if name != self._name:
            return f"the frontmatter name must stay {self._name!r}, got {name!r}"
        return None


def _skill_file_in(folder: Path) -> Path | None:
    """`folder/SKILL.md`, else `folder/skill.md`, as named on disk; None when it has neither.

    Names are read from the folder, not tested with `is_file()`, which on a case-insensitive file
    system finds `skill.md` as `SKILL.md`.
    """
    try:
        names = {entry.name for entry in folder.iterdir()}
    except OSError as error:  # every candidate's copy would fail on it
        raise TypeError(
            f"SkillLibrary: can't read {folder} ({error.strerror}), which every candidate's copy needs"
        ) from error
    for name in ("SKILL.md", "skill.md"):
        if name in names and (folder / name).is_file():
            return folder / name
    return None


def _skill_name(text: str) -> str:
    """The `name` in a `SKILL.md`'s frontmatter, parsed as the Strands skills runtime parses it.

    Raises `ValueError` saying what is wrong: no frontmatter block, YAML the runtime can't read, or
    no `name` or `description`.
    """
    return Skill.from_content(text).name


def _outer_folders(folders: Iterable[Path]) -> tuple[Path, ...]:
    """The folders to copy: each distinct folder not inside another one, shortest path first."""
    outer: list[Path] = []
    for folder in sorted(set(folders), key=lambda f: (len(f.parts), str(f))):
        if not any(folder.is_relative_to(other) for other in outer):
            outer.append(folder)
    return tuple(outer)


def _folder_in_copy(folder: Path) -> Path:
    """`<first 8 hex of the sha256 of its absolute path>/<folder name>`.

    The hash keeps a/prompts and b/prompts apart; the copy keeps the folder's own name, which a
    skills runtime checks against a skill's `name`.
    """
    return Path(hashlib.sha256(str(folder).encode("utf-8")).hexdigest()[:8]) / folder.name


def _location_in_copy(folders: Sequence[Path], file: Path) -> Path:
    """Where `file` lands inside a candidate's copy, relative to it."""
    for folder in folders:
        if file.is_relative_to(folder):
            return _folder_in_copy(folder) / file.relative_to(folder)
    raise KeyError(f"{file} is under none of the copied folders")


def _ignore_containing(destinations: tuple[Path, ...]) -> Any:
    """A `copytree` ignore that skips any entry the copy is being written into."""

    def ignore(folder: str, names: list[str]) -> set[str]:
        # Resolve the folder once; only one on the way to a destination needs its entries checked.
        here = Path(folder).resolve()
        if not any(destination.is_relative_to(here) for destination in destinations):
            return set()
        skipped = set()
        for name in names:
            entry = Path(folder, name).resolve()
            if any(destination.is_relative_to(entry) for destination in destinations):
                skipped.add(name)
        return skipped

    return ignore


def _existing_file(path: str | Path, marker: str) -> Path:
    """`path` as an absolute, resolved path; raises `FileNotFoundError` unless it is a file."""
    if not isinstance(path, (str, Path)):
        raise TypeError(f"{marker} path must be a str or Path, got {path!r}")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        where = "" if Path(path).is_absolute() else f" (looked for {resolved}, from the working directory {Path.cwd()})"
        raise FileNotFoundError(f"{marker}: {path} is not a file{where}")
    return resolved


def _read_text(path: Path, marker: str) -> str:
    """The file's exact text as UTF-8 (no newline translation); raises `TypeError` otherwise."""
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise TypeError(f"{marker}: {path} is not UTF-8 text") from error


def _replace_file(target: Path, text: str) -> None:
    """Write `text` to `target` through a temp file and a rename, never through a symlink.

    The new file keeps `target`'s permission bits (e.g. a script's executable bit).
    """
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temp.write_bytes(text.encode("utf-8"))
    if target.exists():
        shutil.copymode(target, temp)
    os.replace(temp, target)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _loads(text: str) -> Any:
    """`json.loads`, rejecting NaN and Infinity, which aren't JSON."""
    return json.loads(text, parse_constant=_reject_constant)


def _dumps(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def _parse_pointer(pointer: Any) -> tuple[str, ...]:
    """The reference tokens of a JSON pointer, unescaped (`~1` -> `/`, `~0` -> `~`)."""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise TypeError(
            f"JsonField pointer must start with '/' (e.g. '/agent/system_prompt'), got {pointer!r}; "
            "use FileText for the whole file"
        )
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/"))


def _get(document: Any, tokens: tuple[str, ...], pointer: str, path: Path) -> Any:
    """The value at `tokens`; raises `TypeError` naming the pointer when there is none."""
    node = document
    for token in tokens:
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and _is_index(token) and int(token) < len(node):
            node = node[int(token)]
        else:
            raise TypeError(f"JsonField: {path} has no field at {pointer!r}")
    return node


def _set(document: Any, tokens: tuple[str, ...], value: Any) -> None:
    """Set the value at `tokens`, which `_get` found at declaration."""
    node = document
    for token in tokens[:-1]:
        node = node[int(token)] if isinstance(node, list) else node[token]
    last = tokens[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def _json_kind(value: Any) -> str:
    """The JSON type of a parsed value: string, number, boolean, null, array, or object."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


def _is_index(token: str) -> bool:
    """An array index per RFC 6901: digits, no leading zero."""
    return token.isdigit() and token.isascii() and (token == "0" or not token.startswith("0"))
