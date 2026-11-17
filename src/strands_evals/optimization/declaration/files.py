"""File-backed markers; `Space.create_candidate_copy` calls their `write()` (a `SkillLibrary`'s `_write()`)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands import Skill

from .markers import _check_positive_int, _check_str, _default_size_cap, _Marker


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
        _write_text_atomically(target, value)

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
        _write_text_atomically(target, self._render(value, text))

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


# Names a library snapshot never reads, at any depth: tools write them beside skills (the optimizer's
# run folders under the default base folder, a git repository, Python and Finder caches), and the agent
# never loads them. A version can't hold them either, so a stored version reads back as itself.
_SKIPPED_NAMES = frozenset({".strands-evals", ".git", "__pycache__", ".DS_Store"})

# A version's value: `skills/<16 hex>`, also its folder in a run's `artifacts/`.
_VERSION_PREFIX = "skills/"
_VERSION_VALUE = re.compile(r"skills/[0-9a-f]{16}")

_SKILL_FILE_NAMES = ("SKILL.md", "skill.md")  # as the Strands skills runtime looks for them, in its order


@dataclass(frozen=True)
class _Version:
    """One stored library version: every file by its posix path relative to the library (sorted), and the
    executable ones. Never changed once stored."""

    files: dict[str, bytes]
    executable: frozenset[str]


class SkillLibrary(_Marker):
    """A directory of skills whose value is a library version, so skills can be edited, added, and retired.

    The first argument is the path of the folder the agent loads its skills from, the same path
    passed to `AgentSkills`; the declared name (`skills=` below) is the factory's parameter:

        skills/                    # SkillLibrary("skills")
          triage/SKILL.md
          billing/SKILL.md

        @tunable(skills=SkillLibrary("skills"))
        def create_agent(skills: Path) -> Agent:
            return Agent(plugins=[AgentSkills(skills=skills)])  # the candidate's copy of skills/

    The declared name is one key whose value is a library version, `"skills/<hash>"`: the first 16 hex of
    the sha256 of every file in it (its path, whether it is executable, and its bytes), so equal libraries
    are one value. Every file under the folder is read once, at declaration, following symlinks; that is
    the default version. Files and folders named `.strands-evals`, `.git`, `__pycache__`, or `.DS_Store`
    are left out at any depth, and a version holds files only (an empty subfolder isn't part of it). The
    folder may be empty: the default is then the empty version.

    A version holds any set of skills: `revise` makes one from another (edit, add, or remove files), `add`
    stores the one in a folder, and `files` reads one. Versions are kept in memory by this marker; a run
    stores each version its candidates hold in `<run>/artifacts/skills/<hash>/` and reads them back on
    resume. To build a run's candidate in another process, add its version first:
    `library.add(run_dir / "artifacts" / values["skills"])`.

    The skills are found as the Strands skills runtime finds them: the folder itself when it holds a
    `SKILL.md` (or `skill.md`), a single skill, else each subfolder that does. Reflection edits each skill's
    `SKILL.md` as a text of its own, named `<declared name>/<path in the library>`, e.g.
    `skills/triage/SKILL.md`; other files (references, scripts) are kept as they are, and a `FileText` or
    `JsonField` may declare one of them (`Space.validate` then requires every version to hold that file as
    declared).

    Every version is checked when it is stored (`ValueError` from `revise` and `add`):
    - a library keeps no `SKILL.md` at its root, and a single skill keeps its `SKILL.md`;
    - each tuned skill's `SKILL.md` is UTF-8 and opens with a YAML frontmatter block (between `---`
      lines) holding a non-empty `name` and `description`, parsed as the Strands skills runtime parses it;
    - a declared skill keeps its `name`, so it keeps its identity, and a new skill is named after its
      folder, which the runtime checks against the `name`;
    - no tuned skill shares its `name` with another skill (the runtime keeps only one);
    - each tuned `SKILL.md` is within its size cap.

    To tune only some skills of a library, name them: with `SkillLibrary("skills", only=("triage",))`,
    a version may change only files in `triage/`; the other skills stay as declared, and Reflection
    edits only `skills/triage/SKILL.md`. The factory receives the path of this folder in the
    candidate's copy (the original folder in the plain call); the copy keeps the folder's name, which
    the runtime checks against a single skill's `name`.

    Attributes:
        path: The folder the agent loads its skills from.
        default: The declared folder's version.
        only: The skills `only=` names, or None.
        size_cap: Max characters per `SKILL.md`, or None for `max(2000, 3 * len(declared text))` each
            (2000 for a new one).
        edit_instructions: Shown to the proposer next to every `SKILL.md`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        only: list[str] | tuple[str, ...] | None = None,
        size_cap: int | None = None,
        edit_instructions: str | None = None,
    ):
        """Declare a folder of skills as tunable, or only the named skills in it.

        Args:
            path: The folder of skills, or one skill's folder; a relative path is taken from the
                current working directory. It may be empty.
            only: The skills to tune, a list or tuple of folder names, e.g. `("triage", "billing")`;
                None tunes the whole library. The others stay as declared.
            size_cap: Max characters per `SKILL.md`; None means `max(2000, 3 * len(declared text))` for
                each, 2000 for a new one.
            edit_instructions: Shown to the proposer next to every `SKILL.md`.

        Raises:
            FileNotFoundError: `path` is not a directory.
            TypeError: `size_cap` isn't a positive integer, `edit_instructions` isn't a str, `only`
                isn't a non-empty list or tuple of names of skills in the folder (or `path` is a single
                skill), a folder or file in it can't be read (a symlink loop or a broken symlink
                included), or the declared library fails a check above (a tuned `SKILL.md` that isn't
                UTF-8 or lacks the frontmatter, or two skills with the same `name`).
        """
        _check_positive_int("size_cap", size_cap)
        _check_str("edit_instructions", edit_instructions)
        if not isinstance(path, (str, Path)):
            raise TypeError(f"SkillLibrary path must be a str or Path, got {path!r}")
        if only is not None and not (isinstance(only, (list, tuple)) and all(isinstance(name, str) for name in only)):
            raise TypeError(f"SkillLibrary only must be a list or tuple of skill names, got {only!r}")
        if only is not None and not only:
            raise TypeError("SkillLibrary only names no skill; pass None to tune every skill")
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            where = (
                ""
                if Path(path).is_absolute()
                else f" (looked for {self.path}, from the working directory {Path.cwd()})"
            )
            raise FileNotFoundError(f"SkillLibrary: {path} is not a directory{where}")
        self.only = None if only is None else tuple(only)
        self.size_cap = size_cap
        self.edit_instructions = edit_instructions
        files, executable = _read_folder(self.path)
        self._single = any(name in files for name in _SKILL_FILE_NAMES)  # the folder is itself a skill
        if self.only is not None:
            if self._single:
                raise TypeError(f"SkillLibrary: {self.path} is a single skill; only= picks skills of a directory")
            found = _skills_in(files, single=False)
            unknown = sorted(set(self.only) - found.keys())
            if unknown:
                raise TypeError(
                    f"SkillLibrary: no skill named {', '.join(map(repr, unknown))} in {self.path} "
                    f"(it has {', '.join(found) or 'none'})"
                )
        self._declared = _Version(dict(sorted(files.items())), executable)
        self._names: dict[str, str] = {}  # each declared tuned skill's folder ("" for a single skill) -> its name
        try:
            self._names = self._check(files, executable, declared=True)
        except ValueError as error:
            raise TypeError(f"SkillLibrary: {self.path}: {error}") from error
        self.default = _VERSION_PREFIX + _digest(files, executable)
        self._versions: dict[str, _Version] = {self.default: self._declared}
        self._blobs: dict[bytes, bytes] = {data: data for data in files.values()}  # one copy of each content

    def _canonical(self, value: Any) -> Any:
        """The value as is: a version has no other form."""
        return value

    def _format_problem(self, value: str) -> str | None:
        """A one-line problem with `value`, or None: a version is checked when it is stored, so only one not
        stored here has one."""
        if value in self._versions:
            return None
        return f"{value!r} is not a version of this library stored in this process (make one with revise())"

    def files(self, value: str) -> dict[str, bytes]:
        """A version's files.

        Args:
            value: A version stored here, e.g. a candidate's value for this library.

        Returns:
            Posix path relative to the library -> bytes, sorted by path.

        Raises:
            KeyError: `value` is not a version stored here.
        """
        return dict(self._version(value).files)

    def revise(self, value: str, changes: Mapping[str, str | bytes | None]) -> str:
        """Store the version `value` holds with `changes` applied.

        Args:
            value: The version changed, e.g. a candidate's value for this library.
            changes: Posix path relative to the library -> its new content (text is written as UTF-8),
                or None to remove the file. A changed file keeps its executable bit; a new one has none.
                E.g. `{"refunds/SKILL.md": text}` adds a skill, and removing every file of
                `triage/` retires that skill.

        Returns:
            The new version's value, `"skills/<hash>"`.

        Raises:
            KeyError: `value` is not a version stored here.
            TypeError: A content is neither str, bytes, nor None.
            ValueError: A path isn't a plain relative path (empty, absolute, a `.` or `..` part, a
                backslash, a control character, or a name a version can't hold, e.g. `.git`), a removed
                path isn't a file of the version, an added path is both a file and a folder or differs
                from another only in case, or the new version fails a check (see the class).
        """
        version = self._version(value)
        files = dict(version.files)
        for path, content in changes.items():
            problem = _path_problem(path)
            if problem is not None:
                raise ValueError(f"{path!r}: {problem}")
            if content is None:
                if path not in files:
                    raise ValueError(f"{path}: not a file of {value}, so it can't be removed")
                del files[path]
            elif isinstance(content, (str, bytes)):
                files[path] = content.encode("utf-8") if isinstance(content, str) else content
            else:
                raise TypeError(f"{path}: content must be str, bytes, or None, got {type(content).__name__}")
        _check_layout(files, [path for path in changes if path not in version.files])
        return self._store(files, frozenset(version.executable & files.keys()))

    def add(self, folder: str | Path) -> str:
        """Store the library in `folder`, every file read as at declaration.

        E.g. a version a run stored: `library.add(run_dir / "artifacts" / "skills/<hash>")`.

        Args:
            folder: A folder holding a library.

        Returns:
            Its version's value, `"skills/<hash>"`.

        Raises:
            FileNotFoundError: `folder` is not a directory.
            TypeError: A folder or file in it can't be read.
            ValueError: The library fails a check (see the class).
        """
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"SkillLibrary: {folder} is not a directory")
        files, executable = _read_folder(root)
        return self._store(files, executable)

    def _version(self, value: str) -> _Version:
        version = self._versions.get(value)
        if version is None:
            raise KeyError(
                f"{value!r} is not a version of the SkillLibrary at {self.path} stored in this process; "
                "make one with revise(), or read a run's back with add(<run>/artifacts/<value>)"
            )
        return version

    def _store(self, files: Mapping[str, bytes], executable: frozenset[str]) -> str:
        """Check `files` as a version of this library, keep it, and return its value; raises `ValueError`.

        Kept sorted by path, so every way of making a version lists its files alike, and each content once
        across versions.
        """
        value = _VERSION_PREFIX + _digest(files, executable)
        if value not in self._versions:
            self._check(files, executable)
            kept = {path: self._blobs.setdefault(files[path], files[path]) for path in sorted(files)}
            self._versions[value] = _Version(kept, executable)
        return value

    def _holds(self, path: Path) -> bool:
        """Whether files under `path` would be part of this library's versions (inside it, in no left-out folder)."""
        return path.is_relative_to(self.path) and not _SKIPPED_NAMES & set(path.relative_to(self.path).parts)

    def _check(
        self, files: Mapping[str, bytes], executable: frozenset[str], *, declared: bool = False
    ) -> dict[str, str]:
        """Each tuned skill's folder -> its `name`; raises `ValueError` with the first problem (see the class).

        `declared`: `files` is the declared library, whose names every later version keeps.
        """
        roots = [name for name in _SKILL_FILE_NAMES if name in files]
        if self._single and not roots:
            raise ValueError("a single skill's version must keep its SKILL.md")
        if not self._single and roots:
            raise ValueError(f"{roots[0]} at the library's root would make the folder one skill; give it a folder")
        if self.only is not None:
            before = self._declared
            changed = sorted(
                path
                for path in files.keys() | before.files.keys()
                if (files.get(path), path in executable) != (before.files.get(path), path in before.executable)
                and path.split("/", 1)[0] not in self.only
            )
            if changed:
                raise ValueError(f"{changed[0]}: only {', '.join(self.only)} may change (only=)")
        tuned: dict[str, str] = {}  # a tuned skill's folder -> its name
        untuned: dict[str, str] = {}  # an untuned skill's folder -> its name, when the runtime can read one
        for folder, skill_file in _skills_in(files, single=self._single).items():
            data = files[skill_file]
            if self.only is not None and folder not in self.only:  # as declared: the runtime loads it, or skips it
                try:
                    untuned[folder] = _skill_name(data.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    pass
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError(f"{skill_file}: not UTF-8 text") from None
            cap = self._size_cap(skill_file)
            if len(text) > cap:
                raise ValueError(f"{skill_file}: {len(text)} characters, over the size_cap of {cap}")
            try:
                name = _skill_name(text)
            except ValueError as error:
                raise ValueError(f"{skill_file}: {error}") from None
            expected = self._names.get(folder, folder)
            if not declared and name != expected:
                rule = "must stay" if folder in self._names else "must be its folder's name,"
                raise ValueError(f"{skill_file}: the frontmatter name {rule} {expected!r}, got {name!r}")
            tuned[folder] = name
        # The runtime loads one skill per name, so a tuned skill sharing its name with another might never reach
        # the agent; untuned skills sharing one are left as the runtime loads them.
        names: dict[str, str] = {}  # a tuned skill's name -> its folder
        for folder, name in [*tuned.items(), *untuned.items()]:
            other = names.get(name)
            if other is not None:
                raise ValueError(
                    f"{other or '.'} and {folder or '.'} both name their skill {name!r}; "
                    "the runtime keeps only one, so give each a unique name"
                )
            if folder in tuned:
                names[name] = folder
        return tuned

    def _size_cap(self, skill_file: str) -> int:
        """The size cap of the `SKILL.md` at `skill_file`."""
        if self.size_cap is not None:
            return self.size_cap
        declared = self._declared.files.get(skill_file, b"")
        return _default_size_cap(declared.decode("utf-8", errors="replace"))

    def _texts(self, value: str) -> dict[str, str]:
        """Each tuned skill's `SKILL.md` path -> its text in the version `value`, in the runtime's order."""
        files = self._version(value).files
        return {
            skill_file: files[skill_file].decode("utf-8")
            for folder, skill_file in _skills_in(files, single=self._single).items()
            if self.only is None or folder in self.only
        }

    def _write(self, value: str, target: Path) -> None:
        """Write the version `value` into the folder `target`, made if missing (also for the empty version)."""
        version = self._version(value)
        target.mkdir(parents=True, exist_ok=True)
        for path, data in version.files.items():
            file = target / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
            if path in version.executable:
                os.chmod(file, file.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _save(self, value: str, artifacts: Path) -> None:
        """Write the version `value` once into `artifacts/<value>/`: into a temporary folder beside it, then
        renamed into place."""
        target = artifacts / value
        if target.is_dir():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        building = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            self._write(value, building)
        except BaseException:
            shutil.rmtree(building, ignore_errors=True)
            raise
        try:
            os.rename(building, target)
        except OSError:
            shutil.rmtree(building, ignore_errors=True)
            if not target.is_dir():  # else another call stored it first
                raise

    def _load(self, value: str, artifacts: Path) -> None:
        """Store the version `value` from `artifacts/<value>/`, where `_save` wrote it.

        Raises:
            ValueError: `value` isn't a version's value, the folder is missing, doesn't hold the version it is
                named for (it changed since it was written), or holds one that fails this library's checks.
        """
        if not _VERSION_VALUE.fullmatch(value):
            raise ValueError(f"{value!r} is not a skill-library version's value")
        folder = artifacts / value
        if not folder.is_dir():
            raise ValueError(f"{folder} is missing, so the version {value} can't be read back")
        files, executable = _read_folder(folder)
        if _VERSION_PREFIX + _digest(files, executable) != value:
            raise ValueError(f"{folder} doesn't hold the version it is named for; it changed since it was written")
        self._store(files, executable)


def _read_folder(root: Path) -> tuple[dict[str, bytes], frozenset[str]]:
    """Every file under `root` by its posix path relative to it, and the executable ones, following symlinks.

    Leaves out `_SKIPPED_NAMES`. Raises `TypeError` naming what can't be read: a folder or file, a broken
    symlink, a symlink leading back to a folder it is in, or something that is no file (e.g. a named pipe).
    On Windows, which has no executable bit, no file is executable.
    """

    def fail(error: OSError) -> None:
        raise TypeError(
            f"SkillLibrary: can't read {error.filename} ({error.strerror}), which every candidate's copy needs"
        ) from error

    files: dict[str, bytes] = {}
    executable: set[str] = set()
    walked: dict[str, str] = {}  # each folder walked -> its real path
    for here, folders, names in os.walk(root, onerror=fail, followlinks=True):
        real = os.path.realpath(here)
        up = os.path.dirname(here)
        while up in walked:
            if walked[up] == real:
                raise TypeError(f"SkillLibrary: {here} is a symlink to a folder it is in; replace it with a copy")
            up = os.path.dirname(up)
        walked[here] = real
        folders[:] = sorted(name for name in folders if name not in _SKIPPED_NAMES)
        for name in sorted(names):
            if name in _SKIPPED_NAMES:
                continue
            file = Path(here, name)
            try:
                mode = file.stat().st_mode
                if not stat.S_ISREG(mode):
                    raise TypeError(f"SkillLibrary: {file} is not a file; a library holds only files and folders")
                data = file.read_bytes()
            except OSError as error:
                fail(error)
            relative = file.relative_to(root).as_posix()
            files[relative] = data
            if os.name != "nt" and mode & stat.S_IXUSR:
                executable.add(relative)
    return files, frozenset(executable)


def _digest(files: Mapping[str, bytes], executable: frozenset[str]) -> str:
    """First 16 hex of the sha256 of every file: its path, whether it is executable, its length, and its bytes."""
    digest = hashlib.sha256()
    for path in sorted(files):
        data = files[path]
        digest.update(f"{path}\0{'x' if path in executable else '-'}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()[:16]


def _skills_in(files: Iterable[str], *, single: bool) -> dict[str, str]:
    """Each skill's folder ("" for a single skill) -> its `SKILL.md` path, as the Strands skills runtime finds them.

    A single skill is the folder itself; a library's skills are its subfolders holding a `SKILL.md`, else a
    `skill.md`, sorted by name.
    """
    paths = set(files)
    if single:
        found = next((name for name in _SKILL_FILE_NAMES if name in paths), None)
        return {} if found is None else {"": found}
    skills: dict[str, str] = {}
    for folder in sorted({path.split("/", 1)[0] for path in paths if "/" in path}):
        found = next((f"{folder}/{name}" for name in _SKILL_FILE_NAMES if f"{folder}/{name}" in paths), None)
        if found is not None:
            skills[folder] = found
    return skills


def _path_problem(path: Any) -> str | None:
    """Why `path` can't name a file of a version, or None: it must be a plain relative posix path."""
    if not isinstance(path, str) or not path:
        return "a path must be a non-empty str"
    if "\\" in path or path.startswith("/"):
        return "a path must be relative, with / between its parts"
    if any(ord(character) < 32 or character == "\x7f" for character in path):
        return "a path can't hold a control character"
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "a path can't have an empty, '.', or '..' part"
    skipped = next((part for part in parts if part in _SKIPPED_NAMES), None)
    if skipped is not None:
        return f"a version can't hold {skipped!r}, which a library snapshot leaves out"
    return None


def _check_layout(files: Iterable[str], added: Iterable[str]) -> None:
    """Raise `ValueError` when an added path is also a folder of another, or differs from another only in case.

    Every new conflict involves an added path, so only those are checked: a library declared on a
    case-sensitive disk with both `README.md` and `readme.md` can still be revised, those two included.
    """
    every = set(files)
    folders = {"/".join(parts[:i]) for parts in (path.split("/") for path in every) for i in range(1, len(parts))}
    spelled: dict[str, set[str]] = {}  # each path and folder, casefolded -> as written
    for name in every | folders:
        spelled.setdefault(name.casefold(), set()).add(name)
    for path in added:
        if path not in every:
            continue  # added, then removed
        parts = path.split("/")
        prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
        for prefix in prefixes[:-1]:
            if prefix in every:
                raise ValueError(f"{prefix} is a file and a folder (of {path})")
        if path in folders:
            raise ValueError(f"{path} is a file and a folder")
        for name in prefixes:
            others = sorted(spelled[name.casefold()] - {name})
            if others:
                raise ValueError(f"{name} and {others[0]} differ only in case; a file system may hold only one")


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


def _ignore_containing(destinations: tuple[Path, ...], libraries: tuple[Path, ...] = ()) -> Any:
    """A `copytree` ignore that skips any entry the copy is being written into, and each of `libraries`."""

    def ignore(folder: str, names: list[str]) -> set[str]:
        # Resolve the folder once; only one on the way to a destination or a library needs its entries checked.
        here = Path(folder).resolve()
        if not any(path.is_relative_to(here) for path in (*destinations, *libraries)):
            return set()
        skipped = set()
        for name in names:
            path = Path(folder, name)
            entry = path.resolve()
            if any(destination.is_relative_to(entry) for destination in destinations):
                skipped.add(name)
            elif entry in libraries and not path.is_symlink():  # a link to a library is copied as a link
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


def _write_text_atomically(target: Path, text: str) -> None:
    """Write `text` to `target` through a temp file and a rename, never through a symlink.

    A reader sees the old file or the whole new one. The new file keeps `target`'s permission bits
    (e.g. a script's executable bit), or gets a plain write's mode when `target` doesn't exist. The
    parent folder is created, and the temp file is removed if the write fails.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as file:
            file.write(text.encode("utf-8"))
        if target.exists():
            shutil.copymode(target, temp)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


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
