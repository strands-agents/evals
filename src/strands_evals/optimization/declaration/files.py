"""File-backed markers and the per-candidate copy of the user's files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from ...types.optimization import Value

if TYPE_CHECKING:
    from .space import Space


class FileText:
    """Text whose source of truth is a file; the baseline is the file's content.

    With `as_path=True` the factory receives the path of the candidate's copy instead of the text.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        size_cap: int | None = None,
        as_path: bool = False,
        edit_instructions: str | None = None,
    ):
        raise NotImplementedError

    def write(self, value: str, target: Path) -> None:
        """Replace the text of `target`, this value's file inside the copy."""
        raise NotImplementedError


class JsonField:
    """One field of a JSON file, addressed by JSON pointer; the diff lands on `path#pointer`."""

    def __init__(
        self,
        path: str | Path,
        pointer: str,
        *,
        size_cap: int | None = None,
        edit_instructions: str | None = None,
    ):
        raise NotImplementedError

    def write(self, value: str, target: Path) -> None:
        """Set the field at the pointer; a non-string baseline means `value` is parsed as JSON."""
        raise NotImplementedError


class SkillLibrary:
    """A directory of skill files; each member's text is editable.

    Member keys are `<declared name>/<path relative to the library>`, e.g. `skills/triage/SKILL.md`.
    Resolves to the candidate's copy of the directory.
    """

    def __init__(self, path: str | Path, *, size_cap: int | None = None, edit_instructions: str | None = None):
        raise NotImplementedError

    def write(self, value: str, target: Path) -> None:
        """Replace one member's text, keeping its frontmatter well-formed."""
        raise NotImplementedError


def create_candidate_copy(space: Space, values: Mapping[str, Value], directory: Path) -> Path:
    """Copy each file-backed value's folder into `directory`, then write every active file-backed key.

    Each source folder goes to `directory/<folder name>-<short hash of its original path>/`; a folder
    inside another is copied once, as part of the outer one. Built in a temp folder and renamed into
    place, so an existing `directory` is always complete and a second call returns at once.
    Originals are never written.
    """
    raise NotImplementedError
