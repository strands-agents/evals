"""File-backed markers and the per-candidate copy of the user's files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...types.optimization import Values

# Type-checking only: space.py imports this module, so a runtime import would be a cycle. Internal,
# so get_type_hints(create_candidate_copy) failing on `Space` breaks no exported signature.
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
        """Declare a file's text as tunable.

        Args:
            path: The file whose text is the baseline.
            size_cap: Max characters; None means `max(2000, 3 * len(baseline))`.
            as_path: Pass the path of the candidate's copy instead of the text.
            edit_instructions: Shown to the proposer next to this value.
        """
        raise NotImplementedError

    def write(self, value: str, target: Path) -> None:
        """Replace the text of `target`, this value's file inside the copy.

        Args:
            value: The candidate's text.
            target: This value's file inside the copy.
        """
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
        """Declare one JSON field as tunable.

        Args:
            path: The JSON file.
            pointer: JSON pointer to the field, e.g. "/agent/system_prompt".
            size_cap: Max characters; None means `max(2000, 3 * len(baseline))`.
            edit_instructions: Shown to the proposer next to this value.
        """
        raise NotImplementedError

    def write(self, value: str, target: Path) -> None:
        """Set the field at the pointer; a non-string baseline means `value` is parsed as JSON.

        Args:
            value: The field's new value as text.
            target: The JSON file inside the copy.
        """
        raise NotImplementedError


class SkillLibrary:
    """A directory of skill files; each member's text is editable.

    Member keys are `<declared name>/<path relative to the library>`, e.g. `skills/triage/SKILL.md`.
    Resolves to the candidate's copy of the directory.
    """

    def __init__(self, path: str | Path, *, size_cap: int | None = None, edit_instructions: str | None = None):
        """Declare every member's text as tunable.

        Args:
            path: The directory of skill files.
            size_cap: Max characters per member; None means `max(2000, 3 * len(baseline))`.
            edit_instructions: Shown to the proposer next to every member.
        """
        raise NotImplementedError

    def write(self, value: str, target: Path) -> None:
        """Replace one member's text, keeping its frontmatter well-formed.

        Args:
            value: The member's new text.
            target: The member's file inside the copy.
        """
        raise NotImplementedError


def create_candidate_copy(space: Space, values: Values, directory: Path) -> Path:
    """Copy each file-backed value's folder into `directory`, then write every active file-backed key.

    Each source folder goes to `directory/<folder name>-<short hash of its original path>/`; a folder
    inside another is copied once, as part of the outer one. Built in a temp folder and renamed into
    place, so an existing `directory` is always complete and a second call returns at once.
    Originals are never written.

    Args:
        space: Names the file-backed keys and their sources.
        values: The candidate's values.
        directory: The candidate's copy, `<run>/candidates/<id>/optimization_files/`.

    Returns:
        `directory`.
    """
    raise NotImplementedError
