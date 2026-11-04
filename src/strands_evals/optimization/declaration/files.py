"""File-backed markers; `Space.create_candidate_copy` calls their `write()`."""

from __future__ import annotations

from pathlib import Path

from .markers import _Marker


class FileText(_Marker):
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


class JsonField(_Marker):
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


class SkillLibrary(_Marker):
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
