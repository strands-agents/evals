"""A custom kind for tests: a note, a text kept in the run's artifact store and named by its hash."""

import hashlib
import os
import tempfile

from strands_evals.optimization import CustomMarker, Patch, Proposer


def note_value(text):
    """The value naming a note's text: `notes/<hash>`, relative to the artifact store."""
    return f"notes/{hashlib.sha256(text.encode()).hexdigest()[:16]}"


def store_note(artifacts, text):
    """Write a note into the store once, by temp file and rename, and return its value."""
    value = note_value(text)
    target = artifacts / value
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=target.parent)
        with os.fdopen(handle, "w") as file:
            file.write(text)
        os.replace(temporary, target)
    return value


class Note(CustomMarker):
    """A note whose default the marker holds, so the baseline needs nothing stored."""

    kind = "note"

    def __init__(self, seed):
        self.seed = seed
        self.default = note_value(seed)
        self.checked = []  # the store each check was given

    def check_value(self, value, artifacts):
        self.checked.append(artifacts)
        if not isinstance(value, str) or not value.startswith("notes/"):
            return f"expected notes/<hash>, got {value!r}"
        if value == self.default:
            return None
        if artifacts is None or not (artifacts / value).is_file():
            return f"{value} is not in the artifact store"
        return None

    def resolve(self, value, artifacts):
        return self.seed if value == self.default else (artifacts / value).read_text()


class NoteWriter(Proposer):
    """Writes the given texts into the store, one per patch, and proposes their values."""

    kinds = frozenset({"note"})

    def __init__(self, *texts):
        self.texts = list(texts) if texts else ["a second note", "a much longer third note"]
        self.observed = []

    async def propose(self, request):
        patches = []
        for _ in range(request.n):
            if not self.texts:
                break
            text = self.texts.pop(0)
            value = store_note(request.artifacts, text)
            patches.append(Patch(updates={key: value for key in request.keys}, rationale=f"wrote {text!r}"))
        return patches

    def observe(self, observation):
        self.observed.append(observation)
