"""Progressive disclosure over a large keyed knowledge corpus.

`TraceIndex` gives a judge progressive disclosure over a large *trace*. Some
rubrics also need the judge to consult knowledge that lives **outside** the
trace — a catalog of the tools/skills the agent could have used, API specs,
policy or guideline documents. That corpus overflows the judge's context for the
same reason a large trajectory does, and inlining all of it degrades judge
accuracy even when it fits (lost-in-the-middle, position bias).

`KnowledgeIndex` applies the same list / get / search treatment to an arbitrary
collection of **keyed** documents, and adds `render()` for the common
retrieve-then-inject case where the metric already knows which entries are
relevant from the trace.

Two usage modes:

**Mode A — retrieve-then-inject (default, deterministic, one LLM call).** The
metric selects the trace-relevant keys and injects only those::

    index = KnowledgeIndex(catalog)                 # key -> document text
    knowledge_block = index.render(keys=plan_tool_names)
    judged_output = f"{agent_answer}\n{knowledge_block}"
    evaluator = OutputEvaluator(rubric="...")

**Mode B — agentic discovery (reserve for open-ended lookup).** When the needed
knowledge can't be predetermined ("does *any* entry cover this request?"), hand
the judge the tools and let it look things up::

    prompt_section, tools = index.for_judge()
    evaluator = OutputEvaluator(rubric="...", tools=tools)
    output = f"{agent_answer}\n{prompt_section}"

Like `TraceIndex`, this composes with `OutputEvaluator` (caller-controlled
output) and **not** with `TrajectoryEvaluator`, which inlines the full trajectory
unconditionally.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from strands import tool

from ._progressive import DEFAULT_MAX_READ_CHARS, paged_listing, preview, search_matches, window


@dataclass
class KnowledgeEntry:
    """One document in a knowledge corpus.

    Attributes:
        key: Stable identifier the judge uses to fetch the entry (tool name,
            skill domain, doc id). Must be unique within an index.
        content: Full document text.
        description: Optional one-line summary shown in the overview so the judge
            can decide whether to load the full content.
    """

    key: str
    content: str
    description: str | None = None


def _coerce_entries(
    entries: "Mapping[str, str] | Iterable[KnowledgeEntry] | Iterable[tuple[str, str]]",
) -> list[KnowledgeEntry]:
    """Normalize the accepted input shapes into a list of KnowledgeEntry.

    Accepts a ``{key: content}`` mapping, an iterable of `KnowledgeEntry`, or an
    iterable of ``(key, content)`` pairs — whichever is convenient for the caller
    assembling the corpus.
    """
    if isinstance(entries, Mapping):
        return [KnowledgeEntry(key=k, content=v) for k, v in entries.items()]
    out: list[KnowledgeEntry] = []
    for item in entries:
        if isinstance(item, KnowledgeEntry):
            out.append(item)
        else:
            key, content = item  # (key, content) pair
            out.append(KnowledgeEntry(key=key, content=content))
    return out


def _describe(entry: KnowledgeEntry) -> str:
    """One overview line describing an entry without its full content."""
    size = len(entry.content)
    if entry.description:
        return f"{entry.key} — {preview(entry.description, 80)} [{size} chars]: {preview(entry.content)}"
    return f"{entry.key} [{size} chars]: {preview(entry.content)}"


def _search_haystack(entry: KnowledgeEntry) -> str:
    """Plain-text rendering of an entry for search matching (key + description + content)."""
    parts = [entry.key]
    if entry.description:
        parts.append(entry.description)
    parts.append(entry.content)
    return "\n".join(parts)


class KnowledgeIndex:
    """Read-only list / get / search index over a keyed knowledge corpus.

    Attributes:
        max_read_chars: Cap on any single tool return, so a large entry or a long
            overview can't overflow the judge's context in one call. Oversized
            content is windowed and the tool reports how to page through it.
    """

    def __init__(
        self,
        entries: "Mapping[str, str] | Iterable[KnowledgeEntry] | Iterable[tuple[str, str]]",
        max_read_chars: int = DEFAULT_MAX_READ_CHARS,
    ):
        if max_read_chars < 1:
            raise ValueError(f"max_read_chars must be >= 1, got {max_read_chars}")
        self.max_read_chars = max_read_chars

        coerced = _coerce_entries(entries)
        self._entries: dict[str, KnowledgeEntry] = {}
        for entry in coerced:
            if entry.key in self._entries:
                raise ValueError(f"duplicate knowledge key {entry.key!r}")
            self._entries[entry.key] = entry
        # Sorted key order gives the overview and search a stable, predictable
        # ordering independent of insertion/dict order.
        self._keys = sorted(self._entries)
        self._describe_lines = [f"[{k}] {_describe(self._entries[k])}" for k in self._keys]

        # Bind instance state into plain functions so @tool sees clean signatures.
        this = self

        @tool
        def list_entries(offset: int = 0) -> str:
            """List knowledge entries: one line per entry with its key, optional
            description, size, and a truncated preview. Call this first to decide which
            entries to load. Long corpora are paged; the response says how to page with
            offset. Previews are truncated — load an entry with get_entry before you rely
            on its content to score.

            Args:
                offset: Entry index to start the listing from, for paging long corpora.
            """
            return this.overview(offset)

        @tool
        def get_entry(key: str, offset: int = 0) -> str:
            """Get the full content of one knowledge entry by its key.
            Large entries are windowed; the response says how to page with offset.

            Args:
                key: Entry key as shown by list_entries.
                offset: Character offset for paging through oversized entries.
            """
            entry = this._entries.get(key)
            if entry is None:
                return f"ERROR: no entry with key {key!r}. Call list_entries to see available keys."
            return window(entry.content, offset, this.max_read_chars)

        @tool
        def search_entries(pattern: str, max_matches: int = 20, is_regex: bool = False) -> str:
            """Search all knowledge entries for a literal string (default) or a regex.
            Matching is case-insensitive and covers the key, description, and content.
            Regex anchors ^/$ match line boundaries. Returns matching keys with a short
            excerpt and per-entry match count, capped at max_read_chars total; use
            get_entry to load a match in full.

            Args:
                pattern: Text to search for. Treated literally unless is_regex=True.
                max_matches: Maximum number of matching entries to return. Results are
                    also capped at max_read_chars total, whichever comes first.
                is_regex: Set True to treat pattern as a regular expression.
            """
            return search_matches(
                len(this._keys),
                lambda i: _search_haystack(this._entries[this._keys[i]]),
                lambda i: this._keys[i],
                pattern,
                max_matches,
                is_regex,
                this.max_read_chars,
                unit_pl="entries",
            )

        self.tools = [list_entries, get_entry, search_entries]

    @property
    def keys(self) -> list[str]:
        """The entry keys, in sorted order."""
        return list(self._keys)

    def overview(self, offset: int = 0) -> str:
        """Compact one-line-per-entry overview of the corpus, paged by entry index.

        Args:
            offset: Entry index to start from. The listing is capped at
                `max_read_chars`; if it doesn't fit, the response says the next offset.
        """
        total = len(self._keys)
        header = (
            f"Knowledge overview: {total} entries. Previews are truncated; call "
            f"get_entry/search_entries to load full content (up to {self.max_read_chars} "
            f"chars per call) before scoring."
        )
        return paged_listing(
            header,
            self._describe_lines,
            offset,
            self.max_read_chars,
            unit_sg="entry",
            unit_pl="entries",
        )

    def render(self, keys: Iterable[str], *, max_chars: int | None = None) -> str:
        """Render selected entries as a bounded ``<Knowledge>`` block for inline injection.

        This is the deterministic, tool-free retrieve-then-inject path (Mode A): the
        metric picks the trace-relevant keys and injects only those, so the judge scores
        in one LLM call with no discovery round-trips. Unknown keys are reported inline
        (rather than raising) so a metric deriving keys from the trace degrades to a
        visible note instead of a crash. The whole block is capped at ``max_chars``
        (defaults to `max_read_chars`); if the selected entries don't fit, later entries
        are truncated with a marker rather than silently dropped.

        Args:
            keys: Entry keys to include, in the order given.
            max_chars: Cap on the rendered block. Defaults to `max_read_chars`.

        Returns:
            A ``<Knowledge>...</Knowledge>`` block containing the selected entries.
        """
        budget = self.max_read_chars if max_chars is None else max_chars
        seen: set[str] = set()
        blocks: list[str] = []
        used = len("<Knowledge>\n</Knowledge>")
        truncated = 0
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            entry = self._entries.get(key)
            if entry is None:
                block = f"[{key}] ERROR: no such knowledge entry"
            else:
                block = f"[{key}]\n{entry.content}"
            # Always emit at least the first block; otherwise stop once the budget is
            # spent and report how many entries were dropped.
            if blocks and used + len(block) + 1 > budget:
                truncated += 1
                continue
            blocks.append(block)
            used += len(block) + 1
        body = "\n".join(blocks)
        if truncated:
            body += f"\n[TRUNCATED: {truncated} more selected entries omitted at {budget} chars]"
        return f"<Knowledge>\n{body}\n</Knowledge>"

    def for_judge(self) -> tuple[str, list]:
        """Return the overview block and the discovery tools together (Mode B).

        Composing a `KnowledgeIndex` into an evaluator has two halves — the overview must
        go into the judged output and the discovery tools must be passed to the evaluator
        — and doing only one silently degrades the judge. This hands back both::

            prompt_section, tools = index.for_judge()
            evaluator = OutputEvaluator(rubric="...", tools=tools)
            output = f"{agent_answer}\n{prompt_section}"

        Returns:
            A ``(prompt_section, tools)`` pair. ``prompt_section`` is the overview wrapped
            in a ``<KnowledgeOverview>`` block; ``tools`` is `self.tools`.
        """
        prompt_section = f"<KnowledgeOverview>\n{self.overview()}\n</KnowledgeOverview>"
        return prompt_section, self.tools
