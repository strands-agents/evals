"""Shared machinery for progressive-disclosure indexes.

`TraceIndex` (over a `Session`) and `ReferenceIndex` (over a keyed reference
corpus) expose the same list / get / search shape to a judge agent. The paging,
windowing, and search logic that keeps every tool return inside a `max_read_chars`
budget is identical between them and lives here so the two indexes can't drift.

None of these helpers know about spans or entries — callers pass in the already
rendered lines and per-item text accessors, plus the noun to use in the paging
markers (``span``/``spans`` or ``entry``/``entries``).
"""

import re
from typing import Callable

PREVIEW_CHARS = 120
DEFAULT_MAX_READ_CHARS = 8_000


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Collapse whitespace and truncate ``text`` to ``limit`` chars with an ellipsis."""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def window(text: str, offset: int, max_read_chars: int) -> str:
    """Return ``text`` from ``offset``, capped at ``max_read_chars``, with a paging marker.

    Oversized content is windowed rather than returned whole so a single large
    item can't overflow the judge's context; the marker says the next offset.
    """
    if offset < 0:
        return f"ERROR: offset {offset} is negative; use offset >= 0"
    if offset >= len(text):
        return f"ERROR: offset {offset} beyond content length {len(text)}"
    win = text[offset : offset + max_read_chars]
    if offset + len(win) < len(text):
        remaining = len(text) - offset - len(win)
        win += f"\n[TRUNCATED: {remaining} chars remain; call again with offset={offset + len(win)}]"
    return win


def paged_listing(
    header: str,
    lines_all: list[str],
    offset: int,
    max_read_chars: int,
    *,
    unit_sg: str,
    unit_pl: str,
) -> str:
    """Render a one-line-per-item overview, paged by item index to fit ``max_read_chars``.

    Args:
        header: A leading line describing the whole collection.
        lines_all: Pre-rendered per-item lines (already prefixed with ``[id]``).
        offset: Item index to start from.
        max_read_chars: Cap on the returned listing.
        unit_sg / unit_pl: Singular/plural noun for markers (``span``/``spans``).
    """
    total = len(lines_all)
    if offset < 0:
        return f"ERROR: offset {offset} is negative; use offset >= 0"
    if total and offset >= total:
        return f"ERROR: offset {offset} beyond last {unit_sg} index {total - 1}"

    lines: list[str] = []
    used, end = len(header), offset
    for i in range(offset, total):
        line = lines_all[i]
        if lines and used + len(line) + 1 > max_read_chars:
            break
        lines.append(line)
        used += len(line) + 1
        end = i + 1
    shown = f"Showing {unit_pl} {offset}-{end - 1} of {total}." if lines else f"0 {unit_pl} (of {total})."
    parts = [header, shown, *lines]
    if end < total:
        parts.append(f"[MORE: {total - end} {unit_pl} remain; call again with offset={end}]")
    return "\n".join(parts)


def search_matches(
    count: int,
    haystack_of: Callable[[int], str],
    ident_of: Callable[[int], str],
    pattern: str,
    max_matches: int,
    is_regex: bool,
    max_read_chars: int,
    *,
    unit_pl: str,
) -> str:
    """Search ``count`` items, returning matching identifiers with excerpts.

    Case-insensitive; literal by default, regex when ``is_regex`` (with ``re.MULTILINE``
    so ``^``/``$`` anchor to line boundaries in the newline-joined haystacks). Output is
    bounded by ``max_read_chars`` as well as ``max_matches``, whichever comes first.

    Args:
        count: Number of items to search (indices ``0..count-1``).
        haystack_of: Maps an item index to its searchable text.
        ident_of: Maps an item index to the identifier shown in brackets (span index or key).
        unit_pl: Plural noun for the stop markers.
    """
    if is_regex:
        try:
            # MULTILINE so ^/$ anchor to line boundaries in the newline-joined haystack —
            # LLMs write anchored regexes and would otherwise read a silent "No matches"
            # as "claim unsupported".
            rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            return f"ERROR: invalid regex {pattern!r}: {exc}. Retry with is_regex=False for a literal search."

        def matcher(text: str) -> list[tuple[int, int]]:
            return [(m.start(), m.end()) for m in rx.finditer(text)]
    else:
        needle = pattern.lower()

        def matcher(text: str) -> list[tuple[int, int]]:
            out, low, start = [], text.lower(), 0
            while (i := low.find(needle, start)) != -1:
                out.append((i, i + len(needle)))
                start = i + max(len(needle), 1)
            return out

    hits: list[str] = []
    stop_reason: str | None = None
    used = 0
    for i in range(count):
        if len(hits) >= max_matches:
            stop_reason = "max_matches"
            break
        text = haystack_of(i)
        positions = matcher(text)
        if not positions:
            continue
        s, e = positions[0]
        excerpt = preview(text[max(0, s - 60) : e + 60], 160)
        n = len(positions)
        suffix = f" ({n} matches)" if n > 1 else ""
        line = f"[{ident_of(i)}]{suffix} ...{excerpt}..."
        # Bound the whole response by max_read_chars, not max_matches alone: a generous
        # max_matches on a large collection would otherwise blow past the per-call
        # budget every other tool honors. Always keep at least one hit.
        if hits and used + len(line) + 1 > max_read_chars:
            stop_reason = "budget"
            break
        hits.append(line)
        used += len(line) + 1
    if not hits:
        return f"No matches for {pattern!r}"
    if stop_reason == "max_matches":
        hits.append(f"[stopped at {max_matches} {unit_pl}; refine the pattern or raise max_matches for more]")
    elif stop_reason == "budget":
        hits.append(
            f"[budget reached at {max_read_chars} chars ({len(hits)} {unit_pl} shown); "
            f"refine the pattern to narrow results]"
        )
    return "\n".join(hits)
