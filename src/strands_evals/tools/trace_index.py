"""Progressive trace disclosure for judge agents.

A large agent trajectory does not fit in a judge's context window. Rather than
inlining the whole Session into the evaluation prompt (which overflows and gets
scored as a failure), `TraceIndex` builds a small in-memory index over the trace
and gives the judge two things:

1. An `overview()` — one line per span (index, type, tool name, sizes, truncated
   preview) — cheap enough to always fit in context, and
2. **Lookup tools** the judge calls to load only the spans it needs to verify the
   rubric: `list_spans`, `get_span`, `search_spans`.

This is the same list / get / search shape used to query any indexed collection,
and the same progressive-disclosure pattern skills use: the overview is the
"name + description" line; the tools load the full content on demand.

Example::

    from strands_evals.evaluators import TrajectoryEvaluator
    from strands_evals.tools.trace_index import TraceIndex

    index = TraceIndex(session)
    evaluator = TrajectoryEvaluator(
        rubric="Every claim in the final response must be supported by a tool result.",
        tools=index.tools,
    )
    # Compose the prompt with index.overview() instead of the full trajectory.
"""

import json
import re

from strands import tool

from ..types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    Session,
    SpanUnion,
    ToolExecutionSpan,
)

_PREVIEW_CHARS = 120
_DEFAULT_MAX_READ_CHARS = 8_000


def _flatten_spans(session: Session) -> list[SpanUnion]:
    """Flatten all spans across traces in start_time order."""
    spans = [span for trace in session.traces for span in trace.spans]
    spans.sort(key=lambda s: s.span_info.start_time)
    return spans


def _span_text(span: SpanUnion) -> str:
    """Full text content of a span, for search and retrieval."""
    if isinstance(span, ToolExecutionSpan):
        return json.dumps(
            {
                "tool_call": span.tool_call.model_dump(),
                "tool_result": span.tool_result.model_dump(),
            },
            default=str,
        )
    if isinstance(span, AgentInvocationSpan):
        return json.dumps(
            {"user_prompt": span.user_prompt, "agent_response": span.agent_response},
            default=str,
        )
    if isinstance(span, InferenceSpan):
        return json.dumps([m.model_dump() for m in span.messages], default=str)
    return json.dumps(span.model_dump(), default=str)


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _describe(span: SpanUnion) -> str:
    """One overview line describing a span without its full payload."""
    if isinstance(span, ToolExecutionSpan):
        args = json.dumps(span.tool_call.arguments, default=str)
        result_size = len(str(span.tool_result.content))
        return (
            f"TOOL {span.tool_call.name}({_preview(args, 80)}) "
            f"-> result: {result_size} chars: {_preview(str(span.tool_result.content))}"
        )
    if isinstance(span, AgentInvocationSpan):
        return (
            f"AGENT prompt: {_preview(span.user_prompt, 80)} "
            f"-> response: {len(span.agent_response)} chars: {_preview(span.agent_response)}"
        )
    if isinstance(span, InferenceSpan):
        return f"INFERENCE {len(span.messages)} messages"
    return f"{type(span).__name__}"


class TraceIndex:
    """Read-only list / get / search index over a Session for judge agents.

    Attributes:
        session: The Session being evaluated.
        max_read_chars: Cap on any single tool return, so a huge span can't
            overflow the judge's context in one call. Oversized content is
            windowed and the tool reports how to page through it.
    """

    def __init__(self, session: Session, max_read_chars: int = _DEFAULT_MAX_READ_CHARS):
        self.session = session
        self.max_read_chars = max_read_chars
        self._spans = _flatten_spans(session)

        # Bind instance state into plain functions so @tool sees clean signatures.
        # `this` (not `index`) so the public get_span(index=...) arg name is free.
        this = self

        @tool
        def list_spans() -> str:
            """List every span in the trace: one line per span with its index, type,
            tool name, argument preview, and result size. Call this first to decide
            which spans to inspect."""
            return this.overview()

        @tool
        def get_span(index: int, offset: int = 0) -> str:
            """Get the full content of one span by its index from the span list.
            Large spans are windowed; the response says how to page with offset.

            Args:
                index: Span index as shown by list_spans.
                offset: Character offset for paging through oversized spans.
            """
            if not 0 <= index < len(this._spans):
                return f"ERROR: index {index} out of range (0..{len(this._spans) - 1})"
            return this._window(_span_text(this._spans[index]), offset)

        @tool
        def search_spans(pattern: str, max_matches: int = 20) -> str:
            """Search all span content for a regex or literal string. Returns matching
            span indices with a short excerpt around each match. Use get_span to load
            a matching span in full.

            Args:
                pattern: Regex (or literal text) to search for.
                max_matches: Maximum matches to return.
            """
            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except re.error:
                rx = re.compile(re.escape(pattern), re.IGNORECASE)
            hits = []
            for i, span in enumerate(this._spans):
                text = _span_text(span)
                m = rx.search(text)
                if m:
                    start = max(0, m.start() - 60)
                    hits.append(f"[{i}] ...{_preview(text[start : m.end() + 60], 160)}...")
                if len(hits) >= max_matches:
                    break
            return "\n".join(hits) if hits else f"No matches for {pattern!r}"

        self.tools = [list_spans, get_span, search_spans]

    def overview(self) -> str:
        """Compact one-line-per-span overview of the session."""
        lines = [f"Trace overview: {len(self._spans)} spans (session {self.session.session_id})"]
        lines += [f"[{i}] {_describe(span)}" for i, span in enumerate(self._spans)]
        return "\n".join(lines)

    def _window(self, text: str, offset: int) -> str:
        if offset >= len(text):
            return f"ERROR: offset {offset} beyond content length {len(text)}"
        window = text[offset : offset + self.max_read_chars]
        if offset + len(window) < len(text):
            remaining = len(text) - offset - len(window)
            window += f"\n[TRUNCATED: {remaining} chars remain; call again with offset={offset + len(window)}]"
        return window
