"""Progressive trace disclosure for judge agents.

A large agent trajectory does not fit in a judge's context window. Rather than
inlining the whole Session into the evaluation prompt (which overflows and gets
scored as a failure), `TraceIndex` builds a small in-memory index over the trace
and gives the judge two things:

1. An `overview()` — one line per span (index, type, tool name, sizes, truncated
   preview) — paged so it always fits in context, and
2. **Lookup tools** the judge calls to load only the spans it needs to verify the
   rubric: `list_spans`, `get_span`, `search_spans`.

This is the same list / get / search shape used to query any indexed collection:
the overview is the "name + description" line; the tools load the full content on
demand.

`TraceIndex` composes with `OutputEvaluator`, whose prompt is caller-controlled —
put the overview in the judged output and pass the tools. It does **not** compose
with `TrajectoryEvaluator`, which inlines the full `actual_trajectory`
unconditionally and so re-creates the overflow this class exists to prevent.

Example::

    from strands_evals.evaluators import OutputEvaluator
    from strands_evals.tools.trace_index import TraceIndex

    index = TraceIndex(session)
    prompt_section, tools = index.for_judge()  # overview + tools together
    evaluator = OutputEvaluator(
        rubric=(
            "Every claim in the final response must be supported by a tool result. "
            "Use the trace tools to verify each claim before scoring."
        ),
        tools=tools,
    )
    # Put the compact overview next to the answer instead of the full trajectory:
    judged_output = f"{agent_answer}\n{prompt_section}"
"""

import json

from strands import tool

from ..extractors.trace_extractor import _to_aware_utc
from ..types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanUnion,
    TextContent,
    ToolCallContent,
    ToolExecutionSpan,
    ToolResultContent,
    UserMessage,
)
from ._progressive import DEFAULT_MAX_READ_CHARS, paged_listing, preview, search_matches, window


def _flatten_spans(session: Session) -> list[SpanUnion]:
    """Flatten all spans across traces in start_time order.

    Timestamps are normalized to timezone-aware UTC before sorting so a mix of
    naive and aware `start_time` values (produced by different mappers) can't
    raise `TypeError` at construction time.
    """
    spans = [span for trace in session.traces for span in trace.spans]
    spans.sort(key=lambda s: _to_aware_utc(s.span_info.start_time))
    return spans


def _span_text(span: SpanUnion) -> str:
    """Full text content of a span, for retrieval via get_span."""
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
            {
                "user_prompt": span.user_prompt,
                "agent_response": span.agent_response,
                "system_prompt": span.system_prompt,
                "available_tools": [t.model_dump() for t in span.available_tools],
            },
            default=str,
        )
    if isinstance(span, InferenceSpan):
        return json.dumps([m.model_dump() for m in span.messages], default=str)
    return json.dumps(span.model_dump(), default=str)


def _render_message(msg: UserMessage | AssistantMessage) -> str:
    """Plain-text rendering of one inference message from its fields.

    Renders role + each content block by field (text, or tool name/args/result/
    error) like the tool and agent branches, rather than dumping the pydantic dict.
    A raw `model_dump()` repr leaks structural keys such as ``'error': None`` into the
    text, which turns an "error" search into a false positive on every inference span
    and hides genuine failures behind serialization noise.
    """
    parts = [msg.role.value]
    for block in msg.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolCallContent):
            parts.append(f"{block.name}({json.dumps(block.arguments, default=str)})")
        elif isinstance(block, ToolResultContent):
            parts.append(str(block.content))
            if block.error:
                parts.append(f"error: {block.error}")
    return " ".join(p for p in parts if p)


def _search_haystack(span: SpanUnion) -> str:
    """Plain-text rendering of a span for search matching.

    Unlike `_span_text`, this joins the raw field values without JSON escaping so
    a literal a judge copies from the overview (``$150``, ``refund_amount``) matches
    the bytes it sees, and every human-visible field — including `system_prompt`
    and `available_tools` — is searchable.
    """
    if isinstance(span, ToolExecutionSpan):
        parts = [
            span.tool_call.name,
            json.dumps(span.tool_call.arguments, default=str),
            str(span.tool_result.content),
        ]
        if span.tool_result.error:
            # Prefix with an "error:" token so a judge searching the overview's
            # [ERROR] vocabulary finds the failed tool; the raw value alone
            # (e.g. "CardDeclined: insufficient funds") carries no such token.
            parts.append(f"error: {span.tool_result.error}")
        return "\n".join(parts)
    if isinstance(span, AgentInvocationSpan):
        parts = [span.user_prompt, span.agent_response]
        if span.system_prompt:
            parts.append(span.system_prompt)
        parts += [f"{t.name}: {t.description or ''}" for t in span.available_tools]
        return "\n".join(parts)
    if isinstance(span, InferenceSpan):
        return "\n".join(_render_message(m) for m in span.messages)
    return str(span.model_dump())


def _describe(span: SpanUnion) -> str:
    """One overview line describing a span without its full payload."""
    if isinstance(span, ToolExecutionSpan):
        args = json.dumps(span.tool_call.arguments, default=str)
        result_size = len(str(span.tool_result.content))
        status = "ERROR" if span.tool_result.error else "ok"
        line = (
            f"TOOL {span.tool_call.name}({preview(args, 80)}) "
            f"-> [{status}] result: {result_size} chars: {preview(str(span.tool_result.content))}"
        )
        if span.tool_result.error:
            line += f" | error: {preview(str(span.tool_result.error), 80)}"
        return line
    if isinstance(span, AgentInvocationSpan):
        return (
            f"AGENT prompt: {preview(span.user_prompt, 80)} "
            f"-> response: {len(span.agent_response)} chars: {preview(span.agent_response)}"
        )
    if isinstance(span, InferenceSpan):
        rendered = [_render_message(m) for m in span.messages]
        size = sum(len(r) for r in rendered)
        return f"INFERENCE {len(span.messages)} messages, {size} chars: {preview(' '.join(rendered))}"
    return f"{type(span).__name__}"


class TraceIndex:
    """Read-only list / get / search index over a Session for judge agents.

    Attributes:
        session: The Session being evaluated.
        max_read_chars: Cap on any single tool return, so a large span or a long
            overview can't overflow the judge's context in one call. Oversized
            content is windowed and the tool reports how to page through it.
    """

    def __init__(self, session: Session, max_read_chars: int = DEFAULT_MAX_READ_CHARS):
        if max_read_chars < 1:
            raise ValueError(f"max_read_chars must be >= 1, got {max_read_chars}")
        self.session = session
        self.max_read_chars = max_read_chars
        self._spans = _flatten_spans(session)
        # Precompute per-span overview lines once (recomputing per list_spans call
        # is measurable on large sessions).
        self._describe_lines = [f"[{i}] {_describe(span)}" for i, span in enumerate(self._spans)]

        # Bind instance state into plain functions so @tool sees clean signatures.
        # `this` (not `index`) so the public get_span(index=...) arg name is free.
        this = self

        @tool
        def list_spans(offset: int = 0) -> str:
            """List spans in the trace: one line per span with its index, type, tool
            name, argument preview, and result size. Call this first to decide which
            spans to inspect. Long traces are paged; the response says how to page with
            offset. Previews are truncated — load a span with get_span before you rely
            on its content to score.

            Args:
                offset: Span index to start the listing from, for paging long traces.
            """
            return this.overview(offset)

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
            return window(_span_text(this._spans[index]), offset, this.max_read_chars)

        @tool
        def search_spans(pattern: str, max_matches: int = 20, is_regex: bool = False) -> str:
            """Search all span content for a literal string (default) or a regex.
            Matching is case-insensitive and covers every visible field, including
            system prompts and tool configs. Regex anchors ^/$ match line boundaries.
            Returns matching span indices with a short excerpt and per-span match
            count, capped at max_read_chars total; use get_span to load a match in full.

            Args:
                pattern: Text to search for. Treated literally unless is_regex=True.
                max_matches: Maximum number of matching spans to return. Results are
                    also capped at max_read_chars total, whichever comes first.
                is_regex: Set True to treat pattern as a regular expression.
            """
            return search_matches(
                len(this._spans),
                lambda i: _search_haystack(this._spans[i]),
                str,
                pattern,
                max_matches,
                is_regex,
                this.max_read_chars,
                unit_pl="spans",
            )

        self.tools = [list_spans, get_span, search_spans]

    def overview(self, offset: int = 0) -> str:
        """Compact one-line-per-span overview of the session, paged by span index.

        Args:
            offset: Span index to start from. The listing is capped at
                `max_read_chars`; if it doesn't fit, the response says the next offset.
        """
        total = len(self._spans)
        header = (
            f"Trace overview: {total} spans (session {self.session.session_id}). "
            f"Previews are truncated; call get_span/search_spans to load full content "
            f"(up to {self.max_read_chars} chars per call) and verify claims before scoring."
        )
        return paged_listing(
            header,
            self._describe_lines,
            offset,
            self.max_read_chars,
            unit_sg="span",
            unit_pl="spans",
        )

    def for_judge(self) -> tuple[str, list]:
        """Return the two pieces a judge needs, together, so neither is forgotten.

        Composing a `TraceIndex` into an evaluator has two halves — the overview must
        go into the judged output, and the discovery tools must be passed to the
        evaluator — and doing only one silently degrades the judge (previews with no
        way to drill in, or a bare answer with no trace map). This hands back both:

            prompt_section, tools = index.for_judge()
            evaluator = OutputEvaluator(rubric="...", tools=tools)
            output = f"{agent_answer}\n{prompt_section}"
            evaluator.evaluate(EvaluationData(input=..., actual_output=output))

        Returns:
            A ``(prompt_section, tools)`` pair. ``prompt_section`` is the overview
            wrapped in a ``<TraceOverview>`` block ready to concatenate onto the judged
            output; ``tools`` is `self.tools`.
        """
        prompt_section = f"<TraceOverview>\n{self.overview()}\n</TraceOverview>"
        return prompt_section, self.tools
