"""User request extraction from agent sessions.

Extracts user messages from Session traces by parsing InferenceSpan and
AgentInvocationSpan. No LLM calls -- free and instant.

Ported from AgentCoreLens tools/user_request_extractor.py.
"""

from ..types.trace import AgentInvocationSpan, InferenceSpan, Session, TextContent, UserMessage


def extract_user_requests(session: Session) -> list[str]:
    """Extract user request messages from a session.

    Walks traces/spans and pulls user messages from InferenceSpan and
    AgentInvocationSpan. No LLM calls -- free and instant.

    Args:
        session: The Session object to extract from.

    Returns:
        Deduplicated list of user request strings, in order of appearance.
    """
    requests: list[str] = []
    seen: set[str] = set()

    for trace in session.traces:
        for span in trace.spans:
            if isinstance(span, AgentInvocationSpan) and span.user_prompt:
                _add_unique(span.user_prompt.strip(), requests, seen)
            elif isinstance(span, InferenceSpan):
                for msg in span.messages:
                    if isinstance(msg, UserMessage):
                        text = _extract_text(msg).strip()
                        _add_unique(text, requests, seen)

    return requests


def _extract_text(message: UserMessage) -> str:
    """Extract concatenated text from all TextContent blocks in a message."""
    return " ".join(c.text for c in message.content if isinstance(c, TextContent))


def _add_unique(text: str, results: list[str], seen: set[str]) -> None:
    """Append text to results if non-empty and not already seen."""
    if text and text not in seen:
        results.append(text)
        seen.add(text)
