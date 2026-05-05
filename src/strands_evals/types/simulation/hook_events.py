from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreCallHookEvent:
    """
    Event passed to pre_call_hook before the LLM generates a tool response.

    Attributes:
        tool_name: Name of the tool being called.
        parameters: Parsed parameters for the tool call.
        state_key: Key for the state (tool_name or share_state_id).
        previous_calls: List of previous tool call records from the state registry.
    """

    tool_name: str
    parameters: dict[str, Any]
    state_key: str
    previous_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PostCallHookEvent:
    """
    Event passed to post_call_hook after the LLM generates a tool response.

    Attributes:
        tool_name: Name of the tool that was called.
        parameters: Parsed parameters for the tool call.
        state_key: Key for the state (tool_name or share_state_id).
        response: The LLM-generated response dict, which the hook may modify.
    """

    tool_name: str
    parameters: dict[str, Any]
    state_key: str
    response: dict[str, Any] = field(default_factory=dict)
