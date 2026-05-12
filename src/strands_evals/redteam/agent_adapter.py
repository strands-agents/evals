"""Agent adapter for red team testing.

Extracts tool definitions from Strands Agent objects and wraps them as
Callables with tool execution trace capture. Isolates all Agent-internal
API dependencies so runner.py stays decoupled from Agent internals.
"""

import logging
from collections.abc import Callable
from typing import Any

from strands import Agent

logger = logging.getLogger(__name__)


def extract_tool_info(agent: Agent) -> dict:
    """Extract tool definitions and system prompt from an Agent as target_info.

    Returns a dict compatible with red_team(target_info=...) so existing
    goal generation works transparently with Agent targets.

    Args:
        agent: A Strands Agent instance.

    Returns:
        Dict with keys: description, system_prompt, tools (list of tool defs).
    """
    tools: list[dict[str, Any]] = []
    try:
        all_configs: Any = agent.tool_registry.get_all_tools_config()
        for tool_config in all_configs.values():
            input_schema = tool_config.get("inputSchema", {}).get("json", {})
            tools.append(
                {
                    "name": tool_config["name"],
                    "description": tool_config.get("description", ""),
                    "parameters": input_schema.get("properties", {}),
                }
            )
    except (AttributeError, KeyError, TypeError) as e:
        logger.warning("Failed to extract tools from agent: %s", e)

    return {
        "system_prompt": agent.system_prompt or "",
        "tools": tools,
        "description": f"Agent with {len(tools)} tools: {[t['name'] for t in tools]}",
    }


def wrap_agent_with_trace(agent: Agent) -> Callable[[str, list[dict] | None], str]:
    """Wrap an Agent as a Callable that captures tool execution traces.

    The returned Callable behaves like ``lambda msg: str(agent(msg))`` but
    also appends any new tool calls to a caller-supplied list. Only new
    messages produced by each call are scanned, so traces stay accurate
    when the Agent's message history persists across calls.

    The trace list is owned by the caller (typically one list per case),
    so concurrent calls on independent lists do not share mutable state.

    Args:
        agent: A Strands Agent instance.

    Returns:
        Callable ``(message, trace) -> response``. When ``trace`` is a list,
        tool uses from this call are appended as ``{"name": ..., "input": ...}``.
        When ``trace`` is None, tool calls are not recorded.
    """

    def _call(message: str, trace: list[dict] | None = None) -> str:
        messages_before = len(agent.messages)
        result = agent(message)

        if trace is not None:
            try:
                for msg in agent.messages[messages_before:]:
                    for block in msg.get("content", []):
                        if "toolUse" in block:
                            tool_use = block["toolUse"]
                            trace.append(
                                {
                                    "name": tool_use.get("name", ""),
                                    "input": tool_use.get("input", {}),
                                }
                            )
            except (AttributeError, KeyError, TypeError) as e:
                logger.debug("Failed to extract tool trace: %s", e)

        return str(result)

    return _call
