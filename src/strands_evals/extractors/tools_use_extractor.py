from typing import Union, cast

from strands import Agent

from ..types.trace import Session, ToolLevelInput
from .trace_extractor import TraceExtractor


def extract_agent_tools_used_from_messages(agent_messages):
    """
    Extract tool usage information from agent message history.

    Args:
        agent_messages: List of message dictionaries from agent conversation

    Returns:
        list: Tool usage information with name, input, and tool_result
        [{name: str, input: dict, tool_result: str}, ...]
    """
    tools_used = []
    for i, message in enumerate(agent_messages):
        if message.get("role") == "assistant":
            message_info = message.get("content")
            if message_info:
                # Collect tool uses from this message
                tools = [cb.get("toolUse") for cb in message_info if cb.get("toolUse")]
                if not tools:
                    continue

                # Build lookup dict of tool results from subsequent user messages
                tool_ids_needed = {tool.get("toolUseId") for tool in tools}
                tool_results_by_id: dict[str, dict] = {}
                for next_message in agent_messages[i + 1 :]:
                    if next_message.get("role") == "user":
                        for content_block in next_message.get("content") or []:
                            tool_result_dict = content_block.get("toolResult")
                            if tool_result_dict:
                                tool_id = tool_result_dict.get("toolUseId")
                                if tool_id in tool_ids_needed and tool_id not in tool_results_by_id:
                                    tool_results_by_id[tool_id] = tool_result_dict
                    if len(tool_results_by_id) == len(tool_ids_needed):
                        break

                for tool in tools:
                    tool_name = tool.get("name")
                    tool_input = tool.get("input")
                    tool_id = tool.get("toolUseId")
                    tool_result = None
                    is_error = False

                    # Find the matching tool result block
                    tool_result_dict = tool_results_by_id.get(tool_id)
                    if tool_result_dict:
                        tool_result_content = tool_result_dict.get("content", [])
                        for result_item in tool_result_content:
                            if isinstance(result_item, dict) and "text" in result_item:
                                tool_result = result_item.get("text")
                                break
                        is_error = tool_result_dict.get("status") == "error"

                    tools_used.append(
                        {"name": tool_name, "input": tool_input, "tool_result": tool_result, "is_error": is_error}
                    )
    return tools_used


def extract_agent_tools_used_from_metrics(agent_result):
    """
    Extract tool usage metrics from agent execution result.

    Args:
        agent_result: Agent result object containing metrics

    Returns:
        list: Tool metrics with name, input, counts, and timing
        [{
            name: str,
            input: dict,
            call_count: int,
            success_count: int,
            total_time: float
        }, ...]
    """
    tool_metrics = agent_result.metrics.tool_metrics
    tools_used = []
    for tool_name, tool_info in tool_metrics.items():
        tools_used.append(
            {
                "name": tool_name,
                "input": tool_info.tool.get("input"),
                "call_count": tool_info.call_count,
                "success_count": tool_info.success_count,
                "total_time": tool_info.total_time,
            }
        )
    return tools_used


def extract_agent_tools_used_from_trace(session: Session) -> list[dict]:
    """
    Extract tool usage information from trace data (Session object).
    This function uses TraceExtractor to parse the session at TOOL_LEVEL,
    then transforms the ToolLevelInput objects into the same format as
    extract_agent_tools_used_from_messages for consistency.

    Args:
        session: Session object containing trace data

    Returns:
        list: Tool usage information with name, input, and tool_result
        [{name: str, input: dict, tool_result: str}, ...]
    """
    from ..types.trace import EvaluationLevel

    # Use TraceExtractor to get tool-level inputs
    extractor = TraceExtractor(evaluation_level=EvaluationLevel.TOOL_LEVEL)
    tool_inputs = cast(list[ToolLevelInput], extractor.extract(session))

    # Transform to the same format as message-based extraction
    tools_used = []
    for tool_input in tool_inputs:
        tool_execution = tool_input.tool_execution_details
        tool_name = tool_execution.tool_call.name
        tool_input_args = tool_execution.tool_call.arguments
        tool_result = tool_execution.tool_result.content if tool_execution.tool_result else None

        tools_used.append({"name": tool_name, "input": tool_input_args, "tool_result": tool_result})

    return tools_used


def extract_agent_tools_used(source: Union[list, Session]) -> list[dict]:
    """
    Extract tool usage information from either agent messages or trace data.
    This is a unified interface that automatically detects the input type and uses
    the appropriate extraction method:
    - If source is a Session object, uses trace-based extraction
    - If source is a list, uses message-based extraction

    Args:
        source: Either agent_messages (list) or Session object

    Returns:
        list: Tool usage information with name, input, and tool_result
        [{name: str, input: dict, tool_result: str}, ...]
    Raises:
        TypeError: If source is neither a list nor a Session object
    """
    if isinstance(source, Session):
        return extract_agent_tools_used_from_trace(source)
    elif isinstance(source, list):
        return extract_agent_tools_used_from_messages(source)
    else:
        raise TypeError(f"source must be either a list (agent messages) or Session object, got {type(source).__name__}")


def extract_tools_description(agent: Agent, is_short: bool = True):
    """
    Extract a dictionary of all tools used in a given agent.

    Args:
        agent (Agent): Target agent to extract tool registry from
        is_short (bool, optional): Whether to return only the description of the tools or everything. Defaults to True.

    Returns:
        dict: Tool name and its corresponding description
        {<tool_name>: <tool_description>, ...}
    """
    description = agent.tool_registry.get_all_tools_config()
    if is_short:
        shorten_descrip = {}
        for tool_name, tool_info in description.items():
            shorten_descrip[tool_name] = tool_info["description"]
        return shorten_descrip

    return description
