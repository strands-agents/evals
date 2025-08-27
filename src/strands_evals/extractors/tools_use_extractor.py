from strands import Agent
from strands.agent import AgentResult


def extract_agent_tools_used_from_messages(agent_messages: list) -> list[dict]:
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
            if len(message_info) > 0:
                tool = None
                for message in message_info:
                    if "toolUse" in message:
                        tool = message.get("toolUse")

                if tool:
                    tool_name = tool.get("name")
                    tool_input = tool.get("input")
                    tool_id = tool.get("toolUseId")
                    # get the tool result from the next message
                    tool_result = None
                    next_message_i = i + 1
                    while next_message_i < len(agent_messages):
                        next_message = agent_messages[next_message_i]
                        next_message_i += 1

                        if next_message.get("role") == "user":
                            content = next_message.get("content")
                            if content:
                                tool_result_dict = content[0].get("toolResult")
                                if tool_result_dict.get("toolUseId") == tool_id:
                                    tool_result_content = tool_result_dict.get("content", [])
                                    if len(tool_result_content) > 0:
                                        tool_result = tool_result_content[0].get("text")
                                        break

                    tools_used.append({"name": tool_name, "input": tool_input, "tool_result": tool_result})
    return tools_used


def extract_agent_tools_used_from_metrics(agent_result: AgentResult) -> list[dict]:
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


def extract_tools_description(agent: Agent, is_short: bool = True) -> dict[str, str]:
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
