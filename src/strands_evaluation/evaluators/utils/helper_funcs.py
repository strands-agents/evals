from strands.multiagent import SwarmResult
from strands.multiagent import GraphResult
from strands import Agent

def extract_swarm_handoffs(swarm_result: SwarmResult) -> list[dict]:
    """
    Extract handoff information from swarm execution results.
    
    Args:
        swarm_result: Result object from swarm execution
        
    Returns:
        list: Handoff information with from/to agents and output messages
        [{from: str, to: str, messages: list[str]}, ...]
    """
    hand_off_info = []
    for node_name, node_info in swarm_result.results.items():
        messages = [m["text"] for m in node_info.result.message["content"]]
        added = False
        for tool_name, tool_info in node_info.result.metrics.tool_metrics.items():
            if tool_name == "handoff_to_agent":
                hand_off_info.append({
                    "from": node_name,
                    "to": tool_info.tool["input"]["agent_name"],
                    "messages": messages
                })
                added = True
        if not added:
            hand_off_info.append({
                "from": node_name,
                "to": None,
                "messages": messages
            })

    return hand_off_info

def extract_swarm_interactions_from_handoffs(handoffs_info: list[dict]) -> list[dict]:
    """
    Convert handoff information to interaction format for evaluation.
    
    Args:
        handoffs_info: List of handoff information from extract_swarm_handoffs
        
    Returns:
        list: Interactions with node names, messages, and dependencies
        [{node_name: str, messages: list[str], dependencies: list[str]}, ...]
    """
    dependencies = {}
    interactions = []
    for handoff in handoffs_info:
        if handoff['to'] not in dependencies:
            dependencies[handoff['to']] = []
        dependencies[handoff['to']].append(handoff['from'])
        interactions.append({
            "node_name": handoff['from'],
            "messages": handoff['messages']
        })
    for i in interactions:
        node_name = i["node_name"]
        if node_name not in dependencies:
            dependencies[node_name] = []
        
        i['dependencies'] = dependencies[node_name]

    return interactions

def extract_swarm_interactions(swarm_result: SwarmResult) -> list[dict]:
    """
    Extract interactions from swarm execution results.

    Args:
        swarm_result: Result object from swarm execution
    
    Returns:
        list: Interactions with node names, messages, and dependencies
        [{node_name: str, messages: list[str], dependencies: list[str]}, ...]
    """
    handoff_info = extract_swarm_handoffs(swarm_result)
    return extract_swarm_interactions_from_handoffs(handoff_info)

def extract_graph_interactions(graph_result: GraphResult):
    """
    Extract interaction information from graph execution results.
    
    Args:
        graph_result: Result object from graph execution
        
    Returns:
        list: Interactions with node names, dependencies, and messages
        [{node_name: str, dependencies: list[str], messages: list[str]}]
    """
    message_info = []
    for node in graph_result.execution_order:
        node_name = node.node_id
        node_messages = [m["text"] for m in node.result.result.message["content"]]
        dependencies = [n.node_id for n in node.dependencies]
        message_info.append({
            "node_name": node_name,
            "dependencies": dependencies,
            "messages": node_messages
        })
    return message_info

def extract_agent_tools_used_from_messages(agent_messages):
    """
    Extract tool usage information from agent message history.
    
    Args:
        agent_messages: List of message dictionaries from agent conversation
        
    Returns:
        list: Tool usage information with name, input, and associated message
        [{name: str, input: dict, message: str}, ...]
    """
    tools_used = []
    for message in agent_messages:
        if message.get("role") == "assistant":
            message_info = message.get("content")
            message_text = message_info[0].get("text")
            if len(message_info) >= 2:
                tool = message_info[1].get("toolUse")
                if tool:
                    tool_name = tool.get('name')
                    tool_input = tool.get('input')
                    tools_used.append({"name": tool_name, "input": tool_input, "message": message_text})
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
        tools_used.append({
            "name": tool_name,
            "input": tool_info.tool.get("input"),
            "call_count": tool_info.call_count,
            "success_count": tool_info.success_count,
            "total_time": tool_info.total_time,
        })
    return tools_used

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