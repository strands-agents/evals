from strands.multiagent import SwarmResult


def extract_swarm_handoffs(swarm_result: SwarmResult) -> list[dict]:
    """
    Extract handoff information from swarm execution results.

    Args:
        swarm_result: Result object from swarm execution

    Returns:
        list: Handoff information with from/to agents and output messages
        [{from: str, to: str, messages: list}, ...]
    """
    hand_off_info = []
    for node_name, node_info in swarm_result.results.items():  # node_info is type NodeResult
        if isinstance(node_info.result, Exception):  # there's no agent results for exceptions
            messages = [str(node_info.result)]
            hand_off_info.append({"from": node_name, "to": None, "messages": messages})
        else:  # either a singular AgentResult or MultiAgentResult which has nested AgentResults
            agent_results = node_info.get_agent_results()
            current_node = node_name
            for agent_result in agent_results:
                added = False
                messages = [m["text"] for m in agent_result.message["content"]]
                for tool_name, tool_info in agent_result.metrics.tool_metrics.items():
                    if tool_name == "handoff_to_agent":
                        hand_off_info.append(
                            {"from": current_node, "to": tool_info.tool["input"]["agent_name"], "messages": messages}
                        )
                        added = True
                        current_node = tool_info.tool["input"]["agent_name"]  # in case of MultiAgentResult
                if not added:
                    hand_off_info.append({"from": current_node, "to": None, "messages": messages})

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
    dependencies: dict[str, list] = {}
    interactions = []
    for handoff in handoffs_info:
        if handoff["to"] not in dependencies:
            dependencies[handoff["to"]] = []
        dependencies[handoff["to"]].append(handoff["from"])
        interactions.append({"node_name": handoff["from"], "messages": handoff["messages"]})
    for i in interactions:
        node_name = i["node_name"]
        if node_name not in dependencies:
            dependencies[node_name] = []

        i["dependencies"] = dependencies[node_name]

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
