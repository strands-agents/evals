from typing import Any

from strands.multiagent import GraphResult


def extract_graph_interactions(graph_result: GraphResult):
    """
    Extract interaction information from graph execution results.

    Args:
        graph_result: Result object from graph execution

    Returns:
        list: Interactions with node names, dependencies, and messages
        [{node_name: str, dependencies: list[str], messages: list[str]}]
    """
    message_info: list[dict[str, Any]] = []
    for node in graph_result.execution_order:
        # Skip nodes without results
        if node.result is None:
            continue
        # Skip if result doesn't have the expected structure
        if not hasattr(node.result, "result") or not hasattr(node.result.result, "message"):
            continue

        node_name = node.node_id
        node_messages = [m["text"] for m in node.result.result.message["content"]]
        dependencies = [n.node_id for n in node.dependencies]
        message_info.append({"node_name": node_name, "dependencies": dependencies, "messages": node_messages})
    return message_info
