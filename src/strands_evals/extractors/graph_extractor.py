from strands.multiagent import GraphResult

from strands_evals.types import Interaction


def extract_graph_interactions(graph_result: GraphResult) -> list[Interaction]:
    """
    Extract interaction information from graph execution results.

    Each graph node is treated as a single interaction, with all individual agent messages
    within that node collected into the messages list. Dependencies are extracted from
    the graph structure.

    Args:
        graph_result: Result object from graph execution

    Returns:
        list[Interaction]: List of interactions with node names, dependencies, and messages
    """
    interactions = []
    for node in graph_result.execution_order:  # node is type GraphNode
        node_name = node.node_id
        node_messages = []
        # for node_result in node.result:
        node_result = node.result
        if node_result is not None:  # must be type NodeResult
            if isinstance(node_result.result, Exception):  # there's no agent results for exceptions
                node_messages.append(str(node_result))
            else:  # either a singular AgentResult or MultiAgentResult which has nested AgentResults
                agent_results = node_result.get_agent_results()
                for agent_result in agent_results:
                    messages = [m["text"] for m in agent_result.message["content"]]
                    node_messages.extend(messages)
        dependencies = [n.node_id for n in node.dependencies]
        interactions.append(Interaction(node_name=node_name, dependencies=dependencies, messages=node_messages))
    return interactions
