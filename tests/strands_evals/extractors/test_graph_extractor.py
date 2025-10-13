from unittest.mock import Mock

from strands_evals.extractors.graph_extractor import extract_graph_interactions


def _create_mock_node(node_id: str, messages: list[str], dependencies: list = None):
    """Helper function to create a properly mocked node"""
    mock_node = Mock()
    mock_node.node_id = node_id
    # Create a mock result - the hasattr checks in the implementation will handle validation
    mock_result = Mock()
    mock_result.result.message = {"content": [{"text": msg} for msg in messages]}
    mock_node.result = mock_result
    mock_node.dependencies = dependencies or []
    return mock_node


def test_graph_extractor_extract_interactions():
    """Test extracting interactions from graph result"""
    mock_node1 = _create_mock_node("node1", ["Message from node1"])
    mock_node2 = _create_mock_node("node2", ["Message from node2"], [mock_node1])

    mock_graph_result = Mock()
    mock_graph_result.execution_order = [mock_node1, mock_node2]

    result = extract_graph_interactions(mock_graph_result)

    assert len(result) == 2
    assert result[0]["node_name"] == "node1"
    assert result[0]["messages"] == ["Message from node1"]
    assert result[0]["dependencies"] == []
    assert result[1]["node_name"] == "node2"
    assert result[1]["messages"] == ["Message from node2"]
    assert result[1]["dependencies"] == ["node1"]


def test_graph_extractor_extract_interactions_multiple_messages():
    """Test extracting interactions with multiple messages per node"""
    mock_node = _create_mock_node("node1", ["First message", "Second message"])

    mock_graph_result = Mock()
    mock_graph_result.execution_order = [mock_node]

    result = extract_graph_interactions(mock_graph_result)

    assert len(result) == 1
    assert result[0]["node_name"] == "node1"
    assert result[0]["messages"] == ["First message", "Second message"]
    assert result[0]["dependencies"] == []


def test_graph_extractor_extract_interactions_complex_dependencies():
    """Test extracting interactions with complex dependency structure"""
    mock_node1 = _create_mock_node("node1", ["Node1 message"])
    mock_node2 = _create_mock_node("node2", ["Node2 message"])
    mock_node3 = _create_mock_node("node3", ["Node3 message"], [mock_node1, mock_node2])

    mock_graph_result = Mock()
    mock_graph_result.execution_order = [mock_node1, mock_node2, mock_node3]

    result = extract_graph_interactions(mock_graph_result)

    assert len(result) == 3
    assert result[0]["node_name"] == "node1"
    assert result[0]["dependencies"] == []
    assert result[1]["node_name"] == "node2"
    assert result[1]["dependencies"] == []
    assert result[2]["node_name"] == "node3"
    assert result[2]["dependencies"] == ["node1", "node2"]


def test_graph_extractor_extract_interactions_empty():
    """Test extracting interactions from empty graph result"""
    mock_graph_result = Mock()
    mock_graph_result.execution_order = []

    result = extract_graph_interactions(mock_graph_result)

    assert result == []


def test_graph_extractor_extract_interactions_single_node():
    """Test extracting interactions from single node graph"""
    mock_node = _create_mock_node("single_node", ["Single node message"])

    mock_graph_result = Mock()
    mock_graph_result.execution_order = [mock_node]

    result = extract_graph_interactions(mock_graph_result)

    assert len(result) == 1
    assert result[0]["node_name"] == "single_node"
    assert result[0]["messages"] == ["Single node message"]
    assert result[0]["dependencies"] == []


def test_graph_extractor_extract_interactions_empty_messages():
    """Test extracting interactions with empty message content"""
    mock_node = _create_mock_node("node1", [])

    mock_graph_result = Mock()
    mock_graph_result.execution_order = [mock_node]

    result = extract_graph_interactions(mock_graph_result)

    assert len(result) == 1
    assert result[0]["node_name"] == "node1"
    assert result[0]["messages"] == []
    assert result[0]["dependencies"] == []
