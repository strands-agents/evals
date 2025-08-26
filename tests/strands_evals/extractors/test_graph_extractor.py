from unittest.mock import Mock

from strands_evals.extractors.graph_extractor import extract_graph_interactions


def test_graph_extractor_extract_interactions():
    """Test extracting interactions from graph result"""
    # Create mock agent results
    mock_agent_result1 = Mock()
    mock_agent_result1.message = {"content": [{"text": "Message from node1"}]}

    mock_agent_result2 = Mock()
    mock_agent_result2.message = {"content": [{"text": "Message from node2"}]}

    # Create mock node results
    mock_node_result1 = Mock()
    mock_node_result1.get_agent_results.return_value = [mock_agent_result1]

    mock_node_result2 = Mock()
    mock_node_result2.get_agent_results.return_value = [mock_agent_result2]

    # Create mock graph nodes
    mock_node1 = Mock()
    mock_node1.node_id = "node1"
    mock_node1.result = mock_node_result1
    mock_node1.dependencies = []

    mock_node2 = Mock()
    mock_node2.node_id = "node2"
    mock_node2.result = mock_node_result2
    mock_node2.dependencies = [mock_node1]

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
    mock_agent_result = Mock()
    mock_agent_result.message = {"content": [{"text": "First message"}, {"text": "Second message"}]}

    mock_node_result = Mock()
    mock_node_result.get_agent_results.return_value = [mock_agent_result]

    mock_node = Mock()
    mock_node.node_id = "node1"
    mock_node.result = mock_node_result
    mock_node.dependencies = []

    mock_graph_result = Mock()
    mock_graph_result.execution_order = [mock_node]

    result = extract_graph_interactions(mock_graph_result)

    assert len(result) == 1
    assert result[0]["node_name"] == "node1"
    assert result[0]["messages"] == ["First message", "Second message"]
    assert result[0]["dependencies"] == []


def test_graph_extractor_extract_interactions_complex_dependencies():
    """Test extracting interactions with complex dependency structure"""
    # Create mock agent results
    mock_agent_result1 = Mock()
    mock_agent_result1.message = {"content": [{"text": "Node1 message"}]}

    mock_agent_result2 = Mock()
    mock_agent_result2.message = {"content": [{"text": "Node2 message"}]}

    mock_agent_result3 = Mock()
    mock_agent_result3.message = {"content": [{"text": "Node3 message"}]}

    # Create mock node results
    mock_node_result1 = Mock()
    mock_node_result1.get_agent_results.return_value = [mock_agent_result1]

    mock_node_result2 = Mock()
    mock_node_result2.get_agent_results.return_value = [mock_agent_result2]

    mock_node_result3 = Mock()
    mock_node_result3.get_agent_results.return_value = [mock_agent_result3]

    # Create mock graph nodes
    mock_node1 = Mock()
    mock_node1.node_id = "node1"
    mock_node1.result = mock_node_result1
    mock_node1.dependencies = []

    mock_node2 = Mock()
    mock_node2.node_id = "node2"
    mock_node2.result = mock_node_result2
    mock_node2.dependencies = []

    mock_node3 = Mock()
    mock_node3.node_id = "node3"
    mock_node3.result = mock_node_result3
    mock_node3.dependencies = [mock_node1, mock_node2]

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
    mock_agent_result = Mock()
    mock_agent_result.message = {"content": [{"text": "Single node message"}]}

    mock_node_result = Mock()
    mock_node_result.get_agent_results.return_value = [mock_agent_result]

    mock_node = Mock()
    mock_node.node_id = "single_node"
    mock_node.result = mock_node_result
    mock_node.dependencies = []

    mock_graph_result = Mock()
    mock_graph_result.execution_order = [mock_node]

    result = extract_graph_interactions(mock_graph_result)

    assert len(result) == 1
    assert result[0]["node_name"] == "single_node"
    assert result[0]["messages"] == ["Single node message"]
    assert result[0]["dependencies"] == []


def test_graph_extractor_extract_interactions_empty_messages():
    """Test extracting interactions with empty message content"""
    mock_agent_result = Mock()
    mock_agent_result.message = {"content": []}

    mock_node_result = Mock()
    mock_node_result.get_agent_results.return_value = [mock_agent_result]

    mock_node = Mock()
    mock_node.node_id = "node1"
    mock_node.result = mock_node_result
    mock_node.dependencies = []

    mock_graph_result = Mock()
    mock_graph_result.execution_order = [mock_node]

    result = extract_graph_interactions(mock_graph_result)

    assert len(result) == 1
    assert result[0]["node_name"] == "node1"
    assert result[0]["messages"] == []
    assert result[0]["dependencies"] == []


def test_graph_extractor_extract_interactions_with_exception():
    """Test extracting interactions when node result is an Exception"""
    mock_exception_result = Exception("Node execution failed")
    mock_node_result = Mock()
    mock_node_result.result = mock_exception_result

    mock_node = Mock()
    mock_node.node_id = "failed_node"
    mock_node.result = mock_node_result
    mock_node.dependencies = []

    mock_graph_result = Mock()
    mock_graph_result.execution_order = [mock_node]

    result = extract_graph_interactions(mock_graph_result)

    assert len(result) == 1
    assert result[0]["node_name"] == "failed_node"
    assert result[0]["messages"] == [str(mock_node_result)]
    assert result[0]["dependencies"] == []


def test_graph_extractor_extract_interactions_mixed_results():
    """Test extracting interactions with both normal and exception results"""
    # Normal node
    mock_agent_result = Mock()
    mock_agent_result.message = {"content": [{"text": "Success message"}]}

    mock_normal_node_result = Mock()
    mock_normal_node_result.get_agent_results.return_value = [mock_agent_result]

    mock_normal_node = Mock()
    mock_normal_node.node_id = "normal_node"
    mock_normal_node.result = mock_normal_node_result
    mock_normal_node.dependencies = []

    # Exception node
    mock_exception_result = Exception("Processing error")
    mock_exception_node_result = Mock()
    mock_exception_node_result.result = mock_exception_result

    mock_exception_node = Mock()
    mock_exception_node.node_id = "exception_node"
    mock_exception_node.result = mock_exception_node_result
    mock_exception_node.dependencies = [mock_normal_node]

    mock_graph_result = Mock()
    mock_graph_result.execution_order = [mock_normal_node, mock_exception_node]

    result = extract_graph_interactions(mock_graph_result)

    assert len(result) == 2
    assert result[0]["node_name"] == "normal_node"
    assert result[0]["messages"] == ["Success message"]
    assert result[0]["dependencies"] == []
    assert result[1]["node_name"] == "exception_node"
    assert result[1]["messages"] == [str(mock_exception_node_result)]
    assert result[1]["dependencies"] == ["normal_node"]
