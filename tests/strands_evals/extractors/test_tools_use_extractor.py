from unittest.mock import Mock

from strands_evals.extractors.tools_use_extractor import (
    extract_agent_tools_used_from_messages,
    extract_agent_tools_used_from_metrics,
    extract_tools_description,
)


def test_tools_use_extractor_extract_from_messages_with_tools():
    """Test extracting tool usage from messages with tool usage"""
    messages = [
        {"role": "user", "content": [{"text": "Calculate 2+2"}]},
        {
            "role": "assistant",
            "content": [
                {"text": "I'll calculate 2+2 for you."},
                {
                    "toolUse": {
                        "toolUseId": "tooluse_e4VwyXRiT_ymWiTKfU5cLw",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "status": "success",
                        "content": [{"text": "Result: 4"}],
                        "toolUseId": "tooluse_e4VwyXRiT_ymWiTKfU5cLw",
                    }
                }
            ],
        },
        {"role": "assistant", "content": [{"text": "The result of 2+2 is **4**."}]},
    ]

    result = extract_agent_tools_used_from_messages(messages)

    assert len(result) == 1
    assert result[0]["name"] == "calculator"
    assert result[0]["input"] == {"expression": "2+2"}
    assert result[0]["tool_result"] == "Result: 4"
    assert result[0]["is_error"] is False


def test_tools_use_extractor_extract_from_messages_with_multiple_tools():
    """Test extracting multiple tool usages from messages"""
    messages = [
        {"role": "user", "content": [{"text": "Calculate 2+2 and search for weather"}]},
        {
            "role": "assistant",
            "content": [
                {"text": "I'll calculate and search for you."},
                {
                    "toolUse": {
                        "toolUseId": "tool1",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    }
                },
                {
                    "toolUse": {
                        "toolUseId": "tool2",
                        "name": "web_search",
                        "input": {"query": "current weather"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "status": "success",
                        "content": [{"text": "Result: 4"}],
                        "toolUseId": "tool1",
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "status": "success",
                        "content": [{"text": "Sunny, 25°C"}],
                        "toolUseId": "tool2",
                    }
                }
            ],
        },
        {"role": "assistant", "content": [{"text": "Results: 4 and sunny weather."}]},
    ]

    result = extract_agent_tools_used_from_messages(messages)

    assert len(result) == 2
    assert result[0]["name"] == "calculator"
    assert result[0]["input"] == {"expression": "2+2"}
    assert result[0]["tool_result"] == "Result: 4"
    assert result[0]["is_error"] is False
    assert result[1]["name"] == "web_search"
    assert result[1]["input"] == {"query": "current weather"}
    assert result[1]["tool_result"] == "Sunny, 25°C"
    assert result[1]["is_error"] is False


def test_tools_use_extractor_extract_from_messages_no_tools():
    """Test extracting tool usage from messages without tool usage"""
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hello! How can I help you?"}]},
    ]

    result = extract_agent_tools_used_from_messages(messages)

    assert result == []


def test_tools_use_extractor_extract_from_messages_with_error():
    """Test extracting tool usage from messages with error status"""
    messages = [
        {"role": "user", "content": [{"text": "Calculate invalid"}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "tool_123", "name": "calculator", "input": {"expression": "invalid"}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "status": "error",
                        "content": [{"text": "Invalid expression"}],
                        "toolUseId": "tool_123",
                    }
                }
            ],
        },
    ]

    result = extract_agent_tools_used_from_messages(messages)

    assert len(result) == 1
    assert result[0]["name"] == "calculator"
    assert result[0]["tool_result"] == "Invalid expression"
    assert result[0]["is_error"] is True


def test_tools_use_extractor_extract_from_messages_empty():
    """Test extracting tool usage from empty messages"""
    result = extract_agent_tools_used_from_messages([])
    assert result == []


def test_tools_use_extractor_extract_from_messages_malformed():
    """Test extracting tool usage from malformed messages"""
    messages = [
        {"role": "assistant", "content": [{"text": "Response without tool"}]},
        {"role": "assistant", "content": [{"text": "Response with empty tool"}, {"toolUse": {}}]},
    ]

    result = extract_agent_tools_used_from_messages(messages)

    assert len(result) == 0


def test_tools_use_extractor_extract_from_messages_no_tool_result():
    """Test extracting tool usage when no tool result follows"""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"text": "I'll use a tool."},
                {"toolUse": {"name": "calculator", "input": {"expression": "2+2"}}},
            ],
        }
        # No following user message with tool result
    ]

    result = extract_agent_tools_used_from_messages(messages)

    assert len(result) == 1
    assert result[0]["name"] == "calculator"
    assert result[0]["input"] == {"expression": "2+2"}
    assert result[0]["tool_result"] is None
    assert result[0]["is_error"] is False


def test_tools_use_extractor_extract_from_messages_malformed_tool_result():
    """Test extracting tool usage with malformed tool result"""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"text": "I'll use a tool."},
                {"toolUse": {"name": "calculator", "input": {"expression": "2+2"}}},
            ],
        },
        {
            "role": "user",
            "content": [{"toolResult": {}}],  # Missing content field
        },
    ]

    result = extract_agent_tools_used_from_messages(messages)

    assert len(result) == 1
    assert result[0]["name"] == "calculator"
    assert result[0]["input"] == {"expression": "2+2"}
    assert result[0]["tool_result"] is None


def test_tools_use_extractor_extract_from_metrics():
    """Test extracting tool usage from agent metrics"""
    mock_agent_result = Mock()
    mock_agent_result.metrics.tool_metrics = {
        "calculator": Mock(
            tool={"toolUseId": "tooluse_lFRWxFVQTGCzzlw7dW4bAw", "name": "calculator", "input": {"expression": "4^4"}},
            call_count=3,
            success_count=3,
            error_count=0,
            total_time=0.05285072326660156,
        )
    }

    result = extract_agent_tools_used_from_metrics(mock_agent_result)

    assert len(result) == 1
    assert result[0]["name"] == "calculator"
    assert result[0]["input"] == {"expression": "4^4"}
    assert result[0]["call_count"] == 3
    assert result[0]["success_count"] == 3
    assert result[0]["total_time"] == 0.05285072326660156


def test_tools_use_extractor_extract_from_metrics_empty():
    """Test extracting tool usage from empty metrics"""
    mock_agent_result = Mock()
    mock_agent_result.metrics.tool_metrics = {}

    result = extract_agent_tools_used_from_metrics(mock_agent_result)

    assert result == []


def test_tools_use_extractor_extract_tools_description_short():
    """Test extracting tool descriptions in short format"""
    mock_agent = Mock()
    mock_agent.tool_registry.get_all_tools_config.return_value = {
        "calculator": {
            "name": "calculator",
            "description": "Performs mathematical calculations",
            "inputSchema": {"expression": "string"},
        },
        "calculator2": {
            "name": "calculator2",
            "description": "Performs mathematical calculations 2",
            "inputSchema": {"expression": "string"},
        },
    }

    result = extract_tools_description(mock_agent, is_short=True)

    assert result == {
        "calculator": "Performs mathematical calculations",
        "calculator2": "Performs mathematical calculations 2",
    }


def test_tools_use_extractor_extract_tools_description_full():
    """Test extracting tool descriptions in full format"""
    mock_agent = Mock()
    mock_config = {
        "calculator": {
            "name": "calculator",
            "description": "Performs mathematical calculations",
            "inputSchema": {"expression": "string"},
        },
        "calculator2": {
            "name": "calculator2",
            "description": "Performs mathematical calculations 2",
            "inputSchema": {"expression": "string"},
        },
    }
    mock_agent.tool_registry.get_all_tools_config.return_value = mock_config

    result = extract_tools_description(mock_agent, is_short=False)

    assert result == mock_config


def test_tools_use_extractor_extract_tools_description_empty():
    """Test extracting tool descriptions from agent with no tools"""
    mock_agent = Mock()
    mock_agent.tool_registry.get_all_tools_config.return_value = {}

    result = extract_tools_description(mock_agent, is_short=True)

    assert result == {}


def test_tools_use_extractor_extract_from_messages_user_message_without_tool_result():
    """Test extracting tool usage when user message content lacks toolResult key."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "tool_abc", "name": "calculator", "input": {"expression": "5+5"}}},
            ],
        },
        {
            "role": "user",
            "content": [{"text": "Some user text without toolResult"}],  # No toolResult key
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "status": "success",
                        "content": [{"text": "Result: 10"}],
                        "toolUseId": "tool_abc",
                    }
                }
            ],
        },
    ]
    result = extract_agent_tools_used_from_messages(messages)

    assert len(result) == 1
    assert result[0]["name"] == "calculator"
    assert result[0]["input"] == {"expression": "5+5"}
    assert result[0]["tool_result"] == "Result: 10"
    assert result[0]["is_error"] is False
