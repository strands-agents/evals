"""Tests for goal completion assessment tool."""

from unittest.mock import MagicMock, patch

import pytest

from strands_evals.simulation.tools.goal_completion import (
    _format_conversation_for_assessment,
    get_conversation_goal_completion,
)


@pytest.fixture
def sample_conversation():
    """Fixture providing a sample conversation."""
    return [
        {"role": "user", "content": "I want to book a flight to Tokyo"},
        {"role": "assistant", "content": "I can help with that. What dates?"},
        {"role": "user", "content": "Next month"},
    ]


@pytest.fixture
def sample_conversation_with_blocks():
    """Fixture providing conversation with content blocks."""
    return [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}, {"text": "How can I help?"}]},
    ]


def test_format_conversation_simple(sample_conversation):
    """Test formatting simple conversation."""
    result = _format_conversation_for_assessment(sample_conversation)

    assert "USER: I want to book a flight to Tokyo" in result
    assert "ASSISTANT: I can help with that. What dates?" in result
    assert "USER: Next month" in result
    assert result.count("\n\n") == 2  # Two separators for three turns


def test_format_conversation_with_content_blocks(sample_conversation_with_blocks):
    """Test formatting conversation with content blocks."""
    result = _format_conversation_for_assessment(sample_conversation_with_blocks)

    assert "USER: Hello" in result
    assert "ASSISTANT: Hi there How can I help?" in result


def test_format_conversation_mixed_formats():
    """Test formatting conversation with mixed string and block formats."""
    conversation = [
        {"role": "user", "content": "String content"},
        {"role": "assistant", "content": [{"text": "Block content"}]},
    ]

    result = _format_conversation_for_assessment(conversation)

    assert "USER: String content" in result
    assert "ASSISTANT: Block content" in result


def test_format_conversation_empty_list():
    """Test formatting empty conversation raises error."""
    with pytest.raises(ValueError, match="No valid conversation turns found"):
        _format_conversation_for_assessment([])


def test_format_conversation_invalid_turn_type():
    """Test formatting conversation with invalid turn type."""
    with pytest.raises(ValueError, match="must be a dictionary"):
        _format_conversation_for_assessment(["not a dict"])


def test_format_conversation_missing_role():
    """Test formatting conversation skips turns with missing role."""
    conversation = [
        {"role": "user", "content": "Valid turn"},
        {"content": "Missing role"},
        {"role": "assistant", "content": "Another valid turn"},
    ]

    result = _format_conversation_for_assessment(conversation)

    assert "USER: Valid turn" in result
    assert "ASSISTANT: Another valid turn" in result
    assert "Missing role" not in result


def test_format_conversation_missing_content():
    """Test formatting conversation skips turns with missing content."""
    conversation = [
        {"role": "user", "content": "Valid turn"},
        {"role": "assistant"},
        {"role": "user", "content": "Another valid turn"},
    ]

    result = _format_conversation_for_assessment(conversation)

    assert "USER: Valid turn" in result
    assert "USER: Another valid turn" in result
    assert result.count("ASSISTANT") == 0


def test_format_conversation_empty_strings():
    """Test formatting conversation skips turns with empty strings."""
    conversation = [
        {"role": "user", "content": "Valid turn"},
        {"role": "", "content": "Empty role"},
        {"role": "assistant", "content": ""},
    ]

    result = _format_conversation_for_assessment(conversation)

    assert "USER: Valid turn" in result
    assert "Empty role" not in result


def test_format_conversation_whitespace_handling():
    """Test formatting conversation strips whitespace."""
    conversation = [
        {"role": "  user  ", "content": "  Content with spaces  "},
    ]

    result = _format_conversation_for_assessment(conversation)

    assert "USER: Content with spaces" in result


def test_format_conversation_invalid_content_type():
    """Test formatting conversation with invalid content type logs warning."""
    conversation = [
        {"role": "user", "content": "Valid"},
        {"role": "assistant", "content": 123},  # Invalid type
    ]

    # Should not raise, but skip invalid turn
    result = _format_conversation_for_assessment(conversation)
    assert "USER: Valid" in result


@patch("strands_evals.simulation.tools.goal_completion.Agent")
def test_get_conversation_goal_completion(mock_agent_class, sample_conversation):
    """Test goal completion assessment."""
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = MagicMock(return_value="Score: 3 - Goal fully met")
    mock_agent.return_value = mock_response
    mock_agent_class.return_value = mock_agent

    result = get_conversation_goal_completion(
        initial_goal="Book a flight to Tokyo",
        conversation=sample_conversation,
    )

    assert result == "Score: 3 - Goal fully met"
    mock_agent.assert_called_once()


@patch("strands_evals.simulation.tools.goal_completion.Agent")
def test_get_conversation_goal_completion_formats_prompt(mock_agent_class, sample_conversation):
    """Test goal completion formats prompt correctly."""
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = MagicMock(return_value="Assessment")
    mock_agent.return_value = mock_response
    mock_agent_class.return_value = mock_agent

    get_conversation_goal_completion(
        initial_goal="Test goal",
        conversation=sample_conversation,
    )

    # Verify prompt contains goal and conversation
    call_args = mock_agent.call_args[0][0]
    assert "Test goal" in call_args
    assert "USER:" in call_args
    assert "ASSISTANT:" in call_args


def test_get_conversation_goal_completion_empty_conversation():
    """Test goal completion with empty conversation raises error."""
    with pytest.raises(ValueError):
        get_conversation_goal_completion(
            initial_goal="Test goal",
            conversation=[],
        )


def test_get_conversation_goal_completion_invalid_conversation():
    """Test goal completion with invalid conversation raises error."""
    with pytest.raises(ValueError):
        get_conversation_goal_completion(
            initial_goal="Test goal",
            conversation=["not", "valid"],
        )
