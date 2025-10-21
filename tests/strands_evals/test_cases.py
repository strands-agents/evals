import pytest

from strands_evals import Case
from strands_evals.types import Interaction


def test_create_minimal_case():
    """Test creating a basic Case with only input"""
    case = Case(input="What is 2+2?")
    assert case.input == "What is 2+2?"
    assert case.name is None
    assert case.expected_output is None
    assert case.expected_trajectory is None
    assert case.expected_interactions is None
    assert case.metadata is None


def test_create_full_case():
    """Test creating a Case with all fields. Interactions should be able to take in Interaction or dict."""
    interactions = [
        {"node_name": "math_agent", "messages": ["2x2 is 4."], "dependencies": []},
        Interaction(**{"node_name": "math_agent", "messages": ["2x2 is 4."], "dependencies": []}),
    ]
    case = Case[str, str](
        name="Math Test",
        input="What is 2+2?",
        expected_output="4",
        expected_trajectory=["calculator"],
        expected_interactions=interactions,
        metadata={"category": "math", "difficulty": "easy"},
    )

    assert case.name == "Math Test"
    assert case.input == "What is 2+2?"
    assert case.expected_output == "4"
    assert case.expected_trajectory == ["calculator"]
    assert case.expected_interactions == interactions
    assert case.metadata == {"category": "math", "difficulty": "easy"}


def test_case_with_different_types():
    """Test Case with different input/output types"""
    case = Case[int, float](input=42, expected_output=42.0)

    assert case.input == 42
    assert case.expected_output == 42.0


def test_case_with_interactions():
    """Test Case with expected_interactions with varying fields"""
    interactions = [
        {"node_name": "planner", "messages": ["plan"], "dependencies": []},
        {"messages": ["execute"]},
    ]
    case = Case[str, str](input="Complex task", expected_interactions=interactions)

    assert case.expected_interactions == interactions


def test_case_with_interactions_error():
    """Test Case with expected_interactions with weird dict where weird dict would be empty"""
    interactions = [{"hello": "planner", "hi": ["plan"]}, {"messages": ["execute"]}]
    case = Case[str, str](input="Complex task", expected_interactions=interactions)

    assert case.expected_interactions == [{}, interactions[1]]


def test_case_required_input():
    """Test that input is required"""
    with pytest.raises(ValueError):
        Case()
