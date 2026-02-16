from unittest.mock import MagicMock

import pytest
from strands.models.model import Model

from strands_evals.evaluators import Evaluator, OutputEvaluator
from strands_evals.evaluators.evaluator import DEFAULT_BEDROCK_MODEL_ID
from strands_evals.types import EvaluationData, EvaluationOutput
from strands_evals.types.trace import (
    AssistantMessage,
    TextContent,
    ToolCallContent,
    ToolResultContent,
    UserMessage,
)


class SimpleEvaluator(Evaluator[str, str]):
    """Simple implementation for testing"""

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return EvaluationOutput(score=score, test_pass=score > 0.5, reason="Test evaluation")

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        return self.evaluate(evaluation_case)


@pytest.fixture
def evaluation_data():
    return EvaluationData(input="test input", actual_output="actual", expected_output="expected", name="case")


def test_evaluator_not_implemented_evaluate(evaluation_data):
    """Test that base Evaluator raises NotImplementedError evaluate"""
    evaluator = Evaluator[str, str]()

    with pytest.raises(NotImplementedError, match="This method should be implemented in subclasses"):
        evaluator.evaluate(evaluation_data)


@pytest.mark.asyncio
async def test_evaluator_not_implemented_evaluate_async(evaluation_data):
    """Test that base Evaluator raises NotImplementedError for evaluate_async"""
    evaluator = Evaluator[str, str]()

    with pytest.raises(
        NotImplementedError,
        match="This method should be implemented in subclasses,"
        " especially if you want to run evaluations asynchronously.",
    ):
        await evaluator.evaluate_async(evaluation_data)


def test_evaluator_custom_implementation(evaluation_data):
    """Test that simple implementation works"""
    evaluator = SimpleEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert isinstance(result, EvaluationOutput)
    assert result.score == 0.0  # evaluation data has conflicting outputs
    assert result.test_pass is False
    assert result.reason == "Test evaluation"


@pytest.mark.asyncio
async def test_evaluator_custom_evaluate_async(evaluation_data):
    """Test that simple implementation works for evaluate_async"""
    evaluator = SimpleEvaluator()
    result = await evaluator.evaluate_async(evaluation_data)

    assert isinstance(result, EvaluationOutput)
    assert result.score == 0.0  # evaluation data has conflicting outputs
    assert result.test_pass is False
    assert result.reason == "Test evaluation"


def test_to_dict_evaluator():
    """Test that evaluator to_dict works properly"""
    evaluator = Evaluator()
    evaluator_dict = evaluator.to_dict()
    assert evaluator_dict["evaluator_type"] == "Evaluator"
    assert len(evaluator_dict) == 1


def test_get_model_id_with_string():
    """Test _get_model_id with string model ID"""
    evaluator = Evaluator()
    model_id = evaluator._get_model_id("anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_get_model_id_with_none():
    """Test _get_model_id with None returns default Bedrock model ID"""
    evaluator = Evaluator()
    model_id = evaluator._get_model_id(None)
    assert model_id == DEFAULT_BEDROCK_MODEL_ID


def test_get_model_id_with_model_instance():
    """Test _get_model_id with Model instance"""

    evaluator = Evaluator()
    mock_model = MagicMock(spec=Model)
    mock_model.config = {"model_id": "test-model-id"}

    model_id = evaluator._get_model_id(mock_model)
    assert model_id == "test-model-id"


def test_get_model_id_with_model_instance_no_config():
    """Test _get_model_id with Model instance without config returns empty string"""

    evaluator = Evaluator()
    mock_model = MagicMock(spec=Model)
    del mock_model.config

    model_id = evaluator._get_model_id(mock_model)
    assert model_id == ""


def test_to_dict_with_model_instance():
    """Test that to_dict properly serializes Model instances to model_id"""

    mock_model = MagicMock(spec=Model)
    mock_model.config = {"model_id": "test-model-123"}

    evaluator = OutputEvaluator(rubric="test rubric", model=mock_model)
    evaluator_dict = evaluator.to_dict()

    assert evaluator_dict["evaluator_type"] == "OutputEvaluator"
    assert evaluator_dict["rubric"] == "test rubric"
    assert "model" not in evaluator_dict
    assert evaluator_dict["model_id"] == "test-model-123"


def test_to_dict_with_string_model():
    """Test that to_dict handles string model IDs correctly"""

    evaluator = OutputEvaluator(rubric="test rubric", model="bedrock-model-id")
    evaluator_dict = evaluator.to_dict()

    assert evaluator_dict["evaluator_type"] == "OutputEvaluator"
    assert evaluator_dict["rubric"] == "test rubric"
    assert evaluator_dict["model"] == "bedrock-model-id"


def test_to_dict_with_none_model():
    """Test that to_dict handles None model correctly (uses default)"""

    evaluator = OutputEvaluator(rubric="test rubric", model=None)
    evaluator_dict = evaluator.to_dict()

    assert evaluator_dict["evaluator_type"] == "OutputEvaluator"
    assert evaluator_dict["rubric"] == "test rubric"
    assert "model" not in evaluator_dict
    assert evaluator_dict["model_id"] == DEFAULT_BEDROCK_MODEL_ID


def test_has_text_content_with_single_text_at_start():
    """Test _has_text_content with TextContent at index 0"""
    evaluator = SimpleEvaluator()
    msg = AssistantMessage(content=[TextContent(text="Hello")])

    assert evaluator._has_text_content(msg) is True


def test_has_text_content_with_text_after_tool_call():
    """Test _has_text_content with TextContent not at index 0"""
    evaluator = SimpleEvaluator()
    msg = AssistantMessage(
        content=[
            ToolCallContent(name="calculator", arguments={"x": 1}, tool_call_id="t1"),
            TextContent(text="Let me calculate that"),
        ]
    )

    assert evaluator._has_text_content(msg) is True


def test_has_text_content_with_no_text():
    """Test _has_text_content with no TextContent blocks"""
    evaluator = SimpleEvaluator()
    msg = AssistantMessage(content=[ToolCallContent(name="calculator", arguments={"x": 1}, tool_call_id="t1")])

    assert evaluator._has_text_content(msg) is False


def test_has_text_content_with_multiple_text_blocks():
    """Test _has_text_content with multiple TextContent blocks"""
    evaluator = SimpleEvaluator()
    msg = AssistantMessage(content=[TextContent(text="Hello"), TextContent(text="World")])

    assert evaluator._has_text_content(msg) is True


def test_has_text_content_with_empty_content():
    """Test _has_text_content with empty content list"""
    evaluator = SimpleEvaluator()
    msg = AssistantMessage(content=[])

    assert evaluator._has_text_content(msg) is False


def test_extract_text_content_single_block():
    """Test _extract_text_content with single TextContent block"""
    evaluator = SimpleEvaluator()
    msg = UserMessage(content=[TextContent(text="Hello world")])

    result = evaluator._extract_text_content(msg)
    assert result == "Hello world"


def test_extract_text_content_multiple_blocks():
    """Test _extract_text_content with multiple TextContent blocks"""
    evaluator = SimpleEvaluator()
    msg = AssistantMessage(content=[TextContent(text="Hello"), TextContent(text="world")])

    result = evaluator._extract_text_content(msg)
    assert result == "Hello world"


def test_extract_text_content_text_not_at_start():
    """Test _extract_text_content with TextContent not at index 0"""
    evaluator = SimpleEvaluator()
    msg = AssistantMessage(
        content=[
            ToolCallContent(name="calculator", arguments={"x": 1}, tool_call_id="t1"),
            TextContent(text="Let me calculate"),
        ]
    )

    result = evaluator._extract_text_content(msg)
    assert result == "Let me calculate"


def test_extract_text_content_mixed_content():
    """Test _extract_text_content with mixed content types"""
    evaluator = SimpleEvaluator()
    msg = AssistantMessage(
        content=[
            TextContent(text="First text"),
            ToolCallContent(name="calculator", arguments={"x": 1}, tool_call_id="t1"),
            TextContent(text="Second text"),
        ]
    )

    result = evaluator._extract_text_content(msg)
    assert result == "First text Second text"


def test_extract_text_content_no_text():
    """Test _extract_text_content with no TextContent blocks"""
    evaluator = SimpleEvaluator()
    msg = AssistantMessage(content=[ToolCallContent(name="calculator", arguments={"x": 1}, tool_call_id="t1")])

    result = evaluator._extract_text_content(msg)
    assert result == ""


def test_extract_text_content_empty_content():
    """Test _extract_text_content with empty content list"""
    evaluator = SimpleEvaluator()
    msg = UserMessage(content=[])

    result = evaluator._extract_text_content(msg)
    assert result == ""


def test_extract_text_content_user_message_with_tool_result():
    """Test _extract_text_content with UserMessage containing tool results and text"""
    evaluator = SimpleEvaluator()
    msg = UserMessage(
        content=[
            ToolResultContent(content="Result: 42", tool_call_id="t1"),
            TextContent(text="Here's the result"),
        ]
    )

    result = evaluator._extract_text_content(msg)
    assert result == "Here's the result"
