import json
import logging
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import OutputEvaluator
from strands_evals.types import EnvironmentState, EvaluationData, EvaluationOutput


@pytest.fixture
def mock_agent():
    """Mock Agent for testing"""
    agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = EvaluationOutput(score=0.8, test_pass=True, reason="Mock evaluation result")
    agent.return_value = mock_result
    return agent


@pytest.fixture
def mock_async_agent():
    """Mock Agent for testing with async"""
    agent = Mock()

    # Create a mock coroutine function
    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        mock_result.structured_output = EvaluationOutput(
            score=0.8, test_pass=True, reason="Mock async evaluation result"
        )
        return mock_result

    agent.invoke_async = mock_invoke_async
    return agent


@pytest.fixture
def evaluation_data():
    return EvaluationData(input="What is 2+2?", actual_output="4", expected_output="4", name="math_test")


def test_output_evaluator__init__with_defaults():
    """Test OutputEvaluator initialization with default values"""
    evaluator = OutputEvaluator(rubric="Test rubric")

    assert evaluator.rubric == "Test rubric"
    assert evaluator.model is None
    assert evaluator.include_inputs is True
    assert evaluator.system_prompt is not None  # Uses default template


def test_output_evaluator__init__with_custom_values():
    """Test OutputEvaluator initialization with custom values"""
    custom_prompt = "Custom system prompt"
    evaluator = OutputEvaluator(
        rubric="Custom rubric", model="gpt-4", system_prompt=custom_prompt, include_inputs=False
    )

    assert evaluator.rubric == "Custom rubric"
    assert evaluator.model == "gpt-4"
    assert evaluator.include_inputs is False
    assert evaluator.system_prompt == custom_prompt


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_with_inputs(mock_agent_class, evaluation_data, mock_agent):
    """Test evaluation with inputs included (the default behavior) and trajectory should not be included"""
    mock_agent_class.return_value = mock_agent
    evaluator = OutputEvaluator(rubric="Test rubric")

    result = evaluator.evaluate(evaluation_data)

    # Verify Agent was created with correct parameters
    mock_agent_class.assert_called_once_with(
        model=None, tools=None, system_prompt=evaluator.system_prompt, callback_handler=None
    )

    # Verify agent was called
    mock_agent.assert_called_once()
    call_args = mock_agent.call_args
    prompt = call_args[0][0]
    assert call_args[1]["structured_output_model"] == EvaluationOutput
    assert "<Input>What is 2+2?</Input>" in prompt
    assert "<Trajectory>" not in prompt
    assert "<ExpectedTrajectory>" not in prompt
    assert "<Output>4</Output>" in prompt
    assert "<ExpectedOutput>4</ExpectedOutput>" in prompt
    assert "<Rubric>Test rubric</Rubric>" in prompt

    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].test_pass is True


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_without_inputs(mock_agent_class, evaluation_data, mock_agent):
    """Test evaluation without inputs included and trajectory should not be included"""
    mock_agent_class.return_value = mock_agent
    evaluator = OutputEvaluator(rubric="Test rubric", include_inputs=False)

    result = evaluator.evaluate(evaluation_data)

    call_args = mock_agent.call_args
    prompt = call_args[0][0]
    assert "<Input>" not in prompt
    assert "<Trajectory>" not in prompt
    assert "<ExpectedTrajectory>" not in prompt
    assert "<Output>4</Output>" in prompt
    assert "<ExpectedOutput>4</ExpectedOutput>" in prompt
    assert "<Rubric>Test rubric</Rubric>" in prompt

    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].test_pass is True


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_without_expected_output(mock_agent_class, mock_agent):
    """Test evaluation without expected output"""
    mock_agent_class.return_value = mock_agent
    evaluator = OutputEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input="test",
        actual_output="result",
    )

    evaluator.evaluate(evaluation_data)

    call_args = mock_agent.call_args
    prompt = call_args[0][0]
    assert "<ExpectedOutput>" not in prompt
    assert "<Output>result</Output>" in prompt


def test_output_evaluator_evaluate_missing_actual_output():
    """Test evaluation raises exception when actual_output is missing"""
    evaluator = OutputEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(input="test", expected_output="expected")

    with pytest.raises(Exception, match="Please make sure the task function return the output"):
        evaluator.evaluate(evaluation_data)


@pytest.mark.asyncio
@patch("strands_evals.evaluators.output_evaluator.Agent")
async def test_output_evaluator_evaluate_async_with_inputs(mock_agent_class, evaluation_data, mock_async_agent):
    """Test async evaluation with inputs included"""
    mock_agent_class.return_value = mock_async_agent
    evaluator = OutputEvaluator(rubric="Test rubric")

    result = await evaluator.evaluate_async(evaluation_data)

    # Verify Agent was created with correct parameters
    mock_agent_class.assert_called_once_with(
        model=None, tools=None, system_prompt=evaluator.system_prompt, callback_handler=None
    )

    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].test_pass is True
    assert result[0].reason == "Mock async evaluation result"


@pytest.mark.asyncio
@patch("strands_evals.evaluators.output_evaluator.Agent")
async def test_output_evaluator_evaluate_async_without_inputs(mock_agent_class, evaluation_data, mock_async_agent):
    """Test async evaluation without inputs included"""
    mock_agent_class.return_value = mock_async_agent
    evaluator = OutputEvaluator(rubric="Test rubric", include_inputs=False)

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].test_pass is True


@pytest.mark.asyncio
async def test_output_evaluator_evaluate_async_missing_actual_output():
    """Test async evaluation raises exception when actual_output is missing"""
    evaluator = OutputEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(input="test", expected_output="expected")

    with pytest.raises(Exception, match="Please make sure the task function return the output"):
        await evaluator.evaluate_async(evaluation_data)


def test_output_evaluator_default_system_prompt_mentions_environment_state():
    evaluator = OutputEvaluator(rubric="Test rubric")
    assert "<ActualEnvironmentState>" in evaluator.system_prompt
    assert "<ExpectedEnvironmentState>" in evaluator.system_prompt


def test_output_evaluator__init__uses_environment_state_defaults_false():
    evaluator = OutputEvaluator(rubric="Test rubric")
    assert evaluator.uses_environment_state is False


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_includes_environment_state_in_prompt(mock_agent_class, mock_agent):
    mock_agent_class.return_value = mock_agent
    evaluator = OutputEvaluator(rubric="Test rubric", uses_environment_state=True)
    data = EvaluationData(
        input="test",
        actual_environment_state=[EnvironmentState(name="db", state={"rows": 5})],
        expected_environment_state=[EnvironmentState(name="db", state={"rows": 5})],
    )

    evaluator.evaluate(data)

    prompt = mock_agent.call_args[0][0]
    assert "<ActualEnvironmentState>" in prompt
    assert "<ExpectedEnvironmentState>" in prompt


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_environment_state_does_not_require_actual_output(mock_agent_class, mock_agent):
    mock_agent_class.return_value = mock_agent
    evaluator = OutputEvaluator(rubric="Test rubric", uses_environment_state=True)
    data = EvaluationData(
        input="test",
        actual_environment_state=[EnvironmentState(name="db", state={"rows": 5})],
    )

    # Should not raise even though actual_output is None
    evaluator.evaluate(data)


def test_output_evaluator_evaluate_environment_state_raises_without_actual_state():
    evaluator = OutputEvaluator(rubric="Test rubric", uses_environment_state=True)
    data = EvaluationData(input="test")

    with pytest.raises(Exception, match="environment_state"):
        evaluator.evaluate(data)


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_excludes_environment_state_by_default(mock_agent_class, mock_agent):
    mock_agent_class.return_value = mock_agent
    evaluator = OutputEvaluator(rubric="Test rubric")
    data = EvaluationData(
        input="test",
        actual_output="result",
        actual_environment_state=[EnvironmentState(name="db", state={"rows": 5})],
    )

    evaluator.evaluate(data)

    prompt = mock_agent.call_args[0][0]
    assert "<ActualEnvironmentState>" not in prompt


@pytest.mark.asyncio
@patch("strands_evals.evaluators.output_evaluator.Agent")
async def test_output_evaluator_evaluate_async_includes_environment_state(mock_agent_class, mock_async_agent):
    mock_agent_class.return_value = mock_async_agent
    evaluator = OutputEvaluator(rubric="Test rubric", uses_environment_state=True)
    data = EvaluationData(
        input="test",
        actual_environment_state=[EnvironmentState(name="db", state={"rows": 5})],
    )

    result = await evaluator.evaluate_async(data)

    assert len(result) == 1
    assert result[0].test_pass is True


def test_output_evaluator_init_with_tools():
    """Test OutputEvaluator initialization with custom tools"""

    def verify_claim(claim: str) -> str:
        return "verified"

    evaluator = OutputEvaluator(rubric="Test rubric", tools=[verify_claim])

    assert evaluator.tools == [verify_claim]


def test_output_evaluator_init_without_tools_defaults_to_none():
    """Test OutputEvaluator has no tools by default (current behavior preserved)"""
    evaluator = OutputEvaluator(rubric="Test rubric")

    assert evaluator.tools is None


def test_output_evaluator_to_dict_skips_non_serializable_tools(caplog):
    """Test that to_dict() output is JSON-serializable when callable tools are set (issue #373)"""

    def verify_claim(claim: str) -> str:
        return "verified"

    evaluator = OutputEvaluator(rubric="Test rubric", tools=[verify_claim])
    with caplog.at_level(logging.WARNING):
        evaluator_dict = evaluator.to_dict()

    assert "tools" not in evaluator_dict
    json.dumps(evaluator_dict)
    assert "skipping tool that cannot be written as valid utf-8 JSON" in caplog.text


def test_output_evaluator_to_dict_keeps_serializable_tools(caplog):
    """Test that serializable tools survive to_dict() while callables are skipped"""

    def verify_claim(claim: str) -> str:
        return "verified"

    evaluator = OutputEvaluator(rubric="Test rubric", tools=["my_pkg.calculator", verify_claim])
    with caplog.at_level(logging.WARNING):
        evaluator_dict = evaluator.to_dict()

    assert evaluator_dict["tools"] == ["my_pkg.calculator"]
    json.dumps(evaluator_dict)
    assert "skipping tool that cannot be written as valid utf-8 JSON" in caplog.text
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1  # the serializable tool must not warn
    assert "verify_claim" in warnings[0].getMessage()  # names which tool to re-attach


def test_output_evaluator_to_dict_skips_circular_reference_tools(caplog):
    """Test that tools raising ValueError (circular reference) are skipped, not crashed on"""
    circular: dict = {"name": "circular_tool"}
    circular["self"] = circular

    evaluator = OutputEvaluator(rubric="Test rubric", tools=[circular, "my_pkg.calculator"])
    with caplog.at_level(logging.WARNING):
        evaluator_dict = evaluator.to_dict()

    assert evaluator_dict["tools"] == ["my_pkg.calculator"]
    json.dumps(evaluator_dict)
    assert "skipping tool that cannot be written as valid utf-8 JSON" in caplog.text


def test_output_evaluator_to_dict_skips_unpaired_surrogate_tools(caplog):
    """Test that string tools with unpaired surrogates are skipped, so writing the
    experiment to a utf-8 file cannot crash on them (issue #380)"""
    evaluator = OutputEvaluator(rubric="Test rubric", tools=["my_pkg.calc_\udcff", "my_pkg.calculator"])
    with caplog.at_level(logging.WARNING):
        evaluator_dict = evaluator.to_dict()

    assert evaluator_dict["tools"] == ["my_pkg.calculator"]
    json.dumps(evaluator_dict, ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert "skipping tool that cannot be written as valid utf-8 JSON" in caplog.text


def test_output_evaluator_to_dict_skips_nan_tools(caplog):
    """Test that dict tools containing NaN or Infinity are skipped, so the serialized
    output stays valid JSON (issue #380)"""
    nan_tool = {"name": "t", "default": float("nan")}
    evaluator = OutputEvaluator(rubric="Test rubric", tools=[nan_tool, "my_pkg.calculator"])
    with caplog.at_level(logging.WARNING):
        evaluator_dict = evaluator.to_dict()

    assert evaluator_dict["tools"] == ["my_pkg.calculator"]
    json.dumps(evaluator_dict, ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert "skipping tool that cannot be written as valid utf-8 JSON" in caplog.text


def test_output_evaluator_tools_setter():
    """Test that tools can be reassigned after initialization"""

    def verify_claim(claim: str) -> str:
        return "verified"

    evaluator = OutputEvaluator(rubric="Test rubric")
    evaluator.tools = [verify_claim]

    assert evaluator.tools == [verify_claim]
    assert "tools" not in evaluator.to_dict()


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_passes_tools_to_agent(mock_agent_class, evaluation_data, mock_agent):
    """Test that custom tools are passed to the evaluator agent"""
    mock_agent_class.return_value = mock_agent

    def verify_claim(claim: str) -> str:
        return "verified"

    evaluator = OutputEvaluator(rubric="Test rubric", tools=[verify_claim])

    result = evaluator.evaluate(evaluation_data)

    mock_agent_class.assert_called_once_with(
        model=None, tools=[verify_claim], system_prompt=evaluator.system_prompt, callback_handler=None
    )
    assert result[0].score == 0.8


@pytest.mark.asyncio
@patch("strands_evals.evaluators.output_evaluator.Agent")
async def test_output_evaluator_evaluate_async_passes_tools_to_agent(
    mock_agent_class, evaluation_data, mock_async_agent
):
    """Test that custom tools are passed to the evaluator agent in async path"""
    mock_agent_class.return_value = mock_async_agent

    def verify_claim(claim: str) -> str:
        return "verified"

    evaluator = OutputEvaluator(rubric="Test rubric", tools=[verify_claim])

    result = await evaluator.evaluate_async(evaluation_data)

    mock_agent_class.assert_called_once_with(
        model=None, tools=[verify_claim], system_prompt=evaluator.system_prompt, callback_handler=None
    )
    assert result[0].score == 0.8


# --- cache_config tests ---


def test_output_evaluator_init_cache_config_defaults_to_none():
    """Test OutputEvaluator has no cache_config by default"""
    evaluator = OutputEvaluator(rubric="Test rubric")
    assert evaluator.cache_config is None


def test_output_evaluator_init_with_cache_config():
    """Test OutputEvaluator accepts a CacheConfig"""
    from strands.models.model import CacheConfig

    config = CacheConfig(strategy="auto")
    evaluator = OutputEvaluator(rubric="Test rubric", cache_config=config)
    assert evaluator.cache_config is config
    assert evaluator.cache_config.strategy == "auto"


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_creates_bedrock_model_with_cache_config(
    mock_agent_class, evaluation_data, mock_agent
):
    """Test that cache_config creates a BedrockModel when model is a string"""
    from strands.models.model import CacheConfig

    mock_agent_class.return_value = mock_agent
    config = CacheConfig(strategy="auto")
    evaluator = OutputEvaluator(rubric="Test rubric", model="us.anthropic.claude-sonnet-4-6", cache_config=config)

    with patch("strands.models.bedrock.BedrockModel") as mock_bedrock:
        mock_bedrock_instance = Mock()
        mock_bedrock.return_value = mock_bedrock_instance

        # Re-patch Agent at module level since _resolve_model imports BedrockModel
        evaluator.evaluate(evaluation_data)

    # BedrockModel should have been created in _resolve_model
    # and passed to Agent
    call_kwargs = mock_agent_class.call_args[1]
    assert call_kwargs["callback_handler"] is None


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_no_bedrock_model_when_model_is_instance(
    mock_agent_class, evaluation_data, mock_agent
):
    """Test that cache_config does not wrap an already-instantiated Model"""
    from strands.models.model import CacheConfig

    mock_agent_class.return_value = mock_agent
    config = CacheConfig(strategy="auto")
    mock_model = Mock(spec=["converse"])
    evaluator = OutputEvaluator(rubric="Test rubric", model=mock_model, cache_config=config)

    evaluator.evaluate(evaluation_data)

    # Model should be passed through unchanged (it's already a Model instance, not a string)
    call_kwargs = mock_agent_class.call_args[1]
    assert call_kwargs["model"] is mock_model


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_no_bedrock_model_without_cache_config(
    mock_agent_class, evaluation_data, mock_agent
):
    """Test that string model is passed through unchanged when no cache_config"""
    mock_agent_class.return_value = mock_agent
    evaluator = OutputEvaluator(rubric="Test rubric", model="us.anthropic.claude-sonnet-4-6")

    evaluator.evaluate(evaluation_data)

    call_kwargs = mock_agent_class.call_args[1]
    assert call_kwargs["model"] == "us.anthropic.claude-sonnet-4-6"


def test_output_evaluator_to_dict_includes_cache_config():
    """Test that to_dict serializes cache_config"""
    from strands.models.model import CacheConfig

    config = CacheConfig(strategy="auto", ttl="5m")
    evaluator = OutputEvaluator(rubric="Test rubric", cache_config=config)
    result = evaluator.to_dict()

    assert "cache_config" in result
    assert result["cache_config"]["strategy"] == "auto"
    assert result["cache_config"]["ttl"] == "5m"
    json.dumps(result)


def test_output_evaluator_to_dict_excludes_cache_config_when_none():
    """Test that to_dict omits cache_config when not set"""
    evaluator = OutputEvaluator(rubric="Test rubric")
    result = evaluator.to_dict()

    assert "cache_config" not in result


@pytest.mark.asyncio
@patch("strands_evals.evaluators.output_evaluator.Agent")
async def test_output_evaluator_evaluate_async_with_cache_config(
    mock_agent_class, evaluation_data, mock_async_agent
):
    """Test that async path also resolves model with cache_config"""
    from strands.models.model import CacheConfig

    mock_agent_class.return_value = mock_async_agent
    config = CacheConfig(strategy="auto")
    evaluator = OutputEvaluator(rubric="Test rubric", model="us.anthropic.claude-sonnet-4-6", cache_config=config)

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 0.8
