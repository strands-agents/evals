from unittest.mock import Mock, patch

import pytest

from strands_evals.plugins import EvaluationPlugin
from strands_evals.plugins.evaluation_plugin import ImprovementSuggestion
from strands_evals.types import EvaluationData, EvaluationOutput


@pytest.fixture
def mock_evaluator():
    evaluator = Mock()
    return evaluator


class FakeAgent:
    """Minimal agent class for testing __class__ swap without real Agent internals."""

    def __init__(self):
        self.system_prompt = "original prompt"
        self.messages: list = []
        self._result = Mock()
        self._result.__str__ = Mock(return_value="mock output")
        self.call_count = 0

    def __call__(self, prompt=None, **kwargs):
        self.call_count += 1
        return self._result


@pytest.fixture
def mock_agent():
    return FakeAgent()


def test_plugin_name(mock_evaluator):
    plugin = EvaluationPlugin(evaluators=[mock_evaluator])
    assert plugin.name == "strands-evals"


def test_init_with_defaults(mock_evaluator):
    plugin = EvaluationPlugin(evaluators=[mock_evaluator])
    assert plugin._evaluators == [mock_evaluator]
    assert plugin._max_retries == 1
    assert plugin._expected_output is None
    assert plugin._expected_trajectory is None
    assert plugin._model is None


def test_init_with_custom_values(mock_evaluator):
    plugin = EvaluationPlugin(
        evaluators=[mock_evaluator],
        max_retries=3,
        expected_output="expected",
        expected_trajectory=["step1", "step2"],
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    )
    assert plugin._evaluators == [mock_evaluator]
    assert plugin._max_retries == 3
    assert plugin._expected_output == "expected"
    assert plugin._expected_trajectory == ["step1", "step2"]
    assert plugin._model == "us.anthropic.claude-sonnet-4-20250514-v1:0"


# --- Step 2: init_agent + __class__ swap ---


def test_init_agent_wraps_call(mock_evaluator, mock_agent):
    """After init_agent, agent.__class__ should be a subclass of the original."""
    original_class = mock_agent.__class__
    plugin = EvaluationPlugin(evaluators=[mock_evaluator])
    plugin.init_agent(mock_agent)
    assert mock_agent.__class__ is not original_class
    assert issubclass(mock_agent.__class__, original_class)


def test_init_agent_preserves_isinstance(mock_evaluator, mock_agent):
    """isinstance(agent, OriginalClass) should still be True after wrapping."""
    plugin = EvaluationPlugin(evaluators=[mock_evaluator])
    plugin.init_agent(mock_agent)
    assert isinstance(mock_agent, FakeAgent)


def test_wrapped_call_invokes_original(mock_evaluator, mock_agent):
    """Calling agent(prompt) after wrapping should still invoke the original agent logic."""
    mock_evaluator.evaluate.return_value = [Mock(test_pass=True)]
    plugin = EvaluationPlugin(evaluators=[mock_evaluator], max_retries=0)
    plugin.init_agent(mock_agent)

    result = mock_agent("test prompt")

    assert result is not None


# --- Step 3: Evaluation execution on invocation ---


def test_evaluators_run_after_invocation(mock_agent):
    """Evaluators should be called with EvaluationData after invocation."""
    evaluator = Mock()
    evaluator.evaluate.return_value = [EvaluationOutput(score=1.0, test_pass=True, reason="good")]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=0)
    plugin.init_agent(mock_agent)

    mock_agent("What is 2+2?")

    evaluator.evaluate.assert_called_once()
    eval_data = evaluator.evaluate.call_args[0][0]
    assert isinstance(eval_data, EvaluationData)
    assert eval_data.input == "What is 2+2?"
    assert eval_data.actual_output == str(mock_agent._result)


def test_evaluation_data_uses_constructor_expected(mock_agent):
    """EvaluationData should use expected values from constructor."""
    evaluator = Mock()
    evaluator.evaluate.return_value = [EvaluationOutput(score=1.0, test_pass=True)]
    plugin = EvaluationPlugin(
        evaluators=[evaluator],
        max_retries=0,
        expected_output="4",
        expected_trajectory=["calculator"],
    )
    plugin.init_agent(mock_agent)

    mock_agent("What is 2+2?")

    eval_data = evaluator.evaluate.call_args[0][0]
    assert eval_data.expected_output == "4"
    assert eval_data.expected_trajectory == ["calculator"]


def test_evaluation_data_uses_invocation_state_expected(mock_agent):
    """invocation_state expected values should override constructor values."""
    evaluator = Mock()
    evaluator.evaluate.return_value = [EvaluationOutput(score=1.0, test_pass=True)]
    plugin = EvaluationPlugin(
        evaluators=[evaluator],
        max_retries=0,
        expected_output="constructor_value",
    )
    plugin.init_agent(mock_agent)

    mock_agent("prompt", invocation_state={"expected_output": "state_value"})

    eval_data = evaluator.evaluate.call_args[0][0]
    assert eval_data.expected_output == "state_value"


def test_passing_evaluation_returns_immediately(mock_agent):
    """When evaluations pass, result should be returned without retry."""
    evaluator = Mock()
    evaluator.evaluate.return_value = [EvaluationOutput(score=1.0, test_pass=True)]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=3)
    plugin.init_agent(mock_agent)

    result = mock_agent("prompt")

    assert result is mock_agent._result
    evaluator.evaluate.assert_called_once()


# --- Step 4: Retry on failure ---


def test_retry_on_failure(mock_agent):
    """Agent should be re-invoked when evaluation fails."""
    evaluator = Mock()
    evaluator.evaluate.side_effect = [
        [EvaluationOutput(score=0.0, test_pass=False, reason="bad")],
        [EvaluationOutput(score=1.0, test_pass=True, reason="good")],
    ]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=1)
    plugin._suggest_improvements = Mock(
        return_value=ImprovementSuggestion(reasoning="needs improvement", system_prompt="improved prompt")
    )
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    assert mock_agent.call_count == 2
    assert evaluator.evaluate.call_count == 2


def test_max_retries_respected(mock_agent):
    """Should stop retrying after max_retries attempts."""
    evaluator = Mock()
    evaluator.evaluate.return_value = [EvaluationOutput(score=0.0, test_pass=False, reason="always bad")]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=2)
    plugin._suggest_improvements = Mock(
        return_value=ImprovementSuggestion(reasoning="needs improvement", system_prompt="improved prompt")
    )
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    # 1 initial + 2 retries = 3 total calls
    assert mock_agent.call_count == 3
    assert evaluator.evaluate.call_count == 3


def test_max_retries_zero_no_retry(mock_agent):
    """No retry when max_retries=0."""
    evaluator = Mock()
    evaluator.evaluate.return_value = [EvaluationOutput(score=0.0, test_pass=False)]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=0)
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    assert mock_agent.call_count == 1
    assert evaluator.evaluate.call_count == 1


def test_messages_reset_between_retries(mock_agent):
    """Agent messages should be restored to pre-invocation state before each retry."""
    messages_during_calls = []

    original_messages = ["pre-existing"]
    mock_agent.messages = list(original_messages)

    original_call = mock_agent.__class__.__call__

    def tracking_call(self, prompt=None, **kwargs):
        messages_during_calls.append(list(self.messages))
        self.messages.append(f"response-{self.call_count}")
        return original_call(self, prompt, **kwargs)

    mock_agent.__class__.__call__ = tracking_call

    evaluator = Mock()
    evaluator.evaluate.return_value = [EvaluationOutput(score=0.0, test_pass=False)]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=1)
    plugin._suggest_improvements = Mock(
        return_value=ImprovementSuggestion(reasoning="needs improvement", system_prompt="improved prompt")
    )
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    # Both calls should start with the original messages
    assert messages_during_calls[0] == original_messages
    assert messages_during_calls[1] == original_messages


def test_system_prompt_restored_after_all_attempts(mock_agent):
    """Original system prompt should be restored after retries complete."""
    original_prompt = mock_agent.system_prompt
    evaluator = Mock()
    evaluator.evaluate.return_value = [EvaluationOutput(score=0.0, test_pass=False)]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=1)
    plugin._suggest_improvements = Mock(
        return_value=ImprovementSuggestion(reasoning="needs improvement", system_prompt="improved prompt")
    )
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    assert mock_agent.system_prompt == original_prompt


# --- Step 5: Improvement suggestion generation ---


@patch("strands_evals.plugins.evaluation_plugin.Agent")
def test_improvement_suggestion_called_on_failure(mock_agent_class, mock_agent):
    """LLM should be called to generate suggestions when evaluation fails."""
    mock_suggestion_agent = Mock()
    mock_suggestion_result = Mock()
    mock_suggestion_result.structured_output = ImprovementSuggestion(
        reasoning="Output was wrong", system_prompt="Be more careful"
    )
    mock_suggestion_agent.return_value = mock_suggestion_result
    mock_agent_class.return_value = mock_suggestion_agent

    evaluator = Mock()
    evaluator.evaluate.side_effect = [
        [EvaluationOutput(score=0.0, test_pass=False, reason="incorrect")],
        [EvaluationOutput(score=1.0, test_pass=True, reason="correct")],
    ]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=1, model="test-model")
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    mock_agent_class.assert_called_once()
    mock_suggestion_agent.assert_called_once()


@patch("strands_evals.plugins.evaluation_plugin.Agent")
def test_improvement_suggestion_prompt_contains_failures(mock_agent_class, mock_agent):
    """Improvement prompt should include evaluation failure reasons."""
    mock_suggestion_agent = Mock()
    mock_suggestion_result = Mock()
    mock_suggestion_result.structured_output = ImprovementSuggestion(reasoning="Needs fix", system_prompt="improved")
    mock_suggestion_agent.return_value = mock_suggestion_result
    mock_agent_class.return_value = mock_suggestion_agent

    evaluator = Mock()
    evaluator.evaluate.side_effect = [
        [EvaluationOutput(score=0.0, test_pass=False, reason="answer was factually wrong")],
        [EvaluationOutput(score=1.0, test_pass=True)],
    ]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=1)
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    prompt_arg = mock_suggestion_agent.call_args[0][0]
    assert "answer was factually wrong" in prompt_arg
    assert "original prompt" in prompt_arg


@patch("strands_evals.plugins.evaluation_plugin.Agent")
def test_improvement_suggestion_prompt_contains_expected_output(mock_agent_class, mock_agent):
    """Improvement prompt should include expected_output when available."""
    mock_suggestion_agent = Mock()
    mock_suggestion_result = Mock()
    mock_suggestion_result.structured_output = ImprovementSuggestion(reasoning="fix", system_prompt="improved")
    mock_suggestion_agent.return_value = mock_suggestion_result
    mock_agent_class.return_value = mock_suggestion_agent

    evaluator = Mock()
    evaluator.evaluate.side_effect = [
        [EvaluationOutput(score=0.0, test_pass=False, reason="does not match")],
        [EvaluationOutput(score=1.0, test_pass=True)],
    ]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=1, expected_output="Paris")
    plugin.init_agent(mock_agent)

    mock_agent("What is the capital of France?")

    prompt_arg = mock_suggestion_agent.call_args[0][0]
    assert "Paris" in prompt_arg
    assert "ExpectedOutput" in prompt_arg


@patch("strands_evals.plugins.evaluation_plugin.Agent")
def test_improvement_suggestion_prompt_omits_expected_output_when_none(mock_agent_class, mock_agent):
    """Improvement prompt should not include ExpectedOutput tag when no expected_output is set."""
    mock_suggestion_agent = Mock()
    mock_suggestion_result = Mock()
    mock_suggestion_result.structured_output = ImprovementSuggestion(reasoning="fix", system_prompt="improved")
    mock_suggestion_agent.return_value = mock_suggestion_result
    mock_agent_class.return_value = mock_suggestion_agent

    evaluator = Mock()
    evaluator.evaluate.side_effect = [
        [EvaluationOutput(score=0.0, test_pass=False, reason="bad")],
        [EvaluationOutput(score=1.0, test_pass=True)],
    ]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=1)
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    prompt_arg = mock_suggestion_agent.call_args[0][0]
    assert "ExpectedOutput" not in prompt_arg


@patch("strands_evals.plugins.evaluation_plugin.Agent")
def test_improved_system_prompt_applied_before_retry(mock_agent_class, mock_agent):
    """Agent system prompt should be updated with suggestion before retry."""
    mock_suggestion_agent = Mock()
    mock_suggestion_result = Mock()
    mock_suggestion_result.structured_output = ImprovementSuggestion(
        reasoning="Needs specificity", system_prompt="Always answer with the city name only"
    )
    mock_suggestion_agent.return_value = mock_suggestion_result
    mock_agent_class.return_value = mock_suggestion_agent

    system_prompts_during_calls = []
    original_call = mock_agent.__class__.__call__

    def tracking_call(self, prompt=None, **kwargs):
        system_prompts_during_calls.append(self.system_prompt)
        return original_call(self, prompt, **kwargs)

    mock_agent.__class__.__call__ = tracking_call

    evaluator = Mock()
    evaluator.evaluate.side_effect = [
        [EvaluationOutput(score=0.0, test_pass=False, reason="bad")],
        [EvaluationOutput(score=1.0, test_pass=True)],
    ]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=1)
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    assert system_prompts_during_calls[0] == "original prompt"
    assert system_prompts_during_calls[1] == "Always answer with the city name only"


# --- Step 7: Multiple evaluators + edge cases ---


def test_multiple_evaluators_all_must_pass(mock_agent):
    """All evaluators must pass for the invocation to be considered successful."""
    evaluator1 = Mock()
    evaluator1.evaluate.return_value = [EvaluationOutput(score=1.0, test_pass=True)]
    evaluator2 = Mock()
    evaluator2.evaluate.return_value = [EvaluationOutput(score=1.0, test_pass=True)]
    plugin = EvaluationPlugin(evaluators=[evaluator1, evaluator2], max_retries=0)
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    evaluator1.evaluate.assert_called_once()
    evaluator2.evaluate.assert_called_once()
    assert mock_agent.call_count == 1


def test_partial_evaluator_failure_triggers_retry(mock_agent):
    """If any evaluator fails, retry should be triggered."""
    evaluator1 = Mock()
    evaluator1.evaluate.return_value = [EvaluationOutput(score=1.0, test_pass=True)]
    evaluator2 = Mock()
    evaluator2.evaluate.side_effect = [
        [EvaluationOutput(score=0.0, test_pass=False, reason="failed")],
        [EvaluationOutput(score=1.0, test_pass=True)],
    ]
    plugin = EvaluationPlugin(evaluators=[evaluator1, evaluator2], max_retries=1)
    plugin._suggest_improvements = Mock(
        return_value=ImprovementSuggestion(reasoning="needs improvement", system_prompt="improved")
    )
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    assert mock_agent.call_count == 2


def test_evaluator_exception_recorded_as_failure(mock_agent):
    """Evaluator exceptions should be caught and treated as failures."""
    evaluator = Mock()
    evaluator.evaluate.side_effect = [
        RuntimeError("evaluator crashed"),
        [EvaluationOutput(score=1.0, test_pass=True)],
    ]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=1)
    plugin._suggest_improvements = Mock(
        return_value=ImprovementSuggestion(reasoning="needs improvement", system_prompt="improved")
    )
    plugin.init_agent(mock_agent)

    mock_agent("prompt")

    assert mock_agent.call_count == 2


def test_returns_final_attempt_result(mock_agent):
    """Should return the result from the final attempt."""
    results = [Mock(__str__=Mock(return_value="first")), Mock(__str__=Mock(return_value="second"))]
    call_idx = [0]

    original_call = mock_agent.__class__.__call__

    def multi_result_call(self, prompt=None, **kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        original_call(self, prompt, **kwargs)
        return results[idx]

    mock_agent.__class__.__call__ = multi_result_call

    evaluator = Mock()
    evaluator.evaluate.side_effect = [
        [EvaluationOutput(score=0.0, test_pass=False)],
        [EvaluationOutput(score=1.0, test_pass=True)],
    ]
    plugin = EvaluationPlugin(evaluators=[evaluator], max_retries=1)
    plugin._suggest_improvements = Mock(
        return_value=ImprovementSuggestion(reasoning="needs improvement", system_prompt="improved")
    )
    plugin.init_agent(mock_agent)

    result = mock_agent("prompt")

    assert result is results[1]
