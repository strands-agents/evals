from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators.interactions_evaluator import InteractionsEvaluator
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput


@pytest.fixture
def mock_agent():
    """Mock Agent for testing"""
    agent = Mock()
    agent.structured_output.return_value = EvaluationOutput(
        score=0.85, test_pass=True, reason="Mock interaction evaluation"
    )
    return agent


@pytest.fixture
def mock_async_agent():
    """Mock Agent for testing with async"""
    agent = Mock()

    # Create a mock coroutine function
    async def mock_structured_output_async(*args, **kwargs):
        return EvaluationOutput(score=0.85, test_pass=True, reason="Mock async interaction evaluation")

    agent.structured_output_async = mock_structured_output_async
    return agent


@pytest.fixture
def evaluation_data():
    return EvaluationData(
        input="Analyze climate change impact",
        actual_output="Climate change affects agriculture through drought, temperature, and pests.",
        expected_output="Climate change impacts agriculture via multiple factors.",
        actual_trajectory=["planner", "researcher"],
        actual_interactions=[
            {"node_name": "planner", "dependencies": [], "messages": ["Breaking down the analysis task"]},
            {"node_name": "researcher", "dependencies": ["planner"], "messages": ["Found key climate impacts"]},
        ],
        expected_interactions=[
            {"node_name": "planner", "dependencies": [], "messages": ["Plan the analysis approach"]},
            {"node_name": "researcher", "dependencies": ["planner"], "messages": ["Research climate data"]},
        ],
        name="climate_analysis_test",
    )


def test_interactions_evaluator__init__with_defaults():
    """Test InteractionsEvaluator initialization with default values"""
    evaluator = InteractionsEvaluator(rubric="Test interaction rubric")

    assert evaluator.rubric == "Test interaction rubric"
    assert evaluator.interaction_description is None
    assert evaluator.model is None
    assert evaluator.include_inputs is True
    assert evaluator.system_prompt is not None


def test_interactions_evaluator__init__with_custom_values():
    """Test InteractionsEvaluator initialization with custom values"""
    custom_prompt = "Custom interaction prompt"
    evaluator = InteractionsEvaluator(
        rubric="Custom rubric",
        interaction_description={"planner": "Plan rubric"},
        model="gpt-4",
        system_prompt=custom_prompt,
        include_inputs=False,
    )

    assert evaluator.rubric == "Custom rubric"
    assert evaluator.interaction_description == {"planner": "Plan rubric"}
    assert evaluator.model == "gpt-4"
    assert evaluator.include_inputs is False
    assert evaluator.system_prompt == custom_prompt


def test_interactions_evaluator_get_node_rubric_string():
    """Test _get_node_rubric with string rubric"""
    evaluator = InteractionsEvaluator(rubric="General rubric")
    result = evaluator._get_node_rubric("any_node")
    assert result == "General rubric"


def test_interactions_evaluator_get_node_rubric_dict():
    """Test _get_node_rubric with dictionary rubric"""
    rubric_dict = {"planner": "Plan rubric", "researcher": "Research rubric"}
    evaluator = InteractionsEvaluator(rubric=rubric_dict)

    assert evaluator._get_node_rubric("planner") == "Plan rubric"
    assert evaluator._get_node_rubric("researcher") == "Research rubric"


def test_interactions_evaluator_get_node_rubric_dict_missing_key():
    """Test _get_node_rubric raises exception for missing key"""
    rubric_dict = {"planner": "Plan rubric"}
    evaluator = InteractionsEvaluator(rubric=rubric_dict)

    with pytest.raises(Exception, match="Please make sure the rubric dictionary contains the key 'missing_node'"):
        evaluator._get_node_rubric("missing_node")


@patch("strands_evals.evaluators.interactions_evaluator.Agent")
def test_interactions_evaluator_evaluate_with_full_data(mock_agent_class, evaluation_data, mock_agent):
    """Test evaluation with complete interaction data"""
    mock_agent_class.return_value = mock_agent
    evaluator = InteractionsEvaluator(rubric="Test rubric")

    result = evaluator.evaluate(evaluation_data)

    # Verify Agent creation
    mock_agent_class.assert_called_once()
    call_kwargs = mock_agent_class.call_args[1]
    assert call_kwargs["model"] is None
    assert call_kwargs["system_prompt"] == evaluator.system_prompt
    assert call_kwargs["callback_handler"] is None

    # Verify structured_output was called twice (for each interaction)
    assert mock_agent.structured_output.call_count == 2

    assert result[0].score == 0.85
    assert result[0].test_pass is True


@patch("strands_evals.evaluators.interactions_evaluator.Agent")
def test_interactions_evaluator_evaluate_without_inputs(mock_agent_class, evaluation_data, mock_agent):
    """Test evaluation without inputs included"""
    mock_agent_class.return_value = mock_agent
    evaluator = InteractionsEvaluator(rubric="Test rubric", include_inputs=False)

    result = evaluator.evaluate(evaluation_data)

    # Check that structured_output was called
    assert mock_agent.structured_output.call_count == 2
    assert result[0].score == 0.85
    assert result[0].test_pass is True

    call_args = mock_agent.structured_output.call_args
    prompt = call_args[0][1]
    assert "<Input>" not in prompt
    assert "<Trajectory>" not in prompt


def test_interactions_evaluator_evaluate_missing_interactions():
    """Test evaluation raises exception when interactions are missing"""
    evaluator = InteractionsEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(input="test", actual_output="result")

    with pytest.raises(
        KeyError, match="Please make sure the task function returns a dictionary with the key 'interactions'"
    ):
        evaluator.evaluate(evaluation_data)


@patch("strands_evals.evaluators.interactions_evaluator.Agent")
def test_interactions_evaluator_evaluate_missing_interaction_fields(mock_agent_class, mock_agent):
    """Test evaluation raises exception when interaction fields are missing"""
    mock_agent_class.return_value = mock_agent
    evaluator = InteractionsEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input="test",
        actual_interactions=[{}],  # Missing everything
    )

    with pytest.raises(KeyError):
        evaluator.evaluate(evaluation_data)


@patch("strands_evals.evaluators.interactions_evaluator.Agent")
def test_interactions_evaluator_evaluate_missing_interaction_fields_no_error(mock_agent_class, mock_agent):
    """Test evaluation shouldn't raise exception when interaction fields are missing"""
    mock_agent_class.return_value = mock_agent
    evaluator = InteractionsEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input="test",
        actual_interactions=[{"node_name": "test"}],  # Missing dependencies, messages
    )
    # no error
    evaluator.evaluate(evaluation_data)


@patch("strands_evals.evaluators.interactions_evaluator.Agent")
def test_interactions_evaluator_evaluate_with_dict_rubric(mock_agent_class, mock_agent):
    """Test evaluation with dictionary rubric for different nodes"""
    mock_agent_class.return_value = mock_agent
    rubric_dict = {
        "planner": "Evaluate planning quality and task breakdown",
        "researcher": "Assess research thoroughness and accuracy",
    }
    evaluator = InteractionsEvaluator(rubric=rubric_dict)

    evaluation_data = EvaluationData(
        input="Analyze climate change",
        actual_interactions=[
            {"node_name": "planner", "dependencies": [], "messages": ["Breaking down analysis"]},
            {"node_name": "researcher", "dependencies": ["planner"], "messages": ["Research findings"]},
        ],
    )

    result = evaluator.evaluate(evaluation_data)

    # Verify both interactions were evaluated
    assert mock_agent.structured_output.call_count == 2

    # Check that correct rubrics were used in prompts
    call_args_list = mock_agent.structured_output.call_args_list
    first_prompt = call_args_list[0][0][1]
    second_prompt = call_args_list[1][0][1]

    assert "Evaluate planning quality and task breakdown" in first_prompt
    assert "Assess research thoroughness and accuracy" in second_prompt

    assert result[0].score == 0.85
    assert result[0].test_pass is True


@patch("strands_evals.evaluators.interactions_evaluator.Agent")
def test_interactions_evaluator_evaluate_dict_rubric_missing_node(mock_agent_class, mock_agent):
    """Test evaluation with dictionary rubric raises exception for missing node"""
    mock_agent_class.return_value = mock_agent
    rubric_dict = {"planner": "Plan rubric"}
    evaluator = InteractionsEvaluator(rubric=rubric_dict)

    evaluation_data = EvaluationData(
        input="test",
        actual_interactions=[{"node_name": "missing_node", "dependencies": [], "messages": ["test message"]}],
    )

    with pytest.raises(KeyError, match="Please make sure the rubric dictionary contains the key 'missing_node'"):
        evaluator.evaluate(evaluation_data)


@patch("strands_evals.evaluators.interactions_evaluator.Agent")
async def test_interactions_evaluator_evaluate_async_with_dict_rubric(mock_agent_class, mock_async_agent):
    """Test async evaluation with dictionary rubric"""
    mock_agent_class.return_value = mock_async_agent
    rubric_dict = {"planner": "Evaluate planning approach", "researcher": "Assess research quality"}
    evaluator = InteractionsEvaluator(rubric=rubric_dict)

    evaluation_data = EvaluationData(
        input="Climate analysis",
        actual_interactions=[
            {"node_name": "planner", "dependencies": [], "messages": ["Plan analysis"]},
            {"node_name": "researcher", "dependencies": ["planner"], "messages": ["Research data"]},
        ],
    )

    result = await evaluator.evaluate_async(evaluation_data)

    assert result[0].score == 0.85
    assert result[0].test_pass is True


def test_interactions_evaluator_compose_prompt_first_interaction():
    """Test _compose_prompt for first interaction"""
    evaluator = InteractionsEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input="Test input",
        actual_output="Test output",
        expected_output="Expected output",
        actual_interactions=[
            {"node_name": "agent1", "dependencies": [], "messages": ["First message"]},
            {"node_name": "agent2", "dependencies": ["agent1"], "messages": ["Second message"]},
        ],
        expected_interactions=[
            {"node_name": "agent1", "dependencies": [], "messages": ["Expected first"]},
            {"node_name": "agent2", "dependencies": ["agent1"], "messages": ["Expected second"]},
        ],
    )

    prompt = evaluator._compose_prompt(evaluation_data, 0, False)

    assert "Node Name: agent1" in prompt
    assert "First message" in prompt
    assert "<Input>Test input</Input>" in prompt
    assert "<Output>" not in prompt  # Only in last interaction
    assert "<ExpectedSequence>" in prompt
    assert "Test rubric" in prompt


def test_interactions_evaluator_compose_prompt_last_interaction():
    """Test _compose_prompt for last interaction"""
    evaluator = InteractionsEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input="Test input",
        actual_output="Test output",
        expected_output="Expected output",
        actual_interactions=[
            {"node_name": "agent1", "dependencies": [], "messages": ["First message"]},
            {"node_name": "agent2", "dependencies": ["agent1"], "messages": ["Second message"]},
        ],
    )

    prompt = evaluator._compose_prompt(evaluation_data, 1, True)

    assert "Evaluate this final interaction" in prompt
    assert "THE FINAL SCORE MUST BE A DECIMAL" in prompt
    assert "Node Name: agent2" in prompt
    assert "Second message" in prompt
    assert "<Output>Test output</Output>" in prompt
    assert "<ExpectedOutput>Expected output</ExpectedOutput>" in prompt
    assert "<Input>" not in prompt  # Only in first interaction


def test_interactions_evaluator_compose_prompt_without_inputs():
    """Test _compose_prompt without inputs included"""
    evaluator = InteractionsEvaluator(rubric="Test rubric", include_inputs=False)
    evaluation_data = EvaluationData(
        input="Test input",
        actual_interactions=[{"node_name": "agent1", "dependencies": [], "messages": ["First message"]}],
    )

    prompt = evaluator._compose_prompt(evaluation_data, 0, True)

    assert "<Input>" not in prompt
    assert "Node Name: agent1" in prompt


def test_interactions_evaluator_compose_prompt_with_interaction_description():
    """Test _compose_prompt with interaction description"""
    evaluator = InteractionsEvaluator(
        rubric="Test rubric", interaction_description={"agent1": "First agent description"}
    )
    evaluation_data = EvaluationData(
        input="Test input",
        actual_interactions=[
            {"node_name": "agent1", "dependencies": [], "messages": ["First message"]},
            {"node_name": "agent2", "dependencies": [], "messages": ["Second message"]},
        ],
        actual_output="output",
        expected_output="expected output",
    )

    prompt = evaluator._compose_prompt(evaluation_data, 0, False)

    assert "<Input>" in prompt
    assert "<InteractionDescription>" in prompt
    assert "First agent description" in prompt
    assert "<Output>" not in prompt
    assert "<ExpectedOutput>" not in prompt


def test_interactions_evaluator_compose_prompt_missing_interaction_fields():
    """Test _compose_prompt raises exception when all interaction fields are missing"""
    evaluator = InteractionsEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input="Test input",
        actual_interactions=[{}],  # Missing all fields
    )

    with pytest.raises(Exception, match="Please make sure the task function returns a dictionary"):
        evaluator._compose_prompt(evaluation_data, 0, True)


def test_interactions_evaluator_compose_prompt_with_list_input():
    """Test _compose_prompt with input as list matching interactions length"""
    evaluator = InteractionsEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input=["First prompt", "Second prompt"],  # List input matching interactions
        actual_interactions=[
            {"node_name": "agent1", "dependencies": [], "messages": ["First response"]},
            {"node_name": "agent2", "dependencies": ["agent1"], "messages": ["Second response"]},
        ],
    )

    # Test first interaction gets first input
    prompt_0 = evaluator._compose_prompt(evaluation_data, 0, False)
    assert "<Input>First prompt</Input>" in prompt_0
    assert "First response" in prompt_0

    # Test second interaction gets second input
    prompt_1 = evaluator._compose_prompt(evaluation_data, 1, True)
    assert "<Input>Second prompt</Input>" in prompt_1
    assert "Second response" in prompt_1


def test_interactions_evaluator_compose_prompt_with_mismatched_list_input():
    """Test _compose_prompt with input as list not matching interactions length"""
    evaluator = InteractionsEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input=["Only one prompt"],  # List input shorter than interactions
        actual_interactions=[
            {"node_name": "agent1", "dependencies": [], "messages": ["First response"]},
            {"node_name": "agent2", "dependencies": ["agent1"], "messages": ["Second response"]},
        ],
    )

    # Test first interaction gets input (only for first interaction when lengths don't match)
    prompt_0 = evaluator._compose_prompt(evaluation_data, 0, False)
    assert "<Input>['Only one prompt']</Input>" in prompt_0
    assert "First response" in prompt_0

    # Test second interaction doesn't get input (since lengths don't match)
    prompt_1 = evaluator._compose_prompt(evaluation_data, 1, True)
    assert "<Input>" not in prompt_1
    assert "Second response" in prompt_1
