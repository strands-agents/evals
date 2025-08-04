import pytest
from unittest.mock import Mock, patch
from src.strands_evaluation.evaluators.interactions_evaluator import InteractionsEvaluator
from src.strands_evaluation.types.evaluation import EvaluationData, EvaluationOutput


@pytest.fixture
def mock_agent():
    """Mock Agent for testing"""
    agent = Mock()
    agent.structured_output.return_value = EvaluationOutput(
        score=0.85, 
        test_pass=True, 
        reason="Mock interaction evaluation"
    )
    return agent

@pytest.fixture
def mock_async_agent():
    """Mock Agent for testing with async"""
    agent = Mock()
    # Create a mock coroutine function
    async def mock_structured_output_async(*args, **kwargs):
        return EvaluationOutput(
            score=0.85, 
            test_pass=True, 
            reason="Mock async interaction evaluation"
        )
    
    agent.structured_output_async = mock_structured_output_async
    return agent

@pytest.fixture
def evaluation_data():
    return EvaluationData(
        input="Analyze climate change impact",
        actual_output="Climate change affects agriculture through drought, temperature, and pests.",
        expected_output="Climate change impacts agriculture via multiple factors.",
        actual_interactions=[
            {"node_name": "planner", "dependencies": [], "messages": "Breaking down the analysis task"},
            {"node_name": "researcher", "dependencies": ["planner"], "messages": "Found key climate impacts"}
        ],
        expected_interactions=[
            {"node_name": "planner", "dependencies": [], "messages": "Plan the analysis approach"},
            {"node_name": "researcher", "dependencies": ["planner"], "messages": "Research climate data"}
        ],
        name="climate_analysis_test"
    )

class TestInteractionsEvaluator:
    
    def test_init_with_defaults(self):
        """Test InteractionsEvaluator initialization with default values"""
        evaluator = InteractionsEvaluator(rubric="Test interaction rubric")
        
        assert evaluator.rubric == "Test interaction rubric"
        assert evaluator.model is None
        assert evaluator.include_inputs == True
        assert evaluator.system_prompt is not None
    
    def test_init_with_custom_values(self):
        """Test InteractionsEvaluator initialization with custom values"""
        custom_prompt = "Custom interaction prompt"
        evaluator = InteractionsEvaluator(
            rubric="Custom rubric",
            model="gpt-4",
            system_prompt=custom_prompt,
            include_inputs=False
        )
        
        assert evaluator.rubric == "Custom rubric"
        assert evaluator.model == "gpt-4"
        assert evaluator.include_inputs == False
        assert evaluator.system_prompt == custom_prompt
    
    def test_get_node_rubric_string(self):
        """Test _get_node_rubric with string rubric"""
        evaluator = InteractionsEvaluator(rubric="General rubric")
        result = evaluator._get_node_rubric("any_node")
        assert result == "General rubric"
    
    def test_get_node_rubric_dict(self):
        """Test _get_node_rubric with dictionary rubric"""
        rubric_dict = {"planner": "Plan rubric", "researcher": "Research rubric"}
        evaluator = InteractionsEvaluator(rubric=rubric_dict)
        
        assert evaluator._get_node_rubric("planner") == "Plan rubric"
        assert evaluator._get_node_rubric("researcher") == "Research rubric"
    
    def test_get_node_rubric_dict_missing_key(self):
        """Test _get_node_rubric raises exception for missing key"""
        rubric_dict = {"planner": "Plan rubric"}
        evaluator = InteractionsEvaluator(rubric=rubric_dict)
        
        with pytest.raises(Exception, match="Please make sure the rubric dictionary contains the key 'missing_node'"):
            evaluator._get_node_rubric("missing_node")
    
    @patch('src.strands_evaluation.evaluators.interactions_evaluator.Agent')
    def test_evaluate_with_full_data(self, mock_agent_class, evaluation_data, mock_agent):
        """Test evaluation with complete interaction data"""
        mock_agent_class.return_value = mock_agent
        evaluator = InteractionsEvaluator(rubric="Test rubric")
        
        result = evaluator.evaluate(evaluation_data)
        
        # Verify Agent creation
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs['model'] is None
        assert call_kwargs['system_prompt'] == evaluator.system_prompt
        assert call_kwargs['callback_handler'] is None
        
        # Verify structured_output was called twice (for each interaction)
        assert mock_agent.structured_output.call_count == 2
        
        assert result.score == 0.85
        assert result.test_pass == True
    
    @patch('src.strands_evaluation.evaluators.interactions_evaluator.Agent')
    def test_evaluate_without_inputs(self, mock_agent_class, evaluation_data, mock_agent):
        """Test evaluation without inputs included"""
        mock_agent_class.return_value = mock_agent
        evaluator = InteractionsEvaluator(rubric="Test rubric", include_inputs=False)
        
        result = evaluator.evaluate(evaluation_data)
        
        # Check that structured_output was called
        assert mock_agent.structured_output.call_count == 2
        assert result.score == 0.85
        assert result.test_pass == True

        call_args = mock_agent.structured_output.call_args
        prompt = call_args[0][1]
        assert "<Input>" not in prompt

    def test_evaluate_missing_interactions(self):
        """Test evaluation raises exception when interactions are missing"""
        evaluator = InteractionsEvaluator(rubric="Test rubric")
        evaluation_data = EvaluationData(
            input="test",
            actual_output="result"
        )
        
        with pytest.raises(Exception, match="Please make sure the task function returns a dictionary with the key 'interactions'"):
            evaluator.evaluate(evaluation_data)
    
    @patch('src.strands_evaluation.evaluators.interactions_evaluator.Agent')
    def test_evaluate_missing_interaction_fields(self, mock_agent_class, mock_agent):
        """Test evaluation raises exception when interaction fields are missing"""
        mock_agent_class.return_value = mock_agent
        evaluator = InteractionsEvaluator(rubric="Test rubric")
        evaluation_data = EvaluationData(
            input="test",
            actual_interactions=[{"node_name": "test"}]  # Missing dependencies and message
        )
        
        with pytest.raises(Exception, match="Please make sure the task function returns a dictionary with the key 'interactions' that contains 'node_name', 'dependencies', and 'messages'"):
            evaluator.evaluate(evaluation_data)
    
    @patch('src.strands_evaluation.evaluators.interactions_evaluator.Agent')
    def test_evaluate_with_dict_rubric(self, mock_agent_class, mock_agent):
        """Test evaluation with dictionary rubric for different nodes"""
        mock_agent_class.return_value = mock_agent
        rubric_dict = {
            "planner": "Evaluate planning quality and task breakdown",
            "researcher": "Assess research thoroughness and accuracy"
        }
        evaluator = InteractionsEvaluator(rubric=rubric_dict)
        
        evaluation_data = EvaluationData(
            input="Analyze climate change",
            actual_interactions=[
                {"node_name": "planner", "dependencies": [], "messages": "Breaking down analysis"},
                {"node_name": "researcher", "dependencies": ["planner"], "messages": "Research findings"}
            ]
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
        
        assert result.score == 0.85
        assert result.test_pass == True
    
    @patch('src.strands_evaluation.evaluators.interactions_evaluator.Agent')
    def test_evaluate_dict_rubric_missing_node(self, mock_agent_class, mock_agent):
        """Test evaluation with dictionary rubric raises exception for missing node"""
        mock_agent_class.return_value = mock_agent
        rubric_dict = {"planner": "Plan rubric"}
        evaluator = InteractionsEvaluator(rubric=rubric_dict)
        
        evaluation_data = EvaluationData(
            input="test",
            actual_interactions=[
                {"node_name": "missing_node", "dependencies": [], "messages": "test message"}
            ]
        )
        
        with pytest.raises(Exception, match="Please make sure the rubric dictionary contains the key 'missing_node'"):
            evaluator.evaluate(evaluation_data)
    
    @patch('src.strands_evaluation.evaluators.interactions_evaluator.Agent')
    async def test_evaluate_async_with_dict_rubric(self, mock_agent_class, mock_async_agent):
        """Test async evaluation with dictionary rubric"""
        mock_agent_class.return_value = mock_async_agent
        rubric_dict = {
            "planner": "Evaluate planning approach",
            "researcher": "Assess research quality"
        }
        evaluator = InteractionsEvaluator(rubric=rubric_dict)
        
        evaluation_data = EvaluationData(
            input="Climate analysis",
            actual_interactions=[
                {"node_name": "planner", "dependencies": [], "messages": "Plan analysis"},
                {"node_name": "researcher", "dependencies": ["planner"], "messages": "Research data"}
            ]
        )
        
        result = await evaluator.evaluate_async(evaluation_data)
        
        assert result.score == 0.85
        assert result.test_pass == True
    