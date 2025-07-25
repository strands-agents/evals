import pytest
from unittest.mock import Mock, patch
from src.strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
from src.strands_evaluation.types.evaluation import EvaluationData, EvaluationOutput


@pytest.fixture
def mock_agent():
    """Mock Agent for testing"""
    agent = Mock()
    agent.structured_output.return_value = EvaluationOutput(
        score=0.9, 
        test_pass=True, 
        reason="Mock trajectory evaluation"
    )
    return agent

@pytest.fixture
def mock_async_agent():
    """Mock Agent for testing with async"""
    agent = Mock()
    # Create a mock coroutine function
    async def mock_structured_output_async(*args, **kwargs):
        return EvaluationOutput(
            score=0.9, 
            test_pass=True, 
            reason="Mock async trajectory evaluation"
        )
    
    agent.structured_output_async = mock_structured_output_async
    return agent


@pytest.fixture
def evaluation_data():
    return EvaluationData(
        input="What's 2x2?",
        actual_output="2x2 is 4.",
        expected_output="2x2 is 4.",
        actual_trajectory=["calculator"],
        expected_trajectory=["calculator"],
        name="Simple math test"
    )

class TestTrajectoryEvaluator:
    
    def test_init_with_defaults(self):
        """Test TrajectoryEvaluator initialization with default values"""
        evaluator = TrajectoryEvaluator(rubric="Test trajectory rubric")
        
        assert evaluator.rubric == "Test trajectory rubric"
        assert evaluator.model is None
        assert evaluator.include_inputs == True
        assert evaluator.system_prompt is not None # Should have a default
    
    def test_init_with_custom_values(self):
        """Test TrajectoryEvaluator initialization with custom values"""
        custom_prompt = "Custom trajectory prompt"
        evaluator = TrajectoryEvaluator(
            rubric="Custom rubric",
            model="gpt-4",
            system_prompt=custom_prompt,
            include_inputs=False
        )
        
        assert evaluator.rubric == "Custom rubric"
        assert evaluator.model == "gpt-4"
        assert evaluator.include_inputs == False
        assert evaluator.system_prompt == custom_prompt
    
    @patch('src.strands_evaluation.evaluators.trajectory_evaluator.Agent')
    def test_evaluate_with_full_data(self, mock_agent_class, evaluation_data, mock_agent):
        """Test evaluation with complete trajectory data"""
        mock_agent_class.return_value = mock_agent
        evaluator = TrajectoryEvaluator(rubric="Test rubric")
        
        result = evaluator.evaluate(evaluation_data)
        
        # Verify Agent creation
        mock_agent_class.assert_called_once_with(
            model=None,
            system_prompt=evaluator.system_prompt,
            tools = evaluator._tools,
            callback_handler=None
        )
        
        # Verify structured_output call
        mock_agent.structured_output.assert_called_once()
        call_args = mock_agent.structured_output.call_args
        prompt = call_args[0][1]
        
        assert "<Input>What's 2x2?</Input>" in prompt
        assert "<Output>2x2 is 4." in prompt
        assert "<ExpectedOutput>2x2 is 4.</ExpectedOutput>" in prompt
        assert "<Trajectory>['calculator']</Trajectory>" in prompt
        assert "<ExpectedTrajectory>['calculator']</ExpectedTrajectory>" in prompt
        assert "<Rubric>Test rubric</Rubric>" in prompt
        assert result.score == 0.9
        assert result.test_pass == True
    
    @patch('src.strands_evaluation.evaluators.trajectory_evaluator.Agent')
    def test_evaluate_without_inputs(self, mock_agent_class, evaluation_data, mock_agent):
        """Test evaluation without inputs included"""
        mock_agent_class.return_value = mock_agent
        evaluator = TrajectoryEvaluator(rubric="Test rubric", include_inputs=False)
        
        result = evaluator.evaluate(evaluation_data)
        
        call_args = mock_agent.structured_output.call_args
        prompt = call_args[0][1]
        assert "<Input>" not in prompt
        assert "<Output>2x2 is 4." in prompt
        assert "<ExpectedOutput>2x2 is 4.</ExpectedOutput>" in prompt
        assert "<Trajectory>['calculator']</Trajectory>" in prompt
        assert "<ExpectedTrajectory>['calculator']</ExpectedTrajectory>" in prompt
        assert "<Rubric>Test rubric</Rubric>" in prompt
        assert result.score == 0.9
        assert result.test_pass == True
    
    @patch('src.strands_evaluation.evaluators.trajectory_evaluator.Agent')
    def test_evaluate_without_expected_output(self, mock_agent_class, mock_agent):
        """Test evaluation without expected output"""
        mock_agent_class.return_value = mock_agent
        evaluator = TrajectoryEvaluator(rubric="Test rubric")
        evaluation_data = EvaluationData(
            input="test",
            actual_output="result",
            actual_trajectory=["step1", "step2"],
            expected_trajectory=["step1", "step2"]
        )
        
        evaluator.evaluate(evaluation_data)
        
        call_args = mock_agent.structured_output.call_args
        prompt = call_args[0][1]
        assert "<ExpectedOutput>" not in prompt
        assert "<Output>result</Output>" in prompt
        assert "<Trajectory>['step1', 'step2']</Trajectory>" in prompt
        assert "<ExpectedTrajectory>['step1', 'step2']</ExpectedTrajectory>" in prompt
    
    @patch('src.strands_evaluation.evaluators.trajectory_evaluator.Agent')
    def test_evaluate_without_expected_trajectory(self, mock_agent_class, mock_agent):
        """Test evaluation without expected trajectory"""
        mock_agent_class.return_value = mock_agent
        evaluator = TrajectoryEvaluator(rubric="Test rubric")
        evaluation_data = EvaluationData(
            input="test",
            actual_output="result",
            expected_output="result",
            actual_trajectory=["step1", "step2"],
        )
        
        evaluator.evaluate(evaluation_data)
        
        call_args = mock_agent.structured_output.call_args
        prompt = call_args[0][1]
        assert "<ExpectedTrajectory>" not in prompt
        assert "<Trajectory>['step1', 'step2']</Trajectory>" in prompt
        
    @patch('src.strands_evaluation.evaluators.trajectory_evaluator.Agent')
    def test_evaluate_missing_actual_output(self, mock_agent_class, mock_agent):
        """Test evaluation raises exception when actual_output is missing"""
        mock_agent_class.return_value = mock_agent
        evaluator = TrajectoryEvaluator(rubric="Test rubric")
        evaluation_data = EvaluationData(
            input="test",
            actual_trajectory=["step1"]
        )

        evaluator.evaluate(evaluation_data)
        
        call_args = mock_agent.structured_output.call_args
        prompt = call_args[0][1]
        assert "<Output>" not in prompt
        
    def test_evaluate_missing_actual_trajectory(self):
        """Test evaluation raises exception when actual_trajectory is missing"""
        evaluator = TrajectoryEvaluator(rubric="Test rubric")
        evaluation_data = EvaluationData(
            input="test",
            actual_output="result",
        )
        
        with pytest.raises(Exception, match="Please make sure the task function return a dictionary with the key 'trajectory'"):
            evaluator.evaluate(evaluation_data)
            
    @pytest.mark.asyncio
    @patch('src.strands_evaluation.evaluators.trajectory_evaluator.Agent')
    async def test_evaluate_async_with_full_data(self, mock_agent_class, evaluation_data, mock_async_agent):
        """Test async evaluation with complete trajectory data"""
        mock_agent_class.return_value = mock_async_agent
        evaluator = TrajectoryEvaluator(rubric="Test rubric")
        
        result = await evaluator.evaluate_async(evaluation_data)
        
        # Verify Agent creation
        mock_agent_class.assert_called_once_with(
            model=None,
            system_prompt=evaluator.system_prompt,
            tools=evaluator._tools,
            callback_handler=None
        )

        assert result.score == 0.9
        assert result.test_pass == True
        assert result.reason == "Mock async trajectory evaluation"
    
    @pytest.mark.asyncio
    @patch('src.strands_evaluation.evaluators.trajectory_evaluator.Agent')
    async def test_evaluate_async_without_inputs(self, mock_agent_class, evaluation_data, mock_async_agent):
        """Test async evaluation without inputs included"""
        mock_agent_class.return_value = mock_async_agent
        evaluator = TrajectoryEvaluator(rubric="Test rubric", include_inputs=False)
        
        result = await evaluator.evaluate_async(evaluation_data)

        # Verify Agent creation
        mock_agent_class.assert_called_once_with(
            model=None,
            system_prompt=evaluator.system_prompt,
            tools=evaluator._tools,
            callback_handler=None
        )
        
        assert result.score == 0.9
        assert result.test_pass == True
        assert result.reason == "Mock async trajectory evaluation"
    
    @pytest.mark.asyncio
    async def test_evaluate_async_missing_actual_trajectory(self):
        """Test async evaluation raises exception when actual_trajectory is missing"""
        evaluator = TrajectoryEvaluator(rubric="Test rubric")
        evaluation_data = EvaluationData(
            input="test",
            actual_output="result",
        )
        
        with pytest.raises(Exception, match="Please make sure the task function return a dictionary with the key 'trajectory'"):
            await evaluator.evaluate_async(evaluation_data)