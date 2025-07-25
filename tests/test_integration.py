import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.strands_evaluation.dataset import Dataset
from src.strands_evaluation.case import Case
from src.strands_evaluation.evaluators.evaluator import Evaluator
from src.strands_evaluation.evaluators.output_evaluator import OutputEvaluator
from src.strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
from src.strands_evaluation.types.evaluation import EvaluationOutput, EvaluationData


class SimpleEvaluator(Evaluator[str, str]):
    """Simple evaluator for integration testing"""
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return EvaluationOutput(score=score, test_pass=score > 0.5, reason="Integration test")
        
    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        """Async version of evaluate"""
        # Add a small delay to simulate async processing
        await asyncio.sleep(0.01)
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return EvaluationOutput(score=score, test_pass=score > 0.5, reason="Async integration test")


@pytest.fixture
def cases():
    return [
        Case(name="exact_match", input="hello", expected_output="hello"),
        Case(name="no_match", input="foo", expected_output="bar"),
        Case(name="partial", input="test", expected_output="test_result")
    ]

mock_score = 0.8
@pytest.fixture
def mock_agent():
    agent = Mock()
    agent.structured_output.return_value = EvaluationOutput(
        score=mock_score, test_pass=True, reason="LLM evaluation"
    )
    return agent

@pytest.fixture
def mock_async_agent():
    agent = Mock()
    agent.structured_output_async = AsyncMock(return_value=EvaluationOutput(
        score=mock_score, test_pass=True, reason="Async LLM evaluation"
    ))
    return agent


class TestIntegration:
    
    def test_dataset_with_simple_evaluator_end_to_end(self, cases):
        """Test complete workflow: Dataset + Cases + SimpleEvaluator"""
        dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())
        
        def echo_task(input_val):
            return input_val
        
        report = dataset.run_evaluations(echo_task)
        
        # Verify complete workflow
        assert len(report.scores) == 3
        assert report.scores[0] == 1.0  # exact match
        assert report.scores[1] == 0.0  # no match  
        assert report.scores[2] == 0.0  # partial no match
        assert report.overall_score == 1.0/3
        assert report.test_passes == [True, False, False]
        assert len(report.cases) == 3
    
    def test_dataset_with_dict_output_task(self, cases):
        """Test Dataset with task returning dictionary output"""
        dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())
        
        def dict_task(input_val):
            return {
                "output": input_val,
                "trajectory": ["step1", "step2"]
            }
        
        report = dataset.run_evaluations(dict_task)
        
        assert len(report.scores) == 3
        assert report.scores[0] == 1.0  # exact match
        assert report.scores[1] == 0.0  # no match
        assert report.scores[2] == 0.0  # partial no match
        assert report.overall_score == 1.0/3
        assert report.test_passes == [True, False, False]
        assert len(report.cases) == 3
    
    @patch('src.strands_evaluation.evaluators.output_evaluator.Agent')
    def test_dataset_with_output_evaluator(self, mock_agent_class, cases, mock_agent):
        """Test Dataset with OutputEvaluator integration"""
        mock_agent_class.return_value = mock_agent
        
        output_evaluator = OutputEvaluator(rubric="Test if outputs match exactly")
        dataset = Dataset(cases=cases, evaluator=output_evaluator)
        
        def simple_task(input_val):
            return f"processed_{input_val}"
        
        report = dataset.run_evaluations(simple_task)
        
        # Verify LLM evaluator was called for each test case
        assert mock_agent.structured_output.call_count == 3
        assert len(report.scores) == 3
        assert all(abs(score - mock_score) <= .00001 for score in report.scores)
        assert abs(report.overall_score - mock_score) <= .00001
    
    def test_evaluation_report_display_integration(self, cases):
        """Test that EvaluationReport display works with real data"""
        dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())
        
        def mixed_task(input_val):
            if input_val == "hello":
                return "hello"
            return "different"
        
        report = dataset.run_evaluations(mixed_task)
        
        # Test that display method doesn't crash
        try:
            report.display()
            display_success = True
        except Exception:
            display_success = False
        
        assert display_success
        assert len(report.cases) == 3

    @patch('src.strands_evaluation.evaluators.trajectory_evaluator.Agent')
    def test_dataset_with_trajectory_evaluator(self, mock_agent_class, cases, mock_agent):
        """Test Dataset with TrajectoryEvaluator integration"""
        mock_agent_class.return_value = mock_agent
        trajectory_evaluator = TrajectoryEvaluator(rubric="Test if trajectories match exactly")
        dataset = Dataset(cases=cases, evaluator=trajectory_evaluator)

        def simple_task(input_val):
            return {"output": f"processed_{input_val}", "trajectory": ["step1", "step2"]}

        report = dataset.run_evaluations(simple_task)

        # Verify the evaluator was called for each test case
        assert len(report.scores) == 3
        assert all(abs(score - mock_score) <= .00001 for score in report.scores)
        assert abs(report.overall_score - mock_score) <= .00001

    def test_dataset_with_list_inputs(self):
        """Test Dataset with list inputs"""
        cases = [
            Case(input=["hello", "world"], expected_output=["hello", "world"]),
            Case(input=["foo", "bar"], expected_output=["foo", "bar"])
        ]
        dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())

        def list_task(input_val):
            return input_val

        report = dataset.run_evaluations(list_task)

        # no error in display
        report.display()
        assert len(report.scores) == 2
        assert report.scores[0] == 1.0  # exact match
        assert report.scores[1] == 1.0  # exact match
        assert report.overall_score == 1.0
        assert report.test_passes == [True, True]
        assert len(report.cases) == 2
        
    @pytest.mark.asyncio
    async def test_async_dataset_with_simple_evaluator(self, cases):
        """Test async workflow: Dataset + Cases + SimpleEvaluator"""
        dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())
        
        def echo_task(input_val):
            return input_val
        
        report = await dataset.run_evaluations_async(echo_task)
        
        # Verify complete workflow
        assert len(report.scores) == 3
        assert report.scores[0] == 1.0  # exact match
        assert report.scores[1] == 0.0  # no match  
        assert report.scores[2] == 0.0  # partial no match
        assert report.overall_score == 1.0/3
        assert report.test_passes == [True, False, False]
        assert len(report.cases) == 3
    
    @pytest.mark.asyncio
    async def test_async_dataset_with_async_task(self, cases):
        """Test async workflow with async task function"""
        dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())
        
        async def async_echo_task(input_val):
            await asyncio.sleep(0.01)  # Simulate async work
            return input_val
        
        report = await dataset.run_evaluations_async(async_echo_task)
        
        # Verify complete workflow
        assert len(report.scores) == 3
        assert report.scores[0] == 1.0  # exact match
        assert report.scores[1] == 0.0  # no match  
        assert report.scores[2] == 0.0  # partial no match
        assert report.overall_score == 1.0/3
        assert report.test_passes == [True, False, False]
        assert len(report.cases) == 3
    
    @pytest.mark.asyncio
    @patch('src.strands_evaluation.evaluators.output_evaluator.Agent')
    async def test_async_dataset_with_output_evaluator(self, mock_agent_class, cases, mock_async_agent):
        """Test async Dataset with OutputEvaluator integration"""
        mock_agent_class.return_value = mock_async_agent
        
        output_evaluator = OutputEvaluator(rubric="Test if outputs match exactly")
        dataset = Dataset(cases=cases, evaluator=output_evaluator)
        
        def simple_task(input_val):
            return f"processed_{input_val}"
        
        report = await dataset.run_evaluations_async(simple_task)
        
        # Verify results
        assert len(report.scores) == 3
        assert all(abs(score - mock_score) <= .00001 for score in report.scores)
        assert abs(report.overall_score - mock_score) <= .00001
    
    @pytest.mark.asyncio
    async def test_async_dataset_concurrency(self, cases):
        """Test that async evaluations run concurrently"""
        # Create a dataset with more test cases
        many_cases = [Case(name=f"case{i}", input=f"input{i}", expected_output=f"input{i}") 
                     for i in range(10)]
        dataset = Dataset(cases=many_cases, evaluator=SimpleEvaluator())
        
        # Create a task with noticeable delay
        async def slow_task(input_val):
            await asyncio.sleep(0.1)  # Each task takes 0.1s
            return input_val
        
        # Time the execution
        start_time = asyncio.get_event_loop().time()
        report = await dataset.run_evaluations_async(slow_task, max_workers=5)
        end_time = asyncio.get_event_loop().time()
        
        # With 10 tasks taking 0.1s each and 5 workers, should take ~0.2s
        # (two batches of 5 tasks), not 1.0s (if sequential)
        assert end_time - start_time < 0.5  # Allow some overhead
        
        # Verify results
        assert len(report.scores) == 10
        assert all(score == 1.0 for score in report.scores)
        assert report.overall_score == 1.0