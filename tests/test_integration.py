import pytest
from unittest.mock import Mock, patch
from src.strands_evaluation.dataset import Dataset
from src.strands_evaluation.case import Case
from src.strands_evaluation.evaluators.evaluator import Evaluator
from src.strands_evaluation.evaluators.llm_evaluator import LLMEvaluator
from src.strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
from src.strands_evaluation.types.evaluation import EvaluationOutput, EvaluationData


class SimpleEvaluator(Evaluator[str, str]):
    """Simple evaluator for integration testing"""
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return EvaluationOutput(score=score, test_pass=score > 0.5, reason="Integration test")


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
    
    @patch('src.strands_evaluation.evaluators.llm_evaluator.Agent')
    def test_dataset_with_llm_evaluator(self, mock_agent_class, cases, mock_agent):
        """Test Dataset with LLMEvaluator integration"""
        mock_agent_class.return_value = mock_agent
        
        llm_evaluator = LLMEvaluator(rubric="Test if outputs match exactly")
        dataset = Dataset(cases=cases, evaluator=llm_evaluator)
        
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