import pytest
from src.strands_evaluation.evaluators.evaluator import Evaluator
from src.strands_evaluation.types.evaluation import EvaluationData, EvaluationOutput

class SimpleEvaluator(Evaluator[str, str]):
    """Simple implementation for testing"""
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return EvaluationOutput(score=score, test_pass=score > 0.5, reason="Test evaluation")

@pytest.fixture
def evaluation_data():
    return EvaluationData(
        input="test input",
        actual_output="actual",
        expected_output="expected",
        name="case"
    )

class TestEvaluator:
    
    def test_base_evaluator_not_implemented(self, evaluation_data):
        """Test that base Evaluator raises NotImplementedError"""
        evaluator = Evaluator[str, str]()
        
        with pytest.raises(NotImplementedError, match="This method should be implemented in subclasses"):
            evaluator.evaluate(evaluation_data)
            
    @pytest.mark.asyncio
    async def test_async_base_evaluator_not_implemented(self, evaluation_data):
        """Test that base Evaluator raises NotImplementedError"""
        evaluator = Evaluator[str, str]()
        
        with pytest.raises(NotImplementedError, match="This method should be implemented in subclasses, especially if you want to run evaluations asynchronously."):
            await evaluator.evaluate_async(evaluation_data)
    
    def test_simple_evaluator_implementation(self, evaluation_data):
        """Test that simple implementation works"""
        evaluator = SimpleEvaluator()
        
        result = evaluator.evaluate(evaluation_data)
        
        assert isinstance(result, EvaluationOutput)
        assert result.score == 0.0 # evaluation data has conflicting outputs
        assert result.test_pass == False
        assert result.reason == "Test evaluation"
    
    def test_simple_evaluator_matching_output(self):
        """Test evaluator with matching actual and expected output"""
        evaluator = SimpleEvaluator()
        evaluation_data = EvaluationData(
            input="test",
            actual_output="match",
            expected_output="match"
        )
        
        result = evaluator.evaluate(evaluation_data)
        
        assert result.score == 1.0
        assert result.test_pass == True
        assert result.reason == "Test evaluation"
    
    def test_evaluator_generic_types(self):
        """Test evaluator with different generic types"""
        class IntEvaluator(Evaluator[int, int]):
            def evaluate(self, evaluation_case: EvaluationData[int, int]) -> EvaluationOutput:
                return EvaluationOutput(score=0.5, test_pass=True)
        
        evaluator = IntEvaluator()
        evaluation_data = EvaluationData(input=42, actual_output=42, expected_output=42)
        
        result = evaluator.evaluate(evaluation_data)
        assert result.score == 0.5
    
    def test_evaluator_with_none_values(self):
        """Test evaluator handling None values"""
        evaluator = SimpleEvaluator()
        evaluation_data = EvaluationData(
            input="test",
            actual_output=None,
            expected_output="expected"
        )
        
        result = evaluator.evaluate(evaluation_data)
        assert result.score == 0.0
        assert result.test_pass == False
    
    def test_evaluator_with_empty_strings(self):
        """Test evaluator with empty string values"""
        evaluator = SimpleEvaluator()
        evaluation_data = EvaluationData(
            input="test",
            actual_output="",
            expected_output=""
        )
        
        result = evaluator.evaluate(evaluation_data)
        assert result.score == 1.0
        assert result.test_pass == True

    def test_to_dict_evaluator(self):
        """Test that evaluator to_dict works properly"""
        evaluator = Evaluator()
        evaluator_dict = evaluator.to_dict()
        assert evaluator_dict["evaluator_type"] == "Evaluator"
        assert len(evaluator_dict) == 1