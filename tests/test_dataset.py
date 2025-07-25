import pytest
from src.strands_evaluation.dataset import Dataset
from src.strands_evaluation.case import Case
from src.strands_evaluation.evaluators.evaluator import Evaluator
from src.strands_evaluation.types.evaluation import EvaluationOutput, EvaluationData
import asyncio

class MockEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # Simple mock: pass if actual equals expected
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return EvaluationOutput(score=score, test_pass=score > 0.5, reason="Mock evaluation")
    
    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # Add a small delay to simulate async processing
        await asyncio.sleep(0.01)
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return EvaluationOutput(score=score, test_pass=score > 0.5, reason="Async test evaluation")

@pytest.fixture
def mock_evaluator():
    return MockEvaluator()

class TestDataset:
    
    def test_create_dataset(self, mock_evaluator):
        """Test creating a Dataset with test cases and evaluator"""
        cases = [
            Case(name="test1", input="hello", expected_output="world"),
            Case(name="test2", input="foo", expected_output="bar")
        ]
        
        dataset = Dataset(cases=cases, evaluator=mock_evaluator)
        
        assert len(dataset.cases) == 2
        assert dataset.evaluator == mock_evaluator
    
    def test_run_task_simple_output(self, mock_evaluator):
        """Test _run_task with simple output"""
        case = Case(name="test", input="hello", expected_output="world")
        dataset = Dataset(cases=[case], evaluator=mock_evaluator)
        
        def simple_task(input_val):
            return f"response to {input_val}"
        
        result = dataset._run_task(simple_task, case)
        
        assert result.input == "hello"
        assert result.actual_output == "response to hello"
        assert result.expected_output == "world"
        assert result.name == "test"
        assert result.expected_trajectory is None
        assert result.actual_trajectory is None
        assert result.metadata == {}
    
    def test_run_task_dict_output(self, mock_evaluator):
        """Test _run_task with dictionary output containing trajectory"""
        case = Case(name="test", input="hello", expected_output="world")
        dataset = Dataset(cases=[case], evaluator=mock_evaluator)
        
        def dict_task(input_val):
            return {
                "output": f"response to {input_val}",
                "trajectory": ["step1", "step2"]
            }
        
        result = dataset._run_task(dict_task, case)
        
        assert result.actual_output == "response to hello"
        assert result.actual_trajectory == ["step1", "step2"]
    
    def test_run_evaluations(self, mock_evaluator):
        """Test complete evaluation run"""
        cases = [
            Case(name="match", input="hello", expected_output="hello"),
            Case(name="no_match", input="foo", expected_output="bar")
        ]
        dataset = Dataset(cases=cases, evaluator=mock_evaluator)
        
        def echo_task(input_val):
            return input_val
        
        report = dataset.run_evaluations(echo_task)
        
        assert len(report.scores) == 2
        assert report.scores[0] == 1.0  # match
        assert report.scores[1] == 0.0  # no match
        assert report.test_passes[0] == True
        assert report.test_passes[1] == False
        assert report.overall_score == 0.5
        assert len(report.cases) == 2
    
    def test_empty_dataset(self, mock_evaluator):
        """Test dataset with no test cases"""
        dataset = Dataset(cases=[], evaluator=mock_evaluator)
        
        def dummy_task(input_val):
            return input_val
       
        report = dataset.run_evaluations(dummy_task)
        assert report.overall_score == 0
        assert len(report.scores) == 0
        assert len(report.cases) == 0
        assert len(report.test_passes) == 0
        assert len(report.reasons) == 0

    def test_to_dict_empty(self, mock_evaluator):
        """Test converting empty dataset to dictionary"""
        dataset = Dataset(cases=[], evaluator=mock_evaluator)
        assert dataset.to_dict() == {"cases": [], "evaluator": {"evaluator_type": "MockEvaluator"}
                                     }

    def test_to_dict_non_empty(self, mock_evaluator):
        """Test converting non-empty dataset to dictionary"""
        cases = [Case(name="test", input="hello", expected_output="world")]
        dataset = Dataset(cases=cases, evaluator=mock_evaluator)
        assert dataset.to_dict() == {
            "cases": [{"name": "test", "input": "hello", "expected_output": "world", "expected_trajectory": None, "metadata": {}}],
            "evaluator": {"evaluator_type": "MockEvaluator"}
            }
    
    def test_to_dict_LLM_evaluator_full(self):
        """Test converting dataset with LLM evaluator to dictionary with no defaults."""
        from src.strands_evaluation.evaluators.output_evaluator import OutputEvaluator
        cases = [Case(name="test", input="hello", expected_output="world")]
        evaluator = OutputEvaluator(rubric="rubric", model="model", include_inputs=False, system_prompt="system prompt")
        dataset = Dataset(cases=cases, evaluator=evaluator)
        assert dataset.to_dict() == {
            "cases": [{"name": "test", "input": "hello", "expected_output": "world", "expected_trajectory": None, "metadata": {}}],
            "evaluator": {"evaluator_type": "OutputEvaluator",
                        "rubric": "rubric",
                        "model": "model",
                        "include_inputs": False,
                        "system_prompt": "system prompt"}
            }
        
    def test_to_dict_LLM_evaluator_default(self):
        """Test converting dataset with LLM evaluator to dictionary with defaults.
        The evaluator's data should not include default information."""
        from src.strands_evaluation.evaluators.output_evaluator import OutputEvaluator
        cases = [Case(name="test", input="hello", expected_output="world")]
        evaluator = OutputEvaluator(rubric="rubric")
        dataset = Dataset(cases=cases, evaluator=evaluator)
        assert dataset.to_dict() == {
            "cases": [{"name": "test", "input": "hello", "expected_output": "world", "expected_trajectory": None, "metadata": {}}],
            "evaluator": {"evaluator_type": "OutputEvaluator",
                        "rubric": "rubric",
                        }
            }
    
    def test_to_dict_Trajectory_evaluator_default(self):
        """Test converting dataset with Trajectory evaluator to dictionary with defaults.
        The evaluator's data should not include default information."""
        from src.strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
        cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
        evaluator = TrajectoryEvaluator(rubric="rubric")
        dataset = Dataset(cases=cases, evaluator=evaluator)
        assert dataset.to_dict() == {
            "cases": [{"name": "test", "input": "hello", "expected_output": "world", "expected_trajectory": ["step1", "step2"], "metadata": {}}],
            "evaluator": {"evaluator_type": "TrajectoryEvaluator",
                        "rubric": "rubric",
                        }
            }
    
    def test_to_dict_Trajectory_evaluator_full(self):
        """Test converting dataset with Trajectory evaluator to dictionary with no defaults."""
        from src.strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
        cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
        evaluator = TrajectoryEvaluator(rubric="rubric", model="model", include_inputs=False, system_prompt="system prompt")
        dataset = Dataset(cases=cases, evaluator=evaluator)
        assert dataset.to_dict() == {
            "cases": [{"name": "test", "input": "hello", "expected_output": "world", "expected_trajectory": ["step1", "step2"], "metadata": {}}],
            "evaluator": {"evaluator_type": "TrajectoryEvaluator",
                        "rubric": "rubric",
                        "model": "model",
                        "include_inputs": False,
                        "system_prompt": "system prompt"}
            }

    def test_to_dict_case_dict(self):
        """Test converting dataset with Case with dictionaries as types."""
        case = Case(name="test", input={"field1": "hello"}, expected_output={"field2": "world"}) 
        evaluator = MockEvaluator()
        dataset = Dataset(cases=[case], evaluator=evaluator)
        assert dataset.to_dict() == {
            "cases": [{"name": "test", "input": {"field1": "hello"}, "expected_output": {"field2": "world"}, "expected_trajectory": None, "metadata": {}}],
            "evaluator": {"evaluator_type": "MockEvaluator"}
            }
    
    def test_to_dict_case_function(self):
        """Test converting dataset with Case with function as types."""
        def simple_echo(query):
            return query
        case = Case(name="test", input=simple_echo) 
        evaluator = MockEvaluator()
        dataset = Dataset(cases=[case], evaluator=evaluator)
        assert dataset.to_dict() == {
            "cases": [{"name": "test", "input": simple_echo, "expected_output": None, "expected_trajectory": None, "metadata": {}}],
            "evaluator": {"evaluator_type": "MockEvaluator"}
            }

    def test_from_dict_custom(self):
        """Test creating a Dataset with a custom evaluator and empty cases"""
        dict_dataset = {"cases": [], "evaluator": {"evaluator_type": "MockEvaluator"}}
        dataset = Dataset.from_dict(dict_dataset, custom_evaluators=[MockEvaluator])
        assert dataset.cases == []
        assert isinstance(dataset.evaluator, MockEvaluator)
    
    def test_from_dict_llm_evaluator(self):
        """Test creating a Dataset with a LLM evaluator"""
        from src.strands_evaluation.evaluators.output_evaluator import OutputEvaluator
        cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
        dict_dataset = {
            "cases": cases,
            "evaluator": {
                "evaluator_type": "OutputEvaluator",
                "rubric": "rubric",
                "model": "model",
                "include_inputs": False,
                "system_prompt": "system prompt"
            }
        }
        dataset = Dataset.from_dict(dict_dataset, custom_evaluators=[OutputEvaluator])
        assert dataset.cases == cases
        assert isinstance(dataset.evaluator, OutputEvaluator)
        assert dataset.evaluator.rubric == "rubric"
        assert dataset.evaluator.model == "model"
        assert dataset.evaluator.include_inputs == False
        assert dataset.evaluator.system_prompt == "system prompt"

    def test_from_dict_llm_evaluator_defaults(self):
        """Test creating a Dataset with a LLM evaluator with defaults"""
        from src.strands_evaluation.evaluators.output_evaluator import OutputEvaluator
        cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
        dict_dataset = {
            "cases": cases,
            "evaluator": {
                "evaluator_type": "OutputEvaluator",
                "rubric": "rubric"
            }
        }
        dataset = Dataset.from_dict(dict_dataset, custom_evaluators=[OutputEvaluator])
        assert dataset.cases == cases
        assert isinstance(dataset.evaluator, OutputEvaluator)
        assert dataset.evaluator.rubric == "rubric"
        assert dataset.evaluator.model is None
        assert dataset.evaluator.include_inputs == True
    
    def test_from_dict_trajectory_evaluator(self):
        """Test creating a Dataset with a Trajectory evaluator"""
        from src.strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
        cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
        dict_dataset = {
            "cases": cases,
            "evaluator": {
                "evaluator_type": "TrajectoryEvaluator",
                "rubric": "rubric",
                "model": "model",
                "include_inputs": False,
                "system_prompt": "system prompt"
            }
        }
        dataset = Dataset.from_dict(dict_dataset, custom_evaluators=[TrajectoryEvaluator])
        assert dataset.cases == cases
        assert isinstance(dataset.evaluator, TrajectoryEvaluator)
        assert dataset.evaluator.rubric == "rubric"
        assert dataset.evaluator.model == "model"
        assert dataset.evaluator.include_inputs == False
        assert dataset.evaluator.system_prompt == "system prompt"

    def test_from_dict_trajectory_evaluator_defaults(self):
        """Test creating a Dataset with a Trajectory evaluator with defaults"""
        from src.strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
        cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
        dict_dataset = {
            "cases": cases,
            "evaluator": {
                "evaluator_type": "TrajectoryEvaluator",
                "rubric": "rubric"
            }
        }
        dataset = Dataset.from_dict(dict_dataset, custom_evaluators=[TrajectoryEvaluator])
        assert dataset.cases == cases
        assert isinstance(dataset.evaluator, TrajectoryEvaluator)
        assert dataset.evaluator.rubric == "rubric"
        assert dataset.evaluator.model is None
        assert dataset.evaluator.include_inputs == True
    
    @pytest.mark.asyncio
    async def test_run_task_async_with_sync_task(self):
        """Test _run_task_async with a synchronous task function"""
        def sync_task(input_val):
            return input_val

        case = Case(name="test", input="hello", expected_output="world")
        dataset = Dataset(cases=[case], evaluator=MockEvaluator())
        evaluation_context = await dataset._run_task_async(sync_task, case)
        assert evaluation_context.input == "hello"
        assert evaluation_context.actual_output == "hello"
        assert evaluation_context.expected_output == "world"
    
    @pytest.mark.asyncio
    async def test_run_task_async_with_async_task(self):
        """Test _run_task_async with an asynchronous task function"""
        async def async_task(input_val):
            return input_val
        
        case = Case(name="test", input="hello", expected_output="world")
        dataset = Dataset(cases=[case], evaluator=MockEvaluator())
        evaluation_context = await dataset._run_task_async(async_task, case)
        assert evaluation_context.input == "hello"
        assert evaluation_context.actual_output == "hello"
        assert evaluation_context.expected_output == "world"

    @pytest.mark.asyncio
    async def test_run_evaluations_async(self):
        """Test run_evaluations_async with a simple task"""
        def task(input_str):
            return input_str
        
        case = Case(name="test", input="hello", expected_output="hello")
        case1 = Case(name="test1", input="world", expected_output="world")
        dataset = Dataset(cases=[case, case1], evaluator=MockEvaluator())
        
        report = await dataset.run_evaluations_async(task)
        
        assert len(report.scores) == 2
        assert all(score == 1.0 for score in report.scores)
        assert all(test_pass for test_pass in report.test_passes)
        assert report.overall_score == 1.0
    
    @pytest.mark.asyncio
    async def test_run_evaluations_async_with_async_task(self):
        """Test run_evaluations_async with an async task"""
        async def async_task(input_str):
            await asyncio.sleep(0.01)
            return input_str
        
        case = Case(name="test", input="hello", expected_output="hello")
        case1 = Case(name="test1", input="world", expected_output="world")
        dataset = Dataset(cases=[case, case1], evaluator=MockEvaluator())
        report = await dataset.run_evaluations_async(async_task)
        
        assert len(report.scores) == 2
        assert all(score == 1.0 for score in report.scores)
        assert all(test_pass for test_pass in report.test_passes)
        assert report.overall_score == 1.0
    
    @pytest.mark.asyncio
    async def test_run_evaluations_async_with_errors(self):
        """Test run_evaluations_async handles errors gracefully"""
        def failing_task(input_str):
            if input_str == "hello":
                raise ValueError("Test error")
            return input_str
        
        case = Case(name="test", input="hello", expected_output="hello")
        case1 = Case(name="test1", input="world", expected_output="world")
        dataset = Dataset(cases=[case, case1], evaluator=MockEvaluator())
        report = await dataset.run_evaluations_async(failing_task)
        
        assert len(report.scores) == 2
        assert report.scores[0] == 0.0  # case1 fails
        assert report.scores[1] == 1.0  # case2 passes
        assert "Test error" in report.reasons[0]
