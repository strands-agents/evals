import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator, InteractionsEvaluator, OutputEvaluator, TrajectoryEvaluator
from strands_evals.generators import ExperimentGenerator
from strands_evals.generators.topic_planner import Topic, TopicPlan


def test_experiment_generator__init__():
    """Test initialization with no defaults"""
    generator = ExperimentGenerator(
        str,
        int,
        trajectory_type=str,
        include_expected_output=False,
        include_expected_trajectory=True,
        include_expected_interactions=True,
        include_metadata=True,
        model="test-model",
        max_parallel_num_cases=5,
        rubric_system_prompt="rubric system prompt",
        case_system_prompt="case system prompt",
    )
    assert generator.input_type is str
    assert generator.output_type is int
    assert generator.include_expected_output is False
    assert generator.include_expected_trajectory is True
    assert generator.include_expected_interactions is True
    assert generator.include_metadata is True
    assert generator.model == "test-model"
    assert generator.max_parallel_num_cases == 5
    assert generator.rubric_system_prompt == "rubric system prompt"
    assert generator.case_system_prompt == "case system prompt"


def test_experiment_generator__init__minimum():
    """Test initialization with all defaults"""
    generator = ExperimentGenerator(
        str,
        int,
    )
    assert generator.input_type is str
    assert generator.output_type is int


@pytest.mark.asyncio
async def test_experiment_generator_case_worker():
    """Test case worker functionality"""
    generator = ExperimentGenerator(str, str)
    queue = asyncio.Queue()
    queue.put_nowait(None)
    results = []

    mock_agent = AsyncMock()
    mock_case_data = MagicMock()
    mock_case_data.model_dump.return_value = {"name": "test", "input": "hello"}
    mock_agent.structured_output_async.return_value = mock_case_data

    with patch("strands_evals.generators.experiment_generator.Agent", return_value=mock_agent):
        await generator._case_worker(queue, "test prompt", [], results)

    assert len(results) == 1
    assert isinstance(results[0], Case)


@pytest.mark.asyncio
async def test_experiment_generator_generate_cases_async():
    """Test async case generation"""
    generator = ExperimentGenerator(str, str, max_parallel_num_cases=2)

    mock_agent = AsyncMock()
    mock_case_data = MagicMock()
    mock_case_data.model_dump.return_value = {"name": "test", "input": "hello"}
    mock_agent.structured_output_async.return_value = mock_case_data

    with patch("strands_evals.generators.experiment_generator.Agent", return_value=mock_agent):
        cases = await generator.generate_cases_async("test prompt", num_cases=3)

    assert len(cases) == 3
    assert all(isinstance(case, Case) for case in cases)


@pytest.mark.asyncio
async def test_experiment_generator_construct_evaluator_async_output():
    """Test constructing OutputEvaluator"""
    generator = ExperimentGenerator(str, str)

    mock_agent = AsyncMock()
    mock_agent.invoke_async.return_value = "Generated rubric"

    with patch("strands_evals.generators.experiment_generator.Agent", return_value=mock_agent):
        evaluator = await generator.construct_evaluator_async("test prompt", OutputEvaluator)

    assert isinstance(evaluator, OutputEvaluator)
    assert evaluator.rubric == "Generated rubric"


@pytest.mark.asyncio
async def test_experiment_generator_construct_evaluator_async_trajectory():
    """Test constructing TrajectoryEvaluator"""
    generator = ExperimentGenerator(str, str)

    mock_agent = AsyncMock()
    mock_agent.invoke_async.return_value = "Generated rubric"

    with patch("strands_evals.generators.experiment_generator.Agent", return_value=mock_agent):
        evaluator = await generator.construct_evaluator_async("test prompt", TrajectoryEvaluator)

    assert isinstance(evaluator, TrajectoryEvaluator)
    assert evaluator.rubric == "Generated rubric"


@pytest.mark.asyncio
async def test_experiment_generator_construct_evaluator_async_interactions():
    """Test constructing InteractionsEvaluator"""
    generator = ExperimentGenerator(str, str)

    mock_agent = AsyncMock()
    mock_agent.invoke_async.return_value = "Generated rubric"

    with patch("strands_evals.generators.experiment_generator.Agent", return_value=mock_agent):
        evaluator = await generator.construct_evaluator_async("test prompt", InteractionsEvaluator)

    assert isinstance(evaluator, InteractionsEvaluator)
    assert evaluator.rubric == "Generated rubric"


@pytest.mark.asyncio
async def test_experiment_generator_construct_evaluator_async_invalid():
    """Test constructing evaluator with invalid type"""
    generator = ExperimentGenerator(str, str)

    class CustomEvaluator(Evaluator):
        pass

    with pytest.raises(ValueError, match="is not a default evaluator"):
        await generator.construct_evaluator_async("test prompt", CustomEvaluator)


@pytest.mark.asyncio
async def test_experiment_generator_from_scratch_async_no_evaluator():
    """Test generating experiment from scratch without evaluator"""
    generator = ExperimentGenerator(str, str)

    mock_cases = [Case(name="test", input="hello")]

    with patch.object(generator, "generate_cases_async", return_value=mock_cases):
        experiment = await generator.from_scratch_async(["topic1"], "test task", num_cases=1)

    assert isinstance(experiment, Experiment)
    assert experiment.cases == mock_cases
    assert isinstance(experiment.evaluator, Evaluator)


@pytest.mark.asyncio
async def test_experiment_generator_from_scratch_async_with_evaluator():
    """Test generating experiment from scratch with evaluator"""
    generator = ExperimentGenerator(str, str)

    mock_cases = [Case(name="test", input="hello")]
    mock_evaluator = OutputEvaluator(rubric="test rubric")

    with (
        patch.object(generator, "generate_cases_async", return_value=mock_cases),
        patch.object(generator, "construct_evaluator_async", return_value=mock_evaluator),
    ):
        experiment = await generator.from_scratch_async(["topic1"], "test task", evaluator=OutputEvaluator)

    assert isinstance(experiment, Experiment)
    assert experiment.cases == mock_cases
    assert experiment.evaluator == mock_evaluator


@pytest.mark.asyncio
async def test_experiment_generator_from_context_async_no_evaluator():
    """Test generating experiment from context without evaluator"""
    generator = ExperimentGenerator(str, str)

    mock_cases = [Case(name="test", input="hello")]

    with patch.object(generator, "generate_cases_async", return_value=mock_cases):
        experiment = await generator.from_context_async("test context", "test task", num_cases=1)

    assert isinstance(experiment, Experiment)
    assert experiment.cases == mock_cases
    assert isinstance(experiment.evaluator, Evaluator)


@pytest.mark.asyncio
async def test_experiment_generator_from_context_async_with_evaluator():
    """Test generating experiment from context with evaluator"""
    generator = ExperimentGenerator(str, str)

    mock_cases = [Case(name="test", input="hello")]
    mock_evaluator = OutputEvaluator(rubric="test rubric")

    with (
        patch.object(generator, "generate_cases_async", return_value=mock_cases),
        patch.object(generator, "construct_evaluator_async", return_value=mock_evaluator),
    ):
        experiment = await generator.from_context_async("test context", "test task", evaluator=OutputEvaluator)

    assert isinstance(experiment, Experiment)
    assert experiment.cases == mock_cases
    assert experiment.evaluator == mock_evaluator


@pytest.mark.asyncio
async def test_experiment_generator_from_experiment_async_generic_evaluator():
    """Test generating experiment from existing experiment with generic evaluator"""
    generator = ExperimentGenerator(str, str)

    source_cases = [Case(name="source", input="source_input")]
    source_experiment = Experiment(cases=source_cases, evaluator=Evaluator())
    mock_new_cases = [Case(name="new", input="new_input")]

    with patch.object(generator, "generate_cases_async", return_value=mock_new_cases):
        experiment = await generator.from_experiment_async(source_experiment, "test task")

    assert isinstance(experiment, Experiment)
    assert experiment.cases == mock_new_cases
    assert isinstance(experiment.evaluator, Evaluator)


@pytest.mark.asyncio
async def test_experiment_generator_from_experiment_async_default_evaluator():
    """Test generating experiment from existing experiment with default evaluator"""
    generator = ExperimentGenerator(str, str)

    source_cases = [Case(name="source", input="source_input")]
    source_evaluator = OutputEvaluator(rubric="source rubric")
    source_experiment = Experiment(cases=source_cases, evaluator=source_evaluator)
    mock_new_cases = [Case(name="new", input="new_input")]
    mock_new_evaluator = OutputEvaluator(rubric="new rubric")

    with (
        patch.object(generator, "generate_cases_async", return_value=mock_new_cases),
        patch.object(generator, "construct_evaluator_async", return_value=mock_new_evaluator),
    ):
        experiment = await generator.from_experiment_async(source_experiment, "test task")

    assert isinstance(experiment, Experiment)
    assert experiment.cases == mock_new_cases
    assert experiment.evaluator == mock_new_evaluator


@pytest.mark.asyncio
async def test_experiment_generator_update_current_experiment_async_add_cases_only():
    """Test updating experiment by adding new cases only"""
    generator = ExperimentGenerator(str, str)

    source_cases = [Case(name="source", input="source_input")]
    source_evaluator = Evaluator()
    source_experiment = Experiment(cases=source_cases, evaluator=source_evaluator)
    mock_new_cases = [Case(name="new", input="new_input")]

    with patch.object(generator, "generate_cases_async", return_value=mock_new_cases):
        experiment = await generator.update_current_experiment_async(
            source_experiment, "test task", add_new_cases=True, add_new_rubric=False
        )

    assert len(experiment.cases) == 2
    assert experiment.cases == source_cases + mock_new_cases
    assert experiment.evaluator == source_evaluator


@pytest.mark.asyncio
async def test_experiment_generator_update_current_experiment_async_add_rubric_only():
    """Test updating experiment by adding new rubric only"""
    generator = ExperimentGenerator(str, str)

    source_cases = [Case(name="source", input="source_input")]
    source_evaluator = OutputEvaluator(rubric="source rubric")
    source_experiment = Experiment(cases=source_cases, evaluator=source_evaluator)
    mock_new_evaluator = OutputEvaluator(rubric="new rubric")

    with patch.object(generator, "construct_evaluator_async", return_value=mock_new_evaluator):
        experiment = await generator.update_current_experiment_async(
            source_experiment, "test task", add_new_cases=False, add_new_rubric=True
        )

    assert experiment.cases == source_cases
    assert experiment.evaluator == mock_new_evaluator


@pytest.mark.asyncio
async def test_experiment_generator_update_current_experiment_async_new_evaluator_type():
    """Test updating experiment with new evaluator type"""
    generator = ExperimentGenerator(str, str)

    source_cases = [Case(name="source", input="source_input")]
    source_evaluator = OutputEvaluator(rubric="source rubric")
    source_experiment = Experiment(cases=source_cases, evaluator=source_evaluator)
    mock_new_evaluator = TrajectoryEvaluator(rubric="new rubric")

    with patch.object(generator, "construct_evaluator_async", return_value=mock_new_evaluator):
        experiment = await generator.update_current_experiment_async(
            source_experiment,
            "test task",
            add_new_cases=False,
            add_new_rubric=True,
            new_evaluator_type=TrajectoryEvaluator,
        )

    assert experiment.cases == source_cases
    assert isinstance(experiment.evaluator, TrajectoryEvaluator)


@pytest.mark.asyncio
async def test_experiment_generator_update_current_experiment_async_unsupported_evaluator_type():
    """Test updating experiment with unsupported evaluator type falls back to original"""
    generator = ExperimentGenerator(str, str)

    source_cases = [Case(name="source", input="source_input")]
    source_evaluator = OutputEvaluator(rubric="source rubric")
    source_experiment = Experiment(cases=source_cases, evaluator=source_evaluator)

    class UnsupportedEvaluator(Evaluator):
        pass

    experiment = await generator.update_current_experiment_async(
        source_experiment,
        "test task",
        add_new_cases=False,
        add_new_rubric=True,
        new_evaluator_type=UnsupportedEvaluator,
    )

    assert experiment.cases == source_cases
    assert experiment.evaluator == source_evaluator


def test_experiment_generator_default_evaluators_mapping():
    """Test that default evaluators are properly mapped"""
    generator = ExperimentGenerator(str, str)

    assert OutputEvaluator in generator._default_evaluators
    assert TrajectoryEvaluator in generator._default_evaluators
    assert InteractionsEvaluator in generator._default_evaluators


@pytest.mark.asyncio
async def test_experiment_generator_generate_cases_async_with_topics():
    """Test async case generation with topic planning"""
    generator = ExperimentGenerator(str, str)

    mock_agent = AsyncMock()
    mock_case_data = MagicMock()
    mock_case_data.model_dump.return_value = {"name": "test", "input": "hello"}
    mock_agent.structured_output_async.return_value = mock_case_data

    with patch("strands_evals.generators.experiment_generator.Agent", return_value=mock_agent):
        with patch.object(generator, "_prepare_generation_prompts", return_value=[("prompt1", 2), ("prompt2", 1)]):
            cases = await generator.generate_cases_async("test prompt", num_cases=3, num_topics=2)

    assert len(cases) == 3


@pytest.mark.asyncio
async def test_experiment_generator_prepare_generation_prompts_with_topics():
    """Test prompt preparation with topic planning"""
    generator = ExperimentGenerator(str, str)

    mock_plan = TopicPlan(
        topics=[
            Topic(title="T1", description="D1", key_aspects=["a1"]),
            Topic(title="T2", description="D2", key_aspects=["a2"]),
        ]
    )

    with patch("strands_evals.generators.experiment_generator.TopicPlanner") as mock_planner_class:
        mock_planner = AsyncMock()
        mock_planner.plan_topics_async.return_value = mock_plan
        mock_planner_class.return_value = mock_planner

        result = await generator._prepare_generation_prompts("base prompt", num_cases=10, num_topics=2)

    assert len(result) == 2
    assert all(isinstance(prompt, str) and isinstance(count, int) for prompt, count in result)


@pytest.mark.asyncio
async def test_experiment_generator_from_context_async_with_num_topics():
    """Test generating experiment from context with num_topics parameter"""
    generator = ExperimentGenerator(str, str)

    mock_cases = [Case(name="test", input="hello")]

    with patch.object(generator, "generate_cases_async", return_value=mock_cases) as mock_gen:
        experiment = await generator.from_context_async("test context", "test task", num_cases=1, num_topics=3)

    mock_gen.assert_called_once()
    assert mock_gen.call_args[1]["num_topics"] == 3
    assert isinstance(experiment, Experiment)
