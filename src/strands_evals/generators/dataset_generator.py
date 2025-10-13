import asyncio
import logging

from pydantic import create_model
from strands import Agent
from typing_extensions import Any, Generic, TypeVar

from strands_evals.evaluators import Evaluator, InteractionsEvaluator, OutputEvaluator, TrajectoryEvaluator

from ..case import Case
from ..dataset import Dataset
from ..types.evaluation import Interaction
from .prompt_template.prompt_templates import generate_case_template as CASE_SYSTEM_PROMPT
from .prompt_template.prompt_templates import generate_rubric_template as RUBRIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class DatasetGenerator(Generic[InputT, OutputT]):
    """
    Generates evaluation datasets with test cases and rubrics for LLM-based evaluators for agent assessment.

    This class creates structured test cases and evaluation rubrics tailored to specific tasks
    and domains, enabling comprehensive evaluation of agents' performance.
    """

    _default_evaluators = {
        OutputEvaluator: (
            "evaluates only the output response, don't include information about trajectory "
            "nor interactions even if provided"
        ),
        TrajectoryEvaluator: (
            "evaluates the trajectory and output if provided, don't include info about interactions even if provided"
        ),
        InteractionsEvaluator: (
            "evaluates the interactions and output if provided, don't include info about trajectory even if provided"
        ),
    }

    def __init__(
        self,
        input_type: type,
        output_type: type,
        trajectory_type: type | None = None,
        include_expected_output: bool = True,
        include_expected_trajectory: bool = False,
        include_expected_interactions: bool = False,
        include_metadata: bool = False,
        model: str | None = None,
        max_parallel_num_cases: int = 10,
        rubric_system_prompt: str = RUBRIC_SYSTEM_PROMPT,
        case_system_prompt: str = CASE_SYSTEM_PROMPT,
    ):
        """
        Initialize the dataset generator with configuration for test case structure.

        Args:
            input_type: Type of input data for test cases (e.g., str, dict)
            output_type: Type of expected output data (e.g., str, int)
            trajectory_type: Type for trajectory elements, defaults to Any if None
            include_expected_output: Whether to include expected outputs in test cases
            include_expected_trajectory: Whether to include expected tool/action trajectories
            include_expected_interactions: Whether to include expected interaction sequences
            include_metadata: Whether to include metadata fields in test cases
            model: Model identifier for the generation agent, defaults to strands' default model.
            max_parallel_num_cases: Maximum number of test cases to generate in parallel asynchronously
            rubric_system_prompt: System prompt for rubric generation, defaults to one of the available templates.
            case_system_prompt: System prompt for test case generation, defaults to one of the available templates.
        """
        self.model = model
        self.input_type = input_type
        self.output_type = output_type
        self.include_expected_output = include_expected_output
        self.include_expected_trajectory = include_expected_trajectory
        self.include_expected_interactions = include_expected_interactions
        self.include_metadata = include_metadata
        self.max_parallel_num_cases = max_parallel_num_cases

        self.rubric_system_prompt = rubric_system_prompt
        self.case_system_prompt = case_system_prompt

        # Create class structure for Case with stricter/literal types, excluding any fields not needed
        fields: dict[str, Any] = {"name": (str, ...), "input": (self.input_type, ...)}
        if self.include_expected_output:
            fields["expected_output"] = (self.output_type, ...)
        if self.include_expected_trajectory:
            # Use Any for trajectory type since we can't use runtime variables as types
            fields["expected_trajectory"] = (list[Any], ...)
        if self.include_expected_interactions:
            fields["expected_interactions"] = (list[Interaction], ...)
        if self.include_metadata:
            fields["metadata"] = (dict[str, Any], ...)
        self._Case = create_model("_Case", **fields)

    async def _case_worker(self, queue: asyncio.Queue, prompt: str, message_history: list | None, results: list):
        """
        Worker that generates cases from the queue.

        Args:
            queue: Queue containing cases to process
            prompt: Generation prompt describing the test case requirements
            message_history: Optional conversation history to provide context to the generation agent
            results: List to store results

        """
        case_generator = Agent(
            model=self.model,
            system_prompt=self.case_system_prompt,
            callback_handler=None,
            messages=message_history if message_history else [],
        )

        while True:
            try:
                difficulty = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                full_prompt = prompt + f"Ensure that the test case has a difficulty level of {difficulty}."
                gen_case = await case_generator.structured_output_async(self._Case, full_prompt)
                results.append(Case(**gen_case.model_dump()))
            except Exception as e:
                logger.exception(f"Error generating case: {e}")
            finally:
                queue.task_done()

    async def generate_cases_async(
        self, prompt: str, num_cases: int = 5, message_history: list | None = None
    ) -> list[Case]:
        """
        Generate test cases asynchronously using parallel workers.

        Args:
            prompt: Generation prompt describing the test case requirements
            num_cases: Number of test cases to generate
            message_history: Optional conversation history to provide context to the generation agent

        Returns:
            List of generated Case objects matching the configured schema
        """
        queue: asyncio.Queue[str] = asyncio.Queue()
        generated_cases: list = []

        # Fill queue with tasks
        for i in range(num_cases):
            difficulty = "medium"
            if i < num_cases * 0.3:
                difficulty = "easy"
            elif i > num_cases * 0.8:
                difficulty = "hard"
            queue.put_nowait(difficulty)

        num_workers = min(self.max_parallel_num_cases, num_cases)

        workers = [
            asyncio.create_task(self._case_worker(queue, prompt, message_history, generated_cases))
            for _ in range(num_workers)
        ]

        await queue.join()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        return generated_cases

    async def construct_evaluator_async(
        self, prompt: str, evaluator: Evaluator, message_history: list | None = None
    ) -> Evaluator:
        """
        Create an evaluator instance with a generated rubric.

        Currently supports default evaluators: OutputEvaluator, TrajectoryEvaluator,
        and InteractionsEvaluator. Generates task-specific rubrics for evaluation.

        Args:
            prompt: Prompt describing the evaluation context and requirements
            evaluator: Evaluator class to instantiate (must be a default evaluator)
            message_history: Optional conversation history to provide context to the rubric generation agent

        Returns:
            Configured evaluator instance with generated rubric

        Raises:
            ValueError: If evaluator is not one of the supported default evaluators
        """
        if evaluator not in self._default_evaluators:
            raise ValueError(
                f"{evaluator} is not a default evaluator that needs a rubric. Please use one of the "
                f"default evaluators: {list(self._default_evaluators.keys())}."
            )

        rubric_generator_agent = Agent(
            model=self.model,
            system_prompt=self.rubric_system_prompt,
            callback_handler=None,
            messages=message_history if message_history else [],
        )
        evaluator_name = evaluator.get_type_name()
        evaluator_desc = self._default_evaluators[evaluator]
        evaluator_info = f"""The evaluator selected is {evaluator_name}. This evaluator {evaluator_desc}."""
        final_prompt = (
            prompt
            + evaluator_info
            + """
        IMPORTANT: Your response must be ONLY a few sentences describing how to evaluate the test cases."""
        )

        rubric = await rubric_generator_agent.invoke_async(final_prompt)
        return evaluator(rubric=str(rubric))

    async def from_scratch_async(
        self, topics: list[str], task_description: str, num_cases: int = 5, evaluator: Evaluator = None
    ) -> Dataset:
        """
        Generate a dataset from scratch based on specified topics and task description.

        Creates diverse test cases covering the given topics for the specified task,
        with optional evaluator and rubric generation.

        Args:
            topics: List of topics/domains to cover in test cases
            task_description: Description of the task the AI system will perform
            num_cases: Number of test cases to generate
            evaluator: Optional evaluator class for assessment (generates rubric if provided).

        Returns:
            Dataset containing generated test cases and evaluator. Use the generic Evaluator as placeholder
            if no evaluator is passed in.
        """
        topics_str = " ".join(topics)
        case_prompt = (
            f"""Create test cases for the following topics: {topics_str} for this task: """ f"""{task_description}."""
        )
        cases = await self.generate_cases_async(case_prompt, num_cases)
        if evaluator:
            rubric_prompt = (
                f"""Create a rubric for the following topics: {topics_str} for this task: """ f"""{task_description}."""
            )
            _evaluator = await self.construct_evaluator_async(
                prompt=rubric_prompt,
                evaluator=evaluator,
            )
            return Dataset(cases=cases, evaluator=_evaluator)
        else:
            return Dataset(cases=cases)

    async def from_context_async(
        self, context: str, task_description: str, num_cases: int = 5, evaluator: Evaluator = None
    ) -> Dataset:
        """
        Generate a dataset based on specific context that test cases should reference.

        Creates test cases that can be answered using the provided context,
        useful for testing knowledge retrieval, context understanding, or domain-specific tasks.

        Args:
            context: Specific context/information that test cases should reference. If there's any tools
                they need to use, specify them here too. Be sure to include as much information as you can
                about tools or sub-agents for generating interaction and/or trajectory.
            task_description: Description of the task the AI system will perform
            num_cases: Number of test cases to generate
            evaluator: Optional evaluator class for assessment (generates rubric if provided), use Evaluator()
                as a placeholder.

        Returns:
            Dataset containing context-based test cases and evaluator. Use the generic Evaluator as placeholder
            if no evaluator is passed in.
        """
        cases = await self.generate_cases_async(
            f"""Create test cases with the following context: {context}. Ensure that the questions can be """
            f"""answer using the provided context for this task: {task_description} """,
            num_cases=num_cases,
        )
        if evaluator:
            _evaluator = await self.construct_evaluator_async(
                prompt=f"""Create a rubric with the following context: {context} for this task: """
                f"""{task_description} """,
                evaluator=evaluator,
            )
            return Dataset(cases=cases, evaluator=_evaluator)
        else:
            return Dataset(cases=cases)

    async def from_dataset_async(
        self, source_dataset: Dataset, task_description: str, num_cases: int = 5, extra_information: str | None = None
    ) -> Dataset:
        """
        Generate a new dataset using an existing dataset as reference.

        Creates new test cases that are similar in style and structure to the source dataset,
        while adapting them for the specified task. If the source dataset uses a default
        evaluator with a rubric, generates a new rubric based on the original.

        Args:
            source_dataset: Original dataset to use as reference for generating new test cases
            task_description: Description of the task the AI system will perform
            num_cases: Number of test cases to generate
            extra_information: Optional additional context or requirements for the new test cases and rubric,
                be sure to include as much information as you can about tools or sub-agents
                for generating interaction and/or trajectory.

        Returns:
            A new Dataset containing test cases inspired by the source dataset but adapted
            for the new task. Uses an updated evaluator with new rubric if the source
            evaluator is a default type, otherwise uses generic Evaluator.
        """
        source_cases = source_dataset.cases
        source_evaluator = source_dataset.evaluator

        # construct messages to initialize the agent with context about the previous test cases
        messages = [{"role": "user", "content": [{"text": "Here are the reference test cases: "}]}]
        cases_string_list = []
        for i, case in enumerate(source_cases):
            cases_string_list.append({"text": f"{i}. {case.model_dump()}"})
        messages.append({"role": "user", "content": cases_string_list})
        new_cases = await self.generate_cases_async(
            prompt=(
                f"Create new test cases similar to the reference cases. Ensure that the input and output "
                f"are relevant for this task: {task_description}. Here are some extra information: "
                f"{extra_information}."
            ),
            num_cases=num_cases,
            message_history=messages,
        )
        new_evaluator = Evaluator()
        if type(source_evaluator) in self._default_evaluators:
            source_rubric = source_evaluator.rubric
            new_evaluator = await self.construct_evaluator_async(
                prompt=(
                    f"Create a new rubric based on the reference rubric. Ensure that the rubric is relevant "
                    f"for this task: {task_description}. Here are some extra information: {extra_information}."
                ),
                evaluator=type(source_evaluator),
                message_history=[{"role": "user", "content": [{"text": source_rubric}]}],
            )

        return Dataset(cases=new_cases, evaluator=new_evaluator)

    async def update_current_dataset_async(
        self,
        source_dataset: Dataset,
        task_description: str,
        num_cases: int = 5,
        context: str | None = None,
        add_new_cases: bool = True,
        add_new_rubric: bool = True,
        new_evaluator_type: type | None = None,
    ) -> Dataset:
        """
        Update an existing dataset by adding new test cases and/or updating the evaluator.

        Extends the source dataset with additional test cases that complement the existing ones,
        and optionally updates the evaluation rubric. Useful for iteratively improving datasets
        or adapting them to new requirements while preserving the original test cases.

        Args:
            source_dataset: Original dataset to extend and update
            task_description: Description of the task the AI system will perform
            num_cases: Number of new test cases to add (if add_new_cases is True)
            context: Additional context or requirements for new test cases and rubric,
                be sure to include as much information as you can about tools or sub-agents
                for generating interaction and/or trajectory.
            add_new_cases: Whether to generate and add new test cases to the dataset
            add_new_rubric: Whether to generate a new evaluation rubric
            new_evaluator_type: Optional new evaluator type to use instead of the source evaluator type

        Returns:
            Updated Dataset containing original cases plus new cases (if requested) and
            updated evaluator with new rubric (if requested and evaluator supports it).
        """
        source_cases = source_dataset.cases
        source_evaluator = source_dataset.evaluator

        if add_new_cases:
            # construct messages to initialize the agent with context about the previous test cases
            messages = [{"role": "user", "content": [{"text": "Here are the current test cases: "}]}]
            cases_string_list = []
            for i, case in enumerate(source_cases):
                cases_string_list.append({"text": f"{i}. {case.model_dump()}"})
            messages.append({"role": "user", "content": cases_string_list})
            new_cases = await self.generate_cases_async(
                prompt=(
                    f"Create new test cases, expanding on previous cases for the following context: {context}. "
                    f"Ensure that the input and output are relevant for this task: {task_description}."
                ),
                num_cases=num_cases,
                message_history=messages,
            )

        if add_new_rubric:
            evaluator_type: type
            if new_evaluator_type:
                evaluator_type = new_evaluator_type
            else:
                evaluator_type = type(source_evaluator)  # use the previous evaluator if no new evaluator is passed in

            if evaluator_type in self._default_evaluators:
                source_rubric = source_evaluator.rubric if type(source_evaluator) in self._default_evaluators else None
                new_evaluator = await self.construct_evaluator_async(
                    prompt=(
                        f"Create a new rubric based on the reference rubric if provided for the following "
                        f"context: {context}. Ensure that the rubric is relevant for this task: {task_description}."
                    ),
                    evaluator=evaluator_type,
                    message_history=[{"role": "user", "content": [{"text": source_rubric}]}],
                )
            else:  # use the original if it's not supported
                new_evaluator = source_evaluator

        return Dataset(
            cases=source_cases + new_cases if add_new_cases else source_cases,
            evaluator=new_evaluator if add_new_rubric else source_evaluator,
        )
