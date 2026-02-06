import asyncio
import logging
import math
from textwrap import dedent

from pydantic import create_model
from strands import Agent
from typing_extensions import Any, Generic

from strands_evals.evaluators import Evaluator, InteractionsEvaluator, OutputEvaluator, TrajectoryEvaluator

from ..case import Case
from ..experiment import Experiment
from ..types.evaluation import InputT, Interaction, OutputT
from .prompt_template.prompt_templates import generate_case_template as CASE_SYSTEM_PROMPT
from .prompt_template.prompt_templates import generate_rubric_template as RUBRIC_SYSTEM_PROMPT
from .topic_planner import TopicPlanner

logger = logging.getLogger(__name__)


class ExperimentGenerator(Generic[InputT, OutputT]):
    """
    Generates evaluation experiments with test cases and rubrics for LLM-based evaluators for agent assessment.

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
        Initialize the experiment generator with configuration for test case structure.

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
        self, prompt: str, num_cases: int = 5, message_history: list | None = None, num_topics: int | None = None
    ) -> list[Case]:
        """
        Generate test cases asynchronously using parallel workers.

        Args:
            prompt: Generation prompt describing the test case requirements
            num_cases: Number of test cases to generate
            message_history: Optional conversation history to provide context to the generation agent
            num_topics: Optional number of topics for diverse coverage.
                If None, generates all cases from the single prompt.
                If specified, expands prompt into multiple topic-specific prompts.

        Returns:
            List of generated Case objects matching the configured schema
        """
        prompt_specs = await self._prepare_generation_prompts(
            base_prompt=prompt, num_cases=num_cases, num_topics=num_topics
        )

        generated_cases: list = []
        for prompt_text, cases_for_prompt in prompt_specs:
            cases = await self._generate_batch(
                prompt=prompt_text, num_cases=cases_for_prompt, message_history=message_history
            )
            generated_cases.extend(cases)

        return generated_cases

    async def _prepare_generation_prompts(
        self, base_prompt: str, num_cases: int, num_topics: int | None = None
    ) -> list[tuple[str, int]]:
        """
        Prepare generation prompts, optionally expanding via topic planning.

        Returns:
            List of (prompt, num_cases) tuples. Always returns at least one prompt.
        """
        if num_topics is None:
            return [(base_prompt, num_cases)]

        topic_planner = TopicPlanner(model=self.model)

        try:
            topic_plan = await topic_planner.plan_topics_async(
                context=base_prompt, task_description="", num_topics=num_topics, num_cases=num_cases
            )
        except Exception as e:
            logger.warning(f"Topic planning failed: {e}. Using single prompt.")
            return [(base_prompt, num_cases)]

        # Distribute cases across topics
        cases_per_topic = math.ceil(num_cases / len(topic_plan.topics))
        prompt_specs: list[tuple[str, int]] = []

        num_generated_cases = 0
        for topic in topic_plan.topics:
            remaining = num_cases - num_generated_cases
            if remaining <= 0:
                break

            topic_cases = min(cases_per_topic, remaining)
            topic_prompt = dedent(f"""
                {base_prompt}
                Focus on this topic:
                - {topic.title}: {topic.description}
                - Key aspects: {", ".join(topic.key_aspects)}
            """)

            prompt_specs.append((topic_prompt, topic_cases))
            num_generated_cases += topic_cases

        return prompt_specs

    async def _generate_batch(self, prompt: str, num_cases: int, message_history: list | None = None) -> list[Case]:
        """Generate a batch of cases using the existing worker pattern."""
        queue: asyncio.Queue[str] = asyncio.Queue()
        generated_cases: list = []

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
    ) -> Experiment:
        """
        Generate an experiment from scratch based on specified topics and task description.

        Creates diverse test cases covering the given topics for the specified task,
        with optional evaluator and rubric generation.

        Args:
            topics: List of topics/domains to cover in test cases
            task_description: Description of the task the AI system will perform
            num_cases: Number of test cases to generate
            evaluator: Optional evaluator class for assessment (generates rubric if provided).

        Returns:
            Experiment containing generated test cases and evaluator. Use the generic Evaluator as placeholder
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
            return Experiment(cases=cases, evaluators=[_evaluator])
        else:
            return Experiment(cases=cases)

    async def from_context_async(
        self,
        context: str,
        task_description: str,
        num_cases: int = 5,
        evaluator: Evaluator = None,
        num_topics: int | None = None,
    ) -> Experiment:
        """
        Generate an experiment based on specific context that test cases should reference.

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
            num_topics: Optional number of topics for diverse coverage

        Returns:
            Experiment containing context-based test cases and evaluator. Use the generic Evaluator as placeholder
            if no evaluator is passed in.
        """
        cases = await self.generate_cases_async(
            f"""Create test cases with the following context: {context}. Ensure that the questions can be """
            f"""answer using the provided context for this task: {task_description} """,
            num_cases=num_cases,
            num_topics=num_topics,
        )
        if evaluator:
            _evaluator = await self.construct_evaluator_async(
                prompt=f"""Create a rubric with the following context: {context} for this task: """
                f"""{task_description} """,
                evaluator=evaluator,
            )
            return Experiment(cases=cases, evaluators=[_evaluator])
        else:
            return Experiment(cases=cases)

    async def from_experiment_async(
        self,
        source_experiment: Experiment,
        task_description: str,
        num_cases: int = 5,
        extra_information: str | None = None,
    ) -> Experiment:
        """
        Generate a new experiment using an existing experiment as reference.

        Creates new test cases that are similar in style and structure to the source experiment,
        while adapting them for the specified task. If the source experiment uses a default
        evaluator with a rubric, generates a new rubric based on the original.

        Args:
            source_experiment: Original experiment to use as reference for generating new test cases
            task_description: Description of the task the AI system will perform
            num_cases: Number of test cases to generate
            extra_information: Optional additional context or requirements for the new test cases and rubric,
                be sure to include as much information as you can about tools or sub-agents
                for generating interaction and/or trajectory.

        Returns:
            A new Experiment containing test cases inspired by the source experiment but adapted
            for the new task. Uses an updated evaluator with new rubric if the source
            evaluator is a default type, otherwise uses generic Evaluator.
        """
        source_cases = source_experiment.cases
        source_evaluators = source_experiment.evaluators

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
        new_evaluators = []
        for source_evaluator in source_evaluators:
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
                new_evaluators.append(new_evaluator)
            else:
                new_evaluators.append(Evaluator())

        return Experiment(cases=new_cases, evaluators=new_evaluators if new_evaluators else [Evaluator()])

    async def update_current_experiment_async(
        self,
        source_experiment: Experiment,
        task_description: str,
        num_cases: int = 5,
        context: str | None = None,
        add_new_cases: bool = True,
        add_new_rubric: bool = True,
        new_evaluator_type: type | None = None,
    ) -> Experiment:
        """
        Update an existing experiment by adding new test cases and/or updating the evaluator.

        Extends the source experiment with additional test cases that complement the existing ones,
        and optionally updates the evaluation rubric. Useful for iteratively improving experiments
        or adapting them to new requirements while preserving the original test cases.

        Args:
            source_experiment: Original experiment to extend and update
            task_description: Description of the task the AI system will perform
            num_cases: Number of new test cases to add (if add_new_cases is True)
            context: Additional context or requirements for new test cases and rubric,
                be sure to include as much information as you can about tools or sub-agents
                for generating interaction and/or trajectory.
            add_new_cases: Whether to generate and add new test cases to the experiment
            add_new_rubric: Whether to generate a new evaluation rubric
            new_evaluator_type: Optional new evaluator type to use instead of the source evaluator type

        Returns:
            Updated Experiment containing original cases plus new cases (if requested) and
            updated evaluator with new rubric (if requested and evaluator supports it).
        """
        source_cases = source_experiment.cases
        source_evaluators = source_experiment.evaluators

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
            new_evaluators = []
            for source_evaluator in source_evaluators:
                evaluator_type = new_evaluator_type if new_evaluator_type else type(source_evaluator)

                if evaluator_type in self._default_evaluators:
                    source_rubric = (
                        source_evaluator.rubric if type(source_evaluator) in self._default_evaluators else None
                    )
                    new_evaluator = await self.construct_evaluator_async(
                        prompt=(
                            f"Create a new rubric based on the reference rubric if provided for the following "
                            f"context: {context}. Ensure that the rubric is relevant for this task: {task_description}."
                        ),
                        evaluator=evaluator_type,
                        message_history=[{"role": "user", "content": [{"text": source_rubric}]}],
                    )
                    new_evaluators.append(new_evaluator)
                else:
                    new_evaluators.append(source_evaluator)

        return Experiment(
            cases=source_cases + new_cases if add_new_cases else source_cases,
            evaluators=new_evaluators if add_new_rubric else source_evaluators,
        )
