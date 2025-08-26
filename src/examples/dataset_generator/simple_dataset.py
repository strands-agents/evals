import asyncio

from strands import Agent
from strands_evals.evaluators.output_evaluator import OutputEvaluator
from strands_evals.generators.dataset_generator import DatasetGenerator
from strands_evals.types import EvaluationReport


async def simple_dataset_generator() -> EvaluationReport:
    """
    Demonstrates the a simple dataset generation and evaluation process.

    This function:
    1. Defines a task function that uses an agent to generate responses
    2. Creates a DatasetGenerator for string input/output types
    3. Generates a dataset from scratch based on specified topics
    4. Runs evaluations on the generated test cases

    Returns:
        EvaluationReport: Results of running the generated test cases
    """

    ### Step 1: Define task ###
    async def get_response(query: str) -> str:
        """
        Simple task example to get a response from an agent given a query.
        """
        agent = Agent(system_prompt="Be as concise as possible", callback_handler=None)
        response = await agent.invoke_async(query)
        return str(response)

    # Step 2: Initialize the dataset generator for string types
    generator = DatasetGenerator[str, str](str, str)

    # Step 3: Generate dataset from scratch with specified topics
    # This will create test cases and a rubric automatically
    dataset = await generator.from_scratch_async(
        topics=["safety", "red teaming", "leetspeak"],  # Topics to cover in test cases
        task_description="Getting response from an agent given a query",  # What the AI system does
        num_cases=10,  # Number of test cases to generate
        evaluator=OutputEvaluator,  # Type of evaluator to create with generated rubric
    )

    # Step 3.5: (Optional) Save the generated dataset for future use
    dataset.to_file("generate_simple_dataset")

    # Step 4: Run evaluations on the generated test cases
    report = await dataset.run_evaluations_async(get_response)
    return report


if __name__ == "__main__":
    # python -m examples.dataset_generator.simple_dataset
    report = asyncio.run(simple_dataset_generator())
    report.to_file("generated_safety_judge_output_report", "json")
    report.run_display(include_actual_output=True)
