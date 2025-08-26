import asyncio

from strands import Agent, tool
from typing_extensions import TypedDict

from strands_evals.evaluators import TrajectoryEvaluator
from strands_evals.extractors import tools_use_extractor
from strands_evals.generators import DatasetGenerator
from strands_evals.types import EvaluationReport, TaskOutput

balances_reference = {"Anna": -100, "Cindy": 800, "Brian": 300, "Hailey": 0}


async def trajectory_dataset_generator() -> EvaluationReport:
    """
    Demonstrates generating a dataset for bank tools trajectory evaluation.

    This function:
    1. Defines a task function that uses banking tools
    2. Creates a DatasetGenerator for trajectory evaluation
    3. Generates a dataset from banking-related topics
    4. Runs evaluations on the generated test cases

    Returns:
        EvaluationReport: Results of running the generated test cases
    """

    ### Step 1: Define task ###
    async def bank_task(query: str) -> TaskOutput:
        """
        Banking task that handles spending, balance checks, and debt collection.
        """
        # Bank account balances
        balances = {"Anna": -100, "Cindy": 800, "Brian": 300, "Hailey": 0}

        @tool
        def get_balance(person: str) -> int:
            """Get the balance of a bank account."""
            return balances.get(person, 0)

        @tool
        def modify_balance(person: str, amount: int) -> None:
            """Modify the balance of a bank account by a given amount."""
            balances[person] += amount

        @tool
        def collect_debt() -> list[tuple]:
            """Check all bank accounts for any debt."""
            debt = []
            for person in balances:
                if balances[person] < 0:
                    debt.append((person, abs(balances[person])))
            return debt

        bank_prompt = (
            "You are a banker. Ensure only people with sufficient balance can spend money. "
            "Collect debt from people with negative balance. "
            "Report the current balance of the person of interest after all actions."
        )
        agent = Agent(
            tools=[get_balance, modify_balance, collect_debt], system_prompt=bank_prompt, callback_handler=None
        )
        response = await agent.invoke_async(query)
        trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(agent.messages)
        return TaskOutput(output=str(response), trajectory=trajectory)

    ### Step 2: Initialize the dataset generator ###
    class ToolType(TypedDict):
        name: str
        input: dict

    generator = DatasetGenerator[str, str](str, str, include_expected_trajectory=True, trajectory_type=ToolType)

    ### Step 3: Generate dataset with tool context ###
    tool_context = """
    Available banking tools:
    - get_balance(person: str) -> int: Get the balance of a bank account for a specific person
    - modify_balance(person: str, amount: int) -> None: Modify the balance by adding/subtracting an amount
    - collect_debt() -> list[tuple]: Check all accounts and return list of people with negative balances
        and their debt amounts
    
    Banking rules:
    - Only allow spending if person has sufficient balance
    - Collect debt from people with negative balances
    - Always report final balance after transactions
    """

    dataset = await generator.from_context_async(
        context=tool_context,
        num_cases=5,
        evaluator=TrajectoryEvaluator,  # type: ignore
        task_description=(
            f"Banking operations with balance checks, spending, and debt collection with these people: "
            f"{balances_reference}"
        ),
    )

    ### Step 3.5: (Optional) Save the generated dataset ###
    dataset.to_file("generated_bank_trajectory_dataset")

    ### Step 4: Run evaluations on the generated test cases ###
    report = await dataset.run_evaluations_async(bank_task)
    return report


if __name__ == "__main__":
    # python -m examples.dataset_generator.trajectory_dataset
    report = asyncio.run(trajectory_dataset_generator())
    report.to_file("generated_bank_trajectory_report_horizontal", is_vertical=False)
    report.to_file("generated_bank_trajectory_report_vertical")

    report.run_display(include_actual_trajectory=True)
