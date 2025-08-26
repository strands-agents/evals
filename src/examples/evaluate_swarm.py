import asyncio
import datetime

from strands import Agent
from strands.multiagent import Swarm

from strands_evals import Case, Dataset
from strands_evals.evaluators import InteractionsEvaluator, TrajectoryEvaluator
from strands_evals.extractors import swarm_extractor
from strands_evals.types import EvaluationReport


async def async_swarm_example() -> tuple[EvaluationReport, EvaluationReport]:
    """
    Demonstrates evaluating swarm agent interactions and trajectories for content creation tasks.

    This example:
    1. Defines a task function with a swarm of specialized content creation agents
    2. Creates test cases for content creation scenarios
    3. Creates TrajectoryEvaluator and InteractionsEvaluator to assess agent handoffs
    4. Creates datasets with the test cases and evaluators
    5. Runs evaluations and analyzes the reports

    Returns:
        tuple[EvaluationReport, EvaluationReport]: The trajectory and interaction evaluation results
    """

    ### Step 1: Define task ###
    async def content_creation_swarm(task: str) -> dict:
        # Create specialized content creation agents
        researcher = Agent(
            name="researcher",
            system_prompt=(
                "You are a content research specialist. Gather relevant information, "
                "statistics, and sources for the given topic. Provide comprehensive "
                "research that will inform high-quality content creation."
            ),
            callback_handler=None,
        )
        writer = Agent(
            name="writer",
            system_prompt=(
                "You are a content writer. Create engaging, well-structured content based on research "
                "provided. Focus on clear messaging, compelling narratives, and audience engagement."
            ),
            callback_handler=None,
        )
        editor = Agent(
            name="editor",
            system_prompt=(
                "You are a content editor. Review and refine written content for "
                "clarity, flow, grammar, and style. Ensure the content meets quality "
                "standards and is publication-ready."
            ),
            callback_handler=None,
        )
        seo_specialist = Agent(
            name="seo_specialist",
            system_prompt=(
                "You are an SEO specialist. Optimize content for search engines by adding "
                "relevant keywords and meta descriptions."
            ),
            callback_handler=None,
        )

        # Create a swarm with these agents
        swarm = Swarm(
            [researcher, writer, editor, seo_specialist],
            max_handoffs=20,
            max_iterations=20,
            execution_timeout=900.0,  # 15 minutes
            node_timeout=300.0,  # 5 minutes per agent
            repetitive_handoff_detection_window=2,  # There must be >= 2 unique agents in the last 8 handoffs
            repetitive_handoff_min_unique_agents=2,
        )

        result = await swarm.invoke_async(task)
        interaction_info = swarm_extractor.extract_swarm_interactions(result)

        return {"interactions": interaction_info, "trajectory": [node.node_id for node in result.node_history]}

    ### Step 2: Create test cases ###
    test_cases: list[Case] = [
        Case(
            input="Create a comprehensive blog post about the benefits of remote work for productivity.",
            expected_trajectory=["researcher", "writer", "editor", "seo_specialist"],
        ),
        Case(
            input="Write an article about sustainable energy solutions for small businesses.",
            expected_trajectory=["researcher", "writer", "editor", "seo_specialist"],
        ),
    ]

    ### Step 3: Create evaluator ###
    node_descriptions = {
        "researcher": (
            "The researcher node should gather relevant information, statistics, and sources for the given topic."
        ),
        "writer": "The writer node should create engaging, well-structured content based on research provided.",
        "editor": "The editor node should review and refine written content for clarity, flow, grammar, and style.",
        "seo_specialist": (
            "The SEO specialist should optimize content for search engines by "
            "adding relevant keywords and meta descriptions."
        ),
    }
    interaction_evaluator: InteractionsEvaluator = InteractionsEvaluator(
        rubric=(
            "Scoring should measure how well each agent handoff follows logical content creation workflow. "
            "Score 1.0 if handoffs are appropriate, include relevant context, and demonstrate clear content "
            "improvement. Score 0.5 if partially logical, 0.0 if illogical or missing context."
        ),
        interaction_description=node_descriptions,
    )
    trajectory_evaluator: TrajectoryEvaluator = TrajectoryEvaluator(
        rubric=(
            "Scoring should measure how well the swarm utilizes the right sequence of agents for content creation. "
            "Score 1.0 if trajectory follows expected workflow, 0.5 if partially correct sequence, "
            "0.0 if incorrect or inefficient agent usage."
        )
    )

    ### Step 4: Create dataset ###
    trajectory_dataset = Dataset(cases=test_cases, evaluator=trajectory_evaluator)
    interaction_dataset = Dataset(cases=test_cases, evaluator=interaction_evaluator)

    ### Step 5: Run evaluation ###
    trajectory_report = await trajectory_dataset.run_evaluations_async(content_creation_swarm)
    interaction_report = await interaction_dataset.run_evaluations_async(content_creation_swarm)
    return trajectory_report, interaction_report


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.evaluate_swarm
    start = datetime.datetime.now()
    trajectory_report, interaction_report = asyncio.run(async_swarm_example())
    end = datetime.datetime.now()

    trajectory_report.to_file("content_creation_swarm_trajectory_report")
    interaction_report.to_file("content_creation_swarm_interaction_report")

    trajectory_report.run_display(include_actual_trajectory=True)
    interaction_report.run_display(include_actual_interactions=True, include_expected_interactions=True)
