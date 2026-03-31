"""Red team runner utilities.

Provides helper functions for generating red team test cases, building
multi-turn task functions, and running end-to-end red team evaluations.
"""

import logging
import random
from collections.abc import Callable

from strands_evals.case import Case
from strands_evals.evaluators.red_team_judge_evaluator import RedTeamJudgeEvaluator
from strands_evals.experiment import Experiment
from strands_evals.redteam.presets import ATTACK_REGISTRY
from strands_evals.simulation.actor_simulator import ActorSimulator
from strands_evals.types.evaluation_report import EvaluationReport
from strands_evals.types.simulation import ActorProfile

logger = logging.getLogger(__name__)


def generate_cases(
    attack_types: list[str],
    n_per_type: int = 5,
    target_info: dict | None = None,
) -> list[Case]:
    """Generate red team test cases from attack presets.

    Each case maps to one seed input from the specified attack types.
    Seed inputs are sampled (with repetition if n_per_type > available seeds).

    Args:
        attack_types: List of attack type names (keys in ATTACK_REGISTRY).
        n_per_type: Number of cases to generate per attack type.
        target_info: Optional target metadata (description, system_prompt, tools).

    Returns:
        List of Case objects with attack metadata embedded.

    Raises:
        ValueError: If an attack type is not found in ATTACK_REGISTRY.
    """
    cases = []

    for attack_type in attack_types:
        if attack_type not in ATTACK_REGISTRY:
            raise ValueError(
                f"Unknown attack type: '{attack_type}'. "
                f"Available types: {list(ATTACK_REGISTRY.keys())}"
            )

        preset = ATTACK_REGISTRY[attack_type]
        seed_inputs = preset["seed_inputs"]

        # Sample seeds: use all if n_per_type <= len, otherwise sample with replacement
        if n_per_type <= len(seed_inputs):
            selected_seeds = random.sample(seed_inputs, n_per_type)
        else:
            selected_seeds = seed_inputs + random.choices(
                seed_inputs, k=n_per_type - len(seed_inputs)
            )

        for i, seed in enumerate(selected_seeds):
            metadata = {
                "attack_type": attack_type,
                "traits": preset["traits"],
                "context": preset["context"],
                "actor_goal": preset["actor_goal"],
                "system_prompt_template": preset["system_prompt_template"],
                "severity": preset["severity"],
            }
            if target_info:
                metadata["target_info"] = target_info

            cases.append(
                Case(
                    name=f"{attack_type}_{i}",
                    input=seed,
                    metadata=metadata,
                )
            )

    return cases


def build_task_function(
    target: Callable,
    max_turns: int = 10,
    model: str | None = None,
) -> Callable:
    """Build a task function that runs multi-turn red team conversations.

    The returned function creates an ActorSimulator for each case and
    orchestrates the adversarial conversation loop with the target.

    Args:
        target: Callable that takes a string message and returns a string response.
            Typically a lambda wrapping an Agent call (e.g., lambda msg: str(agent(msg))).
        max_turns: Maximum conversation turns per case.
        model: Model ID for the ActorSimulator's underlying agent.

    Returns:
        A task function compatible with Experiment.run_evaluations().
    """

    def task_fn(case: Case) -> dict:
        metadata = case.metadata or {}

        # Build actor profile from case metadata
        actor_profile = ActorProfile(
            traits=metadata.get("traits", {}),
            context=metadata.get("context", ""),
            actor_goal=metadata.get("actor_goal", ""),
        )

        system_prompt_template = metadata.get("system_prompt_template", "")

        # Create adversarial simulator
        simulator = ActorSimulator(
            actor_profile=actor_profile,
            initial_query=case.input,
            system_prompt_template=system_prompt_template,
            model=model,
            max_turns=max_turns,
        )

        # Run multi-turn conversation
        conversation = []
        attacker_message = case.input

        while simulator.has_next():
            # Target agent responds
            try:
                target_response = str(target(attacker_message))
            except Exception as e:
                logger.warning("Target agent error on turn %d: %s", len(conversation), e)
                target_response = f"[Error: {e}]"

            conversation.append({"role": "attacker", "content": attacker_message})
            conversation.append({"role": "target", "content": target_response})

            # Check if simulator wants to continue before generating next attack
            if not simulator.has_next():
                break

            # Attacker generates next message
            attacker_result = simulator.act(target_response)
            attacker_message = str(attacker_result.structured_output.message)

        return {
            "output": conversation,
        }

    return task_fn


def run_red_team(
    attack_types: list[str],
    target: Callable,
    target_info: dict | None = None,
    n_per_type: int = 5,
    max_turns: int = 10,
    evaluators: list | None = None,
    model: str | None = None,
) -> list[EvaluationReport]:
    """Run an end-to-end red team evaluation.

    This is the built-in one-liner for red team testing. It generates cases,
    builds a task function, creates an Experiment, and runs evaluations.

    Args:
        attack_types: List of attack type names (e.g., ["jailbreak", "prompt_extraction"]).
        target: Callable that takes a string and returns a string response.
        target_info: Optional target metadata for context-aware attacks.
        n_per_type: Number of test cases per attack type.
        max_turns: Maximum conversation turns per case.
        evaluators: List of evaluator instances. If None, uses default RedTeamJudgeEvaluator.
        model: Model ID for the ActorSimulator's underlying agent.

    Returns:
        List of EvaluationReport, one per evaluator.

    Example:
        ```python
        from strands import Agent
        from strands_evals.redteam import run_red_team

        agent = Agent(system_prompt="You are a helpful assistant.")

        reports = run_red_team(
            attack_types=["jailbreak", "prompt_extraction"],
            target=lambda msg: str(agent(msg)),
            n_per_type=3,
            max_turns=5,
        )

        for report in reports:
            report.display()
        ```
    """
    # Generate cases
    cases = generate_cases(
        attack_types=attack_types,
        n_per_type=n_per_type,
        target_info=target_info,
    )

    # Build task function
    task_fn = build_task_function(
        target=target,
        max_turns=max_turns,
        model=model,
    )

    # Set up evaluators
    if evaluators is None:
        evaluators = [RedTeamJudgeEvaluator()]

    # Run experiment
    experiment = Experiment(cases=cases, evaluators=evaluators)
    reports = experiment.run_evaluations(task_fn)

    return reports
