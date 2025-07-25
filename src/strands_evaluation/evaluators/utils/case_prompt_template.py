from ...types.evaluation import EvaluationData
from typing_extensions import TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

def compose_test_prompt(evaluation_case: EvaluationData[InputT, OutputT], rubric: str, include_inputs: bool, uses_trajectory: bool = False) -> str:
    """
    Compose the prompt for a test case evaluation.

    Args:
        evaluation_case: The evaluation data containing input, output, and trajectory information
        rubric: The evaluation criteria to be applied
        include_inputs: Whether to include the input in the prompt
        uses_trajectory: Whether this is a trajectory-based evaluation

    Returns:
        str: The formatted evaluation prompt

    Raises:
        Exception: If actual_output is missing for non-trajectory evaluations
        Exception: If actual_trajectory is missing for trajectory evaluations
    """
    evaluation_prompt = "Evaluate this singular test case. THE FINAL SCORE MUST BE A DECIMAL BETWEEN 0.0 AND 1.0 (NOT 0 to 10 OR 0 to 100). \n"
    if include_inputs:   
        evaluation_prompt += f"<Input>{evaluation_case.input}</Input>\n"

    if uses_trajectory: # trajectory evaluations don't require actual_output
        if evaluation_case.actual_output:
            evaluation_prompt += f"<Output>{evaluation_case.actual_output}</Output>\n"
    else:
        if not evaluation_case.actual_output:
            raise Exception("Please make sure the task function return the output or a dictionary with the key 'output'.")
        evaluation_prompt += f"<Output>{evaluation_case.actual_output}</Output>\n"

    if evaluation_case.expected_output:
        evaluation_prompt += f"<ExpectedOutput>{evaluation_case.expected_output}</ExpectedOutput>\n"

    if uses_trajectory: # trajectory evaluations require actual_trajectory
        if not evaluation_case.actual_trajectory:
            raise Exception("Please make sure the task function return a dictionary with the key 'trajectory'.")
        evaluation_prompt += f"<Trajectory>{evaluation_case.actual_trajectory}</Trajectory>\n"

        if evaluation_case.expected_trajectory:
            evaluation_prompt += f"<ExpectedTrajectory>{evaluation_case.expected_trajectory}</ExpectedTrajectory>\n"

    evaluation_prompt += f"<Rubric>{rubric}</Rubric>"

    return evaluation_prompt
