from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from typing_extensions import TypeVar

from ..types.evaluation import EvaluationData, EvaluationOutput
from .evaluator import Evaluator
from .prompt_templates.prompt_templates import judge_interactions_template as SYSTEM_PROMPT

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class InteractionsEvaluator(Evaluator[InputT, OutputT]):
    """
    An evaluator that is designed for evaluating interactions between agents or components.

    Attributes:
        rubric: The user-specified criteria for evaluating a collection of test cases.
            if the rubric is a string, then use the same rubric for all of the evaluations, else
            get the node-specific rubric for evaluation.
        interaction_description: A dictionary describing the evailable interactions.
        model: A string representing the model-id for Bedrock to use.
                    Defaults to strands.models.BedrockModel if None.
        system_prompt: System prompt to guide model behavior.
                    If None, the evaluator will use one of the default template.
        include_inputs: Whether to include inputs to the task in the evaluation or not.
    """

    def __init__(
        self,
        rubric: str | dict[str, str],
        interaction_description: dict | None = None,
        model: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        include_inputs: bool = True,
    ):
        super().__init__()
        self.rubric = rubric
        self.interaction_description = interaction_description
        self.model = model
        self.include_inputs = include_inputs
        self.system_prompt = system_prompt

    def update_interaction_description(self, new_description: dict) -> None:
        """
        Update the description of the available interactions.

        Args:
            new_description: The new description of the available interactions.
        """
        self.interaction_description = new_description

    def _get_node_rubric(self, node_name: str) -> str:
        """
        Get the rubric for the node involved in the interaction.

        Args:
            node_name: The node involved in the interaction.

        Returns:
            The rubric for the given evaluation case.

        Error:
            If the rubric is a dictionary, then expect it to contain the keys for every node.
        """
        if isinstance(self.rubric, dict):  # rubric for each node
            rubric = self.rubric.get(node_name, None)
            if rubric is None:
                raise KeyError(f"Please make sure the rubric dictionary contains the key '{node_name}'.")
            return rubric

        return self.rubric  # use the same rubric for all of the nodes

    def _compose_prompt(
        self, evaluation_case: EvaluationData[InputT, OutputT], current_case_i: int, is_last: bool
    ) -> str:
        """
        Compose the prompt for the given evaluation case.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.
            current_case_i: The index of the current interaction in the list of interactions.
            is_last: Whether the current interaction is the last interaction in the list of interactions.

        Returns:
            The prompt for the given evaluation case.
        """
        if is_last:
            evaluation_prompt = (
                "Evaluate this final interaction. THE FINAL SCORE MUST BE A DECIMAL BETWEEN 0.0 AND 1.0 "
                "(NOT 0 to 10 OR 0 to 100). Your reasoning should include information from all of the "
                "previous interactions evaluated.\n"
            )
        else:
            evaluation_prompt = (
                "Evaluate this interaction. THE SCORE MUST BE A DECIMAL BETWEEN 0.0 AND 1.0 "
                "(NOT 0 to 10 OR 0 to 100). \n"
            )

        if self.include_inputs:
            if (
                isinstance(evaluation_case.input, list)
                and isinstance(evaluation_case.actual_interactions, list)
                and len(evaluation_case.input) == len(evaluation_case.actual_interactions)
            ):
                evaluation_prompt += f"<Input>{evaluation_case.input[current_case_i]}</Input>\n"
            elif current_case_i == 0:  # only include the input for the first interaction
                evaluation_prompt += f"<Input>{evaluation_case.input}</Input>\n"

        interaction = (
            evaluation_case.actual_interactions[current_case_i]
            if evaluation_case.actual_interactions is not None
            else {}
        )
        node_name = interaction.get("node_name", None)
        dependencies = interaction.get("dependencies", None)
        messages = interaction.get("messages", None)
        if node_name is None and dependencies is None and messages is None:
            raise KeyError(
                "Please make sure the task function returns a dictionary with the key 'interactions' "
                "that contains a list of Interactions with 'node_name', and/or 'dependencies', "
                "and/or 'messages'."
            )

        evaluation_prompt += (
            f"<Interaction> Node Name: {node_name}, Depends on {dependencies} \n Message: {messages} </Interaction>\n"
        )

        if evaluation_case.expected_interactions:
            expected_interactions_count = len(evaluation_case.expected_interactions)
            expected_nodes_sequence = [
                i.get("node_name") for i in evaluation_case.expected_interactions
            ]  # quick overview of the whole sequence
            evaluation_prompt += f"<ExpectedSequence>{expected_nodes_sequence}</ExpectedSequence>\n"
            # include a short window of interactions that may be relevant (at most 3)
            relevant_expected_interactions = evaluation_case.expected_interactions[
                max(0, current_case_i - 1) : min(expected_interactions_count, current_case_i + 2)
            ]
            for relevant_expected_interaction in relevant_expected_interactions:
                e_node_name = relevant_expected_interaction.get("node_name", None)
                e_dependencies = relevant_expected_interaction.get("dependencies", None)
                e_messages = relevant_expected_interaction.get("messages", None)
                evaluation_prompt += (
                    f"<RelevantExpectedInteraction> Node Name: {e_node_name}, "
                    f"Depends on {e_dependencies}, Message: {e_messages} </RelevantExpectedInteraction>\n"
                )

        if is_last:  # only include the actual output of the whole interaction in the last interaction
            if evaluation_case.actual_output:
                evaluation_prompt += f"<Output>{evaluation_case.actual_output}</Output>\n"
            if evaluation_case.expected_output:
                evaluation_prompt += f"<ExpectedOutput>{evaluation_case.expected_output}</ExpectedOutput>\n"

        if self.interaction_description:
            evaluation_prompt += f"<InteractionDescription>{self.interaction_description}</InteractionDescription>\n"

        if node_name is not None:
            evaluation_prompt += f"<Rubric>{self._get_node_rubric(node_name)}</Rubric>"

        return evaluation_prompt

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        """
        Evaluate the performance of the task on the given test cases.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Returns:
            The results of the evaluation as EvaluationOutput.
        """
        if evaluation_case.actual_interactions is None:
            raise KeyError(
                "Please make sure the task function returns a dictionary with the key 'interactions' "
                "of type list[Interaction]."
            )
        num_interactions = len(evaluation_case.actual_interactions)

        # keep all of the context
        conversation_manager = SlidingWindowConversationManager(window_size=num_interactions)
        evaluator_agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            callback_handler=None,
            conversation_manager=conversation_manager,
        )

        is_last = False
        result: EvaluationOutput | None = None
        for i in range(num_interactions):  # evaluate one interaction at a time
            if i == num_interactions - 1:
                is_last = True
            evaluation_prompt = self._compose_prompt(evaluation_case, i, is_last)

            ## Evaluate ##
            result = evaluator_agent.structured_output(EvaluationOutput, evaluation_prompt)

        if result is None:
            return EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="No interactions were evaluated. Ensure actual_interactions is not empty.",
            )

        return result

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        """
        Evaluate the performance of the task on the given test cases asynchronously.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Returns:
            The results of the evaluation as EvaluationOutput.
        """
        if not evaluation_case.actual_interactions:
            raise KeyError("Please make sure the task function returns a dictionary with the key 'interactions'.")
        num_interactions = len(evaluation_case.actual_interactions)

        # keep all of the context
        conversation_manager = SlidingWindowConversationManager(window_size=num_interactions)
        evaluator_agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            callback_handler=None,
            conversation_manager=conversation_manager,
        )

        is_last = False
        result: EvaluationOutput | None = None
        for i in range(num_interactions):  # evaluate one interaction at a time
            if i == num_interactions - 1:
                is_last = True

            evaluation_prompt = self._compose_prompt(evaluation_case, i, is_last)

            ## Evaluate ##
            result = await evaluator_agent.structured_output_async(EvaluationOutput, evaluation_prompt)

        if result is None:
            return EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="No interactions were evaluated. Ensure actual_interactions is not empty.",
            )

        return result
