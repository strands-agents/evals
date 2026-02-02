from enum import Enum

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import TypeVar, Union

from ..types.evaluation import EvaluationData, EvaluationOutput
from .evaluator import Evaluator
from .prompt_templates.contextual_faithfulness import get_template

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ContextualFaithfulnessScore(str, Enum):
    """Categorical contextual faithfulness ratings for RAG evaluation."""

    NOT_FAITHFUL = "Not Faithful"
    PARTIALLY_FAITHFUL = "Partially Faithful"
    MOSTLY_FAITHFUL = "Mostly Faithful"
    FULLY_FAITHFUL = "Fully Faithful"


class ContextualFaithfulnessRating(BaseModel):
    """Structured output for contextual faithfulness evaluation."""

    reasoning: str = Field(description="Step by step reasoning analyzing each claim against the retrieval context")
    score: ContextualFaithfulnessScore = Field(description="Categorical faithfulness rating")


class ContextualFaithfulnessEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates whether an LLM response is faithful to the provided retrieval context.

    This evaluator is designed for RAG (Retrieval-Augmented Generation) systems.
    It checks if the claims in the response are grounded in the retrieved documents,
    helping detect hallucinations where the model generates information not present
    in the context.

    Unlike FaithfulnessEvaluator which checks against conversation history,
    this evaluator specifically validates against retrieval context provided
    in the test case.

    Attributes:
        version: The version of the prompt template to use.
        model: A string representing the model-id for Bedrock to use, or a Model instance.
        system_prompt: System prompt to guide model behavior.
        include_input: Whether to include the user's input query in the evaluation prompt.

    Example:
        evaluator = ContextualFaithfulnessEvaluator()
        case = Case(
            input="What is the refund policy?",
            retrieval_context=[
                "Refunds are available within 30 days of purchase.",
                "Items must be unopened for a full refund."
            ]
        )
        # Run with experiment or evaluate directly
    """

    _score_mapping = {
        ContextualFaithfulnessScore.NOT_FAITHFUL: 0.0,
        ContextualFaithfulnessScore.PARTIALLY_FAITHFUL: 0.33,
        ContextualFaithfulnessScore.MOSTLY_FAITHFUL: 0.67,
        ContextualFaithfulnessScore.FULLY_FAITHFUL: 1.0,
    }

    def __init__(
        self,
        version: str = "v0",
        model: Union[Model, str, None] = None,
        system_prompt: str | None = None,
        include_input: bool = True,
    ):
        super().__init__()
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model
        self.include_input = include_input

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Evaluate the contextual faithfulness of the response.

        Args:
            evaluation_case: The test case containing the response and retrieval context.

        Returns:
            A list containing a single EvaluationOutput with the faithfulness score.

        Raises:
            ValueError: If retrieval_context is not provided in the evaluation case.
        """
        self._validate_evaluation_case(evaluation_case)
        prompt = self._format_prompt(evaluation_case)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        rating = evaluator_agent.structured_output(ContextualFaithfulnessRating, prompt)
        return [self._create_output(rating)]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Evaluate the contextual faithfulness of the response asynchronously.

        Args:
            evaluation_case: The test case containing the response and retrieval context.

        Returns:
            A list containing a single EvaluationOutput with the faithfulness score.

        Raises:
            ValueError: If retrieval_context is not provided in the evaluation case.
        """
        self._validate_evaluation_case(evaluation_case)
        prompt = self._format_prompt(evaluation_case)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        rating = await evaluator_agent.structured_output_async(ContextualFaithfulnessRating, prompt)
        return [self._create_output(rating)]

    def _validate_evaluation_case(self, evaluation_case: EvaluationData[InputT, OutputT]) -> None:
        """Validate that the evaluation case has required fields."""
        if not evaluation_case.retrieval_context:
            raise ValueError(
                "retrieval_context is required for ContextualFaithfulnessEvaluator. "
                "Please provide retrieval_context in your Case."
            )
        if evaluation_case.actual_output is None:
            raise ValueError(
                "actual_output is required for ContextualFaithfulnessEvaluator. "
                "Please make sure the task function returns the output."
            )

    def _format_prompt(self, evaluation_case: EvaluationData[InputT, OutputT]) -> str:
        """Format the evaluation prompt with context and response."""
        parts = []

        if self.include_input:
            parts.append(f"# User Query:\n{evaluation_case.input}")

        context_str = "\n\n".join(
            f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(evaluation_case.retrieval_context or [])
        )
        parts.append(f"# Retrieval Context:\n{context_str}")

        parts.append(f"# Assistant's Response:\n{evaluation_case.actual_output}")

        return "\n\n".join(parts)

    def _create_output(self, rating: ContextualFaithfulnessRating) -> EvaluationOutput:
        """Create an EvaluationOutput from the rating."""
        normalized_score = self._score_mapping[rating.score]
        return EvaluationOutput(
            score=normalized_score,
            test_pass=normalized_score >= 0.67,
            reason=rating.reasoning,
            label=rating.score,
        )
