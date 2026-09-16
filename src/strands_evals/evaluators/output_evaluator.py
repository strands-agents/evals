import json
import logging
from typing import Any, cast

from strands import Agent
from strands.models.model import CacheConfig, Model

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from .evaluator import Evaluator
from .prompt_templates.case_prompt_template import compose_test_prompt
from .prompt_templates.prompt_templates import judge_output_template as SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class OutputEvaluator(Evaluator[InputT, OutputT]):
    """
    An evaluator that is LLM-based.

    Attributes:
        rubric: The user-specified criteria for evaluating a collection of test cases.
        model: A string representing the model-id for Bedrock to use, or a Model instance.
                    Defaults to strands.models.BedrockModel if None.
        system_prompt: System prompt to guide model behavior.
                    If None, the evaluator will use one of the default template.
        include_inputs: Whether to include inputs to the task in the evaluation or not.
        tools: Optional tools for the evaluator agent (e.g., domain-specific verification
                    functions the judge can call). Defaults to None (no tools).
        cache_config: Optional CacheConfig for prompt caching. When set and model is a
                    string, the evaluator creates a BedrockModel with caching enabled so the
                    system prompt (rubric) is cached across evaluations, reducing input token
                    consumption by up to 90% for the cached prefix. Defaults to None.
    """

    def __init__(
        self,
        rubric: str,
        model: Model | str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        include_inputs: bool = True,
        uses_environment_state: bool = False,
        name: str | None = None,
        tools: list[Any] | None = None,
        cache_config: CacheConfig | None = None,
    ):
        super().__init__(name=name)
        self.rubric = rubric
        self.model = model
        self.include_inputs = include_inputs
        self.system_prompt = system_prompt
        self.uses_environment_state = uses_environment_state
        self.cache_config = cache_config
        # Stored privately so the base to_dict() skips it; to_dict() below re-adds
        # the JSON-serializable subset so tools like module path strings round-trip.
        self._tools = tools

    @property
    def tools(self) -> list[Any] | None:
        """Optional tools for the evaluator agent.

        Only tools that can be written as valid JSON in a utf-8 file survive `to_dict()`,
        for example module path strings. Tools that cannot (callables, NaN or Infinity
        floats, strings with unpaired surrogates) are skipped with a warning and must be
        re-attached after `from_dict()`.
        """
        return self._tools

    @tools.setter
    def tools(self, value: list[Any] | None) -> None:
        self._tools = value

    def _resolve_model(self) -> Model | str | None:
        """Resolve the model, applying cache_config when the model is a string ID.

        When cache_config is set and model is a plain string (model ID), creates a
        BedrockModel with caching enabled. This allows the system prompt (rubric) to be
        cached across evaluation calls within the provider's cache window, reducing input
        token consumption for repeated evaluations with the same rubric.

        Returns:
            The resolved model: a BedrockModel with caching if applicable, or the
            original model value unchanged.
        """
        if self.cache_config is not None and isinstance(self.model, str):
            from strands.models.bedrock import BedrockModel

            return BedrockModel(model_id=self.model, cache_config=self.cache_config)
        return self.model

    def to_dict(self) -> dict:
        """
        Convert the evaluator into a dictionary.

        Returns:
            dict: A dictionary containing the evaluator's information. Includes only tools
            that can be written as valid JSON in a utf-8 file. Tools that cannot (decorated
            functions, NaN or Infinity floats, strings with unpaired surrogates) are skipped
            with a warning and must be re-attached after `from_dict()`.
        """
        _dict = super().to_dict()
        if self.cache_config is not None:
            from dataclasses import asdict

            _dict["cache_config"] = asdict(self.cache_config)
        if self._tools:
            serializable_tools = []
            for tool in self._tools:
                try:
                    # Check each tool with the same settings Experiment.to_file() uses.
                    # The utf-8 encode rejects unpaired surrogates. allow_nan=False
                    # rejects NaN and Infinity, which are not allowed in valid JSON.
                    # UnicodeEncodeError is a subclass of ValueError.
                    json.dumps(tool, ensure_ascii=False, allow_nan=False).encode("utf-8")
                except (TypeError, ValueError):
                    tool_name = getattr(tool, "tool_name", None) or getattr(tool, "__name__", None)
                    if not isinstance(tool_name, str):
                        # The tool may be a plain string or a dict with no name attribute.
                        # A short ascii() preview identifies it better than a type name.
                        tool_name = ascii(tool)
                        if len(tool_name) > 80:
                            tool_name = tool_name[:77] + "..."
                    logger.warning(
                        "tool_name=<%s> | skipping tool that cannot be written as valid utf-8 JSON, "
                        "re-attach it via the `tools` attribute after loading",
                        tool_name,
                    )
                else:
                    serializable_tools.append(tool)
            if serializable_tools:
                _dict["tools"] = serializable_tools
        return _dict

    def _build_prompt(self, evaluation_case: EvaluationData[InputT, OutputT]) -> str | list:
        """Build the evaluation prompt for a test case.

        Override in subclasses to customize prompt construction (e.g., for multimodal inputs).

        Args:
            evaluation_case: The test case with all of the necessary context to be evaluated.

        Returns:
            Either a text prompt string or a list of content blocks.
        """
        return compose_test_prompt(
            evaluation_case=evaluation_case,
            rubric=self.rubric,
            include_inputs=self.include_inputs,
            uses_environment_state=self.uses_environment_state,
        )

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """
        Evaluate the performance of the task on the given test cases.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Returns:
            The results of the evaluation as EvaluationOutput.
        """
        model = self._resolve_model()
        evaluator_agent = Agent(
            model=model, tools=self.tools, system_prompt=self.system_prompt, callback_handler=None
        )
        evaluation_prompt = self._build_prompt(evaluation_case)
        result = evaluator_agent(evaluation_prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """
        Evaluate the performance of the task on the given test cases asynchronously.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Returns:
            The results of the evaluation as EvaluationOutput.
        """
        model = self._resolve_model()
        evaluator_agent = Agent(
            model=model, tools=self.tools, system_prompt=self.system_prompt, callback_handler=None
        )
        evaluation_prompt = self._build_prompt(evaluation_case)
        result = await evaluator_agent.invoke_async(evaluation_prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]
