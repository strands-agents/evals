import re
from typing import ClassVar

from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..evaluator import Evaluator


class InclusiveLanguage(Evaluator[InputT, OutputT]):
    """Scans actual_output for non-inclusive terms against a configurable term list.

    Performs a case-insensitive word-boundary regex scan of actual_output against a
    mapping of banned terms to suggested replacements. Returns a passing score when
    no banned terms are found, and a failing score with details when any are detected.
    """

    DEFAULT_TERMS: ClassVar[dict[str, str]] = {
        "blacklist": "denylist",
        "whitelist": "allowlist",
        "master": "primary",
        "slave": "replica",
        "blackday": "blocked day",
        "whiteday": "clear day",
    }

    def __init__(self, terms: dict[str, str] | None = None, name: str | None = None):
        """Initialize the inclusive language evaluator.

        Args:
            terms: Mapping of banned terms to suggested replacements.
                If None, uses the built-in DEFAULT_TERMS.
            name: Optional instance name for identification in reports.
        """
        super().__init__(name=name)
        self.terms = terms if terms is not None else self.DEFAULT_TERMS

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Evaluate actual_output for non-inclusive terminology.

        Args:
            evaluation_case: The evaluation data containing the output to scan.

        Returns:
            A list with a single EvaluationOutput. Score is 1.0 (pass) if no
            banned terms are found, 0.0 (fail) otherwise.
        """
        text = str(evaluation_case.actual_output).lower()
        found: list[tuple[str, str]] = []
        for term, suggestion in self.terms.items():
            if re.search(rf"\b{re.escape(term)}\b", text):
                found.append((term, suggestion))

        if not found:
            return [EvaluationOutput(score=1.0, test_pass=True, reason="no non-inclusive terms found")]

        details = ", ".join(f"'{t}' -> '{s}'" for t, s in found)
        return [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason=f"found {len(found)} non-inclusive term(s): {details}",
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Async version of evaluate. Delegates to the synchronous implementation."""
        return self.evaluate(evaluation_case)
