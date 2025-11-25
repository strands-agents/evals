"""
Simple example demonstrating multi-metric evaluation.

This toy evaluator checks multiple aspects of a response:
1. Length check (is response long enough?)
2. Keyword check (does it contain expected keywords?)
3. Sentiment check (is it positive?)

Each check produces its own metric, and they're aggregated into a final score.
"""

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput


class MultiAspectEvaluator(Evaluator[str, str]):
    """Evaluates multiple aspects of a text response."""

    def __init__(self, min_length: int = 10, required_keywords: list[str] | None = None):
        super().__init__()
        self.min_length = min_length
        self.required_keywords = required_keywords or []

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        """Returns one EvaluationOutput per aspect checked."""
        actual = evaluation_case.actual_output or ""
        results = []

        # Metric 1: Length check
        length_ok = len(actual) >= self.min_length
        results.append(
            EvaluationOutput(
                score=1.0 if length_ok else 0.0,
                test_pass=length_ok,
                reason=f"Length: {len(actual)} chars ({'✓' if length_ok else '✗'} min {self.min_length})",
            )
        )

        # Metric 2: Keyword check
        keywords_found = [kw for kw in self.required_keywords if kw.lower() in actual.lower()]
        keyword_score = len(keywords_found) / len(self.required_keywords) if self.required_keywords else 1.0
        results.append(
            EvaluationOutput(
                score=keyword_score,
                test_pass=keyword_score >= 0.5,
                reason=f"Keywords: {len(keywords_found)}/{len(self.required_keywords)} found {keywords_found}",
            )
        )

        # Metric 3: Sentiment check (simple heuristic)
        positive_words = ["good", "great", "excellent", "happy", "wonderful"]
        has_positive = any(word in actual.lower() for word in positive_words)
        results.append(
            EvaluationOutput(
                score=1.0 if has_positive else 0.5,
                test_pass=True,  # Always pass, just informational
                reason=f"Sentiment: {'Positive' if has_positive else 'Neutral'}",
            )
        )

        return results

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        # For this simple example, just call sync version
        return self.evaluate(evaluation_case)


# Task function: simple echo with modifications
def simple_task(case: Case) -> str:
    return f"This is a great response to: {case.input}. It provides excellent information!"


if __name__ == "__main__":
    # Create test cases
    test_cases = [
        Case[str, str](
            name="short-response",
            input="Hi",
            expected_output="Short reply",
        ),
        Case[str, str](
            name="long-response",
            input="Tell me about Python",
            expected_output="Python is great",
        ),
    ]

    # Create evaluator that checks: length >= 20, contains ["response", "information"]
    evaluator = MultiAspectEvaluator(min_length=20, required_keywords=["response", "information"])

    # Create dataset and run
    experiment = Experiment[str, str](cases=test_cases, evaluator=evaluator)
    report = experiment.run_evaluations(simple_task)

    # Show results
    print("\n=== Programmatic Access to Detailed Results ===")
    for i, detailed in enumerate(report.detailed_results):
        print(f"\nCase {i}: {report.cases[i]['name']}")
        print(f"  Aggregate Score: {report.scores[i]:.2f}")
        print(f"  Individual Metrics ({len(detailed)}):")
        for j, metric in enumerate(detailed):
            print(f"    {j + 1}. Score={metric.score:.2f}, Pass={metric.test_pass}, Reason={metric.reason}")

    # Interactive display
    print("\n=== Interactive Display ===")
    report.run_display()
