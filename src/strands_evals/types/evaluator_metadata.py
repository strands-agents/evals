"""Types for evaluator metadata declarations.

Evaluator metadata lets evaluators declare what they check and how they work,
enabling downstream systems to aggregate results by tier, render informative
reports, and route models based on evaluation method.
"""

from typing import get_args

from typing_extensions import Literal, TypedDict

MethodCategory = Literal[
    "llm_judge_output",
    "llm_judge_trajectory",
    "deterministic_string",
    "deterministic_extraction",
    "threshold_comparison",
    "composite",
    "custom",
]
"""Category describing the evaluation method.

- llm_judge_output: An LLM judge evaluates the final output.
- llm_judge_trajectory: An LLM judge evaluates the full trajectory.
- deterministic_string: A deterministic string comparison (contains, equals, etc.).
- deterministic_extraction: Deterministic extraction and comparison of structured data.
- threshold_comparison: A numeric threshold comparison.
- composite: Combines multiple evaluation methods.
- custom: A custom evaluation method not covered by other categories.
"""

Tier = Literal["guardrail", "quality", "diagnostic"]
"""Tier describing how this evaluator's result affects the overall verdict.

- guardrail: Any failure overrides the overall verdict to fail. Non-negotiable rules.
- quality: The primary correctness signal. Feeds the headline score and gates pass/fail.
- diagnostic: Surfaced in reports but does not gate pass/fail.
"""

VALID_METHOD_CATEGORIES: set[str] = set(get_args(MethodCategory))

VALID_TIERS: set[str] = set(get_args(Tier))


class MethodInfo(TypedDict):
    """Describes how an evaluator works.

    Attributes:
        category: The evaluation method category.
        summary: One to two sentences explaining how the evaluator works.
    """

    category: MethodCategory
    summary: str


class EvaluatorMetadata(TypedDict, total=False):
    """Metadata that an evaluator declares about itself.

    Required keys: checks, method, threshold.
    Optional keys: tier, description.

    Attributes:
        checks: One sentence describing what is measured.
        method: How the evaluator works.
        threshold: The pass condition (e.g. "score >= 0.50").
        tier: How this result affects the overall verdict. Defaults to "quality".
        description: A longer explanation of the evaluator.
    """

    checks: str
    method: MethodInfo
    threshold: str
    tier: Tier
    description: str


REQUIRED_METADATA_KEYS: set[str] = {"checks", "method", "threshold"}


def validate_metadata(metadata: EvaluatorMetadata | None, evaluator_name: str) -> None:
    """Validate that evaluator metadata has all required keys and valid values.

    Args:
        metadata: The metadata dict returned by an evaluator's metadata() method.
            If None, the evaluator has not declared metadata and validation is skipped.
        evaluator_name: The evaluator name, used in error messages.

    Raises:
        ValueError: If required keys are missing or values are invalid.
    """
    if metadata is None:
        return

    # Check required keys
    missing_keys = REQUIRED_METADATA_KEYS - set(metadata.keys())
    if missing_keys:
        raise ValueError(f"Evaluator '{evaluator_name}' metadata is missing required keys: {sorted(missing_keys)}")

    # Validate checks is a non-empty string
    checks = metadata.get("checks", "")
    if not isinstance(checks, str) or not checks.strip():
        raise ValueError(f"Evaluator '{evaluator_name}' metadata 'checks' must be a non-empty string")

    # Validate method
    method = metadata.get("method")
    if not isinstance(method, dict):
        raise ValueError(f"Evaluator '{evaluator_name}' metadata 'method' must be a MethodInfo dict")

    if "category" not in method:
        raise ValueError(f"Evaluator '{evaluator_name}' metadata 'method' is missing required key: 'category'")

    if method["category"] not in VALID_METHOD_CATEGORIES:
        raise ValueError(
            f"Evaluator '{evaluator_name}' metadata method.category '{method['category']}' "
            f"is not valid. Must be one of: {sorted(VALID_METHOD_CATEGORIES)}"
        )

    if "summary" not in method:
        raise ValueError(f"Evaluator '{evaluator_name}' metadata 'method' is missing required key: 'summary'")

    if not isinstance(method["summary"], str) or not method["summary"].strip():
        raise ValueError(f"Evaluator '{evaluator_name}' metadata method.summary must be a non-empty string")

    # Validate threshold is a non-empty string
    threshold = metadata.get("threshold", "")
    if not isinstance(threshold, str) or not threshold.strip():
        raise ValueError(f"Evaluator '{evaluator_name}' metadata 'threshold' must be a non-empty string")

    # Validate tier if present
    tier = metadata.get("tier")
    if tier is not None and tier not in VALID_TIERS:
        raise ValueError(
            f"Evaluator '{evaluator_name}' metadata tier '{tier}' is not valid. Must be one of: {sorted(VALID_TIERS)}"
        )
