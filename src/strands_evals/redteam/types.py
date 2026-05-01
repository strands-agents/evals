"""Shared types for red team module."""

from enum import Enum

from pydantic import BaseModel, Field


class RiskCategory(str, Enum):
    """Risk categories for agent-level vulnerabilities."""

    PROMPT_INJECTION = "prompt_injection"
    EXCESSIVE_AGENCY = "excessive_agency"
    SENSITIVE_DATA = "sensitive_data"
    DATA_EXFILTRATION = "data_exfiltration"
    SOCIAL_ENGINEERING = "social_engineering"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"


class AttackGoal(BaseModel):
    """Structured attack objective.

    Used by case generators to define what an attack should achieve,
    and by evaluators to judge whether it succeeded.
    """

    risk_category: RiskCategory
    description: str = Field(description="What the attack should achieve")
    target_tools: list[str] | None = Field(
        default=None,
        description="Tools the attack should target. None means any.",
    )
    success_criteria: str = Field(
        description="How to judge if the attack worked",
    )
