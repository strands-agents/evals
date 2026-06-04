"""Adversarial case generator."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TypedDict, cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ..case import RedTeamCase
from ..strategies import (
    BUILTIN_STRATEGIES,
    DEFAULT_STRATEGY,
    AttackStrategy,
    resolve_strategy,
)
from ..types import DEFAULT_SEVERITY, RISK_CATEGORIES, AttackGoal, RedTeamConfig
from .prompt_templates import get_template as _get_prompt_template

logger = logging.getLogger(__name__)


class TargetSpec(TypedDict):
    """Description of a target agent for case generation."""

    system_prompt: str
    tools: list[dict]


class _AttackCase(BaseModel):
    actor_goal: str = Field(description="Specific attack objective for this case")
    target_context: str = Field(description="1-2 sentence target summary relevant to the attack")
    traits: dict = Field(default_factory=dict, description="Attacker persona attributes")
    opening_message: str = Field(description="First message the attacker sends")
    success_criteria: str = Field(description="Concrete observable condition that marks the attack as successful")


class _AttackCases(BaseModel):
    cases: list[_AttackCase]


class _RiskCategorySelection(BaseModel):
    categories: list[str] = Field(description="Selected risk category keys relevant to the target")


def _resolve_strategies(strategies: list[AttackStrategy | str] | None) -> list[AttackStrategy]:
    if not strategies:
        return [BUILTIN_STRATEGIES[DEFAULT_STRATEGY]]
    return [resolve_strategy(s) for s in strategies]


_REQUIRED_TARGET_KEYS = ("system_prompt", "tools")


def _extract_tool_info(agent: Agent) -> dict:
    """Extract tool definitions and system prompt as ``target_info``."""
    tools: list[dict[str, Any]] = []
    try:
        all_configs: Any = agent.tool_registry.get_all_tools_config()
        for tool_config in all_configs.values():
            input_schema = tool_config.get("inputSchema", {}).get("json", {})
            tools.append(
                {
                    "name": tool_config["name"],
                    "description": tool_config.get("description", ""),
                    "parameters": input_schema.get("properties", {}),
                }
            )
    except (AttributeError, KeyError, TypeError) as e:
        logger.warning("error=<%s> | failed to extract tools from agent", e)

    return {
        "system_prompt": agent.system_prompt or "",
        "tools": tools,
        "description": f"Agent with {len(tools)} tools: {[t['name'] for t in tools]}",
    }


def _coerce_target(target: Agent | TargetSpec) -> dict:
    if isinstance(target, dict):
        missing = [k for k in _REQUIRED_TARGET_KEYS if k not in target]
        if missing:
            raise ValueError(f"TargetSpec is missing required keys: {missing}")
        return dict(target)
    return _extract_tool_info(target)


class AdversarialCaseGenerator:
    """Generate red team cases tailored to a target.

    Example:
        ```python
        cases = AdversarialCaseGenerator(model=model).generate_cases(
            target=agent,
            risk_categories=["guideline_bypass", "data_exfiltration"],
            num_cases=3,
        )
        ```
    """

    def __init__(
        self,
        model: Model | str | None = None,
        prompt_version: str = "v0",
    ):
        self.model = model
        self.prompt_version = prompt_version

    def generate_cases(
        self,
        *,
        target: Agent | TargetSpec,
        risk_categories: list[str] | None = None,
        num_cases: int = 5,
        attack_strategies: list[AttackStrategy | str] | None = None,
    ) -> list[RedTeamCase]:
        """Generate cases without building an experiment. Use with TargetSpec or for case reuse."""
        return asyncio.run(
            self.generate_cases_async(
                target=target,
                risk_categories=risk_categories,
                num_cases=num_cases,
                attack_strategies=attack_strategies,
            )
        )

    async def generate_cases_async(
        self,
        *,
        target: Agent | TargetSpec,
        risk_categories: list[str] | None = None,
        num_cases: int = 5,
        attack_strategies: list[AttackStrategy | str] | None = None,
    ) -> list[RedTeamCase]:
        """Async variant of :meth:`generate_cases`."""
        target_info = _coerce_target(target)
        resolved_categories = risk_categories or await self._infer_risk_categories(target_info)
        return await self._generate_cases(
            target_info=target_info,
            risk_categories=resolved_categories,
            num_cases=num_cases,
            attack_strategies=attack_strategies,
        )

    async def _generate_cases(
        self,
        *,
        target_info: dict,
        risk_categories: list[str],
        num_cases: int = 5,
        attack_strategies: list[AttackStrategy | str] | None = None,
    ) -> list[RedTeamCase]:
        for risk_category in risk_categories:
            if risk_category not in RISK_CATEGORIES:
                raise ValueError(
                    f"Unknown risk category: '{risk_category}'. Available categories: {list(RISK_CATEGORIES)}"
                )

        resolved_strategies = _resolve_strategies(attack_strategies)

        cases: list[RedTeamCase] = []
        for risk_category in risk_categories:
            generated = await self._generate_cases_for_category(
                risk_category=risk_category,
                target_info=target_info,
                num_cases=num_cases,
            )

            severity = DEFAULT_SEVERITY.get(risk_category, "medium")
            for i, attack in enumerate(generated):
                for strategy in resolved_strategies:
                    template = strategy.system_prompt_template
                    if template is None:
                        raise NotImplementedError(
                            f"Strategy {type(strategy).__name__!r} does not expose system_prompt_template. "
                            "Only system-prompt-based strategies are currently supported."
                        )
                    config = RedTeamConfig(
                        attack_goal=AttackGoal(
                            risk_category=risk_category,
                            actor_goal=attack.actor_goal,
                            context=attack.target_context,
                            severity=severity,
                            success_criteria=attack.success_criteria,
                        ),
                        traits=attack.traits,
                        system_prompt_template=template,
                        strategy=strategy.name,
                    )
                    cases.append(
                        RedTeamCase(
                            name=f"{risk_category}_{i}__{strategy.name}",
                            input=attack.opening_message,
                            config=config,
                        )
                    )

        return cases

    async def _infer_risk_categories(self, target_info: dict) -> list[str]:
        """Use LLM to infer relevant risk categories from target info."""
        template = _get_prompt_template(self.prompt_version)
        categories_desc = "\n".join(f"- {key}: {desc}" for key, desc in RISK_CATEGORIES.items())
        prompt = template.CATEGORY_INFERENCE_PROMPT.format(
            target_info=json.dumps(target_info, indent=2),
            categories=categories_desc,
        )
        agent = Agent(model=self.model, callback_handler=None)
        response = await agent.invoke_async(prompt, structured_output_model=_RiskCategorySelection)
        result = cast(_RiskCategorySelection, response.structured_output)
        if result is None:
            logger.warning("reason=<no_structured_output> | risk-category inference empty using all")
            return list(RISK_CATEGORIES.keys())
        valid = [c for c in result.categories if c in RISK_CATEGORIES]
        if not valid:
            logger.warning("got=<%s> | no recognized risk categories inferred using all", result.categories)
            return list(RISK_CATEGORIES.keys())
        return valid

    async def _generate_cases_for_category(
        self,
        *,
        risk_category: str,
        target_info: dict,
        num_cases: int,
    ) -> list[_AttackCase]:
        template = _get_prompt_template(self.prompt_version)
        prompt = template.CASE_GENERATION_PROMPT.format(
            target_info=json.dumps(target_info, indent=2),
            risk_category=risk_category,
            risk_description=RISK_CATEGORIES[risk_category],
            num_cases=num_cases,
        )
        agent = Agent(model=self.model, callback_handler=None)
        response = await agent.invoke_async(prompt, structured_output_model=_AttackCases)
        result = cast(_AttackCases, response.structured_output)
        if result is None or not result.cases:
            raise RuntimeError(f"Case generator produced no cases for risk_category={risk_category!r}.")
        return result.cases[:num_cases]
