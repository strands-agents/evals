"""Tests for AdversarialCaseGenerator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from strands import Agent as RealAgent

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.generators.adversarial import (
    AdversarialCaseGenerator,
    _AttackCase,
    _AttackCases,
)


def _mock_attack_cases(n: int = 2) -> _AttackCases:
    return _AttackCases(
        cases=[
            _AttackCase(
                actor_goal=f"goal-{i}",
                target_context=f"ctx-{i}",
                traits={"role": "tester"},
                opening_message=f"open-{i}",
                success_criteria=f"criteria-{i}",
            )
            for i in range(n)
        ]
    )


def _make_agent_mock():
    return MagicMock(spec=RealAgent)


@pytest.fixture
def patched_agent():
    with (
        patch("strands_evals.experimental.redteam.generators.adversarial.Agent") as mock_agent_cls,
        patch("strands_evals.experimental.redteam.generators.adversarial._extract_tool_info") as mock_extract,
    ):
        agent = MagicMock()
        response = MagicMock()
        response.structured_output = _mock_attack_cases(2)
        agent.invoke_async = AsyncMock(return_value=response)
        mock_agent_cls.return_value = agent
        mock_extract.return_value = {"system_prompt": "x", "tools": []}
        yield mock_agent_cls


def test_unknown_risk_category_raises(patched_agent):
    gen = AdversarialCaseGenerator()
    with pytest.raises(ValueError, match="Unknown risk category"):
        gen.generate_cases(agent=_make_agent_mock(), risk_categories=["nope"])


def test_agent_required():
    gen = AdversarialCaseGenerator()
    with pytest.raises(TypeError):
        gen.generate_cases(risk_categories=["prompt_injection"])  # type: ignore[call-arg]


def test_generate_cases_returns_strategy_agnostic_cases(patched_agent):
    gen = AdversarialCaseGenerator()
    cases = gen.generate_cases(
        agent=_make_agent_mock(),
        risk_categories=["prompt_injection"],
        num_cases=2,
    )
    assert len(cases) == 2
    assert all(isinstance(c, RedTeamCase) for c in cases)
    assert all(c.config.attack_goal.risk_category == "prompt_injection" for c in cases)
    # cases are strategy-agnostic now: no strategy baked into the config or name
    assert not hasattr(cases[0].config, "strategy")
    assert "__" not in cases[0].name
    assert cases[0].name == "prompt_injection_0"
    assert cases[0].input == "open-0"
    assert cases[0].config.attack_goal.actor_goal == "goal-0"
    assert cases[0].config.attack_goal.context == "ctx-0"
    assert cases[0].config.attack_goal.success_criteria == "criteria-0"
    assert cases[0].metadata["success_criteria"] == "criteria-0"


def test_metadata_synced_from_config(patched_agent):
    gen = AdversarialCaseGenerator()
    cases = gen.generate_cases(
        agent=_make_agent_mock(),
        risk_categories=["prompt_injection"],
        num_cases=1,
    )
    case = cases[0]
    assert case.metadata["risk_category"] == "prompt_injection"
    assert case.metadata["actor_goal"] == case.config.attack_goal.actor_goal


def test_generate_cases_with_target_spec(patched_agent):
    """generate_cases() accepts TargetSpec dict without needing an Agent."""
    gen = AdversarialCaseGenerator()
    cases = gen.generate_cases(
        agent={"system_prompt": "x", "tools": []},
        risk_categories=["prompt_injection"],
        num_cases=1,
    )
    assert len(cases) == 1
    assert isinstance(cases[0], RedTeamCase)


def test_generate_risk_categories_optional(patched_agent):
    """generate_cases() infers risk categories when not provided."""
    gen = AdversarialCaseGenerator()
    with patch.object(gen, "_infer_risk_categories", new_callable=AsyncMock) as mock_infer:
        mock_infer.return_value = ["prompt_injection"]
        cases = gen.generate_cases(agent=_make_agent_mock(), num_cases=1)
        mock_infer.assert_called_once()
        assert len(cases) == 1


def test_target_spec_missing_keys_raises():
    gen = AdversarialCaseGenerator()
    with pytest.raises(ValueError, match="missing required keys"):
        gen.generate_cases(
            agent={"system_prompt": "x"},  # missing 'tools'
            risk_categories=["prompt_injection"],
        )


def test_empty_llm_response_raises(patched_agent):
    empty_response = MagicMock()
    empty_response.structured_output = _AttackCases(cases=[])
    patched_agent.return_value.invoke_async = AsyncMock(return_value=empty_response)
    gen = AdversarialCaseGenerator()
    with pytest.raises(RuntimeError, match="produced no cases"):
        gen.generate_cases(
            agent=_make_agent_mock(),
            risk_categories=["prompt_injection"],
            num_cases=1,
        )


async def test_generate_cases_async_returns_cases(patched_agent):
    gen = AdversarialCaseGenerator()
    cases = await gen.generate_cases_async(
        agent={"system_prompt": "x", "tools": []},
        risk_categories=["prompt_injection"],
        num_cases=1,
    )
    assert len(cases) == 1
    assert isinstance(cases[0], RedTeamCase)


# ---------------------------------------------------------------------------
# _extract_tool_info (inlined helper)
# ---------------------------------------------------------------------------


def _make_agent_for_extract(tools_config=None, system_prompt="sys"):
    agent = MagicMock()
    agent.system_prompt = system_prompt
    agent.tool_registry.get_all_tools_config.return_value = tools_config or {}
    return agent


class TestExtractToolInfo:
    def test_returns_system_prompt_and_empty_tools(self):
        from strands_evals.experimental.redteam.generators.adversarial import _extract_tool_info

        info = _extract_tool_info(_make_agent_for_extract())
        assert info["system_prompt"] == "sys"
        assert info["tools"] == []
        assert "0 tools" in info["description"]

    def test_extracts_tool_definitions(self):
        from strands_evals.experimental.redteam.generators.adversarial import _extract_tool_info

        tools_config = {
            "lookup": {
                "name": "lookup",
                "description": "Look up an order",
                "inputSchema": {"json": {"properties": {"id": {"type": "string"}}}},
            }
        }
        info = _extract_tool_info(_make_agent_for_extract(tools_config=tools_config))
        assert info["tools"] == [
            {"name": "lookup", "description": "Look up an order", "parameters": {"id": {"type": "string"}}}
        ]
        assert "lookup" in info["description"]

    def test_handles_missing_input_schema(self):
        from strands_evals.experimental.redteam.generators.adversarial import _extract_tool_info

        info = _extract_tool_info(_make_agent_for_extract(tools_config={"t": {"name": "t"}}))
        assert info["tools"][0]["parameters"] == {}

    def test_none_system_prompt_becomes_empty_string(self):
        from strands_evals.experimental.redteam.generators.adversarial import _extract_tool_info

        info = _extract_tool_info(_make_agent_for_extract(system_prompt=None))
        assert info["system_prompt"] == ""

    def test_swallows_registry_errors(self):
        from strands_evals.experimental.redteam.generators.adversarial import _extract_tool_info

        agent = MagicMock()
        agent.system_prompt = "sys"
        agent.tool_registry.get_all_tools_config.side_effect = AttributeError("boom")
        info = _extract_tool_info(agent)
        assert info["tools"] == []
