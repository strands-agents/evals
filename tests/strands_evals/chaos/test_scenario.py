"""Unit tests for ChaosScenario."""

from strands_evals.chaos import ChaosScenario
from strands_evals.chaos.effects import CorruptValues, ToolCallFailure, TruncateFields


class TestChaosScenario:
    """Tests for the ChaosScenario data model."""

    def test_baseline_scenario_has_no_effects(self):
        scenario = ChaosScenario(name="baseline")
        assert scenario.effects == {}

    def test_scenario_with_multiple_tools(self):
        scenario = ChaosScenario(
            name="compound_failure",
            effects={
                "search_tool": [ToolCallFailure(error_type="timeout")],
                "db_tool": [CorruptValues(corrupt_ratio=0.8)],
            },
        )
        assert len(scenario.effects) == 2
        assert isinstance(scenario.effects["search_tool"][0], ToolCallFailure)
        assert isinstance(scenario.effects["db_tool"][0], CorruptValues)

    def test_scenario_with_multiple_effects_per_tool(self):
        scenario = ChaosScenario(
            name="multi_effect",
            effects={
                "tool_a": [
                    TruncateFields(max_length=5),
                    CorruptValues(corrupt_ratio=0.3),
                ],
            },
        )
        assert len(scenario.effects["tool_a"]) == 2

    def test_repr_shows_effects(self):
        scenario = ChaosScenario(
            name="test",
            effects={"tool": [ToolCallFailure()]},
        )
        repr_str = repr(scenario)
        assert "test" in repr_str
        assert "ToolCallFailure" in repr_str
