"""Unit tests for ChaosCase."""

from strands_evals import Case
from strands_evals.chaos import ChaosCase
from strands_evals.chaos.effects import CorruptValues, ToolCallFailure, TruncateFields


class TestChaosCase:
    """Tests for the ChaosCase data model."""

    def test_baseline_case_has_no_effects(self):
        case = ChaosCase(name="baseline", input="hello")
        assert case.effects == {}

    def test_case_with_effects(self):
        case = ChaosCase(
            name="search_timeout",
            input="hello",
            effects={"search_tool": [ToolCallFailure(error_type="timeout")]},
        )
        assert len(case.effects) == 1
        assert isinstance(case.effects["search_tool"][0], ToolCallFailure)

    def test_case_with_multiple_tools(self):
        case = ChaosCase(
            name="compound",
            input="hello",
            effects={
                "search_tool": [ToolCallFailure(error_type="timeout")],
                "db_tool": [CorruptValues(corrupt_ratio=0.8)],
            },
        )
        assert len(case.effects) == 2

    def test_case_with_multiple_effects_per_tool(self):
        case = ChaosCase(
            name="multi_effect",
            input="hello",
            effects={
                "tool_a": [
                    TruncateFields(max_length=5),
                    CorruptValues(corrupt_ratio=0.3),
                ],
            },
        )
        assert len(case.effects["tool_a"]) == 2

    def test_inherits_case_fields(self):
        case = ChaosCase(
            name="with_expected",
            input="hello",
            expected_output="world",
            expected_trajectory=["tool_a"],
            metadata={"key": "value"},
            effects={"tool_a": [ToolCallFailure()]},
        )
        assert case.input == "hello"
        assert case.expected_output == "world"
        assert case.expected_trajectory == ["tool_a"]
        assert case.metadata == {"key": "value"}

    def test_repr_shows_effects(self):
        case = ChaosCase(
            name="test",
            input="hello",
            effects={"tool": [ToolCallFailure()]},
        )
        repr_str = repr(case)
        assert "test" in repr_str
        assert "ToolCallFailure" in repr_str


class TestChaosCaseExpand:
    """Tests for the ChaosCase.expand() class method."""

    def test_expand_with_baseline(self):
        cases = [
            Case(name="case_a", input="hello"),
            Case(name="case_b", input="world"),
        ]
        effect_maps = {
            "search_timeout": {"search_tool": [ToolCallFailure(error_type="timeout")]},
            "db_corrupt": {"db_tool": [CorruptValues(corrupt_ratio=0.8)]},
        }
        result = ChaosCase.expand(cases, effect_maps, include_no_effect_baseline=True)
        # 2 cases × (2 effect maps + 1 baseline) = 6
        assert len(result) == 6

    def test_expand_without_baseline(self):
        cases = [
            Case(name="case_a", input="hello"),
            Case(name="case_b", input="world"),
        ]
        effect_maps = {
            "search_timeout": {"search_tool": [ToolCallFailure(error_type="timeout")]},
        }
        result = ChaosCase.expand(cases, effect_maps)
        # 2 cases × 1 effect map = 2 (no baseline by default)
        assert len(result) == 2

    def test_expand_baseline_names(self):
        cases = [Case(name="case_a", input="hello")]
        effect_maps = {"timeout": {"tool": [ToolCallFailure()]}}
        result = ChaosCase.expand(cases, effect_maps, include_no_effect_baseline=True)
        names = [c.name for c in result]
        assert "case_a|baseline" in names

    def test_expand_uses_dict_keys_as_names(self):
        cases = [Case(name="case_a", input="hello")]
        effect_maps = {"search_timeout": {"search_tool": [ToolCallFailure(error_type="timeout")]}}
        result = ChaosCase.expand(cases, effect_maps)
        assert result[0].name == "case_a|search_timeout"

    def test_expand_compound_effect_name(self):
        cases = [Case(name="case_a", input="hello")]
        effect_maps = {
            "multi_failure": {
                "search_tool": [ToolCallFailure(error_type="timeout")],
                "db_tool": [CorruptValues()],
            }
        }
        result = ChaosCase.expand(cases, effect_maps)
        assert result[0].name == "case_a|multi_failure"

    def test_expand_unique_session_ids(self):
        cases = [Case(name="case_a", input="hello"), Case(name="case_b", input="world")]
        effect_maps = {"timeout": {"tool": [ToolCallFailure()]}}
        result = ChaosCase.expand(cases, effect_maps)
        session_ids = [c.session_id for c in result]
        assert len(session_ids) == len(set(session_ids))

    def test_expand_preserves_case_fields(self):
        cases = [
            Case(
                name="case_a",
                input="hello",
                expected_output="world",
                expected_trajectory=["tool_a"],
                metadata={"key": "value"},
            )
        ]
        effect_maps = {"timeout": {"tool": [ToolCallFailure()]}}
        result = ChaosCase.expand(cases, effect_maps)
        expanded = result[0]
        assert expanded.input == "hello"
        assert expanded.expected_output == "world"
        assert expanded.expected_trajectory == ["tool_a"]
        assert expanded.metadata == {"key": "value"}

    def test_expand_baseline_has_empty_effects(self):
        cases = [Case(name="case_a", input="hello")]
        effect_maps = {"timeout": {"tool": [ToolCallFailure()]}}
        result = ChaosCase.expand(cases, effect_maps, include_no_effect_baseline=True)
        baseline = [c for c in result if "baseline" in c.name][0]
        assert baseline.effects == {}

    def test_expand_empty_effect_maps_with_baseline(self):
        cases = [Case(name="case_a", input="hello")]
        result = ChaosCase.expand(cases, {}, include_no_effect_baseline=True)
        # Only baseline
        assert len(result) == 1
        assert "baseline" in result[0].name

    def test_expand_empty_effect_maps_without_baseline(self):
        cases = [Case(name="case_a", input="hello")]
        result = ChaosCase.expand(cases, {})
        # No baseline by default, no effect maps → empty
        assert len(result) == 0

    def test_expand_case_without_name(self):
        cases = [Case(input="hello")]
        effect_maps = {"timeout": {"tool": [ToolCallFailure()]}}
        result = ChaosCase.expand(cases, effect_maps, include_no_effect_baseline=True)
        names = [c.name for c in result]
        assert "baseline" in names
        assert "timeout" in names
