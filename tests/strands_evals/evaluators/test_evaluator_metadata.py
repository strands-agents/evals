"""Tests for evaluator metadata types and the metadata() method on evaluators."""

import pytest

from strands_evals.evaluators import (
    Contains,
    CorrectnessEvaluator,
    Equals,
    Evaluator,
    FaithfulnessEvaluator,
    GoalSuccessRateEvaluator,
    HarmfulnessEvaluator,
    StartsWith,
    StateEquals,
    ToolCalled,
    ToolParameterAccuracyEvaluator,
    ToolSelectionAccuracyEvaluator,
)
from strands_evals.types import EvaluatorMetadata, MethodInfo, validate_metadata
from strands_evals.types.evaluator_metadata import REQUIRED_METADATA_KEYS, VALID_METHOD_CATEGORIES, VALID_TIERS


class TestEvaluatorMetadataTypes:
    """Tests for the metadata type definitions."""

    def test_method_info_can_be_constructed(self):
        """MethodInfo accepts category and summary keys."""
        info: MethodInfo = {
            "category": "deterministic_string",
            "summary": "A simple substring check.",
        }
        assert info["category"] == "deterministic_string"
        assert info["summary"] == "A simple substring check."

    def test_evaluator_metadata_all_fields(self):
        """EvaluatorMetadata accepts all defined fields."""
        meta: EvaluatorMetadata = {
            "checks": "Whether output contains keyword",
            "method": {
                "category": "deterministic_string",
                "summary": "Substring search.",
            },
            "threshold": "substring present",
            "tier": "quality",
            "description": "A longer explanation.",
        }
        assert meta["checks"] == "Whether output contains keyword"
        assert meta["tier"] == "quality"
        assert meta["description"] == "A longer explanation."

    def test_evaluator_metadata_required_keys_only(self):
        """EvaluatorMetadata works with only required keys (total=False)."""
        meta: EvaluatorMetadata = {
            "checks": "Something",
            "method": {"category": "custom", "summary": "Custom method."},
            "threshold": "passes",
        }
        assert "tier" not in meta
        assert "description" not in meta

    def test_valid_method_categories_match_literal(self):
        """VALID_METHOD_CATEGORIES contains all expected values."""
        expected = {
            "llm_judge_output",
            "llm_judge_trajectory",
            "deterministic_string",
            "deterministic_extraction",
            "threshold_comparison",
            "composite",
            "custom",
        }
        assert VALID_METHOD_CATEGORIES == expected

    def test_valid_tiers_match_literal(self):
        """VALID_TIERS contains all expected values."""
        expected = {"guardrail", "quality", "diagnostic"}
        assert VALID_TIERS == expected

    def test_required_metadata_keys(self):
        """REQUIRED_METADATA_KEYS lists the mandatory fields."""
        assert REQUIRED_METADATA_KEYS == {"checks", "method", "threshold"}


class TestValidateMetadata:
    """Tests for the validate_metadata function."""

    def _valid_metadata(self) -> EvaluatorMetadata:
        return {
            "checks": "Whether output is correct",
            "method": {
                "category": "llm_judge_output",
                "summary": "An LLM judge evaluates correctness.",
            },
            "threshold": "score >= 0.5",
            "tier": "quality",
        }

    def test_valid_metadata_passes(self):
        """Valid metadata does not raise."""
        validate_metadata(self._valid_metadata(), "TestEvaluator")

    def test_none_metadata_passes(self):
        """None metadata does not raise (evaluator has not declared metadata)."""
        validate_metadata(None, "TestEvaluator")

    def test_valid_metadata_without_optional_fields(self):
        """Metadata without optional fields (tier, description) is valid."""
        meta: EvaluatorMetadata = {
            "checks": "Check something",
            "method": {"category": "custom", "summary": "Custom check."},
            "threshold": "always passes",
        }
        validate_metadata(meta, "TestEvaluator")

    def test_missing_checks_raises(self):
        """Missing 'checks' key raises ValueError."""
        meta = self._valid_metadata()
        del meta["checks"]  # type: ignore[misc]
        with pytest.raises(ValueError, match="missing required keys.*checks"):
            validate_metadata(meta, "MyEvaluator")

    def test_missing_method_raises(self):
        """Missing 'method' key raises ValueError."""
        meta = self._valid_metadata()
        del meta["method"]  # type: ignore[misc]
        with pytest.raises(ValueError, match="missing required keys.*method"):
            validate_metadata(meta, "MyEvaluator")

    def test_missing_threshold_raises(self):
        """Missing 'threshold' key raises ValueError."""
        meta = self._valid_metadata()
        del meta["threshold"]  # type: ignore[misc]
        with pytest.raises(ValueError, match="missing required keys.*threshold"):
            validate_metadata(meta, "MyEvaluator")

    def test_empty_checks_raises(self):
        """Empty string for 'checks' raises ValueError."""
        meta = self._valid_metadata()
        meta["checks"] = "   "
        with pytest.raises(ValueError, match="'checks' must be a non-empty string"):
            validate_metadata(meta, "MyEvaluator")

    def test_invalid_method_type_raises(self):
        """Non-dict 'method' raises ValueError."""
        meta = self._valid_metadata()
        meta["method"] = "not a dict"  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="'method' must be a MethodInfo dict"):
            validate_metadata(meta, "MyEvaluator")

    def test_method_missing_category_raises(self):
        """Method dict missing 'category' raises ValueError."""
        meta = self._valid_metadata()
        meta["method"] = {"summary": "Some summary."}  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="'method' is missing required key: 'category'"):
            validate_metadata(meta, "MyEvaluator")

    def test_method_invalid_category_raises(self):
        """Invalid method category raises ValueError."""
        meta = self._valid_metadata()
        meta["method"] = {"category": "invalid_category", "summary": "Some summary."}  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="method.category 'invalid_category' is not valid"):
            validate_metadata(meta, "MyEvaluator")

    def test_method_missing_summary_raises(self):
        """Method dict missing 'summary' raises ValueError."""
        meta = self._valid_metadata()
        meta["method"] = {"category": "custom"}  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="'method' is missing required key: 'summary'"):
            validate_metadata(meta, "MyEvaluator")

    def test_method_empty_summary_raises(self):
        """Empty method summary raises ValueError."""
        meta = self._valid_metadata()
        meta["method"] = {"category": "custom", "summary": "  "}
        with pytest.raises(ValueError, match="method.summary must be a non-empty string"):
            validate_metadata(meta, "MyEvaluator")

    def test_empty_threshold_raises(self):
        """Empty threshold raises ValueError."""
        meta = self._valid_metadata()
        meta["threshold"] = ""
        with pytest.raises(ValueError, match="'threshold' must be a non-empty string"):
            validate_metadata(meta, "MyEvaluator")

    def test_invalid_tier_raises(self):
        """Invalid tier value raises ValueError."""
        meta = self._valid_metadata()
        meta["tier"] = "critical"  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="tier 'critical' is not valid"):
            validate_metadata(meta, "MyEvaluator")

    def test_all_valid_tiers_pass(self):
        """Each valid tier value passes validation."""
        for tier in VALID_TIERS:
            meta = self._valid_metadata()
            meta["tier"] = tier  # type: ignore[typeddict-item]
            validate_metadata(meta, "TestEvaluator")

    def test_all_valid_method_categories_pass(self):
        """Each valid method category passes validation."""
        for category in VALID_METHOD_CATEGORIES:
            meta = self._valid_metadata()
            meta["method"] = {"category": category, "summary": "Valid."}  # type: ignore[typeddict-item]
            validate_metadata(meta, "TestEvaluator")

    def test_error_message_includes_evaluator_name(self):
        """Error messages include the evaluator name for debugging."""
        meta = self._valid_metadata()
        del meta["checks"]  # type: ignore[misc]
        with pytest.raises(ValueError, match="'SpecificEvaluatorName'"):
            validate_metadata(meta, "SpecificEvaluatorName")


class TestBaseEvaluatorMetadata:
    """Tests for the metadata() method on the base Evaluator class."""

    def test_base_evaluator_returns_none(self):
        """Base Evaluator.metadata() returns None by default."""
        evaluator = Evaluator()
        assert evaluator.metadata() is None

    def test_subclass_without_override_returns_none(self):
        """Subclass that does not override metadata() returns None."""

        class MyEvaluator(Evaluator[str, str]):
            def evaluate(self, evaluation_case):
                return []

        evaluator = MyEvaluator()
        assert evaluator.metadata() is None

    def test_subclass_can_override_metadata(self):
        """Subclass can override metadata() to return valid metadata."""

        class MyEvaluator(Evaluator[str, str]):
            def evaluate(self, evaluation_case):
                return []

            def metadata(self) -> EvaluatorMetadata:
                return {
                    "checks": "Something custom",
                    "method": {"category": "custom", "summary": "Custom check."},
                    "threshold": "passes",
                    "tier": "diagnostic",
                }

        evaluator = MyEvaluator()
        meta = evaluator.metadata()
        assert meta is not None
        assert meta["checks"] == "Something custom"
        assert meta["tier"] == "diagnostic"


class TestDeterministicEvaluatorMetadata:
    """Tests for metadata() on deterministic evaluators."""

    def test_contains_metadata(self):
        """Contains returns valid metadata with case sensitivity info."""
        evaluator = Contains(value="hello")
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether actual_output contains a required substring",
            "method": {
                "category": "deterministic_string",
                "summary": "Case-sensitive substring search on actual_output.",
            },
            "threshold": "substring present",
            "tier": "quality",
        }
        validate_metadata(meta, evaluator.get_name())

    def test_contains_case_insensitive_metadata(self):
        """Contains with case_sensitive=False reports case-insensitive in summary."""
        evaluator = Contains(value="hello", case_sensitive=False)
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether actual_output contains a required substring",
            "method": {
                "category": "deterministic_string",
                "summary": "Case-insensitive substring search on actual_output.",
            },
            "threshold": "substring present",
            "tier": "quality",
        }

    def test_equals_metadata(self):
        """Equals returns valid metadata."""
        evaluator = Equals(value="expected")
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether actual_output exactly equals an expected value",
            "method": {
                "category": "deterministic_string",
                "summary": "Exact equality comparison between actual_output and expected value.",
            },
            "threshold": "exact match",
            "tier": "quality",
        }
        validate_metadata(meta, evaluator.get_name())

    def test_starts_with_metadata(self):
        """StartsWith returns valid metadata with case sensitivity info."""
        evaluator = StartsWith(value="prefix")
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether actual_output starts with a required prefix",
            "method": {
                "category": "deterministic_string",
                "summary": "Case-sensitive prefix check on actual_output.",
            },
            "threshold": "prefix present",
            "tier": "quality",
        }
        validate_metadata(meta, evaluator.get_name())

    def test_starts_with_case_insensitive_metadata(self):
        """StartsWith with case_sensitive=False reports correctly."""
        evaluator = StartsWith(value="prefix", case_sensitive=False)
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether actual_output starts with a required prefix",
            "method": {
                "category": "deterministic_string",
                "summary": "Case-insensitive prefix check on actual_output.",
            },
            "threshold": "prefix present",
            "tier": "quality",
        }

    def test_tool_called_metadata(self):
        """ToolCalled returns valid metadata including the tool name."""
        evaluator = ToolCalled(tool_name="search_web")
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether the tool 'search_web' was called during execution",
            "method": {
                "category": "deterministic_extraction",
                "summary": "Searches the trajectory for a tool execution span matching the target tool name.",
            },
            "threshold": "tool called at least once",
            "tier": "quality",
        }
        validate_metadata(meta, evaluator.get_name())

    def test_state_equals_metadata(self):
        """StateEquals returns valid metadata including the state name."""
        evaluator = StateEquals(name="cart")
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether environment state 'cart' matches the expected value",
            "method": {
                "category": "deterministic_extraction",
                "summary": "Exact equality comparison of a named environment state against an expected value.",
            },
            "threshold": "exact match",
            "tier": "quality",
        }
        validate_metadata(meta, evaluator.get_name())


class TestLLMEvaluatorMetadata:
    """Tests for metadata() on LLM-judge evaluators."""

    def test_faithfulness_metadata(self):
        """FaithfulnessEvaluator returns valid guardrail metadata."""
        evaluator = FaithfulnessEvaluator()
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether the agent's response is grounded in the conversation history",
            "method": {
                "category": "llm_judge_output",
                "summary": (
                    "An LLM judge compares the agent's last response against prior tool outputs"
                    " and conversation for unsupported claims."
                ),
            },
            "threshold": "score >= 0.50",
            "tier": "guardrail",
        }
        validate_metadata(meta, evaluator.get_name())

    def test_harmfulness_metadata(self):
        """HarmfulnessEvaluator returns valid guardrail metadata."""
        evaluator = HarmfulnessEvaluator()
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether the agent's response contains harmful content",
            "method": {
                "category": "llm_judge_output",
                "summary": (
                    "An LLM judge examines the agent's response for harmful, dangerous, or inappropriate content."
                ),
            },
            "threshold": "score == 1.0 (not harmful)",
            "tier": "guardrail",
        }
        validate_metadata(meta, evaluator.get_name())

    def test_correctness_metadata(self):
        """CorrectnessEvaluator returns valid quality metadata."""
        evaluator = CorrectnessEvaluator()
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether the agent's response is factually correct",
            "method": {
                "category": "llm_judge_output",
                "summary": (
                    "An LLM judge evaluates correctness of the response using either"
                    " a 3-level rubric or a binary reference comparison."
                ),
            },
            "threshold": "score >= 1.0 (basic) or CORRECT verdict (reference)",
            "tier": "quality",
        }
        validate_metadata(meta, evaluator.get_name())

    def test_tool_selection_accuracy_metadata(self):
        """ToolSelectionAccuracyEvaluator returns valid metadata."""
        evaluator = ToolSelectionAccuracyEvaluator()
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether each tool call is justified given the conversation context",
            "method": {
                "category": "llm_judge_trajectory",
                "summary": (
                    "An LLM judge evaluates each tool call in the trajectory to determine if it was"
                    " appropriate given the available tools and conversation context."
                ),
            },
            "threshold": "all tool calls scored Yes",
            "tier": "quality",
        }
        validate_metadata(meta, evaluator.get_name())

    def test_tool_parameter_accuracy_metadata(self):
        """ToolParameterAccuracyEvaluator returns valid metadata."""
        evaluator = ToolParameterAccuracyEvaluator()
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether tool call parameters faithfully use information from the conversation context",
            "method": {
                "category": "llm_judge_trajectory",
                "summary": (
                    "An LLM judge evaluates each tool call's parameters to verify they accurately"
                    " reflect information from the preceding conversation and tool results."
                ),
            },
            "threshold": "all tool calls scored Yes",
            "tier": "quality",
        }
        validate_metadata(meta, evaluator.get_name())

    def test_goal_success_rate_metadata(self):
        """GoalSuccessRateEvaluator returns valid metadata."""
        evaluator = GoalSuccessRateEvaluator()
        meta = evaluator.metadata()
        assert meta == {
            "checks": "Whether all user goals were successfully achieved in the conversation",
            "method": {
                "category": "llm_judge_trajectory",
                "summary": (
                    "An LLM judge analyzes the full session to determine if user goals were met,"
                    " either by inferring goals or checking explicit assertions."
                ),
            },
            "threshold": "score == 1.0 (Yes or SUCCESS)",
            "tier": "quality",
        }
        validate_metadata(meta, evaluator.get_name())


class TestMetadataInstanceDependence:
    """Tests that metadata can depend on instance state."""

    def test_contains_metadata_reflects_case_sensitivity(self):
        """Two Contains instances produce different metadata based on case_sensitive."""
        sensitive = Contains(value="hello", case_sensitive=True)
        insensitive = Contains(value="hello", case_sensitive=False)

        meta_s = sensitive.metadata()
        meta_i = insensitive.metadata()

        assert meta_s is not None
        assert meta_i is not None
        assert meta_s["method"]["summary"] != meta_i["method"]["summary"]

    def test_tool_called_metadata_reflects_tool_name(self):
        """Two ToolCalled instances produce different metadata based on tool_name."""
        eval_a = ToolCalled(tool_name="tool_a")
        eval_b = ToolCalled(tool_name="tool_b")

        meta_a = eval_a.metadata()
        meta_b = eval_b.metadata()

        assert meta_a is not None
        assert meta_b is not None
        assert "tool_a" in meta_a["checks"]
        assert "tool_b" in meta_b["checks"]

    def test_state_equals_metadata_reflects_state_name(self):
        """Two StateEquals instances produce different metadata based on name."""
        eval_cart = StateEquals(name="cart")
        eval_balance = StateEquals(name="balance")

        meta_cart = eval_cart.metadata()
        meta_balance = eval_balance.metadata()

        assert meta_cart is not None
        assert meta_balance is not None
        assert "cart" in meta_cart["checks"]
        assert "balance" in meta_balance["checks"]


class TestMetadataImportAccessibility:
    """Tests that metadata types are importable from expected locations."""

    def test_import_from_types(self):
        """Types are importable from strands_evals.types."""
        import strands_evals.types as types_mod

        assert hasattr(types_mod, "EvaluatorMetadata")
        assert hasattr(types_mod, "MethodInfo")
        assert hasattr(types_mod, "MethodCategory")
        assert hasattr(types_mod, "Tier")
        assert hasattr(types_mod, "validate_metadata")

    def test_import_from_types_evaluator_metadata_module(self):
        """Types are importable from the evaluator_metadata module directly."""
        import strands_evals.types.evaluator_metadata as meta_mod

        assert hasattr(meta_mod, "EvaluatorMetadata")
        assert hasattr(meta_mod, "MethodInfo")
        assert hasattr(meta_mod, "MethodCategory")
        assert hasattr(meta_mod, "Tier")
        assert hasattr(meta_mod, "validate_metadata")
        assert hasattr(meta_mod, "REQUIRED_METADATA_KEYS")
        assert hasattr(meta_mod, "VALID_METHOD_CATEGORIES")
        assert hasattr(meta_mod, "VALID_TIERS")
