"""Unit tests for model output chaos via ChaosPlugin two-hook architecture.

Tests cover:
- Effects constructed via keyed dict {"model_effects": {"*": [...]}}
- EmptyResponse as pre-hook: model not called, turn is single space
- FullRefusal as pre-hook: model not called, turn is refusal text
- MalformedJson on structured-output toolUse: toolUse input corrupted
- Post effects (Confabulation, MalformedJson-on-text, SuccessFraming) work
- Mixed pre+post still produces one turn (pre wins)
- MalformedJson DOES reach/corrupt structured-output toolUse
- Effect family validation: wrong-category effects rejected
- Wildcard rejection: non-'*' model_effects keys rejected
- Ordinary dynamic tool not corrupted: isinstance-based detection
"""

import copy
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError as PydanticValidationError
from strands.hooks import BeforeModelCallEvent
from strands.tools.structured_output.structured_output_tool import StructuredOutputTool

from strands_evals.chaos._context import _current_chaos_case
from strands_evals.chaos.case import ChaosCase
from strands_evals.chaos.effects import (
    Confabulation,
    EmptyResponse,
    FullRefusal,
    MalformedJson,
    SuccessFraming,
    Timeout,
)
from strands_evals.chaos.plugin import ChaosPlugin


def _make_event(message: dict, dynamic_tools: dict | None = None) -> MagicMock:
    """Create a mock MessageAddedEvent with the given message.

    Args:
        message: The message dict.
        dynamic_tools: Optional dict of dynamic tool names -> tools (structured-output tools).
            If None, defaults to empty dict (no structured-output tools registered).
    """
    event = MagicMock()
    event.message = message
    event.agent.tool_registry.dynamic_tools = dynamic_tools or {}
    return event


def _final_assistant_message(text: str = "The answer is 42.") -> dict:
    """An end_turn assistant message with text content only (no toolUse)."""
    return {
        "role": "assistant",
        "content": [{"text": text}],
    }


def _tooluse_assistant_message() -> dict:
    """A tool_use assistant message containing a toolUse block."""
    return {
        "role": "assistant",
        "content": [
            {"text": "Let me search for that."},
            {"toolUse": {"toolUseId": "tu_1", "name": "search", "input": {"query": "test"}}},
        ],
    }


def _user_message() -> dict:
    """A user message."""
    return {
        "role": "user",
        "content": [{"text": "Hello, what is 2+2?"}],
    }


def _tool_result_message() -> dict:
    """A tool result message."""
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "tu_1", "status": "success", "content": [{"text": "4"}]}}],
    }


def _set_chaos_case(model_effects):
    """Helper to set the _current_chaos_case ContextVar with given model_effects.

    Uses keyed dict form: effects={"model_effects": {"*": model_effects}}
    """
    case = ChaosCase(
        name="test_case",
        input="test input",
        effects={"model_effects": {"*": model_effects}},
    )
    _current_chaos_case.set(case)
    return case


class TestKeyedDictConstruction:
    """Effects are constructed via keyed dict form."""

    def test_keyed_dict_form_is_valid(self):
        """ChaosCase accepts effects={"model_effects": {"*": [...]}}."""
        case = ChaosCase(
            name="keyed",
            input="test",
            effects={"model_effects": {"*": [MalformedJson()]}},
        )
        assert case.model_effects == [MalformedJson()]

    def test_wildcard_resolver(self):
        """model_effects property resolves '*' wildcard to flat list."""
        case = ChaosCase(
            name="wildcard",
            input="test",
            effects={"model_effects": {"*": [FullRefusal(), MalformedJson()]}},
        )
        assert len(case.model_effects) == 2
        assert isinstance(case.model_effects[0], FullRefusal)
        assert isinstance(case.model_effects[1], MalformedJson)

    def test_empty_effects_baseline(self):
        """Empty effects dict produces no model_effects."""
        case = ChaosCase(name="baseline", input="test", effects={})
        assert case.model_effects == []


class TestEmptyResponsePreHook:
    """EmptyResponse is a pre-hook effect — cancels model call with single space."""

    def test_empty_response_cancels_with_single_space(self):
        """before_model_invocation sets event.cancel to ' ' (single space)."""
        _set_chaos_case([EmptyResponse()])
        plugin = ChaosPlugin()
        event = BeforeModelCallEvent(agent=MagicMock())

        plugin.before_model_invocation(event)

        assert event.cancel == " "

    def test_empty_response_model_not_called(self):
        """When EmptyResponse fires as pre-hook, post-hook does not apply effects."""
        _set_chaos_case([EmptyResponse()])
        plugin = ChaosPlugin()

        # Pre-hook fires
        pre_event = BeforeModelCallEvent(agent=MagicMock())
        plugin.before_model_invocation(pre_event)
        assert pre_event.cancel == " "

        # SDK builds cancel message, MessageAddedEvent fires
        cancel_message = {"role": "assistant", "content": [{"text": " "}]}
        post_event = _make_event(cancel_message)
        plugin.after_model_invocation(post_event)

        # Content should be unchanged (pre effects skip post processing)
        assert cancel_message["content"] == [{"text": " "}]

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestFullRefusalPreHook:
    """FullRefusal is a pre-hook effect — cancels model call with refusal text."""

    def test_full_refusal_cancels_model_call(self):
        """before_model_invocation sets event.cancel to a refusal template."""
        _set_chaos_case([FullRefusal()])
        plugin = ChaosPlugin()
        event = BeforeModelCallEvent(agent=MagicMock())

        plugin.before_model_invocation(event)

        assert event.cancel in FullRefusal._REFUSAL_TEMPLATES

    def test_full_refusal_produces_single_turn(self):
        """FullRefusal cancels model call, SDK builds cancel message, run ends."""
        _set_chaos_case([FullRefusal()])
        plugin = ChaosPlugin()

        # Step 1: before_model_invocation fires
        pre_event = BeforeModelCallEvent(agent=MagicMock())
        plugin.before_model_invocation(pre_event)
        cancel_text = pre_event.cancel
        assert cancel_text in FullRefusal._REFUSAL_TEMPLATES

        # Step 2: SDK builds the cancel message and fires MessageAddedEvent
        cancel_message = {"role": "assistant", "content": [{"text": cancel_text}]}
        post_event = _make_event(cancel_message)
        plugin.after_model_invocation(post_event)

        # Step 3: verify the cancel message is unchanged (not double-corrupted)
        assert cancel_message["content"] == [{"text": cancel_text}]
        assert len(cancel_message["content"]) == 1

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestMalformedJsonStructuredOutput:
    """MalformedJson DOES reach and corrupt structured-output toolUse blocks only."""

    def test_malformed_json_corrupts_structured_output_tooluse(self):
        """MalformedJson corrupts toolUse input when tool is a StructuredOutputTool."""
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        message = {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "so_1", "name": "MyModel", "input": {"field1": "value1"}}},
            ],
        }
        mock_so_tool = MagicMock(spec=StructuredOutputTool)
        event = _make_event(message, dynamic_tools={"MyModel": mock_so_tool})

        plugin.after_model_invocation(event)

        tool_use_block = message["content"][0]["toolUse"]
        corrupted_input = tool_use_block["input"]
        assert isinstance(corrupted_input, str)
        assert not corrupted_input.endswith("}")

    def test_plain_tooluse_not_corrupted_even_with_malformed_json(self):
        """A plain mid-turn toolUse (not in dynamic_tools) is NOT corrupted by MalformedJson."""
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        message = {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "tu_1", "name": "search", "input": {"query": "test"}}},
            ],
        }
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message, dynamic_tools={})

        plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def test_mixed_tooluse_only_structured_output_corrupted(self):
        """In a message with both regular and structured-output toolUse, only SO is corrupted."""
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        message = {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "tu_1", "name": "search", "input": {"query": "test"}}},
                {"toolUse": {"toolUseId": "so_1", "name": "MyModel", "input": {"field1": "value1"}}},
            ],
        }
        mock_so_tool = MagicMock(spec=StructuredOutputTool)
        event = _make_event(message, dynamic_tools={"MyModel": mock_so_tool})

        plugin.after_model_invocation(event)

        # "search" toolUse should be UNCHANGED
        search_block = message["content"][0]["toolUse"]
        assert search_block["input"] == {"query": "test"}
        # "MyModel" toolUse should be CORRUPTED
        so_block = message["content"][1]["toolUse"]
        assert isinstance(so_block["input"], str)
        assert not so_block["input"].endswith("}")

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestPostEffectsOnText:
    """Post effects (Confabulation, MalformedJson-on-text, SuccessFraming) work."""

    def test_confabulation_injects_template(self):
        """Confabulation injects fabricated citations into text content."""
        _set_chaos_case([Confabulation()])
        plugin = ChaosPlugin()
        original_text = "The weather is sunny. It is warm outside. Birds are singing."
        message = _final_assistant_message(original_text)
        event = _make_event(message)

        plugin.after_model_invocation(event)

        result_text = message["content"][0]["text"]
        assert result_text != original_text
        assert "sunny" in result_text or "warm" in result_text

    def test_malformed_json_on_text(self):
        """MalformedJson truncates JSON-like text content."""
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        message = _final_assistant_message('{"key": "value", "nested": {"a": 1}}')
        event = _make_event(message)

        plugin.after_model_invocation(event)

        result_text = message["content"][0]["text"]
        assert result_text != '{"key": "value", "nested": {"a": 1}}'
        assert len(result_text) < len('{"key": "value", "nested": {"a": 1}}')

    def test_success_framing_prepends_prefix(self):
        """SuccessFraming prepends a confident prefix to text content."""
        _set_chaos_case([SuccessFraming()])
        plugin = ChaosPlugin()
        message = _final_assistant_message("Here is the result.")
        event = _make_event(message)

        plugin.after_model_invocation(event)

        result_text = message["content"][0]["text"]
        has_prefix = any(result_text.startswith(p) for p in SuccessFraming._SUCCESS_PREFIXES)
        assert has_prefix
        assert "Here is the result." in result_text

    def test_confabulation_plus_success_framing(self):
        """Confabulation + SuccessFraming compose: citation injected, then prefix prepended."""
        _set_chaos_case([Confabulation(), SuccessFraming()])
        plugin = ChaosPlugin()
        original_text = "The weather is sunny. It is warm outside. Birds are singing."
        message = _final_assistant_message(original_text)
        event = _make_event(message)

        plugin.after_model_invocation(event)

        result_text = message["content"][0]["text"]
        has_prefix = any(result_text.startswith(p) for p in SuccessFraming._SUCCESS_PREFIXES)
        assert has_prefix

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestMixedPrePostCase:
    """Mixed pre+post effects: pre wins, post does NOT double-corrupt."""

    def test_full_refusal_plus_malformed_json(self):
        """FullRefusal (pre) + MalformedJson (post): pre cancels, post skipped."""
        _set_chaos_case([FullRefusal(), MalformedJson()])
        plugin = ChaosPlugin()

        pre_event = BeforeModelCallEvent(agent=MagicMock())
        plugin.before_model_invocation(pre_event)
        cancel_text = pre_event.cancel
        assert cancel_text in FullRefusal._REFUSAL_TEMPLATES

        cancel_message = {"role": "assistant", "content": [{"text": cancel_text}]}
        post_event = _make_event(cancel_message)
        plugin.after_model_invocation(post_event)

        # Post effect (MalformedJson) should NOT have corrupted the content
        assert cancel_message["content"] == [{"text": cancel_text}]
        assert len(cancel_message["content"]) == 1

    def test_empty_response_plus_success_framing(self):
        """EmptyResponse (pre) + SuccessFraming (post): pre cancels, post skipped."""
        _set_chaos_case([EmptyResponse(), SuccessFraming()])
        plugin = ChaosPlugin()

        pre_event = BeforeModelCallEvent(agent=MagicMock())
        plugin.before_model_invocation(pre_event)
        assert pre_event.cancel == " "

        cancel_message = {"role": "assistant", "content": [{"text": " "}]}
        post_event = _make_event(cancel_message)
        plugin.after_model_invocation(post_event)

        # SuccessFraming (post) should NOT have been applied
        assert cancel_message["content"] == [{"text": " "}]

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestMalformedJsonReachesStructuredOutput:
    """MalformedJson reaches structured-output toolUse (relaxed for it)."""

    def test_malformed_json_corrupts_structured_output_tooluse(self):
        """MalformedJson DOES corrupt a structured-output toolUse block."""
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        message = {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "so_1",
                        "name": "MyModel",
                        "input": {"field1": "value1"},
                    }
                },
            ],
        }
        mock_so_tool = MagicMock(spec=StructuredOutputTool)
        event = _make_event(message, dynamic_tools={"MyModel": mock_so_tool})

        plugin.after_model_invocation(event)

        tool_use_block = message["content"][0]["toolUse"]
        assert isinstance(tool_use_block["input"], str)
        assert not tool_use_block["input"].endswith("}")

    def test_other_post_effects_still_skip_tooluse(self):
        """Confabulation on a toolUse message is skipped (only relaxed for MalformedJson)."""
        _set_chaos_case([Confabulation()])
        plugin = ChaosPlugin()
        message = _tooluse_assistant_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestEffectFamilyValidation:
    """Effects placed in the wrong category are rejected structurally by Pydantic."""

    def test_tool_effect_in_model_effects_rejected(self):
        """A ToolEffect under model_effects is rejected by discriminated union."""
        with pytest.raises(PydanticValidationError, match="union_tag_invalid"):
            ChaosCase(
                name="bad",
                input="test",
                effects={"model_effects": {"*": [Timeout()]}},
            )

    def test_model_effect_in_tool_effects_rejected(self):
        """A ModelEffect under tool_effects is rejected by discriminated union."""
        with pytest.raises(PydanticValidationError, match="union_tag_invalid"):
            ChaosCase(
                name="bad",
                input="test",
                effects={"tool_effects": {"search": [FullRefusal()]}},
            )

    def test_model_effect_in_tool_effects_rejected_via_model_validate(self):
        """A ModelEffect under tool_effects is rejected on the model_validate (dict) path."""
        with pytest.raises(PydanticValidationError, match="union_tag_invalid"):
            ChaosCase.model_validate(
                {
                    "name": "bad_tool",
                    "input": "test",
                    "effects": {"tool_effects": {"search": [{"effect_type": "full_refusal"}]}},
                }
            )

    def test_tool_effect_in_model_effects_rejected_via_model_validate(self):
        """A ToolEffect under model_effects is rejected on the model_validate (dict) path."""
        with pytest.raises(PydanticValidationError, match="union_tag_invalid"):
            ChaosCase.model_validate(
                {
                    "name": "bad_model",
                    "input": "test",
                    "effects": {"model_effects": {"*": [{"effect_type": "timeout"}]}},
                }
            )

    def test_named_model_key_rejected(self):
        """A non-'*' key in model_effects is rejected by Literal constraint."""
        with pytest.raises(PydanticValidationError, match="literal_error"):
            ChaosCase(
                name="bad",
                input="test",
                effects={"model_effects": {"claude-sonnet": [MalformedJson()]}},
            )

    def test_bogus_category_rejected(self):
        """An unknown effects category is rejected by extra='forbid'."""
        with pytest.raises(PydanticValidationError, match="extra_forbidden"):
            ChaosCase(
                name="bad",
                input="test",
                effects={"bogus": {"x": []}},
            )


class TestOrdinaryDynamicToolNotCorrupted:
    """An ordinary dynamic tool (not StructuredOutputTool) is NOT corrupted."""

    def test_ordinary_dynamic_tool_unchanged(self):
        """MalformedJson does NOT corrupt a regular dynamic tool's toolUse."""
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        message = {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "dt_1", "name": "my_dynamic_tool", "input": {"key": "val"}}},
            ],
        }
        original_content = copy.deepcopy(message["content"])
        # Register as a plain MagicMock (NOT spec'd to StructuredOutputTool)
        mock_tool = MagicMock()
        event = _make_event(message, dynamic_tools={"my_dynamic_tool": mock_tool})

        plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestGuardRoleFiltering:
    """User and tool result messages are NOT corrupted."""

    def test_user_message_not_corrupted(self):
        _set_chaos_case([Confabulation()])
        plugin = ChaosPlugin()
        message = _user_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def test_tool_result_message_not_corrupted(self):
        _set_chaos_case([Confabulation()])
        plugin = ChaosPlugin()
        message = _tool_result_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestPassthrough:
    """No corruption when no model_effects is set."""

    def test_no_config_passes_through(self):
        _current_chaos_case.set(None)
        plugin = ChaosPlugin()
        message = _final_assistant_message("Hello world")
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def test_empty_effects_passes_through(self):
        """ChaosCase with empty effects dict does not corrupt."""
        case = ChaosCase(name="baseline", input="test", effects={})
        _current_chaos_case.set(case)
        plugin = ChaosPlugin()
        message = _final_assistant_message("Hello world")
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def teardown_method(self):
        _current_chaos_case.set(None)
