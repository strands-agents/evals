"""Tests for model output chaos (ChaosPlugin two-hook architecture)."""

import copy
import logging
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError as PydanticValidationError
from strands.hooks import BeforeModelCallEvent

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
from strands_evals.chaos.plugin import _CHAOS_STATE_KEY, ChaosPlugin


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


class TestStructuredOutputFailureInjection:
    """MalformedJson injects one structured-output parse failure per invocation."""

    _EXPECTED_MESSAGE = "Structured output was malformed and could not be parsed. Please produce a corrected response."

    def _tool_event(self, tool_type, invocation_state=None):
        event = MagicMock()
        event.tool_use = {"name": "MyModel"}
        event.selected_tool = MagicMock(tool_type=tool_type)
        event.invocation_state = {} if invocation_state is None else invocation_state
        event.cancel_tool = False
        return event

    def test_structured_output_attempt_is_failed(self, caplog):
        """The first structured-output attempt is cancelled with the parse-failure message."""
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        event = self._tool_event("structured_output")

        with caplog.at_level(logging.INFO):
            plugin.before_tool_call(event)

        assert event.cancel_tool == self._EXPECTED_MESSAGE
        assert "injected structured output parse failure" in caplog.text

    def test_injection_is_one_shot_per_invocation(self):
        """A second attempt in the same invocation passes through so the agent can recover."""
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        invocation_state = {}

        first = self._tool_event("structured_output", invocation_state)
        plugin.before_tool_call(first)
        assert first.cancel_tool == self._EXPECTED_MESSAGE

        second = self._tool_event("structured_output", invocation_state)
        plugin.before_tool_call(second)
        assert second.cancel_tool is False

    def test_ordinary_tool_is_unaffected(self):
        """A non-structured-output tool is not cancelled by MalformedJson."""
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        event = self._tool_event("function")

        plugin.before_tool_call(event)

        assert event.cancel_tool is False
        assert _CHAOS_STATE_KEY not in event.invocation_state

    def test_no_malformed_json_configured_writes_no_state(self):
        """Without MalformedJson the structured-output tool runs and no marker is written."""
        _set_chaos_case([Confabulation()])
        plugin = ChaosPlugin()
        event = self._tool_event("structured_output")

        plugin.before_tool_call(event)

        assert event.cancel_tool is False
        assert _CHAOS_STATE_KEY not in event.invocation_state

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestToolUseMessagesNeverCorrupted:
    """after_model_invocation leaves every message carrying a toolUse block untouched."""

    def test_structured_output_tooluse_untouched(self, caplog):
        """A structured-output toolUse message is not corrupted and nothing is logged."""
        _set_chaos_case([MalformedJson(), SuccessFraming()])
        plugin = ChaosPlugin()
        message = {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "so_1", "name": "MyModel", "input": {"field1": "value1"}}},
            ],
        }
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        with caplog.at_level(logging.INFO):
            plugin.after_model_invocation(event)

        assert message["content"] == original_content
        assert "applied model output chaos" not in caplog.text

    def test_ordinary_tooluse_untouched(self):
        """An ordinary mid-turn toolUse message is not corrupted."""
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        message = _tooluse_assistant_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def test_text_only_response_still_corrupted(self):
        """Final text responses remain corruptible."""
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        message = _final_assistant_message('{"key": "value", "nested": {"a": 1}}')
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"][0]["text"] != '{"key": "value", "nested": {"a": 1}}'

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


class TestSinglePreModelEffect:
    """At most one pre-hook model effect per case — pre effects cancel the model call."""

    def test_two_pre_effects_rejected(self):
        """FullRefusal + EmptyResponse (both pre) is rejected, naming both effects."""
        with pytest.raises(PydanticValidationError, match="only 1 is allowed"):
            ChaosCase(
                name="two_pre",
                input="test",
                effects={"model_effects": {"*": [FullRefusal(), EmptyResponse()]}},
            )

    def test_two_pre_effects_rejected_via_model_validate(self):
        """Two pre effects are rejected on the model_validate (dict) path."""
        with pytest.raises(PydanticValidationError, match="only 1 is allowed"):
            ChaosCase.model_validate(
                {
                    "name": "two_pre",
                    "input": "test",
                    "effects": {
                        "model_effects": {"*": [{"effect_type": "full_refusal"}, {"effect_type": "empty_response"}]}
                    },
                }
            )

    def test_rejection_names_both_effects(self):
        """The error message identifies both offending pre effects."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ChaosCase(
                name="two_pre",
                input="test",
                effects={"model_effects": {"*": [FullRefusal(), EmptyResponse()]}},
            )
        message = str(exc_info.value)
        assert "FullRefusal" in message
        assert "EmptyResponse" in message

    def test_single_pre_effect_accepted(self):
        """One pre effect alone is valid."""
        case = ChaosCase(
            name="one_pre",
            input="test",
            effects={"model_effects": {"*": [FullRefusal()]}},
        )
        assert len(case.model_effects) == 1

    def test_pre_plus_post_mix_accepted(self):
        """A pre + post mix is valid — only multiple pre effects are rejected."""
        case = ChaosCase(
            name="mixed",
            input="test",
            effects={"model_effects": {"*": [FullRefusal(), MalformedJson()]}},
        )
        assert len(case.model_effects) == 2


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
