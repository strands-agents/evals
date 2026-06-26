"""Unit tests for model output chaos via ChaosPlugin MessageAddedEvent callback.

Tests cover:
- Per-effect corruption on final assistant messages (end_turn, no toolUse)
- Guard: toolUse-carrying messages are NOT corrupted
- Guard: user/tool messages are NOT corrupted
- Guard: passthrough when no config set
- structured_output_model path: messages with toolUse are skipped (deferred scope)
"""

import copy
from unittest.mock import MagicMock

from strands_evals.chaos._context import _current_chaos_case
from strands_evals.chaos.case import ChaosCase
from strands_evals.chaos.effects import (
    Confabulation,
    EmptyResponse,
    FullRefusal,
    MalformedJson,
    SuccessFraming,
)
from strands_evals.chaos.plugin import ChaosPlugin


def _make_event(message: dict) -> MagicMock:
    """Create a mock MessageAddedEvent with the given message."""
    event = MagicMock()
    event.message = message
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
    """Helper to set the _current_chaos_case ContextVar with given model_effects."""
    case = ChaosCase(
        name="test_case",
        input="test input",
        model_effects=model_effects,
    )
    _current_chaos_case.set(case)
    return case


# ---------------------------------------------------------------------------
# Per-effect corruption tests on final end_turn assistant messages
# ---------------------------------------------------------------------------


class TestModelChaosFormatCorruptionMalformedJson:
    """MALFORMED_JSON effect corrupts the final assistant message."""

    def test_malformed_json_corrupts_json_text(self):
        _set_chaos_case([MalformedJson()])
        plugin = ChaosPlugin()
        message = _final_assistant_message('{"key": "value", "nested": {"a": 1}}')
        event = _make_event(message)

        plugin.after_model_invocation(event)

        # Content should be corrupted (JSON truncated)
        result_text = message["content"][0]["text"]
        assert result_text != '{"key": "value", "nested": {"a": 1}}'

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestModelChaosFormatCorruptionEmptyResponse:
    """EMPTY_RESPONSE effect empties the final assistant message content."""

    def test_empty_response_on_final_message(self):
        _set_chaos_case([EmptyResponse()])
        plugin = ChaosPlugin()
        message = _final_assistant_message("Hello world")
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"] == []

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestModelChaosHallucination:
    """Hallucination effect corrupts the final assistant message text."""

    def test_confabulation_injects_template(self):
        _set_chaos_case([Confabulation()])
        plugin = ChaosPlugin()
        original_text = "The weather is sunny. It is warm outside. Birds are singing."
        message = _final_assistant_message(original_text)
        event = _make_event(message)

        plugin.after_model_invocation(event)

        result_text = message["content"][0]["text"]
        assert result_text != original_text
        # Should contain original text fragments
        assert "sunny" in result_text or "warm" in result_text

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestModelChaosRefusal:
    """FULL_REFUSAL is a pre-hook effect — it uses before_model_invocation."""

    def test_refusal_cancels_model_call(self):
        _set_chaos_case([FullRefusal()])
        plugin = ChaosPlugin()
        event = MagicMock()

        plugin.before_model_invocation(event)

        # event.cancel should be set to a refusal template string
        assert event.cancel in FullRefusal._REFUSAL_TEMPLATES

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestModelChaosSuccessFraming:
    """Success framing prepends a confident prefix after corruption."""

    def test_success_framing_with_empty_response(self):
        # SuccessFraming is a post-hook composable effect
        _set_chaos_case([EmptyResponse(), SuccessFraming()])
        plugin = ChaosPlugin()
        message = _final_assistant_message("Here is the code you requested...")
        event = _make_event(message)

        plugin.after_model_invocation(event)

        # EmptyResponse clears content to [], then SuccessFraming prepends a
        # prefix block (disguises the emptied response with confident framing)
        assert len(message["content"]) == 1
        result_text = message["content"][0]["text"]
        assert result_text in SuccessFraming._SUCCESS_PREFIXES

    def teardown_method(self):
        _current_chaos_case.set(None)


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


class TestModelChaosGuardToolUseMessage:
    """Messages with toolUse blocks are NOT corrupted (guard skips them)."""

    def test_empty_response_on_tooluse_message_not_corrupted(self):
        """EMPTY_RESPONSE on a tool_use message passes through; toolUse intact."""
        _set_chaos_case([EmptyResponse()])
        plugin = ChaosPlugin()
        message = _tooluse_assistant_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        # Content should be UNCHANGED — guard skipped corruption
        assert message["content"] == original_content
        # toolUse blocks should be intact
        tool_blocks = [b for b in message["content"] if "toolUse" in b]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["toolUse"]["name"] == "search"

    def test_full_refusal_does_not_affect_post_hook(self):
        """FullRefusal is pre-hook only — after_model_invocation with toolUse still passes through."""
        _set_chaos_case([FullRefusal()])
        plugin = ChaosPlugin()
        message = _tooluse_assistant_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        # FullRefusal is pre-hook, so post hook has no post effects to apply
        assert message["content"] == original_content

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestModelChaosGuardStructuredOutputPath:
    """structured_output_model path fires MessageAddedEvent with toolUse — guard skips it."""

    def test_structured_output_tool_message_not_corrupted(self):
        """A message containing the structured output tool call is NOT corrupted."""
        _set_chaos_case([EmptyResponse()])
        plugin = ChaosPlugin()
        # Simulate the structured output tool invocation message
        message = {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "so_1",
                        "name": "structured_output__MyModel",
                        "input": {"field1": "value1"},
                    }
                },
            ],
        }
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        # Guard should skip — toolUse block present
        assert message["content"] == original_content

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestModelChaosGuardFinalMessage:
    """Final end_turn assistant message IS corrupted."""

    def test_final_end_turn_message_corrupted(self):
        """An end_turn message with text only (no toolUse) gets corrupted."""
        _set_chaos_case([EmptyResponse()])
        plugin = ChaosPlugin()
        message = _final_assistant_message("Hello world")
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"] == []

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestModelChaosGuardRoleFiltering:
    """User and tool result messages are NOT corrupted."""

    def test_user_message_not_corrupted(self):
        _set_chaos_case([EmptyResponse()])
        plugin = ChaosPlugin()
        message = _user_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def test_tool_result_message_not_corrupted(self):
        _set_chaos_case([EmptyResponse()])
        plugin = ChaosPlugin()
        message = _tool_result_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def teardown_method(self):
        _current_chaos_case.set(None)


class TestModelChaosPassthrough:
    """No corruption when no model_effects is set."""

    def test_no_config_passes_through(self):
        # No chaos case set — plugin should pass through
        _current_chaos_case.set(None)
        plugin = ChaosPlugin()
        message = _final_assistant_message("Hello world")
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.after_model_invocation(event)

        assert message["content"] == original_content


# ---------------------------------------------------------------------------
# Pre-hook integration test
# ---------------------------------------------------------------------------


class TestModelChaosPreHookIntegration:
    """Integration test: FullRefusal pre-hook produces one assistant turn."""

    def test_full_refusal_produces_single_turn(self):
        """FullRefusal cancels model call, SDK builds cancel message, run ends."""
        _set_chaos_case([FullRefusal()])
        plugin = ChaosPlugin()

        # Step 1: before_model_invocation fires
        pre_event = MagicMock()
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


# ---------------------------------------------------------------------------
# Mixed pre+post case test
# ---------------------------------------------------------------------------


class TestModelChaosMixedCase:
    """Mixed pre+post effects: pre wins, post does NOT double-corrupt."""

    def test_pre_plus_post_produces_single_uncorrupted_turn(self):
        """ChaosCase with FullRefusal + EmptyResponse: pre cancels, post skipped."""
        _set_chaos_case([FullRefusal(), EmptyResponse()])
        plugin = ChaosPlugin()

        # Pre-hook fires and cancels
        pre_event = MagicMock()
        plugin.before_model_invocation(pre_event)
        cancel_text = pre_event.cancel
        assert cancel_text in FullRefusal._REFUSAL_TEMPLATES

        # SDK builds cancel message, MessageAddedEvent fires
        cancel_message = {"role": "assistant", "content": [{"text": cancel_text}]}
        post_event = _make_event(cancel_message)
        plugin.after_model_invocation(post_event)

        # Post effect (EmptyResponse) should NOT have emptied the content
        assert cancel_message["content"] == [{"text": cancel_text}]
        assert len(cancel_message["content"]) == 1

    def teardown_method(self):
        _current_chaos_case.set(None)
