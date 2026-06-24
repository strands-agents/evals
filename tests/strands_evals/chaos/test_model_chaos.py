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

from strands_evals.chaos.model_types import (
    ModelOutputCorruptionConfig,
    ModelOutputCorruptionType,
    ModelOutputHallucinationType,
)
from strands_evals.chaos.model_utils import (
    _REFUSAL_TEMPLATES,
    _SUCCESS_PREFIXES,
)
from strands_evals.chaos.plugin import ChaosPlugin


def _make_event(message: dict) -> MagicMock:
    """Create a mock MessageAddedEvent with the given message.

    The event's `message` attribute is a real dict (not a Mock) so that
    dict mutation works as in production.
    """
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


# ---------------------------------------------------------------------------
# Per-effect corruption tests on final end_turn assistant messages
# ---------------------------------------------------------------------------


class TestModelChaosFormatCorruptionMalformedJson:
    """MALFORMED_JSON effect corrupts the final assistant message."""

    def test_malformed_json_corrupts_json_text(self):
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputCorruptionType.MALFORMED_JSON,
            )
        )
        message = _final_assistant_message('{"key": "value", "nested": {"a": 1}}')
        event = _make_event(message)

        plugin.message_added(event)

        # Content should be corrupted (JSON truncated)
        result_text = message["content"][0]["text"]
        assert result_text != '{"key": "value", "nested": {"a": 1}}'


class TestModelChaosFormatCorruptionEmptyResponse:
    """EMPTY_RESPONSE effect empties the final assistant message content."""

    def test_empty_response_on_final_message(self):
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputCorruptionType.EMPTY_RESPONSE,
            )
        )
        message = _final_assistant_message("Hello world")
        event = _make_event(message)

        plugin.message_added(event)

        assert message["content"] == []


class TestModelChaosHallucination:
    """Hallucination effect corrupts the final assistant message text."""

    def test_confabulation_injects_template(self):
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputHallucinationType.CONFABULATION,
            )
        )
        original_text = "The weather is sunny. It is warm outside. Birds are singing."
        message = _final_assistant_message(original_text)
        event = _make_event(message)

        plugin.message_added(event)

        result_text = message["content"][0]["text"]
        assert result_text != original_text
        # Should contain original text fragments
        assert "sunny" in result_text or "warm" in result_text


class TestModelChaosRefusal:
    """FULL_REFUSAL replaces the final assistant message content with a refusal."""

    def test_refusal_replaces_content(self):
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputCorruptionType.FULL_REFUSAL,
            )
        )
        message = _final_assistant_message("Here is the code you requested...")
        event = _make_event(message)

        plugin.message_added(event)

        # Content should be a single refusal text block
        assert len(message["content"]) == 1
        assert message["content"][0]["text"] in _REFUSAL_TEMPLATES


class TestModelChaosSuccessFraming:
    """Success framing prepends a confident prefix after corruption."""

    def test_success_framing_with_refusal(self):
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputCorruptionType.FULL_REFUSAL,
                add_success_framing=True,
            )
        )
        message = _final_assistant_message("Here is the code you requested...")
        event = _make_event(message)

        plugin.message_added(event)

        result_text = message["content"][0]["text"]
        assert any(result_text.startswith(prefix) for prefix in _SUCCESS_PREFIXES)


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


class TestModelChaosGuardToolUseMessage:
    """Messages with toolUse blocks are NOT corrupted (guard skips them)."""

    def test_empty_response_on_tooluse_message_not_corrupted(self):
        """EMPTY_RESPONSE on a tool_use message passes through; toolUse intact."""
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputCorruptionType.EMPTY_RESPONSE,
            )
        )
        message = _tooluse_assistant_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.message_added(event)

        # Content should be UNCHANGED — guard skipped corruption
        assert message["content"] == original_content
        # toolUse blocks should be intact
        tool_blocks = [b for b in message["content"] if "toolUse" in b]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["toolUse"]["name"] == "search"

    def test_full_refusal_on_tooluse_message_not_corrupted(self):
        """FULL_REFUSAL on a tool_use message passes through; toolUse intact."""
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputCorruptionType.FULL_REFUSAL,
            )
        )
        message = _tooluse_assistant_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.message_added(event)

        assert message["content"] == original_content


class TestModelChaosGuardStructuredOutputPath:
    """structured_output_model path fires MessageAddedEvent with toolUse — guard skips it."""

    def test_structured_output_tool_message_not_corrupted(self):
        """A message containing the structured output tool call is NOT corrupted."""
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputCorruptionType.EMPTY_RESPONSE,
            )
        )
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

        plugin.message_added(event)

        # Guard should skip — toolUse block present
        assert message["content"] == original_content


class TestModelChaosGuardFinalMessage:
    """Final end_turn assistant message IS corrupted."""

    def test_final_end_turn_message_corrupted(self):
        """An end_turn message with text only (no toolUse) gets corrupted."""
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputCorruptionType.EMPTY_RESPONSE,
            )
        )
        message = _final_assistant_message("Hello world")
        event = _make_event(message)

        plugin.message_added(event)

        assert message["content"] == []


class TestModelChaosGuardRoleFiltering:
    """User and tool result messages are NOT corrupted."""

    def test_user_message_not_corrupted(self):
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputCorruptionType.EMPTY_RESPONSE,
            )
        )
        message = _user_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.message_added(event)

        assert message["content"] == original_content

    def test_tool_result_message_not_corrupted(self):
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputCorruptionType.EMPTY_RESPONSE,
            )
        )
        message = _tool_result_message()
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.message_added(event)

        assert message["content"] == original_content


class TestModelChaosPassthrough:
    """No corruption when no model_output_config is set."""

    def test_no_config_passes_through(self):
        plugin = ChaosPlugin()  # No model_output_config
        message = _final_assistant_message("Hello world")
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.message_added(event)

        assert message["content"] == original_content

    def test_zero_apply_rate_passes_through(self):
        plugin = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=0.0,
                corruption_type=ModelOutputCorruptionType.EMPTY_RESPONSE,
            )
        )
        message = _final_assistant_message("Hello world")
        original_content = copy.deepcopy(message["content"])
        event = _make_event(message)

        plugin.message_added(event)

        assert message["content"] == original_content
