"""Tests for strands_evals.telemetry.tracer module."""

import json
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.trace import NoOpTracerProvider

from strands_evals.telemetry.tracer import get_tracer, serialize


def test_get_tracer_with_sdk_tracer_provider():
    """Test get_tracer returns tracer from SDK tracer provider"""
    mock_provider = MagicMock(spec=SDKTracerProvider)
    mock_tracer = MagicMock()
    mock_provider.get_tracer.return_value = mock_tracer

    with patch("strands_evals.telemetry.tracer.trace_api.get_tracer_provider", return_value=mock_provider):
        tracer = get_tracer()

        mock_provider.get_tracer.assert_called_once_with("strands-evals")
        assert tracer == mock_tracer


def test_get_tracer_with_non_sdk_tracer_provider():
    """Test get_tracer falls back to NoOpTracerProvider for non-SDK providers"""
    mock_provider = MagicMock()  # Not an SDKTracerProvider
    mock_tracer = MagicMock()

    with patch("strands_evals.telemetry.tracer.trace_api.get_tracer_provider", return_value=mock_provider):
        with patch("strands_evals.telemetry.tracer.trace_api.NoOpTracerProvider") as mock_noop_class:
            mock_noop_instance = MagicMock(spec=NoOpTracerProvider)
            mock_noop_instance.get_tracer.return_value = mock_tracer
            mock_noop_class.return_value = mock_noop_instance

            tracer = get_tracer()

            mock_noop_class.assert_called_once()
            mock_noop_instance.get_tracer.assert_called_once_with("strands-evals")
            assert tracer == mock_tracer


def test_serialize_simple_string():
    """Test serialize with a simple string"""
    result = serialize("hello")
    assert result == '"hello"'
    assert isinstance(result, str)


def test_serialize_simple_dict():
    """Test serialize with a simple dictionary"""
    data = {"key": "value", "number": 42}
    result = serialize(data)
    assert json.loads(result) == data


def test_serialize_nested_dict():
    """Test serialize with nested dictionary"""
    data = {"outer": {"inner": "value", "list": [1, 2, 3]}}
    result = serialize(data)
    assert json.loads(result) == data


def test_serialize_list():
    """Test serialize with a list"""
    data = [1, "two", {"three": 3}]
    result = serialize(data)
    assert json.loads(result) == data


def test_serialize_with_unicode():
    """Test serialize preserves unicode characters"""
    data = {"message": "Hello 世界 🌍"}
    result = serialize(data)
    # ensure_ascii=False should preserve unicode
    assert "世界" in result
    assert "🌍" in result
    assert json.loads(result) == data


def test_serialize_with_non_serializable_object():
    """Test serialize handles non-serializable objects with default=str"""

    class CustomObject:
        def __str__(self):
            return "CustomObject instance"

    obj = CustomObject()
    data = {"object": obj}
    result = serialize(data)
    parsed = json.loads(result)
    assert parsed["object"] == "CustomObject instance"


def test_serialize_with_none():
    """Test serialize handles None value"""
    result = serialize(None)
    assert result == "null"
    assert json.loads(result) is None


def test_serialize_with_boolean():
    """Test serialize handles boolean values"""
    assert serialize(True) == "true"
    assert serialize(False) == "false"


def test_serialize_with_number():
    """Test serialize handles numeric values"""
    assert serialize(42) == "42"
    assert serialize(3.14) == "3.14"


def test_serialize_empty_dict():
    """Test serialize with empty dictionary"""
    result = serialize({})
    assert result == "{}"
    assert json.loads(result) == {}


def test_serialize_empty_list():
    """Test serialize with empty list"""
    result = serialize([])
    assert result == "[]"
    assert json.loads(result) == []
