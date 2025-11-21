"""Tests for strands_evals.telemetry.config module."""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

from strands_evals.telemetry.config import StrandsEvalsTelemetry, get_otel_resource


@pytest.fixture
def mock_telemetry_patches():
    """Fixture to provide common mocks for telemetry tests."""
    with (
        patch("strands_evals.telemetry.config.SimpleSpanProcessor") as mock_simple,
        patch("strands_evals.telemetry.config.ConsoleSpanExporter") as mock_console,
        patch("strands_evals.telemetry.config.BatchSpanProcessor") as mock_batch,
        patch("strands_evals.telemetry.config.InMemorySpanExporter") as mock_memory,
        patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter") as mock_otlp,
    ):
        yield {
            "simple_processor": mock_simple,
            "console_exporter": mock_console,
            "batch_processor": mock_batch,
            "memory_exporter": mock_memory,
            "otlp_exporter": mock_otlp,
        }


@pytest.fixture
def mock_tracer_provider():
    """Fixture to provide a mock tracer provider."""
    return MagicMock(spec=SDKTracerProvider)


def test_get_otel_resource():
    """Test get_otel_resource returns properly configured Resource"""
    resource = get_otel_resource()

    assert isinstance(resource, Resource)
    attributes = resource.attributes
    assert attributes["service.name"] == "strands-evals"
    assert attributes["service.version"] == "0.1.0"
    assert attributes["telemetry.sdk.name"] == "opentelemetry"
    assert attributes["telemetry.sdk.language"] == "python"


def test_strands_evals_telemetry_init_without_provider():
    """Test StrandsEvalsTelemetry initialization without custom provider"""
    with (
        patch("strands_evals.telemetry.config.trace_api.set_tracer_provider") as mock_set_provider,
        patch("strands_evals.telemetry.config.trace_api.get_tracer_provider") as mock_get_provider,
        patch("strands_evals.telemetry.config.propagate.set_global_textmap") as mock_set_propagator,
    ):
        # Mock the returned tracer provider from get_tracer_provider
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        telemetry = StrandsEvalsTelemetry()

        # Verify tracer provider was set globally
        mock_set_provider.assert_called_once()
        # Verify we retrieved the global provider
        mock_get_provider.assert_called_once()
        # Verify we stored the retrieved provider
        assert telemetry.tracer_provider == mock_provider
        mock_set_propagator.assert_called_once()

        # Verify resource was created
        assert isinstance(telemetry.resource, Resource)


def test_strands_evals_telemetry_init_with_custom_provider():
    """Test StrandsEvalsTelemetry initialization with custom provider"""
    custom_provider = MagicMock(spec=SDKTracerProvider)

    with patch("strands_evals.telemetry.config.trace_api.set_tracer_provider") as mock_set_provider:
        telemetry = StrandsEvalsTelemetry(tracer_provider=custom_provider)

        # Verify custom provider was used
        assert telemetry.tracer_provider == custom_provider
        # Should not set global provider when custom one is provided
        mock_set_provider.assert_not_called()


def test_strands_evals_telemetry_setup_console_exporter(mock_telemetry_patches, mock_tracer_provider):
    """Test setup_console_exporter configures console exporter"""
    mock_processor = MagicMock()
    mock_telemetry_patches["simple_processor"].return_value = mock_processor
    mock_exporter = MagicMock()
    mock_telemetry_patches["console_exporter"].return_value = mock_exporter

    telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)
    result = telemetry.setup_console_exporter()

    # Verify console exporter was created
    mock_telemetry_patches["console_exporter"].assert_called_once_with()
    mock_telemetry_patches["simple_processor"].assert_called_once_with(mock_exporter)

    # Verify processor was added to tracer provider
    mock_tracer_provider.add_span_processor.assert_called_once_with(mock_processor)

    # Verify method chaining
    assert result is telemetry


def test_strands_evals_telemetry_setup_console_exporter_with_kwargs(mock_telemetry_patches):
    """Test setup_console_exporter passes kwargs to ConsoleSpanExporter"""
    telemetry = StrandsEvalsTelemetry()
    telemetry.setup_console_exporter(service_name="test-service", formatter=lambda x: x)

    # Verify kwargs were passed to exporter
    mock_telemetry_patches["console_exporter"].assert_called_once()
    call_kwargs = mock_telemetry_patches["console_exporter"].call_args[1]
    assert "service_name" in call_kwargs
    assert "formatter" in call_kwargs


def test_strands_evals_telemetry_setup_console_exporter_handles_exception():
    """Test setup_console_exporter handles exceptions gracefully"""
    with patch("strands_evals.telemetry.config.ConsoleSpanExporter", side_effect=Exception("Test error")):
        telemetry = StrandsEvalsTelemetry()

        # Should not raise exception
        result = telemetry.setup_console_exporter()

        # Should still return self for chaining
        assert result is telemetry


def test_strands_evals_telemetry_setup_in_memory_exporter(mock_telemetry_patches, mock_tracer_provider):
    """Test setup_in_memory_exporter configures in-memory exporter"""
    mock_processor = MagicMock()
    mock_telemetry_patches["simple_processor"].return_value = mock_processor
    mock_exporter = MagicMock()
    mock_telemetry_patches["memory_exporter"].return_value = mock_exporter

    telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)
    result = telemetry.setup_in_memory_exporter()

    # Verify in-memory exporter was created
    mock_telemetry_patches["memory_exporter"].assert_called_once_with()
    mock_telemetry_patches["simple_processor"].assert_called_once_with(mock_exporter)

    # Verify processor was added to tracer provider
    mock_tracer_provider.add_span_processor.assert_called_once_with(mock_processor)

    # Verify exporter is stored in instance variable
    assert telemetry._in_memory_exporter == mock_exporter

    # Verify method chaining
    assert result is telemetry


def test_strands_evals_telemetry_setup_in_memory_exporter_with_kwargs(mock_telemetry_patches):
    """Test setup_in_memory_exporter passes kwargs to InMemorySpanExporter"""
    telemetry = StrandsEvalsTelemetry()
    # InMemorySpanExporter doesn't take many kwargs, but we can test the pattern
    telemetry.setup_in_memory_exporter()

    # Verify exporter was called
    mock_telemetry_patches["memory_exporter"].assert_called_once_with()


def test_strands_evals_telemetry_setup_in_memory_exporter_handles_exception():
    """Test setup_in_memory_exporter handles exceptions gracefully"""
    with patch("strands_evals.telemetry.config.InMemorySpanExporter", side_effect=Exception("Test error")):
        telemetry = StrandsEvalsTelemetry()

        # Should not raise exception
        result = telemetry.setup_in_memory_exporter()

        # Should still return self for chaining
        assert result is telemetry


def test_strands_evals_telemetry_in_memory_exporter_property(mock_telemetry_patches, mock_tracer_provider):
    """Test in_memory_exporter property returns the configured exporter"""
    mock_exporter = MagicMock()
    mock_telemetry_patches["memory_exporter"].return_value = mock_exporter

    telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)
    telemetry.setup_in_memory_exporter()

    # Verify property returns the exporter
    assert telemetry.in_memory_exporter == mock_exporter


def test_strands_evals_telemetry_in_memory_exporter_property_raises_when_not_configured():
    """Test in_memory_exporter property raises RuntimeError when not configured"""
    telemetry = StrandsEvalsTelemetry()

    # Should raise RuntimeError when accessing property before setup
    with pytest.raises(RuntimeError, match="In-memory exporter is not configured"):
        _ = telemetry.in_memory_exporter


def test_strands_evals_telemetry_setup_otlp_exporter(mock_telemetry_patches, mock_tracer_provider):
    """Test setup_otlp_exporter configures OTLP exporter"""
    mock_processor = MagicMock()
    mock_telemetry_patches["batch_processor"].return_value = mock_processor
    mock_exporter = MagicMock()
    mock_telemetry_patches["otlp_exporter"].return_value = mock_exporter

    telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)
    result = telemetry.setup_otlp_exporter()

    # Verify OTLP exporter was created
    mock_telemetry_patches["otlp_exporter"].assert_called_once_with()
    mock_telemetry_patches["batch_processor"].assert_called_once_with(mock_exporter)

    # Verify processor was added to tracer provider
    mock_tracer_provider.add_span_processor.assert_called_once_with(mock_processor)

    # Verify method chaining works
    assert result is telemetry


def test_strands_evals_telemetry_setup_otlp_exporter_with_kwargs(mock_telemetry_patches, mock_tracer_provider):
    """Test setup_otlp_exporter passes kwargs to OTLPSpanExporter"""
    telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)
    telemetry.setup_otlp_exporter(endpoint="http://localhost:4318", headers={"key": "value"})

    # Verify kwargs were passed to exporter
    mock_telemetry_patches["otlp_exporter"].assert_called_once()
    call_kwargs = mock_telemetry_patches["otlp_exporter"].call_args[1]
    assert "endpoint" in call_kwargs
    assert "headers" in call_kwargs


def test_strands_evals_telemetry_setup_otlp_exporter_handles_exception():
    """Test setup_otlp_exporter handles exceptions gracefully"""
    with patch(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        side_effect=Exception("Test error"),
    ):
        mock_tracer_provider = MagicMock()
        telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)

        # Should not raise exception
        result = telemetry.setup_otlp_exporter()

        # Should still return self for chaining
        assert result is telemetry


def test_strands_evals_telemetry_method_chaining(mock_telemetry_patches, mock_tracer_provider):
    """Test method chaining works for multiple exporter setups"""
    telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)
    result = telemetry.setup_console_exporter().setup_otlp_exporter()

    # Verify both processors were added
    assert mock_tracer_provider.add_span_processor.call_count == 2

    # Verify chaining returns the same instance
    assert result is telemetry


def test_strands_evals_telemetry_one_liner_setup(mock_telemetry_patches):
    """Test one-liner setup with method chaining"""
    # Mock the tracer provider's add_span_processor method
    mock_tracer_provider = MagicMock()

    with (
        patch("strands_evals.telemetry.config.SDKTracerProvider", return_value=mock_tracer_provider),
        patch("strands_evals.telemetry.config.trace_api.set_tracer_provider"),
        patch("strands_evals.telemetry.config.trace_api.get_tracer_provider", return_value=mock_tracer_provider),
        patch("strands_evals.telemetry.config.propagate.set_global_textmap"),
    ):
        # This should work as a one-liner
        telemetry = StrandsEvalsTelemetry().setup_console_exporter().setup_otlp_exporter()

        assert isinstance(telemetry, StrandsEvalsTelemetry)
        # Verify both processors were added
        assert mock_tracer_provider.add_span_processor.call_count == 2


def test_strands_evals_telemetry_propagators_configured():
    """Test that propagators are configured correctly"""
    with (
        patch("strands_evals.telemetry.config.propagate.set_global_textmap") as mock_set_propagator,
        patch("strands_evals.telemetry.config.CompositePropagator") as mock_composite,
        patch("strands_evals.telemetry.config.W3CBaggagePropagator") as mock_baggage,
        patch("strands_evals.telemetry.config.TraceContextTextMapPropagator") as mock_trace_context,
    ):
        mock_baggage_instance = MagicMock()
        mock_trace_context_instance = MagicMock()
        mock_baggage.return_value = mock_baggage_instance
        mock_trace_context.return_value = mock_trace_context_instance

        StrandsEvalsTelemetry()

        # Verify propagators were created
        mock_baggage.assert_called_once()
        mock_trace_context.assert_called_once()

        # Verify composite propagator was created with both propagators
        mock_composite.assert_called_once()
        composite_args = mock_composite.call_args[0][0]
        assert mock_baggage_instance in composite_args
        assert mock_trace_context_instance in composite_args

        # Verify global textmap was set
        mock_set_propagator.assert_called_once()
