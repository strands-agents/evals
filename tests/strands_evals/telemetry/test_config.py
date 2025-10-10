"""Tests for strands_evals.telemetry.config module."""

from unittest.mock import MagicMock, patch, call

import pytest
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.resources import Resource

from strands_evals.telemetry.config import get_otel_resource, StrandsEvalsTelemetry


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
    with patch("strands_evals.telemetry.config.trace_api.set_tracer_provider") as mock_set_provider:
        with patch("strands_evals.telemetry.config.propagate.set_global_textmap") as mock_set_propagator:
            telemetry = StrandsEvalsTelemetry()

            # Verify tracer provider was created and set
            assert isinstance(telemetry.tracer_provider, SDKTracerProvider)
            mock_set_provider.assert_called_once_with(telemetry.tracer_provider)
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


def test_strands_evals_telemetry_setup_console_exporter():
    """Test setup_console_exporter configures console exporter"""
    with patch("strands_evals.telemetry.config.SimpleSpanProcessor") as mock_processor_class:
        with patch("strands_evals.telemetry.config.ConsoleSpanExporter") as mock_exporter_class:
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            mock_exporter = MagicMock()
            mock_exporter_class.return_value = mock_exporter

            # Mock the tracer provider
            mock_tracer_provider = MagicMock()
            telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)
            result = telemetry.setup_console_exporter()

            # Verify console exporter was created
            mock_exporter_class.assert_called_once_with()
            mock_processor_class.assert_called_once_with(mock_exporter)

            # Verify processor was added to tracer provider
            mock_tracer_provider.add_span_processor.assert_called_once_with(mock_processor)

            # Verify method chaining
            assert result is telemetry


def test_strands_evals_telemetry_setup_console_exporter_with_kwargs():
    """Test setup_console_exporter passes kwargs to ConsoleSpanExporter"""
    with patch("strands_evals.telemetry.config.SimpleSpanProcessor"):
        with patch("strands_evals.telemetry.config.ConsoleSpanExporter") as mock_exporter_class:
            telemetry = StrandsEvalsTelemetry()
            telemetry.setup_console_exporter(service_name="test-service", formatter=lambda x: x)

            # Verify kwargs were passed to exporter
            mock_exporter_class.assert_called_once()
            call_kwargs = mock_exporter_class.call_args[1]
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


def test_strands_evals_telemetry_setup_otlp_exporter():
    """Test setup_otlp_exporter configures OTLP exporter"""
    with patch("strands_evals.telemetry.config.BatchSpanProcessor") as mock_processor_class:
        with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter") as mock_exporter_class:
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            mock_exporter = MagicMock()
            mock_exporter_class.return_value = mock_exporter

            # Mock the tracer provider
            mock_tracer_provider = MagicMock()
            telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)
            result = telemetry.setup_otlp_exporter()

            # Verify OTLP exporter was created
            mock_exporter_class.assert_called_once_with()
            mock_processor_class.assert_called_once_with(mock_exporter)

            # Verify processor was added to tracer provider
            mock_tracer_provider.add_span_processor.assert_called_once_with(mock_processor)

            # Verify method chaining works
            assert result is telemetry


def test_strands_evals_telemetry_setup_otlp_exporter_with_kwargs():
    """Test setup_otlp_exporter passes kwargs to OTLPSpanExporter"""
    with patch("strands_evals.telemetry.config.BatchSpanProcessor"):
        with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter") as mock_exporter_class:
            mock_tracer_provider = MagicMock()
            telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)
            telemetry.setup_otlp_exporter(endpoint="http://localhost:4318", headers={"key": "value"})

            # Verify kwargs were passed to exporter
            mock_exporter_class.assert_called_once()
            call_kwargs = mock_exporter_class.call_args[1]
            assert "endpoint" in call_kwargs
            assert "headers" in call_kwargs


def test_strands_evals_telemetry_setup_otlp_exporter_handles_exception():
    """Test setup_otlp_exporter handles exceptions gracefully"""
    with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter", side_effect=Exception("Test error")):
        mock_tracer_provider = MagicMock()
        telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)

        # Should not raise exception
        result = telemetry.setup_otlp_exporter()

        # Should still return self for chaining
        assert result is telemetry


def test_strands_evals_telemetry_method_chaining():
    """Test method chaining works for multiple exporter setups"""
    with patch("strands_evals.telemetry.config.SimpleSpanProcessor"):
        with patch("strands_evals.telemetry.config.ConsoleSpanExporter"):
            with patch("strands_evals.telemetry.config.BatchSpanProcessor"):
                with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"):
                    mock_tracer_provider = MagicMock()
                    telemetry = StrandsEvalsTelemetry(tracer_provider=mock_tracer_provider)
                    result = telemetry.setup_console_exporter().setup_otlp_exporter()

                    # Verify both processors were added
                    assert mock_tracer_provider.add_span_processor.call_count == 2

                    # Verify chaining returns the same instance
                    assert result is telemetry


def test_strands_evals_telemetry_one_liner_setup():
    """Test one-liner setup with method chaining"""
    with patch("strands_evals.telemetry.config.SimpleSpanProcessor") as mock_simple_processor:
        with patch("strands_evals.telemetry.config.ConsoleSpanExporter"):
            with patch("strands_evals.telemetry.config.BatchSpanProcessor") as mock_batch_processor:
                with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"):
                    # Mock the tracer provider's add_span_processor method
                    mock_tracer_provider = MagicMock()
                    
                    with patch("strands_evals.telemetry.config.SDKTracerProvider", return_value=mock_tracer_provider):
                        with patch("strands_evals.telemetry.config.trace_api.set_tracer_provider"):
                            with patch("strands_evals.telemetry.config.propagate.set_global_textmap"):
                                # This should work as a one-liner
                                telemetry = StrandsEvalsTelemetry().setup_console_exporter().setup_otlp_exporter()

                                assert isinstance(telemetry, StrandsEvalsTelemetry)
                                # Verify both processors were added
                                assert mock_tracer_provider.add_span_processor.call_count == 2


def test_strands_evals_telemetry_propagators_configured():
    """Test that propagators are configured correctly"""
    with patch("strands_evals.telemetry.config.propagate.set_global_textmap") as mock_set_propagator:
        with patch("strands_evals.telemetry.config.CompositePropagator") as mock_composite:
            with patch("strands_evals.telemetry.config.W3CBaggagePropagator") as mock_baggage:
                with patch("strands_evals.telemetry.config.TraceContextTextMapPropagator") as mock_trace_context:
                    mock_baggage_instance = MagicMock()
                    mock_trace_context_instance = MagicMock()
                    mock_baggage.return_value = mock_baggage_instance
                    mock_trace_context.return_value = mock_trace_context_instance

                    telemetry = StrandsEvalsTelemetry()

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
