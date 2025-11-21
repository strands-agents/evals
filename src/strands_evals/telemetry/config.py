"""OpenTelemetry configuration and setup utilities for strands_evals.

This module provides centralized configuration and initialization functionality
for OpenTelemetry tracing in evaluation workflows.
"""

import logging
from typing import Any

import opentelemetry.trace as trace_api
from opentelemetry import propagate
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)


def get_otel_resource() -> Resource:
    """Create a standard OpenTelemetry resource with service information.

    Returns:
        Resource object with standard service information.
    """
    resource = Resource.create(
        {
            "service.name": "strands-evals",
            "service.version": "0.1.0",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
        }
    )

    return resource


class StrandsEvalsTelemetry:
    """OpenTelemetry configuration and setup for strands_evals.

    Automatically initializes a tracer provider with text map propagators.
    Trace exporters (console, OTLP) can be set up individually using dedicated methods
    that support method chaining for convenient configuration.

    Args:
        tracer_provider: Optional pre-configured SDKTracerProvider. If None,
            a new one will be created and set as the global tracer provider.

    Environment Variables:
        Environment variables are handled by the underlying OpenTelemetry SDK:
        - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
        - OTEL_EXPORTER_OTLP_HEADERS: Headers for OTLP requests

    Examples:
        Quick setup with method chaining:
        >>> StrandsEvalsTelemetry().setup_console_exporter().setup_otlp_exporter()

        Using a custom tracer provider:
        >>> StrandsEvalsTelemetry(tracer_provider=my_provider).setup_console_exporter()

        Step-by-step configuration:
        >>> telemetry = StrandsEvalsTelemetry()
        >>> telemetry.setup_console_exporter()
        >>> telemetry.setup_otlp_exporter()

    Note:
        - The tracer provider is automatically initialized upon instantiation
        - When no tracer_provider is provided, the instance sets itself as the global provider
        - Exporters must be explicitly configured using the setup methods
        - Failed exporter configurations are logged but do not raise exceptions
        - All setup methods return self to enable method chaining
    """

    def __init__(
        self,
        tracer_provider: SDKTracerProvider | None = None,
    ) -> None:
        """Initialize the StrandsEvalsTelemetry instance.

        Args:
            tracer_provider: Optional pre-configured tracer provider.
                If None, a new one will be created and set as global.

        The instance is ready to use immediately after initialization, though
        trace exporters must be configured separately using the setup methods.
        """
        self.resource = get_otel_resource()
        self._in_memory_exporter: InMemorySpanExporter | None = None
        self.tracer_provider: TracerProvider

        if tracer_provider:
            self.tracer_provider = tracer_provider
        else:
            self._initialize_tracer()

    def _initialize_tracer(self) -> None:
        """Initialize the OpenTelemetry tracer."""
        logger.info("Initializing tracer for strands-evals")

        # Create tracer provider
        tracer_provider = SDKTracerProvider(resource=self.resource)

        # Set as global tracer provider
        trace_api.set_tracer_provider(tracer_provider)

        # Get the global tracer provider (may be wrapped in a proxy)
        self.tracer_provider = trace_api.get_tracer_provider()

        # Set up propagators
        propagate.set_global_textmap(
            CompositePropagator(
                [
                    W3CBaggagePropagator(),
                    TraceContextTextMapPropagator(),
                ]
            )
        )

    @property
    def in_memory_exporter(self) -> InMemorySpanExporter:
        """Get the in-memory exporter instance.

        Returns:
            The configured in-memory span exporter.

        Raises:
            RuntimeError: If setup_in_memory_exporter() has not been called.
        """
        if self._in_memory_exporter is None:
            raise RuntimeError(
                "In-memory exporter is not configured. Call setup_in_memory_exporter() before accessing this property."
            )
        return self._in_memory_exporter

    def setup_console_exporter(self, **kwargs: Any) -> "StrandsEvalsTelemetry":
        """Set up console exporter for the tracer provider.

        Args:
            **kwargs: Optional keyword arguments passed directly to
                OpenTelemetry's ConsoleSpanExporter initializer.

        Returns:
            self: Enables method chaining.

        This method configures a SimpleSpanProcessor with a ConsoleSpanExporter,
        allowing trace data to be output to the console. Any additional keyword
        arguments provided will be forwarded to the ConsoleSpanExporter.
        """
        try:
            logger.info("Enabling console export for strands-evals")
            console_processor = SimpleSpanProcessor(ConsoleSpanExporter(**kwargs))
            self.tracer_provider.add_span_processor(console_processor)
        except Exception as e:
            logger.exception("error=<%s> | Failed to configure console exporter", e)
        return self

    def setup_in_memory_exporter(self, **kwargs: Any) -> "StrandsEvalsTelemetry":
        """Set up in-memory exporter for the tracer provider.

        Args:
            **kwargs: Optional keyword arguments passed directly to
                OpenTelemetry's InMemorySpanExporter initializer.

        Returns:
            self: Enables method chaining.

        This method configures a SimpleSpanProcessor with an InMemorySpanExporter,
        allowing trace data to be stored in memory for testing and debugging purposes.
        Any additional keyword arguments provided will be forwarded to the InMemorySpanExporter.
        """
        try:
            logger.info("Enabling in-memory export for strands-evals")
            self._in_memory_exporter = InMemorySpanExporter()
            span_processor = SimpleSpanProcessor(self._in_memory_exporter)
            self.tracer_provider.add_span_processor(span_processor)
        except Exception as e:
            logger.exception("error=<%s> | Failed to configure console exporter", e)
        return self

    def setup_otlp_exporter(self, **kwargs: Any) -> "StrandsEvalsTelemetry":
        """Set up OTLP exporter for the tracer provider.

        Args:
            **kwargs: Optional keyword arguments passed directly to
                OpenTelemetry's OTLPSpanExporter initializer.

        Returns:
            self: Enables method chaining.

        This method configures a BatchSpanProcessor with an OTLPSpanExporter,
        allowing trace data to be exported to an OTLP endpoint. Any additional
        keyword arguments provided will be forwarded to the OTLPSpanExporter.
        """
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        try:
            otlp_exporter = OTLPSpanExporter(**kwargs)
            batch_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(batch_processor)
            logger.info("OTLP exporter configured for strands-evals")
        except Exception as e:
            logger.exception("error=<%s> | Failed to configure OTLP exporter", e)
        return self
