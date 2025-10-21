"""OpenTelemetry tracing for strands_evals.

This module provides a simple way to get the OpenTelemetry tracer
for evaluation workflows.
"""

import json
import logging
from typing import Any

import opentelemetry.trace as trace_api
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

logger = logging.getLogger(__name__)


def get_tracer() -> trace_api.Tracer:
    """Get the OpenTelemetry tracer for strands_evals.

    Returns:
        OpenTelemetry Tracer instance from the global tracer provider
    """
    tracer_provider = trace_api.get_tracer_provider()
    if not isinstance(tracer_provider, SDKTracerProvider):
        tracer_provider = trace_api.NoOpTracerProvider()
    return tracer_provider.get_tracer("strands-evals")


def serialize(obj: Any) -> str:
    """Serialize an object to JSON string.

    Args:
        obj: The object to serialize

    Returns:
        JSON string representation
    """
    return json.dumps(obj, ensure_ascii=False, default=str)
