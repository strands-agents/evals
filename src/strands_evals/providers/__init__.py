from typing import Any

from .exceptions import (
    ProviderError,
    SessionNotFoundError,
    TraceProviderError,
)
from .trace_provider import (
    TraceProvider,
)

__all__ = [
    "CloudWatchProvider",
    "LangfuseProvider",
    "OpenSearchProvider",
    "ProviderError",
    "SessionNotFoundError",
    "TraceProvider",
    "TraceProviderError",
]


def __getattr__(name: str) -> Any:
    """Lazy-load providers that depend on optional packages."""
    if name == "CloudWatchProvider":
        from .cloudwatch_provider import CloudWatchProvider

        return CloudWatchProvider
    if name == "LangfuseProvider":
        from .langfuse_provider import LangfuseProvider

        return LangfuseProvider
    if name == "OpenSearchProvider":
        from .opensearch_provider import OpenSearchProvider

        return OpenSearchProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
