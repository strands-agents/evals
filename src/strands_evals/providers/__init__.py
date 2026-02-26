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
    "LangfuseProvider",
    "ProviderError",
    "SessionNotFoundError",
    "TraceProvider",
    "TraceProviderError",
]


def __getattr__(name: str) -> Any:
    """Lazy-load providers that depend on optional packages."""
    if name == "LangfuseProvider":
        from .langfuse_provider import LangfuseProvider

        return LangfuseProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
