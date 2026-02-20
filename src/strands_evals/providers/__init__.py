from .exceptions import (
    ProviderError,
    SessionNotFoundError,
    TraceNotFoundError,
    TraceProviderError,
)
from .trace_provider import (
    SessionFilter,
    TraceProvider,
)


def __getattr__(name: str):
    if name == "LangfuseProvider":
        from .langfuse_provider import LangfuseProvider

        return LangfuseProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LangfuseProvider",
    "ProviderError",
    "SessionFilter",
    "SessionNotFoundError",
    "TraceNotFoundError",
    "TraceProvider",
    "TraceProviderError",
]