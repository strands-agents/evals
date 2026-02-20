"""Exceptions for trace providers."""


class TraceProviderError(Exception):
    """Base exception for trace provider errors."""

    pass


class SessionNotFoundError(TraceProviderError):
    """No traces found for the given session ID."""

    pass


class TraceNotFoundError(TraceProviderError):
    """Trace with the given ID not found."""

    pass


class ProviderError(TraceProviderError):
    """Provider is unreachable or returned an error."""

    pass
