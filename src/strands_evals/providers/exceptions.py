"""Exceptions for trace providers."""


class TraceProviderError(Exception):
    """Base exception for trace provider errors."""


class SessionNotFoundError(TraceProviderError):
    """No traces found for the given session ID."""


class TraceNotFoundError(TraceProviderError):
    """Trace with the given ID not found."""


class ProviderError(TraceProviderError):
    """Provider is unreachable or returned an error."""
