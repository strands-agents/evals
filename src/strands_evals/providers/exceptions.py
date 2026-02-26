"""Exceptions for trace providers."""


class TraceProviderError(Exception):
    """Base exception for trace provider errors."""


class SessionNotFoundError(TraceProviderError):
    """No traces found for the given session ID."""


class ProviderError(TraceProviderError):
    """Provider is unreachable or returned an error."""
