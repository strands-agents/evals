"""Trial status from the real exception object."""

from __future__ import annotations

from collections.abc import Mapping

from strands.types.exceptions import EventLoopException

from ...types.optimization import TrialStatus
from ...utils import is_throttling_error


class InfrastructureError(Exception):
    """Raise from a rollout or environment to mark the trial `infrastructure_invalid`.

    The trial is re-run, never scored, e.g. when a sandbox failed to start.
    """


# The model service's own failure codes, lowercased; botocore reports them in `response["Error"]["Code"]`,
# also for an error inside a streamed response (`EventStreamError`).
_SERVICE_ERROR_CODES = frozenset(
    {
        "internalserverexception",
        "serviceunavailableexception",
        "modelstreamerrorexception",
        "modelnotreadyexception",
        "throttlingexception",
    }
)


def classify(error: BaseException) -> TrialStatus:
    """`INFRASTRUCTURE_INVALID` or `AGENT_FAILURE`, by these rules, in order.

    1. Unwrap: follow `EventLoopException.original_exception` to the cause.
    2. `InfrastructureError` -> `INFRASTRUCTURE_INVALID`.
    3. `utils.is_throttling_error(cause)` -> `INFRASTRUCTURE_INVALID`.
    4. A transport error -> `INFRASTRUCTURE_INVALID`: botocore `ConnectTimeoutError`,
       `EndpointConnectionError`, `NoCredentialsError`, `ReadTimeoutError`, `ResponseStreamingError`,
       `ConnectionClosedError`; urllib3 `TimeoutError`, `ProtocolError`, `NewConnectionError` (each looked
       up defensively, so a missing one is skipped).
    5. The model service's own error code (`response["Error"]["Code"]`, lowercased; read with
       `getattr`, so it also covers botocore `EventStreamError` in a streamed response):
       `internalserverexception`, `serviceunavailableexception`, `modelstreamerrorexception`,
       `modelnotreadyexception`, `throttlingexception` -> `INFRASTRUCTURE_INVALID`.
    6. Anything else -> `AGENT_FAILURE`, a user's own deadline (`asyncio.TimeoutError`) included, so a
       slow agent never sheds its hard cases by being re-run.

    Only these two come from an exception: `make_task` raises an environment setup error as
    `InfrastructureError` (rule 2) whatever its type, and `TrialRunner` sets `EVALUATOR_INVALID` when every
    evaluator is in `evaluator_errors`. Under `agent_failure_score="worst"` an infrastructure failure misread as
    `AGENT_FAILURE` would score the worst end instead of being re-run.
    """
    cause = error
    while isinstance(cause, EventLoopException) and cause.original_exception is not None:
        cause = cause.original_exception
    if isinstance(cause, InfrastructureError) or is_throttling_error(cause):
        return TrialStatus.INFRASTRUCTURE_INVALID
    if isinstance(cause, _transport_errors()):
        return TrialStatus.INFRASTRUCTURE_INVALID
    response = getattr(cause, "response", None)
    details = response.get("Error") if isinstance(response, Mapping) else None
    code = details.get("Code") if isinstance(details, Mapping) else None
    if str(code or "").lower() in _SERVICE_ERROR_CODES:
        return TrialStatus.INFRASTRUCTURE_INVALID
    return TrialStatus.AGENT_FAILURE


def _transport_errors() -> tuple[type[BaseException], ...]:
    """The transport's errors: no connection, no credentials, or a response that stopped or broke while read.

    urllib3's own timeouts reach the agent unwrapped when a streamed response stalls.
    """
    errors: list[type[BaseException]] = []
    try:
        import botocore.exceptions as botocore_errors
    except ImportError:  # pragma: no cover - botocore ships with strands' Bedrock model
        pass
    else:
        for name in (
            "ConnectTimeoutError",
            "EndpointConnectionError",
            "NoCredentialsError",
            "ReadTimeoutError",
            "ResponseStreamingError",
            "ConnectionClosedError",
        ):
            if hasattr(botocore_errors, name):
                errors.append(getattr(botocore_errors, name))
    try:
        import urllib3.exceptions as urllib3_errors
    except ImportError:  # pragma: no cover - urllib3 ships with botocore
        pass
    else:
        for name in ("TimeoutError", "ProtocolError", "NewConnectionError"):
            if hasattr(urllib3_errors, name):
                errors.append(getattr(urllib3_errors, name))
    return tuple(errors)
