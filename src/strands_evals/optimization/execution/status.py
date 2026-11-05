"""Trial status from the real exception object."""

from __future__ import annotations

from collections.abc import Mapping

import botocore.exceptions
import urllib3.exceptions
from strands.types.exceptions import EventLoopException

from ...types.optimization import TrialStatus
from ...utils import is_throttling_error


class InfrastructureError(Exception):
    """Raise from a rollout or environment to mark the trial `infrastructure_invalid`.

    The trial is re-run, never scored, e.g. when a sandbox failed to start.
    """


# No connection, no credentials, or a response that stopped or broke while read. urllib3's own timeouts
# reach the agent unwrapped when a streamed response stalls.
_TRANSPORT_ERRORS = (
    botocore.exceptions.ConnectTimeoutError,
    botocore.exceptions.EndpointConnectionError,
    botocore.exceptions.NoCredentialsError,
    botocore.exceptions.ReadTimeoutError,
    botocore.exceptions.ResponseStreamingError,
    botocore.exceptions.ConnectionClosedError,
    urllib3.exceptions.TimeoutError,
    urllib3.exceptions.ProtocolError,
    urllib3.exceptions.NewConnectionError,
)

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
       `ConnectionClosedError`; urllib3 `TimeoutError`, `ProtocolError`, `NewConnectionError`.
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
    if isinstance(cause, _TRANSPORT_ERRORS):
        return TrialStatus.INFRASTRUCTURE_INVALID
    response = getattr(cause, "response", None)
    details = response.get("Error") if isinstance(response, Mapping) else None
    code = details.get("Code") if isinstance(details, Mapping) else None
    if str(code or "").lower() in _SERVICE_ERROR_CODES:
        return TrialStatus.INFRASTRUCTURE_INVALID
    return TrialStatus.AGENT_FAILURE
