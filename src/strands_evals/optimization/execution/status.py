"""Trial status from the real exception object."""

from __future__ import annotations

import errno
from collections.abc import Mapping

import botocore.exceptions
from botocore.exceptions import (
    ConnectionClosedError,
    ConnectTimeoutError,
    CredentialRetrievalError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
    ReadTimeoutError,
    RefreshWithMFAUnsupportedError,
    ResponseStreamingError,
    SSOError,
    TokenRetrievalError,
)
from strands.types.exceptions import EventLoopException
from urllib3.exceptions import NewConnectionError, ProtocolError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

from ...types.optimization import TrialStatus
from ...utils import is_throttling_error


class InfrastructureError(Exception):
    """Raise from a rollout or environment to mark the trial `infrastructure_invalid`.

    The trial is re-run, never scored, e.g. when a sandbox failed to start.
    """


# No connection, or a response that stopped or broke while read. urllib3's own timeouts reach the agent
# unwrapped when a streamed response stalls.
_TRANSPORT_ERRORS = (
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
    ResponseStreamingError,
    ConnectionClosedError,
    Urllib3TimeoutError,
    ProtocolError,
    NewConnectionError,
)

# Credentials that are missing or can't be loaded or refreshed: a credential process that fails, an expired SSO,
# `aws login`, or MFA role session. The environment's, never the agent's.
_CREDENTIAL_ERRORS = (
    NoCredentialsError,
    PartialCredentialsError,
    CredentialRetrievalError,
    TokenRetrievalError,
    RefreshWithMFAUnsupportedError,
    SSOError,
    # Also what botocore reports when it couldn't open the config file (configparser skips a file it can't read,
    # e.g. out of file descriptors). A misspelled profile is permanent but can't be told apart: it stops the run at
    # the baseline after its re-runs.
    ProfileNotFound,
    # botocore >= 1.41 only: the `aws login` provider's errors.
    *filter(None, [getattr(botocore.exceptions, "LoginError", None)]),
)

# Out of file descriptors: the process (`EMFILE`) or the whole system (`ENFILE`).
_DESCRIPTOR_ERRNOS = frozenset({errno.EMFILE, errno.ENFILE})

# Error codes of the service's own failures and of expired caller credentials, lowercased; botocore reports them
# in `response["Error"]["Code"]`, also for an error inside a streamed response (`EventStreamError`). A wrong key
# (`UnrecognizedClientException`) or missing model access (`AccessDeniedException`) isn't here: no re-run fixes it.
_SERVICE_ERROR_CODES = frozenset(
    {
        "internalserverexception",
        "serviceunavailableexception",
        "modelstreamerrorexception",
        "modelnotreadyexception",
        "throttlingexception",
        "expiredtokenexception",
        "expiredtoken",
        "requestexpired",  # EC2-style APIs: expired temporary credentials, or a request signed too long ago
    }
)


def classify(error: BaseException) -> TrialStatus:
    """`INFRASTRUCTURE_INVALID` or `AGENT_FAILURE`, by these rules, in order.

    1. Unwrap: follow `EventLoopException.original_exception` to the cause.
    2. `InfrastructureError` -> `INFRASTRUCTURE_INVALID`.
    3. `utils.is_throttling_error(cause)` -> `INFRASTRUCTURE_INVALID`.
    4. A transport error -> `INFRASTRUCTURE_INVALID`: botocore `ConnectTimeoutError`, `EndpointConnectionError`,
       `ReadTimeoutError`, `ResponseStreamingError`, `ConnectionClosedError`; urllib3 `TimeoutError`,
       `ProtocolError`, `NewConnectionError`.
    5. A credential error -> `INFRASTRUCTURE_INVALID`: botocore `NoCredentialsError`, `PartialCredentialsError`,
       `CredentialRetrievalError`, `TokenRetrievalError`, `RefreshWithMFAUnsupportedError`, and `SSOError` and
       `LoginError` (botocore >= 1.41) with their subclasses; also `ProfileNotFound`.
    6. An `OSError` whose `errno` is `EMFILE` or `ENFILE` ("Too many open files") -> `INFRASTRUCTURE_INVALID`.
       Any other `OSError` goes on to the next rules.
    7. The service's own failure, or expired credentials, by error code (`response["Error"]["Code"]`,
       lowercased; read with `getattr`, so it also covers botocore `EventStreamError` in a streamed response):
       `internalserverexception`, `serviceunavailableexception`, `modelstreamerrorexception`,
       `modelnotreadyexception`, `throttlingexception`, `expiredtokenexception`, `expiredtoken`,
       `requestexpired` -> `INFRASTRUCTURE_INVALID`. A wrong key (`UnrecognizedClientException`) or missing
       model access (`AccessDeniedException`) is not: no re-run fixes it.
    8. Anything else -> `AGENT_FAILURE`, a user's own deadline (`asyncio.TimeoutError`) included, so a
       slow agent never sheds its hard cases by being re-run.

    Only these two come from an exception: `make_task` raises an environment setup error as
    `InfrastructureError` (rule 2) whatever its type, and `TrialRunner` sets `EVALUATOR_INVALID` when every
    evaluator is in `evaluator_errors`. Under `agent_failure_score="worst"` an infrastructure failure misread as
    `AGENT_FAILURE` would score the worst end instead of being re-run.

    Rules 5 and 6 also cover what models that each build their own `boto3.Session` raise when many trials run at
    once (see `Optimizer`): the trials share one process and one credential source, so running out of file
    descriptors or failing to get credentials says nothing about the candidate.
    """
    cause = error
    while isinstance(cause, EventLoopException) and cause.original_exception is not None:
        cause = cause.original_exception
    if isinstance(cause, InfrastructureError) or is_throttling_error(cause):
        return TrialStatus.INFRASTRUCTURE_INVALID
    if isinstance(cause, _TRANSPORT_ERRORS + _CREDENTIAL_ERRORS):
        return TrialStatus.INFRASTRUCTURE_INVALID
    if isinstance(cause, OSError) and cause.errno in _DESCRIPTOR_ERRNOS:
        return TrialStatus.INFRASTRUCTURE_INVALID
    response = getattr(cause, "response", None)
    details = response.get("Error") if isinstance(response, Mapping) else None
    code = details.get("Code") if isinstance(details, Mapping) else None
    if str(code or "").lower() in _SERVICE_ERROR_CODES:
        return TrialStatus.INFRASTRUCTURE_INVALID
    return TrialStatus.AGENT_FAILURE
