"""Trial status from the real exception object."""

from __future__ import annotations

from ...types.optimization import TrialStatus


class InfrastructureError(Exception):
    """Raise from a rollout or environment to mark the trial `infrastructure_invalid`.

    The trial is re-run, never scored, e.g. when a sandbox failed to start.
    """


def classify(error: BaseException) -> TrialStatus:
    """`INFRASTRUCTURE_INVALID` or `AGENT_FAILURE`, by these rules, in order.

    1. Unwrap: follow `EventLoopException.original_exception` to the cause.
    2. `InfrastructureError` -> `INFRASTRUCTURE_INVALID`.
    3. `utils.is_throttling_error(cause)`, unless the cause is still an `EventLoopException` ->
       `INFRASTRUCTURE_INVALID`.
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
    `InfrastructureError` (rule 2) whatever its type, and `TrialRunner` sets `VERIFIER_INVALID` when every
    evaluator is in `evaluator_errors`. Under `crash_score="worst"` an infrastructure failure misread as
    `AGENT_FAILURE` would score the worst end instead of being re-run.
    """
    raise NotImplementedError
