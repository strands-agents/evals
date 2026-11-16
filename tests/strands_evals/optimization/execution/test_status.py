import asyncio

import botocore.exceptions as botocore_errors
import pytest
import urllib3.exceptions as urllib3_errors
from botocore.exceptions import ClientError, EventStreamError
from strands.types.exceptions import EventLoopException, ModelThrottledException

from strands_evals.optimization import InfrastructureError
from strands_evals.optimization.execution.status import classify
from strands_evals.types.optimization import TrialStatus

INFRASTRUCTURE = TrialStatus.INFRASTRUCTURE_INVALID
AGENT = TrialStatus.AGENT_FAILURE


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "boom"}}, "ConverseStream")


CASES = [
    pytest.param(InfrastructureError("sandbox did not start"), INFRASTRUCTURE, id="InfrastructureError"),
    pytest.param(ModelThrottledException("slow down"), INFRASTRUCTURE, id="ModelThrottledException"),
    pytest.param(_client_error("ThrottlingException"), INFRASTRUCTURE, id="ClientError-ThrottlingException"),
    pytest.param(_client_error("TooManyRequestsException"), INFRASTRUCTURE, id="ClientError-TooManyRequests"),
    pytest.param(
        botocore_errors.ConnectTimeoutError(endpoint_url="https://x"), INFRASTRUCTURE, id="ConnectTimeoutError"
    ),
    pytest.param(
        botocore_errors.EndpointConnectionError(endpoint_url="https://x"), INFRASTRUCTURE, id="EndpointConnectionError"
    ),
    pytest.param(botocore_errors.NoCredentialsError(), INFRASTRUCTURE, id="NoCredentialsError"),
    pytest.param(botocore_errors.ReadTimeoutError(endpoint_url="https://x"), INFRASTRUCTURE, id="ReadTimeoutError"),
    pytest.param(botocore_errors.ResponseStreamingError(error="reset"), INFRASTRUCTURE, id="ResponseStreamingError"),
    pytest.param(
        botocore_errors.ConnectionClosedError(endpoint_url="https://x"), INFRASTRUCTURE, id="ConnectionClosedError"
    ),
    pytest.param(urllib3_errors.TimeoutError("stalled"), INFRASTRUCTURE, id="urllib3-TimeoutError"),
    pytest.param(urllib3_errors.ProtocolError("broken"), INFRASTRUCTURE, id="urllib3-ProtocolError"),
    pytest.param(urllib3_errors.NewConnectionError(None, "refused"), INFRASTRUCTURE, id="urllib3-NewConnectionError"),
    pytest.param(_client_error("InternalServerException"), INFRASTRUCTURE, id="InternalServerException"),
    pytest.param(_client_error("ServiceUnavailableException"), INFRASTRUCTURE, id="ServiceUnavailableException"),
    pytest.param(_client_error("ModelNotReadyException"), INFRASTRUCTURE, id="ModelNotReadyException"),
    pytest.param(
        EventStreamError({"Error": {"Code": "modelStreamErrorException", "Message": "x"}}, "ConverseStream"),
        INFRASTRUCTURE,
        id="EventStreamError-modelStreamErrorException",
    ),
    pytest.param(_client_error("ValidationException"), AGENT, id="ClientError-ValidationException"),
    pytest.param(ValueError("bad output"), AGENT, id="ValueError"),
    pytest.param(asyncio.TimeoutError(), AGENT, id="user-deadline"),
    pytest.param(KeyError("missing"), AGENT, id="KeyError"),
]


@pytest.mark.parametrize(("error", "status"), CASES)
def test_classify_bare(error, status):
    assert classify(error) is status


@pytest.mark.parametrize(("error", "status"), CASES)
def test_classify_unwraps_event_loop_exception(error, status):
    assert classify(EventLoopException(error)) is status


def test_classify_unwraps_nested_event_loop_exceptions():
    assert classify(EventLoopException(EventLoopException(ModelThrottledException("x")))) is INFRASTRUCTURE


def test_classify_ignores_a_response_that_is_not_a_mapping():
    error = ValueError("odd")
    error.response = "not a dict"  # type: ignore[attr-defined]
    assert classify(error) is AGENT
