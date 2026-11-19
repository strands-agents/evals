import asyncio
import errno

import botocore.exceptions as botocore_errors
import pytest
import urllib3.exceptions as urllib3_errors
from botocore.exceptions import ClientError, EventStreamError
from strands.types.exceptions import EventLoopException, ModelThrottledException

from strands_evals.optimization import InfrastructureError
from strands_evals.optimization.execution.status import classify, is_credential_error
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
    pytest.param(
        botocore_errors.PartialCredentialsError(provider="env", cred_var="x"), INFRASTRUCTURE, id="PartialCredentials"
    ),
    pytest.param(
        botocore_errors.CredentialRetrievalError(provider="custom-process", error_msg="run mwinit"),
        INFRASTRUCTURE,
        id="CredentialRetrievalError",
    ),
    pytest.param(
        botocore_errors.TokenRetrievalError(provider="sso", error_msg="expired"), INFRASTRUCTURE, id="TokenRetrieval"
    ),
    pytest.param(botocore_errors.UnauthorizedSSOTokenError(), INFRASTRUCTURE, id="UnauthorizedSSOTokenError"),
    pytest.param(botocore_errors.SSOTokenLoadError(error_msg="no cache"), INFRASTRUCTURE, id="SSOTokenLoadError"),
    pytest.param(_client_error("ExpiredTokenException"), INFRASTRUCTURE, id="ClientError-ExpiredTokenException"),
    pytest.param(_client_error("ExpiredToken"), INFRASTRUCTURE, id="ClientError-ExpiredToken"),
    pytest.param(_client_error("RequestExpired"), INFRASTRUCTURE, id="ClientError-RequestExpired"),
    pytest.param(botocore_errors.SSOError(), INFRASTRUCTURE, id="SSOError"),
    pytest.param(botocore_errors.RefreshWithMFAUnsupportedError(), INFRASTRUCTURE, id="RefreshWithMFAUnsupported"),
    *(
        [
            pytest.param(botocore_errors.LoginRefreshRequired(), INFRASTRUCTURE, id="LoginRefreshRequired"),
            pytest.param(
                botocore_errors.LoginTokenLoadError(error_msg="no cache"), INFRASTRUCTURE, id="LoginTokenLoad"
            ),
        ]
        if hasattr(botocore_errors, "LoginError")  # botocore >= 1.41 only
        else []
    ),
    pytest.param(botocore_errors.ProfileNotFound(profile="bedrock"), INFRASTRUCTURE, id="ProfileNotFound"),
    pytest.param(OSError(errno.EMFILE, "Too many open files"), INFRASTRUCTURE, id="OSError-EMFILE"),
    pytest.param(OSError(errno.ENFILE, "Too many open files in system"), INFRASTRUCTURE, id="OSError-ENFILE"),
    # A wrong key or missing model access: no re-run fixes it.
    pytest.param(_client_error("UnrecognizedClientException"), AGENT, id="ClientError-UnrecognizedClientException"),
    pytest.param(_client_error("AccessDeniedException"), AGENT, id="ClientError-AccessDeniedException"),
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
    pytest.param(OSError(errno.ENOENT, "No such file or directory"), AGENT, id="OSError-ENOENT"),
    pytest.param(OSError("no errno"), AGENT, id="OSError-no-errno"),
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


CREDENTIAL_IDS = {
    "NoCredentialsError",
    "PartialCredentials",
    "CredentialRetrievalError",
    "TokenRetrieval",
    "UnauthorizedSSOTokenError",
    "SSOTokenLoadError",
    "SSOError",
    "RefreshWithMFAUnsupported",
    "LoginRefreshRequired",
    "LoginTokenLoad",
    "ProfileNotFound",
    "ClientError-ExpiredTokenException",
    "ClientError-ExpiredToken",
    "ClientError-RequestExpired",
}


@pytest.mark.parametrize(("error", "status"), CASES)
def test_is_credential_error_picks_exactly_the_credential_family(error, status, request):
    expected = request.node.callspec.id in CREDENTIAL_IDS
    assert is_credential_error(error) is expected
    assert is_credential_error(EventLoopException(error)) is expected  # unwraps like classify
