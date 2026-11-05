from botocore.exceptions import ClientError
from strands.types.exceptions import EventLoopException, ModelThrottledException

THROTTLING_ERROR_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "RequestLimitExceeded",
    "ServiceUnavailable",
    "ProvisionedThroughputExceededException",
}


def is_throttling_error(exception: BaseException) -> bool:
    """
    Check if an exception is a throttling/rate limiting error.

    An `EventLoopException` wraps any failure in the agent event loop, so it is judged by its
    `original_exception`; otherwise non-throttling failures would be retried as if throttled.

    Args:
        exception: The exception to check

    Returns:
        True if the exception indicates throttling, False otherwise
    """
    if isinstance(exception, EventLoopException):
        return is_throttling_error(exception.original_exception)

    # Check for Strands-specific throttling exceptions
    if isinstance(exception, ModelThrottledException):
        return True

    # Check for botocore.errorfactory.ThrottlingException (dynamically generated)
    if type(exception).__name__ == "ThrottlingException":
        return True

    # Check for botocore ClientError with throttling error codes
    if isinstance(exception, ClientError):
        error_code = exception.response.get("Error", {}).get("Code", "")
        if error_code in THROTTLING_ERROR_CODES:
            return True

    return False
