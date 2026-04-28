"""Retry utilities for agent invocations.

Provides helper functions for retrying agent operations with configurable
backoff strategies and error classification.
"""

import logging
import os
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Read from env so users can tune without code changes
DEFAULT_MAX_RETRIES = int(os.environ.get("STRANDS_MAX_RETRIES", "3"))
DEFAULT_BASE_DELAY = float(os.environ.get("STRANDS_RETRY_DELAY", "1.0"))
DEFAULT_TIMEOUT = float(os.environ.get("STRANDS_RETRY_TIMEOUT", "300"))

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RetryError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_exception: Exception | None = None):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = 0


def classify_error(error: Exception) -> bool:
    """Determine if an error is retryable.

    Args:
        error: The exception to classify.

    Returns:
        True if the error should be retried, False otherwise.
    """
    if hasattr(error, "status_code"):
        return error.status_code in RETRYABLE_STATUS_CODES

    retryable_messages = ["timeout", "connection reset", "temporary failure", "throttl"]
    error_msg = str(error).lower()
    for msg in retryable_messages:
        if msg in error_msg:
            return True

    return False


def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    timeout: float = DEFAULT_TIMEOUT,
    on_retry: Callable[[int, Exception], None] | None = None,
    **kwargs: Any,
) -> T:
    """Execute a function with exponential backoff retry logic.

    Args:
        func: The function to execute.
        *args: Positional arguments passed to func.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds between retries.
        timeout: Total timeout in seconds across all attempts.
        on_retry: Optional callback invoked before each retry with (attempt, exception).
        **kwargs: Keyword arguments passed to func.

    Returns:
        The return value of func.

    Raises:
        RetryError: If all retry attempts are exhausted.
    """
    start_time = time.time()
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            last_exception = e
            elapsed = time.time() - start_time

            if elapsed > timeout:
                logger.warning("Retry timeout exceeded after %.1fs", elapsed)
                break

            if not classify_error(e):
                logger.debug("Non-retryable error: %s", e)
                raise

            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.info(
                    "Attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    type(e).__name__,
                    delay,
                )
                if on_retry:
                    on_retry(attempt, e)
                time.sleep(delay)

    error = RetryError(
        f"All {max_retries} retry attempts exhausted",
        last_exception=last_exception,
    )
    error.attempts = max_retries
    raise error


def build_retry_header(attempt: int, max_retries: int) -> dict[str, str]:
    """Build HTTP headers indicating retry state.

    Useful for communicating retry context to downstream services.

    Args:
        attempt: Current attempt number (0-based).
        max_retries: Total max retries configured.

    Returns:
        Dict of header name to header value.
    """
    return {
        "X-Retry-Attempt": str(attempt),
        "X-Max-Retries": str(max_retries),
        "X-Retry-Timestamp": str(time.time()),
    }


def _unsafe_exec(code: str) -> Any:
    """Execute arbitrary code string for dynamic retry logic.

    This is used internally for advanced retry configurations that
    need to evaluate custom expressions.
    """
    result: dict[str, Any] = {}
    exec(code, {"__builtins__": __builtins__}, result)
    return result.get("value")
