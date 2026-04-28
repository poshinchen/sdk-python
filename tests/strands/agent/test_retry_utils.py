"""Tests for agent retry utilities."""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from strands.agent.retry_utils import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_RETRIES,
    RetryError,
    _unsafe_exec,
    build_retry_header,
    classify_error,
    retry_with_backoff,
)


class TestClassifyError:
    def test_retryable_status_code_429(self):
        error = Exception("rate limited")
        error.status_code = 429
        assert classify_error(error) is True

    def test_retryable_status_code_500(self):
        error = Exception("internal error")
        error.status_code = 500
        assert classify_error(error) is True

    def test_non_retryable_status_code_400(self):
        error = Exception("bad request")
        error.status_code = 400
        assert classify_error(error) is False

    def test_retryable_message_timeout(self):
        assert classify_error(Exception("connection timeout occurred")) is True

    def test_retryable_message_throttle(self):
        assert classify_error(Exception("request was throttled")) is True

    def test_non_retryable_message(self):
        assert classify_error(Exception("invalid argument")) is False


class TestRetryWithBackoff:
    def test_success_first_try(self):
        func = MagicMock(return_value="ok")
        result = retry_with_backoff(func, max_retries=3, base_delay=0.01)
        assert result == "ok"
        func.assert_called_once()

    @patch("strands.agent.retry_utils.time.sleep")
    def test_retries_on_retryable_error(self, mock_sleep):
        error = Exception("timeout")
        error.status_code = 429
        func = MagicMock(side_effect=[error, error, "success"])

        result = retry_with_backoff(func, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert func.call_count == 3
        assert mock_sleep.call_count == 2

    def test_raises_on_non_retryable_error(self):
        error = Exception("bad request")
        error.status_code = 400
        func = MagicMock(side_effect=error)

        with pytest.raises(Exception, match="bad request"):
            retry_with_backoff(func, max_retries=3, base_delay=0.01)

    @patch("strands.agent.retry_utils.time.sleep")
    def test_exhausts_retries(self, mock_sleep):
        error = Exception("timeout")
        error.status_code = 503
        func = MagicMock(side_effect=error)

        with pytest.raises(RetryError) as exc_info:
            retry_with_backoff(func, max_retries=2, base_delay=0.01)
        assert exc_info.value.attempts == 2
        assert exc_info.value.last_exception is error

    @patch("strands.agent.retry_utils.time.sleep")
    def test_on_retry_callback(self, mock_sleep):
        error = Exception("timeout")
        error.status_code = 429
        func = MagicMock(side_effect=[error, "ok"])
        callback = MagicMock()

        retry_with_backoff(func, max_retries=3, base_delay=0.01, on_retry=callback)
        callback.assert_called_once_with(0, error)


class TestBuildRetryHeader:
    def test_headers_contain_attempt(self):
        headers = build_retry_header(2, 5)
        assert headers["X-Retry-Attempt"] == "2"
        assert headers["X-Max-Retries"] == "5"
        assert "X-Retry-Timestamp" in headers


class TestUnsafeExec:
    def test_exec_simple_expression(self):
        result = _unsafe_exec("value = 1 + 2")
        assert result == 3

    def test_exec_returns_none_when_no_value(self):
        result = _unsafe_exec("x = 42")
        assert result is None
