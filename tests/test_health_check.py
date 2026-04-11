"""Tests for llama.cpp health check."""

import pytest
from unittest.mock import patch, MagicMock
import httpx

from gpu_session.cli import wait_for_health


class TestHealthCheck:
    """Test llama.cpp health check functionality."""

    def test_wait_for_health_success(self):
        """Test wait_for_health succeeds when endpoint returns 200."""
        call_count = [0]

        def mock_get(url):
            call_count[0] += 1
            response = MagicMock()
            response.status_code = 200
            return response

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = mock_get
            mock_client_class.return_value = mock_client

            wait_for_health("http://test-pod.proxy.runpod.net", timeout=10, poll_interval=1)

            # Should succeed on first try
            assert call_count[0] == 1

    def test_wait_for_health_retries_on_error(self):
        """Test wait_for_health retries on connection errors."""
        call_count = [0]

        def mock_get(url):
            call_count[0] += 1
            if call_count[0] < 3:
                raise httpx.RequestError("Connection refused")
            response = MagicMock()
            response.status_code = 200
            return response

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = mock_get
            mock_client_class.return_value = mock_client

            wait_for_health("http://test-pod.proxy.runpod.net", timeout=10, poll_interval=1)

            # Should retry until success
            assert call_count[0] == 3

    def test_wait_for_health_timeout(self):
        """Test wait_for_health raises TimeoutError after max retries."""
        def mock_get(url):
            raise httpx.RequestError("Connection refused")

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = mock_get
            mock_client_class.return_value = mock_client

            with pytest.raises(TimeoutError, match="Health check failed"):
                wait_for_health("http://test-pod.proxy.runpod.net", timeout=2, poll_interval=1)

    def test_wait_for_health_non_200_response(self):
        """Test wait_for_health retries on non-200 responses."""
        call_count = [0]

        def mock_get(url):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] < 2:
                response.status_code = 503  # Service unavailable
            else:
                response.status_code = 200
            return response

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = mock_get
            mock_client_class.return_value = mock_client

            wait_for_health("http://test-pod.proxy.runpod.net", timeout=10, poll_interval=1)

            assert call_count[0] == 2
