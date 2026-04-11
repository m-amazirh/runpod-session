"""Tests for RunPod client."""

import pytest
from unittest.mock import MagicMock, patch

from gpu_session.runpod_client import RunPodClient, GPUInfo


class TestRunPodClient:
    """Test RunPodClient operations."""

    def test_wait_for_running_with_runtime_ports(self):
        """Test wait_for_running detects running pod via runtime ports."""
        client = RunPodClient(api_key="test-key")

        # Mock pod that's running (desiredStatus=RUNNING, has runtime ports)
        mock_pod = {
            "desiredStatus": "RUNNING",
            "podStatus": None,  # Often None even when running
            "runtime": {
                "ports": [
                    {"privatePort": 8080, "publicPort": 12345, "type": "http"}
                ]
            },
        }

        with patch.object(client, "get_pod", return_value=mock_pod):
            result = client.wait_for_running("test-pod-id", timeout=10, poll_interval=1)

            assert result == mock_pod

    def test_wait_for_running_terminates_on_error(self):
        """Test wait_for_running raises on TERMINATED status."""
        client = RunPodClient(api_key="test-key")

        mock_pod = {
            "desiredStatus": "RUNNING",
            "podStatus": "TERMINATED",
            "runtime": {},
        }

        with patch.object(client, "get_pod", return_value=mock_pod):
            with pytest.raises(RuntimeError, match="Pod failed to start"):
                client.wait_for_running("test-pod-id", timeout=10, poll_interval=1)

    def test_wait_for_running_times_out(self):
        """Test wait_for_running raises TimeoutError when pod never starts."""
        client = RunPodClient(api_key="test-key")

        # Pod stuck in PENDING state
        mock_pod = {
            "desiredStatus": "PENDING",
            "podStatus": None,
            "runtime": {},
        }

        with patch.object(client, "get_pod", return_value=mock_pod):
            with pytest.raises(TimeoutError, match="did not start within"):
                client.wait_for_running("test-pod-id", timeout=2, poll_interval=1)

    def test_wait_for_running_without_runtime(self):
        """Test wait_for_running waits when runtime is None."""
        client = RunPodClient(api_key="test-key")

        call_count = [0]

        def mock_get_pod(pod_id):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: no runtime yet
                return {
                    "desiredStatus": "RUNNING",
                    "podStatus": None,
                    "runtime": None,
                }
            else:
                # Second call: runtime available
                return {
                    "desiredStatus": "RUNNING",
                    "podStatus": None,
                    "runtime": {"ports": [{"privatePort": 8080, "publicPort": 12345}]},
                }

        with patch.object(client, "get_pod", side_effect=mock_get_pod):
            result = client.wait_for_running("test-pod-id", timeout=10, poll_interval=1)

            assert result["runtime"]["ports"] is not None
            assert call_count[0] == 2
