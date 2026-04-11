"""RunPod API client wrapper."""

import os
import time
from dataclasses import dataclass
from typing import Optional

import runpod


@dataclass
class GPUInfo:
    """Information about an available GPU."""

    name: str
    vram: int  # in GB
    price_per_hour: float
    available_count: int


class RunPodClient:
    """Client for RunPod API operations."""

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            raise ValueError(
                "RUNPOD_API_KEY environment variable is required. "
                "Get your key from https://www.runpod.io/console/api"
            )
        runpod.api_key = api_key

    def list_available_gpus(
        self,
        min_vram: int = 48,
        gpu_filter: Optional[str] = None,
        region_filter: Optional[str] = None,
    ) -> list[GPUInfo]:
        """
        List available GPUs with at least min_vram GB.

        Args:
            min_vram: Minimum VRAM in GB (default 48)
            gpu_filter: Optional GPU name filter (e.g., "A6000")
            region_filter: Optional region filter

        Returns:
            List of GPUInfo sorted by price ascending
        """
        # Query GPU types from RunPod (returns list directly)
        gpu_types = runpod.get_gpus()

        gpus: dict[str, GPUInfo] = {}

        for gpu_type in gpu_types:
            # Use correct field names from API
            gpu_name = gpu_type.get("id", "")
            vram = gpu_type.get("memoryInGb", 0)

            # Filter by VRAM
            if vram < min_vram:
                continue

            # Filter by GPU name if specified
            if gpu_filter and gpu_name != gpu_filter:
                continue

            # Get detailed info including pricing and cloud types
            try:
                gpu_detail = runpod.get_gpu(gpu_id=gpu_name, gpu_quantity=1)
                if not gpu_detail:
                    continue

                # API returns dict, not list
                if isinstance(gpu_detail, list):
                    gpu_detail = gpu_detail[0] if gpu_detail else None
                    available_count = len(gpu_detail) if gpu_detail else 0
                else:
                    available_count = gpu_detail.get("maxGpuCount", 0)

                if not gpu_detail:
                    continue

                # Check if SECURE cloud is available
                if not gpu_detail.get("secureCloud", False):
                    continue

                # Get SECURE pricing
                price = gpu_detail.get("securePrice", 0)

                if price == 0:
                    continue
            except Exception:
                continue

            if gpu_name in gpus:
                if price < gpus[gpu_name].price_per_hour:
                    gpus[gpu_name] = GPUInfo(
                        name=gpu_name,
                        vram=vram,
                        price_per_hour=price,
                        available_count=available_count,
                    )
            else:
                gpus[gpu_name] = GPUInfo(
                    name=gpu_name,
                    vram=vram,
                    price_per_hour=price,
                    available_count=available_count,
                )

        # Sort by price
        return sorted(gpus.values(), key=lambda g: g.price_per_hour)

    def create_pod(
        self,
        name: str,
        gpu_name: str,
        container_image: str,
        container_disk: int = 50,
        env_vars: Optional[dict] = None,
        args: Optional[str] = None,
    ) -> dict:
        """
        Create a new pod.

        Args:
            name: Pod name (e.g., "gpu-session-qwen3.5-27b-q6")
            gpu_name: GPU type to use (e.g., "A6000")
            container_image: Docker image to use
            container_disk: Disk size in GB
            env_vars: Environment variables
            args: Container arguments (docker args)

        Returns:
            Pod creation response
        """
        if env_vars is None:
            env_vars = {}

        # Format ports as string: "8080/http"
        ports = "8080/http"

        return runpod.create_pod(
            name=name,
            image_name=container_image,
            gpu_type_id=gpu_name,
            cloud_type="SECURE",
            container_disk_in_gb=container_disk,
            ports=ports,
            env=env_vars,
            docker_args=args or "",
        )

    def get_pod(self, pod_id: str) -> dict:
        """Get pod details."""
        return runpod.get_pod(pod_id)

    def list_pods(self) -> list[dict]:
        """List all pods for the account."""
        from runpod.api import ctl_commands
        return ctl_commands.get_pods()

    def terminate_pod(self, pod_id: str) -> None:
        """Terminate a pod."""
        runpod.terminate_pod(pod_id)

    def wait_for_running(
        self,
        pod_id: str,
        timeout: int = 600,
        poll_interval: int = 5,
    ) -> dict:
        """
        Wait for pod to reach RUNNING status.

        Per RunPod SDK: pod is running when desiredStatus == "RUNNING" AND
        runtime has ports assigned. podStatus field is often None.

        Args:
            pod_id: Pod ID to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Pod details when running

        Raises:
            TimeoutError: If pod doesn't start within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            pod = self.get_pod(pod_id)
            desired_status = pod.get("desiredStatus", "")
            runtime = pod.get("runtime", {})
            pod_status = pod.get("podStatus", "")

            # Per RunPod SDK: check desiredStatus + runtime ports
            if desired_status == "RUNNING" and runtime and "ports" in runtime:
                return pod
            elif pod_status in ["TERMINATED", "ERROR"]:
                raise RuntimeError(f"Pod failed to start: {pod_status}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Pod did not start within {timeout} seconds")
