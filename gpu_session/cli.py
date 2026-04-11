"""CLI entry point for gpu-session."""

import sys
import time
from datetime import datetime, timezone
from typing import Optional

import click
import httpx

from .config import config
from .runpod_client import RunPodClient
from .session import Session, SessionManager, session_manager
from .model_resolver import ModelResolver


# Pre-built Docker image with llama.cpp and huggingface-cli
DEFAULT_DOCKER_IMAGE = "ghcr.io/m-amazirh/runpod-session/gpu-session-runtime:latest"


def get_client() -> RunPodClient:
    """Get RunPod client, raising error if API key missing."""
    try:
        return RunPodClient()
    except ValueError as e:
        click.echo(
            f"\n[red]Error:[/red] {e}\n",
            err=True,
        )
        click.echo(
            "Set the RUNPOD_API_KEY environment variable:\n"
            "  export RUNPOD_API_KEY='your-api-key'\n"
            "Get your key from: https://www.runpod.io/console/api\n",
            err=True,
        )
        sys.exit(1)


def get_active_session(client: RunPodClient) -> Optional[dict]:
    """Get active session from RunPod (no local state)."""
    pods = client.list_pods()
    # Find running gpu-session pods
    for pod in pods:
        if pod.get("name", "").startswith("gpu-session-"):
            if pod.get("desiredStatus") == "RUNNING" and pod.get("runtime", {}).get("ports"):
                return pod
    return None


def check_active_session(client: RunPodClient) -> Optional[dict]:
    """Check for active session on RunPod and warn if found."""
    session = get_active_session(client)
    if session:
        click.echo(
            f"\n[yellow]Warning:[/yellow] Active session found (Pod ID: {session['id']})\n",
            err=True,
        )
        click.echo(
            "Stop the existing session first with: gpu-session stop\n",
            err=True,
        )
        sys.exit(1)
    return None


def wait_for_health(
    endpoint: str,
    timeout: int = 900,
    poll_interval: int = 10,
) -> None:
    """
    Wait for health endpoint to return 200.

    Args:
        endpoint: Health check URL
        timeout: Maximum wait time in seconds
        poll_interval: Polling interval in seconds

    Raises:
        TimeoutError: If health check doesn't pass within timeout
    """
    client = httpx.Client(timeout=30.0)
    start_time = time.time()

    click.echo("Waiting for model to load and server to be ready...")

    while time.time() - start_time < timeout:
        try:
            response = client.get(f"{endpoint}/health")
            if response.status_code == 200:
                click.echo("✓ Server is ready")
                return
        except httpx.RequestError:
            pass

        elapsed = int(time.time() - start_time)
        click.echo(f"  Waiting... ({elapsed}s)", nl=False)
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Health check failed after {timeout} seconds. "
        "The model may have failed to load."
    )


@click.group()
def cli():
    """GPU Session CLI - Provision GPU sessions on RunPod for LLM inference."""
    pass


@cli.command()
@click.option(
    "--model",
    required=True,
    help="HuggingFace model in format repo/name:quantization (e.g., unsloth/Qwen3.5-27B-GGUF:Q8_0)",
)
@click.option(
    "--api-key",
    default=None,
    envvar="GPU_SESSION_API_KEY",
    help="API key for the inference endpoint (or set GPU_SESSION_API_KEY env var)",
)
@click.option(
    "--engine",
    default=None,
    type=click.Choice(["llama-cpp", "vllm"]),
    help="Inference engine (default: llama-cpp)",
)
@click.option(
    "--context-length",
    default=None,
    type=int,
    help="Maximum context length (default: 131072)",
)
@click.option(
    "--idle-timeout",
    default=None,
    type=int,
    help="Minutes of inactivity before auto-destroy (default: 0, disabled)",
)
@click.option(
    "--gpu",
    default=None,
    help="Force specific GPU type (e.g., A6000). Default: cheapest available.",
)
@click.option(
    "--region",
    default=None,
    help="Prefer specific RunPod region. Default: any region.",
)
@click.option(
    "--hf-token",
    default=None,
    envvar="HF_TOKEN",
    help="HuggingFace token for gated models (or set HF_TOKEN env var)",
)
@click.option(
    "--filename",
    default=None,
    help="Explicit GGUF filename (overrides auto-resolution)",
)
@click.option(
    "--disk-size",
    default=50,
    type=int,
    help="Container disk size in GB (default: 50, auto-calculated for large models)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be provisioned without creating anything.",
)
def start(
    model: str,
    api_key: Optional[str],
    engine: Optional[str],
    context_length: Optional[int],
    idle_timeout: Optional[int],
    gpu: Optional[str],
    region: Optional[str],
    hf_token: Optional[str],
    filename: Optional[str],
    disk_size: int,
    dry_run: bool,
) -> None:
    """Provision a GPU, download a model, and start serving."""
    # Validate API key
    if not api_key:
        click.echo(
            "\n[red]Error:[/red] API key is required.\n",
            err=True,
        )
        click.echo(
            "Provide via --api-key or set GPU_SESSION_API_KEY environment variable.\n",
            err=True,
        )
        sys.exit(1)

    # Check for active session
    check_active_session(client)

    # Use defaults from config if not specified
    engine = engine or config.default_engine
    context_length = context_length or config.default_context_length
    idle_timeout = idle_timeout if idle_timeout is not None else config.default_idle_timeout

    # Initialize clients
    client = get_client()
    resolver = ModelResolver()

    # List available GPUs
    click.echo("Searching for available GPUs with 48GB+ VRAM...")
    available_gpus = client.list_available_gpus(min_vram=48, gpu_filter=gpu)

    if not available_gpus:
        gpu_msg = f" for GPU type '{gpu}'" if gpu else ""
        click.echo(
            f"\n[red]Error:[/red] No available GPUs with 48GB+ VRAM found{gpu_msg}\n",
            err=True,
        )
        sys.exit(1)

    # Select cheapest GPU
    selected_gpu = available_gpus[0]

    click.echo(
        f"✓ Found GPU: {selected_gpu.name} ({selected_gpu.vram}GB) at ${selected_gpu.price_per_hour:.2f}/hr"
    )

    if dry_run:
        click.echo("\n[dry-run] Would provision:")
        click.echo(f"  GPU: {selected_gpu.name} ({selected_gpu.vram}GB)")
        click.echo(f"  Rate: ${selected_gpu.price_per_hour:.2f}/hr")
        click.echo(f"  Model: {model}")
        click.echo(f"  Engine: {engine}")
        click.echo(f"  Context length: {context_length}")
        click.echo(f"  Docker image: {DEFAULT_DOCKER_IMAGE}")
        return

    # Parse model
    repo_id, resolved_filename = resolver.get_hf_download_args(model)
    repo, model_name, quant = resolver.parse_model_spec(model)
    
    # Use explicit filename if provided, otherwise use resolved
    if filename:
        resolved_filename = filename
    
    # Generate unique pod name from model spec
    # Format: gpu-session-{model-name}-{quant}
    pod_name = f"gpu-session-{model_name.lower().replace('-', '')}-{quant.lower()}"

    # Prepare environment variables
    env_vars = {
        "MODEL_REPO": repo_id,
        "MODEL_QUANT": quant,
        "MODEL_FILENAME": resolved_filename,
        "API_KEY": api_key,
        "ENGINE": engine,
        "CTX_LEN": str(context_length),
        "IDLE_TIMEOUT": str(idle_timeout),
    }

    # Add HF token if provided
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    # Calculate disk size: model size + 20GB overhead (OS, swap, logs)
    # Default to 50GB, but increase for large models (>20GB)
    if disk_size < 50:
        # Estimate: 27B models are ~15-30GB, need at least 50GB total
        estimated_disk = max(disk_size, 50)
    else:
        estimated_disk = disk_size
    
    click.echo(f"\nCreating pod with {selected_gpu.name}...")
    pod = client.create_pod(
        name=pod_name,
        gpu_name=selected_gpu.name,
        container_image=DEFAULT_DOCKER_IMAGE,
        container_disk=estimated_disk,
        env_vars=env_vars,
    )

    pod_id = pod.get("id") or pod.get("podId")
    click.echo(f"✓ Pod created: {pod_id}")

    # Wait for pod to be running
    click.echo("\nWaiting for pod to start...")
    try:
        pod = client.wait_for_running(pod_id, timeout=600, poll_interval=5)
    except (TimeoutError, RuntimeError) as e:
        click.echo(f"\n[red]Error:[/red] {e}\n", err=True)
        client.terminate_pod(pod_id)
        sys.exit(1)

    click.echo("✓ Pod is running")

    # Build endpoint URL (llama.cpp serves at root, OpenAI API at /v1)
    base_url = f"https://{pod_id}-8080.proxy.runpod.net"
    endpoint = f"{base_url}/v1"

    # Wait for health check (health is at root)
    try:
        wait_for_health(base_url, timeout=900, poll_interval=10)
    except TimeoutError as e:
        click.echo(f"\n[red]Error:[/red] {e}\n", err=True)
        click.echo("Destroying pod...", err=True)
        client.terminate_pod(pod_id)
        sys.exit(1)

    # Extract session info from pod (no local state needed)
    env_dict = {}
    for env_var in pod.get("env", []):
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            env_dict[key] = value

    # Print success message
    click.echo("\n[green]✓ GPU session started[/green]")
    click.echo(f"  Pod ID:    {pod_id}")
    click.echo(f"  Pod Name:  {pod_name}")
    click.echo(f"  GPU:       {selected_gpu.name} ({selected_gpu.vram}GB)")
    click.echo(f"  Rate:      ${selected_gpu.price_per_hour:.2f}/hr")
    click.echo(f"  Endpoint:  {endpoint}")
    click.echo(f"  API Key:   {api_key}")
    click.echo(f"  Engine:    {engine}")
    click.echo(f"  Model:     {model}")

    # Vibe config snippet
    vibe_name = model.split("/")[-1].lower().replace("-gguf", "").replace(":", "-")
    click.echo("\nAdd to Vibe config:")
    click.echo('  [[providers]]')
    click.echo(f'  name = "gpu-session"')
    click.echo(f'  api_base = "{endpoint}"')
    click.echo('  api_key_env_var = ""')
    click.echo('  api_style = "openai"')
    click.echo('  backend = "generic"')
    click.echo()
    click.echo('  [[models]]')
    click.echo(f'  name = "{vibe_name}"')
    click.echo('  provider = "gpu-session"')
    click.echo('  alias = "qwen"')
    click.echo('  temperature = 0.2')
    click.echo('  input_price = 0.0')
    click.echo('  output_price = 0.0')


@cli.command()
def stop() -> None:
    """Destroy the active session."""
    client = get_client()
    pod = get_active_session(client)

    if not pod:
        click.echo("No active session")
        return

    pod_id = pod["id"]
    pod_name = pod.get("name", "unknown")
    
    # Extract env vars
    env_dict = {}
    for env_var in pod.get("env", []):
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            env_dict[key] = value

    click.echo(f"Stopping session (Pod ID: {pod_id}, Name: {pod_name})...")
    client.terminate_pod(pod_id)

    click.echo("\n[green]✓ GPU session stopped[/green]")
    click.echo(f"  Pod ID:    {pod_id}")
    click.echo(f"  Pod Name:  {pod_name}")
    click.echo(f"  Model:     {env_dict.get('MODEL_REPO', 'unknown')}:{env_dict.get('MODEL_QUANT', 'unknown')}")


@cli.command()
def status() -> None:
    """Show the current session status."""
    client = get_client()
    pod = get_active_session(client)

    if not pod:
        click.echo("No active session")
        return

    pod_id = pod["id"]
    pod_name = pod.get("name", "unknown")
    gpu = pod.get("machine", {}).get("gpuDisplayName", "unknown")
    rate = pod.get("costPerHr", 0)
    uptime = pod.get("uptimeSeconds", 0)
    
    # Extract env vars
    env_dict = {}
    for env_var in pod.get("env", []):
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            env_dict[key] = value
    
    api_key = env_dict.get("API_KEY", "****")
    model = f"{env_dict.get('MODEL_REPO', 'unknown')}:{env_dict.get('MODEL_QUANT', 'unknown')}"
    engine = env_dict.get("ENGINE", "llama-cpp")
    
    # Build endpoint from runtime ports
    runtime = pod.get("runtime", {})
    ports = runtime.get("ports", [])
    if ports:
        # Find the 8080 port
        for port in ports:
            if port.get("privatePort") == 8080:
                public_port = port.get("publicPort")
                endpoint = f"https://{pod_id}-{public_port}.proxy.runpod.net/v1"
                base_url = f"https://{pod_id}-{public_port}.proxy.runpod.net"
                break
        else:
            endpoint = "unknown"
            base_url = "unknown"
    else:
        endpoint = "unknown"
        base_url = "unknown"

    # Check health
    try:
        with httpx.Client(timeout=10.0) as http_client:
            response = http_client.get(f"{base_url}/health")
            health = "✓ OK" if response.status_code == 200 else "✗ Failed"
    except Exception:
        health = "✗ Unreachable"

    # Calculate uptime
    if uptime > 0:
        hours = uptime // 3600
        minutes = (uptime % 3600) // 60
        seconds = uptime % 60
        if hours > 0:
            uptime_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            uptime_str = f"{minutes}m {seconds}s"
        else:
            uptime_str = f"{seconds}s"
    else:
        uptime_str = "0s"
    
    # Calculate cost
    cost = rate * (uptime / 3600)

    click.echo("\n[green]● GPU session active[/green]")
    click.echo(f"  Pod ID:    {pod_id}")
    click.echo(f"  Pod Name:  {pod_name}")
    click.echo(f"  GPU:       {gpu}")
    click.echo(f"  Rate:      ${rate:.2f}/hr")
    click.echo(f"  Uptime:    {uptime_str}")
    click.echo(f"  Est. cost: ${cost:.2f}")
    click.echo(f"  Endpoint:  {endpoint}")
    click.echo(f"  Health:    {health}")
    click.echo(f"  Model:     {model}")
    click.echo(f"  Engine:    {engine}")


@cli.command()
def list_gpus() -> None:
    """Show available 48GB+ GPUs on RunPod Secure Cloud and their prices."""
    client = get_client()

    click.echo("Searching for available GPUs...")
    gpus = client.list_available_gpus(min_vram=48)

    if not gpus:
        click.echo("\n[red]No available GPUs with 48GB+ VRAM found[/red]\n")
        return

    click.echo("\nAvailable 48GB+ GPUs (RunPod Secure Cloud):")
    for gpu in gpus:
        click.echo(f"  {gpu.name:<16} ${gpu.price_per_hour:.2f}/hr   {gpu.available_count} available")


if __name__ == "__main__":
    cli()
