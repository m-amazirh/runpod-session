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


def check_active_session() -> Optional[Session]:
    """Check for active session and warn if found."""
    session = session_manager.load()
    if session:
        click.echo(
            f"\n[yellow]Warning:[/yellow] Active session found (Pod ID: {session.pod_id})\n",
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
    check_active_session()

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
    repo_id, filename = resolver.get_hf_download_args(model)
    repo, model_name, quant = resolver.parse_model_spec(model)

    # Prepare environment variables
    env_vars = {
        "MODEL_REPO": repo_id,
        "MODEL_QUANT": quant,
        "MODEL_FILENAME": filename,
        "API_KEY": api_key,
        "ENGINE": engine,
        "CTX_LEN": str(context_length),
        "IDLE_TIMEOUT": str(idle_timeout),
    }

    # Add HF token if provided
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    # Create pod
    click.echo(f"\nCreating pod with {selected_gpu.name}...")
    pod = client.create_pod(
        gpu_name=selected_gpu.name,
        container_image=DEFAULT_DOCKER_IMAGE,
        container_disk=100,  # 28GB model + overhead
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

    # Create session
    session = Session(
        pod_id=pod_id,
        gpu=selected_gpu.name,
        region=region or "unknown",
        rate_per_hour=selected_gpu.price_per_hour,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        engine=engine,
        started_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    # Save session
    session_manager.save(session)

    # Print success message
    click.echo("\n[green]✓ GPU session started[/green]")
    click.echo(f"  Pod ID:    {session.pod_id}")
    click.echo(f"  GPU:       {session.gpu} ({selected_gpu.vram}GB)")
    click.echo(f"  Region:    {session.region}")
    click.echo(f"  Rate:      ${session.rate_per_hour:.2f}/hr")
    click.echo(f"  Endpoint:  {session.endpoint}")
    click.echo(f"  API Key:   {session.api_key}")
    click.echo(f"  Engine:    {session.engine}")
    click.echo(f"  Model:     {session.model}")

    # Vibe config snippet
    vibe_name = session.model.split("/")[-1].lower().replace("-gguf", "").replace(":", "-")
    click.echo("\nAdd to Vibe config:")
    click.echo('  [[providers]]')
    click.echo(f'  name = "gpu-session"')
    click.echo(f'  api_base = "{session.endpoint}"')
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
    session = session_manager.load()

    if not session:
        click.echo("No active session")
        return

    client = get_client()

    click.echo(f"Stopping session (Pod ID: {session.pod_id})...")
    client.terminate_pod(session.pod_id)

    # Calculate duration and cost
    duration = session.uptime_formatted
    cost = session.estimated_cost

    # Delete session state
    session_manager.delete()

    click.echo("\n[green]✓ GPU session stopped[/green]")
    click.echo(f"  Pod ID:    {session.pod_id}")
    click.echo(f"  Duration:  {duration}")
    click.echo(f"  Est. cost: ${cost:.2f}")


@cli.command()
def status() -> None:
    """Show the current session status."""
    session = session_manager.load()

    if not session:
        click.echo("No active session")
        return

    client = get_client()

    # Query pod status
    try:
        pod = client.get_pod(session.pod_id)
        pod_status = pod.get("podStatus", "unknown")
    except Exception:
        pod_status = "unknown"

    # Check health
    try:
        with httpx.Client(timeout=10.0) as http_client:
            response = http_client.get(f"{session.endpoint}/health")
            health = "✓ OK" if response.status_code == 200 else "✗ Failed"
    except Exception:
        health = "✗ Unreachable"

    click.echo("\n[green]● GPU session active[/green]")
    click.echo(f"  Pod ID:    {session.pod_id}")
    click.echo(f"  GPU:       {session.gpu}")
    click.echo(f"  Rate:      ${session.rate_per_hour:.2f}/hr")
    click.echo(f"  Uptime:    {session.uptime_formatted}")
    click.echo(f"  Est. cost: ${session.estimated_cost:.2f}")
    click.echo(f"  Endpoint:  {session.endpoint}")
    click.echo(f"  Health:    {health}")


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
