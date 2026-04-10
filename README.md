# GPU Session CLI

A command-line tool to provision 48GB+ VRAM GPUs on RunPod Secure Cloud, download GGUF models from HuggingFace, and launch OpenAI-compatible inference servers.

## Installation

```bash
# Install with pip
pip install gpu-session

# Or with uv
uv tool install gpu-session

# Or from source
git clone https://github.com/your-username/gpu-session
cd gpu-session
pip install -e .
```

## Setup

1. Get a RunPod API key from https://www.runpod.io/console/api
2. Set the environment variable:

```bash
export RUNPOD_API_KEY='your-api-key'
```

## Usage

### Start a session

```bash
gpu-session start \
  --model "unsloth/Qwen3.5-27B-GGUF:Q8_0" \
  --api-key "my-secret-key" \
  --engine llama-cpp \
  --context-length 150000 \
  --idle-timeout 30
```

### Stop a session

```bash
gpu-session stop
```

### Check status

```bash
gpu-session status
```

### List available GPUs

```bash
gpu-session list-gpus
```

### Dry run (preview only)

```bash
gpu-session start \
  --model "unsloth/Qwen3.5-27B-GGUF:Q8_0" \
  --api-key "my-secret-key" \
  --dry-run
```

## Configuration

Optional config file at `~/.gpu-session/config.toml`:

```toml
[defaults]
engine = "llama-cpp"
context_length = 150000
idle_timeout = 30
preferred_gpus = ["A6000", "A40", "L40S"]

[runpod]
cloud = "SECURE"
```

## Building the Docker Image

```bash
cd docker
docker build -t ghcr.io/your-username/gpu-session-runtime:latest .
docker push ghcr.io/your-username/gpu-session-runtime:latest
```

Update `DEFAULT_DOCKER_IMAGE` in `gpu_session/cli.py` to point to your image.

## License

MIT
